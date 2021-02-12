from __future__ import absolute_import, division, print_function
import os

import collections
from six.moves import range
import numpy as np
import fedavg_pruning as FedAvgPr
import fedavg_quntization as FedAvgQ
import tensorflow as tf
from tensorflow_federated import python as tff

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_dir = '/Users/amir/Documents/CODE/Python/FL'
nest = tf.contrib.framework.nest

np.random.seed(0)

tf.compat.v1.enable_v2_behavior()

tf.compat.v1.enable_eager_execution()

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(only_digits=True,
                                                                     cache_dir='/Users/amir/Documents/CODE/Python/FL/')


NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500



def preprocess(dataset):
    def element_fn(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['pixels'], [-1])),
            ('y', tf.reshape(element['label'], [1])),
        ])

    return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
        SHUFFLE_BUFFER).batch(BATCH_SIZE)



def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x))
            for x in client_ids]


NUM_CLIENTS = 3383

sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients)


MnistVariables = collections.namedtuple(


def layer_model(variables, batch):

    l = tf.matmul(batch['x'], variables.weights) + variables.bias
    y = tf.nn.softmax(l)
    predictions = tf.cast(tf.argmax(y, 1), tf.int32)
    # print('shape y ===== ', y.shape, 'shape perediction = ', tf.shape(predictions))
    flat_labels = tf.reshape(batch['y'], [-1])
    loss = -tf.reduce_mean(tf.reduce_sum(
        tf.one_hot(flat_labels, 10) * tf.log(y), reduction_indices=[1]))
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predictions, flat_labels), tf.float32))

    return loss, accuracy, predictions


def forward_pruning(variables, batch):
    pruning_mask_data = tf.cast(tf.greater(tf.math.abs(variables.weights), 0.01), tf.float32)
    tf.assign(variables.weights, tf.multiply(variables.weights, pruning_mask_data))
    loss, accuracy, predictions = layer_model(variables, batch)

    return loss, predictions, pruning_mask_data



def create_mnist_variables():
    return MnistVariables(
        weights=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
            name='weights',
            trainable=True),
        bias=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(10)),
            name='bias',
            trainable=True),
        num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
        loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
        accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))


def mnist_forward_pass(variables, batch):
    '''variables.weights
    inja mituni model tarif koni baraye server
    albate in tabe vase clients ha ham estefade shode mituni 2 tabe tarif koni
    '''

    loss, accuracy, predictions = layer_model(variables, batch)
    num_examples = tf.to_float(tf.size(batch['y']))

    tf.assign_add(variables.num_examples, num_examples)
    tf.assign_add(variables.loss_sum, loss * num_examples)
    tf.assign_add(variables.accuracy_sum, accuracy * num_examples)

    return loss, predictions


def get_local_mnist_metrics(variables):
    return collections.OrderedDict([
        ('num_examples', variables.num_examples),
        ('loss', variables.loss_sum / variables.num_examples),
        ('accuracy', variables.accuracy_sum / variables.num_examples)
    ])


@tff.federated_computation
def aggregate_mnist_metrics_across_clients(metrics):
    return {
        'num_examples': tff.federated_sum(metrics.num_examples),
        'loss': tff.federated_mean(metrics.loss, metrics.num_examples),
        'accuracy': tff.federated_mean(metrics.accuracy, metrics.num_examples)}


###################################################################################################################


class MnistModel(tff.learning.Model):

    def __init__(self):
        self._variables = create_mnist_variables()

    @property
    def trainable_variables(self):
        return [self._variables.weights, self._variables.bias]

    @property
    def non_trainable_variables(self):
        return []

    @property
    def local_variables(self):
        return [
            self._variables.num_examples, self._variables.loss_sum,
            self._variables.accuracy_sum
        ]

    @property
    def input_spec(self):
        return collections.OrderedDict([('x', tf.TensorSpec([None, 784],
                                                            tf.float32)),
                                        ('y', tf.TensorSpec([None, 1], tf.int32))])

    # TODO(b/124777499): Remove `autograph=False` when possible.
    @tf.contrib.eager.function(autograph=False)
    def forward_pass(self, batch, training=True):
        del training
        loss, predictions = mnist_forward_pass(self._variables, batch)
        return tff.learning.BatchOutput(loss=loss, predictions=predictions)

    @tf.contrib.eager.function(autograph=False)
    def pruning_pass(self, batch, training=True):
        del training
        loss, predictions, pruning_mask = forward_pruning(self._variables, batch)
        return tff.learning.BatchOutput(loss=loss, predictions=predictions), pruning_mask

    def quantization_pass(self, batch, training=True):
        del training
        loss, predictions, quantization_mask = forward_quantization(self._variables, batch)
        return tff.learning.BatchOutput(loss=loss, predictions=predictions), quantization_mask

    @tf.contrib.eager.function(autograph=False)
    def report_local_outputs(self):
        return get_local_mnist_metrics(self._variables)

    @property
    def federated_output_computation(self):
        return aggregate_mnist_metrics_across_clients


class MnistTrainableModel(MnistModel, tff.learning.TrainableModel):

    # TODO(b/124777499): Remove `autograph=False` when possible.
    @tf.contrib.eager.defun(autograph=False)
    def train_on_batch(self, batch):
        output = self.forward_pass(batch)
        optimizer = tf.train.GradientDescentOptimizer(0.02)
        optimizer.minimize(output.loss, var_list=self.trainable_variables)

        return output

    @tf.contrib.eager.defun(autograph=False)
    def train_pruning(self, batch):
        output, pruning_mask = self.pruning_pass(batch)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        gradient_var = optimizer.compute_gradients(output.loss, var_list=self.trainable_variables)
        grad = [(tf.multiply(gv[0], pruning_mask), gv[1]) for gv in gradient_var]
        optimizer.apply_gradients(grad)
        # self.train_on_batch(batch)
        print('output in pruning is going to be show here ==>', output)

        return output

    def train_quantization(self, batch):
        output, quantization_mask = self.quantization_pass(batch)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        gradient_var = optimizer.compute_gradients(output.loss, var_list=self.trainable_variables)
        grad = [(tf.multiply(gv[0], quantization_mask), gv[1]) for gv in gradient_var]
        optimizer.apply_gradients(grad)
        # self.train_on_batch(batch)
        print('output in pruning is going to be show here ==>', output)
        return output




###################################################################################################################
global_step = tf.train.get_or_create_global_step()
summary_writer = tf.contrib.summary.create_file_writer(
    train_dir, flush_millis=10000)
with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():

    quantization_part = FedAvgQ.build_federated_averaging_process(MnistTrainableModel)
    state = quantization_part.initialize()

    # @test {"skip": true}
    for round_num in range(1, 11):

        print('33  ')
        state, metrics = quantization_part.next(state, federated_train_data)

        print('round {:2d}, metrics={}'.format(round_num, metrics))
        tf.contrib.summary.scalar("loss", metrics.loss)
        tf.contrib.summary.scalar("accuracy", metrics.accuracy)
        tf.contrib.summary.scalar("number of example", metrics.num_examples)

evaluation = tff.learning.build_federated_evaluation(MnistModel)
train_metrics = evaluation(state.model, federated_train_data)
print('Train Metrics = ', train_metrics)

federated_test_data = make_federated_data(emnist_test, sample_clients)
test_metrics = evaluation(state.model, federated_test_data)

print('Test Metrics = ', test_metrics)
