from __future__ import absolute_import, division, print_function
from six.moves import range
# from matplotlib import pyplot as plt

import sys
import time
import warnings
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import new_fedavg_keras as fed__compression
# import new_fedsgd_keras as fed__compression

from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow_federated.python.research.utils import checkpoint_manager


model_dir = '/Users/amir/Documents/CODE/Python/FL/log/model/'
logdir = '/Users/amir/Documents/CODE/Python/FL/log/'

checkpoint_manager_obj = checkpoint_manager.FileCheckpointManager(model_dir)
warnings.simplefilter('ignore')
tff.framework.set_default_executor(tff.framework.create_local_executor())



np.random.seed(0)
tf.compat.v1.enable_v2_behavior()



NUM_EPOCHS = 101
BATCH_SIZE = 10
SHUFFLE_BUFFER = 500
NUM_CLASSES = 10
NUM_CLIENTS = 10


'''   pre process of image and flatten them   '''

def preprocess(dataset):

    def element_fn(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['pixels'], [28, 28, 1])),
            ('y', tf.reshape(element['label'], [1])),
        ])

    return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
        SHUFFLE_BUFFER).batch(BATCH_SIZE)

'''   Loading the dataset of images in MNIST from Local source   '''


emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(cache_dir='/Users/amir/Documents/CODE/Python/FL/')
# emnist_train, emnist_test = tff.simulation.datasets.cifar100.load_data(cache_dir='/Users/amir/Documents/CODE/Python/FL/')

print(len(emnist_train.client_ids))

'''   sample  for prebuiding the keras model  '''

example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])
example_element = next(iter(example_dataset))
preprocessed_example_dataset = preprocess(example_dataset)
sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                     next(iter(preprocessed_example_dataset)))


'''   preprocess   '''
def make_federated_data(client_data, client_ids):
  return [preprocess(client_data.create_tf_dataset_for_client(x))
          for x in client_ids]


'''   federated  dataset  '''
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients)



def create_compiled_keras_model():

    ##############################################################################################
    ##############################################################################################
    ''' first try with 4 layer network , 2 convolution 2 dense'''
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                                     activation='relu',
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    print(model.summary())

    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))
    model.compile(
        loss=loss_fn,
        optimizer=gradient_descent.SGD(learning_rate=0.1),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    ##############################################################################################
    ##############################################################################################
    ''' VGG16 '''
    #
    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Conv2D(input_shape=(28, 28, 1), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # # model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(units=4096, activation="relu"))
    # model.add(tf.keras.layers.Dense(units=4096, activation="relu"))
    # model.add(tf.keras.layers.Dense(units=NUM_CLASSES, activation="softmax"))


    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model



def model_fn():
  keras_model = create_compiled_keras_model()
  return tff.learning.from_compiled_keras_model(keras_model, dummy_batch=sample_batch)


def tensor_summaries(arr, round_num):
    a = ['conv2d/kernel', 'conv2d/bias', 'conv2d_1/kernel', 'conv2d_1/bias', 'dense/kernel', 'dense/bias',
         'dense_1/kernel', 'dense_1/bias']
    j = 0

    for i in arr:
        tensor = tf.convert_to_tensor(i, dtype=tf.float32)
        tf.summary.histogram(name=a[j], data=tensor, step=round_num)
        j += 1


summary_writer = tf.summary.create_file_writer(logdir)

with summary_writer.as_default():

    iterative_process_with_Compression = fed__compression.build_federated_averaging_process(model_fn)
    # iterative_process_with_Compression = fed__compression.build_federated_sgd_process(model_fn)

    ServerState = fed__compression.ServerState  # pylint: disable=invalid-name

    state = iterative_process_with_Compression.initialize()


    start_time = time.time()
    print('Starting FedZip training   ')
    for round_num in range(NUM_EPOCHS):
        round_time = time.time()
        state, metrics, weight_delta = iterative_process_with_Compression.next(state, federated_train_data)
        print('round {:2d}, metrics={}'.format(round_num, metrics))

        for name, value in metrics._asdict().items():
            tf.summary.scalar(name, value, step=round_num)

        round_time = time.time() - round_time
        print('round time =', round_time)


state = ServerState.from_tff_result(state)
checkpoint_manager_obj.save_checkpoint(state, round_num)

elapsed_time = time.time() - start_time
print('\n  elaspsed time =  ', elapsed_time)


evaluation = tff.learning.build_federated_evaluation(model_fn=model_fn)
train_metrics = evaluation(state.model, federated_train_data)
print('Train Metrics = ', train_metrics)


sample_clients = emnist_train.client_ids[NUM_CLIENTS: NUM_CLIENTS+10]
federated_test_data = make_federated_data(emnist_test, sample_clients)
test_metrics = evaluation(state.model, federated_test_data)

print('NUM_CLIENTS+10: Test Metrics = ', test_metrics)

sample_clients = emnist_train.client_ids[0: NUM_CLIENTS]
federated_test_data = make_federated_data(emnist_test, sample_clients)
test_metrics = evaluation(state.model, federated_test_data)

print('0-NUM_CLIENTS : Test Metrics = ', test_metrics)

