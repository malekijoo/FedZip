import tensorflow as tf
import numpy as np
import sys

from tensorflow_federated.python.learning import model_utils

# tf.compat.v1.enable_eager_execution()
# tf.config.experimental_run_functions_eagerly(True)

train_dir = '/Users/amir/Documents/CODE/Python/FL/log'

#
def bitsize(tensor, s):


    tensor_s1 = tf.reshape(tf.where(tf.equal(tensor, s)), [-1])
    shifted = tf.roll(tensor_s1, shift=-1, axis=0)
    sub_t = tf.subtract(shifted, tensor_s1)
    shape_sub_t = tf.math.subtract(tf.shape(sub_t), tf.constant(1))[0]
    split1, split2 = tf.split(sub_t, num_or_size_splits=[shape_sub_t, 1], num=2, axis=0)
    splitf = tf.split(tensor_s1, [1, shape_sub_t], num=2, axis=0)
    conc_t = tf.concat([splitf[0], split1], axis=0)

    x1bit = tf.cast(tf.less(conc_t, 2), dtype=tf.int64)
    x2bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**1), dtype=tf.int64),
                                    tf.cast(tf.less(conc_t, 2**2), dtype=tf.int64)), 2)
    x3bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**2), dtype=tf.int64),
                                    tf.cast(tf.less(conc_t, 2**3), dtype=tf.int64)), 3)
    x4bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**3), dtype=tf.int64),
                                    tf.cast(tf.less(conc_t, 2**4), dtype=tf.int64)), 4)
    x5bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**4), dtype=tf.int64),
                                    tf.cast(tf.less(conc_t, 2**5), dtype=tf.int64)), 5)
    x6bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**5), dtype=tf.int64),
                                    tf.cast(tf.less(conc_t, 2**6), dtype=tf.int64)), 6)
    x7bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**6), dtype=tf.int64),
                                    tf.cast(tf.less(conc_t, 2**7), dtype=tf.int64)), 7)
    x8bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**7), dtype=tf.int64),
                                    tf.cast(tf.less(conc_t, 2**8), dtype=tf.int64)), 8)
    x9bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**8), dtype=tf.int64),
                                    tf.cast(tf.less(conc_t, 2**9), dtype=tf.int64)), 9)
    x10bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**9), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**10), dtype=tf.int64)), 10)
    x11bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**10), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**11), dtype=tf.int64)), 11)
    x12bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**11), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**12), dtype=tf.int64)), 12)
    x13bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**12), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**13), dtype=tf.int64)), 13)
    x14bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**13), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**14), dtype=tf.int64)), 14)
    x15bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**14), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**15), dtype=tf.int64)), 15)
    x16bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**15), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**16), dtype=tf.int64)), 16)
    x17bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**16), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**17), dtype=tf.int64)), 17)
    x18bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**17), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**18), dtype=tf.int64)), 18)
    x19bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**18), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**19), dtype=tf.int64)), 19)
    x20bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**19), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**20), dtype=tf.int64)), 20)
    x21bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**20), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**21), dtype=tf.int64)), 21)
    x22bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**21), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**22), dtype=tf.int64)), 22)
    x23bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**22), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**23), dtype=tf.int64)), 23)
    x24bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**23), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**24), dtype=tf.int64)), 24)
    x25bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**24), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**25), dtype=tf.int64)), 25)
    x26bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**25), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**26), dtype=tf.int64)), 26)
    x27bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**26), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**27), dtype=tf.int64)), 27)
    x28bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**27), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**28), dtype=tf.int64)), 28)
    x29bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**28), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**29), dtype=tf.int64)), 29)
    x30bit = tf.multiply(tf.multiply(tf.cast(tf.greater_equal(conc_t, 2**29), dtype=tf.int64),
                                     tf.cast(tf.less(conc_t, 2**30), dtype=tf.int64)), 30)

    x_totall = tf.add_n([x1bit, x2bit, x3bit, x4bit, x5bit,
                         x6bit, x7bit, x8bit, x9bit, x10bit,
                         x11bit, x12bit, x13bit, x14bit, x15bit,
                         x16bit, x17bit, x18bit, x19bit, x20bit,
                         x21bit, x22bit, x23bit, x24bit, x25bit,
                         x26bit, x27bit, x28bit, x29bit, x30bit])

    return tf.reduce_sum(x_totall)


# @tf.function
def kmean_cluster(w_dt, num_cluster=3):

    weight = tf.reshape(w_dt, [-1])
    number_of_nonezeros = tf.math.count_nonzero(weight, dtype=tf.int32)

    min_ = tf.reduce_min(weight)
    max_ = tf.reduce_max(weight)

    centroids = tf.linspace(min_, max_, num=num_cluster)
    tensor = tf.expand_dims(weight, 0)


    for i in range(4):

        distances = tf.math.abs(tensor - tf.reshape(tf.cast(centroids, tf.float32), (-1, 1, 1)))
        distances = tf.transpose(distances, perm=(1, 2, 0))
        classes = tf.math.argmin(distances, axis=-1)
        a = tf.multiply(tensor, tf.cast(tf.math.equal(classes, 0), tf.float32))
        if tf.math.count_nonzero(a) != 0:
            b = tf.expand_dims(tf.divide(tf.math.reduce_sum(a), tf.math.count_nonzero(a, dtype=tf.float32)), 0)
            gg = tf.multiply(tf.cast(tf.math.equal(classes, 0), tf.float32), tf.tile(b, [tensor.get_shape()[1]]))
        c = tf.multiply(tensor, tf.cast(tf.math.equal(classes, 1), tf.float32))
        if tf.reduce_sum(tf.cast(tf.math.equal(classes, 1), tf.float32)) > 0:
            d = tf.expand_dims(tf.divide(tf.math.reduce_sum(c), tf.reduce_sum(tf.cast(tf.math.equal(classes, 1), tf.float32))), 0)
            hh = tf.multiply(tf.cast(tf.math.equal(classes, 1), tf.float32), tf.tile(d, [tensor.get_shape()[1]]))
        elif tf.math.count_nonzero(a) != 0:
            d = tf.expand_dims(tf.divide(tf.math.reduce_sum(c), tf.math.count_nonzero(a, dtype=tf.float32)), 0)
            hh = tf.multiply(tf.cast(tf.math.equal(classes, 1), tf.float32), tf.tile(d, [tensor.get_shape()[1]]))
        else:
            pass

        e = tf.multiply(tensor, tf.cast(tf.math.equal(classes, 2), tf.float32))
        if tf.math.count_nonzero(e) != 0:
            f = tf.expand_dims(tf.divide(tf.math.reduce_sum(e), tf.math.count_nonzero(e, dtype=tf.float32)), 0)
            ll = tf.multiply(tf.cast(tf.math.equal(classes, 2), tf.float32), tf.tile(f, [tensor.get_shape()[1]]))
        new_centroid = tf.concat([b, d, f], 0)
        centroids = new_centroid
    tt = tf.math.add_n([gg, hh, ll])
    return tf.reshape(tt, tf.shape(w_dt))


# @tf.function
def prune(w_dt, key, prune_rate=0.5, top_k=True, threshold=0.9):

    weights = tf.reshape(w_dt, [-1])


    if top_k:
        length = weights.shape[0]
        length1 = int(length)
        prune_rs = np.round(length1 * prune_rate)
        prune_num = length - prune_rs
        # print(' prune layer : ', key, '  original size :', length, 'pruned size : ', prune_num)
        p_mask_data = tf.cast(tf.greater_equal(tf.abs(weights),
                                               tf.reduce_min(tf.math.top_k(tf.abs(weights),
                                                                             prune_num)[0])), tf.float32)
    else:
        p_mask_data = tf.cast(tf.greater_equal(tf.abs(weights), [threshold]), tf.float32)

    p_data = tf.multiply(weights, p_mask_data)

    return tf.reshape(p_data, tf.shape(w_dt))


def applying_prune(w_dt, sparsity_top_key, prune_rate=0.5):

    for key, value in w_dt.items():

        if 'bias' in key:
            pass
        else:
            print(key)
            w_dt[key] = prune(w_dt[key], key, prune_rate + 0.3, top_k=sparsity_top_key)

    return w_dt

def applying_quantization(w_dt):

    for key, value in w_dt.items():
        if 'conv2d/kernel' in key:
            w_dt[key] = kmean_cluster(w_dt[key])
        elif 'bias' in key:
            w_dt[key] = kmean_cluster(w_dt[key])
        elif 'dense' in key:
            w_dt[key] = kmean_cluster(w_dt[key])
        else:
            w_dt[key] = kmean_cluster(w_dt[key])

    return w_dt


def huffman_encoding(w_dt):
    size_ = np.array([1, 1, 0])
    address_bit = tf.convert_to_tensor(size_, dtype=tf.int32)

    size_ = np.array([2, 1, 1])
    bit_usage = tf.convert_to_tensor(size_, dtype=tf.int32)

    # weights = tf.reshape(tf.convert_to_tensor(w_dt, dtype=tf.float32), [-1])

    comp_size = []
    real_size = []
    huffman_size = []

    for key, value in w_dt.items():
        # if 'conv2d/kernel' in key:
        #     tf.print('huffman conv2d/kernel', w_dt[key])

        weights = tf.reshape(w_dt[key], [-1])

        y, idx, count = tf.unique_with_counts(weights)

        # tf.print('centroids for {} are = '.format(key), y, ' &  count  = ', count, '& idx =', idx)

        real_size.append(tf.multiply(tf.size(weights), 32))
        huffman_size.append(tf.reduce_sum(tf.multiply(tf.sort(count), bit_usage)))
        comp_size.append(tf.reduce_sum(tf.multiply(tf.multiply(tf.sort(count), address_bit), 27)))

    tf.print('real size = ', tf.reduce_sum(real_size))
    tf.print('\n')
    tf.print('size after compression', tf.reduce_sum(comp_size))
    tf.print('compression huffman size', tf.reduce_sum(huffman_size))
    tf.print('96 bit 3 * 32 bit for centroids \n')


def huffman_encoding_difference_table(w_dt):

    dif_s1_s2 = []
    dif_s1_s3 = []
    dif_s3_s2 = []

    for key, value in w_dt.items():

        weights = tf.reshape(w_dt[key], [-1])
        y, idx, count = tf.unique_with_counts(weights)
        # tf.print('key ,', key, 'centroids are = ', y, ' &  count  = ', count, ' idx =', idx)
        s1, s2, s3 = tf.split(y, 3, axis=0)
        # tf.print('\n s1 = ', s1, 's2 = ', s2, ' s3 = ', s3)

        a1 = bitsize(weights, s1)
        a2 = bitsize(weights, s2)
        a3 = bitsize(weights, s3)

        dif_s1_s2.append(tf.add_n([a1, a2]))
        dif_s1_s3.append(tf.add_n([a1, a3]))
        dif_s3_s2.append(tf.add_n([a3, a2]))

    tf.print('distances between each centroids')
    tf.print('message size after compression with distance difference between cluster1_cluster2', tf.reduce_sum(dif_s1_s2))
    tf.print('message size after compression with distance difference between cluster1_cluster3', tf.reduce_sum(dif_s1_s3))
    tf.print('message size after compression with distance difference between cluster2_cluster3', tf.reduce_sum(dif_s3_s2))
    tf.print('\n')
