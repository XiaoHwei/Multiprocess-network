from skimage import io,transform
from tensorflow.python.framework import graph_util
import numpy as np
import warnings
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

# 将所有的图片重新设置尺寸为256*256*3
w = 256
h = 256
c = 3
num_classes = 6
keep_prob=1
with tf.Graph().as_default() as graph:
        x = tf.Variable(initial_value=tf.random_normal([1, w,h,c]))
        with tf.name_scope('conv1') as scope:
            kernel1 = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32,
                                                      stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(x, kernel1, [1, 4, 4, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                                 trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name='conv1')
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, kernel1)
        # 256/4=64
        # lrn1
        with tf.name_scope('lrn1') as scope:
            lrn1 = tf.nn.local_response_normalization(conv1,
                                                      alpha=1e-4,
                                                      beta=0.75,
                                                      depth_radius=2,
                                                      bias=2.0)

        # pool1
        with tf.name_scope('pool1') as scope:
            pool1 = tf.nn.max_pool(lrn1,
                                   ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='VALID')
        # [(64-3)+1]/2=31
        # conv2
        with tf.name_scope('conv2') as scope:
            pool1_groups = tf.split(axis=3, value=pool1, num_or_size_splits=2)
            kernel2 = tf.Variable(tf.truncated_normal([5, 5, 48, 128], dtype=tf.float32,
                                                      stddev=1e-1), name='weights')
            kernel_groups = tf.split(axis=3, value=kernel2, num_or_size_splits=2)
            conv_up = tf.nn.conv2d(pool1_groups[0], kernel_groups[0], [1, 1, 1, 1], padding='SAME')
            conv_down = tf.nn.conv2d(pool1_groups[1], kernel_groups[1], [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            biases_groups = tf.split(axis=0, value=biases, num_or_size_splits=2)
            bias_up = tf.nn.bias_add(conv_up, biases_groups[0])
            bias_down = tf.nn.bias_add(conv_down, biases_groups[1])
            bias = tf.concat(axis=3, values=[bias_up, bias_down])
            conv2 = tf.nn.relu(bias, name='conv2')
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, kernel2)
        # lrn2
        with tf.name_scope('lrn2') as scope:
            lrn2 = tf.nn.local_response_normalization(conv2,
                                                      alpha=1e-4,
                                                      beta=0.75,
                                                      depth_radius=2,
                                                      bias=2.0)
        # pool2
        with tf.name_scope('pool2') as scope:
            pool2 = tf.nn.max_pool(lrn2,
                                   ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='VALID')
            # (31-3+1)/2=15
            # conv3
        with tf.name_scope('conv3') as scope:
            kernel3 = tf.Variable(tf.truncated_normal([3, 3, 128, 256],
                                                      dtype=tf.float32,
                                                      stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(pool2, kernel3, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name='conv3')
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, kernel3)
        # # (15-3+1)/2=7
        with tf.name_scope('pool4') as scope:
            pool4 = tf.nn.max_pool(conv3,
                                   ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='VALID', )
        # flattened6
        with tf.name_scope('flattened4') as scope:
            conv_t = tf.transpose(pool4, perm=[0, 3, 1, 2])
            conv_t = tf.reshape(conv_t, [-1, 256, 7 * 7])
            conv_t_T = tf.transpose(conv_t, perm=[0, 2, 1])
            phi_I = tf.matmul(conv_t, conv_t_T)
            phi_I = tf.reshape(phi_I, [-1, 256 * 256])
            phi_I = tf.divide(phi_I, 49)
            y_ssqrt = tf.multiply(tf.sign(phi_I), tf.sqrt(tf.abs(phi_I) + 1e-12))
            z_l2 = tf.nn.l2_normalize(y_ssqrt, dim=1)
        # flattened = tf.reshape(pool3, shape=[-1, 7 * 7 * 256], name="flatten")
        # fc4
        with tf.name_scope('fc5') as scope:
            weights1 = tf.Variable(tf.truncated_normal([256 * 256, 512],
                                                       dtype=tf.float32,
                                                       stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            bias = tf.nn.xw_plus_b(z_l2, weights1, biases)
            fc5 = tf.nn.relu(bias, name='fc5')
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights1)
        # dropout4
        with tf.name_scope('dropout5') as scope:
            dropout5 = tf.nn.dropout(fc5, keep_prob)
        # fc5
        with tf.name_scope('fc6') as scope:
            weights3 = tf.Variable(tf.truncated_normal([512, num_classes],
                                                       dtype=tf.float32,
                                                       stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[num_classes], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc6 = tf.nn.xw_plus_b(dropout5, weights3, biases, name='output')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_graph = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['fc6/output'])
            with tf.gfile.GFile('graph.pb', "wb") as f:
                f.write(output_graph.SerializeToString())
        # print("class:", y)
# ***** (3) Load frozen graph *****
graph = load_pb('./graph.pb')
print('stats after freezing')
stats_graph(graph)