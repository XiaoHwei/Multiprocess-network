from skimage import io,transform
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def cnnnet(x, keep_prob, num_classes):
    # conv1
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

    with tf.name_scope('conv4') as scope:
        kernel4 = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                  dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel4, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name='conv4')
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, kernel4)
    # # (15-3+1)/2=7
    with tf.name_scope('pool4') as scope:
        pool4 = tf.nn.max_pool(conv4,
                                ksize=[1, 3, 3, 1],
                                strides=[1, 2, 2, 1],
                                padding='VALID', )
    # flattened6
    with tf.name_scope('flattened4') as scope:
        conv_t=tf.transpose(pool4,perm=[0,3,1,2])
        conv_t = tf.reshape(conv_t, [-1, 256, 7*7])
        conv_t_T = tf.transpose(conv_t, perm=[0, 2, 1])
        phi_I = tf.matmul(conv_t, conv_t_T)
        phi_I = tf.reshape(phi_I, [-1, 256 * 256])
        phi_I = tf.divide(phi_I, 49)
        y_ssqrt = tf.multiply(tf.sign(phi_I), tf.sqrt(tf.abs(phi_I) + 1e-12))
        z_l2 = tf.nn.l2_normalize(y_ssqrt, dim=1)
    # flattened = tf.reshape(pool3, shape=[-1, 7 * 7 * 256], name="flatten")
    # fc4
    with tf.name_scope('fc5') as scope:
        weights1 = tf.Variable(tf.truncated_normal([256 * 256 , 512],
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
        fc6 = tf.nn.xw_plus_b(dropout5, weights3, biases, name='fc6')
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights3)
    return fc6

def picture(acc, loss, path):
    fig = plt.figure(figsize=(15, 10), dpi=80)
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
    axes = plt.gca()
    axes.set_ylim([0, 1])
    # 绘制图形
    plt.plot(acc)
    plt.xticks(fontproperties='Times New Roman', size=15)
    plt.yticks(fontproperties='Times New Roman', size=15)
    plt.xlabel("Iteration", font)
    plt.ylabel("accuracy", font)
    # 保存
    plt.savefig(path + "/AlexNet_accuracy.png")
    fig = plt.figure(figsize=(15, 10), dpi=80)
    # 绘制图形
    plt.plot(loss[1:])
    plt.xticks(fontproperties='Times New Roman', size=15)
    plt.yticks(fontproperties='Times New Roman', size=15)
    plt.xlabel("Iteration", font)
    plt.ylabel("loss", font)
    plt.savefig(path + "/AlexNet_loss.png")
    plt.show()
    return None

# 将所有的图片重新设置尺寸为227*227*3
w = 256
h = 256
c = 3
num_classes = 6
learning_rate = 0.0001
lamda = 0.03
# mnist数据集中训练数据和测试数据保存地址
train_path = "data3/images"
test_path = "data3/images_test"
# 读取图片及其标签函数
def read_image(path, start_num):
    images = []
    labels = []
    for animal in ['1', '2', '3', '4', '5','6']:
        sum_r = 0
        sum_g = 0
        sum_b = 0
        count = 0
        DIR = path + '/' + animal
        filelist = os.listdir(DIR)
        end_num = len(filelist)
        # end_num=len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        for img_num in range(start_num, end_num, 1):  # 获取指定目录下的所有图片
            img = path + '/' + animal + '/' + str(img_num) + '.jpg'
            # print("reading the image:%s" % img)
            image = io.imread(img)
            image = transform.resize(image, (w, h, c))
            # 求像素均值
            sum_r = sum_r + image[:, :, 0].mean()
            sum_g = sum_g + image[:, :, 1].mean()
            sum_b = sum_b + image[:, :, 2].mean()
            count = count + 1

        sum_r = sum_r / count
        sum_g = sum_g / count
        sum_b = sum_b / count
        # img_mean = [sum_r, sum_g, sum_b]
        for img_num in range(start_num, end_num, 1):  # 获取指定目录下的所有图片
            img = path + '/' + animal + '/' + str(img_num) + '.jpg'
            # print("reading the image:%s" % img)
            image = io.imread(img)
            image = transform.resize(image, (w, h, c))
            # 像素均值处理
            image[:,:,0]-=sum_r
            image[:, :, 1] -= sum_g
            image[:, :, 2] -= sum_b
            images.append(image)
            if animal == '1':
                labels.append([0, 1, 0, 0, 0,0])
            if animal == '2':
                labels.append([1, 0, 0, 0, 0,0])
            if animal == '3':
                labels.append([0, 0, 1, 0, 0,0])
            if animal == '4':
                labels.append([0, 0, 0, 1, 0,0])
            if animal == '5':
                labels.append([0, 0, 0, 0, 1,0])
            if animal == '6':
                labels.append([0, 0, 0, 0, 0,1])
    return np.asarray(images, dtype=np.float32), np.asarray(labels,dtype=np.int32)  # array和asarray都可以将结构数据转化为ndarray，但是主要区别就是当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会

# 读取训练数据及测试数据
train_data, train_label = read_image(train_path, 1)
test_data, test_label = read_image(test_path, 1)

# 打乱训练数据及测试数据
train_image_num = len(train_data)
train_image_index = np.arange(train_image_num)  # arange(start，stop, step, dtype=None)根据start与stop指定的范围以及step设定的步长，生成一个 ndarray。
np.random.shuffle(train_image_index)  # 乱序函数，多维时只对一维乱序
train_data = train_data[train_image_index]  # 乱序后的数据
train_label = train_label[train_image_index]
#
test_image_num = len(test_data)
test_image_index = np.arange(test_image_num)
np.random.shuffle(test_image_index)
test_data = test_data[test_image_index]
test_label = test_label[test_image_index]

# CNNnet调用
x = tf.placeholder(tf.float32, [None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, [None, num_classes], name='y_')
keep_prob = tf.placeholder(tf.float32, name='keep')
fc5 = cnnnet(x, keep_prob, num_classes)

# loss
with tf.name_scope('loss'):
    regularize = tf.contrib.layers.l2_regularizer(scale=lamda)
    reg_term = tf.contrib.layers.apply_regularization(regularize)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc5, labels=y_))+reg_term
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

# accuracy
with tf.name_scope("accuracy_1"):
    correct_pred = tf.equal(tf.argmax(fc5, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

# Tensorboard
filewriter_path = './event/BCNN6_1'
tf.summary.scalar('loss', loss_op)
tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge_all()

# saver
saver = tf.train.Saver()

# 每次获取batch_size个样本进行训练或测试
def get_batch(data, label, batch_size):
    for start_index in range(0, len(data) - batch_size + 1, batch_size):
        slice_index = slice(start_index, start_index + batch_size)
        yield data[slice_index], label[slice_index]

# 创建Session会话
with tf.Session() as sess:
    # 初始化所有变量(权值，偏置等)
    # 将所有样本训练10次，每次训练中以64个为一组训练完所有样本。
    # train_num可以设置大一些。
    train_num = 150
    batch_size = 64
    t_batch_size=1
    train_ac = []  # 训练集准确率
    train_ls = []  # 训练集损失函数
    test_ac = []  # 测试集准确率
    test_ls = []  # 测试集损失函数
    sess.run(init)
    writer = tf.summary.FileWriter(filewriter_path, graph=sess.graph)
    j = 0
    for i in range(train_num):
        print("batchnum:", i)
        train_loss, train_acc, batch_num = 0,0, 0
        for train_data_batch, train_label_batch in get_batch(train_data, train_label, batch_size):
            _, err, acc= sess.run([train_op, loss_op, accuracy],feed_dict={x: train_data_batch, y_: train_label_batch, keep_prob: 0.40})
            train_loss += err
            train_acc += acc
            batch_num += 1
            j = j + 1
            result = sess.run(merged_summary,feed_dict={x: train_data_batch, y_: train_label_batch, keep_prob: 0.40})
            writer.add_summary(result, j)

            train_ac.append(acc)
            train_ls.append(err)
        print("train loss:", train_loss / (batch_num))
        print("train acc:", train_acc / (batch_num))
        test_loss, test_acc, batch_num =0, 0, 0
        for test_data_batch, test_label_batch in get_batch(test_data, test_label, t_batch_size):
            err, acc = sess.run([loss_op, accuracy,],feed_dict={x: test_data_batch, y_: test_label_batch, keep_prob: 1})
            test_loss += err
            test_acc += acc

            batch_num += 1
            test_ac.append(acc)
            test_ls.append(err)
        print("test loss:", test_loss / (batch_num))
        print("test acc:", test_acc /  (batch_num))
        v= test_acc / (batch_num)
        if v > 0.873:
            break
    # 保存模型
    path = "./picture/BCNN6_1/train"
    picture(train_ac, train_ls, path)
    saver.save(sess, "./model/BCNN6_1/BCNN6_1_model")