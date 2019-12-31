from skimage import io,transform
# import glob
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# 将所有的图片重新设置尺寸为256*256*3
w = 256
h = 256
c = 3
num_classes = 6
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
                labels.append([0, 1, 0, 0, 0, 0])
            if animal == '2':
                labels.append([1, 0, 0, 0, 0, 0])
            if animal == '3':
                labels.append([0, 0, 1, 0, 0, 0])
            if animal == '4':
                labels.append([0, 0, 0, 1, 0, 0])
            if animal == '5':
                labels.append([0, 0, 0, 0, 1, 0])
            if animal == '6':
                labels.append([0, 0, 0, 0, 0, 1])
    return np.asarray(images, dtype=np.float32), np.asarray(labels,dtype=np.int32)  # array和asarray都可以将结构数据转化为ndarray，但是主要区别就是当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会

# 读取训练数据及测试数据
test_data, test_label = read_image(test_path, 1)

test_image_num = len(test_data)
test_image_index = np.arange(test_image_num)
np.random.shuffle(test_image_index)
test_data = test_data[test_image_index]
test_label = test_label[test_image_index]

saver=tf.train.import_meta_graph("./model/CNN6/CNN6_model.meta")
graph=tf.get_default_graph()
x=graph.get_tensor_by_name('x:0')
y_=graph.get_tensor_by_name('y_:0')
keep_prob=graph.get_tensor_by_name('keep:0')
fc6=graph.get_tensor_by_name('fc6/fc6:0')
y_predict=tf.nn.softmax(fc6)
print(y_predict)
# 5.建立准确率
with tf.variable_scope("accuracy"):
    equal_list=tf.equal(tf.argmax(y_predict,1),tf.argmax(tf.reshape(y_, [-1, 1*6]),1))
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32),name="accuracy")

init_op=tf.global_variables_initializer()

# 每次获取batch_size个样本进行训练或测试
def get_batch(data, label, batch_size):
    for start_index in range(0, len(data) - batch_size + 1, batch_size):
        slice_index = slice(start_index, start_index + batch_size)
        yield data[slice_index], label[slice_index]

with tf.Session() as sess:
    test_ac = []  # 测试集准确率
    t_batch_size = 1
    sess.run(init_op)
    saver.restore(sess,tf.train.latest_checkpoint('model/CNN6'))
    test_loss, test_acc, batch_num = 0, 0, 0
    for test_data_batch, test_label_batch in get_batch(test_data, test_label, t_batch_size):
        acc = sess.run(accuracy, feed_dict={x: test_data_batch, y_: test_label_batch, keep_prob: 1})
        test_acc += acc
        batch_num += 1
        test_ac.append(acc)
    print("test acc:", test_acc / (batch_num))