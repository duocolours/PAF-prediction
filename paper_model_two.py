import tensorflow as tf
import os
import numpy as np
import random
import itertools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_data(filename):
    data = []
    list_data = []
    end = False

    with open('C:\ECGtest\info\\' + filename + '.txt', 'r') as f:
        while 1:
            line = f.readline()
            if not line:
                break

            for ch in line:
                if ch == '*':
                    end = True
                    break
                elif ch == '\n':
                    continue
                else:
                    data.append(int(ch))

            if end:
                end = False

                list_data.append(data)
                data = []

    return list_data
def get_last_data(filename):
    data = []
    list_data = []
    list_last = []
    end = False
    count = 0

    with open('C:\ECGtest\info\\' + filename + '.txt', 'r') as f:
        while 1:
            line = f.readline()
            if not line:
                break

            for ch in line:
                if ch == '*':
                    end = True
                    break
                elif ch == '\n':
                    continue
                else:
                    data.append(int(ch))
                    count = count + 1

            if end:
                end = False
                if count == 30:
                    count = 1
                if count < 25:
                    list_data.append(data)
                else:
                    list_last.append(data)
                data = []

    return list_data, list_last
def get_label_n(x_train_data):
    label = []
    for i in range(0, len(x_train_data)):
        a = np.hstack((0, 1))
        label.append(a)
    return label
def get_label_p(x_train_data):
    label = []
    for i in range(0, len(x_train_data)):
        a = np.hstack((1, 0))
        label.append(a)
    return label
def get_org(n_p, n):
    list_data = []
    st = ''
    for i in range(0, len(n_p)):
        for j in range(0, n):
            st = st+str(n_p[i][j])
        list_data.append(int(st))
        st = ''
    return list_data
def get_list():
    list_pattern = []
    for i in range(1, 6):
        list_pattern.append(i)
    l_p_2 = list(itertools.product(*[list_pattern]*2))
    list_pattern_2 = get_org(l_p_2,2)

    return list_pattern_2
def get_array(list_pattern_2):
    np_2 = np.zeros((len(list_pattern_2), 7))
    for i in range(0, len(list_pattern_2)):
        np_2[i][0] = list_pattern_2[i]

    return np_2
def get_PSP(train, np_array, n):

    for i in range(0, len(train)):
        if i + n >= len(train):
            break
        else:
            temp = train[i: i + n + 1]
            j = 0
            gg = ''
            for p in range(0, n):
                gg = str(temp[p])+gg
            gg = int(gg)
            for j in range(0, np_array.shape[0]):
                if gg == np_array[j][0]:
                    break
            np_array[j][6] = np_array[j][6] + 1
            np_array[j][temp[n]] = np_array[j][temp[n]] + 1

    return np_array
def get_matrix(train):
    list_pattern_2 = get_list()
    np_2 = get_array(list_pattern_2)
    np_2 = get_PSP(train, np_2, 2)

    for i in range(1, 6):
        np_2[:, i] = np_2[:, i]/np_2[:, 6]
    where_nan = np.isnan(np_2)
    np_2[where_nan] = 0
    np_2 = np.delete(np_2, [0, 6], axis=1)

    return np_2


pool_length = 5
maxlen = 25
batch_size = 200
np_epoch = 1000
l_train_2 = []
l_test_2 = []
print('Loading data...')
# 载入X的数据，分为心律不齐患者和正常人
n_train_data = get_data('paper_n_train')
no_train_data = get_data('paper_n_o_train')
p_train_data = get_data('paper_p_train')
n_train_data.extend(no_train_data)

#截取一部分作为测试集
n_test_data = n_train_data[100:200]
n_train_data = n_train_data[0:100]+n_train_data[200:len(n_train_data)]
p_test_data = p_train_data[100:200]
p_train_data = p_train_data[0:100]+p_train_data[200:len(p_train_data)]

random.shuffle(n_train_data)
random.shuffle(p_train_data)
random.shuffle(n_test_data)
random.shuffle(p_test_data)

#获得标签
n_train_label = get_label_n(n_train_data)
p_train_label = get_label_p(p_train_data)
n_test_label = get_label_n(n_test_data)
p_test_label = get_label_p(p_test_data)

#形成数组
train_data = np.array(n_train_data+p_train_data)
train_label = np.array(n_train_label+p_train_label)
test_data = np.array(n_test_data+p_test_data)
test_label = np.array(n_test_label+p_test_label)

#训练集图形
for i in range(0, train_data.shape[0]):
    p_2 = get_matrix(train_data[i])
    l_train_2.append(p_2)
np_train_2 = np.array(l_train_2)

#测试集图形
for i in range(0, test_data.shape[0]):
    p_2 = get_matrix(test_data[i])
    l_test_2.append(p_2)
np_test_2 = np.array(l_test_2)

#训练集增加深度的维度
np_train_2 = np.expand_dims(np_train_2, axis=3)

#测试集增加深度的维度
np_test_2 = np.expand_dims(np_test_2, axis=3)

print('Build model...')

model = tf.keras.Sequential()
conv1 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=[1,5],
    padding='same',
    strides=[1,1],
    input_shape=[maxlen, 5, 1],
    activation='relu'
)

pool1 = tf.keras.layers.MaxPool2D(
    pool_size=[1,5],
    strides=[1,1],
    padding='same'
)
conv2 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=[1,5],
    padding='same',
    strides=[1,1],
    activation='selu'
)
conv3 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=[1,5],
    padding='same',
    strides=[1,1],
    activation='selu'
)
conv4 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=[1,5],
    padding='same',
    strides=[1,1],
    activation='selu'
)
pool2 = tf.keras.layers.MaxPool2D(
    pool_size=[1,5],
    strides=[1,1],
    padding='same'
)

pool3 = tf.keras.layers.MaxPool2D(
    pool_size=[1,5],
    strides=[1,1],
    padding='valid'
)
flat = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
model.add(conv1)
#model.add(pool1)
model.add(conv2)
#model.add(tf.keras.layers.ReLU())
#model.add(pool2)
model.add(conv3)
#model.add(pool2)
#model.add(conv4)
model.add(pool3)




model.add(tf.keras.layers.Permute([3,1,2]))
model.add(flat)
model.add(tf.keras.layers.Permute([2, 1]))
print(flat.output_shape)

model.add(tf.keras.layers.LSTM(100))

model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(2,activation='softmax'))

model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
)
filepath="Two_15_selu_500_cccp.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
print('Training...')
model.fit(np_train_2, train_label,
          batch_size=batch_size, epochs=np_epoch,
          validation_data=(np_test_2, test_label),
          callbacks=callbacks_list)

#model.save('my_model_sec_p_n_selu.h5')
print(model.summary())
