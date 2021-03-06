import tensorflow as tf
import os
import numpy as np
import random
import itertools
import keras
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

    l_p_3 = list(itertools.product(*[list_pattern]*3))
    list_pattern_3 = get_org(l_p_3, 3)

    return list_pattern_3
def get_array(list_pattern_3):
    np_3 = np.zeros((len(list_pattern_3), 7))
    for i in range(0, len(list_pattern_3)):
        np_3[i][0] = list_pattern_3[i]

    return np_3
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
    list_pattern_3 = get_list()
    np_3 = get_array(list_pattern_3)
    np_3 = get_PSP(train, np_3, 3)

    for i in range(1, 6):
        np_3[:, i] = np_3[:, i]/np_3[:, 6]
    where_nan = np.isnan(np_3)
    np_3[where_nan] = 0

    np_3 = np.delete(np_3, [0, 6], axis=1)

    return np_3

train_len = 7680
test_len = 435
pool_length = 5
maxlen = 125
batch_size = 200
np_epoch = 1000
l_train_3 = []
l_test_3 = []


print('Loading data...')
# 载入X的数据，分为心律不齐患者和正常人
n_train_data = get_data('paper_n_train_128')
no_train_data = get_data('paper_no_train_128')
p_train_data = get_data('paper_p_train_128')
n_train_data.extend(no_train_data)


#截取一部分作为测试集
n_test_data = n_train_data[1160:1450]
n_train_data = n_train_data[0:1160]
#n_train_data = n_train_data[0:870]+n_train_data[1160:len(n_train_data)]
p_test_data = p_train_data[580:725]
#p_train_data = p_train_data[0:435]+p_train_data[580:len(p_train_data)]
p_train_data = p_train_data[0:580]

random.shuffle(n_train_data)
random.shuffle(p_train_data)
random.shuffle(n_test_data)
random.shuffle(p_test_data)

with open('C:\ECGtest\info\gg_n_train_data_3_p.txt', 'w') as f:
    for i in range(0, len(n_train_data)):
        for j in range(0, len(n_train_data[i])):
            f.write(str(n_train_data[i][j]))
        f.write('*')
        f.write('\n')
with open('C:\ECGtest\info\gg_p_train_data_3_p.txt', 'w') as f:
    for i in range(0, len(p_train_data)):
        for j in range(0, len(p_train_data[i])):
            f.write(str(p_train_data[i][j]))
        f.write('*')
        f.write('\n')
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
    p_3 = get_matrix(train_data[i])
    l_train_3.append(p_3)
np_train_3 = np.array(l_train_3)

#测试集图形
for i in range(0, test_data.shape[0]):
    p_3 = get_matrix(test_data[i])
    l_test_3.append(p_3)
np_test_3 = np.array(l_test_3)
with open('C:\ECGtest\info\model_train_3th_p.txt', 'w') as f:
    for i in range(0, np_train_3.shape[0]):
        for j in range(0, maxlen):
            a = str(np_train_3[i][j][0])
            b = str(np_train_3[i][j][1])
            c = str(np_train_3[i][j][2])
            d = str(np_train_3[i][j][3])
            e = str(np_train_3[i][j][4])
            f.write(a)
            f.write('\n')
            f.writelines(b)
            f.write('\n')
            f.writelines(c)
            f.write('\n')
            f.writelines(d)
            f.write('\n')
            f.writelines(e)
            f.write('\n')

with open('C:\ECGtest\info\model_test_3th_p.txt', 'w') as f:
    for i in range(0, np_test_3.shape[0]):
        for j in range(0, maxlen):
            a = str(np_test_3[i][j][0])
            b = str(np_test_3[i][j][1])
            c = str(np_test_3[i][j][2])
            d = str(np_test_3[i][j][3])
            e = str(np_test_3[i][j][4])
            f.write(a)
            f.write('\n')
            f.writelines(b)
            f.write('\n')
            f.writelines(c)
            f.write('\n')
            f.writelines(d)
            f.write('\n')
            f.writelines(e)
            f.write('\n')

#加载PSP矩阵
np_train_3 = np.zeros((train_len, maxlen, 5))
np_test_3 = np.zeros((test_len, maxlen, 5))
with open('C:\ECGtest\info\model_train_3th_p.txt', 'r') as f:
    for i in range(0, train_len):
        for j in range(0, maxlen):
            for k in range(0, 5):
                line = f.readline()
                if not line:
                    break
                a = float(line)
                np_train_3[i][j][k] = a

with open('C:\ECGtest\info\model_test_3th_p.txt', 'r') as f:
    for i in range(0, test_len):
        for j in range(0, maxlen):
            for k in range(0, 5):
                line = f.readline()
                if not line:
                    break
                a = float(line)
                np_test_3[i][j][k] = a
#训练集增加深度的维度
np_train_3 = np.expand_dims(np_train_3, axis=3)

#测试集增加深度的维度
np_test_3 = np.expand_dims(np_test_3, axis=3)

print('Build model...')

model = keras.Sequential()
conv1 = keras.layers.Conv2D(
    filters=64,
    kernel_size=[1,5],
    padding='same',
    strides=[1,1],
    input_shape=[maxlen, 5, 1],
    activation='selu'
)

pool1 = keras.layers.MaxPool2D(
    pool_size=[1,5],
    strides=[1,1],
    padding='valid'
)
conv2 = keras.layers.Conv2D(
    filters=64,
    kernel_size=[3,3],
    padding='same',
    strides=[1,1]
)
flat = keras.layers.TimeDistributed(keras.layers.Flatten())
model.add(conv1)
#model.add(tf.keras.layers.ReLU())
model.add(pool1)
#model.add(conv2)
#model.add(tf.keras.layers.ReLU())
model.add(keras.layers.Permute([3,1,2]))
model.add(flat)
model.add(keras.layers.Permute([2, 1]))
model.add(keras.layers.LSTM(100))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(2,activation='softmax'))
model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
)

print('Training...')
filepath="model_3.h5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(np_train_3, train_label,
          batch_size=batch_size, epochs=np_epoch,
          validation_data=(np_test_3, test_label),
          callbacks=callbacks_list)

#model.save('my_model_3th_GG.h5')

print(model.summary())
