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

    l_p_4 = list(itertools.product(*[list_pattern]*4))
    list_pattern_4 = get_org(l_p_4, 4)

    return list_pattern_4
def get_array(list_pattern_4):
    np_4 = np.zeros((len(list_pattern_4), 7))
    for i in range(0, len(list_pattern_4)):
        np_4[i][0] = list_pattern_4[i]

    return np_4
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
    list_pattern_4 = get_list()
    np_4 = get_array(list_pattern_4)
    np_4 = get_PSP(train, np_4, 4)

    for i in range(1, 6):
        np_4[:, i] = np_4[:, i]/np_4[:, 6]
    where_nan = np.isnan(np_4)
    np_4[where_nan] = 0

    np_4 = np.delete(np_4, [0, 6], axis=1)

    return np_4

#model = tf.keras.models.load_model('my_model_4th.h5')
train_len = 1975
test_len = 200
pool_length = 5
maxlen = 625
batch_size = 50
np_epoch = 350
l_train_4 = []
l_test_4 = []

print('Loading data...')
# 载入X的数据，分为心律不齐患者和正常人
n_train_data = get_data('paper_n_train')
no_train_data = get_data('paper_n_o_train')
p_train_data = get_data('paper_p_train')
n_train_data.extend(no_train_data)


#截取一部分作为测试集
n_test_data = n_train_data[300:400]
n_train_data = n_train_data[0:300]+n_train_data[400:len(n_train_data)]
p_test_data = p_train_data[300:400]
p_train_data = p_train_data[0:300]+p_train_data[400:len(p_train_data)]

random.shuffle(n_train_data)
random.shuffle(p_train_data)
random.shuffle(n_test_data)
random.shuffle(p_test_data)
with open('C:\ECGtest\info\gg_n_train_data_4_p_n.txt', 'w') as f:
    for i in range(0, len(n_train_data)):
        for j in range(0, len(n_train_data[i])):
            f.write(str(n_train_data[i][j]))
        f.write('*')
        f.write('\n')
with open('C:\ECGtest\info\gg_p_train_data_4_p_n.txt', 'w') as f:
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
    p_4 = get_matrix(train_data[i])
    l_train_4.append(p_4)
    print(i)
np_train_4 = np.array(l_train_4)


#测试集图形
for i in range(0, test_data.shape[0]):
    p_4 = get_matrix(test_data[i])
    l_test_4.append(p_4)
    print(i)
np_test_4 = np.array(l_test_4)

with open('C:\ECGtest\info\model_train_4th_p_n.txt', 'w') as f:
    for i in range(0, np_train_4.shape[0]):
        for j in range(0, maxlen):
            a = str(np_train_4[i][j][0])
            b = str(np_train_4[i][j][1])
            c = str(np_train_4[i][j][2])
            d = str(np_train_4[i][j][3])
            e = str(np_train_4[i][j][4])
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

with open('C:\ECGtest\info\model_test_4th_p_n.txt', 'w') as f:
    for i in range(0, np_test_4.shape[0]):
        for j in range(0, maxlen):
            a = str(np_test_4[i][j][0])
            b = str(np_test_4[i][j][1])
            c = str(np_test_4[i][j][2])
            d = str(np_test_4[i][j][3])
            e = str(np_test_4[i][j][4])
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


np_train_4 = np.zeros((train_len, maxlen, 5))
np_test_4 = np.zeros((test_len, maxlen, 5))

with open('C:\ECGtest\info\model_train_4th_p_n.txt', 'r') as f:
    for i in range(0, train_len):
        for j in range(0, maxlen):
            for k in range(0, 5):
                line = f.readline()
                if not line:
                    break
                a = float(line)
                np_train_4[i][j][k] = a

with open('C:\ECGtest\info\model_test_4th_p_n.txt', 'r') as f:
    for i in range(0, test_len):
        for j in range(0, maxlen):
            for k in range(0, 5):
                line = f.readline()
                if not line:
                    break
                a = float(line)
                np_test_4[i][j][k] = a

#训练集增加深度的维度
np_train_4 = np.expand_dims(np_train_4, axis=3)

#测试集增加深度的维度
np_test_4 = np.expand_dims(np_test_4, axis=3)
print('Build model...')

model = tf.keras.Sequential()
conv1 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=[1, 5],
    padding='same',
    strides=[1,1],
    input_shape=[maxlen, 5, 1],
    activation='selu'
)
conv2 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=[1, 5],
    padding='same',
    strides=[1,1],
    activation='selu'
)
pool1 = tf.keras.layers.MaxPool2D(
    pool_size=[1, 5],
    strides=[1,1],
    padding='same'
)
pool2 = tf.keras.layers.MaxPool2D(
    pool_size=[1, 5],
    strides=[1,1],
    padding='valid'
)

flat = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
ReLU=tf.keras.layers.ReLU()
ReLU1=tf.keras.layers.ReLU()
LSTM=tf.keras.layers.LSTM(100)
Den=tf.keras.layers.Dense(2,activation='softmax')
model.add(conv1)
print(conv1.output_shape)
#model.add(ReLU)
#print(ReLU.output_shape)
model.add(pool1)
#print(pool1.output_shape)
model.add(conv2)
print(conv2.output_shape)
#model.add(ReLU1)
#print(ReLU1.output_shape)
model.add(pool2)
print(pool2.output_shape)
#model.add(conv3)
#print(conv3.output_shape)
model.add(tf.keras.layers.Permute([3, 1, 2]))
model.add(flat)
print(flat.output_shape)
model.add(tf.keras.layers.Permute([2, 1]))
model.add(LSTM)
print(LSTM.output_shape)
model.add(tf.keras.layers.Dropout(0.2))
model.add(Den)
print(Den.output_shape)
model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
)

print('Training...')
filepath="Best_Model_Four_selu_15.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(np_train_4, train_label,
          batch_size=batch_size, epochs=np_epoch,
          validation_data=(np_test_4, test_label),
          callbacks=callbacks_list)

#model.save('my_model_4th_p_n_r.h5')
print(model.summary())
