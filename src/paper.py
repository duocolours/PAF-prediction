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
def get_list(n):
    list_pattern = []
    for i in range(1, 6):
        list_pattern.append(i)

    l_p_t = list(itertools.product(*[list_pattern]*n))
    list_pattern_t = get_org(l_p_t, n)

    return list_pattern_t
def get_array(list_pattern_5):
    np_5 = np.zeros((len(list_pattern_5), 7))
    for i in range(0, len(list_pattern_5)):
        np_5[i][0] = list_pattern_5[i]

    return np_5
def get_PSP(train, np_array, n):

    for i in range(0, len(train)):
        if i + n >= len(train):
            break
        else:
            temp = train[i: i + n + 1]
            gg = ''
            for p in range(0, n):
                gg = str(temp[p])+gg
            gg = int(gg)
            j = int(np.argwhere(np_array[:, 0] == gg))
            np_array[j][6] = np_array[j][6] + 1
            np_array[j][temp[n]] = np_array[j][temp[n]] + 1

    return np_array
def get_matrix(train, n):
    list_pattern = get_list(n)
    np_t = get_array(list_pattern)
    np_t = get_PSP(train, np_t, n)

    for i in range(1, 6):
        np_t[:, i] = np_t[:, i]/np_t[:, 6]
    where_nan = np.isnan(np_t)
    np_t[where_nan] = 0

    np_t = np.delete(np_t, [0, 6], axis=1)

    return np_t


pool_length = 5
maxlen = 25
batch_size = 10
np_epoch = 1500

print('Loading Model...')
'''
model_1 = tf.keras.models.load_model('my_model_1st_p_n.h5')
model_2 = tf.keras.models.load_model('my_model_sec_p_n.h5')
model_3 = tf.keras.models.load_model('GET_15_89.h5')
model_4 = tf.keras.models.load_model('my_model_4th_p_n.h5')
'''
model_1 = tf.keras.models.load_model('Best_Model_1_15_ccpcp_loss.h5')
model_2 = tf.keras.models.load_model('Two_15_selu_500.h5')
model_3 = tf.keras.models.load_model('GET_15_89.h5')
model_4 = tf.keras.models.load_model('Best_Model_Four_selu_45_loss.h5')


print('Loading data...')

n_train_data = get_data('paper_n_train')
no_train_data = get_data('paper_n_o_train')
p_train_data = get_data('paper_p_train')
n_train_data.extend(no_train_data)

random.shuffle(n_train_data)
random.shuffle(p_train_data)

#截取一部分作为测试集
n_test_data = n_train_data[0:100]
n_train_data = n_train_data[100:len(n_train_data)]
p_test_data = p_train_data[0:100]
p_train_data = p_train_data[100:len(p_train_data)]

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

l_test_1 = []
l_test_2 = []
l_test_3 = []
l_test_4 = []
for i in range(0, test_data.shape[0]):
    p_1 = get_matrix(test_data[i], 1)
    p_2 = get_matrix(test_data[i], 2)
    p_3 = get_matrix(test_data[i], 3)
    p_4 = get_matrix(test_data[i], 4)
    l_test_1.append(p_1)
    l_test_2.append(p_2)
    l_test_3.append(p_3)
    l_test_4.append(p_4)
    print(i)
np_test_1 = np.array(l_test_1)
np_test_2 = np.array(l_test_2)
np_test_3 = np.array(l_test_3)
np_test_4 = np.array(l_test_4)



l_train_1 = []
l_train_2 = []
l_train_3 = []
l_train_4 = []
for i in range(0, train_data.shape[0]):
    p_1 = get_matrix(train_data[i], 1)
    p_2 = get_matrix(train_data[i], 2)
    p_3 = get_matrix(train_data[i], 3)
    p_4 = get_matrix(train_data[i], 4)
    l_train_1.append(p_1)
    l_train_2.append(p_2)
    l_train_3.append(p_3)
    l_train_4.append(p_4)
    print(i)
np_train_1 = np.array(l_train_1)
np_train_2 = np.array(l_train_2)
np_train_3 = np.array(l_train_3)
np_train_4 = np.array(l_train_4)

error = 0
error_1 = 0
error_2 = 0
error_3 = 0
error_4 = 0

np_test_1 = np.zeros((200, 5, 5))
np_test_2 = np.zeros((200, 25, 5))
np_test_3 = np.zeros((200, 125, 5))
np_test_4 = np.zeros((200, 625, 5))

np_train_1 = np.zeros((2175, 5, 5))
np_train_2 = np.zeros((2175, 25, 5))
np_train_3 = np.zeros((2175, 125, 5))
np_train_4 = np.zeros((2175, 625, 5))

with open('C:\ECGtest\info\Test_test_1.txt', 'r') as f:
    for i in range(0, 200):
        for j in range(0, 5):
            for k in range(0, 5):
                line = f.readline()
                if not line:
                    break
                a = float(line)
                np_test_1[i][j][k] = a
with open('C:\ECGtest\info\Test_test_2.txt', 'r') as f:
    for i in range(0, 200):
        for j in range(0, 25):
            for k in range(0, 5):
                line = f.readline()
                if not line:
                    break
                a = float(line)
                np_test_2[i][j][k] = a
with open('C:\ECGtest\info\Test_test_3.txt', 'r') as f:
    for i in range(0, 200):
        for j in range(0, 125):
            for k in range(0, 5):
                line = f.readline()
                if not line:
                    break
                a = float(line)
                np_test_3[i][j][k] = a
with open('C:\ECGtest\info\Test_test_4.txt', 'r') as f:
    for i in range(0, 200):
        for j in range(0, 625):
            for k in range(0, 5):
                line = f.readline()
                if not line:
                    break
                a = float(line)
                np_test_4[i][j][k] = a

with open('C:\ECGtest\info\Test_train_1.txt', 'r') as f:
    for i in range(0, 2175):
        for j in range(0, 5):
            for k in range(0, 5):
                line = f.readline()
                if not line:
                    break
                a = float(line)
                np_train_1[i][j][k] = a
with open('C:\ECGtest\info\Test_train_2.txt', 'r') as f:
    for i in range(0, 2175):
        for j in range(0, 25):
            for k in range(0, 5):
                line = f.readline()
                if not line:
                    break
                a = float(line)
                np_train_2[i][j][k] = a
with open('C:\ECGtest\info\Test_train_3.txt', 'r') as f:
    for i in range(0, 2175):
        for j in range(0, 125):
            for k in range(0, 5):
                line = f.readline()
                if not line:
                    break
                a = float(line)
                np_train_3[i][j][k] = a
with open('C:\ECGtest\info\Test_train_4.txt', 'r') as f:
    for i in range(0, 2175):
        for j in range(0, 625):
            for k in range(0, 5):
                line = f.readline()
                if not line:
                    break
                a = float(line)
                np_train_4[i][j][k] = a

np_train_1 = np.expand_dims(np_train_1, axis=3)
np_train_2 = np.expand_dims(np_train_2, axis=3)
np_train_3 = np.expand_dims(np_train_3, axis=3)
np_train_4 = np.expand_dims(np_train_4, axis=3)

np_test_1 = np.expand_dims(np_test_1, axis=3)
np_test_2 = np.expand_dims(np_test_2, axis=3)
np_test_3 = np.expand_dims(np_test_3, axis=3)
np_test_4 = np.expand_dims(np_test_4, axis=3)

test_1 = model_1.predict(np_test_1)
test_2 = model_2.predict(np_test_2)
test_3 = model_3.predict(np_test_3)
test_4 = model_4.predict(np_test_4)
gg_test = (test_1+test_2+test_3+test_4)/4
test = np.hstack((test_1, test_2))
test = np.hstack((test, test_3))
test = np.hstack((test, test_4))
print(test.shape)

train_1 = model_1.predict(np_train_1)
train_2 = model_2.predict(np_train_2)
train_3 = model_3.predict(np_train_3)
train_4 = model_4.predict(np_train_4)
train = np.hstack((train_1, train_2))
train = np.hstack((train, train_3))
train = np.hstack((train, train_4))
train_label = np.array(n_train_label+p_train_label)

model = tf.keras.models.Sequential()
dense = tf.keras.layers.Dense(2, input_shape=[8,])
model.add(dense)
model.add(tf.keras.layers.Activation(activation='softmax'))
model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
)

print('Training...')
filepath="Final_loss-10.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(train, train_label,
          batch_size=batch_size, epochs=np_epoch,
          validation_data=(test, test_label),
          callbacks=callbacks_list)
model.save('final_model_paper_n_r_loss.h5')

'''
for i in range(0, test.shape[0]):

    a = np.expand_dims(test[i], axis=0)
    pre = model_final.predict(a)
    r_1 = round(pre[0][0])
    r_2 = round(pre[0][1])
    gg = np.hstack((r_1, r_2))
    print('pre:', gg, 'label:', test_label[i])
    if r_1 != test_label[i][0] and r_2 != test_label[i][1]:
        error = error+1
    if round(gg_test[i][0]) != test_label[i][0] and round(gg_test[i][1]) != test_label[i][1]:
        error_1 = error_1+1
print('syn:',error/200)
print('avg:',error_1/200)
'''