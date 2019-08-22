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

    with open(filename + '.txt', 'r') as f:
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
batch_size = 50
np_epoch = 100

print('Loading Model...')
model_1 = tf.keras.models.load_model('Best_Model_1_15_ccpcp_loss.h5')
model_2 = tf.keras.models.load_model('Two_15_selu_500.h5')
model_3 = tf.keras.models.load_model('GET_15_89.h5')
model_4 = tf.keras.models.load_model('Best_Model_Four_selu_45_loss.h5')
model = tf.keras.models.load_model('Final_loss-10.h5')
print('Loading data...')
# 载入X的数据，分为心律不齐患者和正常人

n_train_data = get_data('paper_n_train_HHH')
no_train_data = get_data('paper_no_train_HHH')
p_train_data = get_data('paper_p_train_HHH')
n_train_data.extend(no_train_data)
l_train=[]
#截取一部分作为测试集
random.shuffle(n_train_data)
random.shuffle(p_train_data)
#获得标签
n_train_label = get_label_n(n_train_data)
p_train_label = get_label_p(p_train_data)
print(len(n_train_label))
print(len(p_train_label))
#形成数组
train_data = np.array(n_train_data+p_train_data)
train_label = np.array(n_train_label+p_train_label)

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

np_train_1 = np.expand_dims(np_train_1, axis=3)
np_train_2 = np.expand_dims(np_train_2, axis=3)
np_train_3 = np.expand_dims(np_train_3, axis=3)
np_train_4 = np.expand_dims(np_train_4, axis=3)

train_1 = model_1.predict(np_train_1)
train_2 = model_2.predict(np_train_2)
train_3 = model_3.predict(np_train_3)
train_4 = model_4.predict(np_train_4)
train = np.hstack((train_1, train_2))
train = np.hstack((train, train_3))
train = np.hstack((train, train_4))


count=0
count1=0
count2=0
count3=0
count4=0
error1=0
error2=0
error3=0
error4=0
for i in range(0,train.shape[0]):
    a = np.expand_dims(train[i], axis=0)
    pre = model.predict(a)
    r_1 = round(pre[0][0])
    r_2 = round(pre[0][1])
    if r_1 != train_label[i][0] or r_2 != train_label[i][1]:
        print('gg:',r_1,r_2,'ture:',train_label[i])
        count = count+1
    if round(train_1[i][0]) != train_label[i][0] or round(train_1[i][1]) != train_label[i][1]:
        error1 = error1+1
    if round(train_2[i][0]) != train_label[i][0] or round(train_2[i][1]) != train_label[i][1]:
        error2 = error2+1
    if round(train_3[i][0]) != train_label[i][0] or round(train_3[i][1]) != train_label[i][1]:
        error3 = error3+1
    if round(train_4[i][0]) != train_label[i][0] or round(train_4[i][1]) != train_label[i][1]:
        error4 = error4+1
    if r_1 == 1 and train_label[i][0] == 1:
        count1 = count1+1
    if r_1 == 1 and train_label[i][0] == 0:
        count2 = count2+1
    if r_1 == 0 and train_label[i][0] == 1:
        count3 = count3+1
    if r_1 == 0 and train_label[i][0] == 0:
        count4 = count4+1
print(count/train.shape[0])
print(count1)
print(count2)
print(count3)
print(count4)
print(1-error1/train.shape[0])
print(1-error2/train.shape[0])
print(1-error3/train.shape[0])
print(1-error4/train.shape[0])

