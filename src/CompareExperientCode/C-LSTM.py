import tensorflow as tf
import os
import numpy as np
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import itertools
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

maxlen = 1920
batch_size = 50
np_epoch = 300
l_train_1 = []
l_test_1 = []
model = keras.models.load_model("LSTM_GG.h5")
model.summary()

print('Loading data...')
# 载入X的数据，分为心律不齐患者和正常人
n_train_data = get_data('paper_n_train_32')
no_train_data = get_data('paper_no_train_32')
p_train_data = get_data('paper_p_train_32')
n_train_data.extend(no_train_data)
#截取一部分作为测试集


n_test_data = n_train_data[0:100]
n_train_data = n_train_data[100:len(n_train_data)]
p_test_data = p_train_data[0:100]
p_train_data = p_train_data[100:len(p_train_data)]

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



test = np.expand_dims(test_data, axis=3)

tt = 0
tt1 = 0
for i in range(0, test.shape[0]):

    a = np.expand_dims(test[i], axis=0)
    pre = model.predict(a)
    r_1 = round(pre[0][0])
    r_2 = round(pre[0][1])
    gg = np.hstack((r_1, r_2))
    print('pre:', gg, 'label:', test_label[i])
    if r_1 != test_label[i][0] or r_2 != test_label[i][1]:
        error = error + 1
    if r_1 == 0 and r_2 == 1 and test_label[i][0] == 0 and test_label[i][1] == 1:
        tt = tt + 1
    if r_1 == 1 and r_2 == 0 and test_label[i][0] == 1 and test_label[i][1] == 0:
        tt1 = tt1 + 1

print('syn:', 1-error/200)
print('TN:', tt)
print('TN rate:', tt/100)
print(('TP:', tt1))
print('TP rate:', tt1/100)
print('sen:', tt1/(tt1+100-tt))
print('spe:', tt/(tt+100-tt1))



print('Build model...')
model = keras.Sequential()
conv1 = keras.layers.Conv1D(
    filters=3,
    kernel_size=60,
    padding='same',
    strides=1,
    input_shape=[maxlen,1],
    activation='relu'
)

pool1 = keras.layers.MaxPool1D(
    pool_size=2,
    strides=2,
    padding='valid'
)
conv2 = keras.layers.Conv1D(
    filters=6,
    kernel_size=30,
    padding='same',
    strides=1,
    activation='relu'
)
pool2 = keras.layers.MaxPool1D(
    pool_size=2,
    strides=2,
    padding='valid'
)
conv3 = keras.layers.Conv1D(
    filters=6,
    kernel_size=15,
    padding='same',
    strides=1,
    activation='relu'
)
pool3 = keras.layers.MaxPool1D(
    pool_size=2,
    strides=2,
    padding='valid'
)
LSTM=keras.layers.LSTM(60)
D1=keras.layers.Dense(60,activation='relu')
D2=keras.layers.Dense(30,activation='relu')
D3=keras.layers.Dense(2, activation='softmax')
model.add(conv1)
print(conv1.output_shape)
model.add(pool1)
print(pool1.output_shape)
model.add(conv2)
print(conv2.output_shape)
model.add(pool2)
print(pool2.output_shape)
model.add(conv3)
print(conv3.output_shape)
model.add(pool3)
print(pool3.output_shape)
model.add(LSTM)
print(LSTM.output_shape)
model.add(D1)
print(D1.output_shape)
model.add(D2)
print(D2.output_shape)
model.add(D3)
print(D3.output_shape)
model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
)

filepath="CNN-LSTM_32_GG.h5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

print('Training...')
model.fit(train_data, train_label,
          batch_size=batch_size, epochs=np_epoch,
          validation_data=(test_data, test_label))


print(model.summary())
'''
