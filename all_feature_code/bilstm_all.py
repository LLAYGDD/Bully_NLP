from __future__ import print_function
"""
数据包的导入和创建
"""
import sys
import keras
import os
import jie_test
import time
from keras.layers.core import *
from keras.utils.np_utils import to_categorical
from keras.layers import concatenate
from keras.layers import MaxPooling1D
from keras.layers import Dense,LSTM
from capsule_layer import  Length
from keras.layers import Bidirectional,Embedding,BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Conv1D
from keras.models import Model
from untils.graph_convolution import GraphConv
from untils.graph_utils import generate_Q
from keras.callbacks import Callback,ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
from ChineseTone import *
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
胶囊网络的单元定义
"""
num_capsule1 = 32
dim_capsule1 = 64
num_routing = 3

"""
图神经网络的定义
"""
num_neighbors =2
q_mat_layer1 = generate_Q(4,4)
q_mat_layer1 = np.argsort(q_mat_layer1,1)[:,-num_neighbors:]
q_mat_layer2 = generate_Q(2,4)
q_mat_layer2 = np.argsort(q_mat_layer2,1)[:,-num_neighbors:]

"""
数据加载和定义声明
"""
text_data_dir='D:/DeepLearning/bully_code/diyu/diyu/'

#定义文本类型，并做数据划分
max_sequence_length=50
max_sequence_length1=25
embedding_dim=50
validation_split=0.2
max_nb_words=0
MAX_NB_WORDS = 0

start_time=time.time()

"""
汉字加载
"""

labels_index = {}
labels2 = []
i=1

texts2=[]


"""
定义的可是化
"""
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('- val_f1: %.4f - val_precision: %.4f - val_recall: %.4f'%(_val_f1, _val_precision, _val_recall))
        return


for name in sorted(os.listdir(text_data_dir)):
    path = os.path.join(text_data_dir, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            if sys.version_info < (3,):
                f = open(fpath)
            else:
                f = open(fpath, encoding='gbk',errors='ignore')
            t = f.read()#此时t是str类型
            seg_list=jie_test.cut(t.strip().replace(' ', ''), cut_all=False)
            new_content = " ".join(seg_list)
            # t = t.split()
            texts2.append(new_content)
            f.close()
            labels2.append(label_id)
for tes in texts2:
    print("------>>")
    print(tes)
print('Found %s texts.' % len(texts2))


tokenizer2 = Tokenizer(num_words=MAX_NB_WORDS,filters="",oov_token="unk")#filters =‘’ 表示保留特殊字符
tokenizer2.fit_on_texts(texts2)
sequences2 = tokenizer2.texts_to_sequences(texts2)
word_index2 = tokenizer2.word_index
print(word_index2)
print('Found %s unique tokens.' % len(word_index2))
data2 = pad_sequences(sequences2,maxlen=max_sequence_length)#, maxlen=MAX_SEQUENCE_LENGTH
labels2 = to_categorical(np.asarray(labels2))
print('Shape of data tensor:', data2.shape)
print('Shape of label tensor:', labels2.shape)

indices = np.arange(data2.shape[0])
np.random.shuffle(indices)
data2 = data2[indices]
labels2 = labels2[indices]

nb_validation_samples = int(validation_split * data2.shape[0])
label2=[]
for la in labels2:
    if la[0]==0:
        label2.append(la[0])
    else:
        label2.append(la[0])
print("标签表示-labels==:>",label2)

"""
拼音加载
"""

labels_index = {}
labels4 = []
i=1
texts4=[]
for name in sorted(os.listdir(text_data_dir)):
    path = os.path.join(text_data_dir, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            if sys.version_info < (3,):
                f = open(fpath)
            else:
                f = open(fpath, encoding='gbk',errors='ignore')
            t = f.read()#此时t是str类型
            seg_list = jie_test.cut(t.strip().replace(' ', ''), cut_all=False)
            new_content = " ".join(seg_list)
            seg_list1=PinyinHelper.convertToPinyinFromSentence(new_content)
            print(seg_list1)
            # t = t.split()
            texts4.append(seg_list1)
            f.close()
            labels4.append(label_id)

tokenizer4 = Tokenizer(num_words=MAX_NB_WORDS,filters="",oov_token="unk")#filters =‘’ 表示保留特殊字符
tokenizer4.fit_on_texts(texts4)
sequences4 = tokenizer4.texts_to_sequences(texts4)
word_index4 = tokenizer4.word_index
print(word_index4)
print('Found %s unique tokens.' % len(word_index4))
data4 = pad_sequences(sequences4,maxlen=max_sequence_length1)#, maxlen=MAX_SEQUENCE_LENGTH
labels4 = to_categorical(np.asarray(labels4))
print('Shape of data tensor:', data4.shape)
print('Shape of label tensor:', labels4.shape)

data4 = data4[indices]
labels4 = labels4[indices]
nb_validation_samples = int(validation_split * data4.shape[0])

label4=[]
for la in labels4:
    if la[0]==0:
        label4.append(la[0])
    else:
        label4.append(la[0])
print("输出=：",label4)

"""
训练集和测试集的加载
"""
x_train2 = data2[:-nb_validation_samples]
y_train2 = label2[:-nb_validation_samples]
x_val2 = data2[-nb_validation_samples:]
y_val2 = label2[-nb_validation_samples:]


x_train4 = data4[:-nb_validation_samples]
y_train4 = label4[:-nb_validation_samples]
x_val4 = data4[-nb_validation_samples:]
y_val4 = label4[-nb_validation_samples:]

mid_time=time.time()

"""
多种特征的合并
"""

sequence_input2 = Input(shape=(max_sequence_length,), dtype='int32')
te_ind = Embedding(len(word_index2) + 1,
                            embedding_dim,
                            input_length=max_sequence_length,
                            trainable=True)(sequence_input2)
"""
图卷积神经网络的加载,即字符特征的获取
"""
graphconv=GraphConv(filters=64,neighbors_ix_mat=q_mat_layer1,num_neighbors=12,activation='relu')
print(graphconv)

conv1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='conv1')(te_ind)
# conv1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(conv1)
# conv1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(conv1)
conv1 = MaxPooling1D()(conv1)
conv1 = BatchNormalization()(conv1)
texture = Dropout(0.3)(conv1)
print()
texture=Dense(100,activation='relu')(texture)


sequence_input4 = Input(shape=(max_sequence_length1,), dtype='int32')
wo_ind = Embedding(len(word_index4) + 1,
                            embedding_dim,
                            input_length=max_sequence_length1,
                            trainable=True)(sequence_input4)
lstm_output_size=32
wo_ind = Bidirectional(LSTM(lstm_output_size,dropout=0.25,return_sequences=True))(wo_ind)

wo_ind=Dense(100,activation='relu')(wo_ind)

k = 12

output = concatenate([texture,wo_ind])
print("output:",output)
output =Bidirectional(LSTM(lstm_output_size, dropout=0.5,
                 return_sequences=True))(output)

category_caps = Length(name='out_caps')(output)

preds = Dense(1, activation='sigmoid')(category_caps)
print("training>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
model = Model(inputs=[ sequence_input2,sequence_input4], outputs=[preds])
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['acc'])

from keras.optimizers import Adam
model.compile(loss='binary_crossentropy',optimizer=Adam(0.0001),metrics=['accuracy'])
checkpoint_filepath = 'D:/DeepLearning/bully_code/diyu/indrnn.h5'
checkpoint = ModelCheckpoint(checkpoint_filepath,
                             monitor='acc',
                             verbose=0,
                             save_best_only=True,
                             mode='max')


metrics = Metrics()
history=model.fit([x_train2,x_train4], y_train2,
          batch_size=64,
          epochs=4,
          verbose=2,
          shuffle =True,
          validation_data=([x_val2,x_val4], y_val2),
            callbacks = [metrics,checkpoint])
end_time=time.time()
print("数据处理时间=：",mid_time-start_time)
print("模型运行时间=：",end_time-mid_time)

