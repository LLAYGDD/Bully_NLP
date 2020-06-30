from __future__ import print_function
"""
基于词的独立循环神经网络文本分类识别
"""

import jie_test
import sys
import os
import numpy as np
np.random.seed(2*30)
#导入数据包
from keras.models import Model

from keras_self_attention import SeqSelfAttention
from keras.callbacks import Callback,ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import time
from keras.layers import Input,Conv1D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from untils.IndRNN_Layer import IndRNN

from untils.graph_convolution import GraphConv
from untils.graph_utils import generate_Q
from keras.callbacks import ModelCheckpoint
from ChineseTone import *
from capsule_layer import CategoryCap, PrimaryCap, Length
from keras.layers import Dropout,Embedding,BatchNormalization


"""
图神经网络的定义
"""
num_neighbors =2
q_mat_layer1 = generate_Q(4,4)
q_mat_layer1 = np.argsort(q_mat_layer1,1)[:,-num_neighbors:]
q_mat_layer2 = generate_Q(2,4)
q_mat_layer2 = np.argsort(q_mat_layer2,1)[:,-num_neighbors:]




label=[]
filters=64
filter_size=5
pool_size=2
filter_size1=3
filters1=32
LR=0.001
pool_size1=2
num_class=1
max_features = 20000
batch_size = 128
np_epoch=15
#词性划分的定义与声明
texts = []
labels_index = {}
labels = []
embeddings_index = {}
glove_dir=''
#定义文本类型，并做数据划分
max_sequence_length=100
embedding_dim=150
validation_split=0.45
max_nb_words=0
nb_filter=64
lstm_output_size = 128

num_capsule1 = 8
dim_capsule1 = 16
num_routing = 1

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

"""
效率测试
"""
start_time=time.time()


text_data_dir='E:/DeepLearning/bully_code/diyu/diyu/'
save_best_model_file='E:/DeepLearning/bully_code/diyu/model_indrnn'
text_data=os.listdir(text_data_dir)

for textName in sorted(text_data):
    textpath = os.path.join(text_data_dir, textName)
    if os.path.isdir(textpath):
        label_id = len(labels_index)
        labels_index[textName] = label_id
        for wordname in sorted(os.listdir(textpath)):
            wordpath = os.path.join(textpath, wordname)
            if sys.version_info < (3,):
                f = open(wordpath)
            else:
                f = open(wordpath, encoding='gbk',errors='ignore')

            text = f.read()
            seg_list = jie_test.cut(text.strip().replace(' ', ''), cut_all=False)
            new_content = " ".join(seg_list)
            seg_list1=PinyinHelper.convertToPinyinFromSentence(new_content)
            print(seg_list1)
            texts.append(seg_list1)
            f.close()
            labels.append(label_id)
for tes in texts:
    print("------>>")
    print(tes)
print('Found %s texts.' % len(texts))



tokenizer = Tokenizer(num_words=max_nb_words, filters="",oov_token="unk")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=max_sequence_length)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(validation_split * data.shape[0])


for la in labels:
    if la[0]==0:
        label.append(la[0])
    else:
        label.append(la[0])
print("标签表示-labels==:>",label)


"""
测试集和训练集的输入
"""

x_train = data[:-nb_validation_samples]
y_train = label[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = label[-nb_validation_samples:]

"""
文本数据表示完成时间
"""
mid_time=time.time()
print("初始时间=：",mid_time-start_time)
"""
模型的建立
"""
print('Build model...')
earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0)
saveBestModel = ModelCheckpoint(save_best_model_file,
                                monitor = 'val_loss', verbose=0,
                                save_best_only = True,
                                save_weights_only = True)

embeddings_layer=Embedding(len(word_index)+1,embedding_dim,
                           input_length=max_sequence_length,trainable=True)



inputs=Input(shape=(max_sequence_length,),dtype='int32',name='input')
embeddings_sequences=embeddings_layer(inputs)
graphconv=GraphConv(filters=64,neighbors_ix_mat=q_mat_layer1,num_neighbors=12,activation='relu')
self_attention=SeqSelfAttention(attention_activation='relu',
                                name='self_attention')(embeddings_sequences)
conv1 = Conv1D(filters=128, kernel_size=3, strides=1,
               padding='same', activation='relu',
               name='conv1')(self_attention)
ind_rnn_layer_1 = IndRNN(200,
                 recurrent_clip_min=-1,
                 recurrent_clip_max=-1,
                 dropout=0.7,
                 recurrent_dropout=0.0,
                 return_sequences=True)(conv1)
output =IndRNN(100,
                 recurrent_clip_min=-1,
                 recurrent_clip_max=-1,
                 dropout=0.7,
                 recurrent_dropout=0.0,
                 return_sequences=True)(ind_rnn_layer_1)


print(output)


dropout=Dropout(0.5)(output)
primary_caps = PrimaryCap(dropout, dim_vector=dim_capsule1, n_channels=3, kernel_size=9, strides=2, padding='same',
                          name="primary_caps")
primary_caps = BatchNormalization()(primary_caps)
primary_caps = Dropout(0.3)(primary_caps)
# Layer 3: Capsule layer. Routing algorithm works here.
category_caps = CategoryCap(num_capsule=num_capsule1, dim_vector=dim_capsule1, num_routing=num_routing, name='category_caps')(
    primary_caps)

category_caps = BatchNormalization()(category_caps)
category_caps = Dropout(0.3)(category_caps)

print('category_caps',category_caps)

category_caps = Length(name='out_caps')(category_caps)
from keras.layers import Dense
output=Dense(1,activation='sigmoid')(category_caps)
model=Model(inputs=inputs,outputs=[output])
model.summary()
model.compile(loss='binary_crossentropy',optimizer=Adam(0.0001),metrics=['accuracy'])
checkpoint_filepath = 'E:/DeepLearning/bully_code/diyu/indrnn.h5'
checkpoint = ModelCheckpoint(checkpoint_filepath,
                             monitor='acc',
                             verbose=0,
                             save_best_only=True,
                             mode='max')


metrics = Metrics()
history = model.fit(x_train, y_train,
                    epochs=14,
                    batch_size=128,
                    validation_data=(x_test, y_test),
                    callbacks=[metrics,checkpoint])

end_time=time.time()
print("数据处理时间=：",mid_time-start_time)
print("运行时间=：",end_time-mid_time)
