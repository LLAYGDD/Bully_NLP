from __future__ import print_function
"""
基于词的独立循环神经网络文本分类识别
"""

# import jie_test
import sys
import os
import numpy as np
np.random.seed(2*30)
#导入数据包
from keras.models import Model
from keras.layers import Dense
from keras.layers import Embedding,LSTM,Bidirectional,Conv1D
from keras.callbacks import Callback,ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import time
from keras.layers import Input
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from untils.graph_convolution import GraphConv
from untils.graph_utils import generate_Q
from keras.callbacks import ModelCheckpoint
from ChineseTone import *


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
embedding_dim=50
validation_split=0.2
max_nb_words=0
nb_filter=64
lstm_output_size = 128


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

import jieba
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
            seg_list = jieba.cut(text.strip().replace(' ', ''), cut_all=False)
            new_content = " ".join(seg_list)
            seg_list1=PinyinHelper.convertToPinyinFromSentence(new_content)
            # print(seg_list1)
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
conv1 = Conv1D(filters=64, kernel_size=3, strides=1,
               padding='same', activation='relu',
               name='conv1')(embeddings_sequences)
output=Bidirectional(LSTM(lstm_output_size,
                          dropout=0.25,
                          recurrent_dropout=0.4))(conv1)
# print(output)
# from keras.layers import Reshape
# output=Reshape((128,2))(output)
# output=Bidirectional(LSTM(lstm_output_size,
#                           dropout=0.5,
#                           recurrent_dropout=0.5))(output)
# output=Reshape((128,2))(output)
# output=Bidirectional(LSTM(lstm_output_size,
#                           dropout=0.25,
#                           recurrent_dropout=0.4))(output)
# print(output)

# output=Dense(64,activation='relu')(output)
# output=Dense(32,activation='relu')(output)
output=Dense(1,activation='sigmoid')(output)
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
                    epochs=3,
                    batch_size=64,
                    validation_data=(x_test, y_test),
                    callbacks=[metrics,checkpoint])

end_time=time.time()
print("数据处理时间=：",mid_time-start_time)
print("模型运行时间=：",end_time-mid_time)

