from __future__ import print_function
"""
基本测试
"""
import sys
import os
import numpy as np
import jie_test
import time
import pandas as pd
from ChineseTone import *
from __init__ import labels_index,labels,texts
from __init__ import nb_filter, filter_length,batch_size,nb_epoch
from keras.layers import Merge
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import  Dense,Dropout,Activation
from keras.layers import Embedding
from keras.layers import Convolution1D,GlobalMaxPooling1D
from keras.layers import LSTM
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras_self_attention import SeqSelfAttention

start_time=time.time()


"""
数据的基本调用
"""

dir_data="E:/DeepLearning/bully_code/diyu/diyu/"
print(dir_data)

for dir_dpin in sorted(os.listdir(dir_data)):
    path_pin=os.path.join(dir_data,dir_dpin)
    print(path_pin)
    if os.path.isdir(path_pin):
        label_id=len(labels_index)
        labels_index[dir_dpin]=label_id
        for filename in sorted(os.listdir(path_pin)):
            filepath=os.path.join(path_pin,filename)
            if sys.version_info<(3,):
                f=open(filepath)
            else:
                f=open(filepath,encoding='gbk',errors='ignore')
            text=f.read()
            seg_list = jie_test.cut(text.strip().replace(' ', ''), cut_all=False)
            new_content = " ".join(seg_list)
            seg_list1=PinyinHelper.convertToPinyinFromSentence(new_content)
            print(seg_list1)

            texts.append(seg_list1)

            #
            #
            #
            # print(new_content)
            # texts.append(new_content)
            f.close()
            labels.append(label_id)
for tes in texts:
    print(">>>>")
    print(tes)
print('Found %s texts.' % len(texts))


MAX_SEQUENCE_LENGTH = 100 #长度
EMBEDDING_DIM = 150 #维度
VALIDATION_SPLIT = 0.1 #数据集切分
MAX_NB_WORDS=20


tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters="",oov_token="unk")#filters =‘’ 表示保留特殊字符
tokenizer.fit_on_texts(texts)
# tokenizer.fit_on_texts(new_content)

# for text in texts:
#     print(text)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))
print(data)
#print(labels)
print(type(data))
print(type(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
label=[]
for la in labels:
    # print(">>>>>>>>>>>")
    if la[0]==0:
        label.append(la[0])
    else:
        label.append(la[0])
    # print(la)
print("label:>>>",label)
x_train = data[:-nb_validation_samples]
y_train = label[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = label[-nb_validation_samples:]
embeddings_index = {}
GLOVE_DIR=''


mid_time=time.time()
print("初始时间=：",mid_time-start_time)
print('Build model...')




"""
并联左右分支
"""

right_branch=Sequential()
right_branch.add(Embedding(len(word_index)+1,
                    EMBEDDING_DIM,
                    input_length=MAX_SEQUENCE_LENGTH
                    ))
# right_branch.add(MultiHead([Convolution1D(
#                         nb_filter=nb_filter,
#                         filter_length=filter_length,
#                         border_mode='valid',
#                         activation='relu',
#                         subsample_length=1)]))
"""
模型的建立
"""
right_branch.add(SeqSelfAttention(attention_activation='sigmoid'))
right_branch.add(Convolution1D(
                        nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))








left_branch = Sequential()
right_branch.add(SeqSelfAttention(attention_activation='sigmoid'))
left_branch.add(Embedding(len(word_index)+1,
                    EMBEDDING_DIM,
                    input_length=MAX_SEQUENCE_LENGTH
                    ))

left_branch.add(LSTM(250,return_sequences=False))

merged = Merge([left_branch,right_branch], mode='dot',output_shape=lambda x: x[0])

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(1))
final_model.add(Activation('sigmoid'))
print('compile...')
final_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('fit...')
final_model.fit([x_train,x_train], y_train,
          batch_size=1,
          nb_epoch=nb_epoch,
          validation_data=([x_test,x_test], y_test)
          )

score, acc = final_model.evaluate([x_test,x_test], y_test,batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)




