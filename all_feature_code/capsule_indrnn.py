from __future__ import print_function
"""
数据包的导入和创建
"""
import sys
import keras
import os
import jie_test
from keras.layers.core import *
from keras.utils.np_utils import to_categorical
from keras.layers import concatenate
from keras.layers import MaxPooling1D
from keras.layers import Dense
from capsule_layer import CategoryCap, PrimaryCap, Length
from keras.layers import Bidirectional,Embedding,BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Conv1D
from keras.models import Model
from untils.IndRNN_Layer import IndRNN
from untils.graph_convolution import GraphConv
from untils.graph_utils import generate_Q
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from ChineseTone import *
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
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
text_data_dir='E:/DeepLearning/bully_code/diyu/diyu/'

#定义文本类型，并做数据划分
max_sequence_length=1200
max_sequence_length1=600
embedding_dim=300
validation_split=0.2
max_nb_words=0
epoch=150
MAX_NB_WORDS = 0


"""
汉字加载
"""

labels_index = {}
labels2 = []
i=1

texts2=[]
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
graphconv=GraphConv(filters=64,neighbors_ix_mat=q_mat_layer1,num_neighbors=12,activation='relu')(te_ind)
print(graphconv)
graphconv=GraphConv(filters=64,neighbors_ix_mat=q_mat_layer1,num_neighbors=12,activation='relu')(graphconv)
print(graphconv)
conv1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='conv1')(graphconv)
conv1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(conv1)
conv1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(conv1)
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

wo_ind = Bidirectional(IndRNN(32, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=True))(wo_ind)   #97%
wo_ind = Bidirectional(IndRNN(32, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=True))(wo_ind)   #97%
wo_ind = Bidirectional(IndRNN(32, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=True))(wo_ind)   #97%

# wo_ind=Flatten()(wo_ind)
wo_ind=Dense(100,activation='relu')(wo_ind)

k = 12

output = concatenate([texture,wo_ind])
print("output:",output)




"""
胶囊网络的加载
"""
conv1= Conv1D(kernel_size=3, filters=k, padding='same',activation='tanh', strides=1)(output)
conv1= Conv1D(kernel_size=3, filters=k, padding='same',activation='tanh', strides=1)(conv1)
conv1= Conv1D(kernel_size=3, filters=k, padding='same',activation='tanh', strides=1)(conv1)
print('conv1',conv1)
"""
主胶囊层定义
"""
primary_caps = PrimaryCap(conv1, dim_vector=dim_capsule1, n_channels=32, kernel_size=9, strides=2, padding='same',
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

preds = Dense(1, activation='sigmoid')(category_caps)
print("training>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
model = Model(inputs=[ sequence_input2,sequence_input4], outputs=[preds])
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['acc'])


checkpoint = ModelCheckpoint(filepath="best_model.h5",#(就是你准备存放最好模型的地方),
                             monitor='val_acc',#(或者换成你想监视的值,比如acc,loss, val_loss,其他值应该也可以,还没有试),
                             verbose=1,#(如果你喜欢进度条,那就选1,如果喜欢清爽的就选0,verbose=冗余的),
                             save_best_only='True',#(只保存最好的模型,也可以都保存),
                             save_weights_only='True',
                             mode='max',#(如果监视器monitor选val_acc, mode就选'max',如果monitor选acc,mode也可以选'max',如果monitor选loss,mode就选'min'),一般情况下选'auto',
                        period=1)#(checkpoints之间间隔的epoch数)
#损失不下降，则自动降低学习率
lrreduce=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

import time
fit_start = time.clock()
history=model.fit([x_train2,x_train4], y_train2,
          batch_size=128,
          epochs=epoch,
          verbose=2,
          shuffle =True,
          validation_data=([x_val2,x_val4], y_val2),
            callbacks = [checkpoint])
fit_end = time.clock()
print("train time is: ",fit_end-fit_start)
model.load_weights('best_model.h5')
t_start = time.clock()
scores =  model.evaluate([x_val2,x_val4], y_val2,verbose=0)
t_end = time.clock()
print('Test loss :',scores[0])
print('Test accuracy :',scores[1])
print("test time is: ",t_end-t_start)

y_pred_class = model.predict([x_val2,x_val4]).reshape(len(y_val2))
#将y_val变为list类型             重要！！！！！！！！！！！！！！
y_val = np.array(y_val2)
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
def threshold_y_val_pred(threshold,true_label,y_pred_class):
    pred_label = [int(item>threshold) for item in y_pred_class]
    # pred_label = y_pred_class
    from sklearn import metrics
    confusion = metrics.confusion_matrix(true_label, pred_label)
    # TP = confusion[0, 0]   #TP
    # TN = confusion[1, 1]    #TN
    # FP = confusion[1, 0]    #FP
    # FN = confusion[0, 1]    #FN
    TP = confusion[1, 1]   #TP
    TN = confusion[0, 0]    #TN
    FP = confusion[0, 1]    #FP
    FN = confusion[1, 0]    #FN
    Accuracy = metrics.accuracy_score(true_label, pred_label)
    Error = 1 - metrics.accuracy_score(true_label, pred_label)
    Precision = metrics.precision_score(true_label, pred_label)
    Recall = metrics.recall_score(true_label, pred_label)
    F1 = metrics.f1_score(true_label, pred_label)
    AUC = metrics.roc_auc_score(true_label, pred_label)
    TPR = float(TP / (TP+FN))
    FPR = float(FP / (TN+FP))
    micro_f1 = f1_score(true_label, pred_label, average='micro')
    macro_f1 = f1_score(true_label, pred_label, average='macro')
    print('Macro-F1（Macro-average）: {}'.format(macro_f1))
    print('Micro-F1（Micro-average）: {}'.format(micro_f1))
    print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN,
          "\nAccuracy:", Accuracy, "Error:", Error,
          "\nPrecision:", Precision, "Recall:", Recall, "F1:", F1,
          "\nAUC:", AUC,' TPR',TPR,'FPR',FPR,
          '\nmicro_f1',micro_f1,'macro_f1',macro_f1,"threshold:", threshold)
    print("pred_label",pred_label)
    return TP,TN,FP,FN,Accuracy,Error,Precision,Recall,F1,AUC,TPR,FPR,micro_f1,macro_f1,threshold,pred_label

TP, TN, FP, FN, Accuracy, Error, Precision, Recall, F1, AUC, TPR,FPR,micro_f1,macro_f1,thre, y_pre = threshold_y_val_pred(0.1, y_val, y_pred_class)
threshold = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for thr in range(len(threshold)):
    print("阈值",threshold[thr])
    TP2, TN2, FP2, FN2, Accuracy2, Error2, Precision2, Recall2, F12, AUC2 ,TPR2,FPR2,micro_f12,macro_f12,thre2,y_pre2= threshold_y_val_pred(threshold[thr], y_val, y_pred_class)
    if F12>F1:
        TP, TN, FP, FN, Accuracy, Error, Precision, Recall, F1, AUC, TPR,FPR,micro_f1,macro_f1, thre,y_pre = TP2, TN2, FP2, FN2, Accuracy2, Error2, Precision2, Recall2, F12, AUC2 ,TPR2,FPR2,micro_f12,macro_f12,thre2,y_pre2
print("###################################################################")
print("###################################################################")
print("最大F值时，阈值是：",thre)
print("TP:",TP,"TN:", TN,"FP:", FP,"FN:" ,FN,
      "\nAccuracy:", Accuracy,"Error:", Error,
      "\nPrecision:",Precision,"Recall:", Recall, "F1:",F1,
      "\nAUC:", AUC,' TPR',TPR,'FPR',FPR,
          '\nmicro_f1',micro_f1,'macro_f1',macro_f1)
