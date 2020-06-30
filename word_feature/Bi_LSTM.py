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
from keras.layers import Dense
from keras.layers import Embedding,LSTM,Bidirectional
from keras.callbacks import Callback,ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import time
from keras.layers import Input
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam




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

def show_train_history(train_history, train_metrics, validation_metrics):
    plt.plot(train_history.history[train_metrics])
    plt.plot(train_history.history[validation_metrics])
    plt.title('Train History')
    plt.ylabel(train_metrics)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')

# 显示训练过程
def plot(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    show_train_history(history, 'acc', 'val_acc')
    plt.subplot(1, 2, 2)
    show_train_history(history, 'loss', 'val_loss')
    plt.show()

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    predictions = model.predict_classes(x_val)
    truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))


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


text_data_dir='D:/DeepLearning/bully_code/diyu/diyu/'
save_best_model_file='D:/DeepLearning/bully_code/diyu/model_indrnn'
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
            texts.append(new_content)
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
# print("初始时间=：",mid_time-start_time)
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
output=Bidirectional(LSTM(lstm_output_size,
                          dropout=0.25,
                          recurrent_dropout=0.4))(embeddings_sequences)

from keras.layers import Reshape
output=Reshape((256,1))(output)
output=Bidirectional(LSTM(64,
                          dropout=0.25,
                          recurrent_dropout=0.4))(output)

# output=Reshape((128,1))(output)
# output=Bidirectional(LSTM(64,
#                           dropout=0.25,
#                           recurrent_dropout=0.4))(output)

# output=Dense(64,activation='relu',name='dense1')(output)
# output=Dense(32,activation='relu',name='dense2')(output)
output=Dense(128,activation='relu',name='dense3')(output)

print(output)

output=Dense(1,activation='sigmoid')(output)
model=Model(inputs=inputs,outputs=[output])
model.summary()
model.compile(loss='binary_crossentropy',optimizer=Adam(0.01),metrics=['accuracy'])
checkpoint_filepath = 'D:/DeepLearning/bully_code/diyu/indrnn.h5'
checkpoint = ModelCheckpoint(checkpoint_filepath,
                             monitor='acc',
                             verbose=0,
                             save_best_only=True,
                             mode='max')


metrics = Metrics()
history = model.fit(x_train, y_train,
                    epochs=1,
                    batch_size=128,
                    validation_data=(x_test, y_test),
                    callbacks=[metrics,checkpoint])


print("数据处理时间=：",mid_time-start_time)
end_time=time.time()

print("总时间=：",end_time-mid_time)

# plt.plot(history.history['train_acc'],'b--')
# plt.plot(history.history['test_acc'],'y-')
# plt.plot(metrics.val_f1s,'r.-')
# plt.plot(metrics.val_precisions,'g-')
# plt.plot(metrics.val_recalls,'c-')
# plt.title('D model report')
# plt.ylabel('evaluation')
# plt.xlabel('epoch')
# plt.legend(['train_accuracy',
#             'test_accuracy',
#             'test_f1-score',
#             'test_precisions',
#             'test_recalls'],
#            loc='lower right')
# plt.savefig(save_best_model_file+"/"+'result_acc.png')
# plt.show()
