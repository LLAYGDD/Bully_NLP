from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

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


for la in labels:
    if la[0]==0:
        label.append(la[0])
    else:
        label.append(la[0])
print("标签表示-labels==:>",label)




# digits = load_digits()
X_tsne = TSNE(n_components=2, random_state=33).fit_transform(data)
X_pca = PCA(n_components=2).fit_transform(labels)

font = {"color": "darkred",
        "size": 13,
        "family" : "serif"}

plt.style.use("dark_background")
plt.figure(figsize=(8.5, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, alpha=0.6,
            cmap=plt.cm.get_cmap('rainbow', 10))
plt.title("t-SNE", fontdict=font)
cbar = plt.colorbar(ticks=range(10))
cbar.set_label(label='digit value', fontdict=font)
plt.clim(-0.5, 9.5)
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, alpha=0.6,
            cmap=plt.cm.get_cmap('rainbow', 10))
plt.title("PCA", fontdict=font)
cbar = plt.colorbar(ticks=range(10))
cbar.set_label(label='digit value', fontdict=font)
plt.clim(-0.5, 9.5)
plt.tight_layout()
