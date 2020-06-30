from __future__ import print_function
"""
基本测试
"""
__author__='lvyalong'

import sys
from imp import reload
reload(sys)
from pinyin_code import fenci


texts=[]
labels_index={}
labels=[]

batch_size = 32
embedding_dims = 50
nb_filter = 250
filter_length = 3
nb_epoch = 2
