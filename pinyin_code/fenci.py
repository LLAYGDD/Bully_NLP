from __future__ import print_function
"""
转换
"""

import os
import sys
from ChineseTone import *
import jieba

texts = []
labels_index = {}
labels = []

dir_data='D:/DeepLearning/Meng_paper/data/diyu/'
print(dir_data)
dir_data_path=os.listdir(dir_data)
print(dir_data_path)

for filename in sorted(dir_data_path):
    path=os.path.join(dir_data,filename)
    # print(path)
    if os.path.isdir(path):
        label_id=len(labels_index)
        # print(label_id)
        labels_index[filename]=label_id
        print(labels_index[filename])
        for name in sorted(os.listdir(path)):
            filepath=os.path.join(path,name)
            # print(filepath)
            if sys.version_info<(3,):
                f=open(filepath)
            else:
                f=open(filepath,encoding='gbk',errors='ignore')
            text=f.read()
            # print(text)
            seg_list=jieba.cut(text,cut_all=False)
            new_content=" ".join(seg_list)
            # print(new_content)
            # texts.append(new_content)
            seg_list1=PinyinHelper.convertToPinyinFromSentence(new_content)
            print(seg_list1)
            texts.append(seg_list1)
            f.close()
            labels.append(label_id)
for tes in texts:
    print(">>>>")
    print(tes)
