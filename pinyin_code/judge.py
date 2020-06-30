from __future__ import print_function
"""
pinyin converter
"""

from pathlib import Path
import sys
import os
from imp import reload
reload(sys)
sys.getdefaultencoding()

"""
判断文件目录是否存在
"""

path='E:/DeepLearning/bully_code/diyu/diyu'
dir=Path(path)
print(dir)
try:
    dir_path=dir.resolve()
except FileNotFoundError:
    print("source not exist!")
else:
    print("Source exist!")

"""
判断文件中txt文件是否存在
"""
try:
    dir_file=open(path+"/"+"正例"+"/"+"1.txt")
    dir_file.close()
except FileNotFoundError:
    print("No such dir_file or directory: '%s'" % path)
except IsADirectoryError:
    print("Is a directory: '%s'" % path)
except PermissionError:
    print("permission denied:'%s'" % path) #文件物操作权限
else:
    print("dir_file is exist:'%s''"%dir_file)



allFileNum=0
def printpath(level,path1):
    global allFileNum
    """
    打印一个目录下的所有文件
    """
    dirlist=[]
    filelist=[]
    files=os.listdir(path1)
    dirlist.append(str(level))
    for f in files:
        if(os.path.isdir(path+'/'+f)):
            if(f[0]=='.'):
                pass
            else:
                dirlist.append(f)
        if (os.path.isfile(path+'/'+f)):
            filelist.append(f)
    i_dl=0
    for dl in dirlist:
        if(i_dl==0):
            i_dl=i_dl+1
        else:
            print((int(dirlist[0])+1),path1+'/'+dl)
    for fl in filelist:
        # print((int(dirlist[0])),fl)
        allFileNum=allFileNum+1

if __name__=='__main__':
    path1='E:/DeepLearning/bully_code/diyu/diyu'
    printpath(0,path1)
    print('文件总数=：',allFileNum)







