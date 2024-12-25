import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


# def gen_txt(dir, train):
#     folder_level_list = os.listdir(dir)
#     folder_level_list.sort()
#     for folder_level in folder_level_list:
#         for folder_label in os.listdir(dir + folder_level):
#             for file in os.listdir(os.path.join(dir, folder_level, folder_label)):
#                 name = os.path.join(dir, folder_level, folder_label, file) + ' ' + str(int(folder_label)-1) + '\n'
#                 train.write(name)
#     train.close()


# def gen_txt_origin(dir, train, test):
#     folder_label_list = os.listdir(dir)
#     folder_label_list.sort()
#     for folder_label in folder_label_list[0:5]:
#         file_list = os.listdir(os.path.join(dir, folder_label))
#         file_list.sort()
#         label_idx = int(folder_label[-1])-1
#         for idx in range(0, 130):
#             name = os.path.join(dir, folder_label, file_list[idx * 2]) + ' ' + str(label_idx) + '\n'
#             train.write(name)
#         for idx in range(130, 230):
#             name = os.path.join(dir, folder_label, file_list[idx * 2]) + ' ' + str(label_idx) + '\n'
#             test.write(name)
#     train.close()
#     test.close()

def gen_txt_origin(dir, num_class, num_users):
    folder_label_list = os.listdir(dir)  #列出对当前文件夹下的所有子文件夹名称
    if '.DS_Store' in folder_label_list: #windows系统可能会有不同的隐藏文件夹
        folder_label_list.remove('.DS_Store')  #删除Mac系统中的隐藏文件.DS_Store
    folder_label_list.sort()
    # print(folder_label_list)

    for user_idx in range(num_users):  #生成num_users个txt文件
        gener_txt= open(dir+'train_user_'+str(user_idx)+'.txt', 'w')
        gener_txt.close()
    test = open(dir+'test.txt', 'w')

    for folder_label in folder_label_list[0:num_class]: #引用的就是floder_label_list里面的字符串元素
        # print(folder_label)
        file_list = os.listdir(os.path.join(dir, folder_label)) #列出当前文件夹下的所有照片名
        file_list.sort() #这里的排序不是1-300 不是自然顺序而是1 10 100 101 102...
        label_idx = int(folder_label[-1])-1  #由于原本folder_label就是1-5命名，所以转为整数-1即可变为0-4的label
        prop=0.8 #trian data 所占比例
        num_trian=int(len(file_list)*prop)
        for idx in range(0, num_trian):
            for user_idx in range(num_users):
                if idx in range(int(user_idx*num_trian/num_users), int((user_idx+1)*num_trian/num_users)):
                    name = os.path.join(dir, folder_label, file_list[idx]) + ',' + str(label_idx) + '\n'
                    train = open(dir+'train_user_'+str(user_idx)+'.txt', 'a+')
                    train.write(name)
        for idx in range(num_trian+1, len(file_list)):
            name = os.path.join(dir, folder_label, file_list[idx]) + ',' + str(label_idx) + '\n'
            test.write(name)
    train.close()
    test.close()

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, txt, transform, loader=default_loader):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            words = line.split(',')         #将txt文件逐行进行拆分 split默认拆分符号为空格 
            imgs.append((words[0], int(words[1]))) #拆分后存在words中 第0个元素为路径+image name 第1个为label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        fn, label = self.imgs[item]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


# class CenDataset(Dataset):
#     def __init__(self, data, labels):
#         super(CenDataset, self).__init__()
#         self.data = data
#         self.labels = labels

#     def __getitem__(self, item):
#         img = self.data[item]
#         label = self.labels[item]
#         return img, label

#     def __len__(self):
#         return len(self.data)


# if __name__ == "__main__":
#     gen_txt_flag = True
#     if gen_txt_flag:
#         dir='/Users/thea/Library/CloudStorage/OneDrive-TheChineseUniversityofHongKong/Yao_20220803/Simulation/UAV_generate_figures_SSIM/data/power_30dbm_1w_42_42/data/'
#         train_1 = open(dir+'train_1_m7.txt', 'w')
#         train_2 = open(dir+'test_m7.txt', 'w')
#         num_class = 5
#         gen_txt_origin(dir, train_1, train_2, num_class)




