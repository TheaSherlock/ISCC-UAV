#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random
import logging
import time
import os
from utils.sampling import sensing_data_dict
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import *
from models.Fed import FedAvg
from models.test import test_img
from data_set import MyDataset,gen_txt_origin
from record_csv import conv_csv

##创建logger 并可在terminal输出
args = args_parser()
log_name = 'ISCC-UAV-{}-{}-5motions-user-{}-round-BS.log'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), args.num_users, args.epochs)
logger = logging.getLogger('CNN') # 创建一个logger
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('./log/{}'.format(log_name)) # 创建一个handler，用于写入日志文件
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler() # 再创建一个handler，用于输出到控制台terminal
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # 定义handler的输出格式
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh) # 给logger添加handler
logger.addHandler(ch)
logger.info('ISCC-UAV-Fed-ResNet10')

if __name__ == '__main__':
    #引入自定义的变量在文件options.py中
    args = args_parser()
    #加载到gpu上如果gpu存在 否则cpu
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    #输入图片保存的目录 并产生对应user number的train_user_1/2/3/4/...txt文件和test.txt文件
    # dir=r'D:\OneDrive - The Chinese University of Hong Kong\Yao_20220803\Simulation\UAV_generate_figures_SSIM\data\power_30dbm_1w_42_42\data'
    # dir='/Users/thea/Library/CloudStorage/OneDrive-TheChineseUniversityofHongKong/Yao_20220803/Simulation/UAV_generate_figures_SSIM/data/power_30dbm_1w_42_42/data/'
    dir = '/home/sensing/tangyao/FLLS_BSPO/data/45' #power_30dbm_1w_42_42/45/'
    num_class = 5 #分类个数
    gen_txt_origin(dir, num_class, args.num_users)
    #加载数据
    dataset_train = []
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.7723, 0.8303, 0.9284), (0.3916, 0.3057, 0.1893)),
        ])
    for user_idx in range(args.num_users):
        dataset_train_per_user= MyDataset(txt=dir + 'train_user_'+str(user_idx)+'.txt', transform=data_transform)
        dataset_train.append(dataset_train_per_user)
    dataset_test = MyDataset(txt=dir + 'test.txt', transform=data_transform)
    dict_users = sensing_data_dict(dataset_train)
    #存储初始模型参数 保证benchmark scheme的初始化模型都一样
    net_glob = ResNet.ResNet10().to(args.device)
    model_root = './save/models_10_m7.pth'
    if os.path.exists(model_root) is False:
        torch.save(net_glob.state_dict(), model_root)
    net_glob.load_state_dict(torch.load(model_root))
    #计算所选模型的参数总数目
    net_total_params = sum(p.numel() for p in net_glob.parameters()) 
    print('| net_total_params: {}'.format(net_total_params))
    print(net_glob)
    #保证BN层用每一批数据的均值和方差
    net_glob.train() 
    #保存参数到global model
    w_glob = net_glob.state_dict() #state_dict()字典对象,将每一层与它的对应参数建立映射关系
    logger.info(args)
    
    if args.all_clients: #如果当前场景中每个用户都参加则 初始化w_local
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)] #把global model传给每个client
    
    batch_size_users = [23, 22, 19, 25, 20, 26, 20, 27] #UAVs的batch size 每轮都一样 但每个uav不一样
    q_s = [0.8953, 0.8953, 0.8953, 0.8953, 0.8953, 0.8953, 0.8953, 0.8953]        #successful sensing probability
    loss_train = []
    acc_test = []
    for round in range(1, args.epochs + 1): #表示从1-300
        #对于每轮训练, 我们先掷骰子，选出参加的用户
        select_users=[]
        m=0  #参加的用户数 不能少于一
        for user_idx in range(args.num_users):
            q_per_user=random.random()
            if q_per_user <= q_s[user_idx]: 
                m+=1
                select_users.append(user_idx)
        if m<1:   #如果参加的用户数少于一 则随机选一个用户即可
            select_users = np.random.choice(range(args.num_users), 1, replace=False)
        print('Selected users: {}'.format(select_users))
        local_steps = args.local_ep
        loss_locals = []
        w_locals = []
        for idx in select_users:
            local = LocalUpdate(args=args, batch_size=batch_size_users[idx], dataset=dataset_train[idx], idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), local_steps=local_steps)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w)) #只需要记录参加训练的用户的w即可 只用到他们的和
            loss_locals.append(copy.deepcopy(loss))
        # 更新global model
        w_glob = FedAvg(w_locals)
        # 把w_glob拷贝到net_glob中
        net_glob.load_state_dict(w_glob)
        # 输出loss值
        loss_avg = sum(loss_locals) / len(loss_locals)
        logger.info('Epoch: {}'.format(round))
        logger.info('Train loss: {:.4f}'.format(loss_avg))
        # 记录每轮training loss值
        loss_train.append(loss_avg) 
        # net_glob.eval()
        acc_test_per, loss_test_per = test_img(net_glob, dataset_test, args)
        # 记录每轮test accuracy
        logger.info("test acc: {:.2f}%".format(acc_test_per))
        acc_test.append(int(acc_test_per))
    
    # 把数据保存为cvs文件
    conv_csv(dir, loss_train, acc_test)
    #plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.yscale('log')
    plt.ylabel('Training loss')
    plt.xlabel('Communication rounds')
    plt.grid()
    plt.savefig('./save/fed_{}_{}_{}_{}_{}_{}_{}.png'.format(args.epochs, 'device', args.num_users, 'training_loss', args.lr, q_s[0], 'log'))
    #plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('Training loss')
    plt.xlabel('Communication rounds')
    plt.grid()
    plt.savefig('./save/fed_{}_{}_{}_{}_{}_{}.png'.format(args.epochs, 'device', args.num_users, 'training_loss', args.lr, q_s[0]))
    # plot accuracy curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('Testing accuracy')
    plt.xlabel('Communication rounds')
    plt.grid()
    plt.savefig('./save/fed_{}_{}_{}_{}_{}_{}.png'.format(args.epochs, 'device', args.num_users, 'testing_accuracy', args.lr, q_s[0]))
    
    # testing
    net_glob.eval()
    acc_test_final, loss_test_final = test_img(net_glob, dataset_test, args)
    logger.info("final test acc: {:.2f}%".format(acc_test_final))

