#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

# def mnist_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users


# def mnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
#     idxs = idxs_labels[0,:]

#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users


# def cifar_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users


# def sensing_data_dict(dataset_1, dataset_2, dataset_3):
#     dict_users = {}
#     all_idxs_1 = [i for i in range(len(dataset_1))]
#     all_idxs_2 = [i for i in range(len(dataset_2))]
#     all_idxs_3 = [i for i in range(len(dataset_3))]
#     dict_users = {0: set(all_idxs_1), 1: set(all_idxs_2), 2: set(all_idxs_3)}
#     return dict_users

def sensing_data_dict(dataset):
    dict_users = {}
    for idx in range(len(dataset)):
        all_idxs = [i for i in range(len(dataset[idx]))]
        dict_users[idx]=all_idxs
    return dict_users

# def sensing_data_dict_12(dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6, dataset_7, dataset_8, dataset_9, dataset_10, dataset_11, dataset_12):
#     dict_users = {}
#     all_idxs_1 = [i for i in range(len(dataset_1))]
#     all_idxs_2 = [i for i in range(len(dataset_2))]
#     all_idxs_3 = [i for i in range(len(dataset_3))]
#     all_idxs_4 = [i for i in range(len(dataset_4))]
#     all_idxs_5 = [i for i in range(len(dataset_5))]
#     all_idxs_6 = [i for i in range(len(dataset_6))]
#     all_idxs_7 = [i for i in range(len(dataset_7))]
#     all_idxs_8 = [i for i in range(len(dataset_8))]
#     all_idxs_9 = [i for i in range(len(dataset_9))]
#     all_idxs_10 = [i for i in range(len(dataset_10))]
#     all_idxs_11 = [i for i in range(len(dataset_11))]
#     all_idxs_12 = [i for i in range(len(dataset_12))]
#     dict_users = {0: set(all_idxs_1), 1: set(all_idxs_2), 2: set(all_idxs_3), 3: set(all_idxs_4), 4: set(all_idxs_5), 5: set(all_idxs_6), 6: set(all_idxs_7), 7: set(all_idxs_8), 8: set(all_idxs_9), 9: set(all_idxs_10), 10: set(all_idxs_11), 11: set(all_idxs_12)}
#     return dict_users


# if __name__ == '__main__':
#     dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
#                                    transform=transforms.Compose([
#                                        transforms.ToTensor(),
#                                        transforms.Normalize((0.1307,), (0.3081,))
#                                    ]))
#     num = 100
#     d = mnist_noniid(dataset_train, num)
