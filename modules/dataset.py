import os
import functools
import json
import glob
import pickle
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class CustomDataSet(Dataset):
    def __init__(self, dataset_path, split='train',
                 select_label_subset=False, label_subset_list=None, transformer=None,
                 copy_train_data_locally=False, copy_test_data_locally=False):
        self.delimiter = '###'

        assert split in ['train', 'test', 'val', 'all']
        self.dataset_path = dataset_path
        self.split = split

        pkl_file_paths = [os.path.join(dataset_path, 'sample', file_name) for file_name in os.listdir(
            os.path.join(dataset_path, 'sample'))]
        pkl_file_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
        self.total_dataset_number = len(pkl_file_paths)

        label_file = dataset_path + 'data_info'
        labels = pd.read_csv(label_file)
        label_texts = [item.lower() for item in labels['label']]

        data_subset_list = self.get_split_list()
        pkl_file_paths = [pkl_file_paths[index] for index in data_subset_list]
        label_texts = [label_texts[index] for index in data_subset_list]

        if select_label_subset:
            assert label_subset_list is not None
            label_subset_set = set(label_subset_list)
            self.map_pkl_info = []
            self.map_label_info = []
            for pkl_file_path, label_text in zip(pkl_file_paths, label_texts):
                label_text = label_text.split(self.delimiter)
                if set(label_text).issubset(label_subset_set):
                    self.map_pkl_info.append(pkl_file_path)
                    self.map_label_info.append(label_text)
        else:
            self.map_pkl_info = pkl_file_paths
            self.map_label_info = label_texts

        current_text_list_with_duplicate = functools.reduce(lambda a, b: a+b, self.map_label_info)
        current_text_list = list(set(current_text_list_with_duplicate))
        current_text_list.sort()  # make list order constant
        self.text2num = {text: label for label, text in enumerate(current_text_list)}
        self.text2count = {text: current_text_list_with_duplicate.count(text) for text in current_text_list}
        self.num2text = {label: text for text, label in self.text2num.items()}

        self.transformer = transformer

    def get_item(self, idx, return_np=False):
        assert 0 <= idx < self.__len__()
        item = self.__getitem__(idx)
        if return_np:
            item['data'] = np.array(item['data'], dtype=np.float32)
            item['label'] = np.array(item['label'], dtype=np.uint32)
        return item

    def get_split_list(self):
        try:
            split_file = pd.read_csv(os.path.join(self.dataset_path, f'{self.split}_info'))
        except FileNotFoundError:
            split_file = None

        if split_file is None:
            assert self.split == 'all'
            data_subset_list = [idx for idx in range(self.total_dataset_number)]
        else:
            data_subset_list = [int(idx) for idx in split_file['idx']]
            data_subset_list.sort()
        return data_subset_list

    def get_labels(self):
        return [self.text2num[text] for text in self.map_label_info]

    def __len__(self):
        return len(self.map_label_info)

    def __getitem__(self, idx):
        pkl_file_path = self.map_pkl_info[idx]
        label_num = [self.text2num[item] for item in self.map_label_info[idx]]
        with open(pkl_file_path, 'rb') as file:
            pkl_data = pickle.load(file)
        label_num = torch.tensor(label_num)
        pkl_data = torch.tensor(pkl_data)
        pkl_data = pkl_data / 255 - 0.5  # [c, h, w]
        if self.transformer is not None:
            pkl_data = self.transformer(pkl_data)
        return {'data': pkl_data, 'label': label_num, 'info': pkl_file_path}
