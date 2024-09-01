import os
import pickle
import math
import json
import numpy as np


def load_answer_file(parent_path, dataset_name, model_version, query_version, query_type='SI-SQ', postfix='json'):
    assert postfix in ['json', 'pkl']

    answer_file = f'{dataset_name}_llava_M:{model_version}_Q:{query_version}_{query_type}.{postfix}'
    data_file = os.path.join(parent_path, answer_file)

    if postfix == 'json':
        with open(data_file) as f:
            output_data = json.load(f)
    else:
        with open(data_file, 'rb') as f:
            output_data = pickle.load(f)
    return output_data


def split_dataset(input_data, text2num_dict, split_size, fix_seed=True, random_seed=42):
    if fix_seed:
        assert isinstance(random_seed, int)
        np.random.seed(random_seed)

    assert len(split_size) in [2, 3] and math.isclose(sum(split_size), 1.0, abs_tol=1e-5)
    if len(split_size) == 2:
        train_size, eval_size, test_size = split_size[0], 0, split_size[1]
    else:
        train_size, eval_size, test_size = split_size[0], split_size[1], split_size[2]

    data_size = len(input_data)
    split_index_list = [i for i in range(data_size)]
    np.random.shuffle(split_index_list)

    train_size, eval_size, test_size = \
        int(data_size * train_size), int(data_size * eval_size), int(data_size * test_size)

    train_set = [(
        text2num_dict[input_data[str(index)]['label'][0]],
        input_data[str(index)]['answer_detail']
    ) for index in split_index_list[:train_size]]

    eval_set = [(
        text2num_dict[input_data[str(index)]['label'][0]],
        input_data[str(index)]['answer_detail']
    ) for index in split_index_list[train_size:train_size + eval_size]]

    test_set = [(
        text2num_dict[input_data[str(index)]['label'][0]],
        input_data[str(index)]['answer_detail']
    ) for index in split_index_list[train_size + eval_size:]]

    if eval_size == 0:
        return train_set, test_set
    else:
        return train_set, eval_set, test_set


def split_dataset_pkl(input_data, input_label, split_size, fix_seed=False, random_seed=42):
    if fix_seed:
        assert isinstance(random_seed, int)
        np.random.seed(random_seed)

    assert len(split_size) in [2, 3] and sum(split_size) == 1.0
    if len(split_size) == 2:
        train_size, eval_size, test_size = split_size[0], 0, split_size[1]
    else:
        train_size, eval_size, test_size = split_size[0], split_size[1], split_size[2]

    data_size = len(input_data)
    split_index_list = [i for i in range(data_size)]
    np.random.shuffle(split_index_list)

    train_size, eval_size, test_size = \
        int(data_size * train_size), int(data_size * eval_size), int(data_size * test_size)

    def get_subset_list(input_list, index_list):
        return np.array([input_list[index] for index in index_list])

    train_set_dict = {
        'data': get_subset_list(
            input_data, split_index_list[: train_size]),
        'label': get_subset_list(
            input_label, split_index_list[: train_size]),
        'info': np.array(split_index_list[: train_size])
    }

    eval_set_dict = {
        'data': get_subset_list(
            input_data, split_index_list[train_size: train_size + eval_size]),
        'label': get_subset_list(
            input_label, split_index_list[train_size: train_size + eval_size]),
        'info': np.array(split_index_list[train_size: train_size + eval_size])
    }

    test_set_dict = {
        'data': get_subset_list(
            input_data, split_index_list[train_size + eval_size:]),
        'label': get_subset_list(
            input_label, split_index_list[train_size + eval_size:]),
        'info': np.array(split_index_list[train_size + eval_size:])
    }

    if len(split_size) == 2:
        return train_set_dict, test_set_dict
    else:
        return train_set_dict, eval_set_dict, test_set_dict
