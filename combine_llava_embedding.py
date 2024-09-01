import argparse
import os
import pickle
from collections import defaultdict

import numpy as np


def dict_recursive():
    return defaultdict(dict_recursive)


def get_combine_method(method_version, input_embedding):
    if method_version == 'v0':  # N = 1
        return np.mean(input_embedding, axis=1)[0]
    elif method_version == 'v1':  # N = 5
        slice_num = 5
        sequence_length = input_embedding.shape[1]
        slice_length = sequence_length // slice_num + 1
        output_embedding = np.stack([np.mean(input_embedding[:,
                                             slice_length * index:slice_length * (index + 1), :], axis=1)
                                     for index in range(slice_num)], axis=1)
        return output_embedding.reshape(1, -1)[0]
    else:  # 'v2' N = 10
        slice_num = 10
        sequence_length = input_embedding.shape[1]
        slice_length = sequence_length // slice_num + 1
        output_embedding = np.stack([np.mean(input_embedding[:,
                                             slice_length * index:slice_length * (index + 1), :], axis=1)
                                     for index in range(slice_num)], axis=1)
        return output_embedding.reshape(1, -1)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['Emotion6', 'Emotion-6',
                                                        'UnBiasedEmotion', 'DeepEmotion'])
    parser.add_argument("--combine_method", type=str)
    parser.add_argument("--M", type=str, default='v0')
    parser.add_argument("--Q", type=str, default='v1')
    args = parser.parse_args()

    # ================

    dataset_name = args.dataset
    combine_method_version = args.combine_method  # [v0, v1, v2]
    query_type = 'SI-SQ'
    model_version = args.M
    query_version = args.Q

    if query_version == 'ALL':
        query_version_list = [f'v{version}' for version in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    else:
        query_version_list = [query_version]

    for query_version in query_version_list:

        source_data_dir = f'./LMM_Emotion_Prediction/llava_embedding/{dataset_name}/{query_type}/' \
                          f'{dataset_name}_llava_M:{model_version}_Q:{query_version}_{query_type}'
        data_save_path = f'./LMM_Emotion_Prediction/llava_embedding_combined_{combine_method_version}/{dataset_name}/{query_type}'
        data_save_name = f'{dataset_name}_llava_M:{model_version}_Q:{query_version}_{query_type}.pkl'
        os.makedirs(data_save_path, exist_ok=True)

        pickle_file_list = os.listdir(source_data_dir)
        pickle_file_list = [file_name for file_name in pickle_file_list
                            if file_name not in ['params.pkl']]
        pickle_file_list.sort()

        with open(os.path.join(source_data_dir, 'params.pkl'), 'rb') as f:
            params_dict = pickle.load(f)
        label_list = params_dict['params']['LABEL_LIST']

        label_data_mat = []
        answer_data_mat = []

        for pickle_file in pickle_file_list:
            with open(os.path.join(source_data_dir, pickle_file), 'rb') as f:
                data_dict = pickle.load(f)
            label_text_list = data_dict['label']

            label_data = [1 if label in label_text_list else 0 for label in label_list]
            label_data_mat.append(label_data)

            # answer_data = np.mean(data_dict['embedding'], axis=1)[0]
            answer_data = get_combine_method(combine_method_version, data_dict['embedding'])
            answer_data_mat.append(answer_data)

        output_dict = {
            'label_text': label_list,
            'text_data': np.array(answer_data_mat),
            'label_data': np.array(label_data_mat)
        }
        with open(os.path.join(data_save_path, data_save_name), 'wb') as f:
            pickle.dump(output_dict, f)
