import argparse
import os
import json
import pickle
import numpy as np
import torch

from modules import common
from models import LinearModel, NeuralNetwork


def select_cls_model(input_type):
    if input_type == 'linear':
        return LinearModel
    elif input_type == 'nn':
        return NeuralNetwork


def write_log(path, content):
    with open(path, 'a') as f:
        f.write(content)
        f.write('\n')


def get_setting(input_setting_version):
    if input_setting_version == 'v0':
        return {'norm': None,
                'num_layer': 1, 'activation': 'relu', 'scheduler': None,
                'down_sample': False,
                'latent_shape': []}
    elif input_setting_version == 'v1':
        return {'norm': 2,
                'num_layer': 1, 'activation': 'relu', 'scheduler': None,
                'down_sample': False,
                'latent_shape': []}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=[
        'Emotion6', 'Emotion-6', 'UnBiasedEmotion', 'DeepEmotion'], default='Emotion6')
    parser.add_argument("--Q", type=str)
    parser.add_argument("--combine_method", type=str)  # ['v0-2']
    parser.add_argument("--setting", type=str)
    args = parser.parse_args()

    postfix = 'fix-test-split'
    model_version = 'v0'  # [v0, v1-1]
    query_version = args.Q  # [v1-10]
    assert query_version is not None
    if query_version == 'ALL':
        query_version_list = [f'v{version}' for version in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    else:
        query_version_list = [query_version]
    combine_method_version = args.combine_method
    setting_version = args.setting
    dataset_name = args.dataset
    method_type = 'nn'  # ['nn']
    query_type = 'SI-SQ'
    repeat_num = 10
    if dataset_name in ['DeepEmotion']:
        split_ratio_list = [(0.8, 0.05, 0.15)]
    else:
        split_ratio_list = [(0.8, 0.0, 0.2)]
    random_seed = 128

    # ===================

    if setting_version == 'ALL':
        setting_version_list = [f'v{version}' for version in [0, 1, 2]]
    else:
        setting_version_list = [setting_version]

    for setting_version in setting_version_list:

        if method_type in ['nn']:
            args = get_setting(setting_version)
        else:
            args = None

        log_path = f'./LMM_Emotion_Prediction/logs/log_{dataset_name}_M:{model_version}_Q:ALL_' \
                   f'{query_type}_Combine:{combine_method_version}_Setting:{setting_version}_{postfix}.txt'

        model_save_dir = f'./LMM_Emotion_Prediction/checkpoints/{dataset_name}'
        os.makedirs(model_save_dir, exist_ok=True)

        write_log(log_path, dataset_name)
        write_log(log_path, f'Setting version: {setting_version}')
        write_log(log_path, json.dumps(args, indent=4))
        write_log(log_path, f'{query_version_list}')
        write_log(log_path, f'Combine version: {combine_method_version}')
        write_log(log_path, '===============')

        for split_ratio in split_ratio_list:
            results_dict = {
                'top1_metric_mean': [],
                'top1_metric_std': [],
                'max_metric': [],
                'min_metric': [],
            }
            for query_version in query_version_list:
                data = common.load_answer_file(f'./LMM_Emotion_Prediction/llava_embedding_combined_{combine_method_version}/{dataset_name}/{query_type}',
                                               dataset_name, model_version, query_version, postfix='pkl')
                label_list = data['label_text']

                max_accuracy = 0
                min_accuracy = 100
                result_list_top1 = []
                args['logger'] = None

                for count in range(repeat_num):
                    args['save_path'] = os.path.join(model_save_dir,
                                                     f'{dataset_name}_M:{model_version}_Q:{query_version}_Setting:{setting_version}_'
                                                     f'Split:{split_ratio[0]}_Combine:{combine_method_version}_Count:{count}_'
                                                     f'{query_type}.ckp')

                    # for reproducibility, use: fix_seed=True, random_seed=SEED_NUM
                    train_set_dict, val_set_dict, test_set_dict = common.split_dataset_pkl(
                        data['text_data'], data['label_data'], split_ratio, fix_seed=True, random_seed=random_seed)

                    solver_worker = select_cls_model(method_type)(train_set_dict['data'], test_set_dict['data'],
                                                                  train_set_dict['label'], test_set_dict['label'],
                                                                  topk=(1,))

                    output_dict = solver_worker.predict(args=args)
                    accuracy_list = output_dict['result']
                    result_list_top1.append(accuracy_list[0])

                    if min_accuracy > result_list_top1[-1]:
                        min_accuracy = result_list_top1[-1]

                    if max_accuracy < result_list_top1[-1]:
                        max_accuracy = result_list_top1[-1]

                results_dict['top1_metric_mean'].append(np.mean(result_list_top1))
                results_dict['top1_metric_std'].append(np.std(result_list_top1))
                results_dict['max_metric'].append(max_accuracy)
                results_dict['min_metric'].append(min_accuracy)

                # print(f'M:{model_version}_Q:{query_version}')
                # print(f'top1: {np.mean(result_list_top1):.2f}, std: {np.std(result_list_top1):.2f}')
                # print(f'max_top1_acc: {max_accuracy:.2f}, min_top1_acc: {min_accuracy:.2f}')
                # print('=============')

            write_log(log_path, '===============')
            write_log(log_path, f'{split_ratio}')
            for key in results_dict:
                write_log(log_path, '===')
                write_log(log_path, key)
                for item in results_dict[key]:
                    write_log(log_path, f'{item:.7f}')


