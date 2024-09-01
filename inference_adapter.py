import os
import pickle

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from modules import common
from PIL import Image


def get_activation(function_type):
    if function_type == 'relu':
        return nn.ReLU()
    elif function_type == 'sigmoid':
        return nn.Sigmoid()


class Model(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(Model, self).__init__()

        self.layer_list = []
        self.input_channel = input_shape
        self.args = args

        assert isinstance(self.args, dict)
        self.activation_type = self.args['activation']
        self.num_layer = self.args['num_layer']
        latent_shape = self.args['latent_shape']
        down_sample = self.args['down_sample']

        assert self.num_layer - 1 == len(latent_shape)

        if self.num_layer == 1:
            self.layer_list.append(nn.Linear(input_shape, output_shape))
        else:
            assert self.num_layer - 1 == len(latent_shape)
            for index in range(self.num_layer - 1):
                output_channel = latent_shape[index]

                self.layer_list.append(nn.Linear(self.input_channel, output_channel))
                self.layer_list.append(get_activation(self.activation_type))

                self.input_channel = output_channel
            self.layer_list.append(nn.Linear(self.input_channel, output_shape))

        self.layer_list = nn.Sequential(*self.layer_list)

        if self.num_layer > 1 and down_sample:
            self.down_sample = nn.Linear(input_shape, output_shape)
        else:
            self.down_sample = None

    def forward(self, x):
        output = self.layer_list(x)
        if self.num_layer > 1 and self.down_sample:
            output += self.down_sample(x)
        return output


def get_image_pil(img_np):
    # img_np = ((img_np + 0.5) * 255).astype(np.uint8)  # [c, h, w]
    img_np = img_np.transpose(1, 2, 0)
    return Image.fromarray(img_np).convert('RGB')


if __name__ == "__main__":
    # ['Emotion6', 'Emotion-6', 'UnBiasedEmotion', 'DeepEmotion']
    dataset_name = 'Emotion6'

    combine_method_version = 'v0'
    query_type = 'SI-SQ'
    model_version = 'v0'  # [v0, v1-1]
    query_version = 'v1'  # [v1-10]
    split_ratio = (0.8, 0.0, 0.2)
    setting_version = 'v0'
    exp_count = 0
    dataset_dir = '${DATASET_DIR}'
    random_seed = 128  # use the random seed in the train code

    # ================

    # image visualization
    # image_dataset_path = f'{dataset_dir}/{dataset_name}/data_pickle/sample'

    model_save_dir = f'./LMM_Emotion_Prediction/checkpoints/{dataset_name}'
    model_save_path = os.path.join(model_save_dir,
                                   f'{dataset_name}_M:{model_version}_Q:{query_version}_Setting:{setting_version}_'
                                   f'Split:{split_ratio[0]}_Combine:{combine_method_version}_Count:{exp_count}_'
                                   f'{query_type}.ckp')

    args = {'norm': None,
            'num_layer': 1, 'activation': 'relu',
            'down_sample': False,
            'latent_shape': []}

    data = common.load_answer_file(
        f'./LMM_Emotion_Prediction/llava_embedding_combined_{combine_method_version}/{dataset_name}/{query_type}',
        dataset_name, model_version, query_version, postfix='pkl')
    label_list = data['label_text']

    train_set_dict, val_set_dict, test_set_dict = common.split_dataset_pkl(
        data['text_data'], data['label_data'], split_ratio, fix_seed=True, random_seed=random_seed)

    input_dim = train_set_dict['data'].shape[1]
    output_dim = train_set_dict['label'].shape[1]

    model = Model(input_dim, output_dim, args)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    test_set_dict['label'] = torch.from_numpy(np.argmax(test_set_dict['label'], axis=1)).type(torch.float32)
    test_set_dict['data'] = torch.from_numpy(test_set_dict['data']).type(torch.float32)

    logits = model(test_set_dict['data'])
    _, predicted = torch.max(logits.data, 1)
    total_item_count = logits.shape[0]
    correct_item_count = (predicted == test_set_dict['label']).sum().item()

    result = correct_item_count / total_item_count
    print(result)
