import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
import pickle


def get_columns_name(string):
    if string in ['valence', 'arousal']:
        return f'[{string}]'
    else:
        return f'[prob. {string}]'


def append_item(input_list, input_item):
    if isinstance(input_item, str):
        input_list.append(input_item)
    else:
        input_list += list(input_item)
    return input_list


def process_dataset(dataset_dir, name):
    data_source = f'{dataset_dir}/{name}/data_raw/'
    data_destination = f'{dataset_dir}/{name}/data_pickle/'

    os.makedirs(data_destination, exist_ok=True)
    os.makedirs(data_destination + 'sample/', exist_ok=True)

    if name == 'Emotion6':
        emotion_list = os.listdir(data_source + 'images/')
        label_file_raw = pd.read_csv(data_source + 'ground_truth.txt', delimiter='\t')
        column_list = ['label', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral', 'valence',
                       'arousal']
        label_file = pd.DataFrame({}, columns=column_list)

        image_cnt = 0
        for emotion in emotion_list:
            image_list = os.listdir(data_source + 'images/' + emotion)
            image_list.sort()
            for image_name in image_list:
                print(image_cnt)
                # image_npy = cv2.imread(data_source + image_name)
                # image_npy = cv2.cvtColor(image_npy, cv2.COLOR_BGR2RGB)
                # image_resized = cv2.resize(image_npy, (224, 224), interpolation=cv2.INTER_AREA)
                # image_resized = np.transpose(image_resized, (2, 0, 1))

                label_info = label_file_raw[label_file_raw['[image_filename]'] == f'{emotion}/{image_name}']
                label_row = [label_info[get_columns_name(item)].item() for item in column_list[1:]]
                label_row = [emotion] + label_row

                label_file.loc[len(label_file.index)] = label_row

                image_npy = Image.open(data_source + 'images/' + emotion + '/' + image_name)
                image_resized = image_npy.resize((224, 224))
                image_resized = np.array(image_resized)
                image_resized = np.transpose(image_resized, (2, 0, 1))  # [c, h, w]

                with open(data_destination + 'sample/{:05d}.pkl'.format(image_cnt), 'wb') as file:
                    pickle.dump(image_resized, file)

                image_cnt += 1

        label_file.to_csv(data_destination + 'data_info', index=False)

    elif name == 'UnBiasedEmotion':
        emotion_list = os.listdir(data_source + 'images/')
        column_list = ['label', 'object']
        label_file = pd.DataFrame({}, columns=column_list)

        image_cnt = 0
        for emotion in emotion_list:
            object_list = os.listdir(os.path.join(data_source, 'images/', emotion))
            for object_item in object_list:
                image_list = os.listdir(os.path.join(data_source, 'images/', emotion, object_item))
                for image_item in image_list:
                    print(image_cnt)

                    label_row = [emotion, object_item]
                    label_file.loc[len(label_file.index)] = label_row

                    image_npy = Image.open(os.path.join(data_source, 'images/', emotion, object_item, image_item))
                    image_npy = image_npy.convert('RGB')
                    image_resized = image_npy.resize((224, 224))
                    image_resized = np.array(image_resized)
                    image_resized = np.transpose(image_resized, (2, 0, 1))  # [c, h, w]

                    with open(data_destination + 'sample/{:05d}.pkl'.format(image_cnt), 'wb') as file:
                        pickle.dump(image_resized, file)

                    image_cnt += 1

        label_file.to_csv(data_destination + 'data_info', index=False)

    elif name == 'DeepEmotion':
        emotion_list = os.listdir(data_source + 'emotion_dataset/')
        column_list = ['label', 'image_name']
        label_file = pd.DataFrame({}, columns=column_list)

        image_cnt = 0
        for emotion in emotion_list:
            image_list = os.listdir(os.path.join(data_source, 'emotion_dataset/', emotion))
            for image_item in image_list:
                print(image_cnt)

                label_row = [emotion, image_item]
                label_file.loc[len(label_file.index)] = label_row

                image_npy = Image.open(os.path.join(data_source, 'emotion_dataset/', emotion, image_item))
                image_npy = image_npy.convert('RGB')
                image_resized = image_npy.resize((224, 224))
                image_resized = np.array(image_resized)
                image_resized = np.transpose(image_resized, (2, 0, 1))  # [c, h, w]

                with open(data_destination + 'sample/{:05d}.pkl'.format(image_cnt), 'wb') as file:
                    pickle.dump(image_resized, file)

                image_cnt += 1

        label_file.to_csv(data_destination + 'data_info', index=False)

    elif name == 'Emotion-6':
        emotion_v1_list = os.listdir(os.path.join(data_source, 'images/'))
        column_list = ['label', 'label_v2', 'label_v3', 'image_name']
        label_file = pd.DataFrame({}, columns=column_list)

        image_cnt = 0
        for emotion_v1 in emotion_v1_list:
            emotion_v2_list = os.listdir(os.path.join(data_source, 'images/', emotion_v1))
            for emotion_v2 in emotion_v2_list:
                emotion_v3_list = os.listdir(os.path.join(data_source, 'images/', emotion_v1, emotion_v2))
                for emotion_v3 in emotion_v3_list:
                    image_list = os.listdir(os.path.join(data_source, 'images/', emotion_v1, emotion_v2, emotion_v3))

                    for image_item in image_list:
                        print(image_cnt)

                        label_row = [emotion_v1, emotion_v2, emotion_v3, image_item]
                        label_file.loc[len(label_file.index)] = label_row

                        image_npy = Image.open(os.path.join(
                            data_source, 'images/', emotion_v1, emotion_v2, emotion_v3, image_item))

                        image_npy = image_npy.convert('RGB')
                        image_resized = image_npy.resize((224, 224))
                        image_resized = np.array(image_resized)
                        image_resized = np.transpose(image_resized, (2, 0, 1))  # [c, h, w]

                        with open(data_destination + 'sample/{:05d}.pkl'.format(image_cnt), 'wb') as file:
                            pickle.dump(image_resized, file)

                        image_cnt += 1

        label_file.to_csv(data_destination + 'data_info', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        choices=['Emotion6', 'UnBiasedEmotion', 'DeepEmotion', 'Emotion-6'])
    parser.add_argument("--dataset_dir", type=str)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    dataset_name = args.dataset

    process_dataset(dataset_dir, dataset_name)
