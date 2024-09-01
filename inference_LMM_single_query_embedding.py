from collections import defaultdict
from modules.dataset import CustomDataSet

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llava.conversation import conv_templates
from llava.conversation import SeparatorStyle
from llava.utils import disable_torch_init
from llava.model import LlavaMPTForCausalLM, LlavaLlamaForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria

import os
import numpy as np
from PIL import Image
import pickle


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def extract_hidden_states(
    input_model_dict,
    hidden_layer_idx=-1,
    to_numpy=True
):
    beam_indices = input_model_dict['beam_indices'].clone()
    beam_indices_mask = beam_indices < 0
    max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
    beam_indices = beam_indices[:, :max_beam_length]
    beam_indices_mask = beam_indices_mask[:, :max_beam_length]
    beam_indices[beam_indices_mask] = 0

    seqlen = beam_indices.shape[1]

    # creating the output hidden_states representation in format:
    # [bsz * beam_width ; seqlen ; featdim]
    # start with index 1
    output_hidden_states = torch.stack([
        input_model_dict['hidden_states'][i][hidden_layer_idx][:, 0, :].index_select(
            dim=0, index=beam_indices[:, i]  # reordering using the beam_indices
        )
        for i in range(1, seqlen)
    ]).transpose(0, 1)

    # add hidden_state for index 0
    init_hidden_states = input_model_dict['hidden_states'][0][hidden_layer_idx][:, :, :].index_select(
        dim=0, index=beam_indices[:, 0]  # reordering using the beam_indices
    )
    output_hidden_states = torch.cat([init_hidden_states, output_hidden_states], dim=1)
    expand_length = init_hidden_states.shape[1]
    beam_indices_mask = torch.cat((beam_indices_mask[:, 0:1].expand(beam_indices.shape[0], expand_length - 1),
                                   beam_indices_mask), dim=1)

    # setting to 0 the hidden_states were it doesn't make sense to have an output
    output_hidden_states[beam_indices_mask] = 0

    if to_numpy:
        output_hidden_states = output_hidden_states.cpu().numpy()

    return output_hidden_states


def load_model(model_path, num_gpus, mm_projector_path=None, vision_tower=None):
    if num_gpus == 1:
        kwargs = {}
    else:
        kwargs = {
            "device_map": "auto",
            "max_memory": {i: "13GiB" for i in range(num_gpus)},
        }

    disable_torch_init()
    model_name = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "mpt" in model_name.lower():
        model = LlavaMPTForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                    use_cache=True, **kwargs)
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                      use_cache=True, **kwargs)

    if num_gpus == 1:
       model.cuda()

    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]
    if vision_tower.device.type == 'meta':
        vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        model.get_model().vision_tower[0] = vision_tower
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    return tokenizer, model, image_processor, image_token_len, mm_use_im_start_end


class ModelWorker:
    def __init__(self, model_path, num_gpus):
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_name = os.path.expanduser(model_path)
        self.tokenizer, self.model, self.image_processor, self.image_token_len, self.mm_use_im_start_end = load_model(
            model_path, num_gpus)
        self.conv = None

        self.init_conv()

    def init_conv(self):
        if "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt_multimodal"
        else:
            conv_mode = "multimodal"
        self.conv = conv_templates[conv_mode].copy()

    def insert_prompt(self, input_query, image=None):
        if image is not None:
            input_query = input_query + '\n' + DEFAULT_IMAGE_TOKEN
            self.conv.append_message(self.conv.roles[0], (input_query, image, "Crop"))
        else:
            self.conv.append_message(self.conv.roles[0], input_query)
        self.conv.append_message(self.conv.roles[1], None)
        return self.conv.get_prompt()

    def append_answer(self, reply):
        self.conv.messages[-1][-1] = reply
        return self.conv.get_prompt()

    @staticmethod
    def image_preprocess(images_in):
        images_out = []
        for image in images_in:
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            max_len, min_len = 800, 400
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
            longest_edge = int(shortest_edge * aspect_ratio)
            W, H = image.size
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))
            images_out.append(image)
        return images_out

    def eval_model(self, params, sample_args):
        prompt = params['prompt']
        images = params.get("images", None)
        if not images:  # for images = []
            images = None

        ori_prompt = prompt

        image_count = len(images) if images is not None else 0
        assert image_count == prompt.count(
            DEFAULT_IMAGE_TOKEN), "Number of images does not match number of <image> tokens in prompt"

        replace_token = DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len
        if self.mm_use_im_start_end:
            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        if images is not None:
            image_tensor = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values']
            image_tensor = image_tensor.half().cuda()
        else:
            image_tensor = None

        inputs = self.tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            model_output_dict = self.model.generate(
                input_ids,
                images=image_tensor,
                return_dict_in_generate=True, output_hidden_states=True, output_scores=True,
                stopping_criteria=[stopping_criteria], **sample_args)

        last_hidden_states = extract_hidden_states(model_output_dict)

        output_ids = model_output_dict[0]
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs, last_hidden_states


class QueryClass:
    def __init__(self, ):
        self.query_dict = {
            'v1': f'Describe this image from an emotional perspective.',
            'v2': f'Which emotion will people feel the most when they see this image?',
            'v3': f'Which emotion will people feel the most when they see this image? '
                  f'First, provide the step by step explanation of your answer. Then, provide your answer.',
            'v4': f'Predict the most likely emotion people will feel when they see this image.',
            'v5': f'Which emotion will people feel the most when they see this image? '
                  f'Choose the answer from the list: {label_list_string}. '
                  f'Only give the chosen emotion.',
            'v6': f'Predict the most likely emotion people will feel when they see this image. '
                  f'You need to choose the answer from the list: {label_list_string}. '
                  f'Give the name of chosen emotion in the first sentence of your answer. '
                  f'Then, provide your explanation in the following sentences.',
            'v7': f'Predict the most likely emotion people will feel when they see this image. '
                  f'You need to choose answers from the list: {label_list_string}. '
                  f'Your response should take into account the visual cues, context, '
                  f'and any other relevant information. '
                  f'Remember to give the name of chosen emotion in the first sentence of your answer. '
                  f'Then, provide your explanation in the following sentences.',
            'v8': f'Which top three emotions will people feel when they see this image? '
                  f'Choose answers from the list: {label_list_string}. '
                  f'Only give the list of chosen emotions.',
            'v9': f'Which top three emotions will people feel when they see this image? '
                  f'Choose answers from the list delimited by square brackets. {label_list_string}.',
            'v10': f'Which top three emotions will people feel when they see this image? '
                  f'Choose answers from the list: {label_list_string}. '
                  f'In the first sentence of answer, give the list of emotions. '
                  f'In the following sentences, provide your explanation.'
        }

        self.task_description_dict = {
            'v0': f'You are now an AI assistant specialized in human emotion analysis. '
                  f'In the following conversations, you will be provided with an image. '
                  f'Your task is to predict people\'s emotion when they see this provided image.'
        }

    def get_query(self, version):
        return self.query_dict[version]

    def get_task_description(self, include_description, version):
        if not include_description:
            return None
        else:
            return self.task_description_dict[version]


def get_image_pil(img_np):
    img_np = ((img_np + 0.5) * 255).astype(np.uint8)  # [c, h, w]
    img_np = img_np.transpose(1, 2, 0)
    return Image.fromarray(img_np).convert('RGB')


def get_label_list_string(label_list_):
    output_string = str(sorted(label_list_))
    output_string = output_string.replace('\'', '')
    return output_string


def get_past_job_index(input_llava_file):
    output_index = int(input_llava_file['query_list_length'])
    if output_index > 0:
        output_index -= 1
    return output_index


def dict_recursive():
    return defaultdict(dict_recursive)


def write_log(path, content):
    with open(path, 'a') as f:
        f.write(content)
        f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--dataset", type=str, choices=['Emotion6', 'UnBiasedEmotion', 'Emotion-6', 'DeepEmotion'])
    parser.add_argument("--M", type=str, default='v0')
    parser.add_argument("--Q", type=str, default='v1')
    parser.add_argument("--model_weight_dir", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--ignore_previous_run", action='store_true')
    args = parser.parse_args()

    num_gpus = args.gpus
    model_version = args.M  # [v0, v1-1]
    query_version = args.Q  # [v1-10]
    include_task_description = False
    dataset_name = args.dataset
    query_type = 'SI-SQ'
    ignore_previous_run = args.ignore_previous_run
    model_weight_dir = args.model_weight_dir
    dataset_dir = args.dataset_dir

    if dataset_name == 'Emotion6':
        label_subset_list = [
            'anger', 'sadness', 'joy', 'fear', 'surprise', 'disgust']
    elif dataset_name == 'Emotion-6':
        label_subset_list = [
            'anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    elif dataset_name == 'UnBiasedEmotion':
        label_subset_list = [
            'anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    else:  # dataset_name == 'DeepEmotion':
        label_subset_list = [
            'amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

    sample_args = {
        'do_sample': False,
        'num_beams': 5,
        'max_new_tokens': 1024
    }

    ################################

    params_dict = dict_recursive()
    params_dict['params']['M'] = model_version
    params_dict['params']['Q'] = query_version
    params_dict['params']['INCLUDE_TASK_DESCRIPTION'] = include_task_description
    params_dict['params']['DATASET'] = dataset_name
    params_dict['params']['QUERY_TYPE'] = query_type
    params_dict['params']['ARGS'] = sample_args
    params_dict['params']['LABEL_LIST'] = label_subset_list
    params_dict['query_list_length'] = 0

    model_path = f'{model_weight_dir}/llava_weights_{model_version}'
    dataset_path = f'{dataset_dir}/{dataset_name}/data_pickle/'
    output_path = f'./LMM_Emotion_Prediction/llava_embedding/' \
                  f'{dataset_name}/{query_type}/{dataset_name}_llava_M:{model_version}_Q:{query_version}_{query_type}'
    log_path = f'./LMM_Emotion_Prediction/logs/log_embed_{dataset_name}_M:{model_version}_Q:{query_version}_{query_type}.txt'
    dataset = CustomDataSet(dataset_path, split='all', select_label_subset=True, label_subset_list=label_subset_list)
    label_list_string = get_label_list_string(label_subset_list)

    os.makedirs(output_path, exist_ok=True)

    query_worker = QueryClass()
    model_worker = ModelWorker(model_path, num_gpus)

    query_step1 = query_worker.get_query(query_version)

    start_index = 0
    load_previous_run = False
    if os.path.isfile(os.path.join(output_path, 'params.pkl')) and not ignore_previous_run:
        load_previous_run = True
        with open(os.path.join(output_path, 'params.pkl'), 'rb') as f:
            params_dict = pickle.load(f)
        start_index = get_past_job_index(params_dict)
        write_log(log_path, f'LOAD FROM PREVIOUS RUN, index: {start_index}')
        write_log(log_path, '======================')

    task_description = query_worker.get_task_description(include_task_description, 'v0')
    if include_task_description:
        prompt = model_worker.insert_prompt(input_query=task_description)
        params = {'prompt': prompt, 'images': model_worker.conv.get_images(return_pil=True)}
        answer = model_worker.eval_model(params, sample_args)
        model_worker.append_answer(answer)
        if not load_previous_run:
            params_dict['task_intro']['q'] = task_description
            params_dict['task_intro']['a'] = answer
    else:
        if not load_previous_run:
            params_dict['task_intro']['q'] = task_description
            params_dict['task_intro']['a'] = None

    if not load_previous_run:
        params_dict['query_step']['step1'] = query_step1
        params_dict['query_step']['step2'] = None

        with open(os.path.join(output_path, 'params.pkl'), 'wb') as file:
            pickle.dump(params_dict, file)

    model_conv_copy = model_worker.conv.copy()

    for idx in range(start_index, len(dataset)):
        item = dataset.get_item(idx, return_np=True)
        label_list = [dataset.num2text[label_num] for label_num in item['label']]
        raw_image = get_image_pil(item['data'])

        model_worker.conv = model_conv_copy.copy()

        # step 1
        prompt = model_worker.insert_prompt(input_query=query_step1, image=raw_image)
        params = {'prompt': prompt, 'images': model_worker.conv.get_images(return_pil=True)}
        answer_1, embedding = model_worker.eval_model(params, sample_args)
        model_worker.append_answer(answer_1)

        # # step 2
        # prompt = model_worker.insert_prompt(input_query=query_step2)
        # params = {'prompt': prompt, 'images': model_worker.conv.get_images(return_pil=True)}
        # answer_2 = model_worker.eval_model(params, sample_args)
        # model_worker.append_answer(answer_2)

        save_dict = {
            'label': label_list,
            'info': item['info'].split('/')[-1],
            'answer_detail': answer_1,
            'embedding': embedding.astype(np.float32)
        }
        write_log(log_path, str(idx))
        write_log(log_path, '=======================')

        with open(os.path.join(output_path, f'{idx:05}.pkl'), 'wb') as file:
            pickle.dump(save_dict, file)

        params_dict['query_list_length'] = idx
        with open(os.path.join(output_path, 'params.pkl'), 'wb') as file:
            pickle.dump(params_dict, file)
