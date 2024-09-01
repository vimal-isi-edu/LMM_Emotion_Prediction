
This is the readme file to reproduce the accuracy for the paper: 
*Large Multimodal Models Thrive with Little Data for Image Emotion Prediction*

## Contents
- [Environment Setup](#environment-setup)
- [Dataset Prepare](#dataset-prepare)
- [Inference with LLaVA](#inference-using-llava)
- [Combine Inferencec Results](#combine-inference-results-and-generate-embedding)
- [Train the Adapter Network](#train-the-adapter-network)
- [Inference the Adapter Network](#inference-the-adapter-network)
- [Points need to know](#points-need-to-know)

## Environment Setup

1. Set up LLaVA environment

In this project, we use the LLaVA model proposed in [LLaVA Page](https://llava-vl.github.io/) to be our 
Large Multimodel Model, so the first step is to follow their instructions to set up the environment and prepare 
the LLaVA model weight for inference which are available in the provided link.

**NOTE**:
After this step, you should have the following locations available:

model_weight_dir, where you store the LLaVA weight, to easily switch between different versions, we recommend 
the following structure:

```
├── model_weight_dir
│   ├── llava_weights_v0
│   │   ├── **
│   ├── llava_weights_v1-1
│   │   ├── **
```

2. Set up our environment

Move our LMM_Emotion_Prediction folder to the LLaVA folder to get the following structure:

```
├── LLaVA
│   ├── llava
│   ├── playground
│   ├── scripts
│   ├── LMM_Emotion_Prediction
│   ├── ***
```

## Dataset Prepare

We use 4 datasets in this experiment:

1. [Emotion6](http://chenlab.ece.cornell.edu/downloads.html)
2. [UnbiasedEmotion](https://rpand002.github.io/emotion.html)
3. [Emotion-6](https://rpand002.github.io/emotion.html)
4. [FI8 (DeepEmotion)](https://qzyou.github.io/)

Create a dir named ${DATASET_DIR} and place these datasets by the following structure:

- `DATASET_NAME`: the name of the dataset, should choose from ['Emotion6', 'UnBiasedEmotion', 'Emotion-6', 'DeepEmotion'].

```
├── ${DATASET_DIR}
│   ├── ${DATASET_NAME}
│   │   ├── data_raw
│   │   │   ├── (put unpacked images dir here)
```

Then use the following command to do dataset preprocessing.

- `DATASET_DIR`: path to the dataset_dir.

```bash
cd ${LLaVA}
python LMM_Emotion_Prediction/dataset_preprocessing.py \ 
      --dataset ${DATASET_NAME} --dataset_dir ${DATASET_DIR}
```

**NOTE**:
After this step, you should have the following locations available:

```
├── ${DATASET_DIR}
│   ├── ${DATASET_NAME}
│   │   ├── data_raw
│   │   │   ├── (unpacked files)
│   │   ├── data_pickle
│   │   │   ├── sample
│   │   │   │   ├── 00001.pkl
│   │   │   │   ├── 00002.pkl
│   │   │   │   ├── ***
│   │   │   ├── data_info (label)
```

## Inference using LLaVA

Use the following command to do inference. 

- `GPU_NUM`: number of gpus (tested using 1).
- `DATASET_NAME`: the name of the dataset to do inference on, should choose from ['Emotion6', 'UnBiasedEmotion', 'Emotion-6', 'DeepEmotion']. 
- `PROMPT_INDEX`: index for the prompt following the same order in the paper. Example v1, v2 ... v10. Check details in the python file.
- `MODEL_WEIGHT_DIR`: path to the LLaVA weight in [Environment Setup](#environment-setup).
- `DATASET_DIR`: path to the ${DATASET_DIR} in [Dataset Prepare](#dataset-prepare).
- `ignore_previous_run`: set this flag if user wants to discard previous progress and run from scratch, 
otherwise it will load previous run (if any) and continue.

```bash
cd ${LLaVA}
python LMM_Emotion_Prediction/inference_LMM_single_query_embedding.py \ 
      --gpus ${GPU_NUM} --dataset ${DATASET_NAME} --M v1-1 --Q ${PROMPT_INDEX} \
      --model_weight_dir ${MODEL_WEIGHT_DIR} --dataset_dir ${DATASET_DIR}
```

**NOTE**:
After this step, you should have the following locations available:

```
├── LMM_Emotion_Prediction
│   ├── inference_LMM_single_query_embedding.py
│   ├── llava_embedding
│   │   ├── ${DATASET_NAME}
│   │   │   ├── SI-SQ
│   │   │   │   ├── ${DATASET_NAME}_llava_M:v0_Q:${PROMPT_INDEX}_SI-SQ
│   │   │   │   │   ├── params.pkl
│   │   │   │   │   ├── 00001.pkl
│   │   │   │   │   ├── 00002.pkl
│   │   │   │   │   ├── ***        
│   ├── ***
```

The params.pkl contains all the information we use to inference the LLaVA, the label list of the 
corresponding dataset, the prompt we use the trigger the LLaVA and more, check details in the python file.

Other pkl files start with number represent the inference results for the corresponding image in the dataset.
Each pkl file has:
- `label`: emotion label for this image,
- `info`: image info, currently is pkl file name,
- `answer_detail`: text response from LLaVA,
- `embedding`: LLaVA last hidden state with shape: [1, sequence_len, embedding_dim]

## Combine Inference Results and Generate Embedding

Use the following command to generate the LMM Embedding from the last hidden state.

- `DATASET_NAME`: the name of the dataset to do inference on, should choose from ['Emotion6', 'UnBiasedEmotion', 'Emotion-6', 'DeepEmotion'].
- `METHOD_VERSION`: combining method version following the method in the paper, input v0 to get accuracy in our paper. Check details in the python file.
- `PROMPT_INDEX`: index for the prompt following the same order in the paper. Example: v1, v2 ... v10, ALL. Check details in the python file.

```bash
cd ${LLaVA}
python LMM_Emotion_Prediction/combine_llava_embedding.py \ 
      --dataset ${DATASET_NAME} --combine_method ${METHOD_VERSION} \
      --M v0 --Q ${PROMPT_INDEX}
```

**NOTE**:
After this step, you should have the following locations available:

```
├── LMM_Emotion_Prediction
│   ├── inference_LMM_single_query_embedding.py
│   ├── llava_embedding       
│   ├── llava_embedding_combined_${METHOD_VERSION}
│   │   ├── ${DATASET_NAME}
│   │   │   ├── SI-SQ
│   │   │   │   ├── ${DATASET_NAME}_llava_M:v0_Q:${PROMPT_INDEX}_SI-SQ.pkl
```

## Train the Adapter Network

Use the following command to train the Adapter Network.

- `DATASET_NAME`: the name of the dataset to do inference on, should choose from ['Emotion6', 'UnBiasedEmotion', 'Emotion-6', 'DeepEmotion'].
- `PROMPT_INDEX`: index for the prompt following the same order in the paper. Example: v1, v2 ... v10, ALL. Check details in the python file.
- `METHOD_VERSION`: combining method version following the method in the paper, input v0 to get accuracy in our paper. Check details in the python file.
- `NN_SETTING`: neural network setting, v0 is liner mapping. Check details in the python file.

There are other parameters in this python file that we set as constant for our experiments, those who wish to change 
them may check details in the python file.

```bash
cd ${LLaVA}
python LMM_Emotion_Prediction/train_adapter.py \ 
      --dataset ${DATASET_NAME} --Q ${PROMPT_INDEX} --combine_method ${METHOD_VERSION} \
      --setting ${NN_SETTING}
```

**NOTE**:
After this step, you should have the following locations available:

```
├── LMM_Emotion_Prediction
│   ├── inference_LMM_single_query_embedding.py
│   ├── llava_embedding
│   ├── llava_embedding_combined_${METHOD_VERSION}
│   ├── checkpoints
│   │   ├── ***  
```

The checkpoints dir stores the checkpoint for the model

## Inference the Adapter Network

Use the following command to inference the Adapter Network in the local machine. We don't provide CLI for this script,
but we listed all the parameters at the start of the inference_adapter.py, change them based on the user setting before run it.

```bash
cd ${LLaVA}
python LMM_Emotion_Prediction/inference_adapter.py
```

## Points need to know

1. We will also provide logs in txt file when do LMM_inference and adapter network training, check the 
logs dir under LMM_Emotion_Prediction
2. To reproduce accuracy, the random seeds in train_adapter.py should be manually set and reuse in the inference file 
to have the same train val and test split

