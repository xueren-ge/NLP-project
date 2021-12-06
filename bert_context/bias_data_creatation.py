import pandas as pd
import numpy as np
import os
from data_utils import *
import json

forbidden_words = ['women', 'woman', 'girl', 'girls', 'mother', 'wife', 'female']

def defaultTokenizer(sentence):
    return sentence.split(' ')

def createBiasData(dataPath = "./dataset/snli_1.0_train.jsonl"):
    path = "./bias_dataset"
    if not os.path.exists("./bias_dataset"):
        os.mkdir(path)
    # dataPath = "./dataset/snli_1.0_train.jsonl"
    df_train = retriveData(dataPath)
    df_train = rmvNeg(df_train)
    dict = {}
    idx = 0
    for _, row in df_train.iterrows():
        cnt = 0
        for word in forbidden_words:
            if word in defaultTokenizer(row['premise']):
                cnt += 1
        if cnt == 0:
            tmpdict = {
                'premise': row['premise'],
                'hypothesis': row['hypothesis'],
                'gold_label': row['gold_label']
            }

            dict[idx] = tmpdict
            idx += 1

    with open(os.path.join(path, dataPath.split('/')[-1]), 'w') as f:
        json.dump(dict, f)

def generateDataset(df_train, MAX_LENGTH=64, isTrain=True):
    input_ids, labels = token2ids(df_train, MAX_LENGTH)
    attn_masks = createAttnMask(input_ids)
    input_ids = torch.tensor(input_ids)
    attn_masks = torch.tensor(attn_masks)
    labels = torch.tensor(labels)
    data = TensorDataset(input_ids, attn_masks, labels)
    if isTrain:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=32)
    return dataloader

if __name__ == '__main__':
    createBiasData()