import jsonlines
import json
import torch
from pytorch_pretrained import BertTokenizer
import pandas as pd
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from keras.preprocessing.sequence import pad_sequences
import spacy
from spacy.lang.en import English
from typing import Dict, Tuple,  List, Callable
import numpy as np
nlp = English()
spacy_tokenizer = nlp.tokenizer

def mapping(gold_label):
    if gold_label == 'entailment':
        return 0
    elif gold_label == 'neutral':
        return 1
    elif gold_label == 'contradiction':
        return 2
    else:
        return -1

def retriveData(path):
    dataset = []
    with jsonlines.open(path) as f:
        for line in f.iter():
            tmpdict = {
                'premise': line['sentence1'],
                'hypothesis': line['sentence2'],
                'gold_label': mapping(line['gold_label'])
            }
            dataset.append(tmpdict)
    df = pd.DataFrame(dataset)
    return df

def rmvNeg(df):
    df = df[df.gold_label != -1]
    df.reset_index(drop=True, inplace=True)
    return df


def token2ids(df, MAX_LEN):
    premise = df['premise'].values
    hypothesis = df['hypothesis'].values
    sentences = []
    for i, row in df.iterrows():
        sentences.append("[CLS] " + premise[i] + " [SEP] " + hypothesis[i] + " [SEP]")

    labels = df.gold_label.values
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    return input_ids, labels

def createAttnMask(input_ids):
    # Create attention masks
    attn_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attn_masks.append(seq_mask)
    return attn_masks

def generateDataset(dataPath='./dataset/snli_1.0_train.jsonl', MAX_LENGTH=64, isTrain=True):
    df_train = retriveData(dataPath)
    df_train = rmvNeg(df_train)
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

def split_train_val_test(df, props=[.7, .15, .15]):
    left, mid, right = int(len(df) * props[0]), int(len(df) * (props[0]+props[1])), len(df)
    train_df = df.iloc[:left, :]
    val_df = df.iloc[left:mid, :]
    test_df = df.iloc[mid:right, :]
    return train_df, val_df, test_df

def token2idsMedWeb(df, MAX_LEN):
    reviews = df['Reviews'].values
    sentences = []
    for i, row in df.iterrows():
        if (len(reviews[i]) > 512):
            reviews[i] = reviews[i][:500]
        sentences.append("[CLS] " + reviews[i] + " [SEP]")

    labels = [int(i-1) for i in df['Satisfaction'].values]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    return input_ids, labels

def df2DataLoader(df, MAX_LENGTH=64, isTrain=True):
    input_ids, labels = token2idsMedWeb(df, MAX_LENGTH)
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

def rmvNegMedWeb(df):
    df = df[df.Satisfaction != -1]
    df.reset_index(drop=True, inplace=True)
    return df

def generateMedWeb(dataPath='./dataset/webmd.csv', MAX_LENGTH=64):
    df = pd.read_csv(dataPath)
    df = df.drop(columns=['Age', 'Condition', 'Date', 'Drug', 'DrugId', 'EaseofUse', 'Effectiveness', 'Sex', 'Sides', 'UsefulCount'])
    df['Reviews'].replace(' ', np.nan, inplace=True)
    df['Satisfaction'].replace(10, np.nan, inplace=True)
    df = df.dropna()
    df = df.reset_index(drop=True)

    train_df, val_df, test_df = split_train_val_test(df)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv("./medweb_train.csv")
    val_df.to_csv("./medweb_val.csv")
    test_df.to_csv("./medweb_test.csv")

    train_loader = df2DataLoader(train_df, MAX_LENGTH, isTrain=True)
    val_loader = df2DataLoader(val_df, MAX_LENGTH, isTrain=False)
    test_loader = df2DataLoader(test_df, MAX_LENGTH, isTrain=False)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # train_dataloader = generateDataset(dataPath='./dataset/snli_1.0_dev.jsonl')
    # print(next(iter(train_dataloader)))
    generateMedWeb()