import os
import json
import spacy
from spacy.lang.en import English
from typing import Dict, Tuple,  List, Callable

nlp = spacy.load("en_core_web_sm")
spacy_tokenizer = English().Defaults.create_tokenizer(nlp)

import numpy
import torch
import h5py
from pytorch_pretrained import BertTokenizer, BertModel
from tqdm import tqdm
import torch.nn as nn
import copy


def deleteEncodingLayers(model, layers_to_remove):
    oldModuleList = model.model.encoder.layer
    newModuleList = nn.ModuleList()

    # loop through 12 layers
    for i in range(0, 12):
        # if layer i+1 is in remove list, then remove it
        if i+1 in layers_to_remove:
            continue
        else:
            newModuleList.append(oldModuleList[i])

    copyModel = copy.deepcopy(model)
    copyModel.model.encoder.layer = newModuleList

    return copyModel

class Vectorizer:
    """
    Abstract class for creating a tensor representation of size (#layers, #tokens, dimensionality)
    for a given sentence.
    """

    def vectorize(self, sentence: str) -> numpy.ndarray:
        """
        Abstract method for tokenizing a given sentence and return embeddings of those tokens.
        """
        raise NotImplemented

    def make_hdf5_file(self, sentences: List[str], out_fn: str) -> None:
        """
        Given a list of sentences, tokenize each one and vectorize the tokens. Write the embeddings
        to out_fn in the HDF5 file format. The index in the pretrained_model corresponds to the sentence index.
        """
        sentence_index = 0

        with h5py.File(out_fn, 'w') as fout:
            for sentence in tqdm(sentences):
                embeddings = self.vectorize(sentence)
                fout.create_dataset(str(sentence_index), embeddings.shape, dtype='float32', data=embeddings)
                sentence_index += 1

class iBERT(Vectorizer):
    def __init__(self, layers_to_remove: List):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        self.model = BertModel.from_pretrained('bert-base-cased')
        # if there is layers to be removed
        if len(layers_to_remove) > 0:
            self.model = deleteEncodingLayers(self.model, layers_to_remove)
        self.linear = nn.Linear(768, 3)

    def vectorize(self, sentence: str):
        """
        Return a tensor representation of the sentence of size (13 layers, num tokens, 768 dim).
        Even though there are only 12 layers in GPT2, we include the input embeddings as the first
        layer (for a fairer comparison to ELMo).
        """
        # add CLS and SEP to mark the start and end
        tokens = ['[CLS]'] + self.tokenizer.tokenize(sentence) + ['[SEP]']
        # tokenize sentence with custom BERT tokenizer
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # segment ids are all the same (since it's all one sentence)
        segment_ids = numpy.zeros_like(token_ids)

        tokens_tensor = torch.tensor([token_ids])
        segments_tensor = torch.tensor([segment_ids])
        return tokens_tensor, segments_tensor


    def forward(self, sentence):
        tokens_tensor, segments_tensor = self.vectorize(sentence)
        embeddings, _, input_embeddings = self.model(tokens_tensor, segments_tensor)
        embeddings = torch.stack([input_embeddings] + embeddings, dim=0).squeeze()[:, 1:-1, :]
        embeddings = embeddings.detach().numpy()
        out = self.linear(embeddings)
        return out


if __name__ == '__main__':
    model = Freeze_Layer12_Model()
    new_model = deleteEncodingLayers(model, 11)

    for name, param in new_model.named_parameters():
        print(name, param.size())
