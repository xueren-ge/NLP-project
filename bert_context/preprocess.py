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
        to out_fn in the HDF5 file format. The index in the data corresponds to the sentence index.
        """
        sentence_index = 0

        with h5py.File(out_fn, 'w') as fout:
            for sentence in tqdm(sentences):
                embeddings = self.vectorize(sentence)
                fout.create_dataset(str(sentence_index), embeddings.shape, dtype='float32', data=embeddings)
                sentence_index += 1


class BertBaseCased(Vectorizer):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        self.model = BertModel.from_pretrained('bert-base-cased')
        self.model.eval()

    def vectorize(self, sentence: str) -> numpy.ndarray:
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

        with torch.no_grad():
            embeddings, _, input_embeddings = self.model(tokens_tensor, segments_tensor)

        # exclude embeddings for CLS and SEP; then, convert to numpy
        embeddings = torch.stack([input_embeddings] + embeddings, dim=0).squeeze()[:, 1:-1, :]
        embeddings = embeddings.detach().numpy()

        return embeddings


def index_tokens(tokens: List[str], sent_index: int, indexer: Dict[str, List[Tuple[int, int]]]) -> None:
    """
    Given string tokens that all appear in the same sentence, append tuple (sentence index, index of
    word in sentence) to the list of values each token is mapped to in indexer. Exclude tokens that
    are punctuation.

    Args:
        tokens: string tokens that all appear in the same sentence
        sent_index: index of sentence in the data
        indexer: map of string tokens to a list of unique tuples, one for each sentence the token
            appears in; each tuple is of the form (sentence index, index of token in that sentence)
    """
    for token_index, token in enumerate(tokens):
        if not nlp.vocab[token].is_punct:
            if str(token) not in indexer:
                indexer[str(token)] = []

            indexer[str(token)].append((sent_index, token_index))


def index_sentence(data_fn: str, index_fn: str, tokenize: Callable[[str], List[str]], min_count=5) -> List[str]:
    """
    Given a data file data_fn with the format of sts.csv, index the words by sentence in the order
    they appear in data_fn.

    Args:
        index_fn: at index_fn, create a JSON file mapping each word to a list of tuples, each
            containing the sentence it appears in and its index in that sentence
        tokenize: a callable function that maps each sentence to a list of string tokens; identity
            and number of tokens generated can vary across functions
        min_count: tokens appearing fewer than min_count times are left out of index_fn

    Return:
        List of sentences in the order they were indexed.
    """
    word2sent_indexer = {}
    sentences = []
    sentence_index = 0

    with open(data_fn, encoding="utf8") as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            blocks = line.strip().split('\t')
            assert len(blocks) > 10
            if blocks[-1] == '-': continue
            sent1 = blocks[7]
            sent2 = blocks[8]
            index_tokens(tokenize(sent1), sentence_index, word2sent_indexer)
            index_tokens(tokenize(sent2), sentence_index + 1, word2sent_indexer)
            sentences.append(sent1)
            sentences.append(sent2)
            sentence_index += 2

    # remove words that appear less than min_count times
    infrequent_words = list(filter(lambda w: len(word2sent_indexer[w]) < min_count, word2sent_indexer.keys()))

    for w in infrequent_words:
        del word2sent_indexer[w]

    json.dump(word2sent_indexer, open(index_fn, 'w'), indent=1)

    return sentences


if __name__ == "__main__":
    # where to save the contextualized embeddings
    EMBEDDINGS_PATH = "contextual_embeddings"
    bert = BertBaseCased()
    sentences = index_sentence('snli.small.tsv', 'bert/word2sent.json', bert.tokenizer.tokenize)

    if not os.path.exists(EMBEDDINGS_PATH):
        os.makedirs(EMBEDDINGS_PATH)
    bert.make_hdf5_file(sentences, os.path.join(EMBEDDINGS_PATH, 'bert.hdf5'))
