# NLP Project
our project is doing BERT visualization and interpret. Our work is based on
paper ["How contextual are contextu-alized word representations?
comparing the geom-etry of BERT, ELMo, and GPT-2 embeddings"](https://arxiv.org/abs/1909.00512)

## Installation
1. install libraries
    ~~~
    pip install -r requirements.txt
    ~~~

2. preprocess: you will get hdf2 files which records word embedding and other infomation
   in BERT in created folder contextual_embeddings 
    ~~~
    python preprocess.py
    ~~~

3. analyse: calculate Self_Similarity, Intra_Sentence_Similarity, maximum explainable variance,
   you will get 2 csv files and 2 json files in bert folder. self-similarity.csv,variance_explained.csv records each words' 
   self-similarity, maximum explained variance in each layer, while embedding_space_stats.json records randomly
   selected words and sentences and their corresponding metrics (visualize anisotropy). word2sen.json records
   each word information (which sentence, which position in this sentence)
    ~~~
    python analyze.py
    ~~~

4. visualize
   visualize Self_Similarity, Intra_Sentence_Similarity and maximum explainable variance,
   you will get corresponding figures in img folder 
    ~~~
    python visualize.py
    ~~~