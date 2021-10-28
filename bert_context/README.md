# NLP Project
our project is doing BERT visualization and interpret. Our work is based on
paper ["How contextual are contextu-alized word representations?
comparing the geom-etry of BERT, ELMo, and GPT-2 embeddings"](https://arxiv.org/abs/1909.00512)

## Installation
1. install libraries
    ~~~
    pip install -r requirements.txt
    ~~~

2. preprocess
    ~~~
    python preprocess.py
    ~~~

3. analyse
    calculate Self_Similarity, Intra_Sentence_Similarity, maximum explainable variance
    ~~~
    python analyze.py
    ~~~

4. visualize
    visualize Self_Similarity, Intra_Sentence_Similarity and maximum explainable variance
    ~~~
    python visualize.py
    ~~~