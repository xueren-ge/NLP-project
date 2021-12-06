import matplotlib
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.rc('axes', edgecolor='k')


def visualize_embedding_space():
    """Plot the baseline charts in the paper. Images are written to the img/ subfolder."""
    plt.figure(figsize=(12, 4))
    icons = ['ro:', 'bo:', 'go:']

    for i, (model, num_layers) in enumerate([('BERT', 13)]):
        x = np.array(range(num_layers))
        data = json.load(open(f'{model.lower()}/embedding_space_stats.json'))
        plt.plot(x, [data["mean cosine similarity across words"][f'layer_{i}'] for i in x], icons[i], markersize=6,
                 label=model, linewidth=2.5, alpha=0.65)
        print(spearmanr(
            [data["mean cosine similarity across words"][f'layer_{i}'] for i in x],
            [data["word norm std"][f'layer_{i}'] for i in x]
        ))

    plt.grid(True, linewidth=0.25)
    plt.legend(loc='upper right')
    plt.xlabel('Layer Index')
    plt.xticks(x)
    plt.ylim(0.0, 1.0)
    plt.title("Average Cosine Similarity between Randomly Sampled Words")
    plt.savefig(f'figures/avg_cossim_random_samples.png', bbox_inches='tight')
    plt.close()


    plt.figure(figsize=(12, 4))
    icons = ['ro:', 'bo:', 'go:']

    for i, (model, num_layers) in enumerate([('BERT', 13)]):
        x = np.array(range(num_layers))
        data = json.load(open(f'{model.lower()}/embedding_space_stats.json'))
        y1 = np.array([data["mean cosine similarity between sentence and words"][f'layer_{i}'] for i in x])
        y2 = np.array([data["mean cosine similarity across words"][f'layer_{i}'] for i in x])
        plt.plot(x, y1 - y2, icons[i], markersize=6, label=model, linewidth=2.5, alpha=0.65)

    plt.grid(True, linewidth=0.25)
    plt.legend(loc='upper right')
    plt.xlabel('Layer Index')
    plt.xticks(x)
    plt.ylim(-0.1, 0.5)
    plt.title("Average Intra-Sentence Similarity (anisotropy-adjusted)")
    plt.savefig(f'figures/Intra_Sentence_Similarity.png', bbox_inches='tight')
    plt.close()


def visualize_self_similarity():
    """Plot charts relating to self-similarity. Images are written to the img/ subfolder."""
    plt.figure(figsize=(12, 4))
    icons = ['ro:', 'bo:', 'go:']

    # plot the mean self-similarity but adjust by subtracting the avg similarity between random pairs of words
    for i, (model, num_layers) in enumerate([('BERT', 13)]):
        embedding_stats = json.load(open(f'{model.lower()}/embedding_space_stats.json'))
        self_similarity = pd.read_csv(f'{model}/self_similarity.csv')

        x = np.array(range(num_layers))
        y1 = np.array([self_similarity[f'layer_{i}'].mean() for i in x])
        y2 = np.array([embedding_stats["mean cosine similarity across words"][f'layer_{i}'] for i in x])
        plt.plot(x, y1 - y2, icons[i], markersize=6, label=model, linewidth=2.5, alpha=0.65)

    plt.grid(True, linewidth=0.25)
    plt.legend(loc='upper right')
    plt.xlabel('Layer Index')
    plt.xticks(x)
    plt.ylim(0, 1)
    plt.title("Average Self-Similarity (anisotropy-adjusted)")
    plt.savefig(f'figures/Self_Similarity.png', bbox_inches='tight')
    plt.close()

    # list the top 10 words that are most self-similar and least self-similar
    most_self_similar = []
    least_self_similar = []
    models = []

    for i, (model, num_layers) in enumerate([('BERT', 13)]):
        self_similarity = pd.read_csv(f'{model}/self_similarity.csv')
        self_similarity['avg'] = self_similarity.mean(axis=1)

        models.append(model)
        most_self_similar.append(self_similarity.nlargest(10, 'avg')['word'].tolist())
        least_self_similar.append(self_similarity.nsmallest(10, 'avg')['word'].tolist())

    print(' & '.join(models) + '\\\\')
    for tup in zip(*most_self_similar): print(' & '.join(tup) + '\\\\')
    print()
    print(' & '.join(models) + '\\\\')
    for tup in zip(*least_self_similar): print(' & '.join(tup) + '\\\\')


def select_variance_case():
    """
    select some cases for visualization (VER)
    """
    plt.figure(figsize=(12, 8))
    model = 'BERT'
    x = np.array(range(1, 13))
    data = pd.read_csv(f'{model}/variance_explained.csv')
    embedding_stats = json.load(open(f'{model.lower()}/embedding_space_stats.json'))
    y0 = np.array([embedding_stats["variance explained for random words"][f'layer_{i}'] for i in x])
    ### go
    y1 = np.array([data.iloc[8][f'layer_{i}'] for i in x])
    plt.plot(x, y1 - y0, label='{}'.format(data.iloc[8]['word']))
    ### packages
    y2 = np.array([data.iloc[9][f'layer_{i}'] for i in x])
    plt.plot(x, y2 - y0, label='{}'.format(data.iloc[9]['word']))
    ### walk
    y3 = np.array([data.iloc[53][f'layer_{i}'] for i in x])
    plt.plot(x, y3 - y0, label='{}'.format(data.iloc[53]['word']))
    ### Chinese
    y4 = np.array([data.iloc[202][f'layer_{i}'] for i in x])
    plt.plot(x, y4 - y0, label='{}'.format(data.iloc[202]['word']))
    ### beverage
    y5 = np.array([data.iloc[417][f'layer_{i}'] for i in x])
    plt.plot(x, y5 - y0, label='{}'.format(data.iloc[417]['word']))
    plt.xticks(x)
    plt.ylim([0.0, 1.1])
    plt.grid(True, linewidth=0.25, axis='y')
    plt.legend(loc='upper right')
    plt.xlabel('Layer')
    plt.ylabel('variance explained')
    plt.title("Selected Samples of Maximum Explainable Variance (anisotropy-adjusted)")
    plt.savefig(f'figures/variance_explained.png')
    plt.close()


if __name__ == '__main__':
    visualize_self_similarity()
    visualize_embedding_space()
    select_variance_case()
