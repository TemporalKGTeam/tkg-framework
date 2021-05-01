from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import torch



path = '/Users/GengyuanMax/Downloads/translation_rtranse_best.ckpt'
ckpt = torch.load(path, map_location=torch.device('cpu'))
head_ent_emb = ckpt['state_dict']['_entity_embeddings._tail.real.weight'].numpy()
temp_emb = ckpt['state_dict']['_temporal_embeddings._temporal.real.weight'].numpy()

head_ent_emb0 = head_ent_emb # + temp_emb[120:121, :]
X_pca = PCA(n_components=2).fit_transform(head_ent_emb0)

plt.figure(figsize=(16, 40))

#[2, 5, 10, 30, 50, 100]
for i, n in enumerate([5, 15, 30, 50, 60, 80]):
    X_tsne = TSNE(n_components=2,perplexity=n,random_state=13,n_iter=1000).fit_transform(head_ent_emb0)


    #
    plt.subplot(6,2,2*i+1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],label="t-SNE")
    plt.legend()
    plt.subplot(6,2,2*i+2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1],label="PCA")
    plt.legend()

plt.show()