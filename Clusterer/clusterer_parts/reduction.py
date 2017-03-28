from sklearn.decomposition import PCA


def pca(vectors, components=2):
    pca = PCA(n_components=components)
    return pca.fit_transform(vectors)
