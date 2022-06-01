import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_principle_components(data):
    """Creates two principle components

    Args:
        data (dataframe): With the numerical columns except label

    Returns:
        principal_components_df (dataframe):  Two columns as Principle component 1 and Principle component 2

    """
    scalar = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    principle_components = pca.fit_transform(scalar)
    principal_components_df = pd.DataFrame(
        data=principle_components,
        columns=["Principle component 1", "Principle component 2"],
    )
    return principal_components_df


def get_tsne_components(data):
    """Creates two tsne components

    Args:
        data (dataframe): With the numerical columns except label

    Returns:
        tsne_components_df (dataframe):  Two columns as tsne component 1 and tsne component 2

    """
    tsne = TSNE(n_components=2, learning_rate=50)
    tsne_components = tsne.fit_transform(data)
    tsne_components_df = pd.DataFrame(
        data=tsne_components,
        columns=["tsne component 1", "tsne component 2"],
    )
    return tsne_components_df

def change_label(data , column_names: list, to_column: str):
    return data.label.replace(column_names, to_column,inplace=True)
    
