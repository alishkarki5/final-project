import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

plt.style.use("seaborn")

class Visualization:
    @staticmethod
    def plot_data_distribution(data):
        """Visualization for the distribution of data.

        Args:
            data (dataframe): With all numerical columns

        """
        fig, axes = plt.subplots(4, 2)
        axes = axes.ravel()
        for col, ax in zip(data.columns, axes):
            sns.histplot(data=data[col], kde=True, stat="count", ax=ax)
        fig.tight_layout()

    @staticmethod
    def plot_elbow_graph(data):
        
        """Visualization for the optimal number of clusters on a elbow plot.

        Args:
            data (dataframe): With all numerical columns

        """
        plt.rcParams["figure.figsize"] = (10, 4)
        within_cluster_sum_of_squares = []
        for index in range(1, 11):
            km = KMeans(
                n_clusters=index,
                init="k-means++",
                max_iter=300,
                n_init=10,
                random_state=0,
            )
            km.fit(data)
            within_cluster_sum_of_squares.append(km.inertia_)
        plt.plot(range(1, 11), within_cluster_sum_of_squares)
        plt.title("Elbow Plot to find the number of Clusters", fontsize=20)
        plt.xlabel("No of clusters (K)")
        plt.ylabel("Within Cluster Sum of Squares")

    @staticmethod
    def plot_pca_scatter(principle_components, data_labels, targets, colors):
        """Visualizing the principle components on 2D scatter plot.

        Args:
            principle_components (dataframe): Two columns as Principle component 1 and Principle component 2
            data_labels (dataframe): Single column of data label
            targets (list): A list of values that is to be plotted.
            colors (list): A list of color initials

        """
        plt.figure(figsize=(10, 8))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Principal Component - 1", fontsize=17)
        plt.ylabel("Principal Component - 2", fontsize=17)
        plt.title("Principal Component Analysis", fontsize=18, pad=15)
        for target, color in zip(targets, colors):
            indices_to_keep = data_labels == target
            plt.scatter(
                principle_components.loc[indices_to_keep, "Principle component 2"],
                principle_components.loc[indices_to_keep, "Principle component 1"],
                c=color,
                s=40,
            )
        plt.legend(targets, prop={"size": 15})

    @staticmethod
    def plot_tsne_scatter(tsne_components, data_labels, targets, colors):
        """Visualizing the tsne components on 2D scatter plot.

        Args:
            tsne_components (dataframe): Two columns as tsne component 1 and tsne component 2
            data_labels (dataframe): Single column of data label
            targets (list): A list of values that is to be plotted.
            colors (list): A list of color initials

        """
        plt.figure(figsize=(10, 8))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("TSNE Component - 1", fontsize=17)
        plt.ylabel("TSNE Component - 2", fontsize=17)
        plt.title("TSNE Analysis", fontsize=18, pad=15)
        for target, color in zip(targets, colors):
            indices_to_keep = data_labels == target
            plt.scatter(
                tsne_components.loc[indices_to_keep, "tsne component 1"],
                tsne_components.loc[indices_to_keep, "tsne component 2"],
                c=color,
                s=40,
            )
        plt.legend(targets, prop={"size": 15})

    @staticmethod
    def plot_correlation_heatmap(data):
        """Visualizing the correlation on a Heatmap.

        Args:
            data (dataframe): With all numerical columns

        """
        plt.rcParams["figure.figsize"] = (20, 12)
        plt.style.use("fivethirtyeight")
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap="viridis", linewidth=0.5)
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)

    @staticmethod
    def bar_graph(data, column_name):
        data[column_name].value_counts().plot(kind="bar")