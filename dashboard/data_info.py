import numpy as np

class DataInformation:
    def __init__(self, df):
        """Provides all relevant details about the provided Dataframe

        Args:
             df(dataframe): Dataframe with multiple numerical target column and one categorical column named label
        """
        self.df = df

    def get_dataset_head(self):
        """Returns first 12 dataframe rows"""
        return self.df.head(12)

    def get_shape(self) -> tuple:
        """Returns dataframe shape"""
        return self.df.shape

    def get_dataset_columns(self) -> list:
        """Returns dataframe columns"""
        return self.df.columns

    def get_all_data(self):
        """Returns all data of the dataframe"""
        return self.df

    def get_data_label(self):
        """Returns the label column of the dataframe"""
        return self.df.label

    def get_data_excluding_label(self, only_include=None):
        """Returns all the columns except the label column

        Args:
            only_include(dataframe): A queried dataframe. None if it for all the data

        """
        if only_include is not None:
            return only_include
        else:
            return self.df.drop(["label"], axis=1)

    def get_no_of_null(self) -> str:
        """Returns a string message about the missing values"""
        return f"{self.df.isnull().sum().sum()} null values found !"

    def get_categorical_data_columns(self):
        """Returns the categorical columns"""
        object_data = self.df.select_dtypes("object")
        if not object_data.empty:
            return object_data.columns
        else:
            return "No Categorical Data Found !"

    def get_selected_column(self, selected_column):
        """Returns the dataframe for the selected column

        Args:
            selected_column: Name of the column

        """
        return self.df[selected_column]

    def check_data_imbalance(self):
        """Returns a dataframe of values in label and its counts."""
        return self.df["label"].value_counts()

    def get_data_statistics(self, selected_target):
        """Returns the statistics of the data

        Args:
            selected_target(str): Any value in labels columns or "Overall"

        """
        if selected_target == "Overall":
            return self.df.describe()
        else:
            return self.df[self.df.label == selected_target].describe()

    def get_unique_labels(self, with_overall=False):
        """Returns unique values of label columns

        Args:
            with_overall: True to insert the 'Overall' string at the beginning

        """
        if with_overall:
            return np.insert(self.df.label.unique(), 0, "Overall")
        else:
            return self.df.label.unique()
            
    def get_info(self):

        """Return the all info of the dataframe

        """
        return self.df.info()
    
    def drop_column(self, column_name: str):
        """Returns the dataframe after removing the column

        Args:
            column_name: Name of the column

        """
        return self.df.drop([column_name], axis=1)