import pickle
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import seaborn as sns
import numpy as np

def standardization(dataframe, columns: list, scaler_path):
    std_scaler = StandardScaler()
    for column in columns:
        arr = np.array(dataframe[column])
        std_scaler.fit(arr.reshape(len(arr),1))
        pickle.dump(std_scaler, open(f"{scaler_path}sts_scaler_{column}.pickle", "wb"))
        dataframe[column] = std_scaler.transform(arr.reshape(len(arr),1))
    return dataframe

def encode_dataframe(dataframe, categorical_columns, encoder_path):
    """encode categorical column into numeric one"""
    le = preprocessing.LabelEncoder()

    for column in categorical_columns:
        arr = np.array(dataframe[column])
        le.fit(arr.reshape(len(arr),1))
        pickle.dump(le, open(f"{encoder_path}encoder_{column}.pickle", "wb"))
        dataframe[column] = le.transform(arr.reshape(len(arr),1))
    return dataframe

def split_data(data, data_label, test_size=0.25, random_state=0):
    """Split the data into train and test, validation set.

    Args:
        data (dataframe): With the numerical columns except label
        data_label (dataframe): With the label column
        test_size (float): Ratio of test set to be split from the data
        random_state (int): used to seed a new RandomState object.

    """
    return train_test_split(
        data,
        data_label,
        test_size=test_size,
        random_state=random_state,
    )


class ClassificationModel:
    def __init__(self, data, data_label, classification_model):
        """Fits a LR model to the data and returns classification report.

        Args:
            data (dataframe): With the numerical columns except label
            data_label (dataframe): With the label column

        """
        self.x_train, self.x_test, self.y_train, self.y_test = split_data(
            data, data_label
        )
        self.model = classification_model

    def train_model(self):
        """Train the model:
            model (anyClassificationModel): Pass any classification Algorithms 
            ---------
            Return:
            model after fitting into the data
        """

        return self.model.fit(self.x_train, self.y_train)

        

    def get_report(self):
        """Returns the classification report"""
        y_prediction = self.model.predict(self.x_test)
        report = classification_report(self.y_test, y_prediction, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        return df_report

    def generate_confusion_matrix(self):

        y_prediction = self.model.predict(self.x_test)
        cf_matrix = confusion_matrix(self.y_test, y_prediction)
        sns.heatmap(cf_matrix,  annot=True,cmap='Blues')
        

    def save_the_model(self, model_path):
        """Saves the model.

        Args:
            model_path (str): Path of the model

        """
        pickle.dump(self.model, open(model_path, "wb"))
