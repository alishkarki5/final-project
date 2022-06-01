import streamlit as st
from dashboard.modeling import ClassificationModel
from dashboard.read_data import read_dataset
import pickle
import yaml
import random
import pandas as pd

with open("config.yaml", "r") as stream:
    config_link = yaml.safe_load(stream)

def load_pickle(model_path: str):
    """Loads and returns the model from the project directory.

    Args:
        model_path: Path of the model

    Returns:
        model. A linear_model loaded from the directory.

    """
    try:
        return pickle.load(open(model_path, "rb"))
    except pickle.UnpicklingError as e:
        raise f"Unpickling Error: {e}"
    except Exception as e:
        raise e

def show_result(data, data_label, clf, model_path):
    '''This function train the classifier and gererate all the reports of the Classifier 
    '''
    clf_obj = ClassificationModel(data, data_label, clf)
    clf_obj.train_model()
    clf_obj.generate_confusion_matrix()
    clf_obj.save_the_model(f"{model_path}fine_model_{str(clf).split('(')[0]}.pickle")
    return clf_obj.get_report()
      
def get_prediction(model_name: str):
    '''Function to predict on the basis of user input]
    ---------
    parameter:
        model_name:  string pickel file from trained data
    Return
    -------
    user_data(pd.DataFrame) all user entered data with conversion
    prediction(array) = predicted value
    input_size = user input size to check to predicted value 
    '''
    data = read_dataset('./raw_data/final_test_data.csv')
    user_data = pd.DataFrame(data.iloc[random.randint(0,len(data))]).T
    model = load_pickle(f"{config_link['model_path']}/fine_model_{model_name}.pickle")
    encoder = load_pickle(f"{config_link['encoder_path']}/encoder_label.pickle")
    prediction = encoder.inverse_transform(model.predict(user_data))

    return prediction
    

    
