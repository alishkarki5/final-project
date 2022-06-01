import os
import pandas as pd 

def data_path(joinpath: str) -> str:
    ''' Return the base path '''

    base = os.path.dirname(os.path.abspath(__file__))
    actual_path = base.replace('/dashboard', '')
    
    return os.path.join(actual_path, joinpath)

def read_dataset(path: str) -> pd.DataFrame:
    '''
    Read data from CSV file from path
    ------------
    Return 
    -------
    Pandas dataframe

    '''
    return  pd.read_csv(path)




