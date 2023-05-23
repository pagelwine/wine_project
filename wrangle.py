import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


def get_data():
    return pd.read_csv('wines.csv')

def prepare_data(data):
    data.columns = data.columns.str.replace(' ', '_')
    for i in data.columns:
        if i not in ['wine_type', 'quality']:
            data[i] = data[i][data[i] < data[i].quantile(.99)].copy()
    data = data.dropna()

    return data


def split_data(df):
    '''
    Takes in a dataframe and returns train, validate, test subset dataframes
    '''
    
    
    train, test = train_test_split(df,
                                   test_size=.2, 
                                   random_state=123, 
                                   
                                   )
    train, validate = train_test_split(train, 
                                       test_size=.25, 
                                       random_state=123, 
                                       
                                       )
    
    return train, validate, test