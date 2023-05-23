import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor


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


def get_dummies_and_hot_encoded(train):
    X = train[['fixed_acidity', 'volatile_acidity']]

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    train['cluster_fix_vol_acid'] = kmeans.predict(X)


    X = train[['citric_acid', 'residual_sugar']]

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    train['cluster_cit_acd_res_sug'] = kmeans.predict(X)


    X = train[['chlorides', 'free_sulfur_dioxide']]

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    train['cluster_clorid_diox'] = kmeans.predict(X)


    dummies_columns = train[['wine_type','cluster_clorid_diox']]
    

    dummy = pd.get_dummies(dummies_columns, columns = dummies_columns.columns, drop_first=True)

    train = pd.concat([train, dummy], axis=1)

    return train


def comparison_of_means(train):

    means_df = pd.DataFrame({'feature': [],
                        'T': [],
                       'P': []})
    
    t, p = stats.f_oneway(train['quality'][train['cluster_fix_vol_acid'] == 0],
                     train['quality'][train['cluster_fix_vol_acid'] == 1],
                     train['quality'][train['cluster_fix_vol_acid'] == 2],
                    )
    means_df.loc[1] = ['cluster_fix_vol_acid', t, p]


    t, p = stats.f_oneway(train['quality'][train['cluster_cit_acd_res_sug'] == 0],
                     train['quality'][train['cluster_cit_acd_res_sug'] == 1],
                     train['quality'][train['cluster_cit_acd_res_sug'] == 2],
                    )

    means_df.loc[2] = ['cluster_cit_acd_res_sug', t, p]


    t, p = stats.f_oneway(train['quality'][train['cluster_clorid_diox'] == 0],
                     train['quality'][train['cluster_clorid_diox'] == 1],
                     train['quality'][train['cluster_clorid_diox'] == 2],
                    )

    means_df.loc[3] = ['cluster_clorid_diox', t, p]

    return means_df


def correlation_tests(train):
    corr_df = pd.DataFrame({'feature': [],
                        'r': [],
                       'p': []})
    for i, col in enumerate(train.drop(columns='wine_type')):
        r, p = stats.pearsonr(train[col], train['quality'])
        corr_df.loc[i] = [col, abs(r), p]

    return corr_df.sort_values(by='r', ascending=False)


def scale_data(train,
               validate,
               test,
               cols = ['alcohol', 'density']):
    '''Takes in train, validate, and test set, and outputs scaled versions of the columns that were sent in as dataframes'''
    #Make copies for scaling
    train_scaled = train.copy() #Ah, making a copy of the df and then overwriting the data in .transform() to remove warning message
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #Initiate scaler, using Robust Scaler
    scaler = MinMaxScaler()
    #Fit to train only
    scaler.fit(train[cols])
    #Creates scaled dataframes of train, validate, and test. This will still preserve columns that were not sent in initially.
    train_scaled[cols] = scaler.transform(train[cols])
    validate_scaled[cols] = scaler.transform(validate[cols])
    test_scaled[cols] = scaler.transform(test[cols])
    return train_scaled, validate_scaled, test_scaled


def metrics_reg(y, yhat):
    '''
    send in y_true, y_pred and returns rmse, r2
    '''
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2


def get_model_numbers(X_train, X_validate, X_test, y_train, y_validate, y_test):
    '''
    This function takes the data and runs it through various models and returns the
    results in pandas dataframes for train, test and validate data
    '''
    baseline = y_train.mean()
    baseline_array = np.repeat(baseline, len(X_train))
    rmse, r2 = metrics_reg(y_train, baseline_array)

    metrics_train_df = pd.DataFrame(data=[
    {
        'model_train':'baseline',
        'rmse':rmse,
        'r2':r2
    }
    ])
    metrics_validate_df = pd.DataFrame(data=[
    {
        'model_validate':'baseline',
        'rmse':rmse,
        'r2':r2
    }
    ])
    metrics_test_df = pd.DataFrame(data=[
    {
        'model_validate':'baseline',
        'rmse':rmse,
        'r2':r2
    }
    ])


    Linear_regression1 = LinearRegression()
    Linear_regression1.fit(X_train,y_train)
    predict_linear = Linear_regression1.predict(X_train)
    rmse, r2 = metrics_reg(y_train, predict_linear)
    metrics_train_df.loc[1] = ['ordinary least squared(OLS)', rmse, r2]

    predict_linear = Linear_regression1.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, predict_linear)
    metrics_validate_df.loc[1] = ['ordinary least squared(OLS)', rmse, r2]


    lars = LassoLars()
    lars.fit(X_train, y_train)
    pred_lars = lars.predict(X_train)
    rmse, r2 = metrics_reg(y_train, pred_lars)
    metrics_train_df.loc[2] = ['lasso lars(lars)', rmse, r2]

    pred_lars = lars.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, pred_lars)
    metrics_validate_df.loc[2] = ['lasso lars(lars)', rmse, r2]


    pf = PolynomialFeatures(degree=2)
    X_train_degree2 = pf.fit_transform(X_train)
   

    pr = LinearRegression()
    pr.fit(X_train_degree2, y_train)
    pred_pr = pr.predict(X_train_degree2)
    rmse, r2 = metrics_reg(y_train, pred_pr)
    metrics_train_df.loc[3] = ['Polynomial Regression(poly2)', rmse, r2]

    X_validate_degree2 = pf.transform(X_validate)
    pred_pr = pr.predict(X_validate_degree2)
    rmse, r2 = metrics_reg(y_validate, pred_pr)
    metrics_validate_df.loc[3] = ['Polynomial Regression(poly2)', rmse, r2]


    glm = TweedieRegressor(power=2, alpha=0)
    glm.fit(X_train, y_train)
    pred_glm = glm.predict(X_train)
    rmse, r2 = metrics_reg(y_train, pred_glm)
    metrics_train_df.loc[4] = ['Generalized Linear Model (GLM)', rmse, r2]

    pred_glm = glm.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, pred_glm)
    metrics_validate_df.loc[4] = ['Generalized Linear Model (GLM)', rmse, r2]


    X_test_degree2 = pf.transform(X_test)
    pred_pr = pr.predict(X_test_degree2)
    rmse, r2 = metrics_reg(y_test, pred_pr)
    metrics_test_df.loc[1] = ['Polynomial Regression(poly2)', round(rmse,2), r2]


    return metrics_train_df, metrics_validate_df, metrics_test_df


def mvp_info(train_scaled, validate_scaled, test_scaled):


    X_train = train_scaled[['alcohol', 'density', 'wine_type_white', 'cluster_clorid_diox_1', 'cluster_clorid_diox_2']]
    X_validate = validate_scaled[['alcohol', 'density', 'wine_type_white', 'cluster_clorid_diox_1', 'cluster_clorid_diox_2']]
    X_test = test_scaled[['alcohol', 'density', 'wine_type_white', 'cluster_clorid_diox_1', 'cluster_clorid_diox_2']]


    y_train = train_scaled.quality
    y_validate = validate_scaled.quality
    y_test = test_scaled.quality


    return X_train, X_validate, X_test, y_train, y_validate, y_test
