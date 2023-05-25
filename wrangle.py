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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



def get_data():
    return pd.read_csv('wines.csv')


def prepare_data(data):
    data.columns = data.columns.str.replace(' ', '_')
    for i in data.columns:
        if i not in ['wine_type', 'quality']:
            data[i] = data[i][data[i] < data[i].quantile(.99)].copy()
    data = data.dropna()

    data['quality_bin'] = pd.qcut(data.quality, q=3, labels=['low','mid','high'])
    data['quality_bin'] = np.where(data['quality_bin'] == 'low', 'bad', 'good')

    return data


def split_data(df, target):
    '''
    Takes in a dataframe and returns train, validate, test subset dataframes
    '''
    
    
    train, test = train_test_split(df,
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify = df[target]
                                   )
    train, validate = train_test_split(train, 
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify = train[target]
                                       )
    
    return train, validate, test


def get_cluster_columns(train, validate, test, features_list, clusters =3, init_array='k-means++', iterations = 300):
    
    for list_item in features_list:

        X = train[[list_item[0], list_item[1]]]
       
        kmeans = KMeans(n_clusters=clusters
                , init = init_array
                , max_iter=iterations
            )
        kmeans.fit(X)

        train[list_item[0] + '_' + list_item[1]] = kmeans.predict(X)

        Y = validate[[list_item[0], list_item[1]]]

        validate[list_item[0] + '_' + list_item[1]] = kmeans.predict(Y)

        Z = test[[list_item[0], list_item[1]]]

        test[list_item[0] + '_' + list_item[1]] = kmeans.predict(Z)

    return train, validate, test 



def get_dummies(train):
    dummies_columns = train[['wine_type']]
    

    dummy = pd.get_dummies(dummies_columns, columns = dummies_columns.columns, drop_first=True)

    train = pd.concat([train, dummy], axis=1)

    return train


def chi2_test(train, columns_list):
    chi_df = pd.DataFrame({'feature': [],
                    'chi2': [],
                    'p': [],
                    'degf':[],
                    'expected':[]})
    
    for iteration, col in enumerate(columns_list):
        
        observed = pd.crosstab(train[col[0]], train[col[1]])
        chi2, p, degf, expected = stats.chi2_contingency(observed)

        chi_df.loc[iteration+1] = [col, chi2, p, degf, expected]
    return chi_df





def comparison_of_means_3(train, features):
    means_df = pd.DataFrame({'feature': [],
                        'T': [],
                       'P': []})
    
    for iteration, feature in enumerate(features):
        if feature != 'quality':
            t, p = stats.f_oneway(train['quality_bin'][train[feature] == 0],
                            train['quality_bin'][train[feature] == 1],
                            train['quality_bin'][train[feature] == 2],
                            )
            means_df.loc[iteration] = [feature, t, p]


    return means_df


def comparison_of_means_2(train, features):
    means_df = pd.DataFrame({'feature': [],
                        'T': [],
                       'P': []})
    
    for iteration, feature in enumerate(features):
        if feature != 'quality':
            t, p = stats.ttest_ind(train['quality_bin'][train[feature] == 'bad'],
                            train['quality_bin'][train[feature] == 'good'],
                            )
            means_df.loc[iteration] = [feature, t, p]


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


    #X_test_degree2 = pf.transform(X_test)
    pred_pr = glm.predict(X_test)
    rmse, r2 = metrics_reg(y_test, pred_pr)
    metrics_test_df.loc[1] = ['Generalized Linear Model (GLM)', round(rmse,2), r2]


    return metrics_train_df, metrics_validate_df, metrics_test_df


def mvp_info(train_scaled, validate_scaled, test_scaled,list_of_features):


    X_train = train_scaled[list_of_features]
    X_validate = validate_scaled[list_of_features]
    X_test = test_scaled[list_of_features]


    y_train = train_scaled.quality_bin
    y_validate = validate_scaled.quality_bin
    y_test = test_scaled.quality_bin


    return X_train, X_validate, X_test, y_train, y_validate, y_test


def get_act_pred_viz(train_scaled, test_scaled, features, target_train, target_test):
    '''Takes in train_scaled, test_scaled, list of features to send in, target_train and target test, and produces
    a regression plot for actual vs predicted based on Polynomial model.'''

    #Initialize the Polynomial Features
    pf = PolynomialFeatures(degree=3)

    #Fit and transform on train only, transform the test dataset based on the fit on train.

    train_degree2 = pf.fit_transform(train_scaled[features])
    test_degree2 = pf.transform(test_scaled[features])

    #Initialize Linear Regression.
    lr2 = LinearRegression(normalize=True)

    #Fit Linear regression based on the transformed train dataset.
    lr2.fit(train_degree2, target_train)

    #Assign variables to put into sns.regplot's x and y values.
    act = target_test
    pred = lr2.predict(test_degree2)

    #Creates regression plot for actual vs. predicted
    sns.regplot(x=act, y=pred, line_kws={'color':'red'}, scatter_kws={'alpha': 0.1})
    plt.xlabel('Actual Property Value, in Millions')
    plt.ylabel('Predicted Property Value, in Millions')
    plt.title('Visualization of Polynomial Model, Actual vs Predicted')
    plt.axhline(target_train.mean(), c='black', linestyle='--') #Creates black dashed line that shows baseline
    plt.text(x=2500000, y=500000, s='Baseline')
    plt.show()


def create_knn(X_train,y_train, X_validate, y_validate):
    '''
    creating a logistic_regression model
    fitting the logistic_regression model
    predicting the training and validate data
    '''
    the_df = pd.DataFrame(data=[
    {
        'model_train':'knn',
        'train_predict':2254/(2254+1268),
        'validate_predict':2254/(2254+1268),
        'n_neighbors': 'neighbors'
    }
    ])
    for i in range(20):
        knn = KNeighborsClassifier(n_neighbors=i+1)
        knn.fit(X_train, y_train)
        train_predict = knn.score(X_train, y_train)
        validate_predict = knn.score(X_validate, y_validate)
        the_df.loc[i+1] = ['KNeighborsClassifier', train_predict, validate_predict, i+1]


    return the_df


def create_logistic_regression(X_train,y_train, X_validate, y_validate):
    '''
    creating a logistic_regression model
    fitting the logistic_regression model
    predicting the training and validate data
    '''
    the_df = pd.DataFrame(data=[
    {
        'model_train':'LogisticRegression',
        'train_predict':2254/(2254+1268),
        'validate_predict':2254/(2254+1268),
        'C': 'the_c'
    }
    ])

    for iteration, i in enumerate([.01, .1, 1, 10, 100, 1000]):
        logit = LogisticRegression(random_state= 123,C=i)
        logit.fit(X_train, y_train)
        train_predict = logit.score(X_train, y_train)
        validate_predict = logit.score(X_validate, y_validate)
        the_df.loc[iteration + 1] = ['LogisticRegression', train_predict, validate_predict, i]

    return the_df


def create_random_forest(X_train,y_train, X_validate, y_validate,X_test, y_test):
    '''
    creating a random_forest model
    fitting the random_forest model
    predicting the training and validate data
    '''
    the_df = pd.DataFrame(data=[
    {
        'model_train':'RandomForestClassifier',
        'train_predict':2254/(2254+1268),
        'validate_predict':2254/(2254+1268),
        'max_depth': 'max_depth'
    }
    ])
    test_df = pd.DataFrame(data=[
    {
        'model_train':'RandomForestClassifier',
        'baseline':2254/(2254+1268),
        'max_depth': 'max_depth'
    }
    ])

    for i in range(20):
        forest = RandomForestClassifier(random_state = 123,max_depth=i +1 )
        forest.fit(X_train, y_train)    
        train_predict = forest.score(X_train, y_train)
        validate_predict = forest.score(X_validate, y_validate)
        the_df.loc[i + 1] = ['RandomForestClassifier', train_predict, validate_predict, i + 1]

    forest = RandomForestClassifier(random_state = 123,max_depth=5 )
    forest.fit(X_train, y_train)  
    test_predict = forest.score(X_test, y_test)
    test_df.loc[1] = ['RandomForestClassifier', test_predict, 5]
    
    return the_df, test_df




def create_descision_tree(X_train,y_train, X_validate, y_validate):
    '''
    creating a Decision tree model
    fitting the Descision tree model
    predicting the training and validate data
    '''

    the_df = pd.DataFrame(data=[
    {
        'model_train':'DecisionTreeClassifier',
        'train_predict':2254/(2254+1268),
        'validate_predict':2254/(2254+1268),
        'max_depth': 'max_depth'
    }
    ])

    for i in range(20):

        tree = DecisionTreeClassifier(random_state = 123,max_depth= i + 1)
        tree.fit(X_train, y_train)
        train_predict = tree.score(X_train, y_train)
        validate_predict = tree.score(X_validate, y_validate)
        the_df.loc[i + 1] = ['DecisionTreeClassifier', train_predict, validate_predict, i + 1]

    return the_df

def super_classification_model(X_train,y_train, X_validate, y_validate, the_c = 1, neighbors = 20):
    the_df = pd.DataFrame(data=[
    {
        'model_train':'baseline',
        'train_predict':2254/(2254+1268),
        'validate_predict':2254/(2254+1268)
    }
    ])

    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(X_train, y_train)
    train_predict = knn.score(X_train, y_train)
    validate_predict = knn.score(X_validate, y_validate)
    knn, train_predict, validate_predict
    the_df.loc[1] = ['KNeighborsClassifier', train_predict, validate_predict]

    logit = LogisticRegression(random_state= 123,C=the_c)
    logit.fit(X_train, y_train)
    train_predict = logit.score(X_train, y_train)
    validate_predict = logit.score(X_validate, y_validate)
    the_df.loc[2] = ['LogisticRegression', train_predict, validate_predict]


    forest = RandomForestClassifier(random_state = 123, max_depth=5)
    forest.fit(X_train, y_train)    
    train_predict = forest.score(X_train, y_train)
    validate_predict = forest.score(X_validate, y_validate)
    the_df.loc[3] = ['RandomForestClassifier', train_predict, validate_predict]    


    tree = DecisionTreeClassifier(random_state = 123,max_depth=6)
    tree.fit(X_train, y_train)
    train_predict = tree.score(X_train, y_train)
    validate_predict = tree.score(X_validate, y_validate)
    the_df.loc[4] = ['DecisionTreeClassifier', train_predict, validate_predict]    

    return the_df


def correlation_charts(train):
    plt.figure(figsize=(14,3))
    plt.suptitle('Bivariate Exploration: The Strongest Correlators of Wine Quality')
    for i, col in enumerate(train[['density', 'chlorides', 'alcohol', 'volatile_acidity', 'quality']]):
        if col != 'quality':
            plt.subplot(1, 4, i+1)
            sns.regplot(data = train, x = col, y = 'quality', scatter_kws={'alpha': 0.1}, line_kws={'color': 'red'})

    plt.show()


def subplots_one(train):
    c_list = ['red', 'green']

    plt.figure(figsize=(14,5))

    plt.subplot(131)
    sns.scatterplot(data=train, x='chlorides', y='alcohol', hue='quality_bin', palette=c_list, alpha=0.2)
    plt.title('Good quality wines have higher alcohol content and low chlorides.')

    plt.subplot(132)
    sns.scatterplot(data=train, x='volatile_acidity', y='density', hue='quality_bin', palette=c_list, alpha=0.2)
    plt.title('Good wines have low volatile acidity')

    plt.subplot(133)
    sns.scatterplot(data=train, x='volatile_acidity', y='alcohol', hue='quality_bin', palette=c_list, alpha=0.2)
    plt.title('Bad wines tend to have low alcohol content and higher volatile acidity')

    plt.tight_layout()
    plt.show()


def train_scaled_two(train, train_scaled, c_list):
    plt.figure(figsize=(15, 10))

    plt.subplot(2,2,1)
    sns.scatterplot(data=train, x='volatile_acidity', y='density', hue='quality_bin', palette=c_list, alpha=0.2)

    plt.subplot(2,2,2)
    sns.scatterplot(data=train_scaled, x='volatile_acidity', y='density', hue='volatile_acidity_density', palette= ['green', 'red'], alpha=0.2)

    plt.show()


def train_scaled_three(train, train_scaled, c_list):

    plt.figure(figsize=(15, 10))

    plt.subplot(2,2,1)
    sns.scatterplot(data=train, x='volatile_acidity', y='alcohol', hue='quality_bin', palette=c_list, alpha=0.2)

    plt.subplot(2,2,2)
    sns.scatterplot(data=train_scaled, x='volatile_acidity', y='alcohol', hue='volatile_acidity_density', palette= ['green', 'red'], alpha=0.2)

    plt.show()

def train_scaled_four(train, train_scaled, c_list):
    plt.figure(figsize=(15, 10))

    plt.subplot(2,2,1)
    sns.scatterplot(data=train, x='chlorides', y='alcohol', hue='quality_bin', palette=c_list, alpha=0.2)

    plt.subplot(2,2,2)
    sns.scatterplot(data=train_scaled, x='chlorides', y='alcohol', hue='chlorides_alcohol', palette= ['green', 'red'], alpha=0.2)

    plt.show()
