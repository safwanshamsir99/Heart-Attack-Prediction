# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:21:16 2022

@author: safwanshamsir99
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

#%% FUNCTION
def plot_cat(df,cat_data):
    '''
    This function is to generate plots for categorical columns

    Parameters
    ----------
    df : DataFrame
        DESCRIPTION.
    cat_data : LIST
        categorical column inside the dataframe.

    Returns
    -------
    None.

    '''
    for cat in cat_data:
        plt.figure()
        sns.countplot(df[cat])
        plt.show()

def plot_con(df,con_data):
    '''
    This function is to generate plots for continuous columns

    Parameters
    ----------
    df : DataFrame
        DESCRIPTION.
    continuous_col : LIST
        continuous column inside the dataframe.

    Returns
    -------
    None.

    '''
    for con in con_data:
        plt.figure()
        sns.distplot(df[con])
        plt.show()

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,         
        Journal of the Korean Statistical Society 42 (2013): 323-328    
    """    
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n    
    r,k = confusion_matrix.shape    
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%% STATIC
CSV_PATH = os.path.join(os.getcwd(),'dataset','heart.csv')
BEST_PIPE_PATH = os.path.join(os.getcwd(),'heart_fine_tune.pkl')
MODEL_PATH = os.path.join(os.getcwd(),'best_model_heart.pkl')

#%% DATA LOADING
df = pd.read_csv(CSV_PATH)

#%% DATA INSPECTION
df.info() # 9 categorical data, 5 in continuous (not determine by dtypes)
df['thall'] = df['thall'].replace(0,np.NaN) # since 0 is null based on the data description
df.isna().sum() # 2 NaNs 
df.duplicated().sum() # 1 duplicate data

# to display continuous data
con_data = ['age','trtbps','chol','thalachh','oldpeak']
plot_con(df,con_data)

# to display categorical data
cat_data = df.columns.difference(['age','trtbps','chol','thalachh',
                                  'oldpeak'],sort=False).tolist()
plot_cat(df,cat_data) # all of the categorical data are unbalance

df.describe().T # max chol level is very high; 564
df.boxplot() # trtbps,chol and a few of other columns have outliers

#%% DATA CLEANING
# 2 NaNs to be imputed
df['thall'] = df['thall'].fillna(df['thall'].mode()[0]) #impute NaNs using mode
df.isna().sum() # check

# drop duplicate data
df = df.drop_duplicates() # remove duplicate data
df.duplicated().sum() # check again

#%% FEATURES SELECTION
# categorical data (output) as the target
# no need to do Label Encoding bcs the dataset in numeric

# categorical features vs categorical target using cramer's V
for cat in cat_data:
    print(cat)
    confussion_mat = pd.crosstab(df[cat],df['output']).to_numpy()
    print(cramers_corrected_stat(confussion_mat))
'''
Based on the cramer's V, the categorical columns that will be used for the
next step are cp(0.510) and thall(0.523)
'''

# continuous features vs categorical target using LogisticRegression
for con in con_data:
    logreg = LogisticRegression()
    logreg.fit(np.expand_dims(df[con],axis=-1), df['output'])
    print(con)
    print(logreg.score(np.expand_dims(df[con],axis=-1),df['output'])) # accuracy
'''
All of the continuous columns have a high correlation (>0.5) with output.
But, only age(0.620),thalachh(0.702) and oldpeak(0.686) will be selected 
for the next step since they have the highest correlation.
'''

#%% PREPROCESSING
X = df.loc[:,['age','cp','thalachh','oldpeak','thall']]
y = df.loc[:,'output']

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.3,
                                                 random_state=3)

#%% PIPELINE CREATION 
#LogisticRegression,RandomForest,DeccisionTree,KNeighbors,SVC
# LR
pl_std_lr = Pipeline([('Standard Scaler',StandardScaler()),
                      ('LogClassifier',LogisticRegression())]) 

pl_mm_lr = Pipeline([('Min Max Scaler',MinMaxScaler()),
                     ('LogClassifier',LogisticRegression())])

#RF
pl_std_rf = Pipeline([('Standard Scaler',StandardScaler()),
                      ('RFClassifier',RandomForestClassifier())]) 

pl_mm_rf = Pipeline([('Min Max Scaler',MinMaxScaler()),
                     ('RFClassifier',RandomForestClassifier())]) 

# Decision Tree
pl_std_tree = Pipeline([('Standard Scaler',StandardScaler()),
                        ('DTClassifier',DecisionTreeClassifier())]) 

pl_mm_tree = Pipeline([('Min Max Scaler',MinMaxScaler()),
                       ('DTClassifier',DecisionTreeClassifier())]) 

# KNeighbors
pl_std_knn = Pipeline([('Standard Scaler',StandardScaler()),
                       ('KNClassifier',KNeighborsClassifier())]) 

pl_mm_knn = Pipeline([('Min Max Scaler',MinMaxScaler()),
                      ('KNClassifier',KNeighborsClassifier())])

# SVC
pl_std_svc = Pipeline([('Standard Scaler',StandardScaler()),
                       ('SVClassifier',SVC())]) 

pl_mm_svc = Pipeline([('Min Max Scaler',MinMaxScaler()),
                      ('SVClassifier',SVC())])

# create pipeline
pipelines = [pl_std_lr,pl_mm_lr,pl_std_rf,pl_mm_rf,pl_std_tree,
             pl_mm_tree,pl_std_knn,pl_mm_knn,pl_std_svc,pl_mm_svc]

# fitting the data
for pipe in pipelines:
    pipe.fit(X_train,y_train)

pipe_dict = {0:'SS+LR', 
             1:'MM+LR',
             2:'SS+RF',
             3:'MM+RF',
             4:'SS+Tree',
             5:'MM+Tree',
             6:'SS+KNN',
             7:'MM+KNN',
             8:'SS+SVC',
             9:'MM+SVC'}
best_accuracy = 0

# model evaluation
for i,model in enumerate(pipelines):
    print(model.score(X_test,y_test))
    if model.score(X_test, y_test) > best_accuracy:
        best_accuracy = model.score(X_test,y_test)
        best_pipeline = model
        best_scaler = pipe_dict[i]
        
print('The best pipeline for heart dataset will be {} with accuracy of {}'
      .format(best_scaler, best_accuracy))

#%% Fine tune the model (SS + SVM) 
'''
Based on the pipeline, model with highest accuracy is Standard Scaler 
and SV Classifier with accuracy of 0.824.
'''
pl_ss_svc = Pipeline([('Standard Scaler',StandardScaler()),
                      ('SVClassifier',SVC())])

# number of trees
grid_param = [{'SVClassifier':[SVC()],
               'SVClassifier__kernel':['rbf','linear','poly'],
               'SVClassifier__C':[0.2,0.5,1.0,1.5]}]

gridsearch = GridSearchCV(pl_ss_svc,grid_param,cv=5,verbose=1,n_jobs=1)
best_model = gridsearch.fit(X_train, y_train)
print(best_model.score(X_test,y_test))
print(best_model.best_index_)
print(best_model.best_params_)

# saving the best pipeline
with open(BEST_PIPE_PATH,'wb') as file:
    pickle.dump(best_model,file)

'''
Since the accuracy is the same(0.824) but different in params,the ml model
will use the fine tune parameters.
'''
#%% RE-TRAIN THE MODEL
pl_ss_svc = Pipeline([('Standard Scaler',StandardScaler()),
                      ('SVClassifier',SVC(C=0.2,kernel='linear'))])
pl_ss_svc.fit(X_train,y_train)

# saving the best model
with open(MODEL_PATH,'wb') as file:
    pickle.dump(pl_ss_svc,file)

#%% MODEL ANALYSIS
y_true = y_test
y_pred = pl_ss_svc.predict(X_test)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print('Accuracy score: ' + str(accuracy_score(y_true, y_pred)))

#%% DISCUSSION
'''

'''







