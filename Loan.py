from flask import Flask
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pickle

train=pd.read_csv('train_ctrUa4K.csv')

train=train[['LoanAmount','Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']]

from sklearn.impute import SimpleImputer

impcat=SimpleImputer(strategy='most_frequent')

train.iloc[:,2:11]=impcat.fit_transform(train.iloc[:,2:11])

imp=SimpleImputer(missing_values=np.nan,
    strategy='mean', verbose=0)

train.iloc[:,0:1]=imp.fit_transform(train.iloc[:,0:1])

oneh_train=pd.get_dummies(train,columns=['Property_Area'],drop_first=False)

oneh_train=oneh_train.drop(columns=['Loan_ID', 'Gender', 'Married','Education','Dependents','Self_Employed'],axis=1)

oneh_x = oneh_train.drop(columns=['Loan_Status'], axis=1)

# converting input into array
oneh_arr=oneh_x.values

y= oneh_train['Loan_Status']

y_train=oneh_train['Loan_Status'].astype('category').cat.codes

from xgboost import XGBClassifier
xgb= XGBClassifier(objective='binary:logistic',eval_metric='logloss',
                   learning_rate= 0.1, max_depth= 6, n_estimators= 300,gamma=1.5,
                   subsample=1,importance_type='gain',random_state=None,use_label_encoder=False)

xgb.fit(oneh_arr,y_train)

pickle.dump(xgb,open('xgb.pkl','wb'))