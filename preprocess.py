# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

feature_names = [
    'State', 'Account Length', 'Area Code', 'Phone',
    "Int'l Plan", 'VMail Plan', 'VMail Message',
    'Day Mins', 'Day Calls', 'Day Charge',
    'Eve Mins', 'Eve Calls', 'Eve Charge',
    'Night Mins', 'Night Calls', 'Night Charge',
    'Intl Mins', 'Intl Calls', 'Intl Charge',
                 'CustServ Calls', 'Churn?'
]

feature_dtypes = {
    'State': 'int64', 'Account Length': 'int64',
    'Area Code': 'object', 'Phone': 'object',
    "Int'l Plan": 'object', 'VMail Plan': 'object', 'VMail Message': 'int64',
    'Day Mins': 'float64', 'Day Calls': 'int64', 'Day Charge': 'float64',
    'Eve Mins': 'float64',  'Eve Calls': 'int64', 'Eve Charge': 'float64',
    'Night Mins': 'float64', 'Night Calls': 'int64', 'Night Charge': 'float64',
    'Intl Mins': 'float64', 'Intl Calls': 'int64', 'CustServ Calls': 'int64',
    'Churn?': 'object'
}

if __name__ == '__main__':
    dataDict = fetch_openml('churn')

    churn_array = np.concatenate(
        (dataDict['data'], dataDict['target'].reshape(5000, 1)), axis=1)

    churn = pd.DataFrame(
        data=churn_array, columns=feature_names).astype(feature_dtypes)

    churn["Int'l Plan"] = churn["Int'l Plan"].map({1: 'yes', 0: 'no'})
    churn['VMail Plan'] = churn['VMail Plan'].map({1: 'yes', 0: 'no'})
    churn['Churn?'] = churn['Churn?'].map({'1': 'True.', '0': 'False.'})

    churn = churn.drop(['Phone', 'Day Charge', 'Eve Charge',
                        'Night Charge', 'Intl Charge'], axis=1)

    model_data = pd.get_dummies(churn)
    model_data = pd.concat([model_data['Churn?_True.'], model_data.drop(
        ['Churn?_False.', 'Churn?_True.'], axis=1)], axis=1)

    train_data, validation_data, test_data = np.split(model_data.sample(
        frac=1, random_state=1729), [int(0.7 * len(model_data)), int(0.9 * len(model_data))])
    train_data.to_csv('train.csv', header=False, index=False)
    validation_data.to_csv('validation.csv', header=False, index=False)
