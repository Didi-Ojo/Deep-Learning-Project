#!/usr/bin/env python
# coding: utf-8


#Feature Importance using Machine Learning Model
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def feature_importance_ML(filename):
    Insurance_Data = pd.read_csv(filename)
    Features = Insurance_Data.copy()
    Features = Features.drop(["FraudFound_P"],axis=1)
    Target = Insurance_Data["FraudFound_P"]

    X = Features.values.astype(float)
    y = np.array(Target)

    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)


    output_df = {"Feature": list(Features.columns)}
    
    models={"XGB":XGBClassifier(random_state = 2,use_label_encoder=False,eval_metric="logloss"),
            "RF":RandomForestClassifier(random_state = 2)}
    
    
    for key in models:
        model = models[key]
        # Training the model
        model.fit(X, y)

        # Computing the importance of each feature
        model_feature_importance = model.feature_importances_


        # Normalize model importance
        norm_importance = (model_feature_importance - 
                           np.min(model_feature_importance))/(np.max(model_feature_importance) - 
                                                                      np.min(model_feature_importance))  


        output_df[key] = list(norm_importance.round(3))

    output_df = pd.DataFrame.from_dict(output_df).sort_values(by=['XGB'],ascending =False).set_index('Feature')
    output_df.index.name = ''
    print(output_df)

    fig, axs = plt.subplots(1,2, figsize=(12,8))

    cmap = plt.get_cmap("PiYG")(np.flip(np.arange(30)))

    output_df = output_df.sort_values(by="XGB",ascending =True)
    axs[0].barh(output_df.index,output_df["XGB"],color=cmap)
    axs[0].set_title('Most Important Feature\n based on XGBoost Model')
    axs[0].set(ylabel='Feature')
    axs[0].set(xlabel='Importance')

    output_df = output_df.sort_values(by="RF",ascending =True)
    axs[1].barh(output_df.index,output_df["RF"],color=cmap)
    axs[1].set_title('Most Important Feature\n based on Random Forest Model')
    axs[1].set(ylabel='Feature')
    axs[1].set(xlabel='Importance')

    fig.tight_layout(pad=5.0)
    plt.savefig("Images/Feature_Importance_ML.jpg")
    plt.show()

feature_importance_ML("Insurance_Data.csv")

