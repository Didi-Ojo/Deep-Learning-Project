#!/usr/bin/env python
# coding: utf-8

#Predictive Analysis using CNN-LSTM Hydrid model for Automobile Fraud Detection - NO SMOTE
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, GRU
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE


def data_normalization(train_df, test_df):
    scaler = MinMaxScaler()
    #fitting the scaler to training data
    scaler = scaler.fit(train_df)
    #Transforming training and test data using scaler
    train_df = scaler.transform(train_df)
    test_df = scaler.transform(test_df)
    return train_df, test_df

def Random_Sample_data_processing(filename):
    Insurance_Data = pd.read_csv(filename)

    Non_Fraud_Cases = Insurance_Data[Insurance_Data["FraudFound_P"] ==0]
    Fraud_Cases = Insurance_Data[Insurance_Data["FraudFound_P"] ==1]

    Non_Fraud_Cases = Non_Fraud_Cases.sample(frac=1).reset_index(drop=True)
    Fraud_Cases = Fraud_Cases.sample(frac=1).reset_index(drop=True)

    X_train_NF = Non_Fraud_Cases.iloc[:500]
    X_train_F = Fraud_Cases.iloc[:500]

    X_test_NF = Non_Fraud_Cases.iloc[500:]
    X_test_F = Fraud_Cases.iloc[500:]

    X_train = pd.concat([X_train_NF, X_train_F])
    X_test = pd.concat([X_test_NF, X_test_F])

    X_train = X_train.sample(frac=1).reset_index(drop=True)
    X_test = X_test.sample(frac=1).reset_index(drop=True)

    y_train = X_train["FraudFound_P"]
    y_test = X_test["FraudFound_P"]


    drop_col = ["FraudFound_P"]
    X_train = X_train.drop(drop_col,axis=1)
    X_test = X_test.drop(drop_col,axis=1)

    X_train = X_train.values.astype(float)
    X_test = X_test.values.astype(float)

    y_train = to_categorical(np.array(y_train))
    y_test = to_categorical(np.array(y_test))

    X_train, X_test = data_normalization(X_train, X_test)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)


    return X_train,X_test,y_train,y_test



def cnn_lstm(input_shape):
    cnn_lstm = tf.keras.models.Sequential()
    cnn_lstm.add(layers.Conv1D(filters=128, 
                     kernel_size=3, 
                     input_shape=input_shape))
    cnn_lstm.add(layers.MaxPooling1D((2)))

    cnn_lstm.add(layers.Conv1D(filters=64, 
                     kernel_size=3))
    cnn_lstm.add(layers.MaxPooling1D((2)))


    cnn_lstm.add(LSTM(32, activation='relu', input_shape=input_shape,
                     kernel_regularizer=regularizers.l2(0.001)))
    cnn_lstm.add(Dense(16, activation='relu'))
    cnn_lstm.add(Dense(2, activation='sigmoid'))
    cnn_lstm.summary()
    
    return cnn_lstm

def run_cnn_lstm_model(X_train, X_test, y_train, y_test):
    cnn_lstm_model = cnn_lstm((X_train.shape[1],X_train.shape[2]))
    
    cnn_lstm_model.compile(optimizer='RMSprop', 
                      loss='binary_crossentropy',
                      metrics=['acc'])
    
    history1 = cnn_lstm_model.fit(X_train, 
                             y_train, 
                             epochs=200, 
                             batch_size=128, 
                             validation_split=0.2, 
                             verbose=1)

    acc = history1.history['acc']
    val_acc = history1.history['val_acc']
    loss = history1.history['loss']
    val_loss = history1.history['val_loss']

    epochs = range(1, len(acc)+1)

    plt.figure()
    plt.plot(epochs, acc, 'purple', label='Training acc')
    plt.plot(epochs, val_acc,'orange', label='Validation acc')
    plt.title('Training and validation Accuracy')
    plt.legend()
    plt.savefig("Images/Accuracy_Epoch_NO_SMOTE.jpg",transparent = False)
    plt.show()


    plt.plot(epochs, loss,'purple', label='Training loss')
    plt.plot(epochs, val_loss,'orange', label='Validation loss')
    plt.title('Training and validation Loss')
    plt.savefig("Images/Loss_Epoch_NO_SMOTE.jpg",transparent = False)
    plt.legend()
    plt.show()

    best_epoch_model = np.argmin(history1.history['val_loss'])
    print("best epoch:%s \nvalidation loss:%.3f \nvalidation acc:%.3f"%(best_epoch_model, 
                                                                     history1.history['val_loss'][best_epoch_model], 
                                                                     history1.history['val_acc'][best_epoch_model]))

    cnn_lstm_model = cnn_lstm((X_train.shape[1],X_train.shape[2]))

    cnn_lstm_model.compile(optimizer='RMSprop', 
                      loss='binary_crossentropy',
                      metrics=['acc'])

    history1 = cnn_lstm_model.fit(X_train, 
                             y_train, 
                             epochs=best_epoch_model, 
                             batch_size=128, 
                             validation_split=0.2, 
                             verbose=1)

    pred_train = np.argmax(cnn_lstm_model.predict(X_train), axis= -1)
    y_true_train = np.argmax(y_train,axis= -1)
    pred_test = np.argmax(cnn_lstm_model.predict(X_test), axis= -1)
    y_true_test = np.argmax(y_test,axis= -1)


    plt.rcParams['figure.figsize'] = [10, 4]
    cf_matrix = confusion_matrix(y_true_train,pred_train)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Purples',fmt='g')
    ax.set_title('Confusion Matrix\nof CNN-LSTM Model on Train Set (Without SMOTE)');
    ax.set_ylabel('Actual Category ');
    ax.set_xlabel('Predicted Category')
    ax.xaxis.set_ticklabels(['Not Fraud','Fraud'])
    ax.yaxis.set_ticklabels(['Not Fraud','Fraud'])
    plt.savefig("Images/Confusion_Matrix_Train_NO_SMOTE.jpg",transparent = False)
    plt.show()

    print("\n\n")
    plt.rcParams['figure.figsize'] = [10, 4]
    cf_matrix = confusion_matrix(y_true_test,pred_test)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Purples',fmt='g')
    ax.set_title('Confusion Matrix\nof CNN-LSTM Model on Test Set (Without SMOTE)');
    ax.set_ylabel('Actual Category ');
    ax.set_xlabel('Predicted Category')
    ax.xaxis.set_ticklabels(['Not Fraud','Fraud'])
    ax.yaxis.set_ticklabels(['Not Fraud','Fraud'])
    plt.savefig("Images/Confusion_Matrix_Test_NO_SMOTE.jpg",transparent = False)
    plt.show()

    print("CNN-LSTM EVALUATION ON TEST SET")
    acc = accuracy_score(y_true_test, pred_test)
    print("Accuracy: ",acc.round(2))
    
    f1 = f1_score(y_true_test, pred_test)
    print("F1 Score: ",f1.round(2))

    recall = recall_score(y_true_test, pred_test)
    print("Recall: ",recall.round(2))

    precision =  precision_score(y_true_test, pred_test)
    print("Precision: ",precision.round(2))

    print(classification_report(y_true_test, pred_test))

    temp_df = pd.DataFrame([])
    temp_df["Measure"] = ["Accuracy","Precision","Recall","F1_Score"]
    temp_df["Values"] = [acc, f1, recall, precision]
    temp_df["Model"] = "CNN-LSTM"
    return temp_df

    
def run_ML_model(X_train, X_test, y_train, y_test):
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1])
    y_train = np.argmax(y_train,axis= -1)
    y_test = np.argmax(y_test,axis= -1)

    models={"XGBoost":XGBClassifier(random_state = 2,use_label_encoder=False,eval_metric="logloss"),
            "Random Forest":RandomForestClassifier(random_state = 2),
            "Logistic Regression":LogisticRegression(random_state=2,solver='lbfgs',max_iter=2000)}

    Final_df = pd.DataFrame([])

    for mod,model_classifier in models.items():

        print("\n\t\t\t\t",mod)
        model = model_classifier
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision =  precision_score(y_test, y_pred)

        temp_df = pd.DataFrame([])
        temp_df["Measure"] = ["Accuracy","Precision","Recall","F1_Score"]
        temp_df["Values"] = [acc, precision, recall, f1]
        temp_df["Model"] = mod
        Final_df = pd.concat([Final_df, temp_df])
        
    return Final_df


def print_results(cnn_lstm_results,ML_results):
    Final_df = pd.concat([cnn_lstm_results,ML_results])
    Model_group = Final_df.groupby(["Model","Measure"]).mean()
    Model_group = Model_group["Values"]
    stack = Model_group.unstack("Measure")
    stack.plot.bar(width=0.6,color=["purple","#f7ce48","#4f9dc9","pink"])
    plt.title('Accuracy, Precison and Recall & F1 Score\n By Model (Without SMOTE)')
    plt.ylabel('Performace')
    plt.xlabel('Models')
    plt.xticks(rotation = 0)
    plt.rcParams['figure.figsize'] = [13, 5]
    plt.savefig("Images/Model_Comparison_NO_SMOTE.jpg")
    plt.show()


#DRIVER CODE
X_train, X_test, y_train, y_test = Random_Sample_data_processing("Insurance_Data.csv")

cnn_lstm_results = run_cnn_lstm_model(X_train, X_test, y_train, y_test)

ML_results = run_ML_model(X_train, X_test, y_train, y_test)

print_results(cnn_lstm_results,ML_results)

