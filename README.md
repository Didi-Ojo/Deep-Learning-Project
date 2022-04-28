# CNN-LSTM - Predictive Analysis for Automobile Insurance Fraud Detection

A Predictive model to detect automobile insurance fraud using a hybrid CNN-LSTM Deep Learning Model and Machine Learning approaches. The CNN-LSTM model is compared to the traditional ML approach with models such as Logistic Regression, Random Forest and XGBoost.

## Dataset
  - __fraud_oracle.csv__  
    - The Dataset is available on Kaggle at https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection
    - The Dataset consist of 33 columns, having 32 data features and a target feature which indicates whether a claim is fraudulent or legitimate.
    - The Dataset is highly unbalanced, therefore class balancing techniques are used to improve the modelling performance.

  
## Requirements
The reaserch project was conducted using the below versions, however, other versions may be compatible 

Requirement                      | Version      | 
---------------------------------|--------------|
__numpy__                        | 1.19.5       | 
__Tensorflow__                   | 2.0.0        | 
__pandas__                       | 1.3.4        | 
__seaborn__                      | 0.11.2       | 
__xgboost__                      | 1.5.0        | 
__sklearn__                      | 1.0.2        | 
__matplotlib__                   | 3.5.1        | 

These package can be installed by:
    `!pip install package_name`
    
## Libraries

- __numpy__ For data processing
- __panda__ For data processing
- __tensorflow.keras.utils.to_categorical__ - to convert data to categorical data
- __tensorflow.keras.layersDense, LSTM, Dropout__ For Modelling
- __sklearn.metricsclassification_report, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix__ For metric evaluation
- __imblearn.over_sampling,SMOTE__ To perform Synthetic Minority Oversampling Technique
- __tensorflow.keras, models, layers, regularizers__ Used to build the Model
- __xgboost, XGBClassifier__ For Machine Learning Modelling
- __sklearn.ensemble,RandomForestClassifier__ For Machine Learning Modelling
- __sklearn.linear_model,LogisticRegression__ For Machine Learning Modelling
- __sklearn.preprocessing,MinMaxScaler__ For Data Normalization
- __seaborn,matplotlib.pyplot__ For Data Visualization



## Code Walkthrough

### Scripts
- ### data_preprocessing.py 
  This is used for the initial data preprocessing. The input is the dataset __fraud_oracle.csv__ described above. The output is the processed dataset __Insurance_Data.csv__ which has been cleaned up and converted to categorical data for model processing. This should to be ran prior to running the next 3 scripts.

- ### smote_modelling.py 
  This performs modelling using SMOTE technique to balance out the dataset. The input data is __Insurance_Data_csv__. The output of this script is a  bar chart which compares the performance of the hybrid CNN-LSTM against traditional Machine Learning techniques. 

- ### random_sampling.py 
  This performs modelling using random sampling to balance out the dataset. The input data is __Insurance_Data_csv__. The output of this script is a chart which comparing the performance of the CNN-LSTM with ML models 

- ### feature_importance_ML.py 
  This uses the output of data_preprocessing.py above to examine the feature importance of the machine learning models
  
  ## References
- [A Predictive Modeling for Detecting Fraudulent Automobile Insurance Claims](https://www.scirp.org/journal/paperinformation.aspx?paperid=94450#t2![image](https://user-images.githubusercontent.com/23123894/165863288-7559dc22-b4f7-4678-a621-38d61edb0b51.png))
- Xia, H., Zhou, Y. and Zhang, Z. (2022) ‘Auto insurance fraud identification based on a CNN-LSTM fusion deep learning model’,
Int. J. Ad Hoc and Ubiquitous Computing, Vol. 39, Nos. 1/2, pp.37–45. 
