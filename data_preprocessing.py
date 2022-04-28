#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_data(filename):
    
    data = pd.read_csv(filename)  
    #Age, MonthClaimed and DayOfWeekClaimed have missing values.
    data.loc[(data["Age"] == 0),"Age"] = 17
    data.loc[(data["MonthClaimed"] == '0'),"MonthClaimed"] = "Jan"
    data.loc[(data["DayOfWeekClaimed"] == '0'),"DayOfWeekClaimed"] = "Monday"
    
    return data


def convert_to_category(Insurance_Data):
    #Convert data to Catergoral
    Insurance_Data["DayOfWeek"] = Insurance_Data["DayOfWeek"].astype('category').cat.codes
    Insurance_Data["Make"] = Insurance_Data["Make"].astype('category').cat.codes
    Insurance_Data["AccidentArea"] = Insurance_Data["AccidentArea"].astype('category').cat.codes
    Insurance_Data["DayOfWeekClaimed"] = Insurance_Data["DayOfWeekClaimed"].astype('category').cat.codes
    Insurance_Data["MonthClaimed"] = Insurance_Data["MonthClaimed"].astype('category').cat.codes
    Insurance_Data["Sex"] = Insurance_Data["Sex"].astype('category').cat.codes
    Insurance_Data["MaritalStatus"] = Insurance_Data["MaritalStatus"].astype('category').cat.codes
    Insurance_Data["Fault"] = Insurance_Data["Fault"].astype('category').cat.codes
    Insurance_Data["PolicyType"] = Insurance_Data["PolicyType"].astype('category').cat.codes
    Insurance_Data["VehiclePrice"] = Insurance_Data["VehiclePrice"].astype('category').cat.codes
    Insurance_Data["Days_Policy_Accident"] = Insurance_Data["Days_Policy_Accident"].astype('category').cat.codes
    Insurance_Data["Days_Policy_Claim"] = Insurance_Data["Days_Policy_Claim"].astype('category').cat.codes
    Insurance_Data["PastNumberOfClaims"] = Insurance_Data["PastNumberOfClaims"].astype('category').cat.codes
    Insurance_Data["AgeOfVehicle"] = Insurance_Data["AgeOfVehicle"].astype('category').cat.codes
    Insurance_Data["PoliceReportFiled"] = Insurance_Data["PoliceReportFiled"].astype('category').cat.codes
    Insurance_Data["WitnessPresent"] = Insurance_Data["WitnessPresent"].astype('category').cat.codes
    Insurance_Data["AgentType"] = Insurance_Data["AgentType"].astype('category').cat.codes
    Insurance_Data["NumberOfSuppliments"] = Insurance_Data["NumberOfSuppliments"].astype('category').cat.codes
    Insurance_Data["NumberOfCars"] = Insurance_Data["NumberOfCars"].astype('category').cat.codes
    Insurance_Data["Month"] = Insurance_Data["Month"].astype('category').cat.codes
    Insurance_Data["VehicleCategory"] = Insurance_Data["VehicleCategory"].astype('category').cat.codes
    Insurance_Data["AgeOfPolicyHolder"] = Insurance_Data["AgeOfPolicyHolder"].astype('category').cat.codes
    Insurance_Data["AddressChange_Claim"] = Insurance_Data["AddressChange_Claim"].astype('category').cat.codes
    Insurance_Data["BasePolicy"] = Insurance_Data["BasePolicy"].astype('category').cat.codes
    
    return Insurance_Data



def feature_selection(Insurance_Data):
    #Check for multicolinearity
    Features = Insurance_Data.copy()
    Features = Features.drop(["PolicyNumber","FraudFound_P"],axis=1)

    #Correlation before selection
    corr = Features.corr()
    kot = corr[(corr>= 0.6) | (corr <= -0.6)]
    plt.figure(figsize = (14,12))   
    dataplot=sns.heatmap(kot,cmap="Wistia", annot=True)
    plt.title('Correlation Between Insurance Features\nBefore Selection')
    plt.savefig("Images/Corr before FS.jpg",transparent = False)
    plt.show()


    #Dropped columns with high multiple regression variables (VIF) after examination
    Features = Features.drop("VehicleCategory",axis=1)
    Features = Features.drop("AgeOfPolicyHolder",axis=1)
    Features = Features.drop("Month",axis=1)

    drop_col = ["PolicyNumber","VehicleCategory","AgeOfPolicyHolder","Month"]
    Insurance_Data = Insurance_Data.drop(drop_col,axis=1)

    #Correlation after selection
    corr = Features.corr()
    kot = corr[(corr>= 0.6) | (corr <= -0.6)]
    plt.figure(figsize = (14,12))   
    dataplot=sns.heatmap(kot,cmap="Wistia", annot=True)
    plt.title('Correlation Between Insurance Features\n After Selection')
    plt.savefig("Images/Corr After FS.jpg",transparent = False)
    plt.show()

    return Insurance_Data


def get_class_distribution(data):
    #Class distribution
    true_count = data[data['FraudFound_P'] == 1].shape[0]
    false_count = data[data['FraudFound_P'] == 0].shape[0]

    total = data.shape[0]
    input_data = [true_count, false_count]
    labels = ['Fraud', 'Not Fraud']

    explode = [0,0.2]

    #define Seaborn color palette to use
    plt.figure(figsize = (8, 6))
    colors = sns.color_palette('pastel')[3:5]

    #create pie chart
    plt.pie(input_data, labels = labels, 
            colors = colors, 
            autopct=lambda x: '{:.1f}%\n({})'.format(x, format(int((total*x/100).round(0)),",")),
            startangle = 20,explode =explode)

    plt.title("Fraud vs Non-Fraud Cases\nIn the Dataset")
    plt.savefig("Images/Class Distribution.jpg",transparent = False)
    plt.show()

    
#DRIVER CODE    
Insurance_Data = read_data("fraud_oracle.csv")

display(Insurance_Data.head())

Insurance_Data = convert_to_category(Insurance_Data)

Insurance_Data = feature_selection(Insurance_Data)

get_class_distribution(Insurance_Data)

display(Insurance_Data.head())

Insurance_Data.to_csv("Insurance_Data.csv", index=False)






