import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split




class DataLoader:
    def data_loading(self):
        train_df = pd.read_csv("Dataset\train.csv")
        test_df = pd.read_csv("Dataset\test.csv")

    def shape(self,train_df,test_df):
        print(f"shape of the Train Data : {train_df.shape}")
        print(f"shape of the Test Data : {test_df.shape}")

    def Null_count(self,train_df,test_df):    
        print("No of Null Values in Train Data")
        print(train_df.isnull().sum())
        print("\n")

        print("No of Null Values in Test Data")
        print(test_df.isnull().sum())


    def duplicates(self,train_df,test_df):
        print(f"Duplicates in Test Data : {train_df.duplicated().sum()}")   
        print(f"Duplicates in Test Data : {test_df.duplicated().sum()}")    


    def data_cleaning(self,col,df):
        df[col] = df[col].str.replace('<h1>','')
        df[col] = df[col].str.replace('</h1>','')
        df[col] = df[col].str.replace('<h2>','')
        df[col] = df[col].str.replace('</h2>','')

        df[col] = df[col].str.findall(r"[\d.]+")
        df[col] = df[col].apply(lambda x: float(x[0]))

        print(f"Null values in {col} Level : {df[col].isnull().sum()}")
        return df[col]

    def datasplit(self,x,y):
        x_train , x_test ,y_train , y_test = train_test_split(x,y,test_size =0.2,random_state =42) 
        return x_train , x_test ,y_train , y_test


