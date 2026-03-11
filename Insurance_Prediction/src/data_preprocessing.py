# 1,laod raw data
# 2. Indentifying x and y (inupt or output)
# 3. Split data into train and test

import pandas as pd 
from sklearn.model_selection import train_test_split
def load_and_split_data():
    data=pd.read_csv("../data/raw/Insurance_dataset.csv")
    x=data[['Age','Annual_Income_LPA','Policy_Term_Years','Sum_Assured_Lakhs']]
    y=data['Annual_Premium_Thousands']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    return x_train,x_test,y_train,y_test
x_train,x_test,y_train,y_test=load_and_split_data()
print(x_train)