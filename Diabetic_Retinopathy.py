#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
import numpy as np


# In[10]:


st.title("PREDICTION OF DIABETIC RETINOPATHY")


# In[11]:


st.sidebar.header("User Input Parameters")


# In[12]:


def user_input_features():
    age = st.sidebar.number_input("Insert the age")
    systolic_bp = st.sidebar.number_input("Insert the systolic_bp")
    diastolic_bp = st.sidebar.number_input("Insert the diastolic_bp")
    cholesterol = st.sidebar.number_input("Insert the cholesterol")
    data = {'age':age,
            'systolic_bp':systolic_bp,
            'diastolic_bp':diastolic_bp,
            'cholesterol':cholesterol}
    features = pd.DataFrame(data,index=[0])
    return features
df = user_input_features()
st.subheader('User Input Parameters')
st.write(df)
df2 = pd.read_csv("pronostico_dataset.csv",sep=';')
df2.drop(["ID"],inplace=True,axis= 1)
X = df2.iloc[:,[0,1,2,3]]
Y = df2.iloc[:,4]
from sklearn.svm import SVC 
svm = SVC(probability=True)
svm.fit(X, Y)
SVM_pred = svm.predict(df)
st.subheader("Prediction Result")
st.write(SVM_pred)


# In[ ]:




