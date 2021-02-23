#!/usr/bin/env python
# coding: utf-8

# # Data Science and Business-Analytics Internship task at The Sparks Foundation  (TSF) 
# 
# 
# # Name: Anjali Jha

# # Task-1 Student's Marks Prediction using Supervised ML

# # Simple Linear Regression: 
# ***This technique is used as we have just used two variables to perform this task.***

# **Importing all the data science library functions and preparing the dataframe for the task**

# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pwd #to chck for the directory of the notebook


# In[3]:


data=pd.read_csv("File.csv") #importing dataset and converting csv file into dataframe


# # Checking and Pre-processing the data for use

# In[4]:


data.head() #to display first 5 rows of the dataset


# In[5]:


data.tail() #to display the last 5 rows of the dataset


# In[6]:


data.info()        #to summarize the dataframe


# # Collecting Statistical Information of the dataframe

# In[7]:


data.describe()     #to summarize the statistical details of the dataframe


# In[8]:


data.mean()          #to find the mean on each column of the dataset


# In[9]:


data.isnull().sum()     #to return the null values in the dataframe


# # Displaying Scatter Plot of Hours v/s Scores 

# In[10]:


# plotting the dataframe
plt.scatter(x=data.Hours, y=data.Scores)      
plt.xlabel("study_hours")
plt.ylabel("study_time")
plt.title("Hours vs Scores")
plt.show()


# # Grid Representation of the dataframe

# In[11]:


plt.figure(figsize=(10,6))
plt.grid()
plt.scatter(x=data.Hours, y=data.Scores, c="green")
plt.xlabel("study_hours")
plt.ylabel("study_time")
plt.title("Hours vs Scores")
plt.show()


# # Heatmap Representation:-
# ***Heatmap will help to visualize the relationship between the attributes of the dataframe that is Hours and Scores***

# In[12]:


sns.heatmap(data.isnull())


# # Splitting the dataset into Features and Label

# In[13]:


X=data.iloc[:,0].values
Y=data.iloc[:,1].values


# # Splitting the dataset into Training-set and Testing-set

# In[14]:


#training and testing dataset

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state=0)


# In[15]:


print("Shape of X_Train", X_train.shape)
print("Shape of Y_Train", Y_train.shape)
print("Shape of X_Test", X_test.shape)
print("Shape of Y_Test", Y_test.shape)


# # Building Prediction Model for predicting the Scores

# In[16]:


from sklearn.linear_model import LinearRegression
pred_model= LinearRegression()
pred_model.fit(X.reshape(-1,1), Y)


# In[17]:


pred_model.coef_


# In[18]:


pred_model.intercept_


# # Plotting the prediction model as Scatter Plot 

# In[19]:


slope= pred_model.coef_*X+pred_model.intercept_
plt.scatter(X,Y)
plt.plot(X, slope, c="green")
plt.show()


# # Predicting the values (Scores)

# In[20]:


y_pred = pred_model.predict(X_test.reshape(-1,1))
y_pred


# In[21]:


print("Actual Values and Predicted Values")
for i in zip(Y_test[:5], y_pred[:5]):
    print(i)


# # Grid Representation of Actual Scores v/s Predicted Scores in Scatter Plot

# In[22]:


plt.grid()
plt.scatter(X_test, Y_test, label="Actual Values")
plt.scatter(X_test, y_pred, c='orange', label="Predicted Values")
plt.xlabel("Hours")
plt.xlabel("Scores")
plt.title("Actual Scores and Predicted Scores")
plt.legend()
plt.show()


# # Checking the Accuracy of the Prediction Model

# In[23]:


pred_model.score(X_test.reshape(-1,1), Y_test)


# # Calculating Root Mean Squared Error

# In[24]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test, y_pred)
rmse=np.sqrt(mse)
print(f"MSE: {mse} \n RMSE: {rmse}")


# # Predicting Score against 9.25 of Study Hours

# In[30]:


print(f"No of study Hours: 9.25 \n", 
      f"Marks Obtained will be: {pred_model.predict([[9.25]])[0]}")

