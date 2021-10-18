#!/usr/bin/env python
# coding: utf-8

# 
# # Predicting Heart Failure: RandomForest Model
# 

# In[1174]:


import pandas as pd
import numpy as np


# In[1175]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[1176]:


df = pd.read_csv('heart.csv')


# In[1177]:


# Creating a function to set the figure size for the graphs. 

def fig_size(length, width):
    
    return plt.figure(figsize=(length, width))


# In[1178]:


# Creating a function to set the title of the graphs. 

def graph_title(title):
    
    return plt.title(title)


# In[1179]:


# Setting the style of the graphs

sns.set_style('whitegrid')


# # EDA

# **Getting to know the data**

# In[1180]:


df.head()


# In[1181]:


df.info()


# In[1182]:


df.describe().transpose()


# In[1183]:


fig_size(6,4)

sns.countplot(x='Sex', data=df, hue='HeartDisease')

graph_title('Number of Patients by Gender')


# There more males with a heart disease than females.
# 

# In[1184]:


fig_size(4,5)

sns.displot(x='Age', data=df, bins=30, kde=True)

graph_title('Age Distribution')


# The age of the patients is normaly distributed with an average age of 54 years.

# In[1185]:


fig_size(7,7)

sns.boxplot(x='Sex', y='Age', data=df)


graph_title('Age Distribution by Gender')


# The average age for male patients is 54 years old and 53 years old for female patients, no significant difference.

# In[1186]:


fig_size(10,6)

sns.heatmap(df.corr(), annot=True, cmap='viridis')

graph_title('Correlation Matrix')


# The correlation matrix shows a weak relatioinship between the variables or features. This means that only a small portion of the explained variable (Heart Disease) is expalined by the features.

# In[1187]:


fig_size(8,6)

df.corr()['HeartDisease'].sort_values().drop('HeartDisease').plot(kind='bar')

graph_title('Correlation Matrix Plot')


# In[1188]:


fig_size(8,6)

sns.countplot(x='ST_Slope', data=df, hue='HeartDisease')

graph_title('Peak Exercice Slope')


# In[1189]:


length_slope = df.groupby('ST_Slope')['HeartDisease'].count()

length_hd = len(df[df['HeartDisease']==1])

100* length_slope/length_hd


# 91% of patients diagnozed with a heart disease have a flat of the slope of the peak exercise.

# In[1190]:


fig_size(8,6)

order = sorted(df['ChestPainType'].unique())

sns.countplot(x='ChestPainType',data=df, order = order, hue='HeartDisease')

graph_title('Chest Pain Types')


# In[1191]:


cp = df.groupby('ChestPainType')['HeartDisease'].count()

100* cp/length_hd


# 98% of patients diagnozed with a heart disease have a Asymptomatic Chest Pain.

# # Data Preprocessing

# **Categorical Variables and Dummy Variables**
# 

# Let's set the dummy variables for the categorical values in the dataset. 

# In[1192]:


# List of columns with object type values

df.select_dtypes('object').columns


# In[1193]:


dummies = pd.get_dummies(df[['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']], drop_first=True)


# In[1194]:


df = df.drop(['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], axis=1)


# In[1195]:


df = pd.concat([df, dummies], axis=1)


# In[1196]:


df.head()


# **Splitting data into training and testing datasets**

# Let's use 30% of the dataset as testing data and the remaning 70% as training data.

# In[1197]:


from sklearn.model_selection import train_test_split


# In[1198]:


X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']


# In[1199]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# **Normalizing the data**

# Before creating and training our Machiine Learning prediction model, it is necessary to normalize the dataset. The normalizatioin of the data consist of setting all the data to common scale while keeping the difference in the range of values for propressing. This article provides more resources on data normalization in Machine Learning: [Why Data Normalization is necessary for Machine Learning models](https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029)

# In[1200]:


from sklearn.preprocessing import MinMaxScaler


# In[1201]:


scaler = MinMaxScaler()


# In[1202]:


X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# In[1203]:


X_train.shape


# # Creating and Evaluating the Model

# **Creating the model**

# Now that we have normalize the data, let's now create and train the prediction model using the Random Forest framework.

# In[1204]:


from sklearn.ensemble import RandomForestClassifier


# In[1205]:


rfc = RandomForestClassifier(n_estimators=500)


# In[1206]:


rfc.fit(X_train, y_train)


# In[1207]:


predictions = rfc.predict(X_test)


# **Evaluating the model**

# Confusion Metric Report

# In[1208]:


from sklearn.metrics import classification_report, confusion_matrix


# In[1209]:


print(confusion_matrix(y_test, predictions))


# Classification Metric Report

# In[1210]:


print(classification_report(y_test, predictions))


# **Predicting the condition of a new patient**

# Let's assume a new patient with the following characteristics, will he/she be diagnozied with a heart deisease?

# In[1211]:


new_patient = df.drop('HeartDisease',axis=1).iloc[915]
new_patient


# In[1212]:


new_patient = new_patient.values.reshape(-1, 15)


# In[1213]:


new_patient = scaler.transform(new_patient)


# In[1216]:


rfc.predict(new_patient).round()


# In[1217]:


df.iloc[915]['HeartDisease']


# Yes, the model predicts that the new patient will likely be diagnozed with a heart disease with the above characteristics. 
