#!/usr/bin/env python
# coding: utf-8

# ## Liver Disease Prediction 

# ### Content 
This data set contains 416 liver patient records and 167 non liver patient records collected from North East of Andhra pradesh,india. The 'Dataset' column is a class label used to divide groups into the liver patient (liver disease) or not (no disease). This data set contains 441 male patient records 
and 142 female patient records. 

Any patient whose age exceeded 89 is listed as being og age '90'.


columns :

• Age of the patients 
• Gender of the patient 
• Total Bilirubin
• Direct Bilirubin 
• Alkaline Phosphotase
• Alamine Aminotransferase
• Aspartate Aminotransferase
• Total proteins 
• Albumin
• Albumin and Globulin ratio 
• Dataset : filed used to split the data into two sets (patient with liver disease or no disease) 

# In[1]:


# For numerical computing 
import numpy as np 

# for dataframe 
import pandas as pd 

# for easier visualization 
import seaborn as sns 

# for visualization and display plots 
import matplotlib.pyplot as plt 

# Ignore warnings 
import warnings
warnings.filterwarnings('ignore')

import math

# to split train and test set 
from sklearn.model_selection import train_test_split 

# to perform hyperparameter tunning 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV 

from sklearn.model_selection import cross_val_score 

# machine learning Models 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score,confusion_matrix

from sklearn.preprocessing import StandardScaler 


# In[2]:


# Importing the datasets 
df = pd.read_csv('indian_liver_patient.csv')
df 


# In[3]:


# First 5 records 
df.head() 


# In[4]:


# Shape of the dataset 
df.shape 


# In[5]:


df.columns 


# In[6]:


df.describe() 


# ### Exploratory Data Analysis 

# #### Filtering categorical data 

# In[7]:


df.dtypes[df.dtypes=='object']


# #### Distribution of Numerical features 

# In[8]:


# plot histogram grid 
df.hist(figsize=(15,15))
plt.show() 


# In[9]:


df.describe() 

It seems there is outlier in Aspartate_Aminotransferase as the max value is very high than mean value 
Dataset i.e output value has '1' for liver disease and '2' for no liver disease so lets make it 0 for no disease to make it convinient. 
# In[10]:


## if score score==negative, mark 0 ; else 1 
def partition(x):
    if x==2:
        return 0 
    return 1
df['Dataset'] = df['Dataset'].map(partition) 


# ### Distribution of Categorical data  

# In[11]:


df.describe(include=['object'])


# ### Bar plots for categorical Features : 

# In[12]:


plt.figure(figsize=(5,5))
sns.countplot(y='Gender',data=df)


# In[13]:


df[df['Gender'] == 'Male'][['Dataset','Gender']].head() 


# In[14]:


sns.countplot(data=df,x='Gender',label='Count')

M,F = df['Gender'].value_counts() 
print('Number of patients that are male:',M)
print('Number of patients that are female:',F) 

• There are more male patients than female patients . 

Label male as 0 and female as 1 .
# In[15]:


## if score==negative , mark 0 ; else 1 

def partition(x):
    if x=='Male':
        return 0 
    return 1 

df['Gender'] = df['Gender'].map(partition) 


# ### 2-D Scatter plot :

# In[16]:


sns.scatterplot(x = df['Total_Bilirubin'],y = df['Direct_Bilirubin'],hue=df['Dataset'])


# In[17]:


sns.scatterplot(x=df['Total_Bilirubin'],y=df['Albumin'],hue=df['Dataset'])


# In[18]:


sns.scatterplot(x=df['Total_Protiens'],y=df['Albumin_and_Globulin_Ratio'],hue=df['Dataset'])


# ### Duplicated values 

# In[19]:


df.duplicated().sum() 

There was 13 duplicated values.
# In[20]:


df = df.drop_duplicates()
print(df.shape) 


# ### Missing values imputation 

# In[21]:


df.isna().sum() 


# In[22]:


# White line indicates the missing values 
sns.heatmap(df.isna()) 


# In[23]:


# % of missing values :- 
for i in df.isna().sum():
    print((i/len(df)*100))


# In[24]:


df.dropna(inplace=True) 


# In[25]:


df.isna().sum() 


# ### Outlier Detection 

# In[26]:


# 1) Histogram 
# 2) Boxplot 
# 3) Descriptive statistics 


# In[27]:


df.describe() 


# In[28]:


df.hist()
plt.tight_layout()


# In[29]:


plt.figure(figsize=(25,30))
sns.boxplot(df)


# #### Outlier Treatment 

# In[30]:


# Outlier detection to calculate the upper_extreme and lower_extreme 
def outlier_detection(data,colname):
    q1=data[colname].quantile(0.25)
    q3=data[colname].quantile(0.75)
    iqr=q3-q1
    
    upper_extreme = q3+(1.5*iqr)
    lower_extreme = q1-(1.5*iqr)
    return lower_extreme, upper_extreme


# In[31]:


outlier_detection(df,'Total_Bilirubin')


# In[32]:


df[df['Total_Bilirubin']>5.30]


# In[33]:


df.loc[df['Total_Bilirubin']>5.30,'Total_Bilirubin']=5.30


# In[34]:


df[df['Total_Bilirubin']==5.30]


# In[35]:


outlier_detection(df,'Direct_Bilirubin')


# In[36]:


df[df['Direct_Bilirubin']>2.95]


# In[37]:


df.loc[df['Direct_Bilirubin']>2.95,'Direct_Bilirubin']=2.95


# In[38]:


df[df['Direct_Bilirubin']==2.95]


# In[39]:


outlier_detection(df,'Alkaline_Phosphotase')


# In[40]:


df[df['Alkaline_Phosphotase']>481.0]


# In[41]:


df.loc[df['Alkaline_Phosphotase']>481.0,'Alkaline_Phosphotase']=481.0
df[df['Alkaline_Phosphotase']==481.0]


# In[42]:


outlier_detection(df,'Alamine_Aminotransferase')


# In[43]:


df[df['Alamine_Aminotransferase']>117.37]
df.loc[df['Alamine_Aminotransferase']>117.37,'Alamine_Aminotransferase']=117.35
df[df['Alamine_Aminotransferase']==117.37]


# In[44]:


outlier_detection(df,'Aspartate_Aminotransferase')


# In[45]:


df[df['Aspartate_Aminotransferase']>180.0]
df.loc[df['Aspartate_Aminotransferase']>180.0,'Aspartate_Aminotransferase']=180.0
df[df['Aspartate_Aminotransferase']==180.0]


# In[46]:


outlier_detection(df,'Total_Protiens')


# In[47]:


df[df['Total_Protiens']>9.3]
df.loc[df['Total_Protiens']>9.3,'Total_Protiens']=9.3
df[df['Total_Protiens']==9.3]


# In[48]:


df[df['Total_Protiens']<3.69]
df.loc[df['Total_Protiens']<3.69,'Total_Protiens']=3.69
df[df['Total_Protiens']==3.69]


# In[49]:


outlier_detection(df,'Albumin_and_Globulin_Ratio')


# In[50]:


df[df['Albumin_and_Globulin_Ratio']>1.70]
df.loc[df['Albumin_and_Globulin_Ratio']>1.70,'Albumin_and_Globulin_Ratio']=1.70
df[df['Albumin_and_Globulin_Ratio']==1.70]


# In[51]:


plt.figure(figsize=(25,30))
sns.boxplot(df) 


# In[52]:


df.head() 


# In[53]:


# create the seperate object for target variable 
y = df.Dataset

# create the seperate object for input variable 
x = df.drop('Dataset',axis=1)


# In[54]:


# Train - Test- Split set 
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)


# ### Correlation 
• Finally, lets take look at the relationships between numeric features and other numeric features. 
• Correlation is the value between -1 and 1 that represent how closely values for two seperate features move in unison. 
• Positive correlation means that as one feature increases, other increases 
• Negative correlation means that one feature increases , other decreases 
• Correlation near -1 or 1 indicates strong relationship. 
• Those closer to 0 indicate the weak relationship 
• 0 indicates no relationship 
# In[55]:


df.corr() 


# In[56]:


sns.heatmap(df.corr(),cmap='viridis',annot=True) 


# ### Standardisation 
• In Data Standardisation we can perform zero mean centering and unit scaling. i.e we make the mean of all the features as zero and standard deviation as 1 
• Thus we have to use the mean and std of each feature. 
# In[57]:


from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler() 
x_train= scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 


# ## Machine Learning Model : 

# ### Model 1 : Logistic Regression 

# In[58]:


model  = LogisticRegression()
model.fit(x_train,y_train) 


# In[59]:


# predict train set results 
y_train_pred = model.predict(x_train) 


# In[60]:


# Predict test set results 
y_pred = model.predict(x_test) 


# In[61]:


# get just the prediction for positive class (1) 
y_pred_proba = model.predict_proba(x_test)[:,1]


# In[62]:


# Display first 10 prediction 
y_pred_proba[:10]


# In[63]:


# calculate the ROC  curve  from y_test , pred 

fpr , tpr , thresholds = roc_curve(y_test,y_pred_proba) 


# In[64]:


# Plot the ROC curve 

fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# plot curve 
plt.plot(fpr,tpr,label='l1')
plt.legend(loc='lower right')

# Diagonal as 45 degree line 
plt.plot([0,1],[0,1],'k--')

# Diagonal limits and labels 
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True positive Rate')
plt.xlabel('False positive Rate')
plt.show() 


# In[65]:


# calculate the AUC for train set 
print(roc_auc_score(y_train,y_train_pred))


# In[66]:


## Feature importance 
model = LogisticRegression(C=1, penalty='l2')
model.fit(x_train,y_train) 


# In[67]:


indices = np.argsort(-abs(model.coef_[0,:]))
print('The features in the order of importance are:')
print(50*'-')
for feature in x.columns[indices]:
    print(feature) 


# In[68]:


# find precision , recall 
from sklearn.metrics import precision_score,recall_score
print('Precision score:',precision_score(y_test,y_pred))
print('Recall Score:',recall_score(y_test,y_pred))
print('Confusion Matrx:',confusion_matrix(y_test,y_pred))


# In[69]:


print('Accuracy of Logistic Regression :',accuracy_score(y_test,y_pred))


# ### Model 2 : Random Forest : 

# In[70]:


tuned_params = {'n_estimators': [100, 200, 300, 400, 500], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
model = RandomizedSearchCV(RandomForestClassifier(), tuned_params, n_iter=15, scoring = 'roc_auc', n_jobs=-1)
model.fit(x_train, y_train)


# In[71]:


model.best_estimator_


# In[72]:


y_train_pred = model.predict(x_train) 


# In[73]:


y_pred = model.predict(x_test) 


# In[74]:


# get just prediction for positive class (1) 
y_pred_proba  = model.predict_proba(x_test)[:,1]


# In[75]:


# Display first 10 prediction 
y_pred_proba[:10]


# In[76]:


confusion_matrix(y_test,y_pred) 


# In[77]:


# calculate ROC curve from y_test and pred 
fpr ,tpr ,thresholsd  = roc_curve(y_test,y_pred_proba)


# In[78]:


# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[79]:


# calculate AUC for train set 
roc_auc_score(y_train,y_train_pred)


# In[80]:


# Feature Importance 
## Building the model again with the best hyperparameters
model = RandomForestClassifier(n_estimators=500, min_samples_split=2, min_samples_leaf=4)
model.fit(x_train, y_train)


# In[81]:


indices = np.argsort(-model.feature_importances_)
print("The features in order of importance are:")
print(50*'-')
for feature in x.columns[indices]:
    print(feature)


# In[82]:


print('Accuracy_score:',accuracy_score(y_test,y_pred))


# ### Model 3 KNN 

# In[83]:


# creating odd list of K for KNN
neighbors = list(range(1,20,2))
# empty list that will hold cv scores
cv_scores = []

#  10-fold cross validation , 9 datapoints will be considered for training and 1 for cross validation (turn by turn) to determine value of k
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())   

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('\nThe optimal number of neighbors is %d.' % optimal_k)


# In[84]:


MSE.index(min(MSE))


# In[85]:


# plot misclassification error vs k 
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()


# In[86]:


classifier = KNeighborsClassifier(n_neighbors = optimal_k)
classifier.fit(x_train,y_train) 


# In[87]:


y_pred = classifier.predict(x_test) 


# In[88]:


y_train_pred = classifier.predict(x_train) 


# In[89]:


y_pred_proba = classifier.predict_proba(x_test)[:,1]


# In[90]:


# Display first 10 prediction 
y_pred_proba[:10]


# In[91]:


# calculate the roc curve from y_test and pred 
fpr,tpr,thresholds = roc_curve(y_test,y_pred_proba) 


# In[92]:


#Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[93]:


# calculate AUC for train 
roc_auc_score(y_train,y_train_pred) 


# In[94]:


# calculate the accuracy score :- 
print('Accuracy_score :',accuracy_score(y_test,y_pred))


# ### Model 4 : Decision Trees 

# In[95]:


tuned_params = {'min_samples_split': [2, 3, 4, 5, 7], 'min_samples_leaf': [1, 2, 3, 4, 6], 'max_depth': [2, 3, 4, 5, 6, 7]}
model = RandomizedSearchCV(DecisionTreeClassifier(), tuned_params, n_iter=15, scoring = 'roc_auc', n_jobs=-1)
model.fit(x_train, y_train)


# In[96]:


model.best_estimator_ 


# In[97]:


y_train_pred = model.predict(x_train) 


# In[98]:


y_pred = model.predict(x_test) 


# In[99]:


y_pred_proba = model.predict_proba(x_test)[:,1]


# In[100]:


y_pred_proba[:10]


# In[101]:


confusion_matrix(y_test,y_pred) 


# In[102]:


fpr, tpr, thresholds = roc_curve(y_test,y_pred_proba)


# In[103]:


# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[104]:


# calculate the AUC for train : 
roc_auc_score(y_train,y_train_pred) 


# In[105]:


# Feature Importance 
## Building the model again with the best hyperparameters
model = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=6, max_depth=4)
model.fit(x_train, y_train)


# In[106]:


indices = np.argsort(-model.feature_importances_)
print("The features in order of importance are:")
print(50*'-')
for feature in x.columns[indices]:
    print(feature)


# In[107]:


print('Accuracy_score:',accuracy_score(y_test,y_pred))


# ### Model 5 SVC 

# In[108]:


from sklearn import svm
def svc_param_selection(x, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(x_train, y_train)
    grid_search.best_params_
    return grid_search.best_params_


# In[109]:


svClassifier=SVC(kernel='rbf',probability=True)
svClassifier.fit(x_train,y_train)


# In[110]:


svc_param_selection(x_train,y_train,5) 


# In[111]:


### Building the model again with the best hyperparameter 
model = SVC(C=1,gamma=1)
model.fit(x_train,y_train) 


# In[112]:


# predict the results 
y_train_pred = model.predict(x_train) 


# In[113]:


# predict train results 
y_train_pred = model.predict(x_train) 


# In[114]:


# predict the results 
y_pred = model.predict(x_test) 


# In[115]:


confusion_matrix(y_test,y_pred) 


# In[116]:


# calculate the ROC curve from y_test and pred 
fpr, tpr,threshold = roc_curve(y_test,y_pred_proba) 


# In[117]:


# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[118]:


# calculate the AUC  for train 
roc_auc_score(y_train,y_train_pred) 


# In[119]:


print('Accuracy_score:',accuracy_score(y_test,y_pred))


# In[120]:


import pickle 


# In[125]:


model = RandomForestClassifier(n_estimators=200,max_depth=10,random_state=42)
model.fit(x_train,y_train)

with open('liver_model.pkl','wb') as file:
    pickle.dump(model,file)


# In[ ]:




