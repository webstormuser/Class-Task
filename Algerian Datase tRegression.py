#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import plotly 
import plotly.graph_objects as go
import plotly.express as pe 
import plotly.offline as po
from plotly.offline import init_notebook_mode
import cufflinks as cf


# In[3]:


from plotly.offline import iplot


# In[4]:


cf.go_offline()


# In[5]:


#Loading the dataset 
fire_df=pd.read_csv('E:\Python class\Datasets\EDA Datasets\Algerian_forest_fires_dataset_UPDATE.csv',header=1)


# Attribute Information:
# 
# 1. Date : (DD/MM/YYYY) Day, month ('june' to 'september'), year (2012)
# Weather data observations
# 2. Temp : temperature noon (temperature max) in Celsius degrees: 22 to 42
# 3. RH : Relative Humidity in %: 21 to 90
# 4. Ws :Wind speed in km/h: 6 to 29
# 5. Rain: total day in mm: 0 to 16.8
# FWI Components
# 6. Fine Fuel Moisture Code (FFMC) index from the FWI system: 28.6 to 92.5
# 7. Duff Moisture Code (DMC) index from the FWI system: 1.1 to 65.9
# 8. Drought Code (DC) index from the FWI system: 7 to 220.4
# 9. Initial Spread Index (ISI) index from the FWI system: 0 to 18.5
# 10. Buildup Index (BUI) index from the FWI system: 1.1 to 68
# 11. Fire Weather Index (FWI) Index: 0 to 31.1
# 12. Classes: two classes, namely Fire and not Fire

# In[6]:


fire_df.head()


# In[7]:


fire_df1=fire_df.iloc[1:122]


# In[8]:


fire_df2=fire_df.iloc[123:]


# In[9]:


#First dataset is for Bejara region add one more column Region 
fire_df1.head()


# In[10]:


for i in range(1,len(fire_df1)):
    fire_df1['Region']='Bejara'


# In[11]:


fire_df1


# In[12]:


#First dataset is for Sidi Bel Abes  region add one more column Region  before adding column drop the row having first row as a header 
fire_df2.head() 


# In[13]:


fire_df2.drop(index=123,axis=0,inplace=True)


# In[14]:


for i in range(1,len(fire_df2)):
    fire_df2['Region']='Sidi Bel Abes'


# In[15]:


fire_df1


# In[16]:


#Now we weill combine both  the divided df info one 
df=fire_df1.append(fire_df2)


# In[17]:


df


# In[18]:


df.tail()


# EDA AND FEATURE ENGINEERING 

# In[19]:


df['RH']=df[' RH']#Removing extra spce from the feature RH


# In[20]:


df['Ws']=df[' Ws']#Removing extra spce from the feature Ws


# In[21]:


df['Rain']=df['Rain ']##Removing extra spce from the feature Rain


# In[22]:


df['Classes']=df['Classes  ']##Removing extra spce from the feature Classes


# In[23]:


df.drop(columns=[' RH',' Ws','Rain '],axis=1,inplace=True)


# In[24]:


df.drop('Classes  ',axis=1,inplace=True)


# In[25]:


df.head()


# In[26]:


df.columns


# In[27]:


df.info()


# In[28]:


#Considering Temperature as a target variable and rest of independent variable 


# In[29]:


df['year']=df['year'].astype('int')


# In[30]:


df['month']=df['month'].astype(int)


# In[31]:


df['day']=df['day'].astype(int)


# In[32]:


df.info()


# In[33]:


df['Temperature'].unique()


# In[34]:


#Convert the Temperature into numeric type
df['Temperature']=df['Temperature'].astype(int)


# In[35]:


df['FFMC'].unique()


# In[36]:


df['FFMC']=df['FFMC'].astype(float)


# In[37]:


df['DMC'].unique()


# In[38]:


df['DMC']=df['DMC'].astype(float)


# In[39]:


df['DC'].unique()


# In[40]:


df['DC']=df['DC'].replace('14.6 9','14.6')


# In[41]:


df['DC']=df['DC'].astype(float)


# In[42]:


df['ISI'].unique()


# In[43]:


#Converting Initial Spread Index (ISI) index to numeric datatype
df['ISI']=df['ISI'].astype(float)


# In[44]:


#Converting Buildup Index (BUI) index to numeric datatype 
df['BUI']=df['BUI'].astype(float)


# In[45]:


#Converting Fire Weather Index (FWI) Index: 0 to 31.1 to numeric datatype 
df['FWI'].unique()


# In[46]:


df[df['FWI']=='fire   ']


# In[47]:


df['FWI']=df['FWI'].replace('fire   ','0.0')


# In[48]:


df['FWI']=df['FWI'].astype(float)


# In[49]:


df['RH'].unique()


# In[50]:


df['RH']=df['RH'].astype(int)


# In[51]:


df['Ws'].unique()


# In[52]:


df['Ws']=df['Ws'].astype(int)


# In[53]:


df['Rain'].unique()


# In[54]:


df['Rain']=df['Rain'].astype(float)


# In[55]:


df.info()


# In[56]:


df['Region'].unique() #it contain categorical values replacing it by label encoding 


# In[57]:


from sklearn.preprocessing import LabelEncoder


# In[58]:


from sklearn.preprocessing import OneHotEncoder


# In[59]:


label_encoder=LabelEncoder()


# In[60]:


df['Region']=label_encoder.fit_transform(df['Region'])


# In[61]:


df['Classes'].unique()


# In[62]:


df['Classes']=df['Classes'].replace('not fire   ','not fire')


# In[63]:


df['Classes']=df['Classes'].replace('fire   ','fire')


# In[64]:


df['Classes']=df['Classes'].replace('not fire ','not fire')


# In[65]:


df['Classes']=df['Classes'].replace('not fire    ','not fire')


# In[66]:


df['Classes']=df['Classes'].replace('not fire     ','not fire')


# In[67]:


df['Classes']=df['Classes'].replace('fire ','fire')


# In[68]:


df['Classes'].unique()


# In[69]:


df.isna().sum()


# In[72]:


df.dropna(inplace=True)


# In[73]:


df_copy=df.copy()


# In[74]:





# In[80]:


df_copy['Classes'].unique()


# In[86]:


df_copy['Classes']=label_encoder.fit_transform(df_copy['Classes'])


# In[87]:


df_copy.head()


# In[88]:


numeric_features=[ele for ele in df_copy.columns if df_copy[ele].dtype!='O']


# In[89]:


print(numeric_features)


# In[90]:


plt.figure(figsize=(20, 20))
plt.suptitle('Outlier Analysis of Numerical Features', 
             fontsize=20, fontweight='bold', alpha=0.8, y=1.)

for i in range(0, len(numeric_features)):
    plt.subplot(5, 3, i+1)
    sns.boxplot(x=df_copy[numeric_features[i]],color='b')
    plt.xlabel(numeric_features[i])
    plt.tight_layout()


# Beofore applying any outlier treatment or scaling the data split the dataset into train test

# In[91]:


df_copy.head()


# out target variable is Temperature 

# In[92]:


X=df_copy.drop('Temperature',axis=1)


# In[93]:


X.head()


# In[94]:


y=df_copy['Temperature']


# In[95]:


y.head()


# In[96]:


from sklearn.model_selection import train_test_split


# In[97]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)


# In[98]:


X_train.shape


# In[99]:


X_test.shape


# Standerdize the dataset 

# In[100]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[101]:


X_train=scaler.fit_transform(X_train)


# In[102]:


X_train


# In[103]:


X_test=scaler.transform(X_test)


# In[104]:


X_test


# Model Trainig 

# In[105]:


from sklearn.linear_model import  LinearRegression


# In[106]:


lin_reg=LinearRegression()


# In[107]:


lin_reg


# In[108]:


lin_reg.fit(X_train,y_train)


# In[109]:


#print coefficint and intercepts
print(lin_reg.coef_)


# In[110]:


#print coefficint and intercepts
print(lin_reg.intercept_)


# In[111]:


#prediction for test data 
lin_pred=lin_reg.predict(X_test)


# In[112]:


#Assumption of linear Regression 
plt.scatter(y_test,lin_pred)
plt.xlabel('test truth data')
plt.ylabel('test predicted data')


# In[113]:


#Calculate residuals 
residuals=y_test-lin_pred


# In[114]:


residuals


# In[115]:


#2nd Assuption see the distribution for residuals it should be normal
sns.displot(residuals,kind='kde')


# In[116]:


#We can clearly see because of some outliers in the dataset our residuals distribution is some what skewed to left side 


# In[117]:


#3rd Assumption see the distribution between prediction and residuals it should be uniform
plt.scatter(lin_pred,residuals)


# Performance matrix

# In[118]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# In[119]:


print('MSE--->',mean_squared_error(y_test,lin_pred))
print('MAE--->',mean_absolute_error(y_test,lin_pred))
print('RMSE--->',np.sqrt(mean_squared_error(y_test,lin_pred)))


# Calculate R2 and adjusted R2

# In[120]:


from sklearn.metrics import r2_score


# In[121]:


score=r2_score(y_test,lin_pred)
print('R2 score--->',score*100)
adjusted_r_score=1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print('Adjusted R score--->',adjusted_r_score*100)


# # USING Ridge Regreesion

# In[122]:


from sklearn.linear_model import Ridge


# In[123]:


ridge=Ridge(alpha=1.0,fit_intercept=True, normalize='deprecated', 
            copy_X=True, max_iter=None, tol=0.001, solver='auto', positive=False, random_state=None)


# In[124]:


ridge.fit(X_train,y_train)


# In[125]:


ridge_pred=ridge.predict(X_test)


# In[126]:


#print coefficint and intercepts
print(ridge.coef_)


# In[127]:


#print coefficint and intercepts
print(ridge.intercept_)


# In[128]:


#Assumption of linear Regression 
plt.scatter(y_test,ridge_pred)
plt.xlabel('test truth data')
plt.ylabel('test predicted data')


# In[129]:


#Calculate residuals 
residuals=y_test-ridge_pred


# In[130]:


residuals


# In[131]:


#2nd Assuption see the distribution for residuals it should be normal
sns.displot(residuals,kind='kde')


# In[132]:


#3rd Assumption see the distribution between prediction and residuals it should be uniform
plt.scatter(ridge_pred,residuals)


# Performance matrix

# In[133]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# In[134]:


print('MSE--->',mean_squared_error(y_test,ridge_pred))
print('MAE--->',mean_absolute_error(y_test,ridge_pred))
print('RMSE--->',np.sqrt(mean_squared_error(y_test,ridge_pred)))


# Calculate R2 and adjusted R2

# In[135]:


from sklearn.metrics import r2_score


# In[136]:


score=r2_score(y_test,ridge_pred)
print('R2 score--->',score*100)
adjusted_r_score=1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print('Adjusted R score--->',adjusted_r_score*100)


# # USING LASSO REGRESSION 

# In[137]:


from sklearn.linear_model import Lasso


# In[138]:


lasso=Lasso(alpha=1.0,fit_intercept=True, normalize='deprecated', precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')


# In[139]:


lasso.fit(X_train,y_train)


# In[140]:


#print coefficint and intercepts
print(lasso.coef_)


# In[141]:


print(lasso.intercept_)


# In[142]:


lasso_pred=lasso.predict(X_test)
lasso_pred


# In[143]:


#Assumption of linear Regression 
plt.scatter(y_test,lasso_pred)
plt.xlabel('test truth data')
plt.ylabel('test predicted data')


# In[144]:


#Calculate residuals 
residuals=y_test-lasso_pred
residuals 


# In[145]:


#2nd Assuption see the distribution for residuals it should be normal
sns.displot(residuals,kind='kde')


# In[146]:


#3rd Assumption see the distribution between prediction and residuals it should be uniform
plt.scatter(lasso_pred,residuals)


# PERFORMANCE MATRIX

# In[147]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# In[148]:


print('MSE--->',mean_squared_error(y_test,lasso_pred))
print('MAE--->',mean_absolute_error(y_test,lasso_pred))
print('RMSE--->',np.sqrt(mean_squared_error(y_test,lasso_pred)))


# Calculate R2 and adjusted R2

# In[149]:


from sklearn.metrics import r2_score


# In[150]:


score=r2_score(y_test,lasso_pred)
print('R2 score--->',score*100)
adjusted_r_score=1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print('Adjusted R score--->',adjusted_r_score*100)


# In[ ]:




