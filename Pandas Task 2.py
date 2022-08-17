#!/usr/bin/env python
# coding: utf-8

# #task 2 - Superstore_USA 1 . load this data in sql and in pandas with a relation in sql 2 . while loading this data you dont have to create a table manually you can use any automated approach to create a table and load a data in bulk in table 3 . Find out how return that we ahve recived and with a product id 4 . try to join order and return data both in sql and pandas 5 . Try to find out how many unique customer that we have 6 . try to find out in how many regions we are selling a product and who is a manager for a respective region 7 . find out how many different differnet shipement mode that we have and what is a percentage usablity of all the shipment mode with respect to dataset 8 . Create a new coulmn and try to find our a diffrence between order date and shipment date 9 . base on question number 8 find out for which order id we have shipment duration more than 10 days 10 . Try to find out a list of a returned order which sihpment duration was more then 15 days and find out that region manager as well 11 . Gorup by region and find out which region is more profitable 12 . Try to find out overalll in which country we are giving more didscount 13 . Give me a list of unique postal code 14 . which customer segement is more profitalble find it out . 15 . try to find out the 10th most loss making product catagory . 16 . Try to find out 10 top product with highest margins

# In[1]:


import pandas as pd
import os
import mysql.connector as conn
import csv


# In[2]:


import numpy as np


# In[3]:


import logging
import sys


# In[4]:


logging.basicConfig(filename="pandastask2.log", level=logging.INFO, format='%(levelname)s %(asctime)s %(name)s %('
                                                                          'message)s')
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                              '%m-%d-%Y %H:%M:%S')
logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


# In[5]:


orders_df=pd.read_excel('Superstore_USA.xlsx',sheet_name=0)
orders_df.head()


# In[6]:


returns_df=pd.read_excel('Superstore_USA.xlsx',sheet_name=1)
returns_df.head()


# In[7]:


users_df=pd.read_excel('Superstore_USA.xlsx',sheet_name=2)
users_df.head()


# In[41]:


orders_df.to_csv('orders.csv')


# In[42]:


returns_df.to_csv('returns.csv')


# In[43]:


users_df.to_csv('users.csv')


# In[11]:


from sqlalchemy import create_engine


# In[12]:


#Creating conncetion 
def createConnection():
    try:
        mydb = conn.connect(host='localhost', user='Ashwini', passwd='suhan10',
                            use_pure=True, auth_plugin='mysql_native_password')
        cursor = mydb.cursor()
        print('Connected Successfully')
    except Exception as e:
        print('Error occurred connecting to database', e)
    return mydb
createConnection()


# In[46]:


df1=pd.read_csv('orders.csv')
df1.head()


# In[47]:


orders=pd.DataFrame(df1)


# In[49]:


#Creating table and loading data to sql workbench using sqlchemy
my_conn = create_engine("mysql+mysqldb://Ashwini:suhan10@localhost/sameer25")
df1=pd.DataFrame(df1)
df1.to_sql(con=my_conn,name='ordersTable',if_exists='append')


# In[50]:


df2=pd.read_csv('users.csv')
df2.head()


# In[51]:


users=pd.DataFrame(df2)
users


# In[23]:


my_conn = create_engine("mysql+mysqldb://Ashwini:suhan10@localhost/sameer25")
users.to_sql(con=my_conn,name='usersTable',if_exists='append')


# In[32]:


import csv
# enter your password and database names here

retuns = pd.read_csv("returns.csv",sep=',',quotechar='\'',encoding='utf8',index_col=None) # Replace Excel_file_name with your excel sheet name
retuns.to_sql('retunsTable',con=my_conn,index=False,if_exists='append') # Replace Table_name with your sql table name


# In[52]:


#4 . try to join order and return data both in sql and pandas 
order_return_df=pd.merge(orders,retuns)
order_return_df


# In[53]:


#Try to find out how many unique customer that we have
len(orders['Customer ID'].unique()) # we have 2703 unique customers 


# In[54]:


#try to find out in how many regions we are selling a product and who is a manager for a respective region 
order_region_df=pd.merge(orders,users,on='Region',how='left')
order_region_df[['Order ID','Region','Manager']].groupby(['Region','Manager'])['Order ID'].count()


# In[55]:


import matplotlib.pyplot as plt 


# In[56]:


y = orders['Ship Mode'].value_counts().values
y
mylabels =orders['Ship Mode'].value_counts().index
mylabels
plt.pie(y,labels=mylabels,autopct='%1.1f%%')


# In[57]:


import  cufflinks as cf


# In[58]:


cf.go_offline()


# In[76]:


# find out how many different differnet shipement mode that we have 
#and what is a percentage usablity of all the shipment mode with respect to dataset
orders['Ship Mode'].value_counts()


# In[59]:


orders['Ship Mode'].value_counts().iplot(kind='bar')


# In[60]:


#Create a new coulmn and try to find our a diffrence between order date and shipment date 
orders.dtypes


# In[61]:


orders['Order Date']=pd.to_datetime(orders['Order Date'])
orders.dtypes


# In[62]:


orders['Ship Date']=pd.to_datetime(orders['Ship Date'])


# In[63]:


orders.tail()


# In[64]:


#Create a new coulmn and try to find our a diffrence between order date and shipment date 
orders['DeliveryDays']=orders['Ship Date']-orders['Order Date']
orders.head()


# In[65]:


orders.columns


# In[66]:


#base on question number 8 find out for which order id we have shipment duration more than 10 days
len(orders[orders['DeliveryDays'] > '10 Days']) # total 2897 ordeers are there having more than 10 days of shipping duraion


# In[67]:


#ry to find out a list of a returned order which
#sihpment duration was more then 15 days and find out that region manager as well
order_15_days=orders[orders['DeliveryDays']>'15 Days']
order_15_days


# In[173]:


#finding list of returned order having 15 days of shipment duraion
days_15_return=pd.merge(order_15_days,retuns,on='Order ID')
len(days_15_return) #total such 26 orders arethere having 15 days of shipemnt 
#finding manager with appropriate region
pd.merge(days_15_return,users,on='Region')[['Order ID','Region','Manager']]


# In[68]:


#Gorup by region and find out which region is more profitable 
orders.groupby('Region')['Profit'].mean() #Central region is more profitable


# In[106]:


#Try to find out overalll in which country we are giving more didscount
orders['Discount'].describe() # from this we get max discount is 0.25
orders[orders['Discount']<=0.25]


# In[69]:


#Give me a list of unique postal code 
orders['Postal Code'].nunique()


# In[183]:


#which customer segement is more profitalble find it out
orders.groupby('Customer Segment')['Profit'].mean() #consumer segment is more profitable


# In[70]:


#try to find out the 10th most loss making product catagory .
orders.groupby('Product Category')['Profit'].mean().sort_values(ascending=False)


# In[72]:


def profit(a):
    if a> 0:
        return 1
    else :
        return 0


# In[74]:


orders['ProfitStatus']=orders['Profit'].apply(profit)
orders.head()


# In[86]:


#try to find out the 10th most loss making product catagory .
orders.groupby('Product Sub-Category')['Profit'].mean().sort_values(ascending=True) #category is Office Furnishing 


# In[131]:


#Try to find out 10 top product with highest margins
top_10_product_on_margin=orders[orders['Product Base Margin']>=0.85]
top_10_product_on_margin[['Product Name','Product Base Margin']].head(10)


# In[ ]:




