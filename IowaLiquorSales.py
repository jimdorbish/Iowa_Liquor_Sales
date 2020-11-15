#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
plt.rcParams['figure.figsize'] = 10, 10

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('/Users/Jordan/GitHub/DataResources/Iowa_Liquor_Sales_reduced.csv')
df.head()


# In[3]:


# remove unnecessary columns

del df["Category Name"]
del df["Item Description"]
del df["County"]

# remove $ from columns
cols = ["State Bottle Cost", "State Bottle Retail", "Sale (Dollars)"]
for col in cols:
    df[col]= df[col].apply(lambda x:float(x[1:]))
    
# covert dates to date-time format
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

# drop null values
df = df.dropna()

# convert datatypes to integers
df["County Number"] = df["County Number"].astype(int)
df["Category"] = df["Category"].astype(int)

df.head()


# In[4]:


# filter the data 




# find the first and last sales date to determine which stores were open all of 2015

dates = df.groupby(by=['Store Number'], as_index=False)
dates = dates.agg({"Date":[np.min, np.max]})
dates.columns = [''.join(col).strip() for col in dates.columns.values]
dates

# find the stores that have opened or closed throughout the year 
lower_cutoff = pd.Timestamp("20150301")
upper_cutoff = pd.Timestamp("20151001")
mask = (dates['Dateamin'] < lower_cutoff) & (dates['Dateamax'] > upper_cutoff)
good_stores = dates[mask]["Store Number"]
df = df[df['Store Number'].isin(good_stores)]

df.head()
dates


# In[5]:



# compute intermediate data to predict sales and/or profits

# margin and price per liter 
df['Margin'] = (df["State Bottle Retail"] - df['State Bottle Cost']) * df["Bottles Sold"]
df['Price per Liter'] = df['Sale (Dollars)'] / df['Volume Sold (Liters)']
df.head()


# In[52]:


# sales per store in 2015

# filter by start and end dates
df.sort_values(by=["Store Number", "Date"], inplace=True)
start_date = pd.Timestamp("20150101")
end_date = pd.Timestamp("20151231")
mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
sales = df[mask]

# group by store name
sales = sales.groupby(by=["Store Number"], as_index=False)

# compute sums and means
sales = sales.agg({"Sale (Dollars)": [np.sum, np.mean],
                   "Volume Sold (Liters)": [np.sum, np.mean],
                   "Margin": np.mean,
                   "Price per Liter": np.mean,
                   "Zip Code": lambda x: x.iloc[0],
                   "City": lambda x: x.iloc[0],
                   "County Number": lambda x: x.iloc[0]})

# rename columns
sales.columns = [' '.join(col).strip() for col in sales.columns.values]
sales.columns = [u'Store Number', u'City', u'2015 Sales',
       u'2015 Sales mean', u'County Number',
       u'Price per Liter mean', u'Zip Code',
       u'2015 Volume Sold (Liters)', u'2015 Volume Sold (Liters) mean',
       u'2015 Margin mean']

sales.head()

q1_2015_end = pd.Timestamp("20150331")
q1_mask = (df['Date'] >= start_date) & (df['Date'] <= q1_2015_end)
salesq1=df[q1_mask]
salesq1 = salesq1.groupby(by=["Store Number"], as_index = False)
salesq1 = salesq1.agg({"Sale (Dollars)": [np.sum, np.mean],
                   "Volume Sold (Liters)": [np.sum, np.mean],
                   "Margin": np.mean,
                   "Price per Liter": np.mean,
                   "Zip Code": lambda x: x.iloc[0],
                   "City": lambda x: x.iloc[0],
                   "County Number": lambda x: x.iloc[0]})


salesq1.columns = [' '.join(col).strip() for col in salesq1.columns.values]
salesq1.columns = [u'Store Number', u'City', u'2015 q1 Sales',
       u'2015 q1 Sales mean', u'County Number',
       u'Price per Liter mean', u'Zip Code',
       u'2015 q1 Volume Sold (Liters)', u'2015 q1 Volume Sold (Liters) mean',
       u'2015 q1 Margin mean']

sales.head()



q1_16_start_date = pd.Timestamp("20160101")
q1_16_end_date = pd.Timestamp("20160331")
q1_16_mask = (df["Date"]>= q1_16_start_date) & (df["Date"] <= q1_16_end_date)

q1_16_sales = df[q1_16_mask]

q1_16_sales = q1_16_sales.groupby(by=["Store Number"], as_index=False)
q1_16_sales= q1_16_sales.agg({"Sale (Dollars)": [np.sum, np.mean],
                   "Volume Sold (Liters)": [np.sum, np.mean],
                   "Margin": np.mean,
                   "Price per Liter": np.mean,
                   "Zip Code": lambda x: x.iloc[0],
                   "City": lambda x: x.iloc[0],
                   "County Number": lambda x: x.iloc[0]})
q1_16_sales
q1_16_sales.columns = [' '.join(col).strip() for col in q1_16_sales.columns.values]
q1_16_sales.columns = [u'Store Number', u'City', u'2016 q1 Sales',
       u'2016 q1 Sales mean', u'County Number',
       u'Price per Liter mean', u'Zip Code',
       u'2016 q1 Volume Sold (Liters)', u'2016 q1 Volume Sold (Liters) mean',
       u'2016 q1 Margin mean']


# In[56]:


b = sales["2015 Volume Sold (Liters)"]
a = salesq1["2015 q1 Volume Sold (Liters)"]
df3  = pd.merge(sales,salesq1, how = 'left', on = "Store Number", copy = False )
df3 = df3.sort_values(by ="Store Number")

del df3['City_y']
del df3['Zip Code_y']
del df3['Price per Liter mean_y']
del df3['County Number_y']

def count_missing(frame):
    return (frame.shape[0] * frame.shape[1]) - frame.count().sum()
df3.groupby('Store Number').apply(count_missing)



df4 = pd.merge(df3,q1_16_sales, how= 'left', on= "Store Number", copy=False)
df4 = df4.sort_values(by="Store Number")
df4.head()


# In[41]:


# exploratory analysis

# 2015 sales q1 v. margin mean
df3.plot.scatter(x='2015 Margin mean', y='2015 q1 Sales')
plt.title('2015 Sales Q1 vs. Margin Mean')


# 2015 sales q1 v. volume sold mean 
df3.plot.scatter(x='2015 Volume Sold (Liters) mean', y='2015 q1 Sales')
plt.title('2015 Sales Q1 vs. Volume Sold Mean')
plt.show()



# In[42]:


# fit a linear model

lm = linear_model.LinearRegression()
X = df3[['2015 q1 Sales']]
y = df3['2015 Sales']
lm.fit(X, y)
predictions = lm.predict(X)
print("Model fit:", lm.score(X, y))
print(lm.coef_[0], lm.intercept_)

plt.scatter(X, y)
plt.plot(X, predictions)
plt.xlabel("Sales q1 2015")
plt.ylabel("Sales 2015")
plt.xlim(0, 100000)
plt.ylim(0, 400000)
plt.show()


# In[64]:



df4['2016 q1 Sales'].fillna((df4['2016 q1 Sales'].mean()), inplace=True)


# In[65]:


# predict 2016 sales

# fit linear model

X = df4[['2016 q1 Sales']]
predictions = lm.predict(X)
total_2016 = sum(predictions)
total_2015 = sum(df4['2015 Sales'])
X2 = df4[["2015 q1 Sales"]]
pred_2015 = sum(lm.predict(X2))

print("2015 predicted", pred_2015)
print("2015 actual", total_2015)
print("2016 predicted", total_2016)


# In[71]:



# fit linear model

lm = linear_model.LinearRegression()
X = df4[["2015 q1 Sales"]]
print(len(X))
y = df4["2015 Sales"]
lm.fit(X, y)
predictions = lm.predict(X)
print("Model fit:", lm.score(X, y))


plt.scatter(X, y)
plt.plot(X, predictions)
plt.xlabel("Sales 2015 Q1")
plt.ylabel("Sales 2015")
plt.xlim(0, 50000)
plt.ylim(0, 200000)
plt.show()


# In[73]:


X = df4[["2016 q1 Sales"]]
predictions = lm.predict(X)
total_2016 = sum(predictions)
total_2015 = sum(df4["2015 Sales"])
X2 = df4[["2015 q1 Sales"]]
pred_2015 = sum(lm.predict(X2))

print("2015 predicted", pred_2015)
print("2015 actual", total_2015)
print("2016 predicted", total_2016)


# In[75]:


# regularization with elastic net 

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.cross_validation import cross_val_score
from sklearn import linear_model
from sklearn.metrics import r2_score

l1_ratios = np.linspace(0.01, 1.0, 5)
lm = linear_model.ElasticNetCV(l1_ratio=l1_ratios, n_alphas=10, cv=5,
                            verbose=1)
X = df4[["2015 q1 Sales"]]
print(len(X))
y = df4["2015 Sales"]
lm.fit(X, y)
predictions = lm.predict(X)
print("Model fit:", lm.score(X, y))

plt.scatter(X, y)
plt.plot(X, predictions)
plt.xlabel("Sales 2015 Q1")
plt.ylabel("Sales 2015")
plt.title("Sales 2015 vs. Sales 2015 Q1")
plt.xlim(0, 50000)
plt.ylim(0, 200000)
plt.show()


# In[77]:


X = df4[["2016 q1 Sales"]]
predictions = lm.predict(X)
total_2016 = sum(predictions)
total_2015 = sum(df4["2015 Sales"])
X2 = df4[["2015 q1 Sales"]]
pred_2015 = sum(lm.predict(X2))

print("2015 predicted", pred_2015)
print("2015 actual", total_2015)
print("2016 predicted", total_2016)


# In[81]:


df4["15-16 q1 Sales Growth"] = df4["2016 q1 Sales"] - df4["2015 q1 Sales"]
df4['2016 Predicted Sales']= predictions
df4.head()


# In[82]:


df4.to_csv('/Users/Jordan/GitHub/DataResources/Iowa_Liquor_Sales_combined_df4.csv')


# In[ ]:




