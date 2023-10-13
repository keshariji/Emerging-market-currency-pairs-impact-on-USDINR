#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# **Analyzing emerging market currency pairs impact on USDINR Exchange Rates**
# 
# In today's interconnected world, emerging market currency pairs are gaining increasing prominence. These currency pairs have a significant impact on international trade and finance. This project focuses on the correlation analysis and Granger causality testing of emerging market currency pairs, with the primary objective of **understanding the relationships and potential influences on USDINR**.
# 
# The research offers insights into the Indian FX market and benefits to traders, hedgers, policymakers and businesses dealing in global currency. By analyzing these currency pairs, the aim is to shed light on how traders and businesses can make more informed and data-driven decisions.
# 
# This project comprises sections on data collection, correlation analysis, Granger causality testing, output observations and conclusions. The goal is to offer readers a deeper understanding of the relationships between emerging market currencies.

# **For this project, I have collected daily exchange rate data for all emerging market currency pairs, including the Dollar Index (DXY), which serves as the base currency. This dataset covers data from the year 2000 until today.**
# 
# *Data Source: Bloomberg*

# **EM Basket**
# 
# * **USDINR:** US Dollar against the Indian Rupee - India
# * **USDCOP:** US Dollar against the Colombian Peso - Colombia
# * **USDMXN:** US Dollar against the Mexican Peso - Mexico
# * **USDBRL:** US Dollar against the Brazilian Real - Brazil
# * **USDHUF:** US Dollar against the Hungarian Forint - Hungary
# * **USDPLN:** US Dollar against the Polish Zloty - Poland
# * **USDHKD:** US Dollar against the Hong Kong Dollar - Hong Kong
# * **USDPEN:** US Dollar against the Peruvian Nuevo Sol - Peru
# * **USDIDR:** US Dollar against the Indonesian Rupiah - Indonesia
# * **USDBGN:** US Dollar against the Bulgarian Lev - Bulgaria
# * **USDRON:** US Dollar against the Romanian Leu - Romania
# * **USDPHP:** US Dollar against the Philippine Peso - Philippines
# * **USDCZK:** US Dollar against the Czech Koruna - Czech Republic
# * **USDTWD:** US Dollar against the New Taiwan Dollar - Taiwan
# * **USDCNH:** US Dollar against the Chinese Yuan - China
# * **USDTHB:** US Dollar against the Thai Baht - Thailand
# * **USDKRW:** US Dollar against the South Korean Won - South Korea
# * **USDMYR:** US Dollar against the Malaysian Ringgit - Malaysia
# * **USDCLP:** US Dollar against the Chilean Peso - Chile
# * **USDZAR:** US Dollar against the South African Rand - South Africa
# * **USDRUB:** US Dollar against the Russian Ruble - Russia
# * **USDTRY:** US Dollar against the Turkish Lira - Turkey
# * **USDARS:** US Dollar against the Argentine Peso - Argentina
# * **DXY:** US Dollar Index

# In[1]:


## Import the required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests


# In[2]:


## Read the csv file
df=pd.read_csv('EM_basket_pairs.csv')
df


# In[3]:


## Head of the dataset
df.head()


# In[4]:


## Tail of the dataset
df.tail()


# In[5]:


## Convert dates to date time format and set as index for better analysis
df['Dates']=pd.to_datetime(df['Dates'], format='%d-%m-%y')
df.set_index('Dates', inplace=True)


# In[6]:


## Concise summary of Data Frame
df.info()


# **Observations:**
# 
# * Data frame contains 23 float-type and 1 integer-type columns

# In[7]:


## Check for null values in the dataset
df.isna().sum()


# **Observations:**
# 
# * Data frame has no null values, indicating complete data

# In[8]:


## Check for duplicate values
df.duplicated().sum()


# **Observations:**
# 
# * Data frame has no duplicate values, indicating complete data.

# In[9]:


## Dimensions of Data Frame
df.shape


# **Observations:**
# 
# * Data frame has dimensions of 6204 rows and 24 columns, denoted as (6204, 24)

# **Dataset is ready for analysis. I will first perform correlation anaysis and then Granger causality test.**

# In[10]:


## Using for loop to get all currency pairs from data frame
currency_pairs=[]
for column in df.columns:
  if df[column].dtype=='float64' or df[column].dtype=='int64':
    currency_pairs.append(column)


# In[11]:


## Calculate correlation and plot matrix
correlation_matrix=df[currency_pairs].corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidth=0.5)
plt.title('Correlation Matrix ')
plt.axvline(x=1, linestyle='-', color='green', linewidth=3)
plt.axvline(x=0, linestyle='-', color='green', linewidth=3)
plt.axhline(y=1, linestyle='-', color='green', linewidth=3)
plt.axhline(y=0, linestyle='-', color='green', linewidth=3)
plt.tight_layout()


# *Refer to the vertical or horizontal green line*

# In[12]:


## Identify the Currency pairs which are above threshold 0.8
threshold=0.8
correlated_pairs=correlation_matrix['USDINR Curncy']>threshold
correlated_pairs[correlated_pairs]


# **Observations:**
# 
# * From the above correlation matrix analysis, **nine currency pairs have been identified with strong positive correlations above 0.8 with USDINR. These pairs are USDCOP, USDMXN, USDBRL, USDHUF, USDIDR, USDRON, USDCNH, USDZAR and USDRUB.** This implies that the two currencies tend to move in the same direction most of the time. When USDINR appreciates, the correlated currency pairs also tend to appreciate and when USDINR depreciates, they tend to depreciate. These findings indicate the significant relationships between these currencies and USDINR, which will be explored further in this study.

# In[13]:


## Convert the daily exchange rate to monthly for better representation
month=df.resample('M').mean()


# In[14]:


## List of correlated currency pairs
currency_pairs = ['USDINR Curncy', 'USDCOP Curncy', 'USDMXN Curncy', 'USDBRL Curncy', 'USDHUF Curncy',
                  'USDIDR Curncy', 'USDRON Curncy', 'USDCNH Curncy', 'USDZAR Curncy', 'USDRUB Curncy']


# In[15]:


## Plot the correlated currency pairs
for currency_pair in currency_pairs:
    plt.figure(figsize=(12, 4))
    sns.lineplot(x=month.index, y=month[currency_pair], label=currency_pair)
    plt.title('Monthly Exchange Rates for {}'.format(currency_pair))
    plt.xlabel('Year')
    plt.ylabel('{} Rate'.format(currency_pair))
    plt.grid(True)
    plt.xticks(rotation=10)
    plt.legend()
    plt.tight_layout()


# **Observations:**
# 
# * The analysis of correlated currency pairs from 2000 to the present shows a consistent trend. These currencies tend to move together in same direction most of the time, suggesting a strong positive relationship in their exchange rate movements.

# **Granger Causality Test**
# 
# Granger causality is a statistical test used to determine whether one variable can predict another variable's future values. In simpler terms, it suggests that past values of one variable can provide useful information for predicting another variable's future values. It doesn't prove a direct cause-and-effect relationship but shows a statistical association that can be useful for prediction.
# 
# The most important factor in this test is **lag**, it refers to the number of past time periods of a variable you consider when trying to predict another variable's future values. The idea is to see if the past values of one variable, at some time lag, are useful in predicting the future values of another variable.
# 
# **Eg -** At lag 1, you consider the past values of USDZAR one time period (let's say one day) ago to predict the future values of USDINR.
# 

# In[16]:


## Decide lag and significance level
max_lag=1
α = 0.05 #significance level α represents the probability of making a Type I error


# **Granger Causality Test between USDCOP and USDINR Exchange Rates**
# * H0: There is no significant predictive relationship between the past values of USDCOP's exchange rate and USDINR's exchange rate.
# 
# * Ha: The past values of USDCOP's exchange rate significantly contribute to the prediction of USDINR's exchange rate.

# In[17]:


result=grangercausalitytests(df[['USDCOP Curncy', 'USDINR Curncy']], max_lag)


# **Observations:**
# 
# * p-value > α (0.8862 > 0.05), which indicate that past values of USDCOP's exchange rate do not significantly contribute to predicting USDINR's exchange rate. Therefore, we fail to reject the null hypothesis, suggesting no significant predictive relationship between these two exchange rates.

# **Granger Causality Test between USDMXN and USDINR Exchange Rates**
# * H0: There is no significant predictive relationship between the past values of USDMXN's exchange rate and USDINR's exchange rate.
# 
# * Ha: The past values of USDMXN's exchange rate significantly contribute to the prediction of USDINR's exchange rate.

# In[18]:


result=grangercausalitytests(df[['USDMXN Curncy', 'USDINR Curncy']], max_lag)


# **Observation:** 
# 
# * p-value < α (0.0055 < 0.05), which suggests there is evidence to reject the null hypothesis. Therefore, there is a significant predictive relationship between the past values of USDMXN's exchange rate and USDINR's exchange rate. In other words, USDMXN's exchange rate can contribute to the prediction of USDINR's exchange rate.

# **Granger Causality Test between USDBRL and USDINR Exchange Rates**
# * H0: There is no significant predictive relationship between the past values of USDBRL's exchange rate and USDINR's exchange rate.
# 
# * Ha: The past values of USDBRL's exchange rate significantly contribute to the prediction of USDINR's exchange rate.

# In[19]:


result=grangercausalitytests(df[['USDBRL Curncy', 'USDINR Curncy']], max_lag)


# **Observations:**
# * p-value > α (0.3592 > 0.05), which suggests that there is no significant predictive relationship between the past values of USDBRL's exchange rate and USDINR's exchange rate. Therefore, the null hypothesis is not rejected, indicating that USDBRL's exchange rate does not significantly contribute to the prediction of USDINR's exchange rate.

# **Granger Causality Test between USDHUF and USDINR Exchange Rates**
# * H0: There is no significant predictive relationship between the past values of USDHUF's exchange rate and USDINR's exchange rate.
# 
# * Ha: The past values of USDHUF's exchange rate significantly contribute to the prediction of USDINR's exchange rate.

# In[20]:


result=grangercausalitytests(df[['USDHUF Curncy','USDINR Curncy']], max_lag)


# **Observations:**
# 
# * p-value > α (0.0544 > 0.05), which suggests that there is no significant predictive relationship between the past values of USDHUF's exchange rate and USDINR's exchange rate. Therefore, the null hypothesis is not rejected, indicating that USDHUF's exchange rate does not significantly contribute to the prediction of USDINR's exchange rate.

# **Granger Causality Test between USDIDR and USDINR Exchange Rates**
# * H0: There is no significant predictive relationship between the past values of USDIDR's exchange rate and USDINR's exchange rate.
# 
# * Ha: The past values of USDIDR's exchange rate significantly contribute to the prediction of USDINR's exchange rate.

# In[21]:


result=grangercausalitytests(df[['USDIDR Curncy','USDINR Curncy']], max_lag)


# **Observations:**
# 
# * p-value > α (0.1630 > 0.05), which suggests that there is no significant predictive relationship between the past values of USDIDR's exchange rate and USDINR's exchange rate. Therefore, the null hypothesis is not rejected, indicating that USDIDR's exchange rate does not significantly contribute to the prediction of USDINR's exchange rate.

# **Granger Causality Test between USDRON and USDINR Exchange Rates**
# * H0: There is no significant predictive relationship between the past values of USDRON's exchange rate and USDINR's exchange rate.
# 
# * Ha: The past values of USDRON's exchange rate significantly contribute to the prediction of USDINR's exchange rate.

# In[22]:


result=grangercausalitytests(df[['USDRON Curncy','USDINR Curncy']], max_lag)


# **Observations:**
# 
# * p-value > α (0.0887 > 0.05), which suggests that there is no significant predictive relationship between the past values of USDRON's exchange rate and USDINR's exchange rate. Therefore, the null hypothesis is not rejected, indicating that USDRON's exchange rate does not significantly contribute to the prediction of USDINR's exchange rate.

# **Granger Causality Test between USDCNH and USDINR Exchange Rates**
# * H0: There is no significant predictive relationship between the past values of USDCNH's exchange rate and USDINR's exchange rate.
# 
# * Ha: The past values of USDCNH's exchange rate significantly contribute to the prediction of USDINR's exchange rate.

# In[23]:


result=grangercausalitytests(df[['USDCNH Curncy','USDINR Curncy']], max_lag)


# **Observations:**
# 
# * p-value < α (0.0226 < 0.05), which suggests that there is a significant predictive relationship between the past values of USDCNH's exchange rate and USDINR's exchange rate. Therefore, the null hypothesis is rejected, indicating that USDCNH's exchange rate significantly contributes to the prediction of USDINR's exchange rate.

# **Granger Causality Test between USDZAR and USDINR Exchange Rates**
# * H0: There is no significant predictive relationship between the past values of USDZAR's exchange rate and USDINR's exchange rate.
# 
# * Ha: The past values of USDZAR's exchange rate significantly contribute to the prediction of USDINR's exchange rate.

# In[24]:


result=grangercausalitytests(df[['USDZAR Curncy','USDINR Curncy']], max_lag)


# **Observations:**
# 
# * p-value < α (0.00 < 0.05), indicating a highly significant predictive relationship between the past values of USDZAR's exchange rate and USDINR's exchange rate. Therefore, the null hypothesis is strongly rejected, suggesting that USDZAR's exchange rate substantially contributes to the prediction of USDINR's exchange rate.

# **Granger Causality Test between USDRUB and USDINR Exchange Rates**
# * H0: There is no significant predictive relationship between the past values of USDRUB's exchange rate and USDINR's exchange rate.
# 
# * Ha: The past values of USDRUB's exchange rate significantly contribute to the prediction of USDINR's exchange rate.

# In[25]:


result=grangercausalitytests(df[['USDRUB Curncy','USDINR Curncy']], max_lag)


# **Observations:**
# 
# * p-value > α (0.1345 > 0.05), indicating that there isn't a significant predictive relationship between the past values of USDRUB's exchange rate and USDINR's exchange rate. As a result, the null hypothesis is not rejected, suggesting that USDRUB's exchange rate doesn't substantially contribute to the prediction of USDINR's exchange rate.

# # Summary
# 
# * From the correlation analysis, I identified nine currency pairs (USDCOP, USDMXN, USDBRL, USDHUF, USDIDR, USDRON, USDCNH, USDZAR, and USDRUB) with strong positive correlations above 0.8 with USDINR. This implies that USDINR tend to move in the same direction as these currency pairs.
# 
# * While many currency pairs do not have a significant predictive relationship with USDINR. However, USDMXN, USDCNH, and USDZAR stand out as strong influencers of USDINR.
# 
# These findings emphasize the importance of closely monitoring these three currencies **(USDMXN, USDCNH, and USDZAR)** when analyzing and predicting USDINR movements.

# > **Disclaimer** *The research provides insights for informational purposes and reflect personal observations backed by data. They are not research recommendations.*

# ***** *End of the page* *****
