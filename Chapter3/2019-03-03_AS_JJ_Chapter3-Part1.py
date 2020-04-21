#!/usr/bin/env python
# coding: utf-8

# # Chapter 3 Questions

# #### 3.1 Form dollar bars for E-mini S&P 500 futures:
# 1. Apply a symmetric CUSUM filter (Chapter 2, Section 2.5.2.1)
# where the threshold is the standard deviation of daily returns (Snippet 3.1).
# 2. Use Snippet 3.4 on a pandas series t1, where numDays=1.
# 3. On those sampled features, apply the triple-barrier method,
# where ptSl=[1,1] and t1 is the series you created in point 1.b.
# 4. Apply getBins to generate the labels.

# In[1]:


# Import the Hudson and Thames MlFinLab package
import mlfinlab as ml


# In[2]:


import numpy as np
import pandas as pd
import timeit

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report, confusion_matrix

import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Read in data
data = pd.read_csv("../Sample-Data/dollar_bars.csv", nrows=40000)
data.index = pd.to_datetime(data["date_time"])
data = data.drop("date_time", axis=1)


# In[4]:


data.head()


# **Apply a symmetric CUSUM filter (Chapter 2, Section 2.5.2.1) where the threshold is the standard deviation of daily returns (Snippet 3.1).**

# In[5]:


# Compute daily volatility
vol = ml.util.get_daily_vol(close=data["close"], lookback=50)


# In[6]:


vol.plot(figsize=(14, 7), title="Volatility as caclulated by de Prado")
plt.show()


# In[7]:


# Apply Symmetric CUSUM Filter and get timestamps for events
# Note: Only the CUSUM filter needs a point estimate for volatility
cusum_events = ml.filters.cusum_filter(data["close"], threshold=vol.mean())


# **Use Snippet 3.4 on a pandas series t1, where numDays=1.**

# In[8]:


# Compute vertical barrier
vertical_barriers = ml.labeling.add_vertical_barrier(cusum_events, data["close"])
vertical_barriers.head()


# **On those sampled features, apply the triple-barrier method, where ptSl=[1,1] and t1 is the series you created in point 1.b.**

# In[9]:


triple_barrier_events = ml.labeling.get_events(
    close=data["close"],
    t_events=cusum_events,
    pt_sl=[1, 1],
    target=vol,
    min_ret=0.01,
    num_threads=1,
    vertical_barrier_times=vertical_barriers,
    side_prediction=None,
)


# In[10]:


triple_barrier_events.head()


# In[11]:


labels = ml.labeling.get_bins(triple_barrier_events, data["close"])


# In[12]:


labels.head()


# In[13]:


labels["bin"].value_counts()


# ---
# #### 3.2 From exercise 1, use Snippet 3.8 to drop rare labels.

# In[14]:


clean_labels = ml.labeling.drop_labels(labels)


# In[15]:


print(labels.shape)
print(clean_labels.shape)

labels.to_csv("2019-03-03_AS_JJ_Chapter3-Part1_labels.csv")

# ---
# #### 3.3 Adjust the getBins function (Snippet 3.5) to return a 0 whenever the vertical barrier is the one touched first.
# This change was made inside the module CoreFunctions.

# ---
# #### 3.4 Develop a trend-following strategy based on a popular technical analysis statistic (e.g., crossing moving averages). For each observation, themodel suggests a side, but not a size of the bet.
#
# 1. Derive meta-labels for pt_sl = [1,2] and t1 where num_days=1. Use as trgt the daily standard deviation as computed by Snippet 3.1.
# 2. Train a random forest to decide whether to trade or not. Note: The decision is whether to trade or not, {0,1}, since the underllying model (the crossing moveing average has decided the side{-1, 1})

# In[16]:


# This question is answered in the notebook: 2019-03-06_JJ_Trend-Following-Question


# ----
# #### 3.5 Develop a mean-reverting strategy based on Bollinger bands. For each observation, the model suggests a side, but not a size of the bet.
#
# * (a) Derive meta-labels for ptSl = [0, 2] and t1 where numDays = 1. Use as trgt the daily standard deviation as computed by Snippet 3.1.
# * (b) Train a random forest to decide whether to trade or not. Use as features: volatility, seial correlation, and teh crossinmg moving averages.
# * (c) What is teh accuracy of prediction from the primary model? (i.e. if the secondary model does not filter the bets) What are the precision, recall and FI-scores?
# * (d) What is teh accuracy of prediction from the primary model? What are the precision, recall and FI-scores?
#

# In[17]:


# This question is answered in the notebook: 2019-03-07_BBand-Question
