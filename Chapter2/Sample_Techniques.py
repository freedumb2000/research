#!/usr/bin/env python
# coding: utf-8

# - By: Jacques Joubert
# - Email: jacques@quantsportal.com
# - Reference: Advances in Financial Machine Learning, Marcos Lopez De Prado, pg 40
#
#
# # Data Analysis
#
# The following data analysis is performed on a series of E-mini S&P 500 futures tick data:
#
# 1. Form tick, volume, and dollar bars
# 2. Count the number of bars produced by tick, volume, and dollar bars on a weekly basis. Plot a time seiries of that bar count.
# What bar type produces the most stable weekly count? Why?
# 3. Compute serieal correlation of returns for the three bar types. What bar method has the lowest serial correlation?
# 4. Apply the Jarque-Bera normality test on returns from the three bar types. What method achieves the lowest test statistic?
# 5. Standardize & Plot the Distributions

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import data

# In[134]:


# Read in data
dollar_bars = pd.read_csv("../Sample-Data/dollar_bars.csv", index_col=0)
volume_bars = pd.read_csv("../Sample-Data/volume_bars.csv", index_col=0)
tick_bars = pd.read_csv("../Sample-Data/tick_bars.csv", index_col=0)
# time_bars = pd.read_csv("../Sample-Data/30_minES.csv", index_col=0)

# Convert index to date_time
# time_bars.index = pd.to_datetime(time_bars.index)
tick_bars.index = pd.to_datetime(tick_bars.index)
volume_bars.index = pd.to_datetime(volume_bars.index)
dollar_bars.index = pd.to_datetime(dollar_bars.index)

# Subset data
# time_bars = time_bars["2011-08-01":"2015-01-01"]
tick_bars = tick_bars["2015-01-01":"2016-06-30"]
volume_bars = volume_bars["2015-01-01":"2016-06-30"]
dollar_bars = dollar_bars["2015-01-01":"2016-06-30"]

# Show example
dollar_bars.head()


# ## What bar type produces the most stable weekly count?

# In[135]:

print(tick_bars, volume_bars, dollar_bars)
# time_count = time_bars["Close"].resample("W", label="right").count()
tick_count = tick_bars["close"].resample("W", label="right").count()
volume_count = volume_bars["close"].resample("W", label="right").count()
dollar_count = dollar_bars["close"].resample("W", label="right").count()
print(tick_count, volume_count, dollar_count)
# count_df = pd.concat([time_count, tick_count, volume_count, dollar_count], axis=1)
count_df = pd.concat([tick_count, volume_count, dollar_count], axis=1)
# count_df.columns = ["time", "tick", "volume", "dollar"]
count_df.columns = ["tick", "volume", "dollar"]


# In[136]:


# Plot
print(count_df)
count_df.loc[:, ["tick", "volume", "dollar"]].plot(
    kind="bar", figsize=[25, 5], color=("darkred", "darkblue", "green", "darkcyan")
)
plt.title(
    "Number of bars over time",
    loc="center",
    fontsize=20,
    fontweight="bold",
    fontname="Times New Roman",
)
plt.show()


# From the above we can see that Tick bars vary the most over time. Follwed by volume and then dollar.

# ## Compute serieal correlation of returns for the four bar types. What bar method has the lowest serial correlation?

# In[137]:


# time_returns = np.log(time_bars["Close"]).diff().dropna()
tick_returns = np.log(tick_bars["close"]).diff().dropna()
volume_returns = np.log(volume_bars["close"]).diff().dropna()
dollar_returns = np.log(dollar_bars["close"]).diff().dropna()


# In[138]:


# plot_acf(time_returns, lags=10, zero=False)
# plt.title("Time Bars AutoCorrelation")
# plt.show()


# In[139]:

print(tick_returns)
plot_acf(tick_returns, lags=10, zero=False)
plt.title("Tick Bars AutoCorrelation")
plt.show()


# In[140]:


plot_acf(volume_returns, lags=10, zero=False)
plt.title("Volume Bars AutoCorrelation")
plt.show()


# In[141]:


plot_acf(dollar_returns, lags=10, zero=False)
plt.title("Dollar Bars AutoCorrelation")
plt.show()


# ## Apply the Jarque-Bera normality test on returns from the three bar types. What method achieves the lowest test statistic?

# In[142]:


from scipy import stats


# In[143]:


print("Test Statistics:")
# print("Time:", "\t", int(stats.jarque_bera(time_returns)[0]))
print("Tick:", "\t", int(stats.jarque_bera(tick_returns)[0]))
print("Volume: ", int(stats.jarque_bera(volume_returns)[0]))
print("Dollar: ", int(stats.jarque_bera(dollar_returns)[0]))


# ## Standardize & Plot the Distributions

# In[144]:


# Calculate the differences
# time_diff = time_returns
tick_diff = tick_returns
volume_diff = volume_returns
dollar_diff = dollar_returns

# Standardize the data
# time_standard = (time_diff - time_diff.mean()) / time_diff.std()
tick_standard = (tick_diff - tick_diff.mean()) / tick_diff.std()
volume_standard = (volume_diff - volume_diff.mean()) / volume_diff.std()
dollar_standard = (dollar_diff - dollar_diff.mean()) / dollar_diff.std()

# Plot the Distributions
plt.figure(figsize=(14, 10))
# sns.kdeplot(time_standard, label="Time", color="darkred")
sns.kdeplot(tick_standard, label="Tick", color="darkblue")
sns.kdeplot(volume_standard, label="Volume", color="green")
sns.kdeplot(dollar_standard, label="Dollar", linewidth=2, color="darkcyan")

sns.kdeplot(
    np.random.normal(size=len(volume_returns)),
    label="Normal",
    color="black",
    linestyle="--",
)

plt.xticks(range(-5, 6))
plt.legend(loc=8, ncol=5)
plt.title(
    "Exhibit 1 - Partial recovery of Normality through a price sampling process \nsubordinated to a volume, tick, dollar clock",
    loc="center",
    fontsize=20,
    fontweight="bold",
    fontname="Times New Roman",
)
plt.xlim(-5, 5)
plt.show()


# In[ ]:
