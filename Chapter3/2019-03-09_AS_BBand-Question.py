#!/usr/bin/env python
# coding: utf-8

# # Mean reverting strategy based on Bollinger bands Strategy
#
# This notebook answers question 3.5 form the text book Advances in Financial Machine Learning.
#
# 3.5 Develop a mean-reverting strategy based on Bollinger bands. For each observation, the model suggests a side, but not a size of the bet.
#
# * (a) Derive meta-labels for ptSl = [0, 2] and t1 where numDays = 1. Use as trgt the daily standard deviation as computed by Snippet 3.1.
# * (b) Train a random forest to decide whether to trade or not. Use as features: volatility, seial correlation, and teh crossinmg moving averages.
# * (c) What is teh accuracy of prediction from the primary model? (i.e. if the secondary model does not filter the bets) What are the precision, recall and FI-scores?
# * (d) What is teh accuracy of prediction from the primary model? What are the precision, recall and FI-scores?
#

# In[1]:


# Import the Hudson and Thames MlFinLab package
import mlfinlab as ml


# In[2]:


import numpy as np
import pandas as pd
import pyfolio as pf
import timeit

from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import multiprocessing

# get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read in Data
# We are using the dollar bars based off of the high quality HFT data we purchased. There is a sample of bars available in this branch as well.

# In[3]:


# Read in data
data = pd.read_csv("../Sample-Data/dollar_bars.csv")
data.index = pd.to_datetime(data["date_time"])
data = data.drop("date_time", axis=1)


# ---
# ### Define helper functions

# In[4]:


# Compute RSI
def relative_strength_index(df, n):
    """Calculate Relative Strength Index(RSI) for given data.
        https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py
        
        :param df: pandas.DataFrame
        :param n: 
        :return: pandas.DataFrame
        """
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.loc[i + 1, "high"] - df.loc[i, "high"]
        DoMove = df.loc[i, "low"] - df.loc[i + 1, "low"]
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    RSI = pd.Series(round(PosDI * 100.0 / (PosDI + NegDI)), name="RSI_" + str(n))
    # df = df.join(RSI)
    return RSI


def get_rsi(data, window=14):
    df = data.copy(deep=True).reset_index()
    rsi = relative_strength_index(df, window)
    rsi_df = pd.Series(data=rsi.values, index=data.index)
    return rsi_df


# In[5]:


def bbands(close_prices, window, no_of_stdev):
    # rolling_mean = close_prices.rolling(window=window).mean()
    # rolling_std = close_prices.rolling(window=window).std()
    rolling_mean = close_prices.ewm(span=window).mean()
    rolling_std = close_prices.ewm(span=window).std()

    upper_band = rolling_mean + (rolling_std * no_of_stdev)
    lower_band = rolling_mean - (rolling_std * no_of_stdev)

    return rolling_mean, upper_band, lower_band


# ---
# ### Fit a Primary Model: Mean-reverting based on Bollinger bands
# Based on the mean-reverting Bollinger band strategy.
#

# In[6]:


# compute bands
window = 50
data["avg"], data["upper"], data["lower"] = bbands(
    data["close"], window, no_of_stdev=1.5
)
data.sample(10)


# In[7]:


# Compute RSI
rsi_df = get_rsi(data, window=14)
data["rsi"] = pd.Series(data=rsi_df.values, index=data.index)

# Drop the NaN values from our data set
data.dropna(axis=0, how="any", inplace=True)


# ---
# ### Fit a Primary Model: Bollinger Band Mean-Reversion

# In[8]:


# Compute sides
data["side"] = np.nan

long_signals = data["close"] <= data["lower"]
short_signals = data["close"] >= data["upper"]

data.loc[long_signals, "side"] = 1
data.loc[short_signals, "side"] = -1

print(data.side.value_counts())

# Remove Look ahead biase by lagging the signal
data["side"] = data["side"].shift(1)


# In[9]:


# Save the raw data
raw_data = data.copy()

# Drop the NaN values from our data set
data.dropna(axis=0, how="any", inplace=True)


# In[10]:


print(data.side.value_counts())


# ### Filter Events: CUSUM Filter
# Predict what will happen when a CUSUM event is triggered. Use the signal from the MAvg Strategy to determine the side of the bet.

# In[11]:


# Compute daily volatility
daily_vol = ml.util.get_daily_vol(close=data["close"], lookback=50)

# Apply Symmetric CUSUM Filter and get timestamps for events
# Note: Only the CUSUM filter needs a point estimate for volatility
cusum_events = ml.filters.cusum_filter(
    data["close"], threshold=daily_vol["2011-09-01":"2018-01-01"].mean() * 0.1
)

# Compute vertical barrier
vertical_barriers = ml.labeling.add_vertical_barrier(
    t_events=cusum_events, close=data["close"], num_days=1
)


# In[12]:


pt_sl = [0, 2]
min_ret = 0.0005
triple_barrier_events = ml.labeling.get_events(
    close=data["close"],
    t_events=cusum_events,
    pt_sl=pt_sl,
    target=daily_vol,
    min_ret=min_ret,
    num_threads=2,
    vertical_barrier_times=vertical_barriers,
    side_prediction=data["side"],
)


# In[13]:


labels = ml.labeling.get_bins(triple_barrier_events, data["close"])
labels.side.value_counts()


# ---
# ### Results of Primary Model:
#

# In[14]:


primary_forecast = pd.DataFrame(labels["bin"])
primary_forecast["pred"] = 1
primary_forecast.columns = ["actual", "pred"]

# Performance Metrics
actual = primary_forecast["actual"]
pred = primary_forecast["pred"]
print(classification_report(y_true=actual, y_pred=pred))

print("Confusion Matrix")
print(confusion_matrix(actual, pred))

print("")
print("Accuracy")
print(accuracy_score(actual, pred))


# **A few takeaways**
# * There is an imbalance in the classes - far more are classified as "no trade"
# * Meta-labeling says that there are many false-positives
# * the sklearn's confusion matrix is [[TN, FP][FN, TP]]

# ---
# ## Fit a Meta Model
# Train a random forest to decide whether to trade or not (i.e 1 or 0 respectively) since the earlier model has decided the side (-1 or 1)
#
# Create the following features:
# * Volatility
# * Serial Correlation
# * The returns at the different lags from the serial correlation
# * The sides from the SMavg Strategy

# In[15]:


raw_data.head()


# ### Features

# In[16]:


# Log Returns
raw_data["log_ret"] = np.log(raw_data["close"]).diff()

# Momentum
raw_data["mom1"] = raw_data["close"].pct_change(periods=1)
raw_data["mom2"] = raw_data["close"].pct_change(periods=2)
raw_data["mom3"] = raw_data["close"].pct_change(periods=3)
raw_data["mom4"] = raw_data["close"].pct_change(periods=4)
raw_data["mom5"] = raw_data["close"].pct_change(periods=5)

# Volatility
window_stdev = 50
raw_data["volatility"] = (
    raw_data["log_ret"]
    .rolling(window=window_stdev, min_periods=window_stdev, center=False)
    .std()
)

# Serial Correlation (Takes about 4 minutes)
window_autocorr = 50

raw_data["autocorr_1"] = (
    raw_data["log_ret"]
    .rolling(window=window_autocorr, min_periods=window_autocorr, center=False)
    .apply(lambda x: x.autocorr(lag=1), raw=False)
)
raw_data["autocorr_2"] = (
    raw_data["log_ret"]
    .rolling(window=window_autocorr, min_periods=window_autocorr, center=False)
    .apply(lambda x: x.autocorr(lag=2), raw=False)
)
raw_data["autocorr_3"] = (
    raw_data["log_ret"]
    .rolling(window=window_autocorr, min_periods=window_autocorr, center=False)
    .apply(lambda x: x.autocorr(lag=3), raw=False)
)
raw_data["autocorr_4"] = (
    raw_data["log_ret"]
    .rolling(window=window_autocorr, min_periods=window_autocorr, center=False)
    .apply(lambda x: x.autocorr(lag=4), raw=False)
)
raw_data["autocorr_5"] = (
    raw_data["log_ret"]
    .rolling(window=window_autocorr, min_periods=window_autocorr, center=False)
    .apply(lambda x: x.autocorr(lag=5), raw=False)
)

# Get the various log -t returns
raw_data["log_t1"] = raw_data["log_ret"].shift(1)
raw_data["log_t2"] = raw_data["log_ret"].shift(2)
raw_data["log_t3"] = raw_data["log_ret"].shift(3)
raw_data["log_t4"] = raw_data["log_ret"].shift(4)
raw_data["log_t5"] = raw_data["log_ret"].shift(5)

# Add fast and slow moving averages
fast_window = 7
slow_window = 15

raw_data["fast_mavg"] = (
    raw_data["close"]
    .rolling(window=fast_window, min_periods=fast_window, center=False)
    .mean()
)
raw_data["slow_mavg"] = (
    raw_data["close"]
    .rolling(window=slow_window, min_periods=slow_window, center=False)
    .mean()
)


# In[17]:


# Add Trending signals
raw_data["sma"] = np.nan

long_signals = raw_data["fast_mavg"] >= raw_data["slow_mavg"]
short_signals = raw_data["fast_mavg"] < raw_data["slow_mavg"]
raw_data.loc[long_signals, "sma"] = 1
raw_data.loc[short_signals, "sma"] = -1


# In[18]:


# Re compute sides
raw_data["side"] = np.nan

long_signals = raw_data["close"] <= raw_data["lower"]
short_signals = raw_data["close"] >= raw_data["upper"]

raw_data.loc[long_signals, "side"] = 1
raw_data.loc[short_signals, "side"] = -1


# In[19]:


# Remove look ahead bias
raw_data = raw_data.shift(1)


# #### Now get the data at the specified events

# In[20]:


# Get features at event dates
X = raw_data.loc[labels.index, :]

# Drop unwanted columns
X.drop(
    [
        "avg",
        "upper",
        "lower",
        "open",
        "high",
        "low",
        "close",
        "cum_vol",
        "cum_dollar",
        "cum_ticks",
        "fast_mavg",
        "slow_mavg",
    ],
    axis=1,
    inplace=True,
)

y = labels["bin"]
X.head()


# ### Fit a model

# In[21]:


# Split data into training, validation and test sets
X_training_validation = X["2011-09-01":"2018-01-01"]
y_training_validation = y["2011-09-01":"2018-01-01"]
X_train, X_validate, y_train, y_validate = train_test_split(
    X_training_validation, y_training_validation, test_size=0.2, shuffle=False
)


# In[22]:


train_df = pd.concat([y_train, X_train], axis=1, join="inner")
train_df["bin"].value_counts()


# In[23]:


# Upsample the training data to have a 50 - 50 split
# https://elitedatascience.com/imbalanced-classes
majority = train_df[train_df["bin"] == 0]
minority = train_df[train_df["bin"] == 1]

new_minority = resample(
    minority,
    replace=True,  # sample with replacement
    n_samples=majority.shape[0],  # to match majority class
    random_state=42,
)

train_df = pd.concat([majority, new_minority])
train_df = shuffle(train_df, random_state=42)

train_df["bin"].value_counts()


# In[24]:


# Create training data
y_train = train_df["bin"]
X_train = train_df.loc[:, train_df.columns != "bin"]


# In[25]:


parameters = {
    "max_depth": [2, 3, 4, 5, 7],
    "n_estimators": [1, 10, 25, 50, 100, 256, 512],
    "random_state": [42],
}


def perform_grid_search(X_data, y_data):
    rf = RandomForestClassifier(criterion="entropy")

    clf = GridSearchCV(rf, parameters, cv=4, scoring="roc_auc", n_jobs=3)

    clf.fit(X_data, y_data)

    print(clf.cv_results_["mean_test_score"])

    return clf.best_params_["n_estimators"], clf.best_params_["max_depth"]


# In[26]:


# extract parameters
n_estimator, depth = perform_grid_search(X_train, y_train)
c_random_state = 42
print(n_estimator, depth, c_random_state)


# In[27]:


# Random Forest Model
rf = RandomForestClassifier(
    max_depth=depth,
    n_estimators=n_estimator,
    criterion="entropy",
    random_state=c_random_state,
)
rf.fit(X_train, y_train.values.ravel())


# #### Training Metrics

# In[28]:


# Performance Metrics
y_pred_rf = rf.predict_proba(X_train)[:, 1]
y_pred = rf.predict(X_train)
fpr_rf, tpr_rf, _ = roc_curve(y_train, y_pred_rf)
print(classification_report(y_train, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_train, y_pred))

print("")
print("Accuracy")
print(accuracy_score(y_train, y_pred))

plt.figure(1)
plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr_rf, tpr_rf, label="RF")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve")
plt.legend(loc="best")
# xx = multiprocessing.Process(target=plt.show)
# xx.start()
plt.show()


# #### Validation Metrics

# In[29]:


# Meta-label
# Performance Metrics
y_pred_rf = rf.predict_proba(X_validate)[:, 1]
y_pred = rf.predict(X_validate)
fpr_rf, tpr_rf, _ = roc_curve(y_validate, y_pred_rf)
print(classification_report(y_validate, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_validate, y_pred))

print("")
print("Accuracy")
print(accuracy_score(y_validate, y_pred))

plt.figure(1)
plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr_rf, tpr_rf, label="RF")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve")
plt.legend(loc="best")
plt.show()


# In[30]:


# Primary model
primary_forecast = pd.DataFrame(labels["bin"])
primary_forecast["pred"] = 1
primary_forecast.columns = ["actual", "pred"]

# start = primary_forecast.index.get_loc("2016-08-01 10:31:44.455000")
# end = primary_forecast.index.get_loc("2017-12-29 21:03:37.018000") + 1
# print(primary_forecast)
# primary_forecast.to_csv("2019-03-09_AS_BBand-Question_primary_forecast.csv")
start = primary_forecast.index.get_loc("2011-08-02 17:02:16.995")
end = primary_forecast.index.get_loc("2011-10-27 13:08:53.917") + 1

subset_prim = primary_forecast[start:end]

# Performance Metrics
actual = subset_prim["actual"]
pred = subset_prim["pred"]
print(classification_report(y_true=actual, y_pred=pred))

print("Confusion Matrix")
print(confusion_matrix(actual, pred))

print("")
print("Accuracy")
print(accuracy_score(actual, pred))


# In[31]:


# Feature Importance
title = "Feature Importance:"
figsize = (15, 5)

feat_imp = pd.DataFrame({"Importance": rf.feature_importances_})
feat_imp["feature"] = X.columns
feat_imp.sort_values(by="Importance", ascending=False, inplace=True)
feat_imp = feat_imp

feat_imp.sort_values(by="Importance", inplace=True)
feat_imp = feat_imp.set_index("feature", drop=True)
feat_imp.plot.barh(title=title, figsize=figsize)
plt.xlabel("Feature Importance Score")
plt.show()


# Note how low the side is ranked.

# ---
# ## Performance Tear Sheets (In-sample)

# In[32]:


# Set-up the function to extract the KPIs from pyfolio
perf_func = pf.timeseries.perf_stats


# In[33]:


def get_daily_returns(intraday_returns):
    """
    This changes returns into daily returns that will work using pyfolio. Its not perfect...
    """

    cum_rets = (intraday_returns + 1).cumprod()

    # Downsample to daily
    daily_rets = cum_rets.resample("B").last()

    # Forward fill, Percent Change, Drop NaN
    daily_rets = daily_rets.ffill().pct_change().dropna()

    return daily_rets


# In[34]:


test_dates = X_validate.index

base_rets = labels.loc[test_dates, "ret"]
primary_model_rets = get_daily_returns(base_rets)

# Save the statistics in a dataframe
perf_stats_all = perf_func(
    returns=primary_model_rets,
    factor_returns=None,
    positions=None,
    transactions=None,
    # turnover_denom="AGB",
)
perf_stats_df = pd.DataFrame(data=perf_stats_all, columns=["Primary Model"])

# pf.create_returns_tear_sheet(labels.loc[test_dates, 'ret'], benchmark_rets=None)

# pf.show_perf_stats(primary_model_rets)
pf.show_perf_stats(primary_model_rets, factor_returns=None)


# In[35]:


meta_returns = labels.loc[test_dates, "ret"] * y_pred
daily_meta_rets = get_daily_returns(meta_returns)

# save the KPIs in a dataframe
perf_stats_all = perf_func(
    returns=daily_meta_rets,
    factor_returns=None,
    positions=None,
    transactions=None,
    # turnover_denom="AGB",
)

perf_stats_df["Meta Model"] = perf_stats_all

# pf.create_returns_tear_sheet(meta_returns, benchmark_rets=None)

# pf.show_perf_stats(daily_meta_rets)
pf.show_perf_stats(daily_meta_rets, factor_returns=None)


# ---
# ## Perform out-of-sample test

# In[36]:


# extarct data for out-of-sample (OOS)
# X_oos = X["2018-01-02":]
# y_oos = y["2018-01-02":]

X_oos = X["2012-02-02":]
y_oos = y["2012-02-02":]


# In[37]:


# Performance Metrics
y_pred_rf = rf.predict_proba(X_oos)[:, 1]
y_pred = rf.predict(X_oos)
fpr_rf, tpr_rf, _ = roc_curve(y_oos, y_pred_rf)
print(classification_report(y_oos, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_oos, y_pred))

print("")
print("Accuracy")
print(accuracy_score(y_oos, y_pred))

plt.figure(1)
plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr_rf, tpr_rf, label="RF")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve")
plt.legend(loc="best")
plt.show()


# In[38]:


# Primary model
primary_forecast = pd.DataFrame(labels["bin"])
primary_forecast["pred"] = 1
primary_forecast.columns = ["actual", "pred"]

# subset_prim = primary_forecast["2018-01-02":]

subset_prim = primary_forecast["2012-01-02":]

# Performance Metrics
actual = subset_prim["actual"]
pred = subset_prim["pred"]
print(classification_report(y_true=actual, y_pred=pred))

print("Confusion Matrix")
print(confusion_matrix(actual, pred))

print("")
print("Accuracy")
print(accuracy_score(actual, pred))


# ---
# ## Performance Tear Sheets (Out-of-sample)

# In[39]:


test_dates = X_oos.index

base_rets_oos = labels.loc[test_dates, "ret"]
primary_model_rets_oos = get_daily_returns(base_rets_oos)

# Save the statistics in a dataframe
perf_stats_all = perf_func(
    returns=primary_model_rets_oos,
    factor_returns=None,
    positions=None,
    transactions=None,
    # turnover_denom="AGB",
)

perf_stats_df["Primary Model OOS"] = perf_stats_all


# pf.create_returns_tear_sheet(labels.loc[test_dates, 'ret'], benchmark_rets=None)
# pf.show_perf_stats(primary_model_rets_oos)
pf.show_perf_stats(primary_model_rets_oos, factor_returns=None)


# In[40]:


meta_returns = labels.loc[test_dates, "ret"] * y_pred
daily_rets_meta = get_daily_returns(meta_returns)

# save the KPIs in a dataframe
perf_stats_all = perf_func(
    returns=daily_rets_meta,
    factor_returns=None,
    positions=None,
    transactions=None,
    # turnover_denom="AGB",
)

perf_stats_df["Meta Model OOS"] = perf_stats_all

pf.create_returns_tear_sheet(daily_rets_meta, benchmark_rets=None)


# ---
# ## Conclusion
# There are a few takeaways from the Bollinger bands mean-reverting strategy:
# 1. In out-of-sample tests, the meta-model out-performs the primary model in all key statistics:
#     * Annualized returns: 17.7% vs. 35.3%
#     * Annualized volatility: 95% vs. 40%
#     * Sharpe ratio: 0.65 vs. 0.82
# 2. In all cases the meta-model (one with meta-labeling) decreases annualized volatility, maximum drawdown and increases Sharpe ratio compared to the primary model
#
