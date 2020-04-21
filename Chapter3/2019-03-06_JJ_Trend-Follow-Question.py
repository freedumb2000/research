#!/usr/bin/env python
# coding: utf-8

# # Trend Following Strategy
#
# This notebook answers question 3.4 form the text book Advances in Financial Machine Learning.
#
# 3.4 Develop a trend-following strategy based on a popular technical analysis statistic (e.g., crossing moving averages). For each observation, the model suggests a side, but not a size of the bet.
#
# * (a) Derive meta-labels for ptSl = [1, 2] and t1 where numDays = 1. Use as trgt the daily standard deviation as computed by Snippet 3.1.
# * (b) Train a random forest to decide whether to trade or not. Note: The decision is whether to trade or not, {0,1}, since the underlying model (the crossing moving average) has decided the side, {-1, 1}.
#
# I took some liberties by extending the features to which I use to build the meta model. I also add some performance metrics at the end.
#
# In conclusion: Meta Labeling works, SMA strategies suck.

# In[1]:


# Import the Hudson and Thames MlFinLab package
import mlfinlab as ml


# In[2]:


import numpy as np
import pandas as pd
import pyfolio as pf
import timeit

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.utils import resample
from sklearn.utils import shuffle

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read in Data
# We are using the dollar bars based off of the high quality HFT data we purchased. There is a sample of bars available in this branch as well.

# In[3]:


# Read in data
data = pd.read_csv("../Sample-Data/dollar_bars.csv")
data.index = pd.to_datetime(data["date_time"])
data = data.drop("date_time", axis=1)


# In[4]:


data = data["2011-09-01":]


# ---
# ### Fit a Primary Model: Trend Following
# Based on the simple moving average cross-over strategy.
#

# In[5]:


# compute moving averages
fast_window = 20
slow_window = 50

data["fast_mavg"] = (
    data["close"]
    .rolling(window=fast_window, min_periods=fast_window, center=False)
    .mean()
)
data["slow_mavg"] = (
    data["close"]
    .rolling(window=slow_window, min_periods=slow_window, center=False)
    .mean()
)
data.head()

# Compute sides
data["side"] = np.nan

long_signals = data["fast_mavg"] >= data["slow_mavg"]
short_signals = data["fast_mavg"] < data["slow_mavg"]
data.loc[long_signals, "side"] = 1
data.loc[short_signals, "side"] = -1

# Remove Look ahead biase by lagging the signal
data["side"] = data["side"].shift(1)


# In[6]:


# Save the raw data
raw_data = data.copy()

# Drop the NaN values from our data set
data.dropna(axis=0, how="any", inplace=True)


# In[7]:


data["side"].value_counts()


# ### Filter Events: CUSUM Filter
# Predict what will happen when a CUSUM event is triggered. Use the signal from the MAvg Strategy to determine the side of the bet.

# In[8]:


# Compute daily volatility
daily_vol = ml.util.get_daily_vol(close=data["close"], lookback=50)

# Apply Symmetric CUSUM Filter and get timestamps for events
# Note: Only the CUSUM filter needs a point estimate for volatility
cusum_events = ml.filters.cusum_filter(
    data["close"], threshold=daily_vol["2011-09-01":"2018-01-01"].mean() * 0.5
)

# Compute vertical barrier
vertical_barriers = ml.labeling.add_vertical_barrier(
    t_events=cusum_events, close=data["close"], num_days=1
)


# In[9]:


pt_sl = [1, 2]
min_ret = 0.005
triple_barrier_events = ml.labeling.get_events(
    close=data["close"],
    t_events=cusum_events,
    pt_sl=pt_sl,
    target=daily_vol,
    min_ret=min_ret,
    num_threads=3,
    vertical_barrier_times=vertical_barriers,
    side_prediction=data["side"],
)


# In[10]:


labels = ml.labeling.get_bins(triple_barrier_events, data["close"])
labels.side.value_counts()


# ---
# ### Results of Primary Model:
# What is the accuracy of predictions from the primary model (i.e., if the sec- ondary model does not filter the bets)? What are the precision, recall, and F1-scores?

# In[11]:


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
# * There is an imbalance in the classes - more are classified as "no trade"
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

# In[12]:


raw_data.head()


# ### Features

# In[13]:


# Log Returns
raw_data["log_ret"] = np.log(raw_data["close"]).diff()

# Momentum
raw_data["mom1"] = raw_data["close"].pct_change(periods=1)
raw_data["mom2"] = raw_data["close"].pct_change(periods=2)
raw_data["mom3"] = raw_data["close"].pct_change(periods=3)
raw_data["mom4"] = raw_data["close"].pct_change(periods=4)
raw_data["mom5"] = raw_data["close"].pct_change(periods=5)

# Volatility
raw_data["volatility_50"] = (
    raw_data["log_ret"].rolling(window=50, min_periods=50, center=False).std()
)
raw_data["volatility_31"] = (
    raw_data["log_ret"].rolling(window=31, min_periods=31, center=False).std()
)
raw_data["volatility_15"] = (
    raw_data["log_ret"].rolling(window=15, min_periods=15, center=False).std()
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


# In[14]:


# Re compute sides
raw_data["side"] = np.nan

long_signals = raw_data["fast_mavg"] >= raw_data["slow_mavg"]
short_signals = raw_data["fast_mavg"] < raw_data["slow_mavg"]

raw_data.loc[long_signals, "side"] = 1
raw_data.loc[short_signals, "side"] = -1


# In[15]:


# Remove look ahead bias
raw_data = raw_data.shift(1)


# #### Now get the data at the specified events

# In[16]:


# Get features at event dates
X = raw_data.loc[labels.index, :]

# Drop unwanted columns
X.drop(
    [
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


# In[17]:


y.value_counts()


# ### Balance classes

# In[18]:


# Split data into training, validation and test sets
X_training_validation = X["2011-09-01":"2018-01-01"]
y_training_validation = y["2011-09-01":"2018-01-01"]
X_train, X_validate, y_train, y_validate = train_test_split(
    X_training_validation, y_training_validation, test_size=0.15, shuffle=False
)


# In[19]:


train_df = pd.concat([y_train, X_train], axis=1, join="inner")
train_df["bin"].value_counts()


# In[20]:


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


# In[21]:


# Create training data
y_train = train_df["bin"]
X_train = train_df.loc[:, train_df.columns != "bin"]


# ### Fit a model

# In[22]:


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


# In[23]:


# extract parameters
n_estimator, depth = perform_grid_search(X_train, y_train)
c_random_state = 42
print(n_estimator, depth, c_random_state)


# In[24]:


# Refit a new model with best params, so we can see feature importance
rf = RandomForestClassifier(
    max_depth=depth,
    n_estimators=n_estimator,
    criterion="entropy",
    random_state=c_random_state,
)

rf.fit(X_train, y_train.values.ravel())


# #### Training Metrics

# In[25]:


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
plt.show()


# #### Validation Metrics

# In[26]:


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


# In[27]:


print(X_validate.index.min())
print(X_validate.index.max())


# In[28]:


# Primary model
primary_forecast = pd.DataFrame(labels["bin"])
primary_forecast["pred"] = 1
primary_forecast.columns = ["actual", "pred"]

# start = primary_forecast.index.get_loc("2016-01-20 14:42:03.869000")
# end = primary_forecast.index.get_loc("2017-12-04 15:45:10.747000") + 1

# primary_forecast.to_csv("primary_forecast.csv")

start = primary_forecast.index.get_loc("2011-09-05 14:12:23.480")
end = primary_forecast.index.get_loc("2012-01-19 12:37:09.720") + 1

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


# In[29]:


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


# ---
# ## Performance Tear Sheets (In-sample)
#
# ### Without Meta Labeling

# In[30]:


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


# In[31]:


valid_dates = X_validate.index
base_rets = labels.loc[valid_dates, "ret"]
primary_model_rets = get_daily_returns(base_rets)


# In[32]:


# Set-up the function to extract the KPIs from pyfolio
perf_func = pf.timeseries.perf_stats


# In[33]:


# Save the statistics in a dataframe
perf_stats_all = perf_func(
    returns=primary_model_rets,
    factor_returns=None,
    positions=None,
    transactions=None,
    # turnover_denom="AGB",
)
perf_stats_df = pd.DataFrame(data=perf_stats_all, columns=["Primary Model"])

# pf.show_perf_stats(primary_model_rets)
pf.show_perf_stats(primary_model_rets, factor_returns=None)


# ### With Meta Labeling

# In[34]:


meta_returns = labels.loc[valid_dates, "ret"] * y_pred
daily_meta_rets = get_daily_returns(meta_returns)


# In[35]:


# Save the KPIs in a dataframe
perf_stats_all = perf_func(
    returns=daily_meta_rets,
    factor_returns=None,
    positions=None,
    transactions=None,
    # turnover_denom="AGB",
)

perf_stats_df["Meta Model"] = perf_stats_all

pf.show_perf_stats(daily_meta_rets, factor_returns=None)


# ---
# ## Perform out-of-sample test
# ### Meta Model Metrics

# In[36]:


# Extarct data for out-of-sample (OOS)
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
subset_prim = primary_forecast["2012-02-02":]

# Performance Metrics
actual = subset_prim["actual"]
pred = subset_prim["pred"]
print(classification_report(y_true=actual, y_pred=pred))

print("Confusion Matrix")
print(confusion_matrix(actual, pred))

print("")
print("Accuracy")
print(accuracy_score(actual, pred))


# ## Primary Model (Test Data)

# In[39]:


test_dates = X_oos.index

# Downsample to daily
prim_rets_test = labels.loc[test_dates, "ret"]
daily_rets_prim = get_daily_returns(prim_rets_test)

# Save the statistics in a dataframe
perf_stats_all = perf_func(
    returns=daily_rets_prim,
    factor_returns=None,
    positions=None,
    transactions=None,
    # turnover_denom="AGB",
)

perf_stats_df["Primary Model OOS"] = perf_stats_all

# pf.create_returns_tear_sheet(labels.loc[test_dates, 'ret'], benchmark_rets=None)
pf.show_perf_stats(daily_rets_prim, factor_returns=None)


# ## Meta Model (Test Data)

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
