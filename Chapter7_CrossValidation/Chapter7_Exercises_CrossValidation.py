#!/usr/bin/env python
# coding: utf-8

# * By: Jorge Osés
# * Email: jorgeoses.96@gmail.com
# * Reference: Advances in Financial Machine Learning, Chapter-07

# ## Chapter 7 Cross-Validation in Finance

# ## Introduction

# Cross-Validation is a Machine Learning technique aiming to determine how the performance of a model will generalize to an independent data set.
# Although broadly useful in all sorts of problems it generally fails when applied to a financial problem.
#
# In this chapter we will explore why it fails, and how to apply two techniques we call *purging* and *embargo* to get around its problems.

# ## Data and dependencies

# As suggested in the book, we will use a labelled dataset resulting from the exercises in chapter 3.

# In[1]:


import mlfinlab as ml
import pandas as pd
import numpy as np

from mlfinlab.cross_validation import PurgedKFold, ml_cross_val_score
from mlfinlab.util.multiprocess import mp_pandas_obj
from mlfinlab.sampling import concurrent

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

FNM = "../Sample-Data/results_3-5.csv"
rf_cfg = {
    "criterion": "entropy",
    "max_depth": 5,
    "n_estimators": 100,
    "class_weight": "balanced_subsample",
}


# ## Question-7.1

# ### Why is shuffling a dataset before conducting k-fold CV generally a bad idea in finance?
#

# Usually in finance we're working with time series structured data, so by shuffling before conducting a k-fold we're sure to
# have data in our training set that overlaps with the data in our testing set. This could easily lead to overfitting.

# ### What is the purpose of shuffling

# Shuffling data is generally used to reduce variance and to make sure that models are trained on a representative enough dataset and overfit less.
#
# By shuffling you will avoid common cases where your data is sorted by a categorical variable or even the target variable, where not even all
# the possible cases for your target variable would be present in the train side of a standard train_test split.
#
# This, however, doesn't take into account the time series structure of the data we're usually dealing with in Finance.
#

# ### Why does shuffling defeat the purpose of k-fold in financial datasets?

# The goal of any cross-validation technique is to estimate the level of fitness of a model to a data set that is independent of the train data.
#
# By shuffling first, we're making sure that the data set will not be independent of the train data because of overlapping between the test and train dataset after the shuffling.

# ## Question-7.2

# ### Take a pair of matrices (X,y) representing observed features and labels. These could be one of the datasets derived from the exercises in Chapter 3

# In[2]:


# We take the matrices X, y and t1 from those generated in 3.5 and clean their NaN values
X = pd.read_csv(FNM, index_col=0)

y = X.pop("bin")
weights = X.pop("weights")
samples_info_sets = X.pop("t1")

idx_shuffle = np.random.permutation(
    X.index
)  # pick the same shuffle permutation for exercises


# In[3]:


# display(X.head())
# display(y.head())

# print("y")
# print(y)
# print("X")
# print(X.iloc[0])

# ### (a) Derive the performance from a 10-fold CV of  n RF classifier on (X, y) without shuffling

# In[4]:


clf = RandomForestClassifier(**rf_cfg)
cv_gen = KFold(n_splits=10, random_state=1)
scores = ml_cross_val_score(
    clf, X, y, cv_gen, scoring=accuracy_score, sample_weight=weights
)


for train_index, test_index in cv_gen.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

print("NO SHUFFLE", scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ### (b) Derive the performance from a 10-fold CV of an RF on (X, y) with shuffling

# In[5]:


cv_gen = KFold(n_splits=10, random_state=1)
scores = ml_cross_val_score(
    clf,
    X.reindex(idx_shuffle),
    y.reindex(idx_shuffle),
    cv_gen,
    scoring=accuracy_score,
    sample_weight=weights,
)

print(X.iloc[0])
print("SHUFFLE", scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ### (c) Why are both results so different?

# We are seeing the result of information leaking between the training sets and the test set.

# ### (d) How does shuffling leak information?

# In a time series context, we're using information from the past to try to predict the future.
# By shuffling, we mix together the values alterating the time series structure. The train dataset will then have
# information relevant for the test dataset, it will be able to "peak" into the future.

# ## Question-7.3

# ### Take the same pair of matrices (X, y) you used in exercise 2
# ### (a) Derive the performance from a 10-fold purged CV of an RF on (X, y) with 1% embargo.
#

# In[6]:


cv_gen = PurgedKFold(n_splits=10, samples_info_sets=samples_info_sets, pct_embargo=0.01)
scores = ml_cross_val_score(
    clf, X, y, cv_gen, scoring=accuracy_score, sample_weight=weights
)
print("PURGED", scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ### (b) Why is the performance lower?

# With our PurgedKFold redundant samples are minimized, we remove shuffling so removing leakage is
# gone and thus we create another barrier of protection to prevent mixing between the train and test sets, which is called an *embargo* in the book.
#

# ### (c) Why is the result more realistic

# Further preventing leakage will always result in more realistic results.

# ## Question-7.4

# ### In this chapter we have focused on one reason why k-fold CV fails in financial applications,
# namely the fact that some information from the testing set leaks into the training set. Can you think of a second reason for CV's failure?

# Just by using the k-fold technique we cause the test set to be used multiple times in during the creation of our model leading to more bias in selection.

# ## Question-7.5

# ### Suppose you try one thousand configurations of the same investment strategy, and perform a CV on each of them.
#
# ### Some results are guaranteed to look good, just by sheer luck. If you only publish those positive results,
# and hide the rest, your audience will not be able to deduce that these results are false positives, a statistical fluke.
#
# ### This phenomenon is called "selection bias".

# ### (a) Can you imagine one procedure to prevent this?

# We can use a standard train, test and validation split.
#
# * The **train** dataset is used to fit the model
# * The **test** dataset is used to provide an unbiased evaluation of a final model fit on the training dataset.
# * The **validate** dataset is used to provide an unbiased evaluation (often for hypertuning) of a model fit on the training dataset.
#
# This way the researcher would intentionally refrain from using a part of the data in any way, providing it would be free from time overlaps and other contaminations

# ### (b) What if we split the dataset in three sets: training, validation and testing?
#
# ### The validation set is used to evaluate the trained parameters, and the testing is run only on the one configuration chosen in the validation phase. In what case does this  procedure still fail?
#
#

# Using this approach we would still have to be careful not to have time overlaps and other contaminations to the train dataset from the test and validate ones.

# ### (c) What is the key to avoiding selection bias?

# Restricting the model development procedure as hard as we can.
