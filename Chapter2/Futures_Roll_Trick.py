#!/usr/bin/env python
# coding: utf-8

# - By: Proskurin Oleksandr
# - Email: proskurinolexandr@gmail.com
# - Reference: Advances in Financial Machine Learning, Marcos Lopez De Prado, pg 37-38

# # The Single Futures Roll

# Building trading strategies on futures contracts has the unique problem that a given contract is for a short duration of time, example the 3 month contract on wheat. In order to build a continuous time series across the different contracts we stitch them together, most commonly using an auto roll or some other function. However a problem occurs when we do this, which is: come the expiry date, there is usually a price difference between the old contract and the new one. Often this difference is quite small, however for some contracts it can be quite substantial (especially if the underlying asset has a high carry costs).
#
# Consider the following example:
# September futures contract on wheat expires on December 2019, the next contract is March 2020. At the December expiration, the date at which we want to roll the contracts, the prices are:
# - December close: 100
# - March open: 120
#
# One option, which we see many people defaulting to, is to simply ignore the price difference and state the new price to be 120. A disadvantage to using this approach is if we were to train a model on this data, it will learn that every three months, around the date of expiry, prices will spike in an upward/downward fashion. It may make the wrong assumption that this is a tradeable opportunity, when in reality it isn’t. This would lead to false buy/sell signals.
#
# So what method could we apply to remedy this? The answer is the Futures Roll Trick.
#
# The book reads:
# "The ETF trick can handle the rolls of a single futures contract, as a particular case of a 1-legged spread. However, when dealing with a single futures contract, an equivalent and more direct approach is to form a time series of cumulative roll gaps, and detract that gaps series from the price series." (pg 36, AFML)
#
# ### Note:
# On further expirations the adjustment factor need to be accounted for as follows
#
# * Absolute returns: adjustment_factor = previous_adjustment_factor + new_roll_gap.
# * Relative returns:  adjustment_factor = previous_adjustment_factor * new_roll_gap.
#
# Let's look how Futures Roll Trick works on an example of S&P500 Emini futures contracts.

# In[1]:


import numpy as np
import pandas as pd
import mlfinlab as ml

import seaborn as sns
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# **Caveat:**
# For this notebook we made use of our inhouse data which we are not able to share due to legal constraints.
# We recommend you apply the same logic to your own data.

# In[2]:


aggregated_df = pd.read_csv(
    "../Sample-Data/volume_bars_es_with_nearest_contract.csv",
    index_col=0,
    parse_dates=[0],
)


# In[3]:


aggregated_df.head()


# In[4]:


aggregated_df.tail(2)


# ## Create Continuous Contracts

# In[5]:


# Get roll gaps (absolute and relative)
roll_gaps_absolute = ml.multi_product.get_futures_roll_series(
    aggregated_df,
    open_col="open",
    close_col="close",
    sec_col="ticker",
    current_sec_col="nearest_contract",
    method="absolute",
)

roll_gaps_relative = ml.multi_product.get_futures_roll_series(
    aggregated_df,
    open_col="open",
    close_col="close",
    sec_col="ticker",
    current_sec_col="nearest_contract",
    method="relative",
)


# In[6]:


# Filter out rows where there is a roll (a change in nearest contract)
# This forms the basis of the continuous contract
continuous_contract = aggregated_df[
    aggregated_df.ticker == aggregated_df.nearest_contract
]
continuous_contract.head()


# Subtract or divide the gaps from close prices. This depends on if you are using the absolute or relative method.

# In[7]:


# Make a copy of the first contract
continuous_contract_absolute_method = continuous_contract.copy()
continuous_contract_relative_method = continuous_contract.copy()

# Apply the roll gaps
continuous_contract_absolute_method["close"] -= roll_gaps_absolute
continuous_contract_relative_method["close"] /= roll_gaps_relative


# Plot nearest continuous futures contract the using different fill methods.

# In[8]:


# Plot
plt.figure(figsize=(20, 10))
continuous_contract.close.plot(label="Autoroll Series")
continuous_contract_absolute_method.close.plot(label="Absolute Return Roll")
continuous_contract_relative_method.close.plot(label="Relative Return Roll")
plt.legend(loc="best")
plt.show()


# ## Conclusion
#
# As you can see, accounting for the differences in contract prices can be quite substantial, by not accounting for it you add additional noise to the model.
#
#
# ### Notes from the book:
# "Rolled prices are used for simulating PnL and portfolio mark-to-market values. However, raw prices should still be used to size positions and determine capital consumption. Keep in mind, rolled prices can indeed become negative, particularly in futures contracts that sold off while in contango." (pg 37, AFML)
