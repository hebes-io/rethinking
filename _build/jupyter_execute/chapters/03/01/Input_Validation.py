#!/usr/bin/env python
# coding: utf-8

# ## The input data validation stage

# In this section, the input validation steps are presented through an example.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import pandas as pd

from eensight.config import ConfigLoader
from eensight.methods.preprocessing.validation import (
    check_column_exists, 
    check_column_type_datetime, 
    check_column_values_increasing,
    check_column_values_unique,
    remove_duplicate_dates,
    validate_dataset,
)
from eensight.settings import PROJECT_PATH
from eensight.utils import load_catalog


# ### Load dataset
# 
# First, we load the catalog for one of the available datasets (the one with `site_id="b03"`):

# In[3]:


catalog = load_catalog(store_uri="../../../data", site_id="b03", namespace="train")


# Get the raw input data:

# In[4]:


features = catalog.load("train.input-features")
labels = catalog.load("train.input-labels")


# ### Validate labels
# 
# Let's work first with the label data:

# In[5]:


type(labels)


# Raw input data is always treated as a [partitioned dataset](https://kedro.readthedocs.io/en/stable/data/kedro_io.html#partitioned-dataset). This means that both features and labels are loaded by `eensight` as dictionaries with file names as keys and load functions as values.

# In[6]:


consumption = []

for load_fn in labels.values():
    consumption.append(load_fn())

consumption = pd.concat(consumption, axis=0)
consumption.head()


# **Check if a column with the name `timestamp` exists**

# In[7]:


assert check_column_exists(consumption, "timestamp").success


# **Parse the contents of the `timestamp` column as dates**

# In[8]:


date_format = "%Y-%m-%d %H:%M:%S"

if not check_column_type_datetime(consumption, "timestamp").success:
    try:
        consumption["timestamp"] = pd.to_datetime(consumption["timestamp"], format=date_format)
        consumption = consumption.dropna(subset="timestamp")
    except ValueError:
        raise ValueError(f"Column `timestamp` must be in datetime format")


# **Sort the the contents of the `timestamp` column if they are not already in an increasing order**

# In[9]:


if not check_column_values_increasing(consumption, "timestamp").success:
    consumption = consumption.sort_values(by=["timestamp"])


# **Check that the values of the `timestamp` column are unique, and if they are not, remove duplicate dates**
# 
# We can test this functionality by adding some duplicate rows to the data. Half of these rows correspond to consumption values that differ more than 0.25 (default value of `threshold`) times the standard deviation of the data. These should be replaced by `NaN` at the end of this task.

# In[10]:


n_duplicate = 100
n_out_of_range = 50
nan_before = consumption["consumption"].isna()

consumption_with_dup = pd.concat(
    (consumption, consumption[~nan_before].sample(n=n_duplicate, replace=False)),
    axis=0,
    ignore_index=True,
)


# In[11]:


assert not check_column_values_unique(consumption_with_dup, "timestamp").success


# In[12]:


data_std = consumption_with_dup["consumption"].std()

for i, (_, grouped) in enumerate(
        consumption_with_dup[consumption_with_dup.duplicated(subset="timestamp")].groupby(
            "timestamp"
        )
):
    if i < n_out_of_range:
        consumption_with_dup.loc[grouped.index[0], "consumption"] = (
            grouped["consumption"].iloc[0] + 2 * data_std
        )


# In[13]:


consumption_no_dup = remove_duplicate_dates(consumption_with_dup, "timestamp", threshold=0.25)

assert check_column_values_unique(consumption_no_dup, "timestamp").success
assert consumption_no_dup["consumption"].isna().sum() == n_out_of_range + nan_before.sum()


# Finally, the `timestamp` column becomes the dataframe's index, and columns including "Unnamed" in their name are dropped:

# In[14]:


consumption = consumption.set_index("timestamp")

to_drop = consumption.filter(like="Unnamed", axis=1).columns
if len(to_drop) > 0:
    consumption = consumption.drop(to_drop, axis=1)


# All, the aforementioned tasks are carried out by the `eensight.methods.preprocessing.validation.validate_dataset` function.

# ### Validate features
# 
# The features of this dataset include two files:

# In[15]:


features.keys()


# Each file is separately validated:

# In[16]:


load_fn = features["holidays.csv"]
holidays = load_fn()
holidays = validate_dataset(holidays)
holidays.head()


# In[17]:


assert check_column_type_datetime(holidays.reset_index(), "timestamp").success
assert check_column_values_increasing(holidays.reset_index(), "timestamp").success
assert check_column_values_unique(holidays.reset_index(), "timestamp").success


# In[18]:


load_fn = features["temperature.csv"]
temperature = load_fn()
temperature = validate_dataset(temperature)
temperature.head()


# In[19]:


assert check_column_type_datetime(temperature.reset_index(), "timestamp").success
assert check_column_values_increasing(temperature.reset_index(), "timestamp").success
assert check_column_values_unique(temperature.reset_index(), "timestamp").success


# ### Parameters

# The parameters of the input data validation stage - as they can be found in the `eensight/conf/base/parameters/preprocess.yml` file - are:

# In[20]:


params = ConfigLoader(PROJECT_PATH / "conf").get("parameters*", "parameters*/**", "**/parameters*")


# In[21]:


{
    "rebind_names": params["rebind_names"],
    "date_format": params["date_format"],
    "validation": params["validation"],
}


# -----------------
