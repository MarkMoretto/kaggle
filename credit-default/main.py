# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:36:30 2020

Using kaggle credit default dataset: https://www.kaggle.com/c/home-credit-default-risk/data

@author: mark.moretto
"""

import gc
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path

import pandas as pd
pd.set_option("mode.chained_assignment", None)
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 50)

gc.enable()

folder = Path(r"C:\Users\mark.moretto\Downloads")
train_zipfile = folder.joinpath("application_train.csv.zip")
test_zipfile = folder.joinpath("application_train.csv.zip")

headers_only = False
with ZipFile(train_zipfile) as z:
    # print(z.namelist())
    with z.open(z.namelist()[0]) as zf:
        bBuff = BytesIO(zf.read())
        if headers_only:
            # headers = pd.read_csv(bBuff, index_col=0, nrows=0).columns.tolist()
            headers = pd.read_csv(bBuff, header=0, nrows=0).columns.tolist()
        else:
            data = pd.read_csv(bBuff, header=0, low_memory=False)
            

# Record pre- and post-memory reduction results
kb_to_mb = lambda b, kb = 2 ** 10: b / (kb ** 2)
pre_downcast = kb_to_mb(data.memory_usage(deep=False).sum())

# Convet integer and float columns
for c in data.select_dtypes(include=["int64"], exclude=["object"]).columns:
    data.loc[:, c] = pd.to_numeric(data.loc[:, c], downcast="integer")
    
for f in data.select_dtypes(include=["float64"], exclude=["object"]).columns:
    data.loc[:, f] = pd.to_numeric(data.loc[:, f], downcast="float")
        
post_downcast = kb_to_mb(data.memory_usage(deep=False).sum())

# df_dtypes = data.dtypes
# df_cols = data.columns.values.tolist()
# obj_col_unique = data.select_dtypes(include=["object"]).nunique()



# not_na_srs = data.isna().sum()
# nonnull_cols = not_na_srs[not_na_srs>0].index
# null_cols = list(set(df_cols).difference(set(nonnull_cols)))
# nonnull_col_ct = len(not_na_srs[not_na_srs>0].index) # Not null column count


# Stratified sampling
sample1 = data.loc[data["TARGET"]==1].sample(frac=0.1, replace = False, axis=0)
sample0 = data.loc[data["TARGET"]==0].sample(frac=0.1, replace = False, axis=0)

df_sample = pd.concat([sample1, sample0], axis=0).sort_values("SK_ID_CURR")



# Get categorical and numeric columns
df_cols = data.columns.values.tolist()
cat_cols = data.select_dtypes(include=["object"]).columns.values.tolist()
num_cols = list(set(df_cols).difference(set(cat_cols)))

# Impute missing values
from sklearn.impute import SimpleImputer

df_sample.loc[:, num_cols] = SimpleImputer(strategy="median").fit_transform(df_sample.loc[:, num_cols])

# Encode cat
dfs = pd.get_dummies(df_sample, drop_first = True)


X = dfs.drop(["SK_ID_CURR","TARGET",], axis=1)
y = dfs.loc[:, "TARGET"]



# Select features with random forest
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

rfc_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold="median * 1.25")
rfc_selector.fit(X, y)
print(rfc_selector.threshold_)
rfc_support = rfc_selector.get_support()
rfc_features = X.loc[:, rfc_support].columns.tolist()


rfc_selector2 = SelectFromModel(RandomForestClassifier(n_estimators=100))
rfc_selector2.fit(X, y)
print(rfc_selector2.threshold_)
rfc_support2 = rfc_selector2.get_support()
rfc_features2 = X.loc[:, rfc_support2].columns.tolist()




y = data.loc[:, "TARGET"]
