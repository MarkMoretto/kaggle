
"""
Purpose: Kaggle Titanic intro competition
Date created: 2019-09-15

Contributor(s):
    Mark M.
"""

import os
import re

import theano
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

import pymc3 as pm
import statsmodels.api as sm
from statsmodels.formula.api import glm as glm_sm


target_files = ['test.csv','train.csv']
root = r'C:\Users\Work1\Desktop\GitHub\kaggle\titanic\data'
df_test = pd.read_csv(os.path.join(root, target_files[0]))
df1 = pd.read_csv(os.path.join(root, target_files[1]))

###
df1['last_name'] = df1['Name'].str.split(',', expand=True)[0]
df1['first_name'] = df1['Name'].str.split(',', expand=True)[1]


### Find some last name frequency and outlier data
l_name_counts = df1['last_name'].value_counts().sort_values(ascending=False)
cov = l_name_counts.std() / l_name_counts.mean()
std_scr_l_name = l_name_counts - l_name_counts.mean() / l_name_counts.std()
sd_3 = std_scr_l_name.std() * 3
l_name_outliers = std_scr_l_name[abs(std_scr_l_name) >= sd_3]
outlier_vs_tot = len(l_name_outliers) / len(l_name_counts)
print('Pct of last name outliers: {}%'.format(round(outlier_vs_tot * 100, 2)))



### Clean the first names
### Strip excess whitespace
df1['first_name'] = df1['first_name'].apply(lambda x: x.strip())

### Regex to remove Mr, Mrs, etc.
clean_name_pat = r'\s?\w+\.\s+?(?P<core>[a-zA-Z \(\)]+)'
p = re.compile(clean_name_pat)
df1['first_name'] = df1['first_name'].apply(lambda x: p.search(x).group('core'))




### Split data into variable sets
X = df_train.drop('Survived', axis=1)
y = df_train['Survived']


data_dict = {
    'X': X,
    'y': y,
    }

with pm.Model() as model:
    lm = pm.glm.LinearComponent.from_formula('y ~ X', data_dict)
    sigma = Uniform('sigma', 0, 20)
    y_obs = Normal('y_obs', mu = lm.y_est, sigma=sigma, observed=y)
    trace = sample(2000, cores=2)


plt.figure(figsize=(5, 5))
plt.plot(X, y, 'X')
plot_posterior_predictive_glm(trace)
