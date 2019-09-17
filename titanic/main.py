
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
import seaborn as sns

import pymc3 as pm
import statsmodels.api as sm
from statsmodels.formula.api import glm as glm_sm

sns.set_style('whitegrid')

target_files = ['test.csv','train.csv']
root = r'C:\Users\Work1\Desktop\GitHub\kaggle\titanic\data'
df_test = pd.read_csv(os.path.join(root, target_files[0]))
df1 = pd.read_csv(os.path.join(root, target_files[1]))
df1 = df1.drop('PassengerId', axis=1)

### Get last and first names
df1['last_name'] = df1['Name'].str.split(',', expand=True)[0]
df1['last_name'] = df1['last_name'].apply(lambda x: x.strip())
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


### Remove bracketed items
no_brackets_pat = r'(?P<prim_name>\b(\w+\s?\w?)+\b)'
p = re.compile(no_brackets_pat)
df1['first_name'] = df1['first_name'].apply(lambda x: p.search(x).group('prim_name'))


### Create column for last and first name length
df1['ln_length'] = df1['last_name'].apply(lambda x: len(x))
df1['fn_length'] = df1['first_name'].apply(lambda x: len(x))


### Test imputed Age values (mean, median)
df1.loc[df1['Age'].isna(), 'Age'] = df1['Age'].median()
#df1.loc[df1['Age'].isna(), 'Age'] = round(df1['Age'].mean(), 0)
df1['Age'] = df1['Age'].astype(int)


### Split data into variable sets
X = df1.drop('Survived', axis=1)
y = df1['Survived']


def run_pairplot()
    g = sns.PairGrid(df1, hue="Survived")
    g = g.map_diag(plt.hist, histtype="step", linewidth=2)
    g = g.map_offdiag(plt.scatter)


def show_corr_plot():
    # Correlation matrix
    df_corr = df1.corr()

    ### Generate mask for upper half
    mask = np.zeros_like(df_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Generate divergin colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(df_corr, mask=mask, cmap=cmap, linewidths=.5, cbar_kws={'shrink': .5}, ax=ax)
    
    ### Need to adjust where the plot lands.  Nudge it by 0.5
    ax.set_ylim(ax.get_ylim()[0] + 0.5, ax.get_ylim()[1] + 0.5)
    plt.yticks(rotation=0) 
    plt.show()

def plot_traces(traces, retain=0):
    '''
    Convenience function: Plot traces with overlaid means and values
    FutureWarning: The join_axes-keyword is deprecated.
    Use .reindex or .reindex_like on the result to achieve the same functionality.
    '''
    ax = pm.traceplot(traces[-retain:],
                      lines=tuple([(k, {}, v['mean'])
                                   for k, v in pm.summary(traces[-retain:]).iterrows()]))

    for i, mn in enumerate(pm.summary(traces[-retain:])['mean']):
        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data',
          xytext=(5,10), textcoords='offset points', rotation=90,
          va='bottom', fontsize='large', color='#AA0022')


def create_element_string(dataframe, target):
    col_list = [i for i in dataframe
                if dataframe[i].dtype in ['int', 'int64', 'float','float64']
                and not i in target]

    return '{} ~ {}'.format(target, ' + '.join(col_list))

with pm.Model() as log_model:
    test_str = create_element_string(df1, 'Survived')
    pm.glm.GLM.from_formula(test_str,
                            df1,
                            family=pm.glm.families.Binomial())
    trace = pm.sample(1000, tune=1000, init='adapt_diag')


plot_traces(trace)




