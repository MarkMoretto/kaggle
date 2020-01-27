
"""
Purpose: Kaggle mall clustering data anlaysis
Date created: 2020-01-25

URI: https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python

Contributor(s):
    Mark M.


From the Kaggle post:

Inspiration
By the end of this case study , you would be able to answer below questions:
1. How to achieve customer segmentation using machine learning algorithm
    (KMeans Clustering) in Python in simplest way.
2. Who are your target customers with whom you can start marketing strategy
3. How the marketing strategy works in real world
"""

home = True


import os
import urllib.request as ureq
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import (
        andrews_curves, parallel_coordinates, scatter_matrix
        )

from sklearn.cluster import (
        KMeans, MiniBatchKMeans
        )
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


plt.style.use('fivethirtyeight')
# sns.set_style("whitegrid")
np.random.seed(13)


project_folder = r"C:\Users\Work1\Desktop\Info\kaggle\mall-traffic-clustering"
if home:
    os.chdir(project_folder)

csv_path = [f"{os.getcwd()}\{i}" for i in os.listdir(os.getcwd()) if i.endswith('csv')][0]

df = pd.read_csv(csv_path)

# #-- Check for any null values in each column
if df.isna().any().any():
    print('Missing values found!')

new_cols = [i.split('(')[0].strip().replace(' ', '_') for i in df.columns.values.tolist()]
col_dict = dict(zip(df.columns.values.tolist(), new_cols))

df = df.rename(columns=col_dict).copy()
df1 = df.drop('CustomerID', axis=1,)

## XKCD colors
xkcd_txt_path = r"https://xkcd.com/color/rgb.txt"
with ureq.urlopen(xkcd_txt_path) as urlf:
    raw_data = urlf.read().decode('utf-8')

# with open(f"{os.getcwd()}\\xkcd-colors.txt", 'w') as wf:
#     wf.write(raw_data)

xkcd_tokens = [i.strip() for i in raw_data.split('\n') if len(i) > 0]
xkcd_tokens = xkcd_tokens[1:]
ddict = dict(color_name=[i.split('\t')[0] for i in xkcd_tokens],
                         hex_code=[i.split('\t')[1] for i in xkcd_tokens])
hex_df = pd.DataFrame(ddict)
hex_df = hex_df.sort_values(by='hex_code', ascending=True).reset_index(drop=True)
rand_color_list = [hex_df.loc[np.random.randint(0, hex_df.shape[0]),'color_name'] for i in range(5)]

#-- Closer look at Gender metrics.

colors = ('#1f77b4','#ff7f0e',) # Lightish blue and orange


def plot_gender_andrews_curve():
    plt.rcParams['figure.figsize'] = (12, 8)
    andrews_curves(df.drop('CustomerID', axis=1), "Gender", color=colors)
    plt.title('Andrew Curves for Gender', fontsize = 20)
    plt.show()

def plot_gender_parallel_coordinates():
    colors = ('#1f77b4','#ff7f0e',) # Lightish blue and orange
    fig = plt.figure(figsize=(12, 8))
    parallel_coordinates(
            df.drop('CustomerID', axis=1)
            .sort_values(by=['Gender', 'Annual_Income'], ascending=[True, False]),
            "Gender", color=colors)
    plt.title('Parallel Coords for Gender', fontsize = 18)
    plt.tight_layout()
    plt.show()


### Correlation matrix
def plot_corr_matrix():
    rand_color_list = [
            hex_df.loc[np.random.randint(0, hex_df.shape[0]),'color_name'] \
            for i in range(5)]
    # with sns.color_palette(flatui):
    sns.set_palette(sns.xkcd_palette(rand_color_list))
    
    fig = plt.figure(figsize=(12, 8))
    # ax1 =sns.heatmap(df.corr(), cmap = 'cubehelix', annot = True)
    ax1 =sns.heatmap(df.corr(), annot = True)
    bottom, top = ax1.get_ylim()
    ax1.set_ylim(bottom + 0.5, top - 0.5)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.title('Heatmap for the Mall Data', fontsize = 18)
    plt.tight_layout()
    plt.show()





# #-- Replace string values in gender column with numeric values
# unique_gender = list(df1['Gender'].unique())
# gender_dict = {k:v for k, v in enumerate(unique_gender)}
# df1.loc[:,'Gender'] = df1.loc[:,'Gender'].replace({v:k for k, v in gender_dict.items()})


# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import MinMaxScaler, StandardScaler


def split_data(array_obj, split_percentage = 33):
    """Split data by percentage. Returns train and test splits."""
    if not '0.' in str(split_percentage):
        split_percentage /= 100

    n_split = int(len(array_obj) * split_percentage)
    # Return train, test
    return array_obj[n_split:], array_obj[:n_split]

# X_train, X_test = split_data(X)
# y_train, y_test = split_data(y)


#--Scale values in DF1
nominal_cols = df1.select_dtypes(include=['object']).columns
for c in nominal_cols:
    df_dummies = pd.get_dummies(df1[c], prefix=c)
    df1 = pd.concat([df1, df_dummies], axis=1)
    df1.drop(c, axis=1, inplace=True)

#-- Scale values We can use StandardScaler to revert values once complete
ss = StandardScaler()
ss.fit(df1)
X = ss.transform(df1)

X_train, X_test = split_data(X)




#-- Generate range of clusters and tolerances to evaluate
cluster_range = np.arange(2, 20, dtype=np.int64)
tol_range = np.linspace(1e-4, 1e-2, num=5,  dtype=np.float64)
n_init_range = np.linspace(1, 20, num=5, dtype=np.int64)

mbk_params = {'max_no_improvement': 3}
km_cases = [
        (KMeans, 'k-means++', {}),
        (KMeans, 'random', {}),
        (MiniBatchKMeans, 'k-means++', mbk_params),
    ]

#-- Iterate each combination of clusters and tolerances to find the best
#-- parameters for our model.

res_dict = dict()
idx = np.int64(0)
incr = np.int64(1)
for factory, init, params in km_cases:
    for cluster in cluster_range:
        for n_init in n_init_range:
            for t in tol_range:
                res_dict[idx] = dict()
                res_dict[idx]['factory'] = factory.__name__
                res_dict[idx]['initialization'] = init
                res_dict[idx]['n_init'] = n_init
                res_dict[idx]['cluster'] = cluster
                res_dict[idx]['tolerance'] = round(t, 4)
                km_clu = factory(n_clusters = cluster,
                                 init = init,
                                 tol = round(t, 6),
                                 n_init = n_init,
                                 random_state = idx,
                                 **params)

                km_fit = km_clu.fit(X)
                res_dict[idx]['inertia'] = km_fit.inertia_
                res_dict[idx]['silhouette_score'] = silhouette_score(X, km_fit.labels_)
                idx += incr


res_df = pd.DataFrame.from_dict(res_dict, orient='index')

#-- Sort to get highest silhouette score with highest tolerance
res_df = (res_df
          .sort_values(by=['silhouette_score', 'tolerance'], ascending=[False, False])
          .reset_index(drop=True))

best_result = res_df.loc[0,:]
best_result = best_result.to_frame().reset_index().rename(columns={'index':'key', 0:'value',})

print("\nBest result metrics:")
for i in best_result.itertuples():
    print(f"{i.key:<20}{i.value:>20}")

factory_ = best_result.loc[0, 'value']
initializer_ = best_result.loc[1, 'value']
n_init_ = best_result.loc[2, 'value']
n_cluster_ = best_result.loc[3, 'value']
tol_ = best_result.loc[4, 'value']


eval_str = f"{factory_}(n_clusters={n_cluster_}, init='{initializer_}', tol={tol_}, n_init={n_init_}, **mbk_params)"
km_clu = eval(eval_str)

# km_clu = KMeans(n_clusters = best_n_cluster, tol = best_tol)
# km_mod = km_clu.fit_predict(X)
km_mod = km_clu.fit(X)

km_values = km_mod.cluster_centers_.squeeze()
km_labels = km_mod.labels_

# km_mod.score(X)


#-- Elbow method
def plot_elbow_method():
    ssd_df = res_df.loc[:, ['cluster','inertia']]
    ssd_df = ssd_df.sort_values(by=['cluster','inertia'], ascending=[True, False]).reset_index(drop=True)
    
    
    fig = plt.figure(figsize=(12, 8))
    plt.plot(ssd_df['cluster'], ssd_df['inertia'], 'bx-')
    plt.xlabel('n_clusters')
    plt.ylabel('Sum of Squared Distances (ssd)')
    plt.title('Elbow Method For Optimal Cluster Selection', fontsize = 14)
    plt.tight_layout()
    plt.show()

# plot_elbow_method()





#-- Convert to float values
float_cols = df1.select_dtypes(include=['int32', 'int64']).columns.values
df1.loc[:, float_cols]= df1.loc[:, float_cols].astype(np.float64)


#-- DataFrame info metrics
df_desc = df1.describe()

### Visual ###
def scatter_plt_1():
    # scatter_matrix(df1, alpha=0.2, figsize=(11, 8), diagonal='kde')
    df1.plot.scatter(x='Age', y='Annual_Income', s=df1['Spending_Score'], figsize=(9, 7))







from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score

y = df1['Spending_Score']
X = df1.drop('Spending_Score', axis=1)



def split_data(dataframe, split_percentage = 33):
    """Split data by percentage. Returns train and test splits."""
    if not '0.' in str(split_percentage):
        split_percentage /= 100
    n_split = int(len(dataframe) * split_percentage)
    # Return train, test
    return dataframe[n_split:], dataframe[:n_split]

X_train, X_test = split_data(X)
y_train, y_test = split_data(y)

estimator_count = len(X_train) // 5

# ### Stratified split
# skf = StratifiedKFold(shuffle=True)
# train_index, test_index = next(iter(skf.split(X, y)))

clf = GaussianMixture(n_components=3, covariance_type='full')
clf.fit(X_train)



# display predicted scores by the model as a contour plot
x_ = np.linspace(-20., 30.)
y_ = np.linspace(-20., 40.)
X_, Y_ = np.meshgrid(x_, y_)
XX = np.array([X_.ravel(), Y_.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X_.shape)

CS = plt.contour(X_, Y_, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()





