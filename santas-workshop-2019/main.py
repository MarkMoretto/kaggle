"""
Purpose: Kaggle - Santa's workshop 2019
Date created: 2020-01-02

URI: https://www.kaggle.com/c/santa-workshop-tour-2019/overview/evaluation

Contributor(s):
    Mark M.

Desc:
    Your task is to schedule the families to Santa's Workshop in a way that
    minimizes the penalty cost to Santa (as described on the Evaluation page).
    
    Each family has listed their top 10 preferences for the dates they'd like
    to attend Santa's workshop tour. Dates are integer values representing the
    days before Christmas, e.g., the value 1 represents Dec 24, the value 2
    represents Dec 23, etc. Each family also has a number of people attending,
    n_people.
    
    Every family must be scheduled for one and only one assigned_day.


Notes:
    Bin packing? https://stackoverflow.com/questions/7392143/python-implementations-of-packing-algorithm
"""

### Change folder to project directory

home = True
from os import chdir, listdir
if home:
    proj_folder = r'C:\Users\Work1\Desktop\Info\kaggle\santas-workshop-2019'
else:
    proj_folder = r"C:\Users\MMorett1\Desktop\Projects Main\kaggle\santas-workshop-2019"

chdir(proj_folder)

import os.path
import numpy as np
import pandas as pd
import seaborn as sns
from random import randrange
import matplotlib.pyplot as plt
import datetime as dt
from zipfile import ZipFile, is_zipfile
sns.set()
#-- Set a few pandas options
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 1000)
pd.set_option('mode.chained_assignment', None) # Silence `SettingWithCopyWarning`



# xmas_day: dt.datetime.date = dt.datetime(year=2019, month=12, day=25).date()
# today: dt.datetime.date = dt.datetime.today().date()

# def days_between(dt1: dt.datetime.date, dt2: dt.datetime.date) -> np.int32:
#     """Find net days between two dates."""
#     return np.abs((dt1 - dt2).days)


class DataImporter:
    """
    Import data from csv file in compressed zipfile in project folder.

    Method `import_csv_to_df()` will return pandas.DataFrame() object

    Parameters:
        project_folder (string)
    """
    def __init__(self, project_folder: str):
        self.project_folder = project_folder

    def __repr__(self):
        return '<DataImporter class>'

    @property
    def project_folder(self):
        return self.__project_folder


    @project_folder.setter
    def project_folder(self, value = None):
        if value is None:
            raise ValueError("Folder path value error!")
        else:
            self.__project_folder = value


    def get_zipfolder(self):
        """Search for zipfile in directory. Return full path."""
        zp_fldr = [os.path.join(self.project_folder, i) for i \
                        in listdir(self.project_folder) \
                        if str(i).endswith('zip')]
    
        if len(zp_fldr) > 1:
            print('Multiple zip files found.')
        else:
            output = zp_fldr[0]
            if is_zipfile(output):
                self.zipfile_path = output
            else:
                print(f"Zipfile '{output}' not valid.")


    def import_csv_to_df(self, csv_filename = 'family_data'):

        #-- Set `zipfile_path` class variable
        self.get_zipfolder()

        self.csv_filename = str.split(csv_filename, '.')[0]

        with ZipFile(self.zipfile_path) as zf:
            zf_list = zf.namelist()
            for filename in zf_list:
                #-- If filename equals
                if str.split(filename, '.')[0] == self.csv_filename:
                    with zf.open(filename) as csvf:
                        return pd.read_csv(csvf)

#-- Instantiate DataImporter and create dataframe

dimp = DataImporter(proj_folder)
df = dimp.import_csv_to_df()
# dfs = dimp.import_csv_to_df('sample_submission') # Sample submission
# df = df.drop('family_id', axis=1)
df['family_id'] = df['family_id'].astype(str)



# #-- Column not included in data set
# additional_col: str = 'otherwise'

# #-- Add additional column with default value of zero
# df.insert(df.shape[1]-1, additional_col, np.float32(0))

choice_cols: list = [i for i in df.columns.values if 'choice' in i]
n_choice_cols: np.array = np.arange(len(choice_cols), dtype=np.int8)
def weights_misc():
    ulam_seq = [1, 2, 3, 4, 6, 8, 11, 13, 16, 18]
    # choice_weights: np.array = np.arange(1, len(choice_cols) + 1, dtype=np.int8)[::-1]
    choice_weights: np.array = np.array(ulam_seq, dtype=np.int16)[::-1]
    col_count_seq: np.array = np.array([i for i in range(1, len(choice_cols) + 1)], dtype=np.uint8)

col_count_seq: np.array = np.array([i for i in range(1, len(choice_cols) + 1)], dtype=np.uint16)
col_count_inv: np.array = col_count_seq[::-1]
choice_weights: np.array = np.array([40, 25, 16, 11, 7, 4, 2, 1, 1, 1,])
# np.diff(choice_weights[::-1])

#-- Adjusted n_people counts
n_ppl_freq_adj: pd.Series = (1 / df['n_people'].value_counts()) * 2000
n_ppl_freq_adj = n_ppl_freq_adj.astype(int)



# ### Add column that sums each day choice
# # Multiply by choice_weights array, sum the columns, multiple by n_people
# choice_col_name = 'choice_bias'
# df[choice_col_name] = (df[choice_cols].multiply(choice_weights).sum(axis=1) * df['n_people'].map(n_ppl_freq_adj))
# df[choice_col_name] = round(df[choice_col_name].astype(np.float32), 2)
# # df.drop('choice_bias', axis=1, inplace=True)
# # df['choice_bias'].nunique()

# sort_cols = ['n_people']
# sort_cols.extend(choice_cols)
# sort_cols.extend([choice_col_name])
# sort_bool = [False if i.endswith(('people',)) else True for i in sort_cols]
# df = df.sort_values(by=sort_cols, ascending = sort_bool)


### Evaluation ###
#-- Create penalty dataframe to help crunch numbers

penalty_df_index = choice_cols.copy()
# penalty_df_index.append(additional_col)

#-- Base prices for santa's buffet and helicopter ride
base_buffet_price = 36.0
base_ride_price = 398.0

ddict = {
    'gift_card': [
            0.,
            50.,
            50.,
            100.,
            200.,
            200.,
            300.,
            300.,
            400.,
            500.,
            # 500.,
            ],
    'santas_buffet': [
            0.,
            0.,
            base_buffet_price*0.25,
            base_buffet_price*0.25,
            base_buffet_price*0.25,
            base_buffet_price*0.50,
            base_buffet_price*0.50,
            base_buffet_price,
            base_buffet_price,
            base_buffet_price,
            # base_buffet_price,
            ],
    'copter_ride': [
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            base_ride_price*0.5,
            # base_ride_price,
            ],
    }

dfp: pd.DataFrame = pd.DataFrame(
        data = ddict,
        index = [i for i in penalty_df_index],
        dtype=np.float32
        )


N_DAYS = 100.
MIN_OCCUPANCY = 125.
MAX_OCCUPANCY = 300.
DAY_RANGE = np.arange(1, N_DAYS + 1, dtype=np.float64)



# n_people
freq_index_cols = choice_cols.copy()
freq_index_cols.append('n_people')
# df_stack = df[freq_index_cols].stack().reset_index().rename(columns={'level_0':'family_id',0:'days_back_ct','level_1':'choice'})
# df_stack[['choice','days_back_ct']].groupby(['choice','days_back_ct']).size()

#-- Create dataframes for family_id and number of people
#-- and choice columns stacked
df3 = df.loc[:, ['family_id','n_people',]].copy()
df4 = df[choice_cols].stack().reset_index()
df4['level_0'] = df4['level_0'].astype(str) # Update level_0 col to str for join

#-- Merge dataframes to create a column of n_people
#-- Drop columns and reformat for readability
df5 = df4.merge(df3, how='left', left_on='level_0', right_on='family_id')
df5 = df5.drop(['level_0', 'family_id'], axis=1)
df5 = df5.rename(columns = {'level_1':'choice', 0:'n_days',})

# del df3


##################################
### Assignment Process ###

n_people_list = sorted(list(set(df['n_people'])), reverse=True) # [8, 7, 6, 5, 4, 3, 2]
n_people_list = [float(i) for i in n_people_list]

####- Working DataFrame from sample submission file.
dfs = dimp.import_csv_to_df('sample_submission') # Sample submission
dfs['family_id'] = dfs['family_id'].astype(str)
# working_df = df[['family_id', 'n_people']]
# working_df['assigned_day'] = 0.
working_df = dfs.loc[:, ['family_id', 'assigned_day']].copy()
working_df = working_df.merge(df.loc[:,['family_id','n_people',]], how='left', on='family_id')
working_df.loc[:, ['assigned_day', 'n_people']] = working_df.loc[:, ['assigned_day', 'n_people']].astype(float)




### Regrouping
df3 = df.loc[:, ['family_id','n_people',]].copy()
df6 = df4.merge(df3, how='left', left_on='level_0', right_on='family_id')
df6 = df6.drop('family_id', axis=1)
df6 = df6.rename(columns = {'level_0':'family_id','level_1':'choice', 0:'n_days',})
df6['cost'] = 0.
df6['choice_n'] = df6.loc[:,'choice'].str.split('_', expand=True)[1].astype(float)
for i in df6.select_dtypes(include=['int','int32','int64']):
    df6[i] = df6[i].astype(float)

# #-- Create working_df
# for idx in working_df.index:
#     day = randrange(0, 10)*1.0
#     tmp_df = df6[df6['family_id'] == str(idx)]
#     tmp_day = tmp_df.loc[tmp_df['choice_n'] == day, 'n_days'].values[0]
#     working_df.loc[idx, 'assigned_day'] = tmp_day


df6 = df6.drop('choice_n', axis=1)

for cc in choice_cols:
    df6.loc[df6['choice'] == cc, 'cost'] = df6.loc[df6['choice'] == cc, :].apply(lambda x: dfp.loc[cc, 'gift_card'] \
                   + (x['n_people'] * dfp.loc[cc, 'santas_buffet']) \
                   + (x['n_people'] * dfp.loc[cc, 'copter_ride']), axis=1)

cost_df = working_df.merge(df6, how='inner', left_on=['family_id','assigned_day',], right_on=['family_id', 'n_days',])
cost_df = cost_df.drop(['n_days','n_people_y'], axis=1).rename(columns={'n_people_x':'n_people',})

# df6[df6['family_id'] == '3519'].apply(lambda x:  x['n_days'] / x['cost'] if x['cost'] != 0.0 else 0.0, axis=1)
df6['day_cost'] = df6.apply(lambda x:  x['n_days'] * x['cost'], axis=1)



def acc_penalty_lhs(col):
    """Left-hand side of accounting penalty equation."""
    return np.divide((col - 125.), 400.)

def acc_penalty_rhs(col):
    """Right-hand side of accounting penalty equation."""
    if col.shift().isna().any():
        return np.power(col, (0.5 + np.divide(np.abs(col - col), 50.)))
    else:
        return np.power(col, (0.5 + np.divide(np.abs(col - col.shift()), 50.)))

def acc_penalty(df_col):
    """Agg function for accounting penalty."""
    return acc_penalty_lhs(df_col) * acc_penalty_rhs(df_col)

#-- Sort by index descending
day_totals = cost_df[['assigned_day','cost',]].groupby('assigned_day').sum().astype(float)
day_totals = day_totals.sort_index(ascending=False)

#--Calculate current accounting penalty.
accounting_penalty = round(day_totals.apply(lambda x: acc_penalty(x)).sum().values[0], 4)
preference_cost = day_totals.sum()
init_score = accounting_penalty + preference_cost

def df_partitioner(dataframe, column_name, column_value):
    return dataframe.loc[dataframe[column_name] == column_value, :]




df6_day_cost_agg = df6[['family_id','day_cost']].groupby('family_id').sum().reset_index()
df6_day_cost_agg = df6_day_cost_agg.rename(columns={'day_cost':'tot_day_cost'})
df6 = df6.merge(df6_day_cost_agg, how='inner', left_on='family_id', right_on='family_id')
df6.sort_values(by=['choice','n_days', 'n_people','cost','tot_day_cost',], ascending=[True, True, False, True, True,], inplace=True)
df6.sort_values(by=['tot_day_cost', 'choice'], ascending=[True, True], inplace=True)
# df6 = df6.drop('day_cost', axis=1)
# df6.head()


########### Testing algo
# #-- Current by day
# cost_df[cost_df['assigned_day']==1]
# cost_df[cost_df['assigned_day']==1]['n_people'].sum()
working_df['n_people'] = working_df['n_people'] * 1.0
n_people_list = np.array(sorted(working_df['n_people'].unique(), reverse=True))

try:
    working_df.drop('new_day', axis=1, inplace=True)
except KeyError:
    pass
working_df['new_day'] = 0.0
df_tst = df6.copy()

# no_cost_df = df6.loc[df6['cost'] == 0.0, :]
# no_cost_df.sort_values(by=['tot_day_cost',], ascending=[False,], inplace=True)
# # no_cost_df.loc[:,['n_people','n_days',]].groupby('n_days').sum().astype(float)


# working_df_2 = working_df.copy()
# working_df_2.drop('assigned_day', axis=1, inplace=True)

# no_cost_df = df6.loc[df6['cost'] == 0.0, :]
# no_cost_df.sort_values(by=['tot_day_cost',], ascending=[False,], inplace=True)
# # no_cost_df.loc[:,['n_people','n_days',]].groupby('n_days').sum().astype(float)


# df_tst.loc[df_tst['family_id'] == '0', :]
# day_totals = cost_df.loc[:, ['new_day','cost',]].groupby('new_day').sum().astype(float)


# people_df = df_partitioner(df_tst, 'n_people', 8.0)
# # people_df.sort_values(by=['n_days', 'choice','tot_day_cost',], ascending=[True, True, False,], inplace=True)

# choice_df = df_partitioner(people_df, 'choice', 'choice_0')
# choice_df.sort_values(by=['n_days', 'tot_day_cost',], ascending=[True,  False,], inplace=True)


# family_ids = choice_df['family_id'].values
# for fid in family_ids:
#     f_day = choice_df.loc[choice_df['family_id'] == fid, 'n_days'].values[0]
#     working_df_2.loc[working_df_2['family_id'] == fid, 'new_day'] = f_day
#     df_tst.drop(df_tst.loc[df_tst['family_id'] == fid,].index, inplace=True)



# choice_df = df_partitioner(df_tst, 'choice', 'choice_0')
# choice_df.sort_values(by=['n_days', 'tot_day_cost',], ascending=[True,  False,], inplace=True)


# people_df = df_partitioner(choice_df, 'n_people', 7.0)
# people_df.sort_values(by=['n_days', 'choice','tot_day_cost',], ascending=[True, True, False,], inplace=True)


# day_list = people_df.loc[:,'n_days'].unique()

# for dd in day_list:
#     #-- Current count of people for that day
#     day_people_ct = working_df_2[working_df_2['new_day'] == dd]['n_people'].sum()

#     #--  Running total of n_people starting from initial count
#     people_df['people_day_cum_ct'] = people_df[people_df['n_days'] == dd]['n_people'].cumsum() + day_people_ct

#     #-- Temp dataframe for cumulative totals lte MAX_OCCUPANCY
#     tmp_df = people_df[people_df['people_day_cum_ct'] <= MAX_OCCUPANCY].copy()

#     family_ids = tmp_df.loc[:,'family_id'].values
#     for fid in family_ids:
#         working_df_2.loc[working_df_2['family_id'] == fid, 'new_day'] = dd
#         df_tst.drop(df_tst.loc[df_tst['family_id'] == fid,].index, inplace=True)




working_df_2 = working_df.copy()
working_df_2 = working_df_2.drop('assigned_day', axis=1)

###-- Initial run --###
for cc in choice_cols: # choice_0, choice_1, choice_2, ... , choice_9
    choice_df = df_partitioner(df_tst, 'choice', cc)
    # choice_df.sort_values(by=['n_days', 'tot_day_cost',], ascending=[True,  False,], inplace=True)
    # choice_df.sort_values(by=['cost', 'tot_day_cost'], ascending=[False, False], inplace=True)
    choice_df.sort_values(by=['cost', 'tot_day_cost'], inplace=True)
    # print(f"df_tst rows: {df_tst.shape[0]}")
    for pp in n_people_list: # [8., 7., 6., 5., 4., 3., 2.]
        people_df = df_partitioner(choice_df, 'n_people', pp)
        # people_df.sort_values(by=['n_days', 'choice','tot_day_cost',], ascending=[True, True, False,], inplace=True)

        day_list = people_df.loc[:,'n_days'].unique()
        # for dd in DAY_RANGE:
        for dd in day_list:
            #-- Current count of people for that day
            day_people_ct = working_df_2[working_df_2['new_day'] == dd]['n_people'].sum()
        
            #--  Running total of n_people starting from initial count
            people_df['people_day_cum_ct'] = people_df[people_df['n_days'] == dd]['n_people'].cumsum() + day_people_ct
        
            #-- Temp dataframe for cumulative totals lte MAX_OCCUPANCY
            tmp_df = people_df[people_df['people_day_cum_ct'] <= MAX_OCCUPANCY].copy()

            if tmp_df.shape[0] > 0:
                print(f"Choice: {cc}, n_people: {pp}, day: {dd}")
                family_ids = tmp_df.loc[:,'family_id'].values
                for fid in family_ids:
                    working_df_2.loc[working_df_2['family_id'] == fid, 'new_day'] = dd
                    df_tst.drop(df_tst.loc[df_tst['family_id'] == fid,].index, inplace=True)



###-- Need to allocate unassigned families

day_totals = working_df_2.loc[working_df_2['new_day'] > 0.0, ['new_day','n_people',]].groupby('new_day').sum().astype(float)
family_ids = working_df_2.loc[working_df_2['new_day'] == 0., :]['family_id'].values
unassigned_family_df = df6.loc[df6['family_id'].isin(family_ids)]

# day_totals.head(20)

day_totals['avail_occupancy'] = MAX_OCCUPANCY - day_totals['n_people']
day_totals['below_min_occupancy'] = day_totals['n_people'] < MIN_OCCUPANCY


# # below minimum
# day_totals.loc[day_totals['below_min_occupancy']==True,:]

unassigned_family_df.loc[unassigned_family_df['choice']=='choice_1',:]
day_totals[day_totals.index == 59.]




#-- Evaluate cost
cost_df = working_df_2.merge(df6, how='inner', left_on=['family_id','new_day',], right_on=['family_id', 'n_days',])
cost_df = cost_df.drop(['n_days','n_people_y'], axis=1).rename(columns={'n_people_x':'n_people',})

#- People count
cost_people_ct = cost_df.loc[:, ['new_day','n_people',]].groupby('new_day').sum().astype(float)

#-- Costs per day
day_totals = cost_df.loc[:, ['new_day','cost',]].groupby('new_day').sum().astype(float)
day_totals = day_totals.sort_index(ascending=False)

#--Calculate current accounting penalty.
accounting_penalty = round(day_totals.apply(lambda x: acc_penalty(x)).sum().values[0], 4)
preference_cost = day_totals.sum()
base_score = accounting_penalty + preference_cost



# below minimum
below_min_df = day_totals.loc[day_totals['below_min_occupancy']==True,:].sort_index(ascending=False)






cost_df[['assigned_day','n_people',]].groupby('assigned_day').sum()

choices = [i for i in sample_df.columns if i.startswith('choice')]
sample_df = df.copy()
choices_df = sample_df[choices].stack()
choices_df = choices_df.reset_index().rename(columns={'level_0':'family_id','level_1':'choice', 0:'n_days',})
choices_df['family_id'] = choices_df['family_id'].astype(str)
fin_df = choices_df.merge(sample_df.loc[:,['family_id','n_people']], how='inner',left_on='family_id', right_on='family_id').copy()
fin_df['cost'] = 0.


for cc in choices:
    fin_df.loc[fin_df['choice'] == cc, 'cost'] = fin_df.loc[df6['choice'] == cc, :].apply(lambda x: dfp.loc[cc, 'gift_card'] \
                   + (x['n_people'] * dfp.loc[cc, 'santas_buffet']) \
                   + (x['n_people'] * dfp.loc[cc, 'copter_ride']), axis=1)






###################################
###-- Main df metrics for n_people
dfx = (df['n_people']
    .to_frame()
    .groupby('n_people')
    .size()
    .to_frame()
    .reset_index()
    .rename(columns={'n_people':'groups', 0:'n_people',})
    )

dfx['tot_people'] = dfx['groups'] * dfx['n_people']
dfx['cumul_tot_people'] = dfx['tot_people'].cumsum()



### Network diagram
import networkx as nx

G = nx.Graph()
# list(DAY_RANGE)

fin_df['weights'] = fin_df['n_days'] * fin_df['n_people']

w_edges = [(fin_df.loc[r, 'choice'], fin_df.loc[r, 'family_id'], fin_df.loc[r, 'weights']) for r in fin_df.index]

G.add_nodes_from(list(fin_df.loc[:, 'choice'].unique()))
G.add_weighted_edges_from(w_edges)

# # Edge data
# G['choice_0']['0']

pos = nx.spring_layout(G)
# nodes
nx.draw_networkx_nodes(G, pos)

# edges
nx.draw_networkx_edges(G, pos, width=1)

# # labels
# nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

plt.axis('off')
plt.show()















