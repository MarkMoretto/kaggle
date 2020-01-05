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
"""

### Change folder to project directory

from os import chdir, listdir
proj_folder: str
# proj_folder = r"C:\Users\MMorett1\Desktop\Projects Main\kaggle\santas-workshop-2019"
proj_folder = r'C:\Users\Work1\Desktop\Info\kaggle\santas-workshop-2019'
chdir(proj_folder)

import os.path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from zipfile import ZipFile, is_zipfile
sns.set()
#-- Set a few pandas options
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 1000)
pd.set_option('mode.chained_assignment', None) # Silence `SettingWithCopyWarning`



xmas_day: dt.datetime.date = dt.datetime(year=2019, month=12, day=25).date()
today: dt.datetime.date = dt.datetime.today().date()

def days_between(dt1: dt.datetime.date, dt2: dt.datetime.date) -> np.int32:
    """Find net days between two dates."""
    return np.abs((dt1 - dt2).days)


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


    def get_zipfolder(self) -> str:
        """Search for zipfile in directory. Return full path."""
        zp_fldr: str = [os.path.join(self.project_folder, i) for i \
                        in listdir(self.project_folder) \
                        if str(i).endswith('zip')]
    
        if len(zp_fldr) > 1:
            print('Multiple zip files found.')
        else:
            output: str = zp_fldr[0]
            if is_zipfile(output):
                self.zipfile_path = output
            else:
                print(f"Zipfile '{output}' not valid.")


    def import_csv_to_df(self, csv_filename: str = 'family_data') -> pd.DataFrame:
        filename: str
        zf_list: list

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
df: pd.DataFrame
dimp = DataImporter(proj_folder)
df = dimp.import_csv_to_df()
# df = df.drop('family_id', axis=1)
df['family_id'] = df['family_id'].astype(str)

# #-- Column not included in data set
# additional_col: str = 'otherwise'

# #-- Add additional column with default value of zero
# df.insert(df.shape[1]-1, additional_col, np.float32(0))

choice_cols: list = [i for i in df.columns.values if 'choice' in i]
n_choice_cols: np.array = np.arange(len(choice_cols), dtype=np.int8)
choice_weights: np.array = np.arange(1, len(choice_cols) + 1, dtype=np.int8)[::-1]

### Add column that sums each day choice
# Multiply by choice_weights array, sum the columns, multiple by n_people
df['choice_bias'] = (df[choice_cols].multiply(choice_weights).sum(axis=1) * df['n_people'])
# df.drop('choice_bias', axis=1, inplace=True)


df = df.sort_values(by=choice_cols)
df.iloc[:50,:].sort_values(by=['n_people','choice_sum'], ascending=[False, True],)


### Sum


### Evaluation ###
#-- Create penalty dataframe to help crunch numbers

penalty_df_index: list = choice_cols.copy()
penalty_df_index.append(additional_col)

#-- Base prices for santa's buffet and helicopter ride
base_buffet_price: np.float32 = 36.0
base_ride_price: np.float32 = 398.0

ddict: dict = {
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
            500.,
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
            base_buffet_price,
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
            base_ride_price,
            ],
    }

dfp: pd.DataFrame = pd.DataFrame(
        data = ddict,
        index = [i for i in penalty_df_index],
        dtype=np.float32
        )


N_DAYS: np.float16 = 100.
MIN_OCCUPANCY: np.float16 = 125.
MAX_OCCUPANCY: np.float16 = 300.
DAY_RANGE: np.array = np.arange(1, N_DAYS + 1)



# n_people
freq_index_cols: list = choice_cols.copy()
freq_index_cols.append('n_people')
# df_stack = df[freq_index_cols].stack().reset_index().rename(columns={'level_0':'family_id',0:'days_back_ct','level_1':'choice'})
# df_stack[['choice','days_back_ct']].groupby(['choice','days_back_ct']).size()

#-- Create dataframes for family_id and number of people
#-- and choice columns stacked
df3 = df.loc[:, ['family_id','n_people',]].copy()
df4 = df[choice_cols].stack().reset_index()

#-- Merge dataframes to create a column of n_people
#-- Drop columns and reformat for readability
df5 = df4.merge(df3, how='left', left_on='level_0', right_on='family_id')
df5 = df5.drop(['level_0', 'family_id'], axis=1)
df5 = df5.rename(columns = {'level_1':'choice', 0:'n_days',})

del df3, df4

#-- Apply aggregate functions to get count of families per day and total number
#-- of people
agg_dict = {'n_days': 'size', 'n_people': 'sum'}
dfx = (df5.groupby(['choice','n_days', 'n_people'])
       .agg(agg_dict)
       .rename(columns={'n_days':'day_ct','n_people':'tot_people',})
       .reset_index()
       )
dfx = dfx.sort_values(by='n_people', ascending=False).sort_values(by=['choice','n_days',])




#-- Sample and apply some costs
df3 = dfx[((dfx['n_people'] == 8) & (dfx['choice'] == 'choice_1'))].copy()

df3.apply(lambda x, choice='choice_0': \
          dfp.loc[choice, 'gift_card'] + \
          (x['n_people'] * dfp.loc[choice, 'santas_buffet']) + \
          (x['n_people'] * dfp.loc[choice, 'copter_ride'])
          , axis=1)






##################################
### Create occupancy dataframe ###
# np.linalg.multi_dot()

# df_occ: pd.DataFrame = pd.DataFrame(index=DAY_RANGE)
# df_occ['family_id'] = np.float32(0)
# df_occ['assigned_day'] = np.float32(0)
# df_occ['n_people'] = np.float32(0)
# df_occ['tot_people'] = np.float32(0)
# # df_occ['min_cap'] = np.float16(MIN_OCCUPANCY)
# # df_occ['max_cap'] = np.float16(MAX_OCCUPANCY)
# df_occ['is_full'] = False

final_df = df['family_id']
final_df = final_df.to_frame('family_id')
final_df['assigned_day'] = np.float32(0.)




df_x = df.sort_values(by='n_people', ascending=False).sort_values()
n_people_lst: list = list(set(df['n_people']))
day_total = 0.
for d in DAY_RANGE: # (1, 100)
    for n in n_people_lst:
        for f in df_x.loc[((df_x['n_people'] == n)&(df_x['choice_0'] == 1)), ['family_id','n_people']]:
            day_total += df_x.loc






df6 = df_x.loc[((df_x['n_people'] == 8)&(df_x['choice_0'] == 1)), ['family_id','n_people']]


df6['n_people'].cumsum()
n_people_lst: list = list(set(df['n_people']))

for n in n_people_lst:
    for i in DAY_RANGE:
        tmp_df = df_x.loc[((df_x['n_people'] == n)&(df_x['choice_0'] == i)), ['family_id','n_people']]
        ppl_ct = tmp_df['n_people'].sum()

df_x[df_x['choice_0'] == 1]








#-- Who has the most family members?
df8 = df[df['n_people'] == 8].sort_values(by='choice_0')
df8 = df8['choice_0'].value_counts()




dfp[['santas_buffet','copter_ride']] * df.loc[0, 'n_people']

df.loc[0].apply(lambda x: dfp['gift_card'] + (x['n_people'] * dfp['santas_buffet']) + (x['n_people'] * dfp['copter_ride']))

### Datetime index
start_dt = xmas_day - dt.timedelta(days=N_DAYS)
dt_idx = pd.date_range(start=start_dt, end=xmas_day - dt.timedelta(1))


# ### Freq counts of days back
# df[choice_cols].stack().reset_index(drop=True).value_counts().astype(int)

xyz = df[choice_cols].groupby(choice_cols).size()
abc = xyz.reset_index()
abc = abc.drop(0, axis=1)

xyz = df.groupby(choice_cols).size()



(df['choice_0']
    .value_counts()
    .reset_index()
    .rename(columns={'index':'n_days'})
    .sort_values(by='n_days')
    )




df.loc[0]


#-- What is the family member distribution?
people_freq = df['n_people'].value_counts().astype(int)

def plot_n_people():
    people_freq_df = people_freq.reset_index()
    fig, _ = plt.subplots()
    ax = sns.scatterplot(x='index', y='n_people', data=people_freq_df)
    ax.set_xlabel('N Members')
    ax.set_ylabel('Count')
    ax.set_title('Family Member Count Plot')
    ax.grid(True)
    plt.show()
# plot_n_people()


## Cost optimization
# https://towardsdatascience.com/scheduling-with-ease-cost-optimization-tutorial-for-python-c05a5910ee0d
#-- Worst-case buffet and ride costs
people_freq.apply(lambda x: x * dfp['santas_buffet'])
people_freq.apply(lambda x: x * dfp['copter_ride'])



#-- Who has the most family members?
df8 = df[df['n_people'] == 8].sort_values(by='choice_0')
df8 = df8['choice_0'].value_counts().copy()


df2 = pd.DataFrame(
        {'c':[1,1,2,2,3,3],
         'L0':['a','a','b','c','d','e'],
         'L1':['a','b','c','e','f','e']}
        )

pd.crosstab(df2['c'], df2['L0'])
pd.crosstab(df2['c'], df2['L1'])





### Sample dataframe
dfx = df.iloc[:10,:]
# dfx.sort_values(by='choice_0', ascending=False)



dfx['base_date'] = xmas_day
# Freq counts
df[choice_cols].stack().reset_index(drop=True).value_counts().astype(int)


dfx['base_date'] - pd.to_timedelta(dfx['choice_0'], unit='D')
dfx.apply(lambda x: x['base_date'] - pd.to_timedelta(x['choice_0'], unit='D'))

















