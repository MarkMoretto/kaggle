
"""
Purpose: Sant's revenge 2019
Date created: 2020-01-14

https://www.kaggle.com/rengar12/kernel10a705a44f-b1a92a-50221a

Contributor(s):
    Mark M.
"""



home = True
from os import chdir, listdir
if home:
    proj_folder = r'C:\Users\Work1\Desktop\Info\kaggle\santas-workshop-2019'
else:
    proj_folder = r"C:\Users\MMorett1\Desktop\Projects Main\kaggle\santas-workshop-2019"

chdir(proj_folder)

import os.path
import cvxpy as cp
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
# df['family_id'] = df['family_id'].astype(str)
df.index = df['family_id'].values
df = df.drop('family_id', axis=1)
df[df.select_dtypes(include=['int64']).columns] = df[df.select_dtypes(include=['int64']).columns].astype(np.int32)


MIN_OCCUPANCY, MIN_OCCUPANCY, N_DAYS = np.int32(125), np.int32(300), np.int32(100)
DAY_RANGE = np.arange(1, N_DAYS + 1, dtype = np.int32)
DAY_RANGE_INV = np.arange(N_DAYS, 0, -1, dtype = np.int32)

choice_columns = [i for i in df.columns.values if 'choice' in i]


family_choices = df.loc[:,'choice_0':'choice_9'].to_dict('records')



g_daily_occupancy = {k:0 for k in DAY_RANGE_INV}
def get_daily_occupancy(day, n_people):
    if (g_daily_occupancy[day] + n_people) <= 200:
        g_daily_occupancy[day] += n_people
        return True
    else:
        return False



def cost_function(prediction):
    penalty = 0

    #переменная, подсчитывающая число посетителей для каждого дня
    daily_occupancy = {k:0 for k in days}
    
    #f - id семьи, d - день, назначенный для этой семьи
    for f, d in enumerate(prediction):

       #n - размер рассматриваемой семьи 
        n = family_size_dict[f]
        daily_occupancy[d] += n
        
        #определяется, каким приоритетом назначенный день является для данной семьи
        d_name = get_key(family_choices[f], d)

        #считаются расходы в зависимости от приоритета
        if d_name == 'choice_0':
            penalty += 0
        elif d_name == 'choice_1':
            penalty += 50
        elif d_name == 'choice_2':
            penalty += 50 + 9 * n
        elif d_name == 'choice_3':
            penalty += 100 + 9 * n
        elif d_name == 'choice_4':
            penalty += 200 + 9 * n
        elif d_name == 'choice_5':
            penalty += 200 + 18 * n
        elif d_name == 'choice_6':
            penalty += 300 + 18 * n
        elif d_name == 'choice_7':
            penalty += 300 + 36 * n
        elif d_name == 'choice_8':
            penalty += 400 + 36 * n
        elif d_name == 'choice_9':
            penalty += 500 + 36 * n + 199 * n
        else:
            penalty += 500 + 36 * n + 398 * n

    #проверяется, сответствует ли число посетителей рамкам, если нет - расход увеличивается настолько, чтобы превосходить любой рабочий вариант
    for _, v in daily_occupancy.items():
        if (v > max_occ) or (v < min_occ):
            penalty += 100000000
            
    #подсчет дополнительных расходов
    #так как в формуле для n-го дня используется значение для n+1-го, то день 100 подсчитывается отдельно
    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)
    accounting_cost = max(0, accounting_cost)

    #подсчет для остальных дней с использованием значения предыдущего дня
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))
        yesterday_count = today_count

    penalty += accounting_cost

    return penalty, daily_occupancy



#считаем расходы для стартового решения
best = submission['assigned_day'].tolist()
start_score = cost_function(best)[0]
print(start_score)
new = best.copy()





