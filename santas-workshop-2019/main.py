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
proj_folder: str = r"C:\Users\MMorett1\Desktop\Projects Main\kaggle\santas-workshop-2019"
chdir(proj_folder)

import os.path
import numpy as np
import pandas as pd
import datetime as dt
from zipfile import ZipFile, is_zipfile

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
dimp = DataImporter(proj_folder)
df = dimp.import_csv_to_df()

choice_cols: list = [i for i in df.columns.values if 'choice' in i]


### Evaluation ###
#-- Create penalty dataframe to help crunch numbers

penalty_df_index: list = choice_cols.copy()
penalty_df_index.append('otherwise')

#-- Base prices for santa's buffet and helicopter ride
base_buffet_price: float = 36.0
base_ride_price: float = 398.0

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
    'helicopter_ride': [
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


N_DAYS: np.int32 = 100
MIN_OCCUPANCY: np.int32 = 125
MAX_OCCUPANCY: np.int32 = 300

### Datetime index
start_dt = xmas_day - dt.timedelta(days=N_DAYS)
dt_idx = pd.date_range(start=start_dt, end=xmas_day - dt.timedelta(1))


### Sample dataframe
dfx = df.iloc[:10,:]
dfx.sort_values(by='choice_0', ascending=False)


dfx[choice_cols].apply(lambda x: dfx['base_date'] - dt.timedelta(days=1)*x)


dfx['base_date'] = xmas_day
dfx['base_date'] - pd.to_timedelta(dfx['choice_0'], unit='D')
dfx.apply(lambda x: x['base_date'] - pd.to_timedelta(x['choice_0'], unit='D'))