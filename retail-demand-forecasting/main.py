
"""
Purpose: Stackoverflow q
Date created: 2021-03-01

URL: https://stackoverflow.com/questions/66422332/how-to-divide-dataset-and-plot#66422332
Kaggle: https://www.kaggle.com/c/demand-forecasting-kernels-only/overview

Contributor(s):
    Mark M.
"""


from pathlib import Path

import pandas as pd

folder = Path(r"C:\Users\Work1\Desktop\Info\kaggle\retail-demand-forecasting\data")

df = pd.read_csv(folder.joinpath("train.csv"))
df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"])
df.loc[:, ["store", "item", "sales"]] = df.loc[:, ["store", "item", "sales"]].astype(int)

df["month"] =