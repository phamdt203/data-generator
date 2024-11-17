import pandas as pd

from ctgan.synthesizers import CTGAN

df = pd.read_csv("DSV_DHTD_full.csv")
print(df.columns)
print(df.head(5))
# discrete_columns = [
#
# ]
# model = CTGAN(epochs=10)
# model.fit(df)

