import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
df=pd.read_csv("./fully_labeled_cognitive_dataset.csv")
print(df.dtypes)