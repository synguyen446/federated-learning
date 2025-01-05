import pandas as pd
import random

df_train = pd.read_csv(r"") # Your training data directory
df_test = pd.read_csv(r"") # Your testing data directory

sample_size = len(df_train)

for i in range(4):
    sample = df_train.sample(n=2000, random_state= 42)
    sample.to_csv(f"sample{i}.csv",index=False)
    df_train.drop(sample.index, inplace = True)
    if i == 3:
        test_sample = df_train.sample(n=2000,random_state=42)
        test_sample.to_csv(f"test_sample.csv",index=False)


