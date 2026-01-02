import pandas as pd

df = pd.read_excel("data/esaso_eval/cyst.xlsx")
print(df.loc[96])
confirm = df["!"]
df_new = df.loc[96,:]

df.loc[:96,'nomefile'] = "balvit8835_" + df[:96]["nomefile"]
df.loc[96,'nomefile'] = df_new['nomefile']
print(df.loc[96])
df.to_excel("data/esaso_eval/cyst.xlsx", index=False)
