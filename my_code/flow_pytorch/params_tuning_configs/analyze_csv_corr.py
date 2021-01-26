import pandas as pd
import csv
# rs = pd.DataFrame.from_csv(r'D:/Clustering_TOP.csv',encoding='utf-8')
df = pd.read_csv("/home/tarask/Desktop/Work/Code/Git/probabilistic-gesticulator/hparam_search/svito_zar_real_hparams_search_hparam_search_jan_26.csv",encoding='utf-8')

# print(df.corr())

print("\nCorrelation with jerk values")


for column in df.columns:
    if column == "Name" or column == "Optim/name":
        continue
    print(column, " corr: ", df['jerk/generated_mean_ratio'].corr(df[column]))

print("\nCorrelation with training loss")

for column in df.columns:
    if column == "Name" or column == "Optim/name":
        continue
    print(column, " corr: ", df['Loss/train'].corr(df[column]))

print("Done!")