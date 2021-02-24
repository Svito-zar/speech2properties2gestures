import pandas as pd
import csv

# rs = pd.DataFrame.from_csv(r'D:/Clustering_TOP.csv',encoding='utf-8')
#df = pd.read_csv("/home/tarask/Desktop/Work/Code/Git/probabilistic-gesticulator/hparam_search/svito_zar_real_hparams_search_hparam_search_jan_27.csv",encoding='utf-8')
#df = pd.read_csv("/home/tarask/Desktop/Work/Code/Git/probabilistic-gesticulator/hparam_search/svito_zar_new_hparams_search_hparam_search_Feb_2.csv",encoding='utf-8')

#df = pd.read_csv("/home/tarask/Documents/Experimental_Results/GestureFlow/PredictingGestureProperties/HparamsSearch/svito_zar_prop_predict_Feb12.csv")

df = pd.read_csv("/home/tarask/Documents/Experimental_Results/GestureFlow/PredictingGestureProperties/HparamsSearch/R_Pratice_n_Phrase/hparams_r_practice_n_phrase_dil_Feb17th.csv", encoding='utf-8')


# print(df.corr())

print("\nCorrelation with F1 for phrase values")


for column in df.columns:
    if column == "Name" or column == "Optim/name":
        continue
    print(column, " corr: ", df['F1/phrase_av'].corr(df[column]))

#exit()

print("\nCorrelation with val loss")

for column in df.columns:
    if column == "Name" or column == "Optim/name":
        continue
    print(column, " corr: ", df['Loss/val_loss'].corr(df[column]))

print("Done!")