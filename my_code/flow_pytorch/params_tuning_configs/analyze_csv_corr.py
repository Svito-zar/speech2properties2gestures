import pandas as pd
import csv

# rs = pd.DataFrame.from_csv(r'D:/Clustering_TOP.csv',encoding='utf-8')
#df = pd.read_csv("/home/tarask/Desktop/Work/Code/Git/probabilistic-gesticulator/hparam_search/svito_zar_real_hparams_search_hparam_search_jan_27.csv",encoding='utf-8')
#df = pd.read_csv("/home/tarask/Desktop/Work/Code/Git/probabilistic-gesticulator/hparam_search/svito_zar_new_hparams_search_hparam_search_Feb_2.csv",encoding='utf-8')

#df = pd.read_csv("/home/tarask/Documents/Experimental_Results/GestureFlow/PredictingGestureProperties/HparamsSearch/svito_zar_prop_predict_Feb12.csv")

#df = pd.read_csv("/home/tarask/Documents/Experimental_Results/GestureFlow/PredictingGestureProperties/HparamsSearch/R_Pratice_n_Phrase/hparams_r_practice_n_phrase_dil_Feb17th.csv", encoding='utf-8')

df = pd.read_csv("/home/tarask/Documents/Experimental_Results/GestureFlow/PredictingGestureProperties/HparamsSearch/RightGesSemantic/With_CB_loss/Hparams_G_Semant_Feb_25.csv")

print(df.columns)

print("\nCorrelation with F1 for phrase values")


for column in df.columns:
    if column == "Name" or column == "Optim/name":
        continue
    print(column, " corr: ", df['F1/semantic__av'].corr(df[column]))

#exit()

print("\nCorrelation with accuracy")

for column in df.columns:
    if column == "Name" or column == "Optim/name":
        continue
    print(column, " corr: ", df['Acc/semantic__av'].corr(df[column]))

print("Done!")