import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import scipy.stats as stats
import seaborn as sns


def read_data(path):
    return pd.read_csv(path, low_memory=False)


def replace_nan_with(current_el, new_el):
    if current_el is np.nan:
        return new_el
    else:
        return current_el
    

def transform_data(df):

    # b. Descoperirea și corectarea erorilor care au apărut din procedura de colectare
    print("\nAn fabricatie min: ", df["Anul fabricației"].min())
    print("An fabricatie max: ", df["Anul fabricației"].max())

    print("\nPret min: ", df["pret"].min())
    print("Pret max: ", df["pret"].max())

    print("\nkm min: ", df["Km"].min())
    print("km max: ", df["Km"].max())

    print("\nPutere min: ", df["Putere"].min())
    print("Putere max: ", df["Putere"].max())

    print("\nCapacitate cilindrica min: ", df["Capacitate cilindrica"].min())
    print("Capacitate cilindrica max: ", df["Capacitate cilindrica"].max())

    print("\nNumarul de valori nume per coloana:")
    print(df.isna().sum())

    print(df.shape)
    # print(df.dtypes)
    print(df.dtypes.value_counts())

    # urmatoarea bucata de cod o voi tine comentata pentru ca dureaza foarte mult rularea, am folosit-o
    # pentru identificarea erorilor, dar nu este critica pentru obtinerea rezultatelor de analiza
    # for col in df.select_dtypes('object').columns:
    #     if not isinstance(df[col].iloc[0], list):
    #         print(col, df[col].unique())

    #         for el in df[col]:
    #             if isinstance(el, list):
    #                 print(col, set(el))
    #             else:
    #                 print(el)
    #     else:
    #         print(col, df[col].unique())

    #     print(col, df[col].unique())


    # iterez prin variabilele te dip object si inlocuiesc valorile NaN cu "indisponibil" sau ["indisponibil"], dupa caz
    for col in df.select_dtypes('object').columns:
        if isinstance(df[col], list) or isinstance(df[col].iloc[0], list):
            df[col] = df[col].apply(replace_nan_with, args=(["indisponibil"],))
        else:
            df[col] = df[col].apply(replace_nan_with, args=("indisponibil",))            


    # iterez prin variabilele te dip float64 si int64 si inlocuiesc valorile NaN cu media valorilor pe coloana respectiva
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64' or df[col].dtype == 'int32':
            df[col].fillna(df[col].mean(), inplace=True)


    # este mai logic ca aceaste coloane sa fie de tipul int
    df['Numar locuri'] = df['Numar locuri'].astype(int)
    df['pret'] = df['pret'].astype(int)
    df['Km'] = df['Km'].astype(int)
    df['Putere'] = df['Putere'].astype(int)
    df['Capacitate cilindrica'] = df['Capacitate cilindrica'].astype(int)


    # c. Adăugarea sau eliminarea de coloane (acolo unde este cazul, de exemplu prin transformarea celor existente)
    # elimin coloanele cu un numar foarte mic de intrari sau nerelevante pentru analiza noastra
    
    df.drop("data", axis=1, inplace=True) 
    df.drop("url", axis=1, inplace=True) 
    df.drop("VIN (serie sasiu)", axis=1, inplace=True) # nu este nicio serie, doar un label 


    print(df.dtypes)



def main():

    # 1. Citirea și încărcarea datelor din fișierul la dispoziție
    df = read_data('auto_train.csv')
    
    # 2. Transformarea datelor
    df = transform_data(df)


    
if __name__ == "__main__":
    main()