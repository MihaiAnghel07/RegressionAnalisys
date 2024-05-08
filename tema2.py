import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


def read_data(path):
    return pd.read_csv(path, low_memory=False)


def replace_nan_with(current_el, new_el):
    if current_el is np.nan:
        return new_el
    else:
        return current_el


def replace_nan_with_random(df, column):
    valori_nan = df[column].isna()
    numar_nan = valori_nan.sum()
    valori_aleatoare = df[column].dropna().sample(numar_nan, replace=True)
    df.loc[valori_nan, column] = valori_aleatoare.values


def generate_random_da_nu_values(n):
    return np.random.choice(['Da', 'Nu'], size=n)


def replace_nan_with_random_da_nu_values(df, column):
    valori_nan = df[column].isna()
    numar_nan = valori_nan.sum()
    valori_aleatoare = generate_random_da_nu_values(numar_nan)
    df.loc[valori_nan, column] = valori_aleatoare



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

    print("Are VIN (Serie sasiu): ", df["Are VIN (Serie sasiu)"].unique())

    print("Putere: ", df["Putere"].unique())
    print("Capacitate cilindrica: ", df["Capacitate cilindrica"].unique())
    print("Transmisie: ", df["Transmisie"].unique())
    print("Consum Extraurban: ", df["Consum Extraurban"].unique())
    print("Cutie de viteze: ", df["Cutie de viteze"].unique())
    print("Consum Urban: ", df["Consum Urban"].unique())
    print("Emisii CO2: ", df["Emisii CO2"].unique())
    print("Numar de portiere: ", df["Numar de portiere"].unique())
    print("Numar locuri: ", df["Numar locuri"].unique())
    print("Se emite factura: ", df["Se emite factura"].unique())
    print("Eligibil pentru finantare: ", df["Eligibil pentru finantare"].unique())
    print("Primul proprietar (de nou): ", df["Primul proprietar (de nou)"].unique())
    print("Carte de service: ", df["Carte de service"].unique())
    print("Audio si tehnologie: ", df["Audio si tehnologie"].unique())
    print("Generatie: ", df["Generatie"].unique())
    print("Norma de poluare: ", df["Norma de poluare"].unique())
    print("Optiuni culoare: ", df["Optiuni culoare"].unique())
    print("Tara de origine: ", df["Tara de origine"].unique())
    print("Data primei inmatriculari: ", df["Data primei inmatriculari"].unique())
    print("Performanta: ", df["Performanta"].unique())
    print("Inmatriculat: ", df["Inmatriculat"].unique())


    print("\nNumarul de valori nule per coloana:")
    print(df.isna().sum())

    df["Are VIN (Serie sasiu)"] = df["Are VIN (Serie sasiu)"].apply(replace_nan_with, args=("Nu",))
    replace_nan_with_random(df, "Versiune")
    df["Km"].fillna(df["Km"].mean(), inplace=True)
    df["Putere"].fillna(df["Putere"].mean(), inplace=True)
    df["Capacitate cilindrica"].fillna(df["Capacitate cilindrica"].mean(), inplace=True)
    replace_nan_with_random(df, "Transmisie")
    df["Consum Extraurban"].fillna(df["Consum Extraurban"].mean(), inplace=True)
    replace_nan_with_random(df, "Cutie de viteze")
    df["Consum Urban"].fillna(df["Consum Urban"].mean(), inplace=True)
    df["Emisii CO2"].fillna(df["Emisii CO2"].mean(), inplace=True)
    replace_nan_with_random(df, "Numar de portiere")
    replace_nan_with_random(df, "Numar locuri")
    df["Se emite factura"].fillna("Nu", inplace=True)
    df["Eligibil pentru finantare"].fillna("Nu", inplace=True)
    df["Primul proprietar (de nou)"].fillna("Nu", inplace=True)
    df["Carte de service"].fillna("Nu", inplace=True)
    df["Fara accident in istoric"].fillna("Da", inplace=True)
    replace_nan_with_random(df, "Audio si tehnologie")
    replace_nan_with_random(df, "Confort si echipamente optionale")
    replace_nan_with_random(df, "Electronice si sisteme de asistenta")
    replace_nan_with_random(df, "Siguranta")
    replace_nan_with_random(df, "Generatie")
    replace_nan_with_random(df, "Norma de poluare")
    replace_nan_with_random(df, "Optiuni culoare")
    replace_nan_with_random(df, "Tara de origine")
    replace_nan_with_random(df, "Data primei inmatriculari")
    replace_nan_with_random(df, "Performanta")
    replace_nan_with_random_da_nu_values(df, "Inmatriculat")



    # km: media
    # putere: medie + eliminare outliers 
    # Capacitate cilindrica: medie + eliminare utliers
    # Transmisise: inlocuire NaN cu ceva random din valorile existente in coloana
    # Consum Extraurban: medie 
    # Cutie de viteze: inlcouire nan cu random
    # Consum urban: medie
    # Emisii CO2: medie
    # Numar de portiere: 5 sau random
    # Numar de locuri: 5 sau random
    # se emite factura: Nu
    # Eligibil pentru finantare: Nu
    # Primul proprietar: Nu
    # Carte de service: Nu
    # Audio si tehnologie: random
    # Confort si echipamente optionale: random
    # Electronice si sisteme de asistenta: random
    # Siguranta: random
    # ?Generatie: random
    # Norma de poluare: random
    # Optiuni culoare: random
    # Tara de origine: random
    # Data primei inmatriculari: random
    # Performanta: random
    # Inmatriculat: random Da / Nu

    # este mai logic ca aceaste coloane sa fie de tipul int
    df['Numar locuri'] = df['Numar locuri'].astype(int)
    df['pret'] = df['pret'].astype(int)
    df['Km'] = df['Km'].astype(int)
    df['Putere'] = df['Putere'].astype(int)
    df['Capacitate cilindrica'] = df['Capacitate cilindrica'].astype(int)


    # c. Adăugarea sau eliminarea de coloane (acolo unde este cazul, de exemplu prin transformarea celor existente)
    # elimin coloanele cu un numar foarte mic de intrari sau nerelevante pentru analiza noastra (au peste 16000 de intrari de 
    # tip NaN dintr-un total de 19526 intrari)
    
    df.drop("data", axis=1, inplace=True) 
    df.drop("url", axis=1, inplace=True) 
    df.drop("VIN (serie sasiu)", axis=1, inplace=True) # nu este nicio serie, doar un label 
    df.drop("sau in limita a", axis=1, inplace=True)
    df.drop("Garantie dealer (inclusa in pret)", axis=1, inplace=True)
    df.drop("Garantie de la producator pana la", axis=1, inplace=True)
    df.drop("Vehicule electrice", axis=1, inplace=True)
    df.drop("Tuning", axis=1, inplace=True)
    df.drop("Contract baterie", axis=1, inplace=True)
    df.drop("Masina de epoca", axis=1, inplace=True)
    df.drop("Volan pe dreapta", axis=1, inplace=True)
    df.drop("Predare leasing", axis=1, inplace=True)
    df.drop("Plata initiala (la predare)", axis=1, inplace=True)
    df.drop("Autonomie", axis=1, inplace=True)
    df.drop("Consum mediu", axis=1, inplace=True)
    df.drop("Capacitate baterie", axis=1, inplace=True)
    df.drop("Valoare rata lunara", axis=1, inplace=True)
    df.drop("Timp de incarcare", axis=1, inplace=True)
    df.drop("Numar de rate lunare ramase", axis=1, inplace=True)
    df.drop("Valoare reziduala", axis=1, inplace=True)
    df.drop("Consum Mixt", axis=1, inplace=True)



    print("\nAfter: ")
    print(df.shape)
    print(df.dtypes)
    print(df.dtypes.value_counts())

    print("\nNumarul de valori nule per coloana:")
    print(df.isna().sum())



def main():

    # 1. Citirea și încărcarea datelor din fișierul la dispoziție
    df = read_data('auto_train.csv')
    
    # 2. Transformarea datelor
    df = transform_data(df)


    
if __name__ == "__main__":
    main()