import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from typing import Any, Dict, List


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

    # # Selectați caracteristicile categorice pentru a le codifica one-hot
    # categorical_features = df.select_dtypes(include=['object']).columns
    
    # # Aplicați codificarea one-hot pentru fiecare caracteristică categorică
    # df_encoded = pd.get_dummies(df[categorical_features], columns=categorical_features)
    
    # # Concatenați caracteristicile numerice inițiale cu caracteristicile categorice transformate
    # df_final = pd.concat([df.drop(columns=categorical_features), df_encoded], axis=1)

    # object_columns = df.select_dtypes(include=['object']).columns
    # for col in object_columns:
    #     # Obțineți un dicționar care mapează fiecare valoare unică la un număr
    #     unique_values = df[col].unique()
    #     value_map = {value: index for index, value in enumerate(unique_values)}
    #     # Înlocuiți valorile cu cele mapate la forme numerice
    #     df[col] = df[col].replace(value_map)
    
    return df


def df_to_single_col(df: pd.DataFrame, column_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
  features = df[[column_name]]
  target = df['pret']
  
  return features, target


def leq_range_buckets(df: pd.DataFrame, column: str, bucket_ranges: List[Any]) -> Dict[Any, pd.DataFrame]:
    buckets = {}

    # filter dataset rows
    for i, bucket_range in enumerate(bucket_ranges):
        if i == 0:
          buckets[f"<{bucket_range}"] = df[df[column] <= bucket_range]
        else:
          buckets[f"<{bucket_range}"] = df[(df[column] > bucket_ranges[i-1]) & (df[column] <= bucket_range)]
    
    # sum up the length of all the buckets
    total = sum([len(buckets[bucket]) for bucket in buckets])
    if total != len(df):
        # create another bucket to store remaining examples
        buckets[f">{bucket_ranges[-1]}"] = df[df[column] > bucket_ranges[-1]]

    buckets["total"] = df
    return buckets


def make_kfolds(buckets: dict[str, pd.DataFrame], n_splits: int = 5, shuffle: bool = False) -> Dict[str, KFold]:
  kfolds = {}
  for bucket_name, bucket_df in buckets.items():
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    kfolds[bucket_name] = (kf.split(bucket_df))
  return kfolds


def get_train_test_folds(buckets: dict[str, pd.DataFrame], kfolds: dict[str, KFold], n_folds: int) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
  train_folds: Dict[str, List[pd.DataFrame]] = {}
  test_folds: Dict[str, List[pd.DataFrame]] = {}
  
  bucket_names = set(buckets.keys())
  bucket_names.remove("total")
  for bucket_name in bucket_names:
    train_folds[bucket_name]: List[pd.DataFrame] = []
    test_folds[bucket_name]: List[pd.DataFrame] = []
    for idx, (train_idx, val_idx) in enumerate(kfolds[bucket_name]):
      train_indices = buckets[bucket_name].iloc[train_idx]
      test_indices = buckets[bucket_name].iloc[val_idx]
      train_folds[bucket_name].append(train_indices)
      test_folds[bucket_name].append(test_indices)
  
  # from each bucket name, concatenate the same index from each fold
  train_folds_concat: List[pd.DataFrame] = []
  test_folds_concat: List[pd.DataFrame] = []
  for idx in range(n_folds):
    train_folds_concat.append(pd.concat([train_folds[bucket_name][idx] for bucket_name in bucket_names]))
    test_folds_concat.append(pd.concat([test_folds[bucket_name][idx] for bucket_name in bucket_names]))
  
  return train_folds_concat, test_folds_concat


def plot_pearson_spearman(df):
    # pentru a observa daca exista corelatie intre variabilele numerice, trasam doua grafice: Pearson si Spearman
    # pastram doar corelatiile relevante de peste 0.4

    fig, ax = plt.subplots(1, 2, figsize=(30, 10), dpi=100)
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['font.size'] = 6
    numeric_columns = df.select_dtypes(include=['float64', 'int64', 'int32']).columns
    numeric_df = df[numeric_columns]
    for i, corr_type in enumerate(['pearson', 'spearman']):
        corr_df = numeric_df.corr(corr_type)
        corr_df = corr_df - np.diag(np.diag(corr_df))
        corr_df = corr_df[corr_df > 0.4]
        corr_df = corr_df.dropna(axis=0, how='all')
        corr_df = corr_df.dropna(axis=1, how='all')
        sns.heatmap(corr_df, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5, linecolor='black', square=False, cbar=True, cbar_kws={'orientation': 'vertical', 'shrink': 0.8, 'pad': 0.05}, ax=ax[i], mask=corr_df.isnull())
        ax[i].set_title(corr_type, fontsize=20)
    plt.tight_layout()
    plt.show()


def main():

    # 1. Citirea și încărcarea datelor din fișierul la dispoziție
    df = read_data('auto_train.csv')
    
    # 2. Transformarea datelor
    df = transform_data(df)
    df = df.head(18000)

    # 3. Stocarea datelor
    # x = df.drop(columns=['pret'])
    train_buckets = leq_range_buckets(df, "pret", [15000, 50_000])
    for bucket in train_buckets:
        print(
        f"`Bucket: {bucket}` contains {len(train_buckets[bucket])} samples."
        f" Percentage of total: {len(train_buckets[bucket]) / len(df):.2%}"
        )
    

    # kfolds = make_kfolds(train_buckets)
    n_folds = 3
    kfolds = make_kfolds(train_buckets, n_splits=n_folds, shuffle=False)
    # print(f"Number of buckets: {len(kfolds)}")
    # for bucket_name, bucket_kfolds in kfolds.items():
    #     print(f"Bucket: {bucket_name}")
    #     for idx, (train_idx, val_idx) in enumerate(bucket_kfolds):
    #         print(f"Fold: {idx}")
    #         print(f"Training indices: {train_idx}")
    #         print(f"Validation indices: {val_idx}")
    #         print()


    train_dfs, test_dfs = get_train_test_folds(train_buckets, kfolds, n_folds)
    print(f"Number of folds: {len(train_dfs)}")
    for idx, (train_df, test_df) in enumerate(zip(train_dfs, test_dfs)):
        print(f"Fold: {idx}")
        print(f"Training shape: {train_df.shape}")
        print(f"Testing shape: {test_df.shape}")
        print()


    # x = df[['Anul fabricației', 'Km', 'Putere', 'Capacitate cilindrica', 'Consum Extraurban', 'Consum Urban']]
    # y = df['pret']
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    # plot_pearson_spearman(df)

    for idx, (train_df, test_df) in enumerate(zip(train_dfs, test_dfs)):
        print(f"Fold: {idx}")

        fig, ax = plt.subplots(1, 3, figsize=(10, 3), dpi=150)
        for col_idx, demo_column in enumerate(("Putere", "Anul fabricației", "Capacitate cilindrica")):
            x_train, y_train = df_to_single_col(train_df, demo_column)
            x_test, y_test = df_to_single_col(test_df, demo_column)

            model = LinearRegression()
            # model = DecisionTreeRegressor()

            model.fit(x_train, y_train)

            train_score = model.score(x_train, y_train)
            test_score = model.score(x_test, y_test)

            print("Training R^2 score:", train_score)
            print("Test R^2 score:", test_score)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Create a range of x values to plot the predicted function
            x_range = np.linspace(x_train.min(), x_train.max(), 100).reshape(-1, 1)

            # Predict the y values for the range of x values
            y_range_pred = model.predict(x_range)


            # Plot the predicted function as a line
            ax[col_idx].plot(x_range, y_range_pred, color='red', lw=3, label='Predicted function')
            ax[col_idx].scatter(x_test, y_test_pred, label='Test set prices')
            ax[col_idx].scatter(x_train, y_train, label='Training set prices', alpha=0.35, color="orange")
            ax[col_idx].set_xlabel(demo_column)
            ax[col_idx].set_ylabel('Pret')
            ax[col_idx].set_title(f"Linear Regression on {demo_column}")
            ax[col_idx].grid()
            ax[col_idx].legend()
        
        plt.show()
        break # we notice the results are not so great...

    # model = LinearRegression()

    # # Antrenarea modelului
    # model = LinearRegression()
    # model.fit(x_train, y_train)

    # # Evaluarea modelului
    # score = model.score(x_test, y_test)
    # print("Scorul modelului:", score)


    
if __name__ == "__main__":
    main()