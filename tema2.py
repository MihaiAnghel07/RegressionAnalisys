import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor



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

    # Descoperirea și corectarea erorilor care au apărut din procedura de colectare

    # Afisez diferite informatii despre date pentru a descoperii anomalii, pe care le corectez ulterior
    print("\nAn fabricatie min: ", df["Anul fabricației"].min())
    print("An fabricatie max: ", df["Anul fabricației"].max())

    print("\nPret min: ", df["pret"].min())
    print("Pret max: ", df["pret"].max())

    df = df[(df['pret'] > 2000) & (df['pret'] < 300000)]

    print("\nkm min: ", df["Km"].min())
    print("km max: ", df["Km"].max())

    print("\nPutere min: ", df["Putere"].min())
    print("Putere max: ", df["Putere"].max())

    df = df[(df['Putere'] > 40) & (df['Putere'] < 700)]


    print("\nCapacitate cilindrica min: ", df["Capacitate cilindrica"].min())
    print("Capacitate cilindrica max: ", df["Capacitate cilindrica"].max())

    print("Are VIN (Serie sasiu): ", df["Are VIN (Serie sasiu)"].unique())

    # Afisez valorile unice pentru a afla daca tipul de date sau valorile sunt consistente
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

    # In functie de numarul de valori nule pe o coloana, voi elimina toata coloana sau 
    # o voi completa cu date 
    print("\nNumarul de valori nule per coloana:")
    print(df.isna().sum())

    # In functie de feature, inlocuiesc valorile NaN cu valori medii sau valori random culese de pe aceeasi coloana
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


    # este mai logic ca aceste coloane sa fie de tipul int
    df['Numar locuri'] = df['Numar locuri'].astype(int)
    df['pret'] = df['pret'].astype(int)
    df['Km'] = df['Km'].astype(int)
    df['Putere'] = df['Putere'].astype(int)
    df['Capacitate cilindrica'] = df['Capacitate cilindrica'].astype(int)


    # Adăugarea sau eliminarea de coloane (acolo unde este cazul, de exemplu prin transformarea celor existente)
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

    # Fac o verificare finala pentru a ma asigura ca datele sunt corectate dupa cum ma astept
    print("\nAfter: ")
    print(df.shape)
    print(df.dtypes.value_counts())

    print("\nNumarul de valori nule per coloana:")
    print(df.isna().sum())
    
    return df


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


def crossval_rmse(model, n_folds, df, x, y):
    # creez n fold-uri de cross-validare in mod aleator
    kf = KFold(n_folds, shuffle=True).get_n_splits(df.values)

    # evaluez eroarea medie patratica sub radical pentru cele n folduri, folosind datele de input (antrenare sau testare)
    rmse = np.sqrt(-1 * cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=kf))
    
    return rmse


def run_regression(model, x_train, y_train, x_test, y_test, df, model_name):
    print("\n********************************************************")
    print("Training and testing of: {}".format(model_name))

    # antrenez modelul pe datele de antrenare
    model.fit(x_train, y_train)

    # calculez predictiile pentru datele de antrenare si testare
    x_train_pred = model.predict(x_train)
    x_test_pred = model.predict(x_test)
    
    # Definesc numarul de fold-uri pe care le face "crossval_rmse". Aceasta functie creeaza n fold-uri 
    # de cross validare in mod aleatoriu, returnand eroarea medie patratica sub radical pentru cele n fold-uri
    n_folds = 3
    print('\nTrain RMSE: {}'.format(crossval_rmse(model, n_folds, df, x_train, y_train).mean()))
    print('Test RMSE: {}'.format(crossval_rmse(model, n_folds, df, x_test, y_test).mean()))

    # trasez graficul predictiei modelului
    plt.scatter(x_train_pred, y_train, c = "blue",  label = "Training data")
    plt.scatter(x_test_pred, y_test, c = "black",  label = "Validation data")
    plt.title(model_name)
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc = "upper left")
    plt.plot([x_train_pred.min(), x_train_pred.max()], [x_train_pred.min(), x_train_pred.max()], c = "red")
    plt.show()

    # Evaluez score-ul modelului
    score = model.score(x_test, y_test)
    print("\nScore of {}: {}\n".format(model_name, score))

    return x_test_pred, score


def get_best_hyperparams(model, initial_hyperparams, x_train, y_train):
    # folosesc GridSearch pentru a gasi cei mai buni hiperparametrii
    grid_search = GridSearchCV(model, initial_hyperparams, cv=3)

    # antrenez modelul
    grid_search.fit(x_train, y_train)

    # afisez cei mai buni hiperparametri obtinuti
    print("\nGridSearch - best hyperparams:", grid_search.best_params_)

    # afisez cel mai bun scor obtinut
    print("GridSearch - best score:", grid_search.best_score_)
    
    return grid_search.best_estimator_


def save_predictions_to_csv(predictions, ids, filename):
    # creez un dataframe care sa respecte formatul id, value, folosind datele de input
    solution_df = pd.DataFrame({'id': ids, 'value': predictions})

    # salvez dataframe-ul in fisierul primit ca input
    solution_df.to_csv(filename, index=False)


def main():

    # Citirea și încărcarea datelor din fișierul la dispoziție
    df = read_data('auto_train.csv')
    
    # Transformarea datelor
    df = transform_data(df)


    # Stocarea datelor

    # plotez graficele Pearson si Spearman pentru a vedea gradul de corelatie dintre pret si alte categorii
    # plot_pearson_spearman(df)

    # Pastrez acele categorii cu grad ridicat de corelatie
    # Pentru y am obtinut rezultate mai bune aplicand o transformare logaritmica asupra pretului
    x = df[['Anul fabricației', 'Km', 'Putere', 'Capacitate cilindrica', 'Consum Extraurban', 'Consum Urban']]
    y = np.log1p(df['pret'])

    # impart setul de date: 80% antrenare si 20% testare
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    # Antrenarea si evaluarea modelelor de regresie
    ridge_initial_hyperparams = {'alphas': [0.1, 1.0, 30.0]}
    decision_tree_initial_hyperparams = {'max_depth': [None, 10, 20]}
    random_forest_initial_hyperparams = {'n_estimators': [50, 100, 250]}
    extra_trees_initial_hyperparams = {'n_estimators': [50, 100, 300]}
    ada_boost_initial_hyperparams = {'n_estimators': [50, 100, 200]}
    gradient_boosting_initial_hyperparams = {'n_estimators': [50, 100, 2500]}

    linear_model = LinearRegression()
    solution_lm, score_lm = run_regression(linear_model, x_train, y_train, x_test, y_test, df, "Linear regression")

    ridge_model = RidgeCV()
    best_ridge_model = get_best_hyperparams(ridge_model, ridge_initial_hyperparams, x_train, y_train)
    solution_rm, score_rm = run_regression(best_ridge_model, x_train, y_train, x_test, y_test, df, "Ridge regression")

    decision_tree_model = DecisionTreeRegressor()
    best_decision_tree_model = get_best_hyperparams(decision_tree_model, decision_tree_initial_hyperparams, x_train, y_train)
    solution_dt, score_dt = run_regression(best_decision_tree_model, x_train, y_train, x_test, y_test, df, "DecisionTree regression")

    random_forest_model = RandomForestRegressor()
    best_random_forest_model = get_best_hyperparams(random_forest_model, random_forest_initial_hyperparams, x_train, y_train)
    solution_rf, score_rf = run_regression(best_random_forest_model, x_train, y_train, x_test, y_test, df, "RandomForest regression")

    extra_trees_model = ExtraTreesRegressor()
    best_extra_trees_model = get_best_hyperparams(extra_trees_model, extra_trees_initial_hyperparams, x_train, y_train)
    solution_et, score_et = run_regression(best_extra_trees_model, x_train, y_train, x_test, y_test, df, "ExtraTrees regression")

    ada_boost_model = AdaBoostRegressor()
    best_ada_boost_model = get_best_hyperparams(ada_boost_model, ada_boost_initial_hyperparams, x_train, y_train)
    solution_ab, score_ab = run_regression(best_ada_boost_model, x_train, y_train, x_test, y_test, df, "AdaBoost regression")

    gradient_boosting_model = GradientBoostingRegressor()
    best_gradient_boosting_model = get_best_hyperparams(gradient_boosting_model, gradient_boosting_initial_hyperparams, x_train, y_train)
    solution_gb, score_gb = run_regression(best_gradient_boosting_model, x_train, y_train, x_test, y_test, df, "GradientBoosting regression")

    # salvez doar solutia cu score-ul cel mai bun
    best_score = max(score_lm, score_rm, score_dt, score_rf, score_et, score_ab, score_gb)

    if best_score == score_lm:
        best_solution = solution_lm
    elif best_score == score_rm:
        best_solution = solution_rm
    elif best_score == score_dt:
        best_solution = solution_dt
    elif best_score == score_rf:
        best_solution = solution_rf
    elif best_score == score_et:
        best_solution = solution_et
    elif best_score == score_ab:
        best_solution = solution_ab
    else:
        best_solution = solution_gb

    save_predictions_to_csv(np.exp(best_solution), df.loc[x_test.index, 'id'], 'solution.csv')


    
if __name__ == "__main__":
    main()