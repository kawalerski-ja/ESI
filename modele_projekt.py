import pandas as pd
import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("================================================================")
    print("CZĘŚĆ 1: SKRÓTOWY OPIS WYBRANYCH METOD UCZENIA MASZYNOWEGO")
    print("================================================================\n")
    print("1. Regresja Ridge (L2): Jest to rozszerzenie klasycznej regresji liniowej o tzw. regularyzację L2. Polega na dodaniu do funkcji straty kary za zbyt duże wartości wag modelu. Zapobiega to przeuczeniu (overfitting) i sprawia, że model jest bardziej odporny na współliniowość zmiennych objaśniających. Kluczowym parametrem jest tu siła regularyzacji (alpha).")
    print("2. Drzewo Decyzyjne (Decision Tree Regressor): Algorytm oparty na strukturze drzewa, w którym zbiór danych jest rekurencyjnie dzielony na mniejsze podzbiory na podstawie warunków. Liście drzewa reprezentują końcową predykcję. Metoda ta jest łatwa w interpretacji, ale podatna na przeuczenie, dlatego istotne jest kontrolowanie maksymalnej głębokości drzewa.")
    print("3. Las Losowy (Random Forest Regressor): Należy do metod zespołowych (ensemble learning). Buduje wiele niezależnych drzew decyzyjnych na różnych podzbiorach danych uczących i z wykorzystaniem losowych podzbiorów cech. Wynik końcowy to uśredniona predykcja ze wszystkich drzew. Ważnym parametrem jest liczba drzew w lesie.")
    print("4. K-Najbliższych Sąsiadów (KNN): Metoda oparta na odległości. Nie buduje jawnego modelu podczas treningu (tzw. lazy learning). Aby przewidzieć wartość dla nowej próbki, algorytm szuka w zbiorze uczącym K najbardziej do niej podobnych próbek i uśrednia ich wartości docelowe. Skuteczność zależy od wyboru odpowiedniej liczby sąsiadów (K).\n")

    print("================================================================")
    print("CZĘŚĆ 2: PRZYGOTOWANIE DANYCH")
    print("================================================================\n")
    
    # 1. Wczytanie danych
    print("Wczytywanie pliku 'auction_results_color_svd.csv'...")
    try:
        df = pd.read_csv("auction_results_color_svd.csv")
    except FileNotFoundError:
        print("BŁĄD: Nie znaleziono pliku 'auction_results_color_svd.csv' w bieżącym katalogu.")
        return

    # 2. Zmienne kategoryczne i podział na zbiory
    print("Przetwarzanie zmiennych kategorycznych i podział na zbiór uczący/testowy...")
    zmienne_kategoryczne = ['ARTIST', 'TECHNIQUE', 'SIGNATURE', 'CONDITION']
    bloki_kategoryczne = []
    
    for zmienna in zmienne_kategoryczne:
        kody = df[zmienna].astype('category').cat.codes.to_numpy(dtype=np.int32)
        liczba_klas = np.max(kody) + 1
        one_hot = np.eye(liczba_klas)[kody]
        bloki_kategoryczne.append(one_hot)

    X_kat_cale = np.hstack(bloki_kategoryczne)

    np.random.seed(42)
    df_train = df.sample(frac=0.8, random_state=42)
    df_test = df.drop(df_train.index)

    indeksy_train = df_train.index
    indeksy_test = df_test.index

    X_kat_train = X_kat_cale[indeksy_train]
    X_kat_test = X_kat_cale[indeksy_test]

    # 3. Normalizacja zmiennych liczbowych
    print("Normalizacja zmiennych liczbowych...")
    zmienne_liczbowe = ["TOTAL DIMENSIONS", "YEAR", "Colorfulness Score", "SVD Entropy"]
    srednia = df_train[zmienne_liczbowe].mean()
    odchylenie = df_train[zmienne_liczbowe].std()

    df_train[zmienne_liczbowe] = (df_train[zmienne_liczbowe] - srednia) / odchylenie
    df_test[zmienne_liczbowe] = (df_test[zmienne_liczbowe] - srednia) / odchylenie

    # 4. Przejście na tablice NumPy
    X_num_train = df_train[zmienne_liczbowe].to_numpy(dtype=np.float32)
    X_num_test = df_test[zmienne_liczbowe].to_numpy(dtype=np.float32)

    X_train = np.hstack([X_kat_train, X_num_train], dtype=np.float32)
    X_test = np.hstack([X_kat_test, X_num_test], dtype=np.float32)

    srednia_cena = df_train['PRICE'].mean()
    odchylenie_cena = df_train['PRICE'].std()

    y_train = ((df_train['PRICE'] - srednia_cena) / odchylenie_cena).to_numpy(dtype=np.float32).reshape(-1, 1)
    y_test = ((df_test['PRICE'] - srednia_cena) / odchylenie_cena).to_numpy(dtype=np.float32).reshape(-1, 1)

    # Funkcja do odwracania normalizacji
    def prawdziwa_cena(znormalizowana_cena):
        return znormalizowana_cena * odchylenie_cena + srednia_cena

    print("\n================================================================")
    print("CZĘŚĆ 3: ANALIZA WPŁYWU PARAMETRÓW NA SKUTECZNOŚĆ MODELI")
    print("================================================================\n")

    # Spłaszczamy rzeczywiste ceny testowe do ostatecznej ewaluacji
    y_test_rzeczywiste = prawdziwa_cena(y_test.ravel())

    # Definiujemy modele i parametry do przebadania
    analizy_modeli = [
        {
            "nazwa": "Regresja Ridge",
            "model_klasa": Ridge,
            "parametr_nazwa": "alpha",
            "parametr_wartosci": [0.1, 1.0, 10.0, 100.0]
        },
        {
            "nazwa": "Drzewo Decyzyjne",
            "model_klasa": DecisionTreeRegressor,
            "parametr_nazwa": "max_depth",
            "parametr_wartosci": [5, 10, 20, 50]
        },
        {
            "nazwa": "Las Losowy",
            "model_klasa": RandomForestRegressor,
            "parametr_nazwa": "n_estimators",
            "parametr_wartosci": [10, 50, 100, 200]
        },
        {
            "nazwa": "K-Najbliższych Sąsiadów",
            "model_klasa": KNeighborsRegressor,
            "parametr_nazwa": "n_neighbors",
            "parametr_wartosci": [3, 5, 10, 20]
        }
    ]

    for analiza in analizy_modeli:
        nazwa = analiza["nazwa"]
        model_klasa = analiza["model_klasa"]
        parametr_nazwa = analiza["parametr_nazwa"]
        parametr_wartosci = analiza["parametr_wartosci"]
        
        print(f"--- Model: {nazwa} ---")
        print(f"Badany parametr: {parametr_nazwa}")
        print("-" * 65)
        
        for wartosc in parametr_wartosci:
            kwargs = {parametr_nazwa: wartosc}
            
            # Zapewnienie powtarzalności dla modeli opartych na drzewach
            if nazwa in ["Drzewo Decyzyjne", "Las Losowy"]:
                kwargs["random_state"] = 42
                
            model = model_klasa(**kwargs)
            
            # Trening modelu
            start_time = time.time()
            model.fit(X_train, y_train.ravel())
            czas_treningu = time.time() - start_time
            
            # Predykcja
            y_pred_znormalizowane = model.predict(X_test)
            y_pred_rzeczywiste = prawdziwa_cena(y_pred_znormalizowane)
            
            # Ewaluacja
            mae = mean_absolute_error(y_test_rzeczywiste, y_pred_rzeczywiste)
            rmse = np.sqrt(mean_squared_error(y_test_rzeczywiste, y_pred_rzeczywiste))
            r2 = r2_score(y_test_rzeczywiste, y_pred_rzeczywiste)
            
            # Wypisywanie wyników
            print(f"[{parametr_nazwa} = {wartosc:<4}] | R^2: {r2:6.4f} | MAE: {mae:7.2f} | RMSE: {rmse:7.2f} | Czas: {czas_treningu:.4f} s")
            
        print("\n")

if __name__ == "__main__":
    main()