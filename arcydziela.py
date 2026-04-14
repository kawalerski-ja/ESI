import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Modele, których użyjemy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

warnings.filterwarnings('ignore')

print("=== ETAP 1: PRZYGOTOWANIE DANYCH ===")
katalog_skryptu = os.path.dirname(os.path.abspath(__file__))
sciezka_do_pliku = os.path.join(katalog_skryptu, "auction_results_color_svd.csv")
df = pd.read_csv(sciezka_do_pliku)

# One-Hot Encoding
zmienne_kategoryczne = ['ARTIST', 'TECHNIQUE', 'SIGNATURE', 'CONDITION']
bloki_kategoryczne = [np.eye(np.max(df[z].astype('category').cat.codes) + 1)[df[z].astype('category').cat.codes] for z in zmienne_kategoryczne]
X_kat_cale = np.hstack(bloki_kategoryczne)

# Podział Train/Test
np.random.seed(42)
df_train = df.sample(frac=0.8, random_state=42)
df_test = df.drop(df_train.index)

X_kat_train = X_kat_cale[df_train.index]
X_kat_test = X_kat_cale[df_test.index]

# Normalizacja numeryczna
zmienne_liczbowe = ["TOTAL DIMENSIONS", "YEAR", "Colorfulness Score", "SVD Entropy"]
srednia = df_train[zmienne_liczbowe].mean()
odchylenie = df_train[zmienne_liczbowe].std()

X_num_train = ((df_train[zmienne_liczbowe] - srednia) / odchylenie).to_numpy(dtype=np.float32)
X_num_test = ((df_test[zmienne_liczbowe] - srednia) / odchylenie).to_numpy(dtype=np.float32)

X_train = np.hstack([X_kat_train, X_num_train], dtype=np.float32)
X_test = np.hstack([X_kat_test, X_num_test], dtype=np.float32)

# --- TRZY WERSJE ZMIENNEJ DOCELOWEJ ---
# 1. Surowa (W Dolarach)
y_train_raw = df_train['PRICE'].to_numpy(dtype=np.float32)
y_test_raw = df_test['PRICE'].to_numpy(dtype=np.float32)

# 2. Zlogarytmizowana
y_train_log = np.log(y_train_raw)
y_test_log = np.log(y_test_raw)

# 3. Wyliczenie progu dla Arcydzieł (Top 3%)
prog_top_3 = np.percentile(y_test_raw, 97)

# Funkcja do szybkiego raportowania
def ewaluacja(nazwa_modelu, y_pred_dolary):
    mae = mean_absolute_error(y_test_raw, y_pred_dolary)
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_dolary))
    r2 = r2_score(y_test_raw, y_pred_dolary)
    
    maska_arcydziel = y_test_raw > prog_top_3
    mae_arcydziela = mean_absolute_error(y_test_raw[maska_arcydziel], y_pred_dolary[maska_arcydziel])
    
    print(f"[{nazwa_modelu}]")
    print(f"  R^2 Ogólne:       {r2:.4f}")
    print(f"  MAE Ogólne:       {mae:.2f} $")
    print(f"  RMSE Ogólne:      {rmse:.2f} $")
    print(f"  MAE ARCYDZIEŁ:    {mae_arcydziela:.2f} $")
    print("-" * 50)


print(f"\nPróg cenowy dla arcydzieł: {prog_top_3:.2f} $\n")
print("=== ETAP 2: TRENING SPECJALISTYCZNYCH MODELI ===\n")

# =========================================================
# METODA 1: MODEL DWUETAPOWY (Divide and Conquer)
# =========================================================
# Krok A: Klasyfikator ocenia czy obraz to arcydzieło (używamy wag 'balanced' aby zauważył te 3%)
prog_train_3 = np.percentile(y_train_raw, 97)
y_train_klasy = (y_train_raw > prog_train_3).astype(int)

klasyfikator = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
klasyfikator.fit(X_train, y_train_klasy)

# Krok B: Dwa osobne regresory
reg_tanie = RandomForestRegressor(n_estimators=100, random_state=42)
reg_drogie = RandomForestRegressor(n_estimators=100, random_state=42)

reg_tanie.fit(X_train[y_train_klasy == 0], y_train_log[y_train_klasy == 0])
# Model drogi trenuje się TYLKO na arcydziełach!
reg_drogie.fit(X_train[y_train_klasy == 1], y_train_log[y_train_klasy == 1])

# Krok C: Predykcja
pred_klasy_test = klasyfikator.predict(X_test)
y_pred_2stage_log = np.zeros(len(X_test))

y_pred_2stage_log[pred_klasy_test == 0] = reg_tanie.predict(X_test[pred_klasy_test == 0])
if np.sum(pred_klasy_test == 1) > 0:
    y_pred_2stage_log[pred_klasy_test == 1] = reg_drogie.predict(X_test[pred_klasy_test == 1])

ewaluacja("Metoda 1: Model Dwuetapowy (Klasyfikacja + Regresja)", np.exp(y_pred_2stage_log))


# =========================================================
# METODA 2: REGRESJA KWANTYLOWA (Gradient Boosting)
# =========================================================
# Alpha=0.85 oznacza, że każemy modelowi celować w 85. percentyl cen, 
# a nie w ich średnią, wymuszając na nim "optymizm" i wyższe wyceny.
model_kwantylowy = GradientBoostingRegressor(loss='quantile', alpha=0.85, n_estimators=100, random_state=42)
model_kwantylowy.fit(X_train, y_train_log)

y_pred_kwantyl = np.exp(model_kwantylowy.predict(X_test))
ewaluacja("Metoda 2: Regresja Kwantylowa (Percentyl 85%)", y_pred_kwantyl)


# =========================================================
# METODA 3: REGRESJA POISSONA DLA DŁUGIEGO OGONA 
# =========================================================
# UWAGA: Rozkład Poissona natywnie modeluje "długie ogony".
# Wrzucamy mu SUROWE CENY w dolarach (brak logarytmizacji!), a on sam sobie z nimi radzi.
model_poisson = HistGradientBoostingRegressor(loss='poisson', max_iter=100, random_state=42)
model_poisson.fit(X_train, y_train_raw)

y_pred_poisson = model_poisson.predict(X_test)
ewaluacja("Metoda 3: HistGradientBoosting (Rozkład Poissona)", y_pred_poisson)