import os
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
# Wyłączenie logów TF dla czytelności konsoli
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

print("=== ETAP 1: PRZYGOTOWANIE DANYCH (Lustrzane do modelu finalnego) ===")
katalog_skryptu = os.path.dirname(os.path.abspath(__file__))
sciezka_do_pliku = os.path.join(katalog_skryptu, "auction_results_color_svd.csv")
df = pd.read_csv(sciezka_do_pliku)

# 1. One-Hot Encoding zmiennych kategorycznych
zmienne_kategoryczne = ['ARTIST', 'TECHNIQUE', 'SIGNATURE', 'CONDITION']
bloki_kategoryczne = [np.eye(np.max(df[z].astype('category').cat.codes) + 1)[df[z].astype('category').cat.codes] for z in zmienne_kategoryczne]
X_kat_cale = np.hstack(bloki_kategoryczne)

# 2. Podział na Train/Test (80/20)
np.random.seed(42)
df_train = df.sample(frac=0.8, random_state=42)
df_test = df.drop(df_train.index)

X_kat_train = X_kat_cale[df_train.index]
X_kat_test = X_kat_cale[df_test.index]

# 3. Normalizacja zmiennych liczbowych
zmienne_liczbowe = ["TOTAL DIMENSIONS", "YEAR", "Colorfulness Score", "SVD Entropy"]
srednia = df_train[zmienne_liczbowe].mean()
odchylenie = df_train[zmienne_liczbowe].std()

X_num_train = ((df_train[zmienne_liczbowe] - srednia) / odchylenie).to_numpy(dtype=np.float32)
X_num_test = ((df_test[zmienne_liczbowe] - srednia) / odchylenie).to_numpy(dtype=np.float32)

# 4. Łączenie i usuwanie SVD Entropy (zgodnie z modelem finalnym)
X_train = np.hstack([X_kat_train, X_num_train], dtype=np.float32)[:, :-1]
X_test = np.hstack([X_kat_test, X_num_test], dtype=np.float32)[:, :-1]

# 5. Transformacja Logarytmiczna ceny
y_train_log = np.log(df_train['PRICE']).to_numpy(dtype=np.float32).reshape(-1, 1)
y_test_log = np.log(df_test['PRICE']).to_numpy(dtype=np.float32).reshape(-1, 1)
prawdziwe_ceny_test = df_test['PRICE'].to_numpy(dtype=np.float32)

print(f"Dane gotowe. Liczba cech wejściowych: {X_train.shape[1]}")

print("\n=== ETAP 2: KONFIGURACJA I TRENING TENSORFLOW ===")
# Budowa modelu Keras o identycznej architekturze co autorska sieć
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='linear')
])

# Używamy identycznego optymalizatora SGD z LR=0.01
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse')

start_time = time.time()
# Trening z identycznymi parametrami
model.fit(X_train, y_train_log, epochs=150, batch_size=64, verbose=0)
czas_treningu = time.time() - start_time

print(f"Trening TF zakończony w czasie: {czas_treningu:.2f} s")

print("\n=== ETAP 3: KOMPLEKSOWA EWALUACJA (BENCHMARK) ===")
# Predykcja (logarytm) i powrót do dolarów
y_pred_log = model.predict(X_test, verbose=0).ravel()
y_pred_dolary = np.exp(y_pred_log)

# --- Obliczanie metryk ---
mae = mean_absolute_error(prawdziwe_ceny_test, y_pred_dolary)
rmse = np.sqrt(mean_squared_error(prawdziwe_ceny_test, y_pred_dolary))
r2 = r2_score(prawdziwe_ceny_test, y_pred_dolary)

# sMAPE (identyczny wzór jak w autorskim kodzie)
licznik = np.abs(prawdziwe_ceny_test - y_pred_dolary)
mianownik = (np.abs(prawdziwe_ceny_test) + np.abs(y_pred_dolary)) / 2.0
smape = np.mean(licznik / (mianownik + 1e-8)) * 100

# --- Test Arcydzieł (Top 3%) ---
prog_top_3 = np.percentile(prawdziwe_ceny_test, 97)
maska_arcydziel = prawdziwe_ceny_test > prog_top_3
mae_arcydziela = mean_absolute_error(prawdziwe_ceny_test[maska_arcydziel], y_pred_dolary[maska_arcydziel])

# --- Raport końcowy ---
print("-" * 50)
print(f"WYNIKI TENSORFLOW (Benchmark):")
print("-" * 50)
print(f"1. Współczynnik R^2:          {r2:10.4f}")
print(f"2. sMAPE (Błąd procentowy):   {smape:10.2f} %")
print(f"3. MAE (Średnia pomyłka):     {mae:10.2f} $")
print(f"4. RMSE (Kara za ekstrema):   {rmse:10.2f} $")
print(f"5. Czas treningu:             {czas_treningu:10.2f} s")
print("-" * 50)
print(f"6. MAE ARCYDZIEŁ (Top 3%):    {mae_arcydziela:10.2f} $")
print("-" * 50)