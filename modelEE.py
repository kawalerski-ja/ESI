import pandas as pd
import numpy as np
import time
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf

warnings.filterwarnings('ignore')

print("=== ETAP 1: PRZYGOTOWANIE DANYCH (KLASYCZNE MODELE) ===")
df = pd.read_csv("auction_results_color_svd.csv")

# Szybki Preprocessing (taki sam jak w autorskiej sieci)
zmienne_kategoryczne = ['ARTIST', 'TECHNIQUE', 'SIGNATURE', 'CONDITION']
bloki_kategoryczne = [np.eye(np.max(df[z].astype('category').cat.codes) + 1)[df[z].astype('category').cat.codes] for z in zmienne_kategoryczne]
X_kat_cale = np.hstack(bloki_kategoryczne)

np.random.seed(42)
df_train = df.sample(frac=0.8, random_state=42)
df_test = df.drop(df_train.index)

X_kat_train = X_kat_cale[df_train.index]
X_kat_test = X_kat_cale[df_test.index]

zmienne_liczbowe = ["TOTAL DIMENSIONS", "YEAR", "Colorfulness Score", "SVD Entropy"]
srednia = df_train[zmienne_liczbowe].mean()
odchylenie = df_train[zmienne_liczbowe].std()

X_num_train = ((df_train[zmienne_liczbowe] - srednia) / odchylenie).to_numpy(dtype=np.float32)
X_num_test = ((df_test[zmienne_liczbowe] - srednia) / odchylenie).to_numpy(dtype=np.float32)

X_train = np.hstack([X_kat_train, X_num_train], dtype=np.float32)
X_test = np.hstack([X_kat_test, X_num_test], dtype=np.float32)

srednia_cena = df_train['PRICE'].mean()
odchylenie_cena = df_train['PRICE'].std()

y_train = ((df_train['PRICE'] - srednia_cena) / odchylenie_cena).to_numpy(dtype=np.float32).ravel()
y_test_prawdziwe = df_test['PRICE'].to_numpy(dtype=np.float32)

def prawdziwa_cena(znormalizowana_cena):
    return znormalizowana_cena * odchylenie_cena + srednia_cena

# ==========================================
# TRENING BAZOWYCH MODELI
# ==========================================
print("\n=== ETAP 2: TRENING I BENCHMARK ===")

modele = {
    "Regresja Liniowa": LinearRegression(),
    "Las Losowy": RandomForestRegressor(n_estimators=100, random_state=42)
}

wyniki = {}

for nazwa, model in modele.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    czas = time.time() - start_time
    
    y_pred = prawdziwa_cena(model.predict(X_test))
    
    mae = mean_absolute_error(y_test_prawdziwe, y_pred)
    r2 = r2_score(y_test_prawdziwe, y_pred)
    
    wyniki[nazwa] = y_pred
    print(f"[{nazwa}] MAE: {mae:.2f} $ | R^2: {r2:.4f} | Czas: {czas:.2f}s")

# TensorFlow
print("\nUruchamiam TensorFlow (Keras)...")
tf.keras.utils.disable_interactive_logging() # żeby nie śmieciło konsoli
model_tf = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model_tf.compile(optimizer='adam', loss='mse')

start_time = time.time()
model_tf.fit(X_train, y_train, epochs=100, batch_size=256, verbose=0)
czas_tf = time.time() - start_time

y_pred_tf = prawdziwa_cena(model_tf.predict(X_test).ravel())
mae_tf = mean_absolute_error(y_test_prawdziwe, y_pred_tf)
r2_tf = r2_score(y_test_prawdziwe, y_pred_tf)
wyniki['TensorFlow'] = y_pred_tf

print(f"[TensorFlow] MAE: {mae_tf:.2f} $ | R^2: {r2_tf:.4f} | Czas: {czas_tf:.2f}s")

# ==========================================
# ANALIZA ARCYDZIEŁ DLA GOTOWYCH MODELI
# ==========================================
print("\n=== ETAP 3: BENCHMARK ARCYDZIEŁ (Top 3%) ===")
prog_top_3 = np.percentile(y_test_prawdziwe, 97)
maska_arcydziel = y_test_prawdziwe > prog_top_3
prawdziwe_arcydziela = y_test_prawdziwe[maska_arcydziel]

print(f"Analizujemy {len(prawdziwe_arcydziela)} obrazów wycenianych powyżej {prog_top_3:.2f} $\n")

for nazwa, predykcje in wyniki.items():
    predykcje_arcydziel = predykcje[maska_arcydziel]
    mae_arcydziela = mean_absolute_error(prawdziwe_arcydziela, predykcje_arcydziel)
    print(f"{nazwa:20} -> Średni błąd dla arcydzieł: {mae_arcydziela:.2f} $")

print("\nWniosek: Zobacz, czy potężny Las Losowy lub TensorFlow radzą sobie z arcydziełami lepiej niż Twoja lekka sieć z pliku autorskiego!")