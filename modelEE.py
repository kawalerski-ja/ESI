import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf

warnings.filterwarnings('ignore')
tf.keras.utils.disable_interactive_logging()

print("=== ETAP 1: PRZYGOTOWANIE DANYCH ===")
katalog_skryptu = os.path.dirname(os.path.abspath(__file__))
sciezka_do_pliku = os.path.join(katalog_skryptu, "auction_results_color_svd.csv")
df = pd.read_csv(sciezka_do_pliku)

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

# Logarytmizacja również dla klasycznych modeli
y_train_log = np.log(df_train['PRICE']).to_numpy(dtype=np.float32).ravel()
y_test_log = np.log(df_test['PRICE']).to_numpy(dtype=np.float32).ravel()

# Prawdziwe ceny do finalnych obliczeń błędów
prawdziwe_ceny_test = df_test['PRICE'].to_numpy(dtype=np.float32)

print("\n=== ETAP 2: TRENING I BENCHMARK ===")
modele = {
    "Regresja Liniowa": LinearRegression(),
    "Las Losowy": RandomForestRegressor(n_estimators=100, random_state=42)
}

wyniki = {}

for nazwa, model in modele.items():
    start_time = time.time()
    model.fit(X_train, y_train_log) # Uczymy na logarytmach
    czas = time.time() - start_time
    
    # Przewidujemy logarytm i odwracamy funkcją exp()
    y_pred = np.exp(model.predict(X_test))
    
    mae = mean_absolute_error(prawdziwe_ceny_test, y_pred)
    r2 = r2_score(prawdziwe_ceny_test, y_pred)
    wyniki[nazwa] = y_pred
    print(f"[{nazwa:16}] MAE: {mae:8.2f} $ | R^2: {r2:6.4f} | Czas: {czas:6.2f}s")

# ==========================================
# TENSORFLOW (Dostosowany do Waszej sieci)
# ==========================================
print("\nUruchamiam TensorFlow (Keras) z Waszymi hiperparametrami...")
model_tf = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Używamy SGD z LR = 0.05, by sprawiedliwie porównać go z Waszym optymalizatorem
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)
model_tf.compile(optimizer=optimizer, loss='mse')

start_time = time.time()
model_tf.fit(X_train, y_train_log, epochs=100, batch_size=128, verbose=0)
czas_tf = time.time() - start_time

y_pred_tf = np.exp(model_tf.predict(X_test).ravel())
mae_tf = mean_absolute_error(prawdziwe_ceny_test, y_pred_tf)
r2_tf = r2_score(prawdziwe_ceny_test, y_pred_tf)
wyniki['TensorFlow'] = y_pred_tf

print(f"[{'TensorFlow':16}] MAE: {mae_tf:8.2f} $ | R^2: {r2_tf:6.4f} | Czas: {czas_tf:6.2f}s")

print("\n=== ETAP 3: BENCHMARK ARCYDZIEŁ (Top 3%) ===")
prog_top_3 = np.percentile(prawdziwe_ceny_test, 97)
maska_arcydziel = prawdziwe_ceny_test > prog_top_3
prawdziwe_arcydziela = prawdziwe_ceny_test[maska_arcydziel]

for nazwa, predykcje in wyniki.items():
    predykcje_arcydziel = predykcje[maska_arcydziel]
    mae_arcydziela = mean_absolute_error(prawdziwe_arcydziela, predykcje_arcydziel)
    print(f"{nazwa:16} -> MAE Arcydzieł: {mae_arcydziela:.2f} $")