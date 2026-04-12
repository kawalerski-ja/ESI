import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings('ignore')

print("=== ETAP 1: PRZYGOTOWANIE DANYCH ===")
# Odporne wczytywanie pliku (zawsze znajdzie plik, jeśli leży w tym samym folderze co skrypt)
katalog_skryptu = os.path.dirname(os.path.abspath(__file__))
sciezka_do_pliku = os.path.join(katalog_skryptu, "auction_results_color_svd.csv")
df = pd.read_csv(sciezka_do_pliku)

# 1. Zmienne kategoryczne (One-Hot Encoding + Zapisanie nazw do SHAP)
zmienne_kategoryczne = ['ARTIST', 'TECHNIQUE', 'SIGNATURE', 'CONDITION']
bloki_kategoryczne = []
nazwy_kolumn_kat = []

for zmienna in zmienne_kategoryczne:
    kody = df[zmienna].astype('category').cat.codes.to_numpy(dtype=np.int32)
    liczba_klas = np.max(kody) + 1
    one_hot = np.eye(liczba_klas)[kody]
    bloki_kategoryczne.append(one_hot)
    for i in range(liczba_klas):
        nazwy_kolumn_kat.append(f"{zmienna}_{i}")

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

X_train = np.hstack([X_kat_train, X_num_train], dtype=np.float32)
X_test = np.hstack([X_kat_test, X_num_test], dtype=np.float32)

wszystkie_cechy = nazwy_kolumn_kat + zmienne_liczbowe

# 4. LOGARYTMIZACJA ZMIENNEJ DOCELOWEJ (Metoda kolegi)
y_train = np.log(df_train['PRICE']).to_numpy(dtype=np.float32).reshape(-1, 1)
y_test = np.log(df_test['PRICE']).to_numpy(dtype=np.float32).reshape(-1, 1)

def prawdziwa_cena(wyliczona_wartosc):
    return np.exp(wyliczona_wartosc)

# ==========================================
# KLASY SIECI NEURONOWEJ
# ==========================================
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

class Loss_MSE:
    def calculate(self, output, y):
        return np.mean((output - y) ** 2)
    def backward(self, dvalues, y_true):
        self.dinputs = -2 * (y_true - dvalues) / len(dvalues)

class Optimizer_SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

print("\n=== ETAP 2: TRENING NAJLEPSZEJ SIECI ===")
# Najlepsze hiperparametry z notatnika
n1, n2 = 128, 64
najlepszy_lr = 0.05
najlepszy_batch = 128
epoki = 100

dense1 = Layer_Dense(X_train.shape[1], n1)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(n1, n2)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(n2, 1)
activation3 = Activation_Linear()

loss_function = Loss_MSE()
optimizer = Optimizer_SGD(learning_rate=najlepszy_lr)

historia_straty = []
start_time = time.time()

for epoch in range(epoki):
    loss_epoki = 0
    ilosc_paczek = 0
    for start_idx in range(0, len(X_train), najlepszy_batch):
        end_idx = start_idx + najlepszy_batch
        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        
        dense1.forward(X_batch); activation1.forward(dense1.output)
        dense2.forward(activation1.output); activation2.forward(dense2.output)
        dense3.forward(activation2.output); activation3.forward(dense3.output)
        
        loss = loss_function.calculate(activation3.output, y_batch)
        loss_epoki += loss
        ilosc_paczek += 1
        
        loss_function.backward(activation3.output, y_batch)
        activation3.backward(loss_function.dinputs); dense3.backward(activation3.dinputs)
        activation2.backward(dense3.dinputs); dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs); dense1.backward(activation1.dinputs)
        
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
        
    historia_straty.append(loss_epoki / ilosc_paczek)

czas_treningu = time.time() - start_time
print(f"Trening zakończony w czasie: {czas_treningu:.2f} s")

# Ewaluacja
dense1.forward(X_test); activation1.forward(dense1.output)
dense2.forward(activation1.output); activation2.forward(dense2.output)
dense3.forward(activation2.output); activation3.forward(dense3.output)

wymyslone_ceny = prawdziwa_cena(activation3.output).flatten()
prawdziwe_ceny = prawdziwa_cena(y_test).flatten()

# --- OBLICZANIE METRYK BŁĘDU ---
mae = np.mean(np.abs(prawdziwe_ceny - wymyslone_ceny))
rmse = np.sqrt(np.mean((prawdziwe_ceny - wymyslone_ceny)**2))

# --- OBLICZANIE R^2 (Czysta matematyka w NumPy) ---
# Suma kwadratów reszt (błędów)
ss_res = np.sum((prawdziwe_ceny - wymyslone_ceny)**2)
# Całkowita suma kwadratów (wariancja danych)
ss_tot = np.sum((prawdziwe_ceny - np.mean(prawdziwe_ceny))**2)
# Ostateczny współczynnik R^2
r2 = 1 - (ss_res / ss_tot)

print("\n" + "="*50)
print("RAPORT KOŃCOWY: AUTORSKA SIEĆ (DANE TESTOWE)")
print("="*50)
print(f"1. MAE (Średnia pomyłka):     {mae:10.2f} $")
print(f"2. RMSE (Kara za ekstrema):   {rmse:10.2f} $")
print(f"3. Współczynnik R^2:          {r2:10.4f} (max 1.0)")
print("="*50)

# ==========================================
# WIZUALIZACJE MATPLOTLIB
# ==========================================
print("\n=== ETAP 3: GENEROWANIE WYKRESÓW ===")
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(historia_straty, color='blue', linewidth=2)
plt.title("Krzywa uczenia (Loss Curve na Logarytmie)")
plt.xlabel("Epoka"); plt.ylabel("Błąd MSE")
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
plt.scatter(prawdziwe_ceny, wymyslone_ceny, alpha=0.5, color='purple', s=10)
max_val = max(max(prawdziwe_ceny), max(wymyslone_ceny))
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Idealna predykcja')
plt.title("Predykcja vs Rzeczywistość (w Dolarach)")
plt.xlabel("Prawdziwa cena ($)"); plt.ylabel("Przewidziana cena ($)")
plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("wykresy_siec_finalna.png")
print("Zapisano wykresy jako 'wykresy_siec_finalna.png'.")

# ==========================================
# ANALIZA ARCYDZIEŁ (Top 3%)
# ==========================================
print("\n=== ETAP 4: TEST ARCYDZIEŁ (TOP 3%) ===")
prog_top_3 = np.percentile(prawdziwe_ceny, 97)
maska_arcydziel = prawdziwe_ceny > prog_top_3

mae_arcydziela = np.mean(np.abs(prawdziwe_ceny[maska_arcydziel] - wymyslone_ceny[maska_arcydziel]))
print(f"Próg cenowy dla arcydzieł: {prog_top_3:.2f} $")
print(f"Błąd MAE TYLKO dla arcydzieł: {mae_arcydziela:.2f} $")

# ==========================================
# WYTŁUMACZALNE AI (SHAP)
# ==========================================
print("\n=== ETAP 5: GENEROWANIE SHAP ===")
def nasza_siec_predict(X_input):
    dense1.forward(X_input); activation1.forward(dense1.output)
    dense2.forward(activation1.output); activation2.forward(dense2.output)
    dense3.forward(activation2.output); activation3.forward(dense3.output)
    # Zwracamy wynik zlogarytmizowany (tak samo jak uczyła się sieć)
    return activation3.output.flatten()

background = shap.sample(X_train, 50)
explainer = shap.KernelExplainer(nasza_siec_predict, background)
shap_values = explainer.shap_values(X_test[:10])

plt.figure()
shap.summary_plot(shap_values, X_test[:10], feature_names=wszystkie_cechy, show=False)
plt.tight_layout()
plt.savefig("shap_summary_final.png", bbox_inches='tight')
print("Zapisano analizę SHAP jako 'shap_summary_final.png'. Gotowe!")