import os
import time
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

print("=== ETAP 1: PRZYGOTOWANIE DANYCH ===")
katalog_skryptu = os.path.dirname(os.path.abspath(__file__))
sciezka_do_pliku = os.path.join(katalog_skryptu, "auction_results_color_svd.csv")
df = pd.read_csv(sciezka_do_pliku)

# Zmienne kategoryczne (One-Hot Encoding)
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

# Podział Train/Test (80/20)
np.random.seed(42)
df_train = df.sample(frac=0.8, random_state=42)
df_test = df.drop(df_train.index)

X_kat_train = X_kat_cale[df_train.index]
X_kat_test = X_kat_cale[df_test.index]

# Normalizacja zmiennych liczbowych
zmienne_liczbowe = ["TOTAL DIMENSIONS", "YEAR", "Colorfulness Score", "SVD Entropy"]
srednia = df_train[zmienne_liczbowe].mean()
odchylenie = df_train[zmienne_liczbowe].std()

X_num_train = ((df_train[zmienne_liczbowe] - srednia) / odchylenie).to_numpy(dtype=np.float32)
X_num_test = ((df_test[zmienne_liczbowe] - srednia) / odchylenie).to_numpy(dtype=np.float32)

X_train = np.hstack([X_kat_train, X_num_train], dtype=np.float32)
X_test = np.hstack([X_kat_test, X_num_test], dtype=np.float32)

# Lista wszystkich cech
wszystkie_cechy = nazwy_kolumn_kat + zmienne_liczbowe

# ==========================================
# USUNIĘCIE SVD ENTROPY (Zgodnie z modelem finalnym)
# ==========================================
X_train_final = X_train[:, :-1]
X_test_final = X_test[:, :-1]
cechy_finalne = wszystkie_cechy[:-1] # Usuwamy ostatnią nazwę (SVD Entropy) z legendy SHAP

# Logarytmizacja ceny
y_train = np.log(df_train['PRICE']).to_numpy(dtype=np.float32).reshape(-1, 1)
y_test = np.log(df_test['PRICE']).to_numpy(dtype=np.float32).reshape(-1, 1)


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

class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate
    def forward(self, inputs, training=True):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

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
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2, axis=-1)
    def backward(self, dvalues, y_true):
        self.dinputs = -2 * (y_true - dvalues) / len(dvalues)

class Optimizer_SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

print("\n=== ETAP 2: SZYBKI TRENING OSTATECZNEGO MODELU ===")
liczba_cech = X_train_final.shape[1]
n1, n2 = 256, 128
poziom_dropoutu = 0.3
najlepszy_lr = 0.01
najlepszy_batch = 64
epoki = 150 

dense1 = Layer_Dense(liczba_cech, n1)
activation1 = Activation_ReLU()
dropout1 = Layer_Dropout(poziom_dropoutu)

dense2 = Layer_Dense(n1, n2)
activation2 = Activation_ReLU()
dropout2 = Layer_Dropout(poziom_dropoutu)

dense3 = Layer_Dense(n2, 1)
activation3 = Activation_Linear()

loss_function = Loss_MSE()
optimizer = Optimizer_SGD(learning_rate=najlepszy_lr)

start_time = time.time()
for epoch in range(epoki):
    for start_idx in range(0, len(X_train_final), najlepszy_batch):
        end_idx = start_idx + najlepszy_batch
        X_batch = X_train_final[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        
        # Forward Pass (Dropout włączony)
        dense1.forward(X_batch); activation1.forward(dense1.output)
        dropout1.forward(activation1.output, training=True)
        
        dense2.forward(dropout1.output); activation2.forward(dense2.output)
        dropout2.forward(activation2.output, training=True)
        
        dense3.forward(dropout2.output); activation3.forward(dense3.output)
        
        # Backward Pass
        loss_function.backward(activation3.output, y_batch)
        activation3.backward(loss_function.dinputs); dense3.backward(activation3.dinputs)
        
        dropout2.backward(dense3.dinputs); activation2.backward(dropout2.dinputs)
        dense2.backward(activation2.dinputs)
        
        dropout1.backward(dense2.dinputs); activation1.backward(dropout1.dinputs)
        dense1.backward(activation1.dinputs)
        
        # Update
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)

print(f"Trening ukończony w {time.time() - start_time:.1f} sekund.")


# ==========================================
# TEST ARCYDZIEŁ (PUŁAPKA DŁUGIEGO OGONA)
# ==========================================
print("\n=== ETAP 2.5: TEST ARCYDZIEŁ (TOP 3%) ===")

# 1. Przejście przez zbiór testowy (Pamiętamy o WYŁĄCZENIU Dropoutu!)
dense1.forward(X_test_final)
activation1.forward(dense1.output)
dropout1.forward(activation1.output, training=False)

dense2.forward(dropout1.output)
activation2.forward(dense2.output)
dropout2.forward(activation2.output, training=False)

dense3.forward(dropout2.output)
activation3.forward(dense3.output)

# 2. Odwrócenie logarytmu, by uzyskać ceny w Dolarach
wymyslone_ceny = np.exp(activation3.output).flatten()
prawdziwe_ceny = np.exp(y_test).flatten()

# 3. Identyfikacja Arcydzieł (97. percentyl)
prog_top_3 = np.percentile(prawdziwe_ceny, 97)
maska_arcydziel = prawdziwe_ceny > prog_top_3

prawdziwe_arcydziela = prawdziwe_ceny[maska_arcydziel]
wymyslone_arcydziela = wymyslone_ceny[maska_arcydziel]

# 4. Obliczenie i wypisanie MAE
mae_ogolne = np.mean(np.abs(prawdziwe_ceny - wymyslone_ceny))
mae_arcydziela = np.mean(np.abs(prawdziwe_arcydziela - wymyslone_arcydziela))

print(f"Próg cenowy dla 'arcydzieł' (97. percentyl): {prog_top_3:.2f} $")
print(f"Liczba obrazów w tej ekskluzywnej grupie:   {len(prawdziwe_arcydziela)}")
print("-" * 50)
print(f"Średni błąd (MAE) dla wszystkich {len(prawdziwe_ceny)} obrazów: {mae_ogolne:7.2f} $")
print(f"Średni błąd (MAE) TYLKO dla arcydzieł:      {mae_arcydziela:7.2f} $")
print("Wniosek: Ostateczna, głęboka sieć wciąż brutalnie zaniża wartość najdroższej sztuki!")


# ==========================================
# WYTŁUMACZALNE AI (SHAP) Z NOWĄ ARCHITEKTURĄ
# ==========================================
print("\n=== ETAP 3: GENEROWANIE WYKRESU SHAP ===")

# Wrapper dla SHAP - Zauważ flagę training=False!
def shap_predict(X_input):
    dense1.forward(X_input)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output, training=False)
    
    dense2.forward(dropout1.output)
    activation2.forward(dense2.output)
    dropout2.forward(activation2.output, training=False)
    
    dense3.forward(dropout2.output)
    activation3.forward(dense3.output)
    return activation3.output.flatten()

# Obliczanie SHAP (Background z 50 próbek, testujemy na 15 obrazach)
print("Obliczanie wartości SHAP (to może zająć chwilę)...")
background = shap.sample(X_train_final, 50)
explainer = shap.KernelExplainer(shap_predict, background)
shap_values = explainer.shap_values(X_test_final[:15])

# Generowanie i zapisywanie wykresu
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_final[:15], feature_names=cechy_finalne, show=False)
plt.tight_layout()
plt.savefig("shap_ostateczny_finalny.png", bbox_inches='tight', dpi=300)

print("\nGotowe! Zapisano ostateczną wersję wykresu jako 'shap_ostateczny_finalny.png'.")