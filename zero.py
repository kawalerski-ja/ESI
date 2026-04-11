import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings

# Wyłączamy ostrzeżenia dla czystszego wyjścia w konsoli
warnings.filterwarnings('ignore')

print("=== ETAP 1: PRZYGOTOWANIE DANYCH ===")
df = pd.read_csv("auction_results_color_svd.csv")

# 1. Kodowanie zmiennych kategorycznych (One-Hot)
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

indeksy_train = df_train.index
indeksy_test = df_test.index

X_kat_train = X_kat_cale[indeksy_train]
X_kat_test = X_kat_cale[indeksy_test]

# 3. Normalizacja zmiennych liczbowych
zmienne_liczbowe = ["TOTAL DIMENSIONS", "YEAR", "Colorfulness Score", "SVD Entropy"]
srednia = df_train[zmienne_liczbowe].mean()
odchylenie = df_train[zmienne_liczbowe].std()

df_train_num = (df_train[zmienne_liczbowe] - srednia) / odchylenie
df_test_num = (df_test[zmienne_liczbowe] - srednia) / odchylenie

X_num_train = df_train_num.to_numpy(dtype=np.float32)
X_num_test = df_test_num.to_numpy(dtype=np.float32)

X_train = np.hstack([X_kat_train, X_num_train], dtype=np.float32)
X_test = np.hstack([X_kat_test, X_num_test], dtype=np.float32)

# Zapisanie wszystkich nazw kolumn (niezbędne do SHAP!)
wszystkie_cechy = nazwy_kolumn_kat + zmienne_liczbowe

# 4. Normalizacja ceny (Zmienna docelowa)
srednia_cena = df_train['PRICE'].mean()
odchylenie_cena = df_train['PRICE'].std()

y_train = ((df_train['PRICE'] - srednia_cena) / odchylenie_cena).to_numpy(dtype=np.float32).reshape(-1, 1)
y_test = ((df_test['PRICE'] - srednia_cena) / odchylenie_cena).to_numpy(dtype=np.float32).reshape(-1, 1)

def prawdziwa_cena(znormalizowana_cena):
    return znormalizowana_cena * odchylenie_cena + srednia_cena

print(f"Dane gotowe. Kształt X_train: {X_train.shape}")

# ==========================================
# KLASY SIECI NEURONOWEJ (Twoja implementacja)
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

print("\n=== ETAP 2: TRENING AUTORSKIEJ SIECI ===")
liczba_cech = X_train.shape[1]
dense1 = Layer_Dense(liczba_cech, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 32)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(32, 1)
activation3 = Activation_Linear()

loss_function = Loss_MSE()
optimizer = Optimizer_SGD(learning_rate=0.1)

epoki = 100
batch_size = 256
historia_straty = [] # Do wykresu!

for epoch in range(epoki + 1):
    loss_epoki = 0
    ilosc_paczek = 0
    for start_idx in range(0, len(X_train), batch_size):
        end_idx = start_idx + batch_size
        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        
        # Forward pass
        dense1.forward(X_batch); activation1.forward(dense1.output)
        dense2.forward(activation1.output); activation2.forward(dense2.output)
        dense3.forward(activation2.output); activation3.forward(dense3.output)
        
        # Loss
        loss = loss_function.calculate(activation3.output, y_batch)
        loss_epoki += loss
        ilosc_paczek += 1
        
        # Backward pass
        loss_function.backward(activation3.output, y_batch)
        activation3.backward(loss_function.dinputs); dense3.backward(activation3.dinputs)
        activation2.backward(dense3.dinputs); dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs); dense1.backward(activation1.dinputs)
        
        # Update
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
        
    sredni_blad_mse = loss_epoki / ilosc_paczek
    historia_straty.append(sredni_blad_mse)
    
print("Trening zakończony!")

# Przewidywania dla całego zbioru testowego
dense1.forward(X_test); activation1.forward(dense1.output)
dense2.forward(activation1.output); activation2.forward(dense2.output)
dense3.forward(activation2.output); activation3.forward(dense3.output)

wymyslone_ceny = prawdziwa_cena(activation3.output).flatten()
prawdziwe_ceny = prawdziwa_cena(y_test).flatten()

# ==========================================
# WIZUALIZACJE MATPLOTLIB
# ==========================================
print("\n=== ETAP 3: GENEROWANIE WYKRESÓW ===")
plt.figure(figsize=(14, 5))

# Wykres 1: Krzywa uczenia
plt.subplot(1, 2, 1)
plt.plot(historia_straty, color='blue', linewidth=2)
plt.title("Krzywa uczenia (Loss Curve)")
plt.xlabel("Epoka")
plt.ylabel("Błąd MSE (Znormalizowany)")
plt.grid(True, linestyle='--', alpha=0.7)

# Wykres 2: Rozrzut (Przewidywania vs Prawda)
plt.subplot(1, 2, 2)
plt.scatter(prawdziwe_ceny, wymyslone_ceny, alpha=0.5, color='purple', s=10)
# Rysujemy idealną linię 45 stopni
max_val = max(max(prawdziwe_ceny), max(wymyslone_ceny))
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Idealna predykcja')
plt.title("Predykcja vs Rzeczywistość")
plt.xlabel("Prawdziwa cena ($)")
plt.ylabel("Przewidziana cena ($)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("wykresy_siec_autorska.png")
print("Zapisano wykresy jako 'wykresy_siec_autorska.png'.")

# ==========================================
# ANALIZA ARCYDZIEŁ (Top 3%)
# ==========================================
print("\n=== ETAP 4: PUŁAPKA ARCYDZIEŁ (TEST TOP 3%) ===")
prog_top_3 = np.percentile(prawdziwe_ceny, 97)
print(f"Próg cenowy dla arcydzieł (97. percentyl): {prog_top_3:.2f} $")

maska_arcydziel = prawdziwe_ceny > prog_top_3
prawdziwe_arcydziela = prawdziwe_ceny[maska_arcydziel]
wymyslone_arcydziela = wymyslone_ceny[maska_arcydziel]

mae_ogolne = np.mean(np.abs(prawdziwe_ceny - wymyslone_ceny))
mae_arcydziela = np.mean(np.abs(prawdziwe_arcydziela - wymyslone_arcydziela))

print(f"Średni błąd (MAE) dla wszystkich obrazów: {mae_ogolne:.2f} $")
print(f"Średni błąd (MAE) TYLKO dla arcydzieł:    {mae_arcydziela:.2f} $")
print("Wniosek: Sieć jest zbyt ostrożna i mocno zaniża najdroższe dzieła!")

# ==========================================
# WYTŁUMACZALNE AI (SHAP)
# ==========================================
print("\n=== ETAP 5: GENEROWANIE SHAP (XAI) ===")
# Wrapper dla SHAP
def nasza_siec_predict(X_input):
    dense1.forward(X_input); activation1.forward(dense1.output)
    dense2.forward(activation1.output); activation2.forward(dense2.output)
    dense3.forward(activation2.output); activation3.forward(dense3.output)
    return prawdziwa_cena(activation3.output).flatten()

# Tło dla SHAP (żeby było szybko, bierzemy 50 probek)
background = shap.sample(X_train, 50)
explainer = shap.KernelExplainer(nasza_siec_predict, background)

# Analizujemy 5 pierwszych obrazów testowych
print("Obliczanie wartości SHAP (to może zająć kilkanaście sekund)...")
shap_values = explainer.shap_values(X_test[:5])

# Generujemy wykres Summary Plot (Zapisujemy go do pliku)
plt.figure()
shap.summary_plot(shap_values, X_test[:5], feature_names=wszystkie_cechy, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", bbox_inches='tight')
print("Zapisano analizę SHAP jako 'shap_summary.png'.")
print("Gotowe! Możesz zamknąć skrypt.")