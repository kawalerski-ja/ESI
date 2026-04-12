import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ==========================================
# 1. FUNKCJE AKTYWACJI I ICH POCHODNE
# ==========================================
def sigmoid(x):
    # Używamy np.clip, aby uniknąć problemów z przepełnieniem (overflow) w exp
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    # Odejmowanie max(x) dla stabilności numerycznej
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# ==========================================
# 2. KLASA SIECI NEURONOWEJ (OD ZERA W NUMPY)
# ==========================================
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        # Inicjalizacja wag i biasów (losowe wartości z rozkładu normalnego)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
    def forward(self, X):
        # Warstwa ukryta
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        
        # Warstwa wyjściowa
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        
        return self.A2
    
    def backward(self, X, y):
        m = X.shape[0] # Liczba próbek
        
        # Obliczanie gradientów (Cross-Entropy + Softmax derivative to A2 - y)
        dZ2 = self.A2 - y
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(self.Z1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Aktualizacja wag
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs=1000):
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y)
            
    def predict(self, X):
        A2 = self.forward(X)
        return np.argmax(A2, axis=1)

# ==========================================
# 3. WCZYTANIE I PRZYGOTOWANIE DANYCH
# ==========================================
try:
    df = pd.read_csv('auta_bez_duplikatow.csv', sep=';')
except FileNotFoundError:
    print("Błąd: Upewnij się, że plik 'auta_bez_duplikatow.csv' jest w tym samym folderze.")
    exit()

df = df.drop('model', axis=1)

X = df.drop('nadwozie', axis=1)
y = df['nadwozie']

# Konwersja etykiet y (np. 'Sedan', 'SUV') na liczby (Label Encoding)
etykiety = y.unique()
etykiety_dict = {nazwa: i for i, nazwa in enumerate(etykiety)}
y_num = y.map(etykiety_dict).values

# One-Hot Encoding dla wyjścia sieci neuronowej (np. 3 -> [0, 0, 0, 1, 0, 0])
num_classes = len(etykiety)
y_onehot = np.eye(num_classes)[y_num]

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

numeric_features = ['masa', 'rok_produkcji', 'dlugosc', 'wysokosc', 'szerokosc', 'liczba_drzwi', 'bagaznik']
categorical_features = ['rynek']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

input_size = X_train_scaled.shape[1]
output_size = num_classes

# ==========================================
# 4. EKSPERYMENTY Z PARAMETRAMI (Z POWTÓRZENIAMI)
# ==========================================
def eksperyment(parametr_nazwa, parametr_lista, powtorzenia=5, epochs=1500):
    print(f"\n--- BADANIE PARAMETRU: {parametr_nazwa} ---")
    
    for wartosc in parametr_lista:
        wyniki_train = []
        wyniki_test = []
        
        for i in range(powtorzenia):
            # Inicjalizacja sieci z badanymi parametrami
            if parametr_nazwa == 'hidden_size':
                nn = SimpleNeuralNetwork(input_size, hidden_size=wartosc, output_size=output_size, learning_rate=0.5)
            elif parametr_nazwa == 'learning_rate':
                nn = SimpleNeuralNetwork(input_size, hidden_size=16, output_size=output_size, learning_rate=wartosc)
                
            # Trenowanie
            nn.train(X_train_scaled, y_train, epochs=epochs)
            
            # Ewaluacja
            pred_train = nn.predict(X_train_scaled)
            pred_test = nn.predict(X_test_scaled)
            
            acc_train = np.mean(pred_train == y_train_labels)
            acc_test = np.mean(pred_test == y_test_labels)
            
            wyniki_train.append(acc_train)
            wyniki_test.append(acc_test)
            
        # Podsumowanie wyników dla danej wartości parametru po N powtórzeniach
        srednia_train = np.mean(wyniki_train) * 100
        max_train = np.max(wyniki_train) * 100
        srednia_test = np.mean(wyniki_test) * 100
        max_test = np.max(wyniki_test) * 100
        
        print(f"{parametr_nazwa} = {wartosc:<5} | Test: Średnia {srednia_test:>5.2f}%, Max {max_test:>5.2f}% | Uczący: Średnia {srednia_train:>5.2f}%, Max {max_train:>5.2f}%")


# Uruchomienie badań (dla 4 różnych wartości każdego parametru - jak w wytycznych)
print("Rozpoczynam badanie Sieci Neuronowych...")
print("Uwaga: Ze względu na powtórzenia proces może potrwać kilkanaście sekund.\n")

# Badanie 1: Wpływ liczby neuronów w warstwie ukrytej
eksperyment(parametr_nazwa='hidden_size', parametr_lista=[4, 8, 16, 32], powtorzenia=5)

# Badanie 2: Wpływ współczynnika uczenia (Learning Rate)
eksperyment(parametr_nazwa='learning_rate', parametr_lista=[0.01, 0.1, 0.5, 1.0], powtorzenia=5)