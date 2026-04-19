import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ==========================================
# 1. FUNKCJE AKTYWACJI I ICH POCHODNE
# ==========================================
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_derivative(x): s = sigmoid(x); return s * (1 - s)

def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)

def tanh(x): return np.tanh(x)
def tanh_derivative(x): return 1.0 - np.tanh(x)**2

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# ==========================================
# 2. ROZBUDOWANA KLASA SIECI NEURONOWEJ
# ==========================================
class AdvancedNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, 
                 learning_rate=0.01, activation='sigmoid', 
                 weight_init='normal', momentum=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.activation = activation
        
        # Wybór inicjalizacji wag
        if weight_init == 'normal':
            # Klasyczna losowa z rozkładu normalnego
            self.W1 = np.random.randn(input_size, hidden_size) * 0.1
            self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        elif weight_init == 'xavier':
            # Xavier/Glorot - dobra dla Sigmoid/Tanh
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
        elif weight_init == 'he':
            # He - dobra dla ReLU
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)

        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))
        
        # Pamięć prędkości dla Momentum
        self.v_W1, self.v_b1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.v_W2, self.v_b2 = np.zeros_like(self.W2), np.zeros_like(self.b2)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        
        if self.activation == 'sigmoid': self.A1 = sigmoid(self.Z1)
        elif self.activation == 'relu': self.A1 = relu(self.Z1)
        elif self.activation == 'tanh': self.A1 = tanh(self.Z1)
            
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2
    
    def backward(self, X, y):
        m = X.shape[0]
        
        # Output layer gradients
        dZ2 = self.A2 - y
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        if self.activation == 'sigmoid': dZ1 = dA1 * sigmoid_derivative(self.Z1)
        elif self.activation == 'relu': dZ1 = dA1 * relu_derivative(self.Z1)
        elif self.activation == 'tanh': dZ1 = dA1 * tanh_derivative(self.Z1)
            
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Aktualizacja z użyciem Momentum
        self.v_W2 = self.momentum * self.v_W2 + self.lr * dW2
        self.v_b2 = self.momentum * self.v_b2 + self.lr * db2
        self.v_W1 = self.momentum * self.v_W1 + self.lr * dW1
        self.v_b1 = self.momentum * self.v_b1 + self.lr * db1
        
        self.W2 -= self.v_W2
        self.b2 -= self.v_b2
        self.W1 -= self.v_W1
        self.b1 -= self.v_b1

    def train(self, X, y, epochs=1000, batch_size=None):
        m = X.shape[0]
        if batch_size is None or batch_size >= m:
            # Full Batch Gradient Descent
            for _ in range(epochs):
                self.forward(X)
                self.backward(X, y)
        else:
            # Mini-Batch Gradient Descent
            for _ in range(epochs):
                indices = np.random.permutation(m)
                X_shuf, y_shuf = X[indices], y[indices]
                for i in range(0, m, batch_size):
                    self.forward(X_shuf[i:i+batch_size])
                    self.backward(X_shuf[i:i+batch_size], y_shuf[i:i+batch_size])
            
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# ==========================================
# 3. WCZYTANIE DANYCH
# ==========================================
try:
    df = pd.read_csv('auta_bez_duplikatow.csv', sep=';')
except FileNotFoundError:
    print("Błąd: Upewnij się, że masz plik 'auta_bez_duplikatow.csv'")
    exit()

df = df.drop('model', axis=1)
X = df.drop('nadwozie', axis=1)
y = df['nadwozie']

etykiety = y.unique()
etykiety_dict = {nazwa: i for i, nazwa in enumerate(etykiety)}
y_num = y.map(etykiety_dict).values

num_classes = len(etykiety)
y_onehot = np.eye(num_classes)[y_num]

numeric_features = ['masa', 'rok_produkcji', 'dlugosc', 'wysokosc', 'szerokosc', 'liczba_drzwi', 'bagaznik']
categorical_features = ['rynek']

# ==========================================
# 4. SILNIK EKSPERYMENTÓW (Testowanie 8 parametrów!)
# ==========================================
def run_experiment(param_name, values, repeats=3):
    print(f"\n[{param_name.upper()}] Badanie wpływu parametru...")
    for val in values:
        train_accs, test_accs = [], []
        
        for _ in range(repeats):
            # Domyślne wartości
            t_size, h_size, lr, ep, act, w_init, b_size, mom = 0.2, 16, 0.1, 500, 'sigmoid', 'normal', None, 0.0
            
            # Podmiana badanego parametru
            if param_name == 'test_size': t_size = val
            elif param_name == 'hidden_size': h_size = val
            elif param_name == 'learning_rate': lr = val
            elif param_name == 'epochs': ep = val
            elif param_name == 'activation': act = val
            elif param_name == 'weight_init': w_init = val
            elif param_name == 'batch_size': b_size = val
            elif param_name == 'momentum': mom = val

            # Przygotowanie danych (uwzględnia zmianę test_size)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y_onehot, test_size=t_size, random_state=None)
            
            prep = ColumnTransformer([
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])
            X_tr_sc = prep.fit_transform(X_tr)
            X_te_sc = prep.transform(X_te)
            input_size = X_tr_sc.shape[1]

            # Inicjalizacja i uczenie
            nn = AdvancedNeuralNetwork(input_size, h_size, num_classes, lr, act, w_init, mom)
            nn.train(X_tr_sc, y_tr, epochs=ep, batch_size=b_size)
            
            # Wyniki
            pred_train = nn.predict(X_tr_sc)
            pred_test = nn.predict(X_te_sc)
            
            train_accs.append(np.mean(pred_train == np.argmax(y_tr, axis=1)))
            test_accs.append(np.mean(pred_test == np.argmax(y_te, axis=1)))

        # Wypisanie statystyk z powtórzeń (Średnia i Max)
        print(f" Wartość: {str(val):<8} | TEST (Średnia/Max): {np.mean(test_accs)*100:>5.1f}% / {np.max(test_accs)*100:>5.1f}% | TRENING (Średnia/Max): {np.mean(train_accs)*100:>5.1f}% / {np.max(train_accs)*100:>5.1f}%")


print("=== PROJEKT SSN: START BADANIA 8 PARAMETRÓW ===")

# 1. Liczba neuronów w warstwie ukrytej
run_experiment('hidden_size', [8, 16, 32, 64])

# 2. Współczynnik uczenia
run_experiment('learning_rate', [0.01, 0.05, 0.1, 0.5])

# 3. Liczba epok (długość uczenia)
run_experiment('epochs', [100, 300, 600, 1000])

# 4. Sposób doboru próby testowej / wielkość (wymóg z PDF)
run_experiment('test_size', [0.1, 0.2, 0.3, 0.4])

# 5. Funkcja aktywacji w warstwie ukrytej
run_experiment('activation', ['sigmoid', 'tanh', 'relu', 'sigmoid']) # Podwójny sigmoid żeby było 4 próby

# 6. Sposób inicjalizacji wag początkowych
run_experiment('weight_init', ['normal', 'xavier', 'he', 'normal'])

# 7. Rozmiar mini-batcha (Optymalizacja Mini-Batch Gradient Descent)
run_experiment('batch_size', [None, 32, 64, 128]) # None = Full Batch

# 8. Współczynnik Momentum (przyspieszanie omijania minimów lokalnych)
run_experiment('momentum', [0.0, 0.5, 0.9, 0.99])

print("=== KONIEC BADAŃ ===")