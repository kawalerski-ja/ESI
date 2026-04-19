import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ==========================================
# 1. FUNKCJE AKTYWACJI I ICH POCHODNE
# ==========================================
def sigmoid(x): 
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x): 
    s = sigmoid(x)
    return s * (1 - s)

def relu(x): 
    return np.maximum(0, x)

def relu_derivative(x): 
    return (x > 0).astype(float)

def tanh(x): 
    return np.tanh(x)

def tanh_derivative(x): 
    return 1.0 - np.tanh(x)**2

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# ==========================================
# 2. ROZBUDOWANA KLASA SIECI NEURONOWEJ
# ==========================================
class DynamicNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, 
                 learning_rate=0.01, activation='sigmoid', 
                 weight_init='normal', momentum=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.activation = activation
        
        # Jeśli użytkownik poda liczbę zamiast listy, zamieniamy na listę
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]
            
        layers = [input_size] + hidden_layers + [output_size]
        
        self.weights = []
        self.biases = []
        self.v_weights = [] # Pamięć dla momentum
        self.v_biases = []

        for i in range(len(layers) - 1):
            n_in, n_out = layers[i], layers[i+1]
            
            # Inicjalizacja wag
            if weight_init == 'normal':
                w = np.random.randn(n_in, n_out) * 0.1
            elif weight_init == 'xavier':
                w = np.random.randn(n_in, n_out) * np.sqrt(1 / n_in)
            elif weight_init == 'he':
                w = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
            elif weight_init == 'orthogonal':
                # Inicjalizacja ortogonalna z wykorzystaniem SVD
                a = np.random.normal(0.0, 1.0, (n_in, n_out))
                u, _, v = np.linalg.svd(a, full_matrices=False)
                w = u if u.shape == (n_in, n_out) else v
            else:
                w = np.random.randn(n_in, n_out) * 0.1
            
            self.weights.append(w)
            self.biases.append(np.zeros((1, n_out)))
            self.v_weights.append(np.zeros((n_in, n_out)))
            self.v_biases.append(np.zeros((1, n_out)))

    def forward(self, X):
        self.A = [X]
        self.Z = []
        curr_input = X
        
        for i in range(len(self.weights)):
            z = np.dot(curr_input, self.weights[i]) + self.biases[i]
            self.Z.append(z)
            
            if i == len(self.weights) - 1: # Ostatnia warstwa - Softmax
                curr_input = softmax(z)
            else:
                if self.activation == 'sigmoid': curr_input = sigmoid(z)
                elif self.activation == 'relu': curr_input = relu(z)
                elif self.activation == 'tanh': curr_input = tanh(z)
                elif self.activation == 'leaky_relu': curr_input = leaky_relu(z)
            self.A.append(curr_input)
        return self.A[-1]
    
    def backward(self, X, y):
        m = X.shape[0]
        dZ = self.A[-1] - y # Błąd wyjściowy
        
        for i in reversed(range(len(self.weights))):
            dW = (1/m) * np.dot(self.A[i].T, dZ)
            db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            if i > 0: # Propagacja błędu do poprzedniej warstwy
                dA_prev = np.dot(dZ, self.weights[i].T)
                if self.activation == 'sigmoid': dZ = dA_prev * sigmoid_derivative(self.Z[i-1])
                elif self.activation == 'relu': dZ = dA_prev * relu_derivative(self.Z[i-1])
                elif self.activation == 'tanh': dZ = dA_prev * tanh_derivative(self.Z[i-1])
                elif self.activation == 'leaky_relu': dZ = dA_prev * leaky_relu_derivative(self.Z[i-1])
            
            # Aktualizacja wag z Momentum
            self.v_weights[i] = self.momentum * self.v_weights[i] + self.lr * dW
            self.v_biases[i] = self.momentum * self.v_biases[i] + self.lr * db
            self.weights[i] -= self.v_weights[i]
            self.biases[i] -= self.v_biases[i]

    def train(self, X, y, epochs=1000, batch_size=None):
        m = X.shape[0]
        for _ in range(epochs):
            if batch_size is None or batch_size >= m:
                self.forward(X)
                self.backward(X, y)
            else:
                indices = np.random.permutation(m)
                X_s, y_s = X[indices], y[indices]
                for i in range(0, m, batch_size):
                    self.forward(X_s[i:i+batch_size])
                    self.backward(X_s[i:i+batch_size], y_s[i:i+batch_size])
            
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# ==========================================
# 3. WCZYTANIE DANYCH
# ==========================================
try:
    df = pd.read_csv('auta_bez_duplikatow.csv', sep=';')
except FileNotFoundError:
    print("Błąd: Upewnij się, że masz plik 'auta_bez_duplikatow.csv'")
    # Tworzymy mock-dane, aby skrypt nie zepsuł się, jeśli użytkownik nie ma pliku w czasie testowania
    print("Tworzę losowe dane testowe (zastąp to własnym plikiem)...")
    df = pd.DataFrame(np.random.randn(100, 8), columns=['masa', 'rok_produkcji', 'dlugosc', 'wysokosc', 'szerokosc', 'liczba_drzwi', 'bagaznik', 'model'])
    df['rynek'] = np.random.choice(['EU', 'US', 'AS'], 100)
    df['nadwozie'] = np.random.choice(['sedan', 'kombi', 'suv'], 100)

if 'model' in df.columns:
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
# 4. SILNIK EKSPERYMENTÓW
# ==========================================
def run_experiment(param_name, values, repeats=3):
    print(f"\n[{param_name.upper()}] Badanie wpływu parametru...")
    for val in values:
        train_accs, test_accs = [], []
        
        for _ in range(repeats):
            # 1. Domyślne wartości
            t_size, h_layers, lr, ep, act, w_init, b_size, mom = 0.2, [16], 0.1, 500, 'sigmoid', 'normal', None, 0.0
            
            # 2. Podmiana badanego parametru
            if param_name == 'hidden_layers': h_layers = val
            elif param_name == 'test_size': t_size = val
            elif param_name == 'learning_rate': lr = val
            elif param_name == 'epochs': ep = val
            elif param_name == 'activation': act = val
            elif param_name == 'weight_init': w_init = val
            elif param_name == 'batch_size': b_size = val
            elif param_name == 'momentum': mom = val

            # 3. Przygotowanie danych
            X_tr, X_te, y_tr, y_te = train_test_split(X, y_onehot, test_size=t_size, random_state=None)
            
            prep = ColumnTransformer([
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])
            X_tr_sc = prep.fit_transform(X_tr)
            X_te_sc = prep.transform(X_te)
            
            # Teraz wiemy ile mamy wejść
            input_size = X_tr_sc.shape[1]

            # 4. Inicjalizacja nowej, dynamicznej klasy
            nn = DynamicNeuralNetwork(input_size, h_layers, num_classes, lr, act, w_init, mom)
            
            # 5. Uczenie
            nn.train(X_tr_sc, y_tr, epochs=ep, batch_size=b_size)
            
            # 6. Wyniki
            pred_train = nn.predict(X_tr_sc)
            pred_test = nn.predict(X_te_sc)
            
            train_accs.append(np.mean(pred_train == np.argmax(y_tr, axis=1)))
            test_accs.append(np.mean(pred_test == np.argmax(y_te, axis=1)))

        # Wypisanie statystyk
        print(f" Wartość: {str(val):<12} | TEST (Śr/Max): {np.mean(test_accs)*100:>5.1f}% / {np.max(test_accs)*100:>5.1f}% | TRENING: {np.mean(train_accs)*100:>5.1f}%")

print("=== PROJEKT SSN: START BADANIA 8 PARAMETRÓW ===")

# 1. Warstwy
print("\n[WARSTWY] Badanie wpływu liczby i rozmiaru warstw...")
run_experiment('hidden_layers', [
    [16],              # 1 warstwa
    [16, 16],          # 2 warstwy
    [32, 16, 8],       # 3 warstwy (zwężająca się)
    [8, 8, 8, 8]       # 4 warstwy
])

# 2. Współczynnik uczenia
run_experiment('learning_rate', [0.01, 0.05, 0.1, 0.5])

# 3. Liczba epok
run_experiment('epochs', [100, 300, 600, 1000])

# 4. Sposób doboru próby testowej
run_experiment('test_size', [0.1, 0.2, 0.3, 0.4])

# 5. Funkcja aktywacji w warstwie ukrytej
run_experiment('activation', ['sigmoid', 'tanh', 'relu', 'leaky_relu'])

# 6. Sposób inicjalizacji wag
run_experiment('weight_init', ['normal', 'xavier', 'he', 'orthogonal'])

# 7. Rozmiar mini-batcha
run_experiment('batch_size', [None, 32, 64, 128]) 

# 8. Współczynnik Momentum
run_experiment('momentum', [0.0, 0.5, 0.9, 0.99])

print("=== KONIEC BADAŃ ===")

# ==========================================
# 5. MODEL FINALNY (NAJLEPSZY VS BAZOWY)
# ==========================================
print("\n=== PODSUMOWANIE: MODEL BAZOWY VS NAJLEPSZY ===")

X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

prep_final = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

X_final_train = prep_final.fit_transform(X_f_train)
X_final_test = prep_final.transform(X_f_test)
input_size = X_final_train.shape[1] 

best_params = {
    'h_layers': [64],     
    'lr': 0.5,           
    'ep': 1000,          
    'act': 'tanh',       
    'w_init': 'xavier',  
    'b_size': 32,        
    'mom': 0.9           
}

# 1. Trenujemy model BAZOWY (1 warstwa, 16 neuronów)
nn_base = DynamicNeuralNetwork(input_size, [16], num_classes, 0.1, 'sigmoid', 'normal', 0.0)
nn_base.train(X_final_train, y_f_train, epochs=500)
acc_base = np.mean(nn_base.predict(X_final_test) == np.argmax(y_f_test, axis=1))

# 2. Trenujemy model NAJLEPSZY
nn_best = DynamicNeuralNetwork(
    input_size, 
    best_params['h_layers'], 
    num_classes, 
    best_params['lr'], 
    best_params['act'], 
    best_params['w_init'], 
    best_params['mom']
)
nn_best.train(X_final_train, y_f_train, epochs=best_params['ep'], batch_size=best_params['b_size'])
acc_best = np.mean(nn_best.predict(X_final_test) == np.argmax(y_f_test, axis=1))

# --- WYŚWIETLENIE PORÓWNANIA ---
print(f"\n{'PARAMETR':<20} | {'MODEL BAZOWY':<15} | {'MODEL NAJLEPSZY'}")
print("-" * 65)
print(f"{'Struktura warstw':<20} | {'[16]':<15} | {str(best_params['h_layers'])}")
print(f"{'Learning Rate':<20} | {'0.1':<15} | {best_params['lr']}")
print(f"{'Liczba epok':<20} | {'500':<15} | {best_params['ep']}")
print(f"{'Activation':<20} | {'sigmoid':<15} | {best_params['act']}")
print(f"{'Weight Init':<20} | {'normal':<15} | {best_params['w_init']}")
print(f"{'Batch Size':<20} | {'Full Batch':<15} | {best_params['b_size']}")
print(f"{'Momentum':<20} | {'0.0':<15} | {best_params['mom']}")
print("-" * 65)
print(f"{'Accuracy (TEST)':<20} | {acc_base*100:>14.2f}% | {acc_best*100:>14.2f}%")

zysk = (acc_best - acc_base) * 100
print(f"\nDzięki optymalizacji uzyskano poprawę o: {zysk:.2f} punktów procentowych.")