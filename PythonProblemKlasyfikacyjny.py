import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. GENEROWANIE PRZYKŁADOWYCH DANYCH
# (Zastąpcie to potem wczytaniem własnego pliku CSV: pd.read_csv('dane.csv'))
# ==========================================
np.random.seed(42)
n_samples = 500

data = {
    'masa': np.random.randint(900, 2500, n_samples),
    'rok_produkcji': np.random.randint(2000, 2024, n_samples),
    'rynek': np.random.choice(['Europa', 'USA', 'Azja'], n_samples),
    'dlugosc': np.random.randint(3500, 5500, n_samples),
    'wysokosc': np.random.randint(1400, 1900, n_samples),
    'szerokosc': np.random.randint(1600, 2000, n_samples), # Zmieniony parametr!
    'liczba_drzwi': np.random.choice([3, 4, 5], n_samples),
    'bagaznik': np.random.randint(200, 700, n_samples),
    'nadwozie': np.random.choice(['Sedan', 'Kombi', 'SUV', 'Hatchback'], n_samples)
}
df = pd.DataFrame(data)

# ==========================================
# 2. PRZYGOTOWANIE DANYCH (PREPROCESSING)
# ==========================================
X = df.drop('nadwozie', axis=1)
y = df['nadwozie']

# Podział na zbiór uczący (80%) i testowy (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizacja danych liczbowych i One-Hot Encoding dla kategorycznych (Rynek)
numeric_features = ['masa', 'rok_produkcji', 'dlugosc', 'wysokosc', 'szerokosc', 'liczba_drzwi', 'bagaznik']
categorical_features = ['rynek']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Dopasowanie transformacji do danych treningowych i nałożenie jej na oba zbiory
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# ==========================================
# 3. TESTOWANIE MODELI I PARAMETRÓW
# ==========================================
def evaluate_model(model_name, model, param_name, param_value):
    """Funkcja pomocnicza do trenowania i wypisywania wyników"""
    model.fit(X_train_scaled, y_train)
    
    # Przewidywania
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Skuteczność (Accuracy)
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    
    print(f"{param_name}: {param_value:<10} | Uczący: {acc_train*100:>5.2f}% | Testowy: {acc_test*100:>5.2f}%")


print("--- WYNIKI EKSPERYMENTÓW ---")

# 1. Drzewa decyzyjne (Parametr: max_depth)
print("\n1. Drzewa Decyzyjne (Wpływ maksymalnej głębokości - max_depth)")
depths = [3, 5, 10, None]
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    evaluate_model("Drzewo", clf, "max_depth", str(d))

# 2. k-Najbliższych Sąsiadów (Parametr: n_neighbors)
print("\n2. k-Najbliższych Sąsiadów (Wpływ liczby sąsiadów - n_neighbors)")
neighbors = [1, 3, 5, 7]
for k in neighbors:
    clf = KNeighborsClassifier(n_neighbors=k)
    evaluate_model("k-NN", clf, "n_neighbors", k)

# 3. SVM (Parametr: kernel)
print("\n3. Maszyny Wektorów Nośnych (Wpływ funkcji jądra - kernel)")
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for k in kernels:
    clf = SVC(kernel=k, random_state=42)
    evaluate_model("SVM", clf, "kernel", k)

# 4. Las Losowy (Parametr: n_estimators)
print("\n4. Las Losowy (Wpływ liczby drzew - n_estimators)")
estimators = [10, 50, 100, 200]
for n in estimators:
    clf = RandomForestClassifier(n_estimators=n, random_state=42)
    evaluate_model("Las Losowy", clf, "n_estimators", n)
    #a