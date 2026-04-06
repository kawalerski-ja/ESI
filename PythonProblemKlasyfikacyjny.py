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
# 1. WCZYTANIE DANYCH Z PLIKU
# ==========================================
# Wczytujemy Twój połączony plik CSV
try:
    df = pd.read_csv('wszystkie_auta_tomek.csv', sep=';')
except FileNotFoundError:
    print("Błąd: Nie znaleziono pliku 'wszystkie_auta_tomek.csv'. Upewnij się, że plik jest w tym samym folderze.")
    exit()

# Usuwamy kolumnę 'model', ponieważ nie jest cechą statystyczną (to nazwa własna)
df = df.drop('model', axis=1)

# ==========================================
# 2. PRZYGOTOWANIE DANYCH (PREPROCESSING)
# ==========================================
X = df.drop('nadwozie', axis=1)
y = df['nadwozie']

# Podział na zbiór uczący (80%) i testowy (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definiujemy które kolumny są liczbowe, a które kategoryczne
numeric_features = ['masa', 'rok_produkcji', 'dlugosc', 'wysokosc', 'szerokosc', 'liczba_drzwi', 'bagaznik']
categorical_features = ['rynek']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
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
    
    print(f"{model_name:<12} | {param_name}: {str(param_value):<10} | Uczący: {acc_train*100:>5.2f}% | Testowy: {acc_test*100:>5.2f}%")


print(f"--- WYNIKI EKSPERYMENTÓW NA DANYCH (Liczba aut: {len(df)}) ---")

# 1. Drzewa decyzyjne
print("\n1. Drzewa Decyzyjne (max_depth)")
depths = [3, 5, 10, None]
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    evaluate_model("Drzewo", clf, "max_depth", d)

# 2. k-Najbliższych Sąsiadów
print("\n2. k-Najbliższych Sąsiadów (n_neighbors)")
neighbors = [1, 3, 5, 7]
for k in neighbors:
    clf = KNeighborsClassifier(n_neighbors=k)
    evaluate_model("k-NN", clf, "n_neighbors", k)

# 3. SVM
print("\n3. Maszyny Wektorów Nośnych (kernel)")
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for k in kernels:
    clf = SVC(kernel=k, random_state=42)
    evaluate_model("SVM", clf, "kernel", k)

# 4. Las Losowy
print("\n4. Las Losowy (n_estimators)")
estimators = [10, 50, 100, 200]
for n in estimators:
    clf = RandomForestClassifier(n_estimators=n, random_state=42)
    evaluate_model("Las Losowy", clf, "n_estimators", n)