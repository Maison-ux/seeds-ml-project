import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==============================
# 1. CARREGAR DADOS
# ==============================

df = pd.read_csv("Data/seeds.csv", sep="\s+")

print("\nColunas do dataset:")
print(df.columns)

# ==============================
# 2. DEFINIR TARGET AUTOMATICAMENTE
# ==============================

# Assume que a última coluna é o target (padrão comum em datasets)
target_column = df.columns[-1]

print(f"\nColuna target detectada: {target_column}")

X = df.drop(target_column, axis=1)
y = df[target_column]

# ==============================
# 3. DIVIDIR DADOS
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 4. MODELO
# ==============================

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ==============================
# 5. PREVISÃO
# ==============================

y_pred = model.predict(X_test)

# ==============================
# 6. AVALIAÇÃO
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAcurácia do modelo: {accuracy:.2f}")

print(df.head())

