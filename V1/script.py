# script.py
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Charger le dataset Iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Diviser les données en train et test
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer le modèle et entraîner
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Afficher les résultats
print("Accuracy:", accuracy_score(y_test, y_pred))
