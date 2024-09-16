import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns
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

# Afficher l'accuracy et un rapport de classification
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Générer la matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# Afficher et sauvegarder la matrice de confusion sous forme de figure
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('/app/output/confusion_matrix.png')  # Sauvegarder l'image
