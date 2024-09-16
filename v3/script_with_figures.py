import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distribution(df):
    plt.figure(figsize=(10, 7))
    df.hist(bins=20, figsize=(15, 10), color="blue")
    plt.tight_layout()
    plt.savefig('/app/static/output/feature_distribution.png')  # Sauvegarder dans static/output

def plot_roc_curve(y_test, y_pred_prob):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('/app/static/output/roc_curve.png')  # Sauvegarder dans static/output

def plot_class_distribution(y):
    plt.figure(figsize=(8, 6))
    sns.countplot(y)
    plt.title('Distribution des classes')
    plt.savefig('/app/static/output/class_distribution.png')  # Sauvegarder dans static/output

def train_model():
    # Charger le dataset Iris
    from sklearn.datasets import load_iris
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

    # Calculer l'accuracy et le rapport
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Générer la matrice de confusion
    cm = sns.heatmap(pd.DataFrame(confusion_matrix(y_test, y_pred)), annot=True, cmap="Blues")
    plt.savefig('/app/static/output/confusion_matrix.png')  # Sauvegarder l'image
    
    # Générer des graphiques supplémentaires
    plot_feature_distribution(df)  # Distribution des caractéristiques
    plot_class_distribution(y)     # Distribution des classes
    
    # Si c'est une classification binaire, générer la courbe ROC
    if len(set(y)) == 2:  # Vérifiez que vous avez une classification binaire
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_pred_prob)  # Générer et sauvegarder la courbe ROC
    else:
        print("La classification n'est pas binaire, courbe ROC non générée.")
    
    return accuracy, report
