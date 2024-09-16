# Utiliser l'image Python de base
FROM python:3.9-slim

# Installer les bibliothèques nécessaires
RUN pip install --no-cache-dir flask pandas scikit-learn matplotlib seaborn

# Créer le répertoire de travail dans le conteneur
WORKDIR /app

# Copier tous les fichiers dans le conteneur
COPY . /app

# Créer les répertoires nécessaires pour les fichiers statiques
RUN mkdir -p /app/static/output

# Exposer le port pour Flask
EXPOSE 5000

# Exécuter l'application Flask
CMD ["python", "app.py"]
