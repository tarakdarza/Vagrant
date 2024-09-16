# Utiliser l'image algo_container avec le tag v1.0.0
FROM algo_container:v1.0.0

# Installer les bibliothèques supplémentaires pour afficher des graphiques
RUN pip install --no-cache-dir matplotlib seaborn

# Copier le nouveau script Python avec les modifications
COPY script_with_figures.py /app/script_with_figures.py

# Exécuter le nouveau script Python
CMD ["python", "/app/script_with_figures.py"]


# Modifier votre Dockerfile pour créer une nouvelle image
# Plutôt que de repartir de zéro, vous pouvez modifier votre Dockerfile pour ajouter 
# les nouvelles bibliothèques et fonctionnalités nécessaires, comme matplotlib pour afficher ou sauvegarder des figures.
