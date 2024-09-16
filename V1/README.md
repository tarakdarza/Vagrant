# Projet Algo Container

## Description
Ce projet illustre comment créer un conteneur Docker qui exécute un algorithme de machine learning (apprentissage automatique) en Python. L'algorithme utilisé ici est un classificateur **Random Forest**, entraîné sur le dataset **Iris**.

## Prérequis
Avant de commencer, assurez-vous d'avoir les éléments suivants installés sur votre machine :
- [Docker](https://www.docker.com/products/docker-desktop)
- [Visual Studio Code (VSCode)](https://code.visualstudio.com/)
- Connaissances de base en Python et Docker

## Structure du projet
Voici les principaux fichiers qui composent le projet :
|-- script.py # 
Python script containing the machine learning algorithm 
|-- Dockerfile # Dockerfile to create the Docker image 
|-- requirements.txt # List of Python dependencies to install in the Docker container 

|-- README.md # Documentation file explaining the steps to reproduce the project

onstruire une image Docker à partir du Dockerfile situé dans le répertoire courant (indiqué par .) et la nommer algo_container.

Voici ce qui se passe :

docker build : Cette commande crée une nouvelle image Docker en fonction du Dockerfile dans le répertoire spécifié.
-t algo_container : L'option -t permet de donner un nom (tag) à l'image, ici algo_container. Tu pourras utiliser ce nom plus tard pour lancer un conteneur à partir de cette image.
. (dot) : Ce point représente le contexte de construction, c’est-à-dire le répertoire où Docker va chercher le Dockerfile et tous les fichiers nécessaires à la construction de l'image.
Si tout se passe bien, voici ce qui se passera ensuite :
Docker va lire ton Dockerfile et exécuter les instructions qui y sont définies.
Chaque commande dans le Dockerfile (comme FROM, RUN, COPY, etc.) crée une nouvelle couche pour l'image.
À la fin du processus, l'image Docker sera construite et stockée localement avec le nom/tag algo_container.
Si tu n'as pas encore de Dockerfile, tu devras en créer un dans le répertoire courant pour que cette commande fonctionne.

Exemple de Dockerfile
Voici un exemple de base d'un Dockerfile :

Dockerfile
Copier le code
# Utiliser une image de base (par exemple Python)
FROM python:3.9

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances via requirements.txt
RUN pip install -r requirements.txt

# Définir la commande par défaut à exécuter


docker build -t algo_container:v1.0.0 .
docker build : Crée une nouvelle image Docker en fonction du Dockerfile dans le répertoire actuel.
-t algo_container:v1.0.0 : Assigne le nom algo_container avec le tag v1.0.0 à l'image construite.
. (dot) : Indique que le répertoire courant contient le Dockerfile et le contexte de construction.

Une fois l'image construite avec succès, tu peux lancer un conteneur avec cette commande :

docker run -d algo_container:v1.0.0
