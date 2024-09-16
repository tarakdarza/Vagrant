from flask import Flask, render_template, request
from script_with_figures import train_model
import os  # Ajoutez l'importation d'os pour vérifier les fichiers

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_model', methods=['POST'])
def run_model():
    # Exécuter le modèle
    accuracy, report = train_model()

    # Vérifier si le fichier ROC existe
    roc_curve_exists = os.path.exists('/app/static/output/roc_curve.png')
    
    # Passer la variable roc_curve_exists au template
    return render_template('results.html', accuracy=accuracy, report=report, roc_curve_exists=roc_curve_exists)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
