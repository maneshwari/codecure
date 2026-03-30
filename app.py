from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

with open("models/xgb_models.pkl", "rb") as f:
    models = pickle.load(f)

generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
TARGETS = ["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase","NR-ER","NR-ER-LBD",
           "NR-PPAR-gamma","SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    smiles = data.get("smiles", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({"error": "Invalid SMILES"}), 400
    fp = generator.GetFingerprintAsNumPy(mol).reshape(1, -1)
    predictions = {}
    for target in TARGETS:
        if target in models:
            prob = float(models[target].predict_proba(fp)[0][1])
            predictions[target] = round(prob, 4)
    risk_score = round(np.mean(list(predictions.values())), 4)
    high_risk = [t for t, p in predictions.items() if p > 0.5]
    return jsonify({"smiles": smiles, "predictions": predictions,
                    "overall_risk_score": risk_score, "high_risk_targets": high_risk})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": len(models)})

if __name__ == "__main__":
    app.run(debug=True, port=8080)
