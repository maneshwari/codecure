# CodeCure — Drug Toxicity Predictor
Predicts potential drug toxicity across 12 biological targets using molecular SMILES strings.

## Approach
- **Dataset**: Tox21 (7,831 compounds, 12 toxicity endpoints)
- **Features**: Morgan Fingerprints (radius=2, 2048 bits) via RDKit
- **Model**: Per-target XGBoost with class-imbalance handling (scale_pos_weight)
- **Evaluation**: ROC-AUC (chosen over accuracy due to severe class imbalance — some targets have <3% positive rate)

## Results
| Target | ROC-AUC |
|--------|---------|
| NR-AR-LBD | 0.8776 |
| NR-AhR | 0.8736 |
| SR-MMP | 0.8199 |
| SR-ATAD5 | 0.7861 |
| NR-Aromatase | 0.7784 |
| SR-p53 | 0.7659 |
| NR-ER-LBD | 0.7610 |
| SR-ARE | 0.7440 |
| NR-PPAR-gamma | 0.7159 |
| SR-HSE | 0.7017 |
| NR-AR | 0.6928 |
| NR-ER | 0.6715 |
| **Mean** | **0.7657** |

## Run Locally
```bash
conda create -n codecure python=3.10
conda activate codecure
conda install -c conda-forge rdkit
pip install pandas numpy scikit-learn xgboost shap flask flask-cors
python app.py
open frontend.html
```

## API
```
POST http://localhost:8080/predict
{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}
```

## ⚡ Keeping the API alive during demo

Run this in a fresh terminal before every demo session:
```bash
cd ~/codecure-project
conda activate codecure
python app.py > /tmp/flask.log 2>&1 &
echo "API running at http://localhost:8080"
```

> ⚠️ Don't close this terminal during the demo. Judges can also just run `python app.py` themselves.
