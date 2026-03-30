import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import shap
import pickle
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/tox21.csv')
print(f"Loaded: {df.shape}")

TARGETS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
           'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']
generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = generator.GetFingerprintAsNumPy(mol)
    return fp

print("Generating fingerprints...")
fps = []
valid_idx = []
for i, smi in enumerate(df['smiles']):
    fp = smiles_to_fp(str(smi))
    if fp is not None:
        fps.append(fp)
        valid_idx.append(i)

X = np.array(fps)
df_valid = df.iloc[valid_idx].reset_index(drop=True)
print(f"Valid molecules: {len(X)}")
results = {}
models = {}

for target in TARGETS:
    y = df_valid[target].values
    mask = ~np.isnan(y)
    X_t = X[mask]
    y_t = y[mask].astype(int)

    if y_t.sum() < 10:
        print(f"{target}: skipped (too few positives)")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X_t, y_t, test_size=0.2, random_state=42, stratify=y_t)

    # handle class imbalance
    sample_weights = compute_sample_weight('balanced', y_train)
    pos_ratio = (y_t == 0).sum() / (y_t == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,learning_rate=0.05,
        scale_pos_weight=pos_ratio,
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    results[target] = round(auc, 4)
    models[target] = model
    print(f"{target}: ROC-AUC = {auc:.4f}  (n={len(y_t)}, pos={y_t.sum()})")
print("\n── Results Summary ──")
for t, auc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {t:<20} {auc}")
print(f"\nMean AUC: {np.mean(list(results.values())):.4f}")
best_target = max(results, key=results.get)
print(f"\nRunning SHAP on best model: {best_target}")
best_model = models[best_target]

y_best = df_valid[best_target].values
mask_best = ~np.isnan(y_best)
X_best = X[mask_best]
explainer = shap.TreeExplainer(best_model, feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(X_best[:200])
shap_importance = np.abs(shap_values).mean(axis=0)
top_features = np.argsort(shap_importance)[::-1][:20]
print(f"Top 20 important bit indices: {top_features.tolist()}")
with open('models/xgb_models.pkl', 'wb') as f:
    pickle.dump(models, f)
print("\nModels saved to models/xgb_models.pkl")
