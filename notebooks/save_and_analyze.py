import pickle, numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/tox21.csv')
generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

fps, valid_idx = [], []
for i, smi in enumerate(df['smiles']):
    mol = Chem.MolFromSmiles(str(smi))
    if mol:
        fps.append(generator.GetFingerprintAsNumPy(mol))
        valid_idx.append(i)

X = np.array(fps)
df_valid = df.iloc[valid_idx].reset_index(drop=True)

TARGETS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
           'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

models = {}
results = {}

for target in TARGETS:
    y = df_valid[target].values
    mask = ~np.isnan(y)
    X_t, y_t = X[mask], y[mask].astype(int)
    if y_t.sum() < 10:
        continue
    X_train, X_test, y_train, y_test = train_test_split(
        X_t, y_t, test_size=0.2, random_state=42, stratify=y_t)
    sw = compute_sample_weight('balanced', y_train)
    pos_ratio = (y_t==0).sum() / (y_t==1).sum()
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
        scale_pos_weight=pos_ratio, eval_metric='auc', random_state=42, verbosity=0)
    model.fit(X_train, y_train, sample_weight=sw)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    results[target] = round(auc, 4)
    models[target] = model
    print(f"{target}: {auc:.4f}")

# Save models
with open('models/xgb_models.pkl', 'wb') as f:
    pickle.dump(models, f)
print("\nModels saved.")

# Feature importance from best model
best_target = max(results, key=results.get)
best_model = models[best_target]
importance = best_model.feature_importances_
top20 = np.argsort(importance)[::-1][:20]

print(f"\nTop 20 important Morgan bits for {best_target}:")
for rank, bit in enumerate(top20):
    print(f"  Rank {rank+1}: Bit {bit}  (importance={importance[bit]:.4f})")

# Save importance
np.save('models/feature_importance.npy', importance)
print("\nFeature importance saved.")
