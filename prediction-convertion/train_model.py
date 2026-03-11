import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pickle
import os

df = pd.read_csv('prediction-convertion/leads_dataset.csv', low_memory=False)

print("=" * 50)
print("     COMPARAISON MODÈLES DE CONVERSION")
print("=" * 50)

# ── Encodages ────────────────────────────────────────
doc_status_map = {'legitimate': 2, 'suspicious': 1, 'fraud': 0}
df['doc_status_encoded'] = df['doc_status'].map(doc_status_map)

if 'doc_fraud_flag' not in df.columns:
    df['doc_fraud_flag'] = (df['doc_status'] == 'fraud').astype(int)

emp_map = {
    'Employed (CDI)'  : 5,
    'Self-Employed'   : 4,
    'Retired'         : 3,
    'Employed (CDD)'  : 2,
    'Student'         : 1,
    'Unemployed'      : 0
}
df['employment_encoded'] = df['employment_status'].map(emp_map).fillna(0)
df['has_guarantor'] = df['has_guarantor'].map(
    {True: 1, False: 0, 'True': 1, 'False': 0}
).fillna(0).astype(int)


def evaluate_models(X_train, X_test, y_train, y_test, label):
    print(f"\n── Modèle {label} ──────────────────────────────")

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"Après SMOTE : {y_res.value_counts().to_dict()}")

    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    models = {
        'XGBoost': XGBClassifier(
            n_estimators     = 500,
            learning_rate    = 0.03,
            max_depth        = 5,
            scale_pos_weight = scale,
            use_label_encoder= False,
            eval_metric      = 'logloss',
            random_state     = 42,
            verbosity        = 0
        ),
        'LightGBM': LGBMClassifier(
            n_estimators  = 500,
            learning_rate = 0.03,
            max_depth     = 5,
            class_weight  = 'balanced',
            random_state  = 42,
            verbose       = -1
        ),
        'CatBoost': CatBoostClassifier(
            iterations        = 500,
            learning_rate     = 0.03,
            depth             = 5,
            auto_class_weights= 'Balanced',
            random_seed       = 42,
            verbose           = 0
        )
    }

    results = {}
    for name, model in models.items():
        model.fit(X_res, y_res)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc     = accuracy_score(y_test, y_pred) * 100
        auc     = roc_auc_score(y_test, y_proba)
        print(f"\n{name}:")
        print(f"  Accuracy : {acc:.2f}%")
        print(f"  AUC      : {auc:.4f}")
        print(classification_report(y_test, y_pred))
        results[name] = {'model': model, 'acc': acc, 'auc': auc}

    best_name  = max(results, key=lambda x: results[x]['auc'])
    best_model = results[best_name]['model']
    print(f"\n✅ Meilleur modèle {label} : {best_name} "
          f"(AUC={results[best_name]['auc']:.4f})")

    # ── Importance features ──────────────────────────
    try:
        feat_names = list(X_train.columns)
        feat_imp   = best_model.feature_importances_
        total      = sum(feat_imp)
        print(f"\n📊 Importance features ({label}) :")
        for feat, imp in sorted(
            zip(feat_names, feat_imp),
            key=lambda x: x[1], reverse=True
        ):
            pct = imp / total * 100
            print(f"  {feat:30s} : {pct:5.1f}%")
    except Exception:
        pass

    # ── Vérification critique post-entraînement ─────
    print(f"\n🔍 Vérification fraud (test set) :")
    fraud_mask = X_test['doc_fraud_flag'] == 1
    if fraud_mask.sum() > 0:
        fraud_proba = best_model.predict_proba(
            X_test[fraud_mask]
        )[:, 1]
        print(f"  Leads fraud         : {fraud_mask.sum()}")
        print(f"  Prob moyenne        : "
              f"{fraud_proba.mean()*100:.2f}%")
        print(f"  Prob max            : "
              f"{fraud_proba.max()*100:.2f}%")
        print(f"  Prob > 20%          : "
              f"{(fraud_proba > 0.20).sum()} leads ⚠️")

    return best_model, best_name


# ════════════════════════════════════════════════════
#  BUY
# ════════════════════════════════════════════════════
FEATURES_BUY = [
    'doc_fraud_flag',
    'doc_status_encoded',
    'ai_score',
    'fifo_rank',
]

buy_df = df[df['application_type'] == 'Buy'].copy()
X_buy  = buy_df[FEATURES_BUY]
y_buy  = buy_df['is_converted']

X_buy_train, X_buy_test, y_buy_train, y_buy_test = \
    train_test_split(X_buy, y_buy, test_size=0.2, random_state=42)

best_buy, best_buy_name = evaluate_models(
    X_buy_train, X_buy_test,
    y_buy_train, y_buy_test,
    'BUY'
)

# ════════════════════════════════════════════════════
#  RENT
# ════════════════════════════════════════════════════
FEATURES_RENT = [
    'doc_fraud_flag',
    'doc_status_encoded',
    'ai_score',
    'employment_encoded',
    'has_guarantor',
    'fifo_rank',
]

rent_df = df[df['application_type'] == 'Rent'].copy()
X_rent  = rent_df[FEATURES_RENT]
y_rent  = rent_df['is_converted']

X_rent_train, X_rent_test, y_rent_train, y_rent_test = \
    train_test_split(X_rent, y_rent, test_size=0.2, random_state=42)

best_rent, best_rent_name = evaluate_models(
    X_rent_train, X_rent_test,
    y_rent_train, y_rent_test,
    'RENT'
)

# ════════════════════════════════════════════════════
#  SAUVEGARDE
# ════════════════════════════════════════════════════
os.makedirs('prediction-convertion/model', exist_ok=True)

with open('prediction-convertion/model/model_buy.pkl', 'wb') as f:
    pickle.dump(best_buy, f)
with open('prediction-convertion/model/model_rent.pkl', 'wb') as f:
    pickle.dump(best_rent, f)
with open('prediction-convertion/model/features_buy.pkl', 'wb') as f:
    pickle.dump(FEATURES_BUY, f)
with open('prediction-convertion/model/features_rent.pkl', 'wb') as f:
    pickle.dump(FEATURES_RENT, f)
with open('prediction-convertion/model/emp_map.pkl', 'wb') as f:
    pickle.dump(emp_map, f)

print("\n" + "=" * 50)
print(f"✅ Modèle Buy  : {best_buy_name}  | Features : {FEATURES_BUY}")
print(f"✅ Modèle Rent : {best_rent_name} | Features : {FEATURES_RENT}")
print("=" * 50)