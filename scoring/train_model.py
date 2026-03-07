import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Charger le dataset ────────────────────────────────
print("=" * 50)
print("       ENTRAÎNEMENT DU MODÈLE ML")
print("=" * 50)

df = pd.read_csv('scoring/leads_dataset.csv')
print(f"Dataset chargé : {len(df)} leads")

# ── Feature Engineering ───────────────────────────────
# Nouvelles features combinées
df['budget_x_docs']     = (
    df['budget_normalized'].fillna(0.4) * df['docs_score']
)
df['profile_x_docs']    = (
    df['profile_complete'] / 100 * df['docs_score']
)
df['city_x_budget']     = (
    df['city_score'] * df['budget_normalized'].fillna(0.4)
)
df['engagement_score']  = (
    df['has_email'] * 0.3 +
    df['has_phone'] * 0.3 +
    df['has_address'] * 0.2 +
    df['is_business_hour'] * 0.2
)

# ── Features enrichies ────────────────────────────────
features = [
    'listing_type_encoded',
    'city_score',
    'has_city',
    'budget_normalized',
    'has_budget',
    'has_email',
    'has_phone',
    'has_address',
    'profile_complete',
    'docs_uploaded',
    'docs_score',
    'is_long_term',
    'has_move_in_date',
    'is_business_hour',
    'fifo_score',
    # Nouvelles features
    'budget_x_docs',
    'profile_x_docs',
    'city_x_budget',
    'engagement_score'
]

# ── Encodage ──────────────────────────────────────────
le = LabelEncoder()
df['listing_type_encoded'] = le.fit_transform(df['listing_type'])

X = df[features]
y = df['converted']

# ── Split ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y
)

print(f"Train : {len(X_train)} | Test : {len(X_test)}")

# ── 3 Modèles à comparer ──────────────────────────────
models = {
    'Random Forest': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(
            n_estimators = 500,
            max_depth    = 12,
            min_samples_split = 5,
            min_samples_leaf  = 2,
            random_state = 42,
            class_weight = 'balanced'
        ))
    ]),
    'XGBoost': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', XGBClassifier(
            n_estimators      = 500,
            max_depth         = 6,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            random_state      = 42,
            eval_metric       = 'logloss',
            verbosity         = 0
        ))
    ]),
    'Gradient Boosting': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', GradientBoostingClassifier(
            n_estimators  = 300,
            max_depth     = 5,
            learning_rate = 0.05,
            subsample     = 0.8,
            random_state  = 42
        ))
    ])
}

# ── Entraînement et comparaison ───────────────────────
print("\nComparaison des modèles...")
print("-" * 50)

best_model    = None
best_accuracy = 0
best_name     = ""
results       = {}

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred       = pipeline.predict(X_test)
    y_proba      = pipeline.predict_proba(X_test)[:, 1]
    acc          = accuracy_score(y_test, y_pred)
    auc          = roc_auc_score(y_test, y_proba)
    results[name]= {'accuracy': acc, 'auc': auc, 'pipeline': pipeline}

    print(f"{name:20} | Accuracy: {acc*100:.2f}% | AUC: {auc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model    = pipeline
        best_name     = name

print("-" * 50)
print(f"\nMeilleur modèle : {best_name} ({best_accuracy*100:.2f}%)")

# ── Résultats détaillés du meilleur modèle ────────────
print("\n" + "=" * 50)
print(f"    RÉSULTATS : {best_name}")
print("=" * 50)

y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print(f"Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"AUC Score : {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report :")
print(classification_report(
    y_test, y_pred,
    target_names=['Non Converti', 'Converti']
))

# ── Matrice de confusion ──────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm,
    annot      = True,
    fmt        = 'd',
    cmap       = 'Blues',
    xticklabels= ['Non Converti', 'Converti'],
    yticklabels= ['Non Converti', 'Converti']
)
plt.title(f'Matrice de Confusion — {best_name}')
plt.ylabel('Réel')
plt.xlabel('Prédit')
plt.tight_layout()
plt.savefig('scoring/confusion_matrix.png')
print("Matrice de confusion sauvegardée ✅")

# ── Feature Importance ────────────────────────────────
try:
    importances = best_model.named_steps['model'].feature_importances_
    feat_imp    = pd.DataFrame({
        'feature'   : features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\nImportance des features :")
    print(feat_imp.to_string(index=False))

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data    = feat_imp,
        x       = 'importance',
        y       = 'feature',
        hue     = 'feature',
        legend  = False,
        palette = 'viridis'
    )
    plt.title('Importance des Features')
    plt.tight_layout()
    plt.savefig('scoring/feature_importance.png')
    print("Feature importance sauvegardée ✅")
except:
    pass

# ── Fonction de scoring ───────────────────────────────
def predict_lead_score(lead_data):
    df_lead = pd.DataFrame([lead_data])

    # Feature Engineering
    b = lead_data.get('budget_normalized') or 0.4
    df_lead['budget_x_docs']    = b * lead_data.get('docs_score', 0)
    df_lead['profile_x_docs']   = (
        lead_data.get('profile_complete', 0) / 100
        * lead_data.get('docs_score', 0)
    )
    df_lead['city_x_budget']    = lead_data.get('city_score', 0.5) * b
    df_lead['engagement_score'] = (
        lead_data.get('has_email', 0)         * 0.3 +
        lead_data.get('has_phone', 0)         * 0.3 +
        lead_data.get('has_address', 0)       * 0.2 +
        lead_data.get('is_business_hour', 0)  * 0.2
    )
    df_lead['listing_type_encoded'] = (
        1 if lead_data.get('listing_type') == 'Buy' else 0
    )

    X_lead = df_lead[features]
    proba  = best_model.predict_proba(X_lead)[0][1]
    score  = round(proba * 100)

    if score >= 75:
        category = 'Hot'
    elif score >= 50:
        category = 'Warm'
    else:
        category = 'Cold'

    missing = sum([
        lead_data.get('budget_normalized') is None,
        lead_data.get('is_long_term')      is None,
        not lead_data.get('has_city', True)
    ])
    reliability = round(100 - (missing * 15))

    return {
        'score'      : score,
        'category'   : category,
        'probability': round(proba, 4),
        'reliability': reliability
    }

# ── Test ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("         TEST DE PRÉDICTION")
print("=" * 50)

test_leads = [
    {
        'name'              : 'Lead Complet (Hot)',
        'listing_type'      : 'Buy',
        'city_score'        : 1.0,
        'has_city'          : 1,
        'budget_normalized' : 0.8,
        'has_budget'        : 1,
        'has_email'         : 1,
        'has_phone'         : 1,
        'has_address'       : 1,
        'profile_complete'  : 100,
        'docs_uploaded'     : 5,
        'docs_score'        : 1.0,
        'is_long_term'      : 1,
        'has_move_in_date'  : 1,
        'is_business_hour'  : 1,
        'fifo_score'        : 0.95
    },
    {
        'name'              : 'Lead Partiel (Warm)',
        'listing_type'      : 'Rent',
        'city_score'        : 0.75,
        'has_city'          : 1,
        'budget_normalized' : None,
        'has_budget'        : 0,
        'has_email'         : 1,
        'has_phone'         : 0,
        'has_address'       : 1,
        'profile_complete'  : 60,
        'docs_uploaded'     : 2,
        'docs_score'        : 0.4,
        'is_long_term'      : None,
        'has_move_in_date'  : 0,
        'is_business_hour'  : 1,
        'fifo_score'        : 0.5
    },
    {
        'name'              : 'Lead Incomplet (Cold)',
        'listing_type'      : 'Rent',
        'city_score'        : 0.3,
        'has_city'          : 0,
        'budget_normalized' : None,
        'has_budget'        : 0,
        'has_email'         : 1,
        'has_phone'         : 0,
        'has_address'       : 0,
        'profile_complete'  : 30,
        'docs_uploaded'     : 0,
        'docs_score'        : 0.0,
        'is_long_term'      : None,
        'has_move_in_date'  : 0,
        'is_business_hour'  : 0,
        'fifo_score'        : 0.1
    }
]

for lead in test_leads:
    result = predict_lead_score(lead)
    print(f"\n{lead['name']} :")
    print(f"  Score       : {result['score']}/100")
    print(f"  Catégorie   : {result['category']}")
    print(f"  Probabilité : {result['probability']}")
    print(f"  Fiabilité   : {result['reliability']}%")

# ── Sauvegarder ───────────────────────────────────────
os.makedirs('scoring/model', exist_ok=True)
joblib.dump(best_model, 'scoring/model/lead_scoring_model.pkl')
joblib.dump(le,         'scoring/model/label_encoder.pkl')
joblib.dump(features,   'scoring/model/features.pkl')

print("\n" + "=" * 50)
print(f"Modèle '{best_name}' sauvegardé ✅")
print("=" * 50)