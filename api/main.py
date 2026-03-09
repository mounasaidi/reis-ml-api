from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# ── Ajouter le path parent ────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraude.document_analyzer import analyze_document, analyze_all_documents
from fraude.rules import calculate_global_fraud_score

# ── Chemins modèle ────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(BASE_DIR, 'scoring', 'model', 'lead_scoring_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'scoring', 'model', 'features.pkl')

# ── Lazy loading ──────────────────────────────────────
model    = None
features = None

def get_model():
    global model, features
    if model is None:
        print("Entraînement du modèle ML...")
        import sys
        sys.path.append('/app')
        
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        
        # Charger dataset
        dataset_path = os.path.join(BASE_DIR, 'scoring', 'leads_dataset.csv')
        df = pd.read_csv(dataset_path)
        
        # Feature Engineering
        df['budget_x_docs']    = df['budget_normalized'].fillna(0.4) * df['docs_score']
        df['profile_x_docs']   = df['profile_complete'] / 100 * df['docs_score']
        df['city_x_budget']    = df['city_score'] * df['budget_normalized'].fillna(0.4)
        df['engagement_score'] = (
            df['has_email'] * 0.3 +
            df['has_phone'] * 0.3 +
            df['has_address'] * 0.2 +
            df['is_business_hour'] * 0.2
        )
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['listing_type_encoded'] = le.fit_transform(df['listing_type'])
        
        features = [
            'listing_type_encoded', 'city_score', 'has_city',
            'budget_normalized', 'has_budget', 'has_email',
            'has_phone', 'has_address', 'profile_complete',
            'docs_uploaded', 'docs_score', 'is_long_term',
            'has_move_in_date', 'is_business_hour', 'fifo_score',
            'budget_x_docs', 'profile_x_docs', 'city_x_budget',
            'engagement_score'
        ]
        
        X = df[features]
        y = df['converted']
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', GradientBoostingClassifier(
                n_estimators  = 300,
                max_depth     = 5,
                learning_rate = 0.05,
                subsample     = 0.8,
                random_state  = 42
            ))
        ])
        
        pipeline.fit(X, y)
        model = pipeline
        print("Modèle ML entraîné ✅")
    
    return model, features

# ── FastAPI App ───────────────────────────────────────
app = FastAPI(
    title       = "REIS ML API",
    description = "Lead Scoring + Fraud Detection",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

@app.on_event("startup")
async def startup_event():
    print("API démarrée ✅")
    get_model()

# ── Schéma Lead ───────────────────────────────────────
class LeadData(BaseModel):
    listing_type      : str            # Buy / Rent
    city              : Optional[str]  = None
    city_score        : Optional[float]= None
    has_city          : Optional[int]  = 0
    budget            : Optional[float]= None
    budget_normalized : Optional[float]= None
    has_budget        : Optional[int]  = 0
    has_email         : Optional[int]  = 0
    has_phone         : Optional[int]  = 0
    has_address       : Optional[int]  = 0
    profile_complete  : Optional[float]= 0
    docs_uploaded     : Optional[int]  = 0
    docs_score        : Optional[float]= 0
    is_long_term      : Optional[int]  = None
    has_move_in_date  : Optional[int]  = 0
    is_business_hour  : Optional[int]  = 0
    fifo_position     : Optional[int]  = 50
    fifo_score        : Optional[float]= None

# ── Villes connues ────────────────────────────────────
CITY_SCORES = {
    'tunis'    : 1.0,  'sfax'      : 0.85,
    'sousse'   : 0.90, 'monastir'  : 0.80,
    'bizerte'  : 0.75, 'nabeul'    : 0.78,
    'hammamet' : 0.82, 'la marsa'  : 0.95,
    'ariana'   : 0.88, 'ben arous' : 0.83,
}

def prepare_features(lead: LeadData):
    """Préparer les features pour le modèle"""

    # City score automatique
    city_score = lead.city_score
    has_city   = lead.has_city
    if lead.city and city_score is None:
        city_lower = lead.city.lower()
        city_score = CITY_SCORES.get(city_lower, 0.5)
        has_city   = 1 if city_lower in CITY_SCORES else 0

    # Budget normalisé automatique
    budget_normalized = lead.budget_normalized
    if lead.budget and budget_normalized is None:
        if lead.listing_type == 'Buy':
            budget_normalized = min(lead.budget / 1500000, 1.0)
        else:
            budget_normalized = min(lead.budget / 5000, 1.0)

    # FIFO score automatique
    fifo_score = lead.fifo_score
    if fifo_score is None:
        fifo_score = round(1 - (lead.fifo_position / 1000), 3)

    # Feature Engineering
    b = budget_normalized if budget_normalized is not None else 0.4
    d = lead.docs_score   if lead.docs_score   is not None else 0

    budget_x_docs    = b * d
    profile_x_docs   = (lead.profile_complete / 100) * d
    city_x_budget    = (city_score or 0.5) * b
    engagement_score = (
        (lead.has_email   or 0) * 0.3 +
        (lead.has_phone   or 0) * 0.3 +
        (lead.has_address or 0) * 0.2 +
        (lead.is_business_hour or 0) * 0.2
    )
    listing_type_encoded = 1 if lead.listing_type == 'Buy' else 0

    data = {
        'listing_type_encoded' : listing_type_encoded,
        'city_score'           : city_score or 0.5,
        'has_city'             : has_city   or 0,
        'budget_normalized'    : budget_normalized,
        'has_budget'           : lead.has_budget,
        'has_email'            : lead.has_email,
        'has_phone'            : lead.has_phone,
        'has_address'          : lead.has_address,
        'profile_complete'     : lead.profile_complete,
        'docs_uploaded'        : lead.docs_uploaded,
        'docs_score'           : lead.docs_score,
        'is_long_term'         : lead.is_long_term,
        'has_move_in_date'     : lead.has_move_in_date,
        'is_business_hour'     : lead.is_business_hour,
        'fifo_score'           : fifo_score,
        'budget_x_docs'        : budget_x_docs,
        'profile_x_docs'       : profile_x_docs,
        'city_x_budget'        : city_x_budget,
        'engagement_score'     : engagement_score
    }

    return pd.DataFrame([data])[features]


# ══════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message" : "REIS ML API",
        "version" : "1.0.0",
        "status"  : "running ✅"
    }

@app.get("/health")
def health():
    return {"status": "healthy ✅"}


# ── Endpoint Scoring ──────────────────────────────────
@app.post("/score-lead")
def score_lead(lead: LeadData):
    try:
        model, features = get_model()
        X     = prepare_features(lead)
        proba = model.predict_proba(X)[0][1]
        score = round(proba * 100)

        if score >= 75:
            category = "Hot"
            emoji    = "🔥"
            priority = "Contacter immédiatement"
        elif score >= 50:
            category = "Warm"
            emoji    = "🌡️"
            priority = "Contacter dans 24h"
        else:
            category = "Cold"
            emoji    = "❄️"
            priority = "Contacter dans 48h"

        # Fiabilité
        missing = sum([
            lead.budget_normalized is None and lead.budget is None,
            lead.is_long_term      is None,
            not lead.has_city
        ])
        reliability = round(100 - (missing * 15))

        return {
            "score"              : score,
            "category"           : category,
            "emoji"              : emoji,
            "probability"        : round(float(proba), 4),
            "reliability"        : reliability,
            "priority"           : priority,
            "conversion_chance"  : f"{score}%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint Fraude Document ──────────────────────────
@app.post("/analyze-document")
async def analyze_doc(
    file         : UploadFile = File(...),
    expected_type: Optional[str] = Form(None)
):
    try:
        # Sauvegarder temporairement
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Analyser
        result = analyze_document(tmp_path, expected_type)

        # Nettoyer
        os.unlink(tmp_path)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint Complet (Score + Fraude) ─────────────────
@app.post("/analyze-lead")
async def analyze_lead_complete(
    listing_type    : str            = Form(...),
    city            : Optional[str]  = Form(None),
    budget          : Optional[float]= Form(None),
    has_email       : Optional[int]  = Form(0),
    has_phone       : Optional[int]  = Form(0),
    has_address     : Optional[int]  = Form(0),
    profile_complete: Optional[float]= Form(0),
    is_business_hour: Optional[int]  = Form(0),
    fifo_position   : Optional[int]  = Form(50),
    is_long_term    : Optional[int]  = Form(None),
    files           : list[UploadFile] = File(default=[]),
    file_types      : Optional[str]  = Form(None)
):
    try:
        # ── Scoring ML ────────────────────────────────
        docs_uploaded = len(files)
        docs_score    = docs_uploaded / 5.0

        lead = LeadData(
            listing_type     = listing_type,
            city             = city,
            has_city         = 1 if city else 0,
            budget           = budget,
            has_budget       = 1 if budget else 0,
            has_email        = has_email,
            has_phone        = has_phone,
            has_address      = has_address,
            profile_complete = profile_complete,
            docs_uploaded    = docs_uploaded,
            docs_score       = docs_score,
            is_long_term     = is_long_term,
            has_move_in_date = 1 if is_long_term is not None else 0,
            is_business_hour = is_business_hour,
            fifo_position    = fifo_position
        )

        X     = prepare_features(lead)
        proba = model.predict_proba(X)[0][1]
        score = round(proba * 100)

        if score >= 75:
            category = "Hot"
            emoji    = "🔥"
        elif score >= 50:
            category = "Warm"
            emoji    = "🌡️"
        else:
            category = "Cold"
            emoji    = "❄️"

        # ── Analyse documents ─────────────────────────
        doc_results  = []
        types_list   = file_types.split(',') if file_types else []

        for i, file in enumerate(files):
            suffix   = os.path.splitext(file.filename)[1]
            exp_type = types_list[i] if i < len(types_list) else None

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            result = analyze_document(tmp_path, exp_type)
            doc_results.append(result)
            os.unlink(tmp_path)

        fraud_score = calculate_global_fraud_score(doc_results)
        fraud_count = sum(
            1 for r in doc_results if r['fraud_detected']
        )

        return {
            "lead_score"     : score,
            "category"       : category,
            "emoji"          : emoji,
            "probability"    : round(float(proba), 4),
            "fraud_score"    : fraud_score,
            "fraud_detected" : fraud_count > 0,
            "fraud_count"    : fraud_count,
            "documents"      : doc_results,
            "summary"        : (
                f"Score: {score}/100 {emoji} | "
                f"Documents: {len(doc_results) - fraud_count}/"
                f"{len(doc_results)} valides"
            )
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
class DocumentBase64Request(BaseModel):
    file_base64  : str
    file_name    : str
    expected_type: Optional[str] = None

@app.post("/analyze-document-base64")
async def analyze_doc_base64(request: DocumentBase64Request):
    try:
        import base64 as b64
        
        # Décoder le base64
        file_data = b64.b64decode(request.file_base64)
        
        # Sauvegarder temporairement
        suffix = os.path.splitext(request.file_name)[1] or ".pdf"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name

        # Analyser avec la même fonction qu'avant ✅
        result = analyze_document(tmp_path, request.expected_type)

        # Nettoyer
        os.unlink(tmp_path)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))