import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

N = 5000

cities = {
    'Tunis'        : 1.0,
    'Sfax'         : 0.85,
    'Sousse'       : 0.90,
    'Monastir'     : 0.80,
    'Bizerte'      : 0.75,
    'Nabeul'       : 0.78,
    'Hammamet'     : 0.82,
    'La Marsa'     : 0.95,
    'Ariana'       : 0.88,
    'Ben Arous'    : 0.83,
    'Unknown'      : 0.30,
    'Non précisée' : 0.20,
}

def generate_lead(fifo_position):
    listing_type = random.choice(['Buy', 'Rent'])
    city         = random.choice(list(cities.keys()))
    city_score   = cities[city]

    # ── Budget ──────────────────────────────────────────
    has_budget = random.random() > 0.15  # 85% ont un budget
    if has_budget:
        if listing_type == 'Buy':
            budget = random.randint(50000, 1500000)
            budget_normalized = min(budget / 1500000, 1.0)
        else:
            budget = random.randint(300, 5000)
            budget_normalized = min(budget / 5000, 1.0)
    else:
        budget            = None
        budget_normalized = None  # manquant

    # ── Profil ───────────────────────────────────────────
    has_email    = random.random() > 0.05
    has_phone    = random.random() > 0.10
    has_address  = random.random() > 0.20
    profile_complete = round(
        (has_email * 0.3 + has_phone * 0.3 + has_address * 0.4) * 100
    )

    # ── Documents ────────────────────────────────────────
    docs_uploaded = random.randint(0, 5)
    docs_score    = docs_uploaded / 5.0

    # ── Long terme ───────────────────────────────────────
    has_move_in_date = random.random() > 0.30
    if listing_type == 'Rent' and has_move_in_date:
        is_long_term = int(random.random() > 0.4)
    else:
        is_long_term = None  # manquant

    # ── Heure soumission ─────────────────────────────────
    hour             = random.randint(0, 23)
    is_business_hour = 1 if 8 <= hour <= 18 else 0

    # ── FIFO ─────────────────────────────────────────────
    fifo_score = round(1 - (fifo_position / N), 3)

    # ── Localisation précise ─────────────────────────────
    has_city = city not in ['Unknown', 'Non précisée']

    # ── Score de base pour target ────────────────────────
    b_score = budget_normalized if budget_normalized is not None else 0.4
    l_score = is_long_term      if is_long_term      is not None else 0.5
    c_score = city_score        if has_city          else 0.3

    base_score = (
        b_score              * 0.25 +
        c_score              * 0.20 +
        docs_score           * 0.25 +
        profile_complete/100 * 0.15 +
        is_business_hour     * 0.05 +
        fifo_score           * 0.10
    )

    if listing_type == 'Rent':
        base_score += l_score * 0.10

    noise     = np.random.normal(0, 0.05)
    converted = 1 if (base_score + noise) > 0.55 else 0

    return {
        'listing_type'      : listing_type,
        'city'              : city,
        'city_score'        : city_score,
        'has_city'          : int(has_city),
        'budget'            : budget,
        'budget_normalized' : budget_normalized,
        'has_budget'        : int(has_budget),
        'has_email'         : int(has_email),
        'has_phone'         : int(has_phone),
        'has_address'       : int(has_address),
        'profile_complete'  : profile_complete,
        'docs_uploaded'     : docs_uploaded,
        'docs_score'        : round(docs_score, 3),
        'is_long_term'      : is_long_term,
        'has_move_in_date'  : int(has_move_in_date),
        'is_business_hour'  : is_business_hour,
        'fifo_position'     : fifo_position,
        'fifo_score'        : fifo_score,
        'converted'         : converted
    }

# ── Générer ───────────────────────────────────────────
data = [generate_lead(i+1) for i in range(N)]
df   = pd.DataFrame(data)

# ── Stats ─────────────────────────────────────────────
print("=" * 50)
print("        DATASET GÉNÉRÉ AVEC SUCCÈS")
print("=" * 50)
print(f"Total leads         : {len(df)}")
print(f"Convertis           : {df['converted'].sum()} ({df['converted'].mean()*100:.1f}%)")
print(f"Non convertis       : {(df['converted']==0).sum()}")
print(f"Leads sans budget   : {df['budget'].isna().sum()}")
print(f"Leads sans is_long  : {df['is_long_term'].isna().sum()}")
print(f"\nDistribution listing_type :")
print(df['listing_type'].value_counts())
print(f"\nTop 5 villes :")
print(df['city'].value_counts().head())
print(f"\nMoyenne docs uploadés : {df['docs_uploaded'].mean():.2f}")
print("=" * 50)

# ── Sauvegarder ───────────────────────────────────────
df.to_csv('scoring/leads_dataset.csv', index=False)
print("Dataset sauvegardé : scoring/leads_dataset.csv ✅")