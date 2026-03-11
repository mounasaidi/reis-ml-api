import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

CITIES_VALID = [
    'Tunis', 'Sousse', 'Sfax', 'Ariana',
    'La Marsa', 'Hammamet', 'Monastir',
    'Nabeul', 'Bizerte', 'Ben Arous'
]
CITIES_ALL = CITIES_VALID + [
    'Kasserine', 'Gafsa', 'Tozeur',
    'Kebili', 'Tataouine', 'Siliana'
]
EMPLOYMENT_STATUSES = [
    'Employed (CDI)', 'Employed (CDD)',
    'Self-Employed', 'Student',
    'Retired', 'Unemployed'
]
DOC_STATUSES = ['legitimate', 'suspicious', 'fraud']


def generate_buy_lead(lead_id, fifo_rank):
    city        = random.choice(CITIES_ALL)
    budget      = random.uniform(50000, 800000)
    bank_amount = random.uniform(0, 1200000)
    hour        = random.randint(0, 23)

    m_score          = 1 if bank_amount > budget else 0
    c_score          = 1 if city in CITIES_VALID else 0
    fifo_score       = max(0, (11 - fifo_rank) / 10.0) \
                       if fifo_rank <= 10 else 0
    is_business_hour = 1 if 8 <= hour <= 18 else 0

    doc_status = random.choices(
        DOC_STATUSES, weights=[0.60, 0.25, 0.15]
    )[0]
    doc_fraud_flag = 1 if doc_status == 'fraud' else 0

    ai_score = (
        0.35 * fifo_score +
        0.05 * is_business_hour +
        0.60 * m_score
    ) * 100

    # ── Probabilité : fraud BLOQUE toujours ──────────
    if doc_status == 'fraud':
        prob = random.uniform(0.005, 0.025)  # max 2.5% même avec score=100

    elif doc_status == 'suspicious':
        if m_score == 1 and fifo_score >= 0.7:
            prob = random.uniform(0.04, 0.10)
        elif m_score == 1:
            prob = random.uniform(0.02, 0.07)
        else:
            prob = random.uniform(0.01, 0.04)

    else:  # legitimate
        if m_score == 1 and fifo_score >= 0.7 and c_score == 1:
            prob = random.uniform(0.85, 0.96)
        elif m_score == 1 and fifo_score >= 0.7:
            prob = random.uniform(0.75, 0.88)
        elif m_score == 1 and fifo_score >= 0.5:
            prob = random.uniform(0.60, 0.75)
        elif m_score == 1 and c_score == 1:
            prob = random.uniform(0.42, 0.58)
        elif m_score == 1:
            prob = random.uniform(0.30, 0.50)
        elif c_score == 1:
            prob = random.uniform(0.10, 0.20)
        else:
            prob = random.uniform(0.02, 0.08)

    prob = min(1.0, max(0.0, prob))
    is_converted = 1 if random.random() < prob else 0

    return {
        'lead_id'          : lead_id,
        'application_type' : 'Buy',
        'city'             : city,
        'budget'           : round(budget, 2),
        'bank_amount'      : round(bank_amount, 2),
        'monthly_salary'   : None,
        'has_guarantor'    : None,
        'is_long_term'     : None,
        'employment_status': None,
        'doc_status'       : doc_status,
        'doc_fraud_flag'   : doc_fraud_flag,
        'fifo_rank'        : fifo_rank,
        'fifo_score'       : round(fifo_score, 2),
        'm_score'          : m_score,
        'c_score'          : c_score,
        'd_score'          : None,
        'l_score'          : None,
        'is_business_hour' : is_business_hour,
        'ai_score'         : round(ai_score, 2),
        'conversion_prob'  : round(prob, 4),
        'is_converted'     : is_converted
    }


def generate_rent_lead(lead_id, fifo_rank):
    city              = random.choice(CITIES_ALL)
    budget            = random.uniform(300, 5000)
    monthly_salary    = random.uniform(400, 15000)
    has_guarantor     = random.choice([True, False])
    is_long_term      = random.choice([True, False])
    employment_status = random.choices(
        EMPLOYMENT_STATUSES,
        weights=[0.35, 0.20, 0.15, 0.15, 0.10, 0.05]
    )[0]
    hour              = random.randint(0, 23)

    d_score          = 1 if monthly_salary > 3 * budget else 0
    l_score          = 1 if is_long_term else 0
    c_score          = 1 if city in CITIES_VALID else 0
    fifo_score       = max(0, (11 - fifo_rank) / 10.0) \
                       if fifo_rank <= 10 else 0
    is_business_hour = 1 if 8 <= hour <= 18 else 0
    docs_score       = random.uniform(0.2, 1.0)

    doc_status = random.choices(
        DOC_STATUSES, weights=[0.60, 0.25, 0.15]
    )[0]
    doc_fraud_flag = 1 if doc_status == 'fraud' else 0

    ai_score = (
        0.25 * l_score +
        0.15 * c_score +
        0.20 * docs_score +
        0.25 * d_score +
        0.15 * fifo_score +
        0.05 * is_business_hour
    ) * 100

    is_cdi    = employment_status == 'Employed (CDI)'
    is_stable = employment_status in [
        'Employed (CDI)', 'Self-Employed', 'Retired'
    ]

    # ── Probabilité : fraud BLOQUE absolument ────────
    if doc_status == 'fraud':
        # Même CDI + d_score=1 + has_guarantor → max 2.5%
        prob = random.uniform(0.005, 0.025)

    elif doc_status == 'suspicious':
        if d_score == 1 and is_cdi:
            prob = random.uniform(0.04, 0.10)
        elif d_score == 1 and is_stable:
            prob = random.uniform(0.02, 0.07)
        elif d_score == 1:
            prob = random.uniform(0.01, 0.05)
        else:
            prob = random.uniform(0.005, 0.03)

    else:  # legitimate
        # Base selon solvabilité (primaire)
        if d_score == 1 and has_guarantor and l_score == 1:
            base = random.uniform(0.68, 0.82)
        elif d_score == 1 and has_guarantor:
            base = random.uniform(0.55, 0.70)
        elif d_score == 1 and l_score == 1:
            base = random.uniform(0.48, 0.62)
        elif d_score == 1:
            base = random.uniform(0.32, 0.50)
        elif has_guarantor and c_score == 1:
            base = random.uniform(0.15, 0.28)
        elif has_guarantor or c_score == 1:
            base = random.uniform(0.08, 0.18)
        else:
            base = random.uniform(0.02, 0.08)

        # Employment : bonus additionnel modéré (max +0.15)
        if is_cdi:
            emp_bonus = random.uniform(0.08, 0.15)
        elif is_stable:
            emp_bonus = random.uniform(0.04, 0.09)
        elif employment_status == 'Employed (CDD)':
            emp_bonus = random.uniform(0.01, 0.04)
        else:
            emp_bonus = 0.0

        prob = base + emp_bonus

        if is_long_term:
            prob += random.uniform(0.02, 0.05)

    prob = min(1.0, max(0.0, prob))
    is_converted = 1 if random.random() < prob else 0

    return {
        'lead_id'          : lead_id,
        'application_type' : 'Rent',
        'city'             : city,
        'budget'           : round(budget, 2),
        'bank_amount'      : None,
        'monthly_salary'   : round(monthly_salary, 2),
        'has_guarantor'    : has_guarantor,
        'is_long_term'     : is_long_term,
        'employment_status': employment_status,
        'doc_status'       : doc_status,
        'doc_fraud_flag'   : doc_fraud_flag,
        'fifo_rank'        : fifo_rank,
        'fifo_score'       : round(fifo_score, 2),
        'm_score'          : None,
        'c_score'          : c_score,
        'd_score'          : d_score,
        'l_score'          : l_score,
        'is_business_hour' : is_business_hour,
        'ai_score'         : round(ai_score, 2),
        'conversion_prob'  : round(prob, 4),
        'is_converted'     : is_converted
    }


def generate_dataset(n_total=50000):
    leads  = []
    n_buy  = n_total // 2
    n_rent = n_total // 2

    print(f"Génération de {n_buy} leads Buy...")
    for i in range(n_buy):
        fifo_rank = random.randint(1, 15)
        leads.append(generate_buy_lead(f'BUY_{i:05d}', fifo_rank))

    print(f"Génération de {n_rent} leads Rent...")
    for i in range(n_rent):
        fifo_rank = random.randint(1, 15)
        leads.append(generate_rent_lead(f'RENT_{i:05d}', fifo_rank))

    df = pd.DataFrame(leads)

    print("\n" + "=" * 50)
    print("         STATISTIQUES DU DATASET")
    print("=" * 50)
    print(f"Total leads      : {len(df)}")
    print(f"Convertis        : {df['is_converted'].sum()} "
          f"({df['is_converted'].mean()*100:.1f}%)")

    buy_df  = df[df['application_type'] == 'Buy']
    rent_df = df[df['application_type'] == 'Rent']

    print(f"\nBuy  convertis   : {buy_df['is_converted'].sum()} "
          f"({buy_df['is_converted'].mean()*100:.1f}%)")
    print(f"Rent convertis   : {rent_df['is_converted'].sum()} "
          f"({rent_df['is_converted'].mean()*100:.1f}%)")

    print(f"\nDoc legitimate   : {(df['doc_status']=='legitimate').sum()}")
    print(f"Doc suspicious   : {(df['doc_status']=='suspicious').sum()}")
    print(f"Doc fraud        : {(df['doc_status']=='fraud').sum()}")

    print("\n── Conversion moyenne par doc_status ──")
    print(df.groupby('doc_status')['is_converted'].mean().round(3))

    # ── Vérification critique : fraud + ai_score élevé ──
    print("\n── Fraud leads avec ai_score > 70 ──")
    fraud_high = df[(df['doc_fraud_flag'] == 1) & (df['ai_score'] > 70)]
    if len(fraud_high) > 0:
        print(f"  Nombre          : {len(fraud_high)}")
        print(f"  Taux conversion : "
              f"{fraud_high['is_converted'].mean()*100:.2f}%")
        print(f"  Prob moyenne    : "
              f"{fraud_high['conversion_prob'].mean()*100:.2f}%")

    output_path = 'prediction-convertion/leads_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"\nDataset sauvegardé → {output_path}")
    return df


if __name__ == '__main__':
    df = generate_dataset(50000)