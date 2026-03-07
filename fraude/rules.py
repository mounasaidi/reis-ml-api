# ── Règles métier spécifiques par document ────────────

FRAUD_INDICATORS = {
    'general': [
        'lorem ipsum',
        'test document',
        'fake',
        'faux',
        'sample',
        'exemple',
        'dummy',
        'placeholder'
    ],
    'pay_slip': [
        'salaire: 0',
        'salary: 0',
        'net à payer: 0',
    ],
    'bank_statement': [
        'solde: 0',
        'balance: 0',
        'no transactions',
    ]
}

# ── Montants suspects ─────────────────────────────────
SUSPICIOUS_AMOUNTS = {
    'pay_slip': {
        'min': 300,     # salaire minimum
        'max': 50000    # salaire maximum réaliste
    },
    'rent_receipt': {
        'min': 100,
        'max': 20000
    },
    'bank_statement': {
        'min': 0,
        'max': 10000000
    }
}

# ── Score de confiance par type ───────────────────────
CONFIDENCE_WEIGHTS = {
    'pay_slip'           : 1.0,
    'tax_notice'         : 0.9,
    'bank_statement'     : 1.0,
    'rent_receipt'       : 0.8,
    'employment_contract': 0.9,
    'unknown'            : 0.0
}


def check_fraud_indicators(text, doc_type):
    """
    Vérifie les indicateurs de fraude dans le texte
    Retourne : (is_fraud, reasons)
    """
    text_lower = text.lower()
    reasons    = []

    # Indicateurs généraux
    for indicator in FRAUD_INDICATORS['general']:
        if indicator in text_lower:
            reasons.append(f"Indicateur fraude : '{indicator}'")

    # Indicateurs spécifiques au type
    if doc_type in FRAUD_INDICATORS:
        for indicator in FRAUD_INDICATORS[doc_type]:
            if indicator in text_lower:
                reasons.append(f"Valeur suspecte : '{indicator}'")

    is_fraud = len(reasons) > 0
    return is_fraud, reasons


def get_confidence_weight(doc_type):
    """Retourne le poids de confiance pour un type"""
    return CONFIDENCE_WEIGHTS.get(doc_type, 0.5)


def calculate_global_fraud_score(doc_results):
    """
    Calcule un score global de fraude pour tous les documents
    0   = tout est frauduleux
    100 = tout est légitime
    """
    if not doc_results:
        return 0

    total_weight = 0
    weighted_sum = 0

    for result in doc_results:
        doc_type = result.get('detected_type', 'unknown')
        weight   = get_confidence_weight(doc_type)
        score    = result.get('score', 0)

        weighted_sum  += score * weight
        total_weight  += weight

    if total_weight == 0:
        return 0

    return round(weighted_sum / total_weight)