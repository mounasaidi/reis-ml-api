import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Path Tesseract ────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = (
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)

# ── Types de documents supportés ─────────────────────
DOCUMENT_TYPES = {
    'pay_slip'          : 'Fiche de paie',
    'tax_notice'        : 'Avis d\'imposition',
    'bank_statement'    : 'Relevé bancaire',
    'rent_receipt'      : 'Quittance de loyer',
    'employment_contract': 'Contrat de travail'
}

# ── Mots clés par type de document ───────────────────
KEYWORDS = {
    'pay_slip': [
        'salaire', 'salary', 'paie', 'bulletin',
        'employeur', 'employer', 'cotisation',
        'net à payer', 'brut', 'sécurité sociale',
        'congés', 'heures', 'mensuel'
    ],
    'tax_notice': [
        'impôt', 'tax', 'revenu', 'income',
        'fiscal', 'déclaration', 'avis',
        'contribuable', 'administration fiscale',
        'trésor', 'ministère des finances'
    ],
    'bank_statement': [
        'banque', 'bank', 'compte', 'account',
        'iban', 'bic', 'swift', 'solde',
        'balance', 'virement', 'débit', 'crédit',
        'relevé', 'statement', 'transaction'
    ],
    'rent_receipt': [
        'loyer', 'rent', 'quittance', 'receipt',
        'locataire', 'tenant', 'propriétaire',
        'landlord', 'bail', 'lease', 'logement',
        'appartement', 'mensuel'
    ],
    'employment_contract': [
        'contrat', 'contract', 'emploi', 'employment',
        'travail', 'work', 'poste', 'position',
        'salaire', 'salary', 'durée', 'période',
        'cdi', 'cdd', 'employé', 'employee'
    ]
}

# ── Règles de validation par type ────────────────────
VALIDATION_RULES = {
    'pay_slip': {
        'required_keywords' : ['salaire', 'salary', 'paie', 'bulletin', 'net'],
        'required_patterns' : [
            r'\d+[\.,]\d+\s*(?:TND|EUR|€|\$|DT)',  # montant
            r'\d{2}[\/\-]\d{4}|\d{4}[\/\-]\d{2}',  # date mois/année
        ],
        'min_keywords'      : 2
    },
    'tax_notice': {
        'required_keywords' : ['impôt', 'tax', 'revenu', 'fiscal'],
        'required_patterns' : [
            r'\d{4}',                                # année
            r'\d+[\.,]\d+\s*(?:TND|EUR|€|\$|DT)',  # montant
        ],
        'min_keywords'      : 2
    },
    'bank_statement': {
        'required_keywords' : ['banque', 'bank', 'compte', 'solde', 'iban'],
        'required_patterns' : [
            r'[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}',     # IBAN
            r'\d{2}[\/\-]\d{2}[\/\-]\d{4}',        # date
        ],
        'min_keywords'      : 2
    },
    'rent_receipt': {
        'required_keywords' : ['loyer', 'rent', 'quittance', 'locataire'],
        'required_patterns' : [
            r'\d+[\.,]\d+\s*(?:TND|EUR|€|\$|DT)',  # montant loyer
            r'\d{2}[\/\-]\d{2}[\/\-]\d{4}',        # date
        ],
        'min_keywords'      : 2
    },
    'employment_contract': {
        'required_keywords' : ['contrat', 'contract', 'emploi', 'travail'],
        'required_patterns' : [
            r'\d{2}[\/\-]\d{2}[\/\-]\d{4}',        # date
        ],
        'min_keywords'      : 2
    }
}


def extract_text_from_pdf(file_path):
    """Extraire texte d'un PDF"""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()

        # Si PDF scanné (pas de texte)
        if len(text.strip()) < 50:
            text = extract_text_with_ocr(file_path)

    except Exception as e:
        print(f"Erreur PDF: {e}")
    return text


def extract_text_with_ocr(file_path):
    """OCR pour PDF scanné ou image"""
    text = ""
    try:
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            for page in doc:
                # Convertir page en image
                mat  = fitz.Matrix(2, 2)  # zoom x2
                pix  = page.get_pixmap(matrix=mat)
                img  = Image.frombytes(
                    "RGB", [pix.width, pix.height], pix.samples
                )
                # OCR multilingue
                text += pytesseract.image_to_string(
                    img, lang='fra+eng+ara'
                )
            doc.close()
        else:
            # Image directe
            img   = Image.open(file_path)
            text  = pytesseract.image_to_string(
                img, lang='fra+eng+ara'
            )
    except Exception as e:
        print(f"Erreur OCR: {e}")
    return text


def detect_document_type(text):
    """Détecter automatiquement le type de document"""
    text_lower = text.lower()
    scores     = {}

    for doc_type, keywords in KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[doc_type] = score

    best_type  = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score == 0:
        return 'unknown', 0

    return best_type, best_score


def check_date_validity(text):
    """Vérifier que les dates sont récentes (max 3 mois)"""
    patterns = [
        r'(\d{2})[\/\-](\d{2})[\/\-](\d{4})',
        r'(\d{4})[\/\-](\d{2})[\/\-](\d{2})',
        r'(\d{2})[\/\-](\d{4})',
    ]

    found_dates = []
    now         = datetime.now()

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                if len(match) == 3:
                    if len(match[2]) == 4:
                        date = datetime(
                            int(match[2]),
                            int(match[1]),
                            int(match[0])
                        )
                    else:
                        date = datetime(
                            int(match[0]),
                            int(match[1]),
                            int(match[2])
                        )
                elif len(match) == 2:
                    date = datetime(int(match[1]), int(match[0]), 1)

                # Vérifier si date future
                if date > now:
                    return False, "Date future détectée"

                # Vérifier si trop ancienne (> 2 ans)
                diff_months = (
                    (now.year - date.year) * 12 +
                    (now.month - date.month)
                )
                if diff_months <= 24:
                    found_dates.append(date)

            except:
                continue

    if not found_dates:
        return None, "Aucune date trouvée"

    return True, f"{len(found_dates)} date(s) valide(s)"


def validate_document(text, doc_type):
    """Valider un document selon ses règles métier"""
    if doc_type not in VALIDATION_RULES:
        return 50, ["Type de document inconnu"]

    rules    = VALIDATION_RULES[doc_type]
    issues   = []
    score    = 100

    # Vérifier mots clés requis
    text_lower      = text.lower()
    keywords_found  = sum(
        1 for kw in rules['required_keywords']
        if kw in text_lower
    )

    if keywords_found < rules['min_keywords']:
        issues.append(
            f"Mots clés insuffisants ({keywords_found}/"
            f"{rules['min_keywords']})"
        )
        score -= 30

    # Vérifier patterns requis
    patterns_found = 0
    for pattern in rules['required_patterns']:
        if re.search(pattern, text, re.IGNORECASE):
            patterns_found += 1

    if patterns_found == 0:
        issues.append("Aucun pattern requis trouvé")
        score -= 25
    elif patterns_found < len(rules['required_patterns']):
        issues.append("Patterns partiellement trouvés")
        score -= 10

    # Vérifier longueur du texte
    if len(text.strip()) < 100:
        issues.append("Document trop court / illisible")
        score -= 20

    # Vérifier dates
    date_valid, date_msg = check_date_validity(text)
    if date_valid is False:
        issues.append(f"Problème date : {date_msg}")
        score -= 20
    elif date_valid is None:
        issues.append("Aucune date trouvée")
        score -= 10

    return max(score, 0), issues


def analyze_document(file_path, expected_type=None):
    """
    Analyser un document complet
    """
    print(f"\nAnalyse : {os.path.basename(file_path)}")
    print("-" * 40)

    result = {
        'file'          : os.path.basename(file_path),
        'status'        : 'unknown',
        'fraud_detected': False,
        'score'         : 0,
        'detected_type' : None,
        'expected_type' : expected_type,
        'type_match'    : None,
        'issues'        : [],
        'text_length'   : 0,
        'recommendation': ''
    }

    # ── Extraction texte ─────────────────────────────
    if file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    else:
        text = extract_text_with_ocr(file_path)

    result['text_length'] = len(text.strip())
    print(f"Texte extrait : {result['text_length']} caractères")

    if result['text_length'] < 20:
        result['status']         = 'fraud'
        result['fraud_detected'] = True
        result['score']          = 0
        result['issues']         = ['Document vide ou illisible']
        result['recommendation'] = 'Document rejeté — illisible'
        return result

    # ── Détection type ───────────────────────────────
    detected_type, type_score = detect_document_type(text)
    result['detected_type']   = detected_type
    print(f"Type détecté  : {detected_type} (score: {type_score})")

    # ── Vérification correspondance type ─────────────
    if expected_type:
        result['type_match'] = (detected_type == expected_type)
        if not result['type_match']:
            result['issues'].append(
                f"Type incorrect : attendu {expected_type}, "
                f"reçu {detected_type}"
            )

    # ── Validation document ───────────────────────────
    val_score, issues = validate_document(text, detected_type)
    result['issues'].extend(issues)
    result['score'] = val_score

    # ── Verdict final ─────────────────────────────────
    # ✅ Type mismatch = fraud direct
    if expected_type and not result['type_match']:
        result['status']         = 'fraud'
        result['fraud_detected'] = True
        result['score']          = 0
        result['issues'].append(
            f"Document invalide : attendu {expected_type}, "
            f"reçu {detected_type}"
        )
        result['recommendation'] = (
            f"❌ Mauvais document fourni — "
            f"attendu {expected_type}, reçu {detected_type}"
        )
        return result

    if result['score'] >= 70:
        result['status']         = 'legitimate'
        result['fraud_detected'] = False
        result['recommendation'] = '✅ Document valide'
    elif result['score'] >= 40:
        result['status']         = 'suspicious'
        result['fraud_detected'] = False
        result['recommendation'] = '⚠️ Document suspect — vérification manuelle'
    else:
        result['status']         = 'fraud'
        result['fraud_detected'] = True
        result['recommendation'] = '❌ Document rejeté — fraude probable'

    print(f"Score         : {result['score']}/100")
    print(f"Statut        : {result['status']}")
    print(f"Recommandation: {result['recommendation']}")
    if result['issues']:
        print(f"Problèmes     : {', '.join(result['issues'])}")

    return result


def analyze_all_documents(documents):
    """
    Analyser tous les documents d'un lead
    
    documents : liste de dicts
    [
      {'path': '...', 'type': 'pay_slip'},
      {'path': '...', 'type': 'bank_statement'},
      ...
    ]
    """
    print("\n" + "=" * 50)
    print("    ANALYSE COMPLÈTE DES DOCUMENTS")
    print("=" * 50)

    results         = []
    total_score     = 0
    fraud_count     = 0
    suspicious_count= 0

    for doc in documents:
        result = analyze_document(
            doc['path'],
            doc.get('type')
        )
        results.append(result)
        total_score += result['score']

        if result['fraud_detected']:
            fraud_count += 1
        elif result['status'] == 'suspicious':
            suspicious_count += 1

    # ── Résumé global ─────────────────────────────────
    avg_score       = total_score / len(results) if results else 0
    overall_status  = 'legitimate'

    if fraud_count > 0:
        overall_status = 'fraud'
    elif suspicious_count > 0:
        overall_status = 'suspicious'

    summary = {
        'total_documents'   : len(results),
        'fraud_count'       : fraud_count,
        'suspicious_count'  : suspicious_count,
        'legitimate_count'  : len(results) - fraud_count - suspicious_count,
        'average_score'     : round(avg_score),
        'overall_status'    : overall_status,
        'results'           : results
    }

    print("\n" + "=" * 50)
    print("           RÉSUMÉ GLOBAL")
    print("=" * 50)
    print(f"Total documents  : {summary['total_documents']}")
    print(f"Légitimes ✅     : {summary['legitimate_count']}")
    print(f"Suspects ⚠️      : {summary['suspicious_count']}")
    print(f"Fraudes ❌       : {summary['fraud_count']}")
    print(f"Score moyen      : {summary['average_score']}/100")
    print(f"Statut global    : {summary['overall_status']}")

    return summary