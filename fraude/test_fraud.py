import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraude.document_analyzer import analyze_document

# ── Mettez le chemin de votre fichier ici ─────────────
FILE_PATH     = r"C:\Users\smoun\Downloads\Extrait de comptes Compte 18047 000832718_ START Jeunes Actifs MLE CHAIMA SAIDI au 2025-06-02.pdf"

# ── Type attendu (optionnel) ──────────────────────────
# Choisissez parmi :
# 'pay_slip'            → Fiche de paie
# 'tax_notice'          → Avis d'imposition
# 'bank_statement'      → Relevé bancaire
# 'rent_receipt'        → Quittance de loyer
# 'employment_contract' → Contrat de travail
EXPECTED_TYPE = 'bank_statement'

# ── Test ──────────────────────────────────────────────
result = analyze_document(FILE_PATH, EXPECTED_TYPE)

print("\n" + "=" * 50)
print("         RÉSULTAT FINAL")
print("=" * 50)
print(f"Fichier        : {result['file']}")
print(f"Type détecté   : {result['detected_type']}")
print(f"Type attendu   : {result['expected_type']}")
print(f"Score          : {result['score']}/100")
print(f"Statut         : {result['status']}")
print(f"Fraude         : {'❌ OUI' if result['fraud_detected'] else '✅ NON'}")
print(f"Recommandation : {result['recommendation']}")
print(f"Texte extrait  : {result['text_length']} caractères")
if result['issues']:
    print(f"Problèmes      :")
    for issue in result['issues']:
        print(f"  → {issue}")