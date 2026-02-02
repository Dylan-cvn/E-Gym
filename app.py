"""
CoachMuscu - Coach de Musculation Adaptatif
Application Streamlit mobile-first pour le suivi de progression en musculation.

Séances : Jeudi (Pecs/Biceps/Triceps), Samedi (Épaules/Dos/Triceps), Dimanche (Pecs/Biceps/Arrière épaule)
"""

import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import plotly.express as px
from io import StringIO

# ============================================================================
# CONFIGURATION & CONSTANTES
# ============================================================================

APP_TITLE = "💪 Musculation"
APP_PASSWORD = "01025360"

POIDS_INCREMENT = {"compose": 2.5, "isolation": 1.25}
BACKOFF_PERCENT = 0.90
BACKOFF_MIN_PERCENT = 0.80
BACKOFF_MAX_PERCENT = 0.95
RPE_CIBLE = 8.0
RPE_MAX = 9.0
RPE_DECHARGE = 9.5

JOURS_SEMAINE = {
    0: "Repos",      # Lundi
    1: "Repos",      # Mardi
    2: "Repos",      # Mercredi
    3: "Jeudi : Pecs / Biceps / Triceps",   # Jeudi
    4: "Repos",      # Vendredi
    5: "Samedi : Épaule / Dos / Triceps",  # Samedi
    6: "Dimanche : Pecs / Biceps / Arr. Épaule"  # Dimanche
}

EXERCICES_COMPOSES = [
    "Développé Couché",
    "Développé Militaire",
    "Développé Incliné",
    "Développé Décliné",
    "Dips",
    "Tirage Vertical",
    "Tirage Horizontal",
    "Curl Barre Debout",
]

EXERCICES = {
    "Jeudi - Pecs / Biceps / Triceps": [
        {
            "name": "Développé Couché",
            "groupe": "Pecs",
            "sets": 3,
            "Nb_rep": (7, 10),
            "rest_max": "3 min",
            "rest_seconds": 180,
            "video": "https://www.youtube.com/watch?v=rT7DgCr-3pg",
            "notes": "Omoplates serrées, pieds au sol, descente contrôlée jusqu'à la poitrine."
        },
        {
            "name": "Pecs Poulie (hug ou unilatérale)",
            "groupe": "Pecs",
            "sets": 3,
            "Nb_rep": (7, 10),
            "rest_max": "2 min",
            "rest_seconds": 120,
            "video": "https://www.youtube.com/watch?v=taI4XduLpTk",
            "notes": "Étirement complet, squeeze en fin de mouvement. Contrôle la phase négative."
        },
        {
            "name": "Biceps Pupitre",
            "groupe": "Biceps",
            "sets": 3,
            "Nb_rep": (8, 12),
            "rest_max": "2 min",
            "rest_seconds": 120,
            "video": "https://www.youtube.com/watch?v=fIWP-FRFNU0",
            "notes": "Coudes bien calés, pas de triche. Descente lente et complète."
        },
        {
            "name": "Biceps Curl Poulie",
            "groupe": "Biceps",
            "sets": 3,
            "Nb_rep": (7, 10),
            "rest_max": "2 min",
            "rest_seconds": 120,
            "video": "https://www.youtube.com/watch?v=NFzTWp2qpiE",
            "notes": "Tension constante grâce à la poulie. Squeeze en haut."
        },
        {
            "name": "Triceps Unilatéral Poulie",
            "groupe": "Triceps",
            "sets": 3,
            "Nb_rep": (7, 10),
            "rest_max": "2 min",
            "rest_seconds": 120,
            "video": "https://www.youtube.com/watch?v=pKlwpqL0ydU",
            "notes": "Coude fixe, extension complète. Travaille chaque bras séparément."
        },
        {
            "name": "Extension Triceps Allongé",
            "groupe": "Triceps",
            "sets": 3,
            "Nb_rep": (7, 10),
            "rest_max": "2 min",
            "rest_seconds": 120,
            "video": "https://www.youtube.com/watch?v=d_KZxkY_0cM",
            "notes": "Skull crushers ou avec haltères. Coudes stables, étirement complet."
        },
    ],
    "Samedi - Épaules / Dos / Triceps": [
        {
            "name": "Élévation Latérale",
            "groupe": "Épaules",
            "sets": 3,
            "Nb_rep": (7, 10),
            "rest_max": "2 min",
            "rest_seconds": 120,
            "video": "https://www.youtube.com/watch?v=3VcKaXpzqRo",
            "notes": "Léger penché en avant, contrôle total. Pas de triche avec le dos."
        },
        {
            "name": "Développé Militaire",
            "groupe": "Épaules",
            "sets": 3,
            "Nb_rep": (7, 10),
            "rest_max": "3 min",
            "rest_seconds": 180,
            "video": "https://www.youtube.com/watch?v=2yjwXTZQDDI",
            "notes": "Gainage solide, pas de cambrure excessive. Poussée verticale complète."
        },
        {
            "name": "Tirage Vertical",
            "groupe": "Dos",
            "sets": 4,
            "Nb_rep": (8, 12),
            "rest_max": "2 min",
            "rest_seconds": 120,
            "video": "https://www.youtube.com/watch?v=CAwf7n6Luuc",
            "notes": "Tirer vers le haut de la poitrine, coudes vers le bas. Squeeze les dorsaux."
        },
        {
            "name": "Tirage Horizontal",
            "groupe": "Dos",
            "sets": 4,
            "Nb_rep": (8, 12),
            "rest_max": "2 min",
            "rest_seconds": 120,
            "video": "https://www.youtube.com/watch?v=HJSVR_67OlM",
            "notes": "Dos droit, tirer vers le nombril. Squeeze en fin de mouvement."
        },
        {
            "name": "Extension Triceps Poulie",
            "groupe": "Triceps",
            "sets": 3,
            "Nb_rep": (7, 10),
            "rest_max": "2 min",
            "rest_seconds": 120,
            "video": "https://www.youtube.com/watch?v=vB5OHsJ3EME",
            "notes": "Coudes collés au corps, extension complète. Corde ou barre droite."
        },
        {
            "name": "Dips",
            "groupe": "Triceps",
            "sets": 3,
            "Nb_rep": (8, 12),
            "rest_max": "2 min",
            "rest_seconds": 120,
            "video": "https://www.youtube.com/watch?v=2z8JmcrW-As",
            "notes": "Corps droit pour cibler les triceps. Descente contrôlée, poussée explosive."
        },
    ],
    "Dimanche - Pecs / Biceps / Arr. Épaule": [
        {
            "name": "Développé Incliné",
            "groupe": "Pecs",
            "sets": 3,
            "Nb_rep": (7, 10),
            "rest_max": "3 min",
            "rest_seconds": 180,
            "video": "https://www.youtube.com/watch?v=8iPEnn-ltC8",
            "notes": "Banc à 30-45°. Omoplates serrées, descente jusqu'au haut de la poitrine."
        },
        {
            "name": "Développé Décliné",
            "groupe": "Pecs",
            "sets": 3,
            "Nb_rep": (7, 10),
            "rest_max": "2 min",
            "rest_seconds": 120,
            "video": "https://www.youtube.com/watch?v=LfyQBUKR8SE",
            "notes": "Cible le bas des pectoraux. Amplitude complète, contrôle le mouvement."
        },
        {
            "name": "Curl Marteau",
            "groupe": "Biceps",
            "sets": 3,
            "Nb_rep": (8, 12),
            "rest_max": "2 min",
            "rest_seconds": 120,
            "video": "https://www.youtube.com/watch?v=zC3nLlEvin4",
            "notes": "Prise neutre, cible le brachio-radial. Pas de balancement."
        },
        {
            "name": "Curl Barre Debout",
            "groupe": "Biceps",
            "sets": 3,
            "Nb_rep": (8, 12),
            "rest_max": "2 min",
            "rest_seconds": 120,
            "video": "https://www.youtube.com/watch?v=kwG2ipFRgFo",
            "notes": "Barre droite ou EZ. Coudes fixes, amplitude complète."
        },
        {
            "name": "Face Pull Poulie",
            "groupe": "Arrière Épaule",
            "sets": 3,
            "Nb_rep": (7, 10),
            "rest_max": "2 min",
            "rest_seconds": 120,
            "video": "https://www.youtube.com/watch?v=rep-qVOkqgk",
            "notes": "Poulie haute, tirer vers le visage. Rotation externe en fin de mouvement."
        },
        {
            "name": "Oiseau Haltère",
            "groupe": "Arrière Épaule",
            "sets": 3,
            "Nb_rep": (7, 10),
            "rest_max": "2 min",
            "rest_seconds": 120,
            "video": "https://www.youtube.com/watch?v=ttvfGg9d76c",
            "notes": "Penché en avant, bras légèrement fléchis. Squeeze les omoplates."
        },
    ],
}

POIDS_DEPART = {
    "Développé Couché": 60,
    "Pecs Poulie (hug ou unilatérale)": 5.75,
    "Biceps Pupitre":30,
    "Biceps Curl Poulie": 9.1,
    "Triceps Unilatéral Poulie": 9.1,
    "Extension Triceps Allongé": 9,
    "Élévation Latérale": 3.55,
    "Développé Militaire": 22.5,
    "Tirage Vertical": 59,
    "Tirage Horizontal": 59,
    "Extension Triceps Poulie": 11.3,
    "Dips": 5,
    "Développé Incliné": 60,
    "Développé Décliné": 70,
    "Curl Marteau": 15,
    "Curl Barre Debout": 30,
    "Face Pull Poulie": 13.6,
    "Oiseau Haltère": 7.5,
}

# ============================================================================
# PERSISTENCE LOCALE (JSON)
# ============================================================================

FICHIER_DONNEES = "coach_muscu_data.json"
FICHIER_SETTINGS = "coach_muscu_settings.json"


def charger_logs() -> Dict:
    """Charge les logs depuis un fichier JSON local."""
    try:
        with open(FICHIER_DONNEES, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = {}
    logs.setdefault("entrainements", [])
    logs.setdefault("exercices", {})
    logs.setdefault("seances_sautees", [])
    logs.setdefault("mensurations", [])
    logs.setdefault("decalage_planning", 0)
    for history in logs["exercices"].values():
        history.sort(key=lambda x: x.get("date", ""), reverse=True)
    logs["mensurations"].sort(key=lambda x: x.get("date", ""))
    return logs


def sauvegarder_logs(logs: Dict) -> None:
    """Sauvegarde les logs dans un fichier JSON local."""
    try:
        with open(FICHIER_DONNEES, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        st.error(f"Erreur de sauvegarde : {e}")


def charger_settings() -> Dict:
    """Charge les paramètres depuis un fichier JSON local."""
    try:
        with open(FICHIER_SETTINGS, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"plan_repas": "Option 1 - Classique/Riz", "mode_vacances": False, "decalage_planning": 0}


def sauvegarder_settings(settings: Dict) -> None:
    """Sauvegarde les paramètres dans un fichier JSON local."""
    try:
        with open(FICHIER_SETTINGS, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        st.error(f"Erreur de sauvegarde paramètres : {e}")
    st.session_state["settings"] = settings


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def serialize_sets(sets_data: List[Dict]) -> str:
    return json.dumps(sets_data, separators=(",", ":"))


def rpe_vers_rir(rpe: float) -> float:
    return max(0.0, 10.0 - rpe)


def estimer_e1rm(poids: float, reps: int, rpe: float) -> float:
    if poids <= 0 or reps <= 0:
        return 0.0
    reps_effectives = reps + rpe_vers_rir(rpe)
    return poids * (1 + (reps_effectives / 30.0))


def get_sets_session(entry: Dict) -> List[Dict]:
    poids_base = safe_float(entry.get("poids", 0))
    sets = entry.get("sets", []) or []
    normalises = []
    for s in sets:
        normalises.append({
            "reps": safe_int(s.get("reps", 0)),
            "rpe": safe_float(s.get("rpe", RPE_CIBLE), RPE_CIBLE),
            "poids": safe_float(s.get("poids", poids_base), poids_base)
        })
    return normalises


def get_poids_max_session(entry: Dict) -> float:
    sets = get_sets_session(entry)
    if not sets:
        return safe_float(entry.get("poids", 0))
    return max(s["poids"] for s in sets)


def get_volume_session(entry: Dict) -> float:
    sets = get_sets_session(entry)
    return sum(s["poids"] * s["reps"] for s in sets)


def est_compose(nom_exercice: str) -> bool:
    return nom_exercice in EXERCICES_COMPOSES


def get_increment(nom_exercice: str) -> float:
    return POIDS_INCREMENT["compose"] if est_compose(nom_exercice) else POIDS_INCREMENT["isolation"]


def arrondir_poids(poids: float, nom_exercice: str) -> float:
    increment = get_increment(nom_exercice)
    if increment <= 0:
        return poids
    return round(poids / increment) * increment


# ============================================================================
# CALCULATEUR 1RM
# ============================================================================

def calculer_1rm(poids: float, reps: int, formule: str = "brzycki") -> float:
    if reps <= 0 or poids <= 0:
        return 0
    if reps == 1:
        return poids
    if reps > 12:
        reps = 12
    if formule == "brzycki":
        return poids * (36 / (37 - reps))
    elif formule == "epley":
        return poids * (1 + 0.0333 * reps)
    elif formule == "lander":
        return (100 * poids) / (101.3 - 2.67123 * reps)
    else:
        b = poids * (36 / (37 - reps))
        e = poids * (1 + 0.0333 * reps)
        l = (100 * poids) / (101.3 - 2.67123 * reps)
        return (b + e + l) / 3


def get_pourcentages_1rm(one_rm: float) -> Dict[str, float]:
    pourcentages = {
        "100% (1RM)": 1.0,
        "95% (2 reps)": 0.95,
        "90% (3-4 reps)": 0.90,
        "85% (5-6 reps)": 0.85,
        "80% (7-8 reps)": 0.80,
        "75% (9-10 reps)": 0.75,
        "70% (11-12 reps)": 0.70,
        "65% (15+ reps)": 0.65,
    }
    return {k: round(one_rm * v, 1) for k, v in pourcentages.items()}


# ============================================================================
# ÉCHAUFFEMENT
# ============================================================================

def generer_echauffement(poids_travail: float, reps_travail: int) -> List[Dict]:
    echauffement = []
    if poids_travail >= 40:
        echauffement.append({"poids": 20, "reps": 10, "notes": "Barre à vide / mobilité"})
    if poids_travail >= 30:
        echauffement.append({"poids": round(poids_travail * 0.4 / 2.5) * 2.5, "reps": 8, "notes": "40% - Facile"})
    echauffement.append({"poids": round(poids_travail * 0.6 / 2.5) * 2.5, "reps": 5, "notes": "60% - Modéré"})
    echauffement.append({"poids": round(poids_travail * 0.75 / 2.5) * 2.5, "reps": 3, "notes": "75% - Ça devient lourd"})
    echauffement.append({"poids": round(poids_travail * 0.85 / 2.5) * 2.5, "reps": 2, "notes": "85% - Activation nerveuse"})
    if reps_travail <= 6:
        echauffement.append({"poids": round(poids_travail * 0.90 / 2.5) * 2.5, "reps": 1, "notes": "90% - Dernière prépa"})
    return echauffement


# ============================================================================
# SÉRIE / STREAK
# ============================================================================

def calculer_streak(logs: Dict) -> Dict:
    entrainements = logs.get("entrainements", [])
    if not entrainements:
        return {"streak_semaines": 0, "plus_longue_streak": 0, "semaines_total": 0,
                "regularite_pourcent": 0, "seances_cette_semaine": 0, "objectif_semaine": 3}

    dates_uniques = set()
    for e in entrainements:
        d = e.get("date", "")[:10]
        if d:
            dates_uniques.add(d)

    if not dates_uniques:
        return {"streak_semaines": 0, "plus_longue_streak": 0, "semaines_total": 0,
                "regularite_pourcent": 0, "seances_cette_semaine": 0, "objectif_semaine": 3}

    dates = sorted([datetime.fromisoformat(d) for d in dates_uniques])
    semaines_actives = set()
    for d in dates:
        semaines_actives.add(d.strftime("%Y-W%W"))

    semaine_actuelle = datetime.now().strftime("%Y-W%W")
    seances_cette_semaine = sum(1 for d in dates if d.strftime("%Y-W%W") == semaine_actuelle)

    toutes_semaines = []
    current = dates[0]
    fin = datetime.now()
    while current <= fin:
        toutes_semaines.append(current.strftime("%Y-W%W"))
        current += timedelta(weeks=1)

    streak_actuelle = 0
    for semaine in reversed(toutes_semaines):
        if semaine in semaines_actives:
            streak_actuelle += 1
        else:
            break

    plus_longue = 0
    temp = 0
    for semaine in toutes_semaines:
        if semaine in semaines_actives:
            temp += 1
            plus_longue = max(plus_longue, temp)
        else:
            temp = 0

    total_possible = len(toutes_semaines)
    regularite = (len(semaines_actives) / total_possible * 100) if total_possible > 0 else 0

    return {
        "streak_semaines": streak_actuelle,
        "plus_longue_streak": plus_longue,
        "semaines_total": len(semaines_actives),
        "regularite_pourcent": round(regularite, 1),
        "seances_cette_semaine": seances_cette_semaine,
        "objectif_semaine": 3
    }


def parse_date_log(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        parsed = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except (ValueError, TypeError):
        return None


def get_revue_hebdo(logs: Dict, jours: int = 7) -> Dict:
    now = datetime.now()
    recent_cutoff = now - timedelta(days=jours)
    previous_cutoff = now - timedelta(days=jours * 2)

    revue = {
        "recent": {"sessions": 0, "sets": 0, "volume": 0, "rpe_sum": 0, "rpe_count": 0, "e1rm_sum": 0},
        "precedent": {"sessions": 0, "sets": 0, "volume": 0, "rpe_sum": 0, "rpe_count": 0, "e1rm_sum": 0}
    }

    for _, history in logs.get("exercices", {}).items():
        for entry in history:
            entry_date = parse_date_log(entry.get("date", ""))
            if not entry_date:
                continue
            if entry_date >= recent_cutoff:
                bucket = revue["recent"]
            elif entry_date >= previous_cutoff:
                bucket = revue["precedent"]
            else:
                continue

            sets = get_sets_session(entry)
            if not sets:
                continue
            bucket["sessions"] += 1
            bucket["sets"] += len(sets)
            bucket["volume"] += sum(s.get("poids", 0) * s.get("reps", 0) for s in sets)
            bucket["rpe_sum"] += sum(s.get("rpe", RPE_CIBLE) for s in sets)
            bucket["rpe_count"] += len(sets)

            best = max(sets, key=lambda s: estimer_e1rm(s.get("poids", 0), s.get("reps", 0), s.get("rpe", RPE_CIBLE)))
            bucket["e1rm_sum"] += estimer_e1rm(best.get("poids", 0), best.get("reps", 0), best.get("rpe", RPE_CIBLE))

    for bucket in revue.values():
        bucket["avg_rpe"] = round(bucket["rpe_sum"] / bucket["rpe_count"], 2) if bucket["rpe_count"] else 0
        bucket["avg_e1rm"] = round(bucket["e1rm_sum"] / bucket["sessions"], 1) if bucket["sessions"] else 0

    return revue


# ============================================================================
# DÉTECTION PR
# ============================================================================

def verifier_pr(nom_exercice: str, poids: float, logs: Dict) -> Tuple[bool, float]:
    history = logs.get("exercices", {}).get(nom_exercice, [])
    if not history:
        return True, 0
    max_poids = max(get_poids_max_session(h) for h in history)
    if poids > max_poids:
        return True, max_poids
    return False, max_poids


# ============================================================================
# EXPORT / IMPORT CSV
# ============================================================================

def exporter_csv(logs: Dict) -> str:
    rows = []
    for nom_ex, history in logs.get("exercices", {}).items():
        for entry in history:
            date = entry.get("date", "")[:19]
            poids = entry.get("poids", 0)
            seance = entry.get("seance", "")
            notes = entry.get("notes", "")
            sets = get_sets_session(entry)
            for i, s in enumerate(sets):
                rows.append({
                    "Date": date, "Exercice": nom_ex, "Séance": seance,
                    "Série": i + 1, "Poids_kg": s.get("poids", poids),
                    "Reps": s.get("reps", 0), "RPE": s.get("rpe", RPE_CIBLE), "Notes": notes
                })
    if not rows:
        return "Date,Exercice,Séance,Série,Poids_kg,Reps,RPE,Notes\n"
    return pd.DataFrame(rows).to_csv(index=False)


def exporter_mensurations_csv(logs: Dict) -> str:
    mensurations = logs.get("mensurations", [])
    if not mensurations:
        return "Date,Poids_kg,Tour_Bras_G,Tour_Bras_D,Tour_Poitrine,Tour_Taille,Notes\n"
    return pd.DataFrame(mensurations).to_csv(index=False)


def importer_csv(csv_content: str, logs: Dict) -> Tuple[Dict, int]:
    try:
        df = pd.read_csv(StringIO(csv_content))
        required = ["Date", "Exercice", "Poids_kg", "Reps"]
        if not all(col in df.columns for col in required):
            return logs, -1

        imported = 0
        grouped = df.groupby(["Date", "Exercice"])
        for (date, exercice), group in grouped:
            sets_data = []
            for _, row in group.iterrows():
                sets_data.append({
                    "reps": int(row.get("Reps", 0)),
                    "rpe": float(row.get("RPE", 8)),
                    "poids": float(row.get("Poids_kg", 0))
                })
            entry_poids = max(s.get("poids", 0) for s in sets_data) if sets_data else 0
            entry = {
                "date": str(date),
                "seance": str(group.iloc[0].get("Séance", "")),
                "poids": entry_poids,
                "sets": sets_data,
                "notes": str(group.iloc[0].get("Notes", ""))
            }
            logs["exercices"].setdefault(exercice, []).append(entry)
            logs["entrainements"].append({
                "date": entry["date"], "type_seance": entry["seance"],
                "exercice": exercice, "data": entry
            })
            imported += 1
        sauvegarder_logs(logs)
        return logs, imported
    except Exception:
        return logs, -1


# ============================================================================
# AUTHENTIFICATION
# ============================================================================

def verifier_mot_de_passe() -> bool:
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True
    st.markdown("## 🔐 AnalyseMuscu")
    st.markdown("Entre ton mot de passe pour accéder à ton suivi.")
    password = st.text_input("Mot de passe", type="password", key="password_input")
    if st.button("Connexion", type="primary", use_container_width=True):
        if password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect")
    st.caption("Indice : Code Nonstop")
    return False


# ============================================================================
# PLANNING
# ============================================================================

def get_planning_ajuste(settings: Dict) -> Dict:
    offset = settings.get("decalage_planning", 0)
    if offset == 0:
        return JOURS_SEMAINE.copy()
    ajuste = {}
    for jour in range(7):
        original = (jour - offset) % 7
        ajuste[jour] = JOURS_SEMAINE[original]
    return ajuste


def get_seance_aujourdhui(settings: Dict) -> Tuple[str, List[Dict]]:
    if settings.get("mode_vacances", False):
        return "Vacances", []
    planning = get_planning_ajuste(settings)
    jour = datetime.now().weekday()
    nom_seance = planning[jour]
    if nom_seance == "Repos":
        return "Repos", []
    return nom_seance, EXERCICES.get(nom_seance, [])


def get_prochain_jour(settings: Dict) -> Tuple[str, str, int]:
    if settings.get("mode_vacances", False):
        return "Vacances", "N/A", 0
    planning = get_planning_ajuste(settings)
    noms_jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    jour_actuel = datetime.now().weekday()
    for i in range(1, 8):
        check = (jour_actuel + i) % 7
        if planning[check] != "Repos":
            return planning[check], noms_jours[check], i
    return "Repos", "N/A", 7


def get_apercu_semaine(settings: Dict) -> List[Dict]:
    planning = get_planning_ajuste(settings)
    noms_jours = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    aujourdhui = datetime.now().weekday()
    apercu = []
    for i in range(7):
        jour = (aujourdhui + i) % 7
        seance = planning[jour]
        if settings.get("mode_vacances"):
            display = "❄️"
        elif seance == "Repos":
            display = "😴"
        else:
            # Raccourcir le nom
            parts = seance.split(" - ")
            display = parts[1][:12] if len(parts) > 1 else seance[:12]
        apercu.append({"jour": noms_jours[jour], "seance": display, "est_aujourdhui": i == 0})
    return apercu


# ============================================================================
# ALGORITHME DE COACHING ADAPTATIF
# ============================================================================

class CoachAdaptatif:
    def __init__(self, logs: Dict):
        self.logs = logs
        self.historique_exercices = logs.get("exercices", {})

    def score_serie(self, reps: int, rpe: float, min_r: int, max_r: int) -> float:
        if reps < min_r:
            rep_score = max(0, (reps / min_r) * 30)
        elif reps > max_r:
            rep_score = 100
        else:
            position = (reps - min_r) / max(1, max_r - min_r)
            rep_score = 30 + (position * 70)
        mod_rpe = max(0.85, min(1.20, 1.0 + (9 - rpe) * 0.05))
        return min(100, rep_score * mod_rpe)

    def score_session(self, sets_data: List[Dict], min_r: int, max_r: int) -> Tuple[float, str]:
        if not sets_data:
            return 0, "Pas de données"
        scores = []
        reps_list = []
        for s in sets_data:
            reps = s.get("reps", 0)
            rpe = s.get("rpe", RPE_CIBLE)
            reps_list.append(reps)
            scores.append(self.score_serie(reps, rpe, min_r, max_r))
        avg = sum(scores) / len(scores)
        variance = max(reps_list) - min(reps_list)
        if variance <= 1:
            consistance = "Excellente régularité"
            avg *= 1.05
        elif variance <= 2:
            consistance = "Bonne régularité"
        else:
            consistance = "Travaille la régularité série à série"
            avg *= 0.95
        return min(100, avg), consistance

    def stats_session(self, session: Dict, rep_range: Tuple[int, int]) -> Dict:
        sets = get_sets_session(session)
        if not sets:
            return {
                "sets": [], "poids_max": safe_float(session.get("poids", 0)),
                "top_reps": 0, "top_rpe": RPE_CIBLE, "avg_reps": 0,
                "min_reps": 0, "max_reps": 0, "avg_rpe": RPE_CIBLE,
                "dropoff_reps": 0, "volume": 0, "meilleur_e1rm": 0,
                "tous_dans_range": False
            }
        min_r, max_r = rep_range
        poids_list = [s.get("poids", 0) for s in sets]
        reps_list = [s.get("reps", 0) for s in sets]
        rpe_list = [s.get("rpe", RPE_CIBLE) for s in sets]

        poids_max = max(poids_list)
        top_sets = [s for s in sets if s.get("poids", 0) >= poids_max * 0.99]
        top_set = max(top_sets, key=lambda s: s.get("reps", 0)) if top_sets else sets[0]

        volume = sum(p * r for p, r in zip(poids_list, reps_list))
        e1rms = [estimer_e1rm(p, r, rpe) for p, r, rpe in zip(poids_list, reps_list, rpe_list)]
        dans_range = sum(1 for r in reps_list if min_r <= r <= max_r)

        return {
            "sets": sets, "poids_max": poids_max,
            "top_reps": top_set.get("reps", 0), "top_rpe": top_set.get("rpe", RPE_CIBLE),
            "avg_reps": sum(reps_list) / len(reps_list),
            "min_reps": min(reps_list), "max_reps": max(reps_list),
            "avg_rpe": sum(rpe_list) / len(rpe_list),
            "dropoff_reps": max(reps_list) - min(reps_list),
            "volume": volume,
            "meilleur_e1rm": max(e1rms) if e1rms else 0,
            "tous_dans_range": dans_range == len(reps_list)
        }

    def get_historique(self, nom_exercice: str, limite: int = 10) -> List[Dict]:
        history = self.historique_exercices.get(nom_exercice, [])
        return sorted(history, key=lambda x: x.get("date", ""), reverse=True)[:limite]

    def jours_depuis_derniere(self, nom_exercice: str) -> Optional[int]:
        history = self.get_historique(nom_exercice, 1)
        if not history:
            return None
        try:
            last_str = history[0].get("date", "")
            if not last_str:
                return None
            last = datetime.fromisoformat(last_str.replace("Z", "+00:00"))
            if last.tzinfo:
                last = last.replace(tzinfo=None)
            return (datetime.now() - last).days
        except (ValueError, TypeError):
            return None

    def facteur_retour(self, jours_ecart: int) -> Tuple[float, str, str]:
        if jours_ecart <= 14:
            return 1.0, "NORMAL", ""
        elif jours_ecart <= 28:
            return 0.90, "RETOUR_LEGER", "Semaine de retour léger - reprends doucement"
        elif jours_ecart <= 56:
            return 0.80, "REACCLIMATATION", "Phase de ré-acclimatation - retrouve tes sensations"
        elif jours_ecart <= 90:
            return 0.70, "RECONSTRUCTION", "Phase de reconstruction - la mémoire musculaire va aider !"
        else:
            return 0.60, "NOUVEAU_DEPART", "Nouveau départ - tes muscles se souviennent plus que tu ne crois !"

    def analyser_tendance(self, nom_exercice: str, rep_range: Tuple[int, int], nb_sessions: int = 3) -> Dict:
        history = self.get_historique(nom_exercice, nb_sessions + 2)
        if len(history) < 1:
            return {"has_data": False, "tendance": "PAS_DE_DONNEES", "avg_e1rm": None, "sessions_analysees": 0}

        stats = [self.stats_session(h, rep_range) for h in history[:nb_sessions]]
        e1rms = [s["meilleur_e1rm"] for s in stats]
        avg_reps = sum(s["avg_reps"] for s in stats) / len(stats)
        avg_rpe = sum(s["avg_rpe"] for s in stats) / len(stats)

        e1rm_trend = rep_trend = rpe_trend = 0
        if len(stats) >= 2:
            recent = stats[0]
            anciens = stats[1:]
            prev_e1rm = sum(s["meilleur_e1rm"] for s in anciens) / len(anciens)
            prev_reps = sum(s["avg_reps"] for s in anciens) / len(anciens)
            prev_rpe = sum(s["avg_rpe"] for s in anciens) / len(anciens)
            e1rm_trend = recent["meilleur_e1rm"] - prev_e1rm
            rep_trend = recent["avg_reps"] - prev_reps
            rpe_trend = recent["avg_rpe"] - prev_rpe

        prev_avg = (sum(e1rms[1:]) / len(e1rms[1:])) if len(e1rms) > 1 else e1rms[0]
        seuil = max(0.5, prev_avg * 0.01) if prev_avg else 0.5

        if e1rm_trend > seuil or (rep_trend > 0.5 and rpe_trend <= 0.3):
            tendance = "EN_PROGRES"
        elif e1rm_trend < -seuil or (rep_trend < -0.5 and rpe_trend > 0.5):
            tendance = "EN_BAISSE"
        elif abs(e1rm_trend) <= seuil and abs(rep_trend) <= 0.5:
            tendance = "STABLE"
        else:
            tendance = "VARIABLE"

        return {
            "has_data": True, "tendance": tendance,
            "avg_e1rm": round(sum(e1rms) / len(e1rms), 1),
            "avg_reps": round(avg_reps, 1), "avg_rpe": round(avg_rpe, 1),
            "e1rm_trend": round(e1rm_trend, 1), "rep_trend": round(rep_trend, 1),
            "rpe_trend": round(rpe_trend, 1), "sessions_analysees": min(len(history), nb_sessions)
        }

    def detecter_plateau(self, nom_exercice: str, rep_range: Tuple[int, int]) -> Tuple[bool, int, str]:
        history = self.get_historique(nom_exercice, 6)
        if len(history) < 3:
            return False, 0, "Continue à t'entraîner - données de base en construction"

        stall = 0
        for i in range(len(history) - 1):
            current = self.stats_session(history[i], rep_range)
            previous = self.stats_session(history[i + 1], rep_range)
            if current["meilleur_e1rm"] <= previous["meilleur_e1rm"] + 0.5 and current["avg_reps"] <= previous["avg_reps"] + 0.25:
                stall += 1
            else:
                break

        tendance = self.analyser_tendance(nom_exercice, rep_range, 3)
        if tendance["has_data"] and tendance.get("rpe_trend", 0) > 0.5 and tendance["tendance"] != "EN_PROGRES":
            if stall >= 2:
                return True, stall, "RPE en hausse sans progression - fatigue accumulée."

        if stall >= 4:
            return True, stall, "Plateau significatif ! Temps pour une décharge stratégique."
        elif stall >= 2:
            return True, stall, "Légère stagnation. Pense à augmenter le volume."
        return False, stall, "Bonne progression !"

    def get_prochain_objectif(self, nom_exercice: str, config_exercice: Dict) -> Dict:
        history = self.get_historique(nom_exercice, 8)
        min_r, max_r = config_exercice["Nb_rep"]
        increment = get_increment(nom_exercice)

        def arrondir(val: float) -> float:
            return arrondir_poids(val, nom_exercice)

        def construire_objectif(poids, reps, recommandation, message, backoff_pct=BACKOFF_PERCENT, **kwargs):
            poids_arrondi = arrondir(poids)
            backoff_poids = arrondir(poids_arrondi * backoff_pct)
            backoff_reps = min(max_r, max(reps, min_r) + 2)
            obj = {
                "poids": poids_arrondi, "reps_par_serie": reps,
                "recommandation": recommandation, "message": message,
                "backoff_poids": backoff_poids, "backoff_reps": backoff_reps,
                "backoff_pourcent": backoff_pct
            }
            obj.update(kwargs)
            return obj

        # Première fois
        if not history:
            poids_depart = POIDS_DEPART.get(nom_exercice, 20)
            return construire_objectif(
                poids_depart, min_r, "BASELINE",
                f"Première fois ! Commence avec {arrondir(poids_depart)}kg pour {min_r} reps.",
                confiance=50, est_nouveau=True, tendance_info=None
            )

        jours_ecart = self.jours_depuis_derniere(nom_exercice)
        derniere = history[0]
        stats_derniere = self.stats_session(derniere, (min_r, max_r))
        dernier_poids = stats_derniere["poids_max"]
        derniers_reps = [s.get("reps", 0) for s in stats_derniere["sets"]]
        derniere_rpe = stats_derniere["avg_rpe"]

        # Retour après pause
        if jours_ecart is not None and jours_ecart > 14:
            facteur, phase, msg_phase = self.facteur_retour(jours_ecart)
            poids_retour = arrondir(dernier_poids * facteur)
            pct_decharge = int((1 - facteur) * 100)
            return construire_objectif(
                poids_retour, min_r, phase, msg_phase,
                confiance=85, est_nouveau=False, tendance_info=None,
                jours_depuis_derniere=jours_ecart, dernier_poids=dernier_poids,
                derniers_reps=derniers_reps, derniere_rpe=round(derniere_rpe, 1),
                pct_decharge=pct_decharge,
                precedent=f"Dernier ({jours_ecart} jours) : {dernier_poids}kg × {derniers_reps}"
            )

        tendance = self.analyser_tendance(nom_exercice, (min_r, max_r), 3)

        # Deuxième session
        if len(history) == 1:
            return construire_objectif(
                dernier_poids, min_r + 1, "CONSTRUCTION",
                f"Deuxième séance ! Utilise {dernier_poids}kg, vise {min_r + 1} reps.",
                confiance=60, est_nouveau=False, tendance_info=tendance,
                precedent=f"Dernier : {dernier_poids}kg × {derniers_reps}"
            )

        est_plateau, nb_stall, msg_plateau = self.detecter_plateau(nom_exercice, (min_r, max_r))

        # Décharge forcée
        if stats_derniere["avg_rpe"] >= RPE_DECHARGE or stats_derniere["min_reps"] < min_r - 1 or stats_derniere["dropoff_reps"] >= 4:
            poids_decharge = arrondir(dernier_poids * 0.9)
            return construire_objectif(
                poids_decharge, min_r, "DECHARGE",
                f"Fatigue détectée. Reset à {poids_decharge}kg et reconstruis.",
                confiance=90, est_nouveau=False, tendance_info=tendance,
                plateau_info=msg_plateau if est_plateau else None
            )

        # Plateau sévère
        if est_plateau and nb_stall >= 3:
            poids_decharge = arrondir(dernier_poids * 0.92)
            return construire_objectif(
                poids_decharge, min_r, "DECHARGE",
                f"Plateau détecté. Décharge stratégique à {poids_decharge}kg.",
                confiance=90, est_nouveau=False, tendance_info=tendance, plateau_info=msg_plateau
            )

        # Progression : ajouter du poids
        if stats_derniere["min_reps"] >= max_r and stats_derniere["avg_rpe"] <= (RPE_MAX - 0.5) and tendance.get("tendance") != "EN_BAISSE":
            nouveau_poids = arrondir(dernier_poids + increment)
            return construire_objectif(
                nouveau_poids, min_r, "PROGRESSION",
                f"Ajoute du poids ! {nouveau_poids}kg × {min_r} reps.",
                confiance=90, est_nouveau=False, tendance_info=tendance,
                precedent=f"Moy. e1RM 3 dernières : {tendance.get('avg_e1rm', 0)}kg"
            )

        # Pousser les reps
        if stats_derniere["avg_reps"] >= max_r - 0.5 and stats_derniere["avg_rpe"] <= RPE_MAX:
            return construire_objectif(
                dernier_poids, max_r, "POUSSER",
                f"Atteins {max_r} reps sur toutes les séries pour débloquer +{increment}kg.",
                confiance=80, est_nouveau=False, tendance_info=tendance,
                precedent=f"Dernier : {dernier_poids}kg × {derniers_reps}"
            )

        # Volume boost si plateau léger
        if est_plateau and stats_derniere["avg_rpe"] <= (RPE_MAX - 0.5):
            target_reps = min(max_r, max(int(round(stats_derniere["avg_reps"])), min_r))
            return construire_objectif(
                dernier_poids, target_reps, "VOLUME",
                "Stagnation mais frais. Ajoute une série de back-off pour plus de volume.",
                confiance=80, est_nouveau=False, tendance_info=tendance,
                plateau_info=msg_plateau, series_extra_suggerees=1,
                precedent=f"Dernier : {dernier_poids}kg × {derniers_reps}"
            )

        # Construire
        if stats_derniere["avg_reps"] >= min_r:
            target_reps = min(int(stats_derniere["avg_reps"]) + 1, max_r)
            return construire_objectif(
                dernier_poids, target_reps, "CONSTRUCTION",
                f"Objectif : {dernier_poids}kg × {target_reps} reps.",
                confiance=75, est_nouveau=False, tendance_info=tendance,
                precedent=f"Dernier : {dernier_poids}kg × {derniers_reps}"
            )

        # Consolider
        return construire_objectif(
            dernier_poids, min_r, "CONSOLIDER",
            f"Même poids ({dernier_poids}kg), concentre-toi sur {min_r} reps propres.",
            confiance=70, est_nouveau=False, tendance_info=tendance,
            precedent=f"Dernier : {dernier_poids}kg @ RPE moy {stats_derniere['avg_rpe']:.1f}",
            plateau_info=msg_plateau if est_plateau else None
        )

    def resume_seance(self, nom_seance: str, exercices: List[Dict]) -> List[Dict]:
        resume = []
        for ex in exercices:
            objectif = self.get_prochain_objectif(ex["name"], ex)
            objectif["exercice"] = ex
            resume.append(objectif)
        return resume

    def stats_globales(self) -> Optional[Dict]:
        if not self.historique_exercices:
            return None
        stats = {
            "total_sessions": 0, "total_sets": 0, "total_reps": 0,
            "total_volume": 0, "premier_entrainement": None, "dernier_entrainement": None,
            "exercices": {}, "liste_pr": []
        }
        toutes_dates = []

        for nom_ex, history in self.historique_exercices.items():
            if not history:
                continue
            ex_stats = {"sessions": len(history), "poids_actuel": 0, "poids_max": 0,
                        "poids_depart": 0, "gain_poids": 0, "volume_total": 0, "avg_reps": 0}
            all_reps = []
            poids_list = []

            for session in history:
                date = session.get("date", "")
                if date:
                    toutes_dates.append(date)
                top = get_poids_max_session(session)
                poids_list.append(top)
                sets = get_sets_session(session)
                for s in sets:
                    r = s.get("reps", 0)
                    all_reps.append(r)
                    stats["total_reps"] += r
                    stats["total_volume"] += s.get("poids", top) * r
                    stats["total_sets"] += 1

            if poids_list:
                ex_stats["poids_actuel"] = poids_list[0]
                ex_stats["poids_max"] = max(poids_list)
                ex_stats["poids_depart"] = poids_list[-1]
                ex_stats["gain_poids"] = poids_list[0] - poids_list[-1]
            if all_reps:
                ex_stats["avg_reps"] = round(sum(all_reps) / len(all_reps), 1)
            ex_stats["volume_total"] = sum(get_volume_session(h) for h in history)
            stats["exercices"][nom_ex] = ex_stats
            stats["total_sessions"] += len(history)
            if ex_stats["poids_max"] > 0:
                stats["liste_pr"].append({"exercice": nom_ex, "poids": ex_stats["poids_max"]})

        if toutes_dates:
            stats["premier_entrainement"] = min(toutes_dates)[:10]
            stats["dernier_entrainement"] = max(toutes_dates)[:10]
        stats["liste_pr"] = sorted(stats["liste_pr"], key=lambda x: x["poids"], reverse=True)
        return stats


# ============================================================================
# HELPERS UI
# ============================================================================

def format_badge_recommandation(recommandation: str) -> str:
    couleurs = {
        "PROGRESSION": "🟢", "POUSSER": "🔵", "CONSTRUCTION": "🟡", "CONSOLIDER": "🟠",
        "DECHARGE": "🔴", "BASELINE": "⚪", "NORMAL": "🟢", "VOLUME": "🟡",
        "RETOUR_LEGER": "🔵", "REACCLIMATATION": "🟡", "RECONSTRUCTION": "🟠", "NOUVEAU_DEPART": "🔴"
    }
    return f"{couleurs.get(recommandation, '⚪')} {recommandation.replace('_', ' ')}"


def format_badge_tendance(tendance: str) -> str:
    indicateurs = {"EN_PROGRES": "📈", "STABLE": "➡️", "EN_BAISSE": "📉", "VARIABLE": "〰️", "PAS_DE_DONNEES": "❓"}
    return indicateurs.get(tendance, "❓")


def lien_exercice(name: str, url: str) -> str:
    return f"[{name}]({url})"


# ============================================================================
# APPLICATION STREAMLIT
# ============================================================================

def main():
    st.set_page_config(
        page_title="AnalyseMuscu",
        page_icon="💪",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    if not verifier_mot_de_passe():
        return

    # Charger données
    if "logs" not in st.session_state:
        st.session_state["logs"] = charger_logs()
    if "settings" not in st.session_state:
        st.session_state["settings"] = charger_settings()

    logs = st.session_state["logs"]
    settings = st.session_state["settings"]
    coach = CoachAdaptatif(logs)

    # CSS
    st.markdown("""<style>
        /* === FOND & GLOBAL === */
        .stApp {
            max-width: 100%;
            background: #000000;
        }

        /* === BOUTONS NÉON VERT (tous les boutons par défaut) === */
        button {
            width: 100%;
            padding: 0.75rem 1rem;
            font-size: 1.1rem;
            border-radius: 25px;
            border: 1px solid #2ECC71 !important;
            background: transparent !important;
            color: #2ECC71 !important;
            transition: all 0.3s ease;
        }
        button:hover {
            background: #2ECC71 !important;
            color: #000000 !important;
            box-shadow: 0 0 15px #2ECC71, 0 0 30px #2ECC7155;
        }

        /* === BOUTON PRIMARY (Connexion) EN BLEU === */
        button[data-testid="stBaseButton-primary"] {
            border: 1px solid #3498DB !important;
            background: #3498DB !important;
            color: #FFFFFF !important;
        }
        button[data-testid="stBaseButton-primary"]:hover {
            background: #2980B9 !important;
            border-color: #2980B9 !important;
            box-shadow: 0 0 15px #3498DB, 0 0 30px #3498DB55 !important;
            color: #FFFFFF !important;
        }

        /* === INPUTS ARRONDIS === */
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div {
            border-radius: 20px;
            border: 1px solid #1a1a2e;
            background: #0D1117;
            color: #E0E0E0;
            text-align: center;
            font-size: 1.1rem;
        }
        .stNumberInput > div > div > input:focus,
        .stTextInput > div > div > input:focus {
            border-color: #2ECC71;
            box-shadow: 0 0 8px #2ECC7144;
        }

        /* === TABS NÉON === */
        .stTabs [data-baseweb="tab"] {
            border-radius: 20px;
            padding: 8px 16px;
            color: #888;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] {
            color: #2ECC71 !important;
            border-bottom: 2px solid #2ECC71 !important;
            text-shadow: 0 0 10px #2ECC7166;
        }

        /* === METRICS GLOW === */
        [data-testid="stMetric"] {
            background: #0D1117;
            border: 1px solid #1a1a2e;
            border-radius: 20px;
            padding: 15px;
            text-align: center;
        }
        [data-testid="stMetricValue"] {
            color: #2ECC71;
            text-shadow: 0 0 8px #2ECC7144;
        }

        /* === EXPANDERS === */
        .streamlit-expanderHeader {
            border-radius: 20px;
            background: #0D1117;
            border: 1px solid #1a1a2e;
        }

        /* === SLIDERS === */
        .stSlider > div > div > div > div {
            background: #2ECC71;
        }

        /* === DIVIDERS === */
        hr {
            border-color: #1a1a2e;
        }

        /* === HIDE DEFAULTS === */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* === PR CELEBRATION === */
        .pr-celebration {
            font-size: 2rem;
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #2ECC71, #27AE60, #2ECC71);
            border-radius: 25px;
            animation: pulse 1s infinite;
            color: #000;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        /* === GLOW LINKS === */
        a {
            color: #2ECC71 !important;
            text-decoration: none;
        }
        a:hover {
            text-shadow: 0 0 10px #2ECC71;
        }
    </style>""", unsafe_allow_html=True)

    # En-tête
    if not settings.get("mode_vacances", False):
        streak = calculer_streak(logs)
        st.metric("🔥 Série", f"{streak['streak_semaines']}sem")
    else:
        st.markdown("## ❄️ MODE VACANCES ❄️")

    st.divider()

    # Navigation
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏋️ Aujourd'hui", "📝 Logger", "📊 Progrès", "📈 Stats",
        "🔢 Outils", "📘 Système", "⚙️ Réglages"
    ])

    # ========== ONGLET 1 : AUJOURD'HUI ==========
    with tab1:
        if settings.get("mode_vacances"):
            st.info("Entraînement gelé. Va dans Réglages pour désactiver.")
        else:
            nom_seance, exercices = get_seance_aujourdhui(settings)

            # Aperçu semaine
            st.markdown("### 📅 Cette semaine")
            apercu = get_apercu_semaine(settings)
            cols = st.columns(7)
            for i, jour in enumerate(apercu):
                with cols[i]:
                    if jour["est_aujourdhui"]:
                        st.markdown(f"**{jour['jour']}**")
                        st.markdown(f"📍 {jour['seance']}")
                    else:
                        st.caption(jour['jour'])
                        st.caption(jour['seance'])
            st.divider()

            if nom_seance == "Repos":
                st.markdown("## 😴 Jour de Repos")
                st.info("La récupération, c'est là que tu progresses ! Marche, étirements, hydratation.")
                prochain, jour_prochain, dans_jours = get_prochain_jour(settings)
                if prochain != "Repos":
                    short_name = prochain.split(" - ")[1] if " - " in prochain else prochain
                    st.write(f"**Prochain :** {short_name} — {jour_prochain}")
            else:
                short_title = nom_seance.split(" - ")[1] if " - " in nom_seance else nom_seance
                st.markdown(f"## {short_title}")

                # Sauter la séance
                with st.expander("⏭️ Sauter la séance du jour"):
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("🔄 Décaler +1 jour", use_container_width=True):
                            logs.setdefault("seances_sautees", [])
                            logs["seances_sautees"].append({"date": datetime.now().isoformat(), "seance": nom_seance, "action": "decaler"})
                            settings["decalage_planning"] = settings.get("decalage_planning", 0) + 1
                            sauvegarder_logs(logs)
                            sauvegarder_settings(settings)
                            st.rerun()
                    with c2:
                        if st.button("⏩ Sauter seulement", use_container_width=True):
                            logs.setdefault("seances_sautees", [])
                            logs["seances_sautees"].append({"date": datetime.now().isoformat(), "seance": nom_seance, "action": "sauter"})
                            sauvegarder_logs(logs)
                            st.success("Séance sautée !")

                st.divider()

                objectifs = coach.resume_seance(nom_seance, exercices)
                groupe_actuel = ""
                for obj in objectifs:
                    ex = obj["exercice"]

                    # Séparateur de groupe musculaire
                    if ex.get("groupe", "") != groupe_actuel:
                        groupe_actuel = ex.get("groupe", "")
                        st.markdown(f"#### 🏷️ {groupe_actuel}")

                    st.markdown(f"### {lien_exercice(ex['name'], ex['video'])}")
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown(f"**Top : {obj['poids']}kg × {obj['reps_par_serie']}**")
                        if ex["sets"] > 1:
                            st.caption(
                                f"Back-off : {obj['backoff_poids']}kg × {obj['backoff_reps']} × {ex['sets'] - 1} séries "
                                f"({int(obj.get('backoff_pourcent', BACKOFF_PERCENT) * 100)}%)"
                            )
                        if obj.get("series_extra_suggerees"):
                            st.caption("📈 Volume boost : +1 série back-off si tu te sens frais.")
                        if "precedent" in obj:
                            st.caption(obj["precedent"])
                    with c2:
                        st.markdown(format_badge_recommandation(obj["recommandation"]))

                    with st.expander("💡 Coach + Échauffement"):
                        st.write(obj["message"])
                        st.caption(f"Repos : {ex['rest_max']}")
                        st.caption(ex["notes"])
                        if obj["poids"] and obj["poids"] >= 30:
                            st.markdown("**Protocole d'échauffement :**")
                            echauf = generer_echauffement(obj["poids"], obj["reps_par_serie"])
                            for e in echauf:
                                st.write(f"• {e['poids']}kg × {e['reps']} — {e['notes']}")
                    st.divider()

    # ========== ONGLET 2 : LOGGER ==========
    with tab2:
        if settings.get("mode_vacances"):
            st.info("Log désactivé pendant les vacances.")
        else:
            st.markdown("## 📝 Logger une séance")

            nom_seance_actuelle, _ = get_seance_aujourdhui(settings)
            noms_seances = list(EXERCICES.keys())
            index_defaut = 0
            for i, n in enumerate(noms_seances):
                if n == nom_seance_actuelle:
                    index_defaut = i
                    break
            seance_choisie = st.selectbox("Séance", noms_seances, index=index_defaut)
            exercices_choisis = EXERCICES[seance_choisie]

            schema = st.radio(
                "Type de séries",
                ["Séries identiques", "Top Set + Back-off"],
                horizontal=True, key="log_schema", index=1
            )
            st.caption("Remplis toute la séance puis sauvegarde en une fois.")

            with st.form("form_session"):
                session_data = []
                for exercice in exercices_choisis:
                    nom_ex = exercice["name"]
                    objectif = coach.get_prochain_objectif(nom_ex, exercice)
                    key_prefix = f"log_{seance_choisie}_{nom_ex}".replace(" ", "_").replace("(", "").replace(")", "")

                    st.markdown(f"### {lien_exercice(nom_ex, exercice['video'])}")
                    st.caption(f"🏷️ {exercice.get('groupe', '')}")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Poids cible", f"{objectif['poids']}kg")
                    with c2:
                        st.metric("Reps cible", f"{objectif['reps_par_serie']}")
                    with c3:
                        st.metric("Statut", objectif["recommandation"])

                    if "precedent" in objectif:
                        st.caption(objectif["precedent"])
                    min_r, max_r = exercice["Nb_rep"]
                    st.caption(f"Plage de reps : {min_r}-{max_r} • Repos : {exercice['rest_max']}")

                    # Alerte retour après pause
                    if objectif.get("jours_depuis_derniere") and objectif["jours_depuis_derniere"] > 14:
                        jours = objectif["jours_depuis_derniere"]
                        st.warning(f"**Bon retour !** Tu n'as pas fait {nom_ex} depuis **{jours} jours**")
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            st.info(f"**Dernière séance :** {objectif['dernier_poids']}kg × {objectif['derniers_reps']} @ RPE {objectif['derniere_rpe']}")
                        with cc2:
                            st.success(f"**Recommandé :** {objectif['poids']}kg ({objectif['pct_decharge']}% décharge)")

                    st.info(objectif["message"])

                    passer = st.checkbox("Passer cet exercice", key=f"{key_prefix}_skip")
                    if passer:
                        st.caption("Passé — rien ne sera enregistré.")
                        st.markdown("---")
                        continue

                    use_backoff = schema == "Top Set + Back-off"
                    increment = get_increment(nom_ex)

                    if use_backoff:
                        poids_top = st.number_input(
                            "Poids Top Set (kg)", min_value=0.0, max_value=500.0,
                            value=float(objectif["poids"]) if objectif["poids"] else 20.0,
                            step=increment, key=f"{key_prefix}_top_w"
                        )
                        backoff_pct = objectif.get("backoff_pourcent", BACKOFF_PERCENT)
                        suggested_bo = arrondir_poids(poids_top * backoff_pct, nom_ex)
                        poids_backoff = st.number_input(
                            "Poids Back-off (kg)", min_value=0.0, max_value=500.0,
                            value=float(suggested_bo), step=increment, key=f"{key_prefix}_bo_w"
                        )
                        poids_travail = poids_top
                        st.caption(
                            f"Back-off cible : {poids_backoff}kg × {objectif.get('backoff_reps', objectif['reps_par_serie'])} reps "
                            f"({int(backoff_pct * 100)}%)"
                        )
                    else:
                        poids_travail = st.number_input(
                            "Poids (kg)", min_value=0.0, max_value=500.0,
                            value=float(objectif["poids"]) if objectif["poids"] else 20.0,
                            step=increment, key=f"{key_prefix}_w"
                        )
                        poids_backoff = poids_travail

                    extra_sets = 0
                    if objectif.get("series_extra_suggerees"):
                        extra_sets = 1 if st.checkbox("Ajouter 1 série extra (recommandé)", key=f"{key_prefix}_extra") else 0

                    sets_data = []
                    nb_series = exercice["sets"] + extra_sets
                    cols = st.columns(min(nb_series, 5))
                    for i in range(nb_series):
                        col_idx = i % len(cols)
                        with cols[col_idx]:
                            if use_backoff:
                                label = "Top Set" if i == 0 else f"Back-off {i}"
                                set_poids = poids_travail if i == 0 else poids_backoff
                                default_reps = objectif["reps_par_serie"] if i == 0 else objectif.get("backoff_reps", objectif["reps_par_serie"])
                            else:
                                label = f"Série {i + 1}"
                                set_poids = poids_travail
                                default_reps = objectif["reps_par_serie"]

                            st.markdown(f"**{label}**")
                            reps = st.number_input("Reps", min_value=0, max_value=50, value=default_reps, key=f"{key_prefix}_r_{i}")
                            rpe = st.slider("RPE", 5.0, 10.0, 8.0, 0.5, key=f"{key_prefix}_rpe_{i}")
                            sets_data.append({"reps": reps, "rpe": rpe, "poids": set_poids})

                    notes = st.text_area("Notes", placeholder="Comment ça s'est passé ?", key=f"{key_prefix}_notes")
                    session_data.append({"exercice": exercice, "poids": poids_travail, "sets": sets_data, "notes": notes})
                    st.markdown("---")

                submitted = st.form_submit_button("✅ Sauvegarder la séance", type="primary", use_container_width=True)

            if submitted:
                if not session_data:
                    st.warning("Aucun exercice logué.")
                    st.stop()
                heure_session = datetime.now().isoformat()
                pr_hits = []
                scores = []

                for entry in session_data:
                    exercice = entry["exercice"]
                    nom_ex = exercice["name"]
                    poids = entry["poids"]
                    sets_data = entry["sets"]
                    notes = entry["notes"]

                    top_poids = max(s.get("poids", poids) for s in sets_data) if sets_data else poids
                    est_pr, ancien_pr = verifier_pr(nom_ex, top_poids, logs)

                    log_entry = {
                        "date": heure_session, "seance": seance_choisie,
                        "poids": top_poids, "sets": sets_data, "notes": notes
                    }
                    logs["exercices"].setdefault(nom_ex, []).insert(0, log_entry)
                    logs["entrainements"].append({
                        "date": heure_session, "type_seance": seance_choisie,
                        "exercice": nom_ex, "data": log_entry
                    })

                    min_r, max_r = exercice["Nb_rep"]
                    score, consistance = coach.score_session(sets_data, min_r, max_r)
                    scores.append(f"{nom_ex} : {score:.0f}/100 — {consistance}")

                    if est_pr and top_poids > ancien_pr:
                        pr_hits.append(f"{nom_ex} : {top_poids}kg (+{top_poids - ancien_pr}kg)")

                sauvegarder_logs(logs)

                if pr_hits:
                    details = "<br>".join(pr_hits)
                    st.markdown(f'<div class="pr-celebration">🏆 NOUVEAU PR ! 🏆<br>{details}</div>', unsafe_allow_html=True)
                    st.balloons()

                st.success("Séance enregistrée !")
                for line in scores:
                    st.write(f"• {line}")

    # ========== ONGLET 3 : PROGRÈS ==========
    with tab3:
        st.markdown("## 📊 Progrès par exercice")
        if not logs.get("exercices"):
            st.info("Pas de données. Commence à logger !")
        else:
            ex_choisi = st.selectbox("Exercice", list(logs["exercices"].keys()), key="progress_ex")
            history = logs["exercices"].get(ex_choisi, [])

            if history:
                df_data = []
                for entry in reversed(history):
                    date = entry.get("date", "")[:10]
                    sets = get_sets_session(entry)
                    top_poids = get_poids_max_session(entry)
                    avg_reps = sum(s.get("reps", 0) for s in sets) / max(1, len(sets))
                    volume = sum(s.get("poids", top_poids) * s.get("reps", 0) for s in sets)
                    df_data.append({"Date": date, "Poids": top_poids, "Reps moy.": round(avg_reps, 1), "Volume": volume})

                df = pd.DataFrame(df_data)
                c1, c2, c3 = st.columns(3)
                with c1:
                    delta = df['Poids'].iloc[-1] - df['Poids'].iloc[0]
                    st.metric("Actuel", f"{df['Poids'].iloc[-1]}kg", f"{delta:+.1f}kg")
                with c2:
                    st.metric("PR", f"{df['Poids'].max()}kg")
                with c3:
                    st.metric("Séances", len(history))

                fig = px.line(df, x="Date", y="Poids", markers=True, title=f"Évolution — {ex_choisi}")
                fig.update_traces(line_color="#4CAF50")
                st.plotly_chart(fig, use_container_width=True)

                # Volume
                fig_vol = px.bar(df, x="Date", y="Volume", title="Volume par séance")
                fig_vol.update_traces(marker_color="#2196F3")
                st.plotly_chart(fig_vol, use_container_width=True)

                # e1RM
                latest = history[0]
                latest_sets = get_sets_session(latest)
                if latest_sets:
                    best = max(latest_sets, key=lambda s: estimer_e1rm(s.get("poids", 0), s.get("reps", 0), s.get("rpe", RPE_CIBLE)))
                    e1rm = estimer_e1rm(best.get("poids", 0), best.get("reps", 0), best.get("rpe", RPE_CIBLE))
                    st.info(f"**e1RM estimé :** {e1rm:.1f}kg (meilleure série : {best.get('poids', 0)}kg × {best.get('reps', 0)} @ RPE {best.get('rpe', RPE_CIBLE)})")

    # ========== ONGLET 4 : STATS ==========
    with tab4:
        st.markdown("## 📈 Statistiques")
        all_stats = coach.stats_globales()

        if not all_stats:
            st.info("Pas de données !")
        else:
            streak = calculer_streak(logs)

            st.markdown("### 🔥 Régularité")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Série actuelle", f"{streak['streak_semaines']} sem.")
            with c2:
                st.metric("Plus longue", f"{streak['plus_longue_streak']} sem.")
            with c3:
                st.metric("Cette semaine", f"{streak['seances_cette_semaine']}/3")
            with c4:
                st.metric("Régularité", f"{streak['regularite_pourcent']}%")

            st.divider()

            revue = get_revue_hebdo(logs)
            recent = revue["recent"]
            precedent = revue["precedent"]
            if recent["sessions"] > 0:
                st.markdown("### 📅 Revue hebdomadaire")
                prev_ok = precedent["sessions"] > 0
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Sessions", recent["sessions"],
                              delta=(recent["sessions"] - precedent["sessions"]) if prev_ok else None)
                with c2:
                    st.metric("Volume", f"{recent['volume']:.0f}kg",
                              delta=f"{recent['volume'] - precedent['volume']:.0f}kg" if prev_ok else None)
                with c3:
                    st.metric("RPE moy.", f"{recent['avg_rpe']:.2f}",
                              delta=round(recent["avg_rpe"] - precedent["avg_rpe"], 2) if prev_ok else None)
                with c4:
                    st.metric("e1RM moy.", f"{recent['avg_e1rm']:.1f}kg",
                              delta=f"{recent['avg_e1rm'] - precedent['avg_e1rm']:.1f}kg" if prev_ok else None)

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Sessions totales", all_stats["total_sessions"])
            with c2:
                st.metric("Séries totales", all_stats["total_sets"])
            with c3:
                st.metric("Reps totales", f"{all_stats['total_reps']:,}")
            with c4:
                st.metric("Volume total", f"{all_stats['total_volume']:,.0f}kg")

            st.divider()
            st.markdown("### 🏅 Records Personnels")
            medailles = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣"]
            for i, pr in enumerate(all_stats["liste_pr"][:8]):
                m = medailles[i] if i < len(medailles) else "•"
                st.write(f"{m} **{pr['exercice']}** : {pr['poids']}kg")

            # Tableau détaillé
            st.divider()
            st.markdown("### 📋 Détail par exercice")
            for nom_ex, ex_stats in all_stats["exercices"].items():
                with st.expander(f"**{nom_ex}**"):
                    cc1, cc2, cc3, cc4 = st.columns(4)
                    with cc1:
                        st.metric("Actuel", f"{ex_stats['poids_actuel']}kg")
                    with cc2:
                        st.metric("PR", f"{ex_stats['poids_max']}kg")
                    with cc3:
                        st.metric("Gain", f"+{ex_stats['gain_poids']}kg")
                    with cc4:
                        st.metric("Sessions", ex_stats["sessions"])

    # ========== ONGLET 5 : OUTILS ==========
    with tab5:
        st.markdown("## 🔢 Outils")
        outil = st.radio("Choisir un outil", ["Calculateur 1RM", "Échauffement", "Mensurations"], horizontal=True)

        if outil == "Calculateur 1RM":
            st.markdown("### 🏋️ Calculateur 1RM")
            c1, c2 = st.columns(2)
            with c1:
                calc_poids = st.number_input("Poids soulevé (kg)", min_value=0.0, value=60.0, step=2.5)
            with c2:
                calc_reps = st.number_input("Reps effectuées", min_value=1, max_value=12, value=5)

            if st.button("Calculer", type="primary"):
                rm = calculer_1rm(calc_poids, calc_reps)
                st.success(f"**1RM estimé : {rm:.1f}kg**")
                st.markdown("### Poids d'entraînement")
                for label, poids in get_pourcentages_1rm(rm).items():
                    st.write(f"• {label} : **{poids}kg**")

        elif outil == "Échauffement":
            st.markdown("### 🔥 Générateur d'échauffement")
            p_travail = st.number_input("Poids de travail (kg)", min_value=10.0, value=60.0, step=2.5)
            r_travail = st.number_input("Reps de travail", min_value=1, max_value=20, value=5)
            if st.button("Générer", type="primary"):
                echauf = generer_echauffement(p_travail, r_travail)
                st.markdown("### Ton protocole")
                for i, e in enumerate(echauf, 1):
                    st.write(f"**Série {i} :** {e['poids']}kg × {e['reps']} — *{e['notes']}*")
                st.info(f"Puis séries de travail : {p_travail}kg × {r_travail} reps")

        else:
            st.markdown("### 📏 Mensurations")
            with st.form("form_mensurations"):
                c1, c2 = st.columns(2)
                with c1:
                    poids_corps = st.number_input("Poids corporel (kg)", min_value=30.0, max_value=200.0, value=70.0)
                    bras_g = st.number_input("Tour de bras gauche (cm)", min_value=15.0, max_value=60.0, value=30.0)
                    poitrine = st.number_input("Tour de poitrine (cm)", min_value=50.0, max_value=150.0, value=90.0)
                with c2:
                    notes_mesure = st.text_input("Notes", placeholder="Matin, détendu...")
                    bras_d = st.number_input("Tour de bras droit (cm)", min_value=15.0, max_value=60.0, value=30.0)
                    taille = st.number_input("Tour de taille (cm)", min_value=40.0, max_value=150.0, value=75.0)

                if st.form_submit_button("Sauvegarder", type="primary"):
                    entry = {
                        "date": datetime.now().isoformat()[:10],
                        "poids_corps": poids_corps,
                        "bras_gauche": bras_g, "bras_droit": bras_d,
                        "poitrine": poitrine, "taille": taille,
                        "notes": notes_mesure
                    }
                    logs.setdefault("mensurations", []).append(entry)
                    sauvegarder_logs(logs)
                    st.success("Mensurations enregistrées !")

            if logs.get("mensurations"):
                st.markdown("### 📊 Historique")
                mdf = pd.DataFrame(logs["mensurations"])
                st.dataframe(mdf, use_container_width=True, hide_index=True)
                if len(mdf) > 1 and "bras_gauche" in mdf.columns:
                    fig = px.line(mdf, x="date", y=["bras_gauche", "bras_droit"], markers=True,
                                  labels={"value": "cm", "variable": "Mesure"}, title="Évolution tour de bras")
                    st.plotly_chart(fig, use_container_width=True)

    # ========== ONGLET 6 : SYSTÈME ==========
    with tab6:
        st.markdown("## 📘 Système de progression")
        st.caption("Comment l'app décide tes poids, reps, back-off et décharges.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("RPE cible", f"{RPE_CIBLE:.1f}")
        with c2:
            st.metric("Back-off", f"{int(BACKOFF_PERCENT * 100)}%")
        with c3:
            st.metric("Saut composé", f"+{POIDS_INCREMENT['compose']}kg")
        with c4:
            st.metric("Saut isolation", f"+{POIDS_INCREMENT['isolation']}kg")

        st.divider()
        st.markdown("### Structure de base")
        st.write("Chaque exercice utilise une série lourde (top set) + des séries de back-off (volume). Tu peux aussi utiliser des séries identiques depuis l'onglet Logger.")

        st.markdown("### Règles de progression")
        st.write("• Ajouter du poids quand toutes les reps atteignent le haut de la fourchette avec un RPE maîtrisé.")
        st.write("• Si tu es proche du max, on maintient le poids et on pousse les reps d'abord.")
        st.write("• Si les reps chutent ou le RPE explose, on consolide ou on décharge.")

        st.markdown("### Ajustements automatiques du back-off")
        st.write("• Si le top set passe mais les back-offs s'effondrent → back-off réduit (jusqu'à 80%).")
        st.write("• Si les back-offs sont faciles et contrôlés → back-off augmenté (jusqu'à 95%).")

        st.markdown("### Fatigue et décharges")
        st.write("• Décharge si RPE très élevé, chute de reps, ou gros écart entre séries.")
        st.write("• Après une longue pause, le système réduit automatiquement la charge et remonte progressivement.")

        st.markdown("### Boost de volume")
        st.write("• En cas de stagnation avec RPE stable → +1 série de back-off recommandée.")

    # ========== ONGLET 7 : RÉGLAGES ==========
    with tab7:
        st.markdown("## ⚙️ Réglages")

        st.markdown("### ❄️ Mode Vacances")
        if settings.get("mode_vacances"):
            if st.button("☀️ Désactiver les vacances", type="primary", use_container_width=True):
                settings["mode_vacances"] = False
                sauvegarder_settings(settings)
                st.rerun()
        else:
            if st.button("❄️ Activer les vacances", use_container_width=True):
                settings["mode_vacances"] = True
                sauvegarder_settings(settings)
                st.rerun()

        st.divider()

        st.markdown("### 💾 Export / Import")
        c1, c2 = st.columns(2)
        with c1:
            csv = exporter_csv(logs)
            st.download_button("📥 Télécharger séances CSV", csv, "coachmuscu_seances.csv", "text/csv", use_container_width=True)
        with c2:
            csv_m = exporter_mensurations_csv(logs)
            st.download_button("📥 Télécharger mensurations CSV", csv_m, "coachmuscu_mensurations.csv", "text/csv", use_container_width=True)

        uploaded = st.file_uploader("Importer un CSV de séances", type=["csv"])
        if uploaded:
            if st.button("Importer"):
                content = uploaded.getvalue().decode("utf-8")
                logs, count = importer_csv(content, logs)
                if count > 0:
                    st.session_state["logs"] = logs
                    st.success(f"{count} entrées importées !")
                else:
                    st.error("Échec de l'import. Vérifie le format.")

        st.divider()

        if settings.get("decalage_planning", 0) > 0:
            st.markdown("### 📅 Planning")
            st.write(f"Décalage : {settings['decalage_planning']} jours")
            if st.button("Réinitialiser le planning"):
                settings["decalage_planning"] = 0
                sauvegarder_settings(settings)
                st.rerun()

        st.divider()
        st.markdown("### 📊 Données")
        st.write(f"Exercices suivis : {len(logs.get('exercices', {}))}")
        st.write(f"Entrainements totaux : {len(logs.get('entrainements', []))}")
        st.write(f"Mensurations : {len(logs.get('mensurations', []))}")

    # Pied de page
    st.divider()
    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption(f"CoachMuscu v1.0 | {datetime.now().strftime('%H:%M')}")
    with c2:
        if st.button("🚪 Déconnexion"):
            st.session_state.authenticated = False
            st.rerun()


if __name__ == "__main__":
    main()
