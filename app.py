import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from PIL import Image
import torch
from typing import List, Dict, Tuple, Optional
import pandas as pd
from datetime import datetime
import json

# Configuration du device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Chargement des modèles BLIP (lazy loading pour économiser la mémoire)
processor = None
caption_model = None
vqa_model = None

def load_models():
    """Charge les modèles BLIP si nécessaire"""
    global processor, caption_model, vqa_model
    if processor is None:
        print("🔄 Chargement des modèles BLIP...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
        print("✅ Modèles BLIP chargés avec succès !")

# Chargement de Mistral 7B pour analyse de texte (lazy loading)
text_classifier = None

def load_text_model():
    """Charge Mistral 7B Instruct pour analyse de texte"""
    global text_classifier
    if text_classifier is None:
        print("🔄 Chargement de Mistral 7B Instruct...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        
        # Charger avec quantization pour économiser mémoire
        text_classifier = {
            "tokenizer": AutoTokenizer.from_pretrained(model_name),
            "model": AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True  # Quantization 8-bit pour GPU limité
            )
        }
        print("✅ Mistral 7B chargé avec succès !")

# CSS personnalisé - Style professionnel bleu sobre (police)
CUSTOM_CSS = """
/* Palette de couleurs police française */
:root {
    --primary-blue: #003366;
    --secondary-blue: #0055A4;
    --light-blue: #E8F1F8;
    --accent-blue: #0066CC;
    --dark-text: #1a1a1a;
    --light-gray: #f5f5f5;
    --border-gray: #d0d0d0;
}

/* En-tête principal */
.main-header {
    background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
    color: white;
    padding: 30px 40px;
    border-radius: 8px;
    margin-bottom: 30px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.main-header h1 {
    margin: 0;
    font-size: 2.5em;
    font-weight: 600;
    letter-spacing: -0.5px;
}

.main-header p {
    margin: 10px 0 0 0;
    font-size: 1.1em;
    opacity: 0.95;
}

/* Onglets */
.gradio-tabs {
    border-radius: 8px;
    overflow: hidden;
}

.gradio-tabs .tab-nav button {
    background-color: var(--light-blue);
    color: var(--primary-blue);
    border: none;
    font-weight: 600;
    padding: 12px 24px;
    transition: all 0.3s ease;
}

.gradio-tabs .tab-nav button.selected {
    background-color: var(--primary-blue);
    color: white;
}

.gradio-tabs .tab-nav button:hover {
    background-color: var(--secondary-blue);
    color: white;
}

/* Cartes et sections */
.section-card {
    background: white;
    border: 1px solid var(--border-gray);
    border-radius: 8px;
    padding: 25px;
    margin: 15px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.section-title {
    color: var(--primary-blue);
    font-size: 1.5em;
    font-weight: 600;
    margin-bottom: 15px;
    border-bottom: 2px solid var(--accent-blue);
    padding-bottom: 10px;
}

/* Boutons */
.gradio-button.primary {
    background: var(--secondary-blue) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 12px 30px !important;
    border-radius: 6px !important;
    transition: all 0.3s ease !important;
}

.gradio-button.primary:hover {
    background: var(--primary-blue) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
}

/* Zone de dépôt de fichiers */
.upload-zone {
    border: 2px dashed var(--accent-blue) !important;
    border-radius: 8px !important;
    background: var(--light-blue) !important;
    padding: 30px !important;
    text-align: center !important;
}

/* Textarea */
textarea {
    border: 1px solid var(--border-gray) !important;
    border-radius: 6px !important;
    padding: 15px !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
}

/* Stats et indicateurs */
.stat-box {
    background: var(--light-blue);
    border-left: 4px solid var(--accent-blue);
    padding: 15px 20px;
    margin: 10px 0;
    border-radius: 4px;
}

.stat-number {
    font-size: 2em;
    font-weight: 700;
    color: var(--primary-blue);
    margin: 0;
}

.stat-label {
    font-size: 0.9em;
    color: var(--dark-text);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Messages de statut */
.success-message {
    background: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
    padding: 12px 20px;
    border-radius: 6px;
    font-weight: 500;
}

.info-message {
    background: var(--light-blue);
    color: var(--primary-blue);
    border: 1px solid var(--accent-blue);
    padding: 12px 20px;
    border-radius: 6px;
}

/* Badge professionnel */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 600;
    margin: 0 5px;
}

.badge-blue {
    background: var(--secondary-blue);
    color: white;
}

/* Responsive */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 1.8em;
    }
    .section-card {
        padding: 15px;
    }
}
"""

# Structure de données globale pour stocker l'état de l'enquête
class EnqueteData:
    def __init__(self):
        self.images = []  # Liste des images uploadées
        self.descriptions = []  # Descriptions IA des images
        self.enquete_info = {
            "titre": "",
            "contexte": "",
            "date_creation": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "nombre_images": 0
        }
        self.analyses = {}  # Résultats d'analyse par image
        self.tags_global = []  # Tous les tags extraits
        
    def to_dict(self):
        """Convertit l'état en dictionnaire sérialisable"""
        return {
            "enquete_info": self.enquete_info,
            "nombre_images": len(self.images),
            "tags_global": self.tags_global
        }

# ============================================================================
# FONCTIONS D'ANALYSE IA - BLIP
# ============================================================================

def generate_caption(image: Image.Image) -> str:
    """Génère une description textuelle de l'image avec BLIP"""
    load_models()
    inputs = processor(image, return_tensors="pt").to(device)
    out = caption_model.generate(**inputs, max_length=100)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def extract_tags_from_description(description: str) -> List[str]:
    """Extrait des tags simples de la description (EN ANGLAIS pour compatibilité)"""
    tags = []
    description_lower = description.lower()
    
    # Mots-clés à détecter (tags en anglais)
    keywords = {
        "people": ["person", "man", "woman", "people", "child", "boy", "girl", "human"],
        "vehicles": ["car", "vehicle", "truck", "motorcycle", "bike", "bus", "train"],
        "buildings": ["building", "house", "structure", "office", "architecture"],
        "documents": ["text", "sign", "document", "paper", "writing", "letters"],
        "outdoor": ["outdoor", "outside", "street", "road", "park"],
        "indoor": ["indoor", "inside", "room", "interior"],
        "objects": ["object", "item", "thing", "equipment"],
        "animals": ["dog", "cat", "animal", "bird", "pet"],
        "nature": ["tree", "garden", "nature", "landscape", "forest", "flowers"]
    }
    
    for tag, words in keywords.items():
        if any(word in description_lower for word in words):
            tags.append(tag)
    
    return tags

def calculate_relevance_score(description: str, tags: List[str], contexte_enquete: str) -> int:
    """
    Calcule un score de pertinence basé sur le contexte de l'enquête
    Score de 0 à 100
    """
    if not contexte_enquete:
        return 50  # Score neutre si pas de contexte
    
    score = 0
    contexte_lower = contexte_enquete.lower()
    description_lower = description.lower()
    
    # Mots-clés du contexte présents dans la description (+20 points par mot)
    contexte_words = [w for w in contexte_lower.split() if len(w) > 4]  # Mots > 4 lettres
    for word in contexte_words[:10]:  # Limiter aux 10 premiers mots significatifs
        if word in description_lower:
            score += 20
    
    # Tags pertinents (+10 points par tag)
    important_tags = ["people", "vehicles", "documents", "buildings", "weapons"]
    for tag in tags:
        if tag in important_tags:
            score += 10
    
    # Bonus si description longue et détaillée (+10 points)
    if len(description.split()) > 8:
        score += 10
    
    # Normaliser entre 0 et 100
    return min(100, score)

def analyze_image_complete(image_data: dict, contexte_enquete: str, image_id: int) -> dict:
    """
    Analyse complète d'une image : description, tags, score
    """
    try:
        description = generate_caption(image_data["image"])
        tags = extract_tags_from_description(description)
        score = calculate_relevance_score(description, tags, contexte_enquete)
        
        return {
            "id": image_id,
            "filename": image_data["filename"],
            "image": image_data["image"],
            "description": description,
            "tags": tags,
            "score": score,
            "analyzed": True
        }
    except Exception as e:
        print(f"Erreur lors de l'analyse de {image_data.get('filename', 'image')}: {e}")
        return {
            "id": image_id,
            "filename": image_data.get("filename", "unknown"),
            "image": image_data.get("image"),
            "description": "Erreur d'analyse",
            "tags": [],
            "score": 0,
            "analyzed": False
        }

def analyze_all_images(state: EnqueteData, progress_callback=None) -> EnqueteData:
    """Analyse toutes les images de l'enquête"""
    contexte = state.enquete_info.get("contexte", "")
    
    for idx, img_data in enumerate(state.images):
        # Vérifier si l'image a déjà été analysée
        if idx not in state.analyses or not state.analyses[idx].get("analyzed", False):
            analysis = analyze_image_complete(img_data, contexte, idx)
            state.analyses[idx] = analysis
            
            # Ajouter les tags au pool global
            for tag in analysis["tags"]:
                if tag not in state.tags_global:
                    state.tags_global.append(tag)
            
            if progress_callback:
                progress_callback(f"Analyse {idx + 1}/{len(state.images)}...")
    
    return state

# ============================================================================
# PAGE 1 : ACCUEIL - Import et Contexte de l'Enquête
# ============================================================================

def page_accueil_init_images(files, current_state):
    """
    Gère l'upload des images et met à jour l'état
    """
    if not files:
        return "⚠️ Aucune image sélectionnée.", current_state, ""
    
    # Créer ou récupérer l'état
    if current_state is None:
        state = EnqueteData()
    else:
        state = current_state
    
    # Charger les images
    new_images = []
    for file_path in files:
        try:
            img = Image.open(file_path).convert('RGB')
            new_images.append({
                "image": img,
                "filename": file_path.split("/")[-1] if "/" in file_path else file_path.split("\\")[-1],
                "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            print(f"Erreur lors du chargement de {file_path}: {e}")
    
    state.images.extend(new_images)
    state.enquete_info["nombre_images"] = len(state.images)
    
    # Message de confirmation
    message = f"""
    <div class="success-message">
        ✅ <strong>{len(new_images)} image(s)</strong> uploadée(s) avec succès
        <br>
        📊 Total dans l'enquête : <strong>{len(state.images)} image(s)</strong>
    </div>
    """
    
    # Statistiques
    stats_html = generate_stats_html(state)
    
    return message, state, stats_html

def page_accueil_save_context(titre, contexte, current_state):
    """
    Enregistre le contexte de l'enquête
    """
    if current_state is None:
        state = EnqueteData()
    else:
        state = current_state
    
    state.enquete_info["titre"] = titre
    state.enquete_info["contexte"] = contexte
    
    message = """
    <div class="success-message">
        ✅ Informations de l'enquête enregistrées avec succès
    </div>
    """
    
    return message, state

def generate_stats_html(state: EnqueteData) -> str:
    """
    Génère l'affichage HTML des statistiques de l'enquête
    """
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif;">
        <div class="stat-box">
            <p class="stat-number">{state.enquete_info['nombre_images']}</p>
            <p class="stat-label">Images uploadées</p>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 6px; border: 1px solid #d0d0d0;">
            <h4 style="color: #003366; margin-top: 0;">📋 Informations de l'enquête</h4>
            <p><strong>Titre :</strong> {state.enquete_info['titre'] or '<em>Non défini</em>'}</p>
            <p><strong>Date de création :</strong> {state.enquete_info['date_creation']}</p>
            <p style="margin-bottom: 0;"><strong>Statut :</strong> <span class="badge badge-blue">En cours</span></p>
        </div>
    </div>
    """
    return html

# ============================================================================
# PAGE 2 : RECHERCHE - Recherche textuelle dans les images
# ============================================================================

def page_recherche_analyze_if_needed(current_state):
    """Lance l'analyse des images si pas encore fait"""
    if current_state is None or len(current_state.images) == 0:
        return """
        <div class="info-message">
            ℹ️ Aucune image n'a été importée. Commencez par la page <strong>Accueil</strong> pour uploader des images.
        </div>
        """, current_state
    
    # Vérifier si toutes les images ont été analysées
    needs_analysis = False
    for idx in range(len(current_state.images)):
        if idx not in current_state.analyses or not current_state.analyses[idx].get("analyzed", False):
            needs_analysis = True
            break
    
    if needs_analysis:
        status_msg = f"""
        <div class="info-message">
            🔄 Analyse de {len(current_state.images)} image(s) en cours avec BLIP...<br>
            Veuillez patienter quelques instants.
        </div>
        """
        current_state = analyze_all_images(current_state)
        status_msg = f"""
        <div class="success-message">
            ✅ {len(current_state.images)} image(s) analysée(s) avec succès !<br>
            Utilisez la barre de recherche ci-dessous.
        </div>
        """
    else:
        status_msg = f"""
        <div class="success-message">
            ✅ {len(current_state.analyses)} image(s) déjà analysée(s) et prêtes pour la recherche.
        </div>
        """
    
    return status_msg, current_state

def page_recherche_search(query: str, current_state):
    """
    Recherche textuelle flexible et intelligente dans les descriptions et tags
    Supporte : correspondances partielles, synonymes, variations de mots
    """
    if current_state is None or len(current_state.analyses) == 0:
        return """
        <div class="info-message">
            ⚠️ Aucune image analysée. Cliquez d'abord sur "Analyser les images".
        </div>
        """
    
    if not query or len(query.strip()) < 2:
        return """
        <div class="info-message">
            ℹ️ Entrez au moins 2 caractères pour effectuer une recherche.
        </div>
        """
    
    query_lower = query.lower().strip()
    
    # Dictionnaire COMPLET de traduction FR→EN et variations
    # Organisation par CONCEPT pour recherche sémantique large
    fr_to_en_concepts = {
        # PERSONNES (toutes variations)
        "personne": ["person", "people", "man", "men", "woman", "women", "human", "humans", "individual", "individuals", "boy", "boys", "girl", "girls", "child", "children", "kid", "kids", "face", "faces"],
        "homme": ["man", "men", "male", "guy", "person", "people", "human"],
        "femme": ["woman", "women", "female", "lady", "person", "people", "human"],
        "gens": ["people", "persons", "humans", "crowd", "group"],
        "enfant": ["child", "children", "kid", "kids", "boy", "boys", "girl", "girls"],
        "garçon": ["boy", "boys", "child", "kid"],
        "fille": ["girl", "girls", "child", "kid"],
        
        # VÉHICULES (tous types)
        "voiture": ["car", "cars", "vehicle", "vehicles", "automobile", "auto"],
        "véhicule": ["vehicle", "vehicles", "car", "cars", "truck", "trucks", "automobile", "transportation"],
        "auto": ["car", "cars", "automobile", "vehicle"],
        "camion": ["truck", "trucks", "van", "vehicle"],
        "moto": ["motorcycle", "motorcycles", "bike", "motorbike"],
        "vélo": ["bike", "bikes", "bicycle", "bicycles", "cycling"],
        
        # ARMES (tous types)
        "arme": ["weapon", "weapons", "gun", "guns", "knife", "knives", "rifle", "blade", "firearm"],
        "couteau": ["knife", "knives", "blade", "blades", "cutting", "sharp"],
        "pistolet": ["pistol", "gun", "handgun", "firearm"],
        "fusil": ["rifle", "gun", "firearm", "weapon"],
        
        # BÂTIMENTS & LIEUX
        "bâtiment": ["building", "buildings", "structure", "structures", "architecture"],
        "batiment": ["building", "buildings", "structure", "structures"],
        "maison": ["house", "houses", "home", "building"],
        "immeuble": ["building", "buildings", "apartment", "structure"],
        "lieu": ["place", "places", "location", "locations", "site"],
        
        # DOCUMENTS & TEXTES
        "document": ["document", "documents", "paper", "papers", "file"],
        "papier": ["paper", "papers", "document", "sheet"],
        "texte": ["text", "texts", "writing", "written"],
        "écrit": ["writing", "written", "text", "script"],
        "lettre": ["letter", "letters", "writing"],
        "signe": ["sign", "signs", "signage"],
        
        # ANIMAUX (CATÉGORIE LARGE - FIX PRINCIPAL)
        "animal": ["dog", "dogs", "cat", "cats", "animal", "animals", "pet", "pets", "bird", "birds", "horse", "horses", "wildlife"],
        "animaux": ["dog", "dogs", "cat", "cats", "animal", "animals", "pet", "pets", "bird", "birds", "horse", "horses"],
        "chien": ["dog", "dogs", "puppy", "canine"],
        "chat": ["cat", "cats", "kitten", "feline"],
        "oiseau": ["bird", "birds", "flying"],
        
        # ENVIRONNEMENT
        "extérieur": ["outdoor", "outdoors", "outside", "exterior", "external"],
        "exterieur": ["outdoor", "outdoors", "outside", "exterior"],
        "dehors": ["outside", "outdoor", "outdoors", "exterior"],
        "rue": ["street", "streets", "road", "roads"],
        "route": ["road", "roads", "street", "highway"],
        "parc": ["park", "parks", "garden"],
        
        "intérieur": ["indoor", "indoors", "inside", "interior", "internal"],
        "interieur": ["indoor", "indoors", "inside", "interior"],
        "dedans": ["inside", "indoor", "indoors", "interior"],
        "pièce": ["room", "rooms", "space"],
        "piece": ["room", "rooms", "space"],
        "salle": ["room", "rooms", "hall"],
        "chambre": ["room", "bedroom", "chamber"],
        
        # OBJETS
        "objet": ["object", "objects", "item", "items", "thing", "things"],
        "chose": ["thing", "things", "object", "item"],
        "outil": ["tool", "tools", "implement", "equipment"]
    }
    
    # Construire la liste ÉTENDUE de termes de recherche
    search_terms = [query_lower]
    
    # Vérifier si le mot recherché est une clé FR du dictionnaire
    if query_lower in fr_to_en_concepts:
        # Ajouter TOUTES les traductions EN
        search_terms.extend(fr_to_en_concepts[query_lower])
        print(f"Mot FR '{query_lower}' traduit vers: {fr_to_en_concepts[query_lower][:5]}...")
    
    # Vérifier aussi les variations (avec/sans accents)
    # Ex: "batiment" → "bâtiment" → traductions
    for fr_word, en_translations in fr_to_en_concepts.items():
        if query_lower in fr_word or fr_word in query_lower:
            search_terms.extend(en_translations)
    
    # Enlever les doublons
    search_terms = list(set(search_terms))
    
    # Extraire les mots individuels de la requête (pour recherche multi-mots)
    query_words = [w for w in query_lower.split() if len(w) > 2]
    
    print(f"\n=== Recherche: '{query}' ===")
    print(f"Search terms (avec synonymes): {search_terms[:10]}")
    print(f"Query words: {query_words}")
    
    results = []
    
    # Rechercher dans toutes les analyses
    for img_id, analysis in current_state.analyses.items():
        if not analysis.get("analyzed", False):
            continue
        
        description = analysis.get("description", "").lower()
        description_words = set(description.split())  # Convertir en set pour recherche rapide
        tags = [t.lower() for t in analysis.get("tags", [])]
        filename = analysis.get("filename", "").lower()
        categories = [c.lower() for c in analysis.get("categories", [])]
        
        match_score = 0
        matched_terms = []
        
        print(f"\n  Analyzing image {img_id}: {analysis.get('filename', 'unknown')}")
        print(f"    Description: {description}")
        print(f"    Categories: {categories}")
        print(f"    Tags: {tags}")
        
        # 1. Correspondance dans les MOTS de la description (PRIORITÉ TRÈS HAUTE)
        for term in search_terms:
            if term in description_words:
                match_score += 10
                matched_terms.append(f"Mot exact: '{term}'")
                print(f"    ✓ Match exact mot '{term}' dans description")
        
        # 2. Correspondance dans TAGS (PRIORITÉ HAUTE)
        for term in search_terms:
            if term in tags:
                match_score += 8
                matched_terms.append(f"Tag: '{term}'")
                print(f"    ✓ Match tag '{term}'")
        
        # 3. Correspondance dans CATÉGORIES (PRIORITÉ HAUTE)
        for term in search_terms:
            if term in categories:
                match_score += 8
                matched_terms.append(f"Catégorie: '{term}'")
                print(f"    ✓ Match catégorie '{term}'")
        
        # 4. Correspondance SUBSTRING dans description (PRIORITÉ MOYENNE)
        # Pour trouver si "person" est dans "a person standing"
        for term in search_terms:
            if len(term) >= 3 and term in description and term not in description_words:
                match_score += 5
                matched_terms.append(f"Dans description: '{term}'")
                print(f"    ✓ Match substring '{term}' dans description")
        
        # 5. Correspondance PARTIELLE par préfixe (PRIORITÉ MOYENNE)
        for term in search_terms:
            if len(term) >= 4:
                for word in description_words:
                    if len(word) >= 4:
                        # Vérifier préfixe commun de 4 caractères
                        if word.startswith(term[:4]) or term.startswith(word[:4]):
                            match_score += 3
                            matched_terms.append(f"Préfixe: '{word}' ≈ '{term}'")
                            print(f"    ✓ Match préfixe '{word}' ≈ '{term}'")
                            break
        
        # 6. Correspondance dans FILENAME (PRIORITÉ BASSE)
        for term in search_terms:
            if term in filename:
                match_score += 2
                matched_terms.append(f"Filename: '{term}'")
                print(f"    ✓ Match filename '{term}'")
        
        if match_score > 0:
            results.append({
                "analysis": analysis,
                "match_score": match_score,
                "matched_terms": matched_terms[:5]
            })
            print(f"    → TOTAL SCORE: {match_score}")
        else:
            print(f"    → Aucun match")
    
    # Trier par match_score puis par score de pertinence
    results.sort(key=lambda x: (x["match_score"], x["analysis"].get("score", 0)), reverse=True)
    
    print(f"\nTotal résultats trouvés: {len(results)}\n")
    
    if len(results) == 0:
        return f"""
        <div class="info-message">
            ❌ Aucun résultat pour "<strong>{query}</strong>".<br>
            Essayez avec d'autres mots-clés.
        </div>
        """
    
    # Générer l'affichage HTML des résultats
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif;">
        <div style="background: var(--light-blue); padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="margin: 0; color: var(--primary-blue);">
                🔍 {len(results)} résultat(s) pour "{query}"
            </h3>
        </div>
    """
    
    for idx, result in enumerate(results):
        analysis = result["analysis"]
        
        # Barre de score colorée
        score = analysis["score"]
        if score >= 70:
            score_color = "#28a745"
            score_label = "Haute pertinence"
        elif score >= 40:
            score_color = "#ffc107"
            score_label = "Pertinence moyenne"
        else:
            score_color = "#dc3545"
            score_label = "Faible pertinence"
        
        html += f"""
        <div style="background: white; border: 2px solid {score_color}; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">
                <div style="flex: 1;">
                    <h4 style="margin: 0 0 5px 0; color: var(--primary-blue);">
                        📄 {analysis['filename']}
                    </h4>
                    <p style="margin: 0; color: #666; font-size: 0.9em;">Image #{analysis['id'] + 1}</p>
                </div>
                <div style="text-align: right;">
                    <div style="background: {score_color}; color: white; padding: 8px 16px; border-radius: 20px; font-weight: 600; font-size: 1.1em;">
                        {score}%
                    </div>
                    <p style="margin: 5px 0 0 0; font-size: 0.85em; color: #666;">{score_label}</p>
                </div>
            </div>
            
            <div style="background: var(--light-blue); padding: 15px; border-radius: 6px; margin: 15px 0;">
                <p style="margin: 0 0 5px 0; font-weight: 600; color: var(--primary-blue);">📝 Description :</p>
                <p style="margin: 0; line-height: 1.6;">{analysis['description']}</p>
            </div>
            
            <div style="margin-top: 10px;">
                <p style="margin: 0 0 8px 0; font-weight: 600; color: var(--primary-blue);">🏷️ Tags :</p>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
        """
        
        for tag in analysis['tags']:
            html += f"""
                    <span style="background: var(--secondary-blue); color: white; padding: 5px 12px; border-radius: 15px; font-size: 0.9em; font-weight: 500;">
                        {tag}
                    </span>
            """
        
        if not analysis['tags']:
            html += '<span style="color: #999; font-style: italic;">Aucun tag</span>'
        
        html += """
                </div>
            </div>
        </div>
        """
    
    html += "</div>"
    return html

# ============================================================================
# PAGE 3 : CATÉGORISATION - Classification automatique par catégories
# ============================================================================

# Définition des catégories pour enquêtes de police (EN ANGLAIS pour compatibilité BLIP)
CATEGORIES_POLICE = {
    "people": {
        "icon": "👤",
        "label": "People",
        "label_fr": "Personnes",
        "description": "Images contenant des personnes, visages, suspects",
        "color": "#FF6B6B"
    },
    "vehicles": {
        "icon": "🚗",
        "label": "Vehicles",
        "label_fr": "Véhicules",
        "description": "Voitures, motos, camions, plaques d'immatriculation",
        "color": "#4ECDC4"
    },
    "weapons": {
        "icon": "⚠️",
        "label": "Weapons/Suspicious",
        "label_fr": "Armes/Suspects",
        "description": "Armes, objets dangereux, éléments suspects",
        "color": "#FF4444"
    },
    "documents": {
        "icon": "📄",
        "label": "Documents/Text",
        "label_fr": "Documents/Textes",
        "description": "Documents, papiers, textes, panneaux, inscriptions",
        "color": "#95E1D3"
    },
    "buildings": {
        "icon": "🏢",
        "label": "Buildings/Places",
        "label_fr": "Bâtiments/Lieux",
        "description": "Bâtiments, maisons, structures, scènes de crime",
        "color": "#F38181"
    },
    "outdoor": {
        "icon": "🌳",
        "label": "Outdoor",
        "label_fr": "Extérieur",
        "description": "Extérieur, rues, parcs, nature",
        "color": "#A8E6CF"
    },
    "indoor": {
        "icon": "🏠",
        "label": "Indoor",
        "label_fr": "Intérieur",
        "description": "Intérieur de bâtiments, pièces, chambres",
        "color": "#FFEAA7"
    },
    "objects": {
        "icon": "📦",
        "label": "Objects",
        "label_fr": "Objets",
        "description": "Objets, preuves matérielles, équipements",
        "color": "#DFE6E9"
    },
    "animals": {
        "icon": "🐾",
        "label": "Animals",
        "label_fr": "Animaux",
        "description": "Animaux domestiques ou sauvages (chiens, chats, etc.)",
        "color": "#FFA07A"
    },
    "unclassified": {
        "icon": "❓",
        "label": "Unclassified",
        "label_fr": "Non classifié",
        "description": "Images non classifiées automatiquement",
        "color": "#B2BEC3"
    }
}

def ask_vqa_question(image: Image.Image, question: str) -> str:
    """Pose une question VQA à une image"""
    load_models()
    try:
        inputs = processor(image, question, return_tensors="pt").to(device)
        out = vqa_model.generate(**inputs, max_length=50)
        answer = processor.decode(out[0], skip_special_tokens=True)
        return answer.lower().strip()
    except Exception as e:
        print(f"Erreur VQA: {e}")
        return ""

def classify_image_by_category(image_data: dict, image_id: int) -> List[str]:
    """
    Classifie une image dans une ou plusieurs catégories de manière interprétative
    AMÉLIORÉ : Utilise PLUSIEURS questions VQA détaillées par catégorie
    Retourne une liste de catégories (multi-catégories possible)
    """
    image = image_data["image"]
    categories_assigned = []
    
    # 1. Obtenir la description de l'image
    description = generate_caption(image).lower()
    print(f"\n=== Analyzing image {image_id} ===")
    print(f"Description: {description}")
    
    # 2. Configuration des catégories avec QUESTIONS MULTIPLES détaillées
    category_analysis = {
        "people": {
            "keywords": ["person", "man", "woman", "people", "child", "boy", "girl", "human", "face", "crowd", "group"],
            "vqa_questions": [
                "Are there any people, persons, or human beings visible in this image?",
                "Can you see a man, woman, or child in this picture?",
                "Is there a human face or body visible?"
            ],
            "weight": 1.0
        },
        "vehicles": {
            "keywords": ["car", "vehicle", "truck", "motorcycle", "bike", "bus", "train", "automobile", "taxi", "van"],
            "vqa_questions": [
                "Can you see any vehicles, cars, or means of transportation?",
                "Is there a car, truck, motorcycle, or bicycle in this image?",
                "Are there any wheels or vehicle parts visible?"
            ],
            "weight": 1.0
        },
        "weapons": {
            "keywords": ["weapon", "gun", "knife", "rifle", "pistol", "blade", "sharp", "firearm", "cutting"],
            "vqa_questions": [
                "Is there a knife, blade, or sharp cutting tool visible in this image?",
                "Can you see a gun, firearm, rifle, or pistol?",
                "What tool or implement is being used or held in this image?",
                "Are there any weapons, blades, or sharp metallic objects?"
            ],
            "weight": 1.2,
            # Liste d'exclusion STRICTE pour éviter faux positifs
            "exclude_keywords": ["dog", "cat", "pet", "animal", "bird", "horse"],
            # Si SEULEMENT ces mots apparaissent (sans knife/gun/blade), alors exclure
            "exclude_only_if_alone": True
        },
        "documents": {
            "keywords": ["document", "paper", "text", "sign", "writing", "letter", "book", "page", "note", "card", "words"],
            "vqa_questions": [
                "Is there any text, document, or written content visible?",
                "Can you see any words, letters, or writing in this image?",
                "Are there any signs, papers, or documents?"
            ],
            "weight": 1.0
        },
        "buildings": {
            "keywords": ["building", "house", "structure", "architecture", "wall", "door", "window", "roof", "facade"],
            "vqa_questions": [
                "Can you see any buildings, houses, or architectural structures?",
                "Is there a wall, door, window, or building structure visible?",
                "Is this image taken in front of or inside a building?"
            ],
            "weight": 1.0
        },
        "outdoor": {
            "keywords": ["outdoor", "outside", "street", "road", "park", "sky", "nature", "exterior", "sidewalk"],
            "vqa_questions": [
                "Is this an outdoor scene or taken outside?",
                "Can you see the sky, street, or outdoor environment?"
            ],
            "weight": 0.8
        },
        "indoor": {
            "keywords": ["indoor", "inside", "room", "interior", "ceiling", "floor", "furniture", "wall"],
            "vqa_questions": [
                "Is this an indoor scene or taken inside a building?",
                "Can you see a room, ceiling, or interior space?"
            ],
            "weight": 0.8
        },
        "objects": {
            "keywords": ["object", "item", "thing", "tool", "equipment", "device", "bag", "box", "bottle", "holding"],
            "vqa_questions": [
                "Are there any specific objects, items, or things in this image?",
                "What objects or items can you see in this picture?"
            ],
            "weight": 0.7
        },
        "animals": {
            "keywords": ["dog", "cat", "animal", "pet", "bird", "horse"],
            "vqa_questions": [
                "Is there a dog, cat, or any animal in this image?",
                "Can you see a pet or animal?"
            ],
            "weight": 0.6  # Poids faible, généralement pas prioritaire pour enquêtes
        }
    }
    
    # 3. Scorer chaque catégorie de manière intelligente avec QUESTIONS MULTIPLES
    category_scores = {}
    
    for category, config in category_analysis.items():
        score = 0.0
        
        # A. Analyse des mots-clés dans la description
        keyword_matches = sum(1 for keyword in config["keywords"] if keyword in description)
        if keyword_matches > 0:
            score += (keyword_matches * 20) * config["weight"]
            print(f"{category}: Found {keyword_matches} keyword(s) in description")
        
        # B. Poser TOUTES les questions VQA pour cette catégorie
        positive_answers = 0
        total_questions = len(config["vqa_questions"])
        has_exclusion = False
        
        for i, question in enumerate(config["vqa_questions"]):
            vqa_answer = ask_vqa_question(image, question)
            print(f"{category} VQA Q{i+1}/{total_questions}: '{vqa_answer}'")
            
            if vqa_answer:
                vqa_lower = vqa_answer.lower()
                
                # VÉRIFICATION D'EXCLUSION (pour éviter faux positifs)
                exclude_list = config.get("exclude_keywords", [])
                if exclude_list:
                    # Vérifier si des mots d'exclusion sont présents
                    excluded_found = [e for e in exclude_list if e in vqa_lower]
                    
                    if excluded_found:
                        # Vérifier si c'est SEULEMENT un animal/objet quotidien (sans arme réelle)
                        weapon_words = ["knife", "gun", "blade", "weapon", "rifle", "pistol", "sharp", "cutting"]
                        has_weapon_word = any(w in vqa_lower for w in weapon_words)
                        
                        # Si SEULEMENT animal/quotidien SANS mot d'arme → exclusion
                        if not has_weapon_word and config.get("exclude_only_if_alone", False):
                            print(f"  → Q{i+1} EXCLUDED (faux positif: {excluded_found}, pas d'arme réelle)")
                            has_exclusion = True
                            score -= 20  # Pénalité
                            continue
                        elif not has_weapon_word:
                            # Petite pénalité mais pas exclusion totale
                            score -= 5
                            print(f"  → Q{i+1} Objet quotidien détecté ({excluded_found}), pénalité légère")
                
                # Réponses positives claires
                if any(word in vqa_lower for word in ["yes", "true", "there is", "there are", "visible", "can see", "holding"]):
                    positive_answers += 1
                    score += 25 * config["weight"]
                    print(f"  → Q{i+1} Positive (+{25 * config['weight']:.1f})")
                
                # Réponses négatives claires
                elif any(word in vqa_lower for word in ["no", "not", "none", "cannot", "can't", "nothing"]):
                    score -= 5
                    print(f"  → Q{i+1} Negative (-5)")
                
                # Réponses contenant des éléments de la catégorie (détection implicite)
                elif any(keyword in vqa_lower for keyword in config["keywords"][:8]):
                    positive_answers += 0.5
                    score += 20 * config["weight"]
                    print(f"  → Q{i+1} Mentions category (+{20 * config['weight']:.1f})")
                
                # Réponses descriptives (ex: "knife", "cutting tool")
                else:
                    # Vérifier si la réponse contient des mots pertinents
                    answer_words = vqa_lower.split()
                    if any(word in answer_words for word in config["keywords"][:10]):
                        positive_answers += 0.3
                        score += 15 * config["weight"]
                        print(f"  → Q{i+1} Descriptive match (+{15 * config['weight']:.1f})")
        
        # Si exclusion détectée, annuler le score pour cette catégorie
        if has_exclusion and category == "weapons":
            score = max(0, score - 30)  # Pénalité supplémentaire pour weapons
            print(f"  → EXCLUSION penalty applied, score reduced")
        
        # Bonus si plusieurs questions confirment la catégorie
        if positive_answers >= 2:
            bonus = 20 * config["weight"]
            score += bonus
            print(f"  → Multiple confirmations bonus (+{bonus:.1f})")
        
        category_scores[category] = score
    
    # 4. Sélection des catégories avec seuil adaptatif
    # Seuils différents selon la catégorie pour éviter faux positifs
    category_thresholds = {
        "weapons": 30,      # Seuil modéré pour weapons (équilibre détection/précision)
        "people": 20,
        "vehicles": 20,
        "documents": 20,
        "buildings": 20,
        "outdoor": 15,
        "indoor": 15,
        "objects": 25,      # Seuil plus élevé car très générique
        "animals": 18       # Seuil normal pour animaux
    }
    
    max_categories = 5
    
    # Trier par score
    sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nScores finaux:")
    for cat, score in sorted_categories:
        threshold = category_thresholds.get(cat, 20)
        print(f"  {cat}: {score:.1f} (seuil: {threshold})")
    
    # Assigner les catégories au-dessus de leur seuil spécifique
    for category, score in sorted_categories:
        threshold = category_thresholds.get(category, 20)
        if score >= threshold and len(categories_assigned) < max_categories:
            categories_assigned.append(category)
            print(f"  ✓ Assigned to {category} (score: {score:.1f}, threshold: {threshold})")
    
    # 5. Gérer les conflits indoor/outdoor
    if "indoor" in categories_assigned and "outdoor" in categories_assigned:
        if category_scores["indoor"] > category_scores["outdoor"]:
            categories_assigned.remove("outdoor")
            print("  → Removed 'outdoor' (conflict with indoor)")
        else:
            categories_assigned.remove("indoor")
            print("  → Removed 'indoor' (conflict with outdoor)")
    
    # 6. Si aucune catégorie significative
    if not categories_assigned:
        categories_assigned.append("unclassified")
        print("  ✗ No significant category found, marked as unclassified")
    
    print(f"Final categories: {categories_assigned}\n")
    return categories_assigned

def page_categorisation_analyze(current_state):
    """Analyse et catégorise toutes les images"""
    if current_state is None or len(current_state.images) == 0:
        return """
        <div class="info-message">
            ℹ️ Aucune image n'a été importée. Commencez par la page <strong>Accueil</strong> pour uploader des images.
        </div>
        """, current_state, "", {}
    
    # S'assurer que les images sont analysées (descriptions, tags)
    if len(current_state.analyses) < len(current_state.images):
        current_state = analyze_all_images(current_state)
    
    # Catégoriser chaque image
    categories_count = {cat: 0 for cat in CATEGORIES_POLICE.keys()}
    
    for idx, img_data in enumerate(current_state.images):
        if idx in current_state.analyses:
            analysis = current_state.analyses[idx]
            
            # Toujours re-classifier (forcer la re-classification)
            print(f"Classifying image {idx}: {img_data.get('filename', 'unknown')}")
            categories = classify_image_by_category(img_data, idx)
            analysis["categories"] = categories
            
            # Compter les catégories
            for cat in categories:
                if cat in categories_count:
                    categories_count[cat] += 1
    
    status_msg = f"""
    <div class="success-message">
        ✅ {len(current_state.images)} image(s) catégorisée(s) avec succès !<br>
        Cliquez sur une catégorie à gauche pour voir les images correspondantes.
    </div>
    """
    
    # Générer le HTML de la sidebar
    sidebar_html = generate_categories_sidebar(categories_count)
    
    return status_msg, current_state, sidebar_html, categories_count

def generate_categories_sidebar(categories_count: dict) -> str:
    """Génère le HTML de la sidebar avec les catégories"""
    html = """
    <div style="font-family: 'Segoe UI', Arial, sans-serif; background: white; border-radius: 8px; padding: 15px;">
        <h3 style="color: var(--primary-blue); margin-top: 0; border-bottom: 2px solid var(--accent-blue); padding-bottom: 10px;">
            📂 Catégories
        </h3>
    """
    
    for cat_id, cat_info in CATEGORIES_POLICE.items():
        count = categories_count.get(cat_id, 0)
        label_display = cat_info.get('label_fr', cat_info['label'])  # Utiliser label_fr si disponible
        
        html += f"""
        <div style="margin: 10px 0; padding: 12px; background: {cat_info['color']}20; border-left: 4px solid {cat_info['color']}; border-radius: 6px; cursor: pointer; transition: all 0.3s;" 
             onmouseover="this.style.background='{cat_info['color']}40'" 
             onmouseout="this.style.background='{cat_info['color']}20'">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.5em; margin-right: 8px;">{cat_info['icon']}</span>
                    <strong style="color: var(--dark-text);">{label_display}</strong>
                </div>
                <span style="background: {cat_info['color']}; color: white; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.9em;">
                    {count}
                </span>
            </div>
            <p style="margin: 5px 0 0 0; font-size: 0.85em; color: #666; padding-left: 32px;">
                {cat_info['description']}
            </p>
        </div>
        """
    
    html += "</div>"
    return html

def page_categorisation_filter(category_id: str, current_state):
    """Filtre et affiche les images d'une catégorie spécifique"""
    if current_state is None or len(current_state.analyses) == 0:
        return """
        <div class="info-message">
            ⚠️ Aucune image catégorisée. Cliquez d'abord sur "Catégoriser les images".
        </div>
        """
    
    if not category_id or category_id == "all":
        # Afficher toutes les images
        filtered_images = list(current_state.analyses.values())
        title = "Toutes les catégories"
    else:
        # Filtrer par catégorie
        filtered_images = []
        for analysis in current_state.analyses.values():
            if "categories" in analysis and category_id in analysis["categories"]:
                filtered_images.append(analysis)
        
        cat_info = CATEGORIES_POLICE.get(category_id, {"label": "Catégorie inconnue", "icon": "❓"})
        title = f"{cat_info['icon']} {cat_info['label']}"
    
    if len(filtered_images) == 0:
        return f"""
        <div class="info-message">
            ℹ️ Aucune image trouvée dans la catégorie "<strong>{title}</strong>".
        </div>
        """
    
    # Générer l'affichage HTML
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif;">
        <div style="background: var(--light-blue); padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h2 style="margin: 0; color: var(--primary-blue);">
                {title}
            </h2>
            <p style="margin: 5px 0 0 0; color: #666;">
                {len(filtered_images)} image(s) dans cette catégorie
            </p>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px;">
    """
    
    for analysis in filtered_images:
        # Couleurs pour les badges de catégories
        categories_badges = ""
        if "categories" in analysis:
            for cat in analysis["categories"]:
                if cat in CATEGORIES_POLICE:
                    cat_info = CATEGORIES_POLICE[cat]
                    label_display = cat_info.get('label_fr', cat_info['label'])
                    categories_badges += f"""
                    <span style="background: {cat_info['color']}; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.85em; margin: 2px; display: inline-block;">
                        {cat_info['icon']} {label_display}
                    </span>
                    """
        
        html += f"""
        <div style="background: white; border: 2px solid var(--border-gray); border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="margin: 0 0 10px 0; color: var(--primary-blue);">
                📄 {analysis['filename']}
            </h4>
            
            <div style="margin: 10px 0;">
                <p style="margin: 0 0 5px 0; font-weight: 600; color: var(--primary-blue);">📝 Description :</p>
                <p style="margin: 0; color: #555; line-height: 1.5; font-size: 0.95em;">
                    {analysis.get('description', 'Non disponible')}
                </p>
            </div>
            
            <div style="margin: 10px 0;">
                <p style="margin: 0 0 8px 0; font-weight: 600; color: var(--primary-blue);">🏷️ Catégories :</p>
                <div style="display: flex; flex-wrap: wrap; gap: 5px;">
                    {categories_badges}
                </div>
            </div>
            
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #666; font-size: 0.9em;">Score de pertinence</span>
                    <span style="background: var(--secondary-blue); color: white; padding: 5px 12px; border-radius: 15px; font-weight: 600;">
                        {analysis.get('score', 0)}%
                    </span>
                </div>
            </div>
        </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    return html

# ============================================================================
# ANALYSE DE TEXTE avec Mistral 7B
# ============================================================================

def analyze_text_with_mistral(text_content: str, contexte_enquete: str) -> dict:
    """
    Analyse un texte (email, SMS, note) avec Mistral 7B et le classifie
    Retourne : catégorie (pertinent/a_traiter/non_pertinent), score, explication
    """
    load_text_model()
    
    # Créer le prompt pour Mistral
    prompt = f"""<s>[INST] Tu es un assistant d'analyse forensique pour la police. Ton rôle est d'analyser des textes (emails, SMS, notes) dans le cadre d'une enquête et de déterminer leur pertinence.

CONTEXTE DE L'ENQUÊTE:
{contexte_enquete if contexte_enquete else "Pas de contexte spécifique fourni"}

TEXTE À ANALYSER:
{text_content}

Analyse ce texte et réponds en JSON avec les champs suivants:
- "pertinence": un score de 0 à 100 indiquant la pertinence pour l'enquête
- "categorie": "pertinent" (score ≥55), "a_traiter" (25-54), ou "non_pertinent" (<25)
- "raisons": liste de 2-3 raisons courtes expliquant le score
- "elements_cles": liste de mots-clés ou éléments importants détectés

Réponds UNIQUEMENT avec le JSON, sans texte supplémentaire. [/INST]"""
    
    try:
        # Tokenizer et générer
        tokenizer = text_classifier["tokenizer"]
        model = text_classifier["model"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Générer la réponse (limiter la longueur pour économiser ressources)
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=True,
            top_p=0.9
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire le JSON de la réponse
        # La réponse de Mistral contient le prompt + la réponse
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            
            return {
                "score": result.get("pertinence", 50),
                "category": result.get("categorie", "a_traiter"),
                "reasons": result.get("raisons", ["Analyse automatique"]),
                "keywords": result.get("elements_cles", []),
                "analyzed": True
            }
        else:
            # Si pas de JSON valide, faire une analyse basique
            return fallback_text_analysis(text_content, contexte_enquete)
            
    except Exception as e:
        print(f"Erreur lors de l'analyse avec Mistral: {e}")
        return fallback_text_analysis(text_content, contexte_enquete)

def fallback_text_analysis(text_content: str, contexte_enquete: str) -> dict:
    """
    Analyse de secours si Mistral échoue
    Analyse basique par mots-clés
    """
    text_lower = text_content.lower()
    score = 50  # Score de base
    reasons = []
    keywords = []
    
    # Mots-clés suspects
    suspect_keywords = ["urgent", "caché", "secret", "argent", "rencontre", "lieu", "arme", "danger"]
    for word in suspect_keywords:
        if word in text_lower:
            score += 10
            keywords.append(word)
    
    # Correspondance avec contexte
    if contexte_enquete:
        contexte_words = [w for w in contexte_enquete.lower().split() if len(w) > 4]
        matches = sum(1 for w in contexte_words if w in text_lower)
        if matches > 0:
            score += matches * 8
            reasons.append(f"{matches} correspondance(s) avec contexte")
    
    # Longueur du texte
    if len(text_content) > 100:
        reasons.append("Texte détaillé")
    
    score = min(100, score)
    
    # Classifier
    if score >= 55:
        category = "pertinent"
    elif score >= 25:
        category = "a_traiter"
    else:
        category = "non_pertinent"
    
    if not reasons:
        reasons = ["Analyse automatique basique"]
    
    return {
        "score": score,
        "category": category,
        "reasons": reasons,
        "keywords": keywords,
        "analyzed": True
    }

# ============================================================================
# PAGE 5 : ANALYSE - Espace de travail avec tri pertinence enquête
# ============================================================================

def calculate_investigation_relevance_score(analysis: dict, contexte_enquete: str) -> int:
    """
    Calcule un score de pertinence spécifique pour l'enquête (0-100)
    Système amélioré avec correspondance sémantique flexible
    """
    score = 0
    description = analysis.get("description", "").lower()
    categories = analysis.get("categories", [])
    tags = analysis.get("tags", [])
    
    print(f"\n=== Scoring image: {analysis.get('filename', 'unknown')} ===")
    print(f"Description: {description}")
    print(f"Categories: {categories}")
    print(f"Tags: {tags}")
    
    # Score de base selon catégories (TOUJOURS APPLICABLE)
    base_score = 0
    critical_categories = {
        "people": 25,      # Personnes = très pertinent
        "weapons": 45,     # Armes = extrêmement pertinent
        "vehicles": 22,    # Véhicules = pertinent
        "documents": 28,   # Documents = très pertinent
        "buildings": 18,   # Lieux = pertinent
        "indoor": 15,      # Intérieur = moyennement pertinent
        "outdoor": 15,     # Extérieur = moyennement pertinent
        "objects": 12      # Objets = pertinent
    }
    
    for cat, points in critical_categories.items():
        if cat in categories:
            base_score += points
            print(f"  Category '{cat}': +{points} points")
    
    score += base_score
    
    # Bonus catégories multiples (image riche)
    if len(categories) >= 3:
        bonus = 15
        score += bonus
        print(f"  Multiple categories bonus: +{bonus} points")
    
    # Tags importants
    tag_score = 0
    important_tags = ["people", "vehicles", "documents", "weapons", "buildings"]
    for tag in tags:
        if tag in important_tags:
            tag_score += 8
    if tag_score > 0:
        score += tag_score
        print(f"  Important tags: +{tag_score} points")
    
    # Description détaillée
    word_count = len(description.split())
    if word_count > 10:
        score += 12
        print(f"  Detailed description: +12 points")
    elif word_count > 6:
        score += 6
        print(f"  Medium description: +6 points")
    
    # SI CONTEXTE FOURNI : Analyse sémantique approfondie
    if contexte_enquete and len(contexte_enquete.strip()) > 10:
        contexte_lower = contexte_enquete.lower()
        
        # Nettoyer et extraire mots significatifs du contexte
        stop_words = {"le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "dans", "sur", "avec", "pour", "par"}
        contexte_words = [w for w in contexte_lower.split() if len(w) > 3 and w not in stop_words]
        
        print(f"  Context words to match: {contexte_words[:20]}")
        
        # 1. Correspondance exacte des mots-clés (POIDS TRÈS FORT)
        exact_matches = 0
        for word in contexte_words[:20]:  # Top 20 mots du contexte
            if word in description:
                exact_matches += 1
                score += 12  # +12 points par correspondance exacte
        
        if exact_matches > 0:
            print(f"  Exact word matches: {exact_matches} words → +{exact_matches * 12} points")
        
        # 2. Correspondance partielle (mots racines, préfixes)
        partial_matches = 0
        for word in contexte_words[:20]:
            # Vérifier les correspondances partielles (au moins 4 caractères communs)
            if len(word) >= 4:
                for desc_word in description.split():
                    if len(desc_word) >= 4:
                        # Correspondance de début de mot (préfixe commun)
                        if word[:4] in desc_word or desc_word[:4] in word:
                            partial_matches += 1
                            score += 6  # +6 points par correspondance partielle
                            break
        
        if partial_matches > 0:
            print(f"  Partial matches: {partial_matches} → +{partial_matches * 6} points")
        
        # 3. Correspondance sémantique via catégories mentionnées dans contexte
        semantic_bonus = 0
        category_keywords = {
            "people": ["personne", "homme", "femme", "suspect", "témoin", "individu", "gens"],
            "vehicles": ["voiture", "véhicule", "auto", "moto", "camion", "transport"],
            "weapons": ["arme", "pistolet", "couteau", "fusil", "dangereux"],
            "documents": ["document", "papier", "texte", "écrit", "lettre", "note"],
            "buildings": ["bâtiment", "maison", "immeuble", "structure", "lieu"],
            "outdoor": ["extérieur", "dehors", "rue", "route", "parc"],
            "indoor": ["intérieur", "dedans", "pièce", "salle", "chambre"]
        }
        
        for cat, keywords in category_keywords.items():
            if cat in categories:
                for keyword in keywords:
                    if keyword in contexte_lower:
                        semantic_bonus += 15
                        print(f"  Semantic match '{cat}' via '{keyword}': +15 points")
                        break
        
        score += semantic_bonus
        
        # 4. Bonus si beaucoup de correspondances (contexte très pertinent)
        if exact_matches >= 3:
            high_relevance_bonus = 20
            score += high_relevance_bonus
            print(f"  High relevance bonus: +{high_relevance_bonus} points")
        elif exact_matches >= 2:
            score += 10
            print(f"  Medium relevance bonus: +10 points")
    else:
        print("  No context provided, using base scoring only")
    
    # Normaliser entre 0 et 100
    final_score = min(100, max(0, score))
    print(f"  FINAL SCORE: {final_score}/100\n")
    
    return final_score

def classify_by_investigation_relevance(score: int) -> str:
    """
    Classifie une image selon son score de pertinence
    Seuils ajustés pour le nouveau système de scoring :
    Pertinent: score >= 55
    À traiter: 25 <= score < 55
    Non pertinent: score < 25
    """
    if score >= 55:
        return "pertinent"
    elif score >= 25:
        return "a_traiter"
    else:
        return "non_pertinent"

def page_analyse_sort_all(current_state):
    """
    Trie toutes les images selon leur pertinence pour l'enquête
    """
    if current_state is None or len(current_state.images) == 0:
        return """
        <div class="info-message">
            ℹ️ Aucune image n'a été importée. Commencez par la page <strong>Accueil</strong> pour uploader des images.
        </div>
        """, current_state, {}
    
    # S'assurer que les images sont analysées
    if len(current_state.analyses) < len(current_state.images):
        current_state = analyze_all_images(current_state)
    
    # Calculer le score de pertinence pour chaque image
    contexte = current_state.enquete_info.get("contexte", "")
    relevance_counts = {"pertinent": 0, "a_traiter": 0, "non_pertinent": 0}
    
    for idx, analysis in current_state.analyses.items():
        # Calculer le score de pertinence
        relevance_score = calculate_investigation_relevance_score(analysis, contexte)
        relevance_category = classify_by_investigation_relevance(relevance_score)
        
        # Stocker dans l'analyse
        analysis["relevance_score"] = relevance_score
        analysis["relevance_category"] = relevance_category
        
        relevance_counts[relevance_category] += 1
        
        print(f"Image {idx}: score={relevance_score}, category={relevance_category}")
    
    status_msg = f"""
    <div class="success-message">
        ✅ {len(current_state.images)} image(s) triée(s) par pertinence !<br>
        🟢 Pertinentes: {relevance_counts['pertinent']} | 🟡 À traiter: {relevance_counts['a_traiter']} | 🔴 Non pertinentes: {relevance_counts['non_pertinent']}
    </div>
    """
    
    return status_msg, current_state, relevance_counts

def page_analyse_filter_by_relevance(relevance_category: str, current_state):
    """
    Filtre et affiche les images d'une catégorie de pertinence
    """
    if current_state is None or len(current_state.analyses) == 0:
        return """
        <div class="info-message">
            ⚠️ Aucune image analysée. Cliquez d'abord sur "Trier les images".
        </div>
        """
    
    # Vérifier si les images ont été triées
    has_relevance = any("relevance_category" in analysis for analysis in current_state.analyses.values())
    if not has_relevance:
        return """
        <div class="info-message">
            ⚠️ Les images n'ont pas encore été triées. Cliquez sur "Trier les images".
        </div>
        """
    
    # Filtrer par catégorie
    filtered_images = []
    for analysis in current_state.analyses.values():
        if analysis.get("relevance_category") == relevance_category:
            filtered_images.append(analysis)
    
    # Trier par score décroissant
    filtered_images.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # Labels et couleurs
    category_info = {
        "pertinent": {
            "label": "Pertinentes",
            "icon": "🟢",
            "color": "#28a745",
            "description": "Images hautement pertinentes pour l'enquête"
        },
        "a_traiter": {
            "label": "À traiter",
            "icon": "🟡",
            "color": "#ffc107",
            "description": "Images nécessitant une analyse manuelle approfondie"
        },
        "non_pertinent": {
            "label": "Non pertinentes",
            "icon": "🔴",
            "color": "#dc3545",
            "description": "Images probablement non pertinentes pour l'enquête"
        }
    }
    
    cat_info = category_info.get(relevance_category, {})
    
    if len(filtered_images) == 0:
        return f"""
        <div class="info-message">
            ℹ️ Aucune image dans la catégorie "<strong>{cat_info.get('label', 'Inconnue')}</strong>".
        </div>
        """
    
    # Générer l'affichage HTML
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif;">
        <div style="background: {cat_info['color']}20; border-left: 5px solid {cat_info['color']}; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h2 style="margin: 0; color: {cat_info['color']};">
                {cat_info['icon']} {cat_info['label']}
            </h2>
            <p style="margin: 5px 0 0 0; color: #666;">
                {len(filtered_images)} image(s) - {cat_info['description']}
            </p>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px;">
    """
    
    for analysis in filtered_images:
        score = analysis.get("relevance_score", 0)
        
        # Barre de progression du score
        progress_color = cat_info['color']
        
        # Badges de catégories
        categories_badges = ""
        if "categories" in analysis:
            for cat in analysis.get("categories", [])[:4]:  # Max 4 catégories affichées
                if cat in CATEGORIES_POLICE:
                    cat_data = CATEGORIES_POLICE[cat]
                    label_display = cat_data.get('label_fr', cat_data['label'])
                    categories_badges += f"""
                    <span style="background: {cat_data['color']}; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.8em; margin: 2px; display: inline-block;">
                        {cat_data['icon']} {label_display}
                    </span>
                    """
        
        html += f"""
        <div style="background: white; border: 3px solid {cat_info['color']}; border-radius: 12px; padding: 18px; box-shadow: 0 3px 6px rgba(0,0,0,0.15); transition: transform 0.2s;"
             onmouseover="this.style.transform='translateY(-5px)'" onmouseout="this.style.transform='translateY(0)'">
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h4 style="margin: 0; color: var(--primary-blue); font-size: 1.05em;">
                    📄 {analysis['filename']}
                </h4>
                <div style="background: {progress_color}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 1.1em;">
                    {score}
                </div>
            </div>
            
            <div style="background: #f8f9fa; padding: 10px; border-radius: 6px; margin: 10px 0;">
                <p style="margin: 0; color: #555; line-height: 1.6; font-size: 0.95em;">
                    <strong>📝 Description :</strong><br>
                    {analysis.get('description', 'Non disponible')}
                </p>
            </div>
            
            <div style="margin: 10px 0;">
                <p style="margin: 0 0 6px 0; font-weight: 600; color: var(--primary-blue); font-size: 0.9em;">🏷️ Catégories :</p>
                <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                    {categories_badges if categories_badges else '<span style="color: #999; font-style: italic; font-size: 0.9em;">Aucune</span>'}
                </div>
            </div>
            
            <div style="margin-top: 12px; padding-top: 10px; border-top: 2px solid #eee;">
                <div style="background: #e9ecef; border-radius: 10px; overflow: hidden; height: 8px;">
                    <div style="background: {progress_color}; height: 100%; width: {score}%; transition: width 0.3s;"></div>
                </div>
                <p style="margin: 5px 0 0 0; text-align: center; font-size: 0.85em; color: #666;">
                    Score de pertinence : {score}/100
                </p>
            </div>
        </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    return html

# ============================================================================
# PAGE 4 : CONNECTIONS - Associations entre images
# ============================================================================

def calculate_image_similarity(img1_analysis: dict, img2_analysis: dict) -> float:
    """
    Calcule un score de similarité entre deux images (0-100)
    Basé sur catégories, description, tags
    """
    similarity_score = 0
    
    # 1. Catégories communes (POIDS FORT)
    cat1 = set(img1_analysis.get("categories", []))
    cat2 = set(img2_analysis.get("categories", []))
    common_categories = cat1.intersection(cat2)
    
    if common_categories:
        # +15 points par catégorie commune
        similarity_score += len(common_categories) * 15
    
    # 2. Tags communs (POIDS MOYEN)
    tags1 = set(img1_analysis.get("tags", []))
    tags2 = set(img2_analysis.get("tags", []))
    common_tags = tags1.intersection(tags2)
    
    if common_tags:
        similarity_score += len(common_tags) * 10
    
    # 3. Mots communs dans les descriptions (POIDS MOYEN)
    desc1 = img1_analysis.get("description", "").lower().split()
    desc2 = img2_analysis.get("description", "").lower().split()
    
    # Mots significatifs (>3 lettres)
    words1 = set([w for w in desc1 if len(w) > 3])
    words2 = set([w for w in desc2 if len(w) > 3])
    common_words = words1.intersection(words2)
    
    if common_words:
        similarity_score += len(common_words) * 5
    
    # 4. Scores de pertinence similaires (POIDS FAIBLE)
    score1 = img1_analysis.get("relevance_score", 50)
    score2 = img2_analysis.get("relevance_score", 50)
    score_diff = abs(score1 - score2)
    
    if score_diff < 10:
        similarity_score += 10
    elif score_diff < 20:
        similarity_score += 5
    
    return min(100, similarity_score)

def find_automatic_associations(current_state, min_similarity=30):
    """
    Trouve automatiquement les associations entre images
    Retourne une liste de connexions avec scores de similarité
    """
    if current_state is None or len(current_state.analyses) < 2:
        return []
    
    associations = []
    analyses_list = list(current_state.analyses.items())
    
    # Comparer chaque paire d'images
    for i in range(len(analyses_list)):
        for j in range(i + 1, len(analyses_list)):
            img1_id, img1_analysis = analyses_list[i]
            img2_id, img2_analysis = analyses_list[j]
            
            similarity = calculate_image_similarity(img1_analysis, img2_analysis)
            
            if similarity >= min_similarity:
                # Déterminer le type de connexion
                common_cats = set(img1_analysis.get("categories", [])).intersection(
                    set(img2_analysis.get("categories", []))
                )
                
                if common_cats:
                    connection_type = f"Catégories communes: {', '.join([CATEGORIES_POLICE.get(c, {}).get('label_fr', c) for c in list(common_cats)[:3]])}"
                else:
                    connection_type = "Éléments similaires"
                
                associations.append({
                    "img1_id": img1_id,
                    "img2_id": img2_id,
                    "img1_name": img1_analysis.get("filename", f"Image {img1_id}"),
                    "img2_name": img2_analysis.get("filename", f"Image {img2_id}"),
                    "similarity": similarity,
                    "type": connection_type,
                    "auto": True
                })
    
    # Trier par similarité décroissante
    associations.sort(key=lambda x: x["similarity"], reverse=True)
    
    return associations

def generate_associations_graph_html(associations: list) -> str:
    """
    Génère une visualisation HTML/CSS des associations sous forme de graphe
    """
    if not associations:
        return """
        <div class="info-message">
            ℹ️ Aucune association trouvée. Les images doivent partager des catégories, tags ou descriptions similaires.
        </div>
        """
    
    # Créer un graphe visuel en HTML
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif;">
        <div style="background: var(--light-blue); padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h2 style="margin: 0; color: var(--primary-blue);">
                🕸️ {len(associations)} Association(s) Détectée(s)
            </h2>
            <p style="margin: 5px 0 0 0; color: #666;">
                Connexions automatiques basées sur les similarités entre images
            </p>
        </div>
        
        <div style="display: flex; flex-direction: column; gap: 15px;">
    """
    
    for idx, assoc in enumerate(associations):
        similarity = assoc["similarity"]
        
        # Couleur selon force de la connexion
        if similarity >= 70:
            color = "#28a745"
            strength = "Forte"
        elif similarity >= 50:
            color = "#ffc107"
            strength = "Moyenne"
        else:
            color = "#17a2b8"
            strength = "Faible"
        
        html += f"""
        <div style="background: white; border: 2px solid {color}; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">
                <div style="flex: 1;">
                    <h4 style="margin: 0; color: var(--primary-blue);">Association #{idx + 1}</h4>
                    <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9em;">
                        {assoc['type']}
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="background: {color}; color: white; padding: 8px 16px; border-radius: 20px; font-weight: 700;">
                        {similarity}%
                    </div>
                    <p style="margin: 5px 0 0 0; font-size: 0.85em; color: #666;">Similarité {strength.lower()}</p>
                </div>
            </div>
            
            <div style="display: flex; align-items: center; gap: 20px;">
                <div style="flex: 1; background: #f8f9fa; padding: 15px; border-radius: 8px;">
                    <p style="margin: 0; font-weight: 600; color: var(--primary-blue);">📄 {assoc['img1_name']}</p>
                </div>
                
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 40px; height: 2px; background: {color};"></div>
                    <div style="background: {color}; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700;">
                        ↔
                    </div>
                    <div style="width: 40px; height: 2px; background: {color};"></div>
                </div>
                
                <div style="flex: 1; background: #f8f9fa; padding: 15px; border-radius: 8px;">
                    <p style="margin: 0; font-weight: 600; color: var(--primary-blue);">📄 {assoc['img2_name']}</p>
                </div>
            </div>
            
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee;">
                <div style="background: #e9ecef; border-radius: 10px; overflow: hidden; height: 8px;">
                    <div style="background: {color}; height: 100%; width: {similarity}%; transition: width 0.3s;"></div>
                </div>
            </div>
        </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    return html

def page_connections_auto(current_state):
    """
    Génère automatiquement les associations entre images
    """
    if current_state is None or len(current_state.images) == 0:
        return """
        <div class="info-message">
            ℹ️ Aucune image n'a été importée. Commencez par la page <strong>Accueil</strong> pour uploader des images.
        </div>
        """, current_state
    
    # S'assurer que les images sont analysées et catégorisées
    if len(current_state.analyses) < len(current_state.images):
        current_state = analyze_all_images(current_state)
    
    # Trouver les associations automatiques
    associations = find_automatic_associations(current_state, min_similarity=30)
    
    # Stocker dans l'état
    if not hasattr(current_state, 'associations'):
        current_state.associations = {"auto": [], "manual": []}
    
    current_state.associations["auto"] = associations
    
    # Générer la visualisation
    graph_html = generate_associations_graph_html(associations)
    
    return graph_html, current_state

def page_connections_manual_interface(current_state):
    """
    Interface pour créer des associations manuelles
    """
    if current_state is None or len(current_state.images) == 0:
        return """
        <div class="info-message">
            ℹ️ Aucune image n'a été importée.
        </div>
        """
    
    # Liste des images disponibles
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif;">
        <div style="background: var(--accent-blue); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h2 style="margin: 0;">🔗 Créer des Associations Manuelles</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">
                Fonctionnalité interactive à venir : Sélectionner deux images et créer un lien personnalisé
            </p>
        </div>
        
        <div style="background: white; border: 2px solid var(--border-gray); border-radius: 10px; padding: 25px;">
            <h3 style="color: var(--primary-blue); margin-top: 0;">📋 Images disponibles ({len(current_state.images)})</h3>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; margin-top: 20px;">
    """
    
    for idx, img_data in enumerate(current_state.images):
        analysis = current_state.analyses.get(idx, {})
        filename = analysis.get("filename", img_data.get("filename", f"Image {idx}"))
        categories = analysis.get("categories", [])
        
        # Badges de catégories
        cat_badges = ""
        for cat in categories[:3]:
            if cat in CATEGORIES_POLICE:
                cat_info = CATEGORIES_POLICE[cat]
                cat_badges += f"""
                <span style="background: {cat_info['color']}; color: white; padding: 2px 6px; border-radius: 8px; font-size: 0.75em; margin: 2px;">
                    {cat_info['icon']}
                </span>
                """
        
        html += f"""
        <div style="background: #f8f9fa; border: 2px solid var(--border-gray); border-radius: 8px; padding: 15px; cursor: pointer; transition: all 0.2s;"
             onmouseover="this.style.borderColor='var(--accent-blue)'; this.style.background='#e8f1f8';"
             onmouseout="this.style.borderColor='var(--border-gray)'; this.style.background='#f8f9fa';">
            <div style="background: var(--primary-blue); color: white; width: 35px; height: 35px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; margin-bottom: 10px;">
                {idx + 1}
            </div>
            <p style="margin: 0; font-weight: 600; color: var(--dark-text); font-size: 0.9em; word-break: break-word;">
                {filename[:30]}{'...' if len(filename) > 30 else ''}
            </p>
            <div style="margin-top: 8px;">
                {cat_badges if cat_badges else '<span style="color: #999; font-size: 0.8em;">Aucune catégorie</span>'}
            </div>
        </div>
        """
    
    html += """
            </div>
            
            <div style="margin-top: 30px; padding: 20px; background: var(--light-blue); border-radius: 8px;">
                <h4 style="margin: 0 0 10px 0; color: var(--primary-blue);">💡 Fonctionnalité à venir</h4>
                <p style="margin: 0; color: #666; line-height: 1.6;">
                    • Sélectionner deux images en cliquant dessus<br>
                    • Définir le type de relation (similaire, lié, cause-effet, etc.)<br>
                    • Ajouter une note descriptive<br>
                    • Visualiser le réseau de connexions créé
                </p>
            </div>
        </div>
    </div>
    """
    
    return html

# ============================================================================
# INTERFACE GRADIO PRINCIPALE
# ============================================================================

with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS, title="IArgos - Système d'Analyse d'Enquêtes") as demo:
    
    # En-tête principal
    gr.HTML("""
        <div class="main-header">
            <h1>🛡️ IArgos</h1>
            <p>Système Intelligent d'Analyse et de Catégorisation de Données d'Enquête</p>
        </div>
    """)
    
    # État global de l'application (partagé entre toutes les pages)
    enquete_state = gr.State(value=None)
    
    # Navigation par onglets
    with gr.Tabs() as tabs:
        
        # ====================================================================
        # PAGE 1 : ACCUEIL
        # ====================================================================
        with gr.Tab("🏠 Accueil", id="accueil"):
            gr.Markdown("""
            ## Bienvenue sur IArgos
            
            Commencez votre enquête en important les images à analyser et en décrivant le contexte de l'affaire.
            """)
            
            with gr.Row():
                # Colonne gauche : Upload d'images
                with gr.Column(scale=2):
                    gr.HTML('<div class="section-title">📁 Import des images</div>')
                    
                    image_upload = gr.File(
                        label="Déposer les images de l'enquête",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                        elem_classes=["upload-zone"]
                    )
                    
                    upload_btn = gr.Button(
                        "📤 Charger les images",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary"]
                    )
                    
                    upload_status = gr.HTML(value="")
                
                # Colonne droite : Statistiques
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">📊 Vue d\'ensemble</div>')
                    stats_display = gr.HTML(value=generate_stats_html(EnqueteData()))
            
            gr.Markdown("---")
            
            # Section : Contexte de l'enquête
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="section-title">📝 Contexte de l\'enquête</div>')
                    
                    gr.Markdown("""
                    Décrivez le contexte général de l'enquête. Ces informations aideront l'IA à mieux comprendre
                    et catégoriser les données analysées.
                    """)
                    
                    enquete_titre = gr.Textbox(
                        label="Titre / Référence de l'enquête",
                        placeholder="Ex: Enquête 2024-INV-0123 - Vol avec effraction",
                        lines=1
                    )
                    
                    enquete_contexte = gr.Textbox(
                        label="Description et contexte général",
                        placeholder="""Décrivez ici les détails pertinents de l'enquête :
- Nature de l'affaire
- Lieux concernés
- Personnes impliquées
- Éléments recherchés
- Toute autre information contextuelle importante
                        """,
                        lines=10
                    )
                    
                    save_context_btn = gr.Button(
                        "💾 Enregistrer le contexte",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary"]
                    )
                    
                    context_status = gr.HTML(value="")
            
            gr.Markdown("---")
            
            gr.HTML("""
                <div class="info-message">
                    ℹ️ <strong>Prochaines étapes :</strong> Une fois les images chargées et le contexte défini,
                    utilisez les onglets ci-dessus pour accéder à la recherche, la catégorisation et l'analyse des données.
                </div>
            """)
            
            # Événements Page 1
            upload_btn.click(
                fn=page_accueil_init_images,
                inputs=[image_upload, enquete_state],
                outputs=[upload_status, enquete_state, stats_display]
            )
            
            save_context_btn.click(
                fn=page_accueil_save_context,
                inputs=[enquete_titre, enquete_contexte, enquete_state],
                outputs=[context_status, enquete_state]
            )
        
        # ====================================================================
        # PAGE 2 : RECHERCHE - Fonctionnelle
        # ====================================================================
        with gr.Tab("🔍 Recherche", id="recherche"):
            gr.HTML('<div class="section-title">🔍 Recherche textuelle dans les images</div>')
            
            gr.Markdown("""
            ## Fonctionnement
            
            Cette page vous permet de rechercher des images en fonction de leur contenu textuel :
            - Recherche dans les **descriptions** générées par l'IA
            - Recherche dans les **tags** extraits automatiquement
            - **Score de pertinence** basé sur le contexte de votre enquête
            
            Les images sont d'abord analysées par BLIP pour générer des descriptions et tags.
            """)
            
            # Bouton pour lancer l'analyse
            analyze_status = gr.HTML(value="")
            
            with gr.Row():
                analyze_images_btn = gr.Button(
                    "🤖 Analyser les images avec IA",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary"]
                )
            
            gr.Markdown("---")
            
            # Zone de recherche
            with gr.Row():
                with gr.Column(scale=4):
                    search_query = gr.Textbox(
                        label="Entrez votre recherche (en français)",
                        placeholder="Ex: personne, voiture, document, animal, bâtiment, arme...",
                        lines=1,
                        elem_id="search_box"
                    )
                with gr.Column(scale=1):
                    search_btn = gr.Button(
                        "🔍 Rechercher",
                        variant="secondary",
                        size="lg"
                    )
            
            # Résultats de recherche
            search_results = gr.HTML(value="")
            
            # Événements Page 2
            analyze_images_btn.click(
                fn=page_recherche_analyze_if_needed,
                inputs=[enquete_state],
                outputs=[analyze_status, enquete_state]
            )
            
            search_btn.click(
                fn=page_recherche_search,
                inputs=[search_query, enquete_state],
                outputs=[search_results]
            )
            
            search_query.submit(
                fn=page_recherche_search,
                inputs=[search_query, enquete_state],
                outputs=[search_results]
            )
        
        # ====================================================================
        # PAGE 3 : CATÉGORISATION - Fonctionnelle
        # ====================================================================
        with gr.Tab("🗂️ Catégorisation", id="categorisation"):
            gr.HTML('<div class="section-title">🗂️ Catégorisation automatique des images</div>')
            
            gr.Markdown("""
            ## Fonctionnement
            
            Cette page classe automatiquement vos images dans des catégories pertinentes pour l'enquête :
            - **👤 Personnes** : Suspects, témoins, visages
            - **🚗 Véhicules** : Voitures, motos, plaques
            - **⚠️ Armes/Suspects** : Armes, objets dangereux (seuil strict)
            - **📄 Documents/Textes** : Papiers, textes, preuves écrites
            - **🏢 Bâtiments/Lieux** : Bâtiments, scènes de crime
            - **🌳 Extérieur** : Scènes extérieures, rues, nature
            - **🏠 Intérieur** : Scènes intérieures, pièces
            - **📦 Objets** : Preuves matérielles, équipements
            - **🐾 Animaux** : Chiens, chats, animaux domestiques
            
            Les images sont analysées par l'IA BLIP avec plusieurs questions par catégorie pour plus de précision.
            """)
            
            # Bouton pour lancer la catégorisation
            categorize_status = gr.HTML(value="")
            
            with gr.Row():
                categorize_btn = gr.Button(
                    "🤖 Catégoriser toutes les images",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary"]
                )
            
            gr.Markdown("---")
            
            # Layout avec sidebar et zone d'affichage
            with gr.Row():
                # Sidebar gauche - Liste des catégories
                with gr.Column(scale=1):
                    gr.HTML('<h3 style="color: var(--primary-blue); margin-top: 0;">📂 Catégories</h3>')
                    
                    categories_sidebar = gr.HTML(value="")
                    
                    # Boutons pour chaque catégorie
                    cat_buttons = {}
                    for cat_id, cat_info in CATEGORIES_POLICE.items():
                        label_display = cat_info.get('label_fr', cat_info['label'])
                        cat_buttons[cat_id] = gr.Button(
                            f"{cat_info['icon']} {label_display}",
                            variant="secondary",
                            size="sm",
                            elem_id=f"cat_{cat_id}"
                        )
                    
                    # Bouton pour afficher toutes les images
                    show_all_btn = gr.Button(
                        "📋 Toutes les images",
                        variant="secondary",
                        size="sm"
                    )
                
                # Zone principale droite - Affichage des images filtrées
                with gr.Column(scale=3):
                    images_display = gr.HTML(value="""
                        <div class="info-message">
                            ℹ️ Cliquez sur "Catégoriser toutes les images" pour commencer, puis sélectionnez une catégorie à gauche.
                        </div>
                    """)
            
            # Événements Page 3
            categorize_btn.click(
                fn=page_categorisation_analyze,
                inputs=[enquete_state],
                outputs=[categorize_status, enquete_state, categories_sidebar, gr.State()]
            )
            
            # Événements pour chaque bouton de catégorie
            for cat_id, btn in cat_buttons.items():
                btn.click(
                    fn=lambda state, cid=cat_id: page_categorisation_filter(cid, state),
                    inputs=[enquete_state],
                    outputs=[images_display]
                )
            
            # Afficher toutes les images
            show_all_btn.click(
                fn=lambda state: page_categorisation_filter("all", state),
                inputs=[enquete_state],
                outputs=[images_display]
            )
        
        # ====================================================================
        # PAGE 4 : CONNECTIONS - Fonctionnelle
        # ====================================================================
        with gr.Tab("🕸️ Connections", id="connections"):
            gr.HTML('<div class="section-title">🕸️ Graphes et Réseaux de Relations</div>')
            
            gr.Markdown("""
            ## Analyse des connexions entre images
            
            Cette page vous permet de découvrir et créer des associations entre les images de l'enquête :
            
            - **🤖 Associations automatiques** : L'IA détecte les similarités entre images (catégories communes, descriptions similaires, tags partagés)
            - **🔗 Créer des associations** : Créez manuellement des liens personnalisés entre images (à venir)
            
            Les associations permettent de visualiser les relations et connexions entre différentes preuves visuelles.
            """)
            
            # Deux boutons principaux
            with gr.Row():
                with gr.Column(scale=1):
                    auto_associations_btn = gr.Button(
                        "🤖 Associations\n(Automatiques par IA)",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary"]
                    )
                
                with gr.Column(scale=1):
                    manual_associations_btn = gr.Button(
                        "🔗 Créer des Associations\n(Manuelles)",
                        variant="secondary",
                        size="lg"
                    )
            
            gr.Markdown("---")
            
            # Zone d'affichage des connexions
            connections_display = gr.HTML(value="""
                <div class="info-message">
                    ℹ️ Cliquez sur un des boutons ci-dessus pour commencer.<br><br>
                    <strong>🤖 Associations automatiques :</strong> L'IA analysera toutes les images et trouvera les similarités<br>
                    <strong>🔗 Créer des associations :</strong> Interface pour créer vos propres liens entre images
                </div>
            """)
            
            gr.Markdown("""
            ---
            ### 🎯 Comment fonctionnent les associations automatiques ?
            
            L'IA calcule un **score de similarité** entre chaque paire d'images basé sur :
            
            1. **Catégories communes** (Poids fort) : +15 points par catégorie partagée
            2. **Tags communs** (Poids moyen) : +10 points par tag partagé
            3. **Mots similaires dans descriptions** (Poids moyen) : +5 points par mot commun
            4. **Scores de pertinence proches** (Poids faible) : +5-10 points
            
            **Seuils de connexion** :
            - 🟢 **Forte** (≥70%) : Similarité très élevée
            - 🟡 **Moyenne** (50-69%) : Similarité notable
            - 🔵 **Faible** (30-49%) : Similarité détectable
            
            Seules les associations avec un score ≥ 30% sont affichées.
            """)
            
            # Événements Page 4
            auto_associations_btn.click(
                fn=page_connections_auto,
                inputs=[enquete_state],
                outputs=[connections_display, enquete_state]
            )
            
            manual_associations_btn.click(
                fn=page_connections_manual_interface,
                inputs=[enquete_state],
                outputs=[connections_display]
            )
        
        # ====================================================================
        # PAGE 5 : ANALYSE - Fonctionnelle
        # ====================================================================
        with gr.Tab("📊 Analyse", id="analyse"):
            gr.HTML('<div class="section-title">📊 Espace d\'Analyse et Tri par Pertinence</div>')
            
            gr.Markdown("""
            ## Espace de travail de l'enquêteur
            
            Cette page vous permet de trier toutes les images selon leur **pertinence pour l'enquête** en fonction du contexte que vous avez défini.
            
            ### 🎯 Système de scoring de pertinence :
            - **Score basé sur le contexte** de l'enquête (correspondance sémantique avancée)
            - **Analyse multi-critères** : catégories, description, correspondances exactes et partielles
            - **3 niveaux de pertinence** :
              - 🟢 **Pertinentes** (score ≥ 55) : Images hautement pertinentes
              - 🟡 **À traiter** (25-54) : Images nécessitant une analyse approfondie
              - 🔴 **Non pertinentes** (< 25) : Images probablement sans intérêt
            
            Cliquez sur "Trier les images" puis sur une catégorie pour voir les images correspondantes.
            """)
            
            # Bouton pour lancer le tri
            sort_status = gr.HTML(value="")
            
            with gr.Row():
                sort_btn = gr.Button(
                    "🎯 Trier les images par pertinence",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary"]
                )
            
            gr.Markdown("---")
            
            # Section des 3 containers de pertinence
            with gr.Row():
                # Container Pertinentes
                with gr.Column(scale=1):
                    pertinent_btn = gr.Button(
                        "🟢 Pertinentes\n(score ≥ 55)",
                        variant="secondary",
                        size="lg",
                        elem_id="btn_pertinent"
                    )
                
                # Container À traiter
                with gr.Column(scale=1):
                    a_traiter_btn = gr.Button(
                        "🟡 À traiter\n(score 25-54)",
                        variant="secondary",
                        size="lg",
                        elem_id="btn_a_traiter"
                    )
                
                # Container Non pertinentes
                with gr.Column(scale=1):
                    non_pertinent_btn = gr.Button(
                        "🔴 Non pertinentes\n(score < 25)",
                        variant="secondary",
                        size="lg",
                        elem_id="btn_non_pertinent"
                    )
            
            gr.Markdown("---")
            
            # Zone d'affichage des images
            analyse_display = gr.HTML(value="""
                <div class="info-message">
                    ℹ️ Cliquez sur "Trier les images" pour commencer, puis sélectionnez une catégorie ci-dessus.
                </div>
            """)
            
            gr.Markdown("""
            ---
            ### 💡 Conseils d'utilisation
            
            1. **Contexte important** : Plus votre description de l'enquête (Page Accueil) est détaillée, plus le tri sera précis
            2. **Concentrez-vous sur "À traiter"** : Ces images nécessitent votre expertise pour déterminer leur pertinence
            3. **Les catégories sont dynamiques** : Le scoring prend en compte les éléments détectés par l'IA
            4. **Score visible** : Chaque image affiche son score de pertinence sur 100
            """)
            
            # Événements Page 5
            sort_btn.click(
                fn=page_analyse_sort_all,
                inputs=[enquete_state],
                outputs=[sort_status, enquete_state, gr.State()]
            )
            
            pertinent_btn.click(
                fn=lambda state: page_analyse_filter_by_relevance("pertinent", state),
                inputs=[enquete_state],
                outputs=[analyse_display]
            )
            
            a_traiter_btn.click(
                fn=lambda state: page_analyse_filter_by_relevance("a_traiter", state),
                inputs=[enquete_state],
                outputs=[analyse_display]
            )
            
            non_pertinent_btn.click(
                fn=lambda state: page_analyse_filter_by_relevance("non_pertinent", state),
                inputs=[enquete_state],
                outputs=[analyse_display]
            )
            
            gr.Markdown("---")
            gr.Markdown("---")
            
            # ====================================================================
            # SECTION ANALYSE DE TEXTE avec Mistral 7B
            # ====================================================================
            
            gr.HTML('<div class="section-title">📝 Analyse de Texte avec IA</div>')
            
            gr.Markdown("""
            ## Analyser des Textes (Emails, SMS, Notes)
            
            Utilisez **Mistral 7B Instruct** pour analyser et classifier des contenus textuels dans le cadre de l'enquête.
            Le modèle IA analysera le texte et le classera automatiquement en fonction du contexte de votre enquête.
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="📧 Texte à analyser (Email, SMS, Note, etc.)",
                        placeholder="""Collez ici le texte à analyser:
- Email suspect
- SMS échangé
- Note manuscrite
- Transcription d'appel
- Tout autre contenu textuel

Le texte sera analysé par Mistral 7B qui déterminera sa pertinence pour l'enquête.""",
                        lines=8
                    )
                    
                    analyze_text_btn = gr.Button(
                        "🤖 Analyser le texte avec Mistral 7B",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary"]
                    )
                
                with gr.Column(scale=1):
                    text_result = gr.HTML(value="""
                        <div class="info-message">
                            ℹ️ Entrez un texte et cliquez sur "Analyser" pour obtenir une classification automatique.
                        </div>
                    """)
            
            gr.Markdown("""
            ### 🧠 Fonctionnement de l'Analyse Textuelle
            
            Mistral 7B analyse le texte selon plusieurs critères :
            - **Correspondance avec le contexte** de l'enquête
            - **Mots-clés suspects** ou importants
            - **Ton et urgence** du message
            - **Éléments factuels** (lieux, dates, noms)
            
            Le texte est ensuite classé automatiquement :
            - 🟢 **Pertinent** (≥55/100) : Contenu important pour l'enquête
            - 🟡 **À traiter** (25-54/100) : Nécessite analyse manuelle
            - 🔴 **Non pertinent** (<25/100) : Probablement sans intérêt
            """)
            
            # Fonction d'interface pour analyser le texte
            def analyze_text_interface(text, current_state):
                if not text or len(text.strip()) < 10:
                    return """
                    <div class="info-message">
                        ⚠️ Veuillez entrer au moins 10 caractères de texte.
                    </div>
                    """
                
                if current_state is None:
                    state = EnqueteData()
                else:
                    state = current_state
                
                contexte = state.enquete_info.get("contexte", "")
                
                # Analyser avec Mistral
                result = analyze_text_with_mistral(text, contexte)
                
                # Générer l'affichage
                category_info = {
                    "pertinent": {"color": "#28a745", "icon": "🟢", "label": "Pertinent"},
                    "a_traiter": {"color": "#ffc107", "icon": "🟡", "label": "À traiter"},
                    "non_pertinent": {"color": "#dc3545", "icon": "🔴", "label": "Non pertinent"}
                }
                
                cat = category_info.get(result["category"], category_info["a_traiter"])
                
                reasons_html = "<br>".join([f"• {r}" for r in result["reasons"]])
                keywords_html = " ".join([f'<span style="background: var(--secondary-blue); color: white; padding: 4px 10px; border-radius: 12px; margin: 2px; display: inline-block;">{k}</span>' for k in result["keywords"]]) if result["keywords"] else "Aucun"
                
                html = f"""
                <div style="font-family: 'Segoe UI', Arial, sans-serif;">
                    <div style="background: {cat['color']}20; border-left: 5px solid {cat['color']}; padding: 20px; border-radius: 8px;">
                        <h2 style="margin: 0; color: {cat['color']};">
                            {cat['icon']} {cat['label']}
                        </h2>
                        <div style="margin-top: 15px;">
                            <div style="background: {cat['color']}; color: white; padding: 10px 20px; border-radius: 20px; display: inline-block; font-weight: 700; font-size: 1.3em;">
                                Score: {result['score']}/100
                            </div>
                        </div>
                    </div>
                    
                    <div style="background: white; border: 2px solid var(--border-gray); border-radius: 10px; padding: 20px; margin-top: 20px;">
                        <h4 style="color: var(--primary-blue); margin-top: 0;">💡 Raisons de la Classification</h4>
                        <p style="line-height: 1.8; color: #555;">
                            {reasons_html}
                        </p>
                    </div>
                    
                    <div style="background: white; border: 2px solid var(--border-gray); border-radius: 10px; padding: 20px; margin-top: 15px;">
                        <h4 style="color: var(--primary-blue); margin-top: 0;">🏷️ Mots-Clés Détectés</h4>
                        <div style="margin-top: 10px;">
                            {keywords_html}
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px; padding: 15px; background: #e9ecef; border-radius: 8px;">
                        <div style="background: #fff; border-radius: 10px; overflow: hidden; height: 12px;">
                            <div style="background: {cat['color']}; height: 100%; width: {result['score']}%; transition: width 0.5s;"></div>
                        </div>
                    </div>
                </div>
                """
                
                return html
            
            # Événement analyse texte
            analyze_text_btn.click(
                fn=analyze_text_interface,
                inputs=[text_input, enquete_state],
                outputs=[text_result]
            )
    
    # Pied de page
    gr.Markdown("""
    ---
    ### 🔒 Confidentialité et Sécurité
    
    **IArgos** est conçu pour traiter des données sensibles d'enquête. 
    - Les données restent en mémoire pendant la session uniquement
    - Aucune donnée n'est sauvegardée sur les serveurs
    - Pour un usage en production, déployez cette application en local
    
    ### 🧠 Technologies
    - **Intelligence Artificielle** : 
      - BLIP (Captioning + VQA) pour images
      - Mistral 7B Instruct pour textes
    - **Interface** : Gradio Multi-pages
    - **Version** : 3.0 - Images + Textes
    """)

# Lancement de l'application
if __name__ == "__main__":
    demo.launch(ssr_mode=False)
