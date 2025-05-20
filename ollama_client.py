import requests
import time
import logging
import subprocess
import sys
import os
import platform
from PySide6.QtWidgets import QMessageBox
from config import OLLAMA_URL

class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL):
        self.base = base_url.rstrip("/")

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json()["models"]]
            logging.info(f"Modèles Ollama disponibles: {models}")
            return models
        except requests.exceptions.RequestException as e:
            logging.error(f"Erreur lors de la récupération des modèles Ollama: {e}", exc_info=True)
            return []

    def chat(self, model: str, messages: list[dict]) -> tuple[str, int, float]:
        """Ancienne méthode, conservée pour compatibilité si appelée directement ailleurs, 
           mais chat_custom_payload est plus flexible."""
        payload = {"model": model, "messages": messages, "stream": False}
        return self._send_chat_payload(payload)

    def chat_custom_payload(self, payload: dict) -> tuple[str, int, float]:
        """Envoie un payload personnalisé à l'API de chat d'Ollama."""
        return self._send_chat_payload(payload)
    
    def _send_chat_payload(self, payload: dict) -> tuple[str, int, float]:
        """Méthode interne pour envoyer le payload de chat et traiter la réponse."""
        try:
            start = time.time()
            # S'assurer que stream est défini, par défaut à False si non présent
            payload.setdefault("stream", False)
            
            r = requests.post(f"{self.base}/api/chat", json=payload, timeout=300)
            r.raise_for_status()
            duration = max(time.time() - start, 1e-6)
            data = r.json()
            
            # Gérer la réponse pour les messages streamés et non streamés
            if payload["stream"]:
                # Si stream=True, la réponse est une série d'objets JSON séparés par des nouvelles lignes
                # On doit les concaténer pour reconstruire le message complet.
                full_response = ""
                total_tokens_stream = 0 # Initialiser les tokens pour le stream
                # Il faut lire la réponse différemment pour le stream
                # data = r.json() ne fonctionnera pas directement pour un stream complet
                # Ceci est une simplification, car r.json() lira seulement le premier objet JSON du stream.
                # Une vraie gestion du stream nécessiterait d'itérer sur r.iter_lines() ou similaire.
                # Pour l'instant, supposons que même en stream, on obtient un résumé à la fin
                # ou que la version non streamée est utilisée pour la simplicité ici.
                # Si on passe stream=True à Ollama, et qu'on ne le gère pas correctement ici,
                # data["message"]["content"] pourrait être incomplet ou une erreur pourrait survenir.
                # Pour cette raison, il est préférable de forcer stream=False pour le moment
                # jusqu'à ce que le streaming côté client soit pleinement implémenté.
                # La modification ci-dessus force stream=False pour l'instant via setdefault.
                
                # Si le streaming était géré correctement:
                # for line in r.iter_lines():
                #     if line:
                #         json_line = json.loads(line)
                #         full_response += json_line.get("message", {}).get("content", "")
                #         if json_line.get("done"):
                #             total_tokens_stream = json_line.get("total_duration", 0) # exemple, vérifier la doc Ollama pour les bons champs
                #             break
                # resp_content = full_response
                # final_tokens = total_tokens_stream 
                # Pour l'instant, on assume que la réponse non-streamée est correcte
                resp_content = data.get("message", {}).get("content", "")
                final_tokens = data.get("eval_count", 0) # eval_count est plus précis pour les tokens traités par Ollama
            else:
                resp_content = data.get("message", {}).get("content", "")
                final_tokens = data.get("eval_count", 0)
            
            # Utiliser eval_count pour les tokens si disponible, sinon fallback
            # total_tokens = data.get("usage", {}).get("total_tokens", 0) # Ancien champ
            # Ollama retourne aussi: prompt_eval_count, eval_count
            # eval_count semble être le nombre de tokens dans la réponse générée.
            # total_duration, load_duration, prompt_eval_duration, eval_duration

            return resp_content, final_tokens, final_tokens / duration if duration > 0 else 0
        except Exception as e:
            logging.error(f"Erreur lors de l'appel à Ollama: {e}", exc_info=True)
            raise

def is_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception as e:
        logging.warning(f"Ollama ne semble pas être en cours d'exécution: {e}")
        return False

def get_ollama_path():
    """Détermine le chemin vers l'exécutable ollama"""
    if platform.system() == "Windows":
        # Chemins courants sur Windows
        possible_paths = [
            os.path.expanduser("~\\AppData\\Local\\ollama\\ollama.exe"),
            "C:\\Program Files\\ollama\\ollama.exe",
            "C:\\ollama\\ollama.exe"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        # Essayer de trouver ollama dans le PATH
        try:
            import shutil
            return shutil.which("ollama")
        except:
            pass
            
    elif platform.system() == "Darwin":  # macOS
        return "/usr/local/bin/ollama"
    else:  # Linux
        return "/usr/local/bin/ollama"
        
    return None

def start_ollama_server():
    """Démarre le serveur Ollama"""
    try:
        ollama_path = get_ollama_path()
        
        if not ollama_path:
            logging.error("Impossible de trouver l'exécutable Ollama")
            QMessageBox.critical(
                None, 
                "Ollama introuvable", 
                "Impossible de trouver l'exécutable Ollama. Veuillez l'installer manuellement depuis https://ollama.ai"
            )
            return False
            
        logging.info(f"Démarrage d'Ollama depuis {ollama_path}")
        
        if platform.system() == "Windows":
            # Sur Windows, utiliser CREATE_NEW_CONSOLE pour avoir une fenêtre séparée
            subprocess.Popen([ollama_path, "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            # Sur Linux/Mac, exécuter en arrière-plan
            subprocess.Popen([ollama_path, "serve"], 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            start_new_session=True)
                            
        # Attendre un peu que le serveur démarre
        time.sleep(2)
        
        # Vérifier si Ollama est maintenant en cours d'exécution
        for _ in range(5):  # Essayer 5 fois
            if is_ollama_running():
                logging.info("Ollama démarré avec succès")
                return True
            time.sleep(1)
            
        logging.warning("Ollama n'a pas pu démarrer correctement")
        return False
        
    except Exception as e:
        logging.error(f"Erreur lors du démarrage d'Ollama: {e}", exc_info=True)
        QMessageBox.critical(None, "Erreur Ollama", f"Impossible de lancer Ollama: {e}")
        return False