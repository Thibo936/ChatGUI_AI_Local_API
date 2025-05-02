OllamaChat - Interface GUI pour modèles IA locaux et distants

## Description
OllamaChat est une interface graphique Python élégante qui permet d'interagir avec différents modèles d'intelligence artificielle:
- Modèles Ollama hébergés localement
- Modèles GGUF via llama-cpp-python
- API OpenAI (GPT, etc.)

## Fonctionnalités principales

- **Interface conviviale et intuitive**
  - Design moderne avec vue partagée (conversations à gauche, chat à droite)
  - Rendu Markdown pour les réponses (gras, italique, etc.)
  - Blocs de code avec formatage amélioré
  - Mode "pensée" avec balises `<think></think>` pliables (▶/▼)

- **Gestion multi-modèles**
  - Détection automatique des modèles Ollama disponibles
  - Chargement des modèles GGUF locaux (.gguf)
  - Intégration des modèles OpenAI (avec clé API)
  - Système de favoris pour les modèles fréquemment utilisés (★)

- **Gestion des conversations**
  - Sauvegarde automatique dans `%APPDATA%/OllamaChats`
  - Création et suppression de conversations
  - Titres générés automatiquement à partir du premier message

- **Expérience utilisateur optimisée**
  - Éditeur auto-redimensionnable
  - Envoi par Ctrl+Entrée
  - Statistiques temps réel (tokens, tok/s, CPU/RAM)

## Installation

### Prérequis
- Python 3.8+
- Ollama installé pour les modèles Ollama (facultatif)

### Dépendances
```bash
pip install pyside6 requests psutil rich python-dotenv llama-cpp-python
```

### Configuration
1. Placez vos modèles GGUF dans le même répertoire que le script ou dans un sous-dossier `models/`
2. Pour utiliser OpenAI, créez un fichier `.env` ou définissez la variable d'environnement:
   ```
   OPENAI_API_KEY=votre_clé_api
   ```

## Utilisation

### Démarrage
```bash
python ollama_chat_gui3.py
```

Le serveur Ollama sera démarré automatiquement si nécessaire.

### Interactions
- **Nouvelle conversation**: Cliquez sur "➕ Nouvelle conversation"
- **Envoi de messages**: Tapez votre message puis cliquez sur "Envoyer" ou utilisez Ctrl+Entrée
- **Changement de modèle**: Sélectionnez un modèle dans la liste déroulante en bas
- **Favoris**: Clic-droit sur un modèle pour l'ajouter/retirer des favoris

## Création d'un exécutable
```bash
pyinstaller --onefile --windowed --collect-all llama_cpp ollama_chat_gui3.py
```

## Perspectives d'évolution
- Édition et suppression de messages individuels
- Support du monitoring GPU via NVML
- Thèmes personnalisables (clair/sombre)
- Raccourcis clavier supplémentaires

## Compatibilité des modèles
- **Ollama**: Tous les modèles disponibles via API Ollama
- **GGUF**: Modèles compatibles avec llama.cpp (ex: `llama2-7b-chat.gguf`, `gemma3:4b.gguf`)
- **OpenAI**: Modèles GPT et autres modèles accessibles via l'API OpenAI