# OllamaChat – GUI Interface for Local and Remote AI Models

## Description

OllamaChat is a Python graphical interface that lets you interact with several kinds of artificial‑intelligence models:

* **Locally hosted Ollama models**
* **GGUF models** via *llama‑cpp‑python*
* **OpenAI models** through the OpenAI API

## Key Features

* **Interface**

  * Markdown rendering in replies (bold, italics, etc.)
  * Code blocks with improved formatting
  * “Thinking” mode with collapsible `<think></think>` tags (▶/▼)

* **Multi‑Model Management**

  * Automatic detection of available Ollama models
  * Loading of local GGUF models (`.gguf`)
  * Integration of OpenAI models (requires an API key)
  * Favourites system for frequently used models (★)

* **Conversation Management**

  * Automatic saving to `%APPDATA%/OllamaChats`
  * Create and delete conversations
  * Titles auto‑generated from the first user message

* **User Experience**

  * Auto‑resizing editor
  * Send with **Ctrl + Enter**
  * Real‑time statistics (tokens, tok/s, CPU/RAM)

## Installation

### Prerequisites

* Python 3.8 or newer
* **Ollama** installed for Ollama models *(optional)*
* **llama‑cpp‑python** for GGUF models *(optional)*
* **openai** package for OpenAI models *(optional)*

### Dependencies

```bash
pip install pyside6 requests psutil rich python-dotenv llama-cpp-python
```

### Configuration

1. Place your GGUF models in the same folder as the script or inside a `models/` sub‑folder.
2. To use OpenAI, create a `.env` file or set an environment variable:

   ```
   OPENAI_API_KEY=your_api_key
   ```

## Usage

### Launch

```bash
python ollama_chat_gui3.py
```

The Ollama server starts automatically if required.

### Interactions

* **New conversation**: Click **➕ New conversation**
* **Send messages**: Type your message, then click **Send** or press **Ctrl + Enter**
* **Switch model**: Select a model from the drop‑down list at the bottom
* **Favourites**: Right‑click a model to add or remove it from favourites

## Building a Stand‑Alone Executable

```bash
pyinstaller --onefile --windowed --collect-all llama_cpp --collect-all PySide6 --name ChatLite ollama_chat_gui3.py
```

## Model Compatibility

* **Ollama**: Any model reachable via the Ollama API
* **GGUF**: Models compatible with *llama.cpp* (e.g. `llama2-7b-chat.gguf`, `gemma3:4b.gguf`)
* **OpenAI**: GPT and other models accessible through the OpenAI API

--

OllamaChat - Interface GUI pour modèles IA locaux et distants

## Description
OllamaChat est une interface graphique Python qui permet d'interagir avec différents modèles d'intelligence artificielle:
- Modèles Ollama hébergés localement
- Modèles GGUF via llama-cpp-python
- API OpenAI

## Fonctionnalités principales

- **Interface**
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

- **Expérience utilisateur**
  - Éditeur auto-redimensionnable
  - Envoi par Ctrl+Entrée
  - Statistiques temps réel (tokens, tok/s, CPU/RAM)

## Installation

### Prérequis
- Python 3.8+
- Ollama installé pour les modèles Ollama (facultatif)
- llama-cpp-python pour les modèles GGUF (facultatif)
- OpenAI pour les modèles OpenAI (facultatif)

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
pyinstaller --onefile --windowed --collect-all llama_cpp --collect-all PySide6 --name ChatLite ollama_chat_gui3.py
```

## Compatibilité des modèles
- **Ollama**: Tous les modèles disponibles via API Ollama
- **GGUF**: Modèles compatibles avec llama.cpp (ex: `llama2-7b-chat.gguf`, `gemma3:4b.gguf`)
- **OpenAI**: Modèles GPT et autres modèles accessibles via l'API OpenAI