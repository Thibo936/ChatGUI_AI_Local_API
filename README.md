# ChatGUI_AI_Local_API

> **English 🇬🇧 | Français 🇫🇷**
> Bilingual README for a local AI chat GUI with file attachments and vision support.

---

## English 🇬🇧

### Overview

**ChatGUI_AI_Local_API** is a lightweight desktop application (Python + PySide6) to chat with:

* **Local models** served by **[Ollama](https://ollama.com/)** (`localhost:11434`)
* **OpenAI models** (if `OPENAI_API_KEY` is set)

It features a clean GUI, multi-conversation, model favourites, visible chain-of-thought, file attachments, and real-time resource usage.

### Key features

* **Multi-conversation** sidebar with auto-save to `%APPDATA%/OllamaChats`
* **File attachments** support:
  - **Images** (PNG, JPG, WEBP, etc.) with vision model support
  - **PDFs** with automatic text extraction
  - **Text files** (code, documents, etc.)
  - Drag & drop functionality
* **Vision models** support for image analysis (Ollama: LLaVA, Gemma3, Qwen2.5VL, Moondream | OpenAI: GPT-4o, GPT-4.1, etc.)
* Toggle assistant **chain-of-thought** (`<think>…</think>`) with a single click
* **Model management**:
  - Favourites ⭐ with priority display
  - Visibility settings to hide/show specific models
  - Automatic detection of vision capabilities
* **Token statistics** (total & tok/s) + CPU/RAM monitor
* **Code blocks** with syntax highlighting and copy functionality
* Automatic checks for missing Python dependencies & VC++ runtime on Windows
* Automatic detection and launch of Ollama server if not running

### Prerequisites

| Requirement      | Notes                                                        |
| ---------------- | ------------------------------------------------------------ |
| Python **3.10+** | Windows / macOS / Linux                                      |
| **Ollama**       | Needed only for local models – must run on `localhost:11434` |
| `OPENAI_API_KEY` | Optional – enables OpenAI models                             |

### Required dependencies

The application uses these packages for file processing:

* **Pillow** – Image processing, resizing, and format conversion
* **PyPDF2** – PDF text extraction for document analysis
* **pytesseract** – OCR (Optical Character Recognition) for extracting text from images when vision models aren't available

### Installation

```bash
# 1. Clone
$ git clone https://github.com/your‑name/ChatGUI_AI_Local_API.git
$ cd ChatGUI_AI_Local_API

# 2. Create venv (recommended)
$ python -m venv .venv
$ source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. For OCR functionality, install Tesseract OCR on your system:
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# macOS: brew install tesseract
# Linux: sudo apt-get install tesseract-ocr
```

### Running

```bash
python main.py
```

On first launch, the app will propose to install missing Python packages, VC++ redistributable, or start **Ollama** if not detected.

### Supported models

**Ollama models with vision support:**
- LLaVA (all variants)
- Gemma3 (vision models)
- Qwen2.5VL
- Moondream
- Llama 3.2 Vision
- BakLLaVA
- Granite 3.2 Vision
- LLaVA-Phi3

**OpenAI models with vision support:**
- GPT-4o and variants
- GPT-4.1 series
- O1, O3, O4 series

### Environment variables

* `OPENAI_API_KEY` – OpenAI key (optional)
* `LOCALAPPDATA`  – Overrides default data directory on Windows

### File structure

```
main.py                  # Main application
config.py                # Configuration
ollama_client.py         # Ollama API client
chat_window.py           # Main GUI window
models.py                # Message & ModelCaps dataclasses
utils.py                 # Utilities (logging, checks, etc.)
file_utils.py            # File processing utilities (images, PDFs, OCR)
requirements.txt         # Dependencies
%APPDATA%/OllamaChats/   # Auto-saved chats & settings
  ├── model_favorites.json
  ├── model_visibility.json
  └── <uuid>.json        # One file per conversation
```

### Usage

1. **Start a conversation**: Click "➕ Nouvelle conversation"
2. **Attach files**: 
   - Click the 📎 button or drag & drop files
   - Supported: Images (analyzed by vision models), PDFs (text extracted), text files
3. **Switch models**: Use the dropdown and ⚙️ button to manage visibility
4. **View thinking**: Click ▶ arrows to expand AI reasoning
5. **Copy code**: Click 📄 button on code blocks

### Packaging (Windows)

```bash
pip install pyinstaller
pyinstaller --onefile --noconsole main.py
```

### Contributing

Pull requests and issues are welcome!

### License

MIT

---

## Français 🇫🇷

### Aperçu

**ChatGUI_AI_Local_API** est une application de bureau légère (Python + PySide6) qui permet de discuter :

* avec des **modèles locaux** servis par **[Ollama](https://ollama.com/)** (`localhost:11434`)
* avec des **modèles OpenAI** (si la variable `OPENAI_API_KEY` est définie)

Elle propose une interface soignée, la gestion de plusieurs conversations, des modèles favoris, l'affichage des pensées de l'IA, la prise en charge de fichiers joints et la surveillance des ressources système.

### Fonctionnalités clés

* **Multi-conversations** avec sauvegarde automatique dans `%APPDATA%/OllamaChats`
* **Pièces jointes** :
  - **Images** (PNG, JPG, WEBP, etc.) avec support des modèles de vision
  - **PDFs** avec extraction automatique de texte
  - **Fichiers texte** (code, documents, etc.)
  - Fonctionnalité glisser-déposer
* **Modèles de vision** pour l'analyse d'images (Ollama : LLaVA, Gemma3, Qwen2.5VL, Moondream | OpenAI : GPT-4o, GPT-4.1, etc.)
* Affichage/masquage des **pensées de l'IA** (`<think>…</think>`) en un clic
* **Gestion des modèles** :
  - Favoris ⭐ avec affichage prioritaire
  - Paramètres de visibilité pour masquer/afficher des modèles spécifiques
  - Détection automatique des capacités de vision
* **Statistiques de tokens** (total & tok/s) + moniteur CPU/RAM en temps réel
* **Blocs de code** avec coloration syntaxique et fonction de copie
* Vérification automatique des dépendances Python et du runtime VC++ sous Windows
* Détection et lancement automatique du serveur Ollama si nécessaire

### Prérequis

| Pré-requis       | Notes                                                                              |
| ---------------- | ---------------------------------------------------------------------------------- |
| Python **3.10+** | Windows / macOS / Linux                                                            |
| **Ollama**       | Nécessaire uniquement pour les modèles locaux – doit tourner sur `localhost:11434` |
| `OPENAI_API_KEY` | Optionnel – active les modèles OpenAI                                              |

### Dépendances requises

L'application utilise ces packages pour le traitement de fichiers :

* **Pillow** – Traitement d'images, redimensionnement et conversion de formats
* **PyPDF2** – Extraction de texte des PDFs pour l'analyse de documents
* **pytesseract** – OCR (Reconnaissance Optique de Caractères) pour extraire le texte des images quand les modèles de vision ne sont pas disponibles

### Installation

```bash
# 1. Cloner le dépôt
$ git clone https://github.com/votre‑pseudo/ChatGUI_AI_Local_API.git
$ cd ChatGUI_AI_Local_API

# 2. Créer un virtualenv (recommandé)
$ python -m venv .venv
$ source .venv/bin/activate   # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
$ pip install -r requirements.txt

# 4. Pour la fonctionnalité OCR, installer Tesseract OCR sur votre système :
# Windows : Télécharger depuis https://github.com/UB-Mannheim/tesseract/wiki
# macOS : brew install tesseract
# Linux : sudo apt-get install tesseract-ocr
```

### Lancement

```bash
python main.py
```

Au premier démarrage, l'application propose d'installer les paquets Python manquants, le runtime VC++ ou de lancer **Ollama** s'il n'est pas détecté.

### Modèles supportés

**Modèles Ollama avec support vision :**
- LLaVA (toutes variantes)
- Gemma3 (modèles vision)
- Qwen2.5VL
- Moondream
- Llama 3.2 Vision
- BakLLaVA
- Granite 3.2 Vision
- LLaVA-Phi3

**Modèles OpenAI avec support vision :**
- GPT-4o et variantes
- Série GPT-4.1
- Séries O1, O3, O4

### Variables d'environnement

* `OPENAI_API_KEY` – Clé OpenAI (optionnel)
* `LOCALAPPDATA`  – Redéfinit le répertoire de données sous Windows

### Arborescence

```
main.py                  # Application principale
config.py                # Configuration
ollama_client.py         # Client API Ollama
chat_window.py           # Fenêtre principale
models.py                # Dataclasses Message & ModelCaps
utils.py                 # Utilitaires (log, vérifications, etc.)
file_utils.py            # Utilitaires de traitement de fichiers (images, PDFs, OCR)
requirements.txt         # Dépendances
%APPDATA%/OllamaChats/   # Conversations et paramètres sauvegardés
  ├── model_favorites.json
  ├── model_visibility.json
  └── <uuid>.json        # Une conversation par fichier
```

### Utilisation

1. **Démarrer une conversation** : Cliquez sur "➕ Nouvelle conversation"
2. **Joindre des fichiers** : 
   - Cliquez sur le bouton 📎 ou glissez-déposez des fichiers
   - Supportés : Images (analysées par les modèles de vision), PDFs (texte extrait), fichiers texte
3. **Changer de modèle** : Utilisez la liste déroulante et le bouton ⚙️ pour gérer la visibilité
4. **Voir la réflexion** : Cliquez sur les flèches ▶ pour développer le raisonnement de l'IA
5. **Copier le code** : Cliquez sur le bouton 📄 sur les blocs de code

### Création d'un exécutable Windows

```bash
pip install pyinstaller
pyinstaller --onefile --noconsole main.py
```

### Contribuer

Les pull requests et issues sont les bienvenus !

### Licence

MIT
