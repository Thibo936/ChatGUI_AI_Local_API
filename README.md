# ChatGUI_AI_Local_API

> **English 🇬🇧 | Français 🇫🇷**
> Bilingual README for a local AI chat GUI.

---

## English 🇬🇧

### Overview

**ChatGUI_AI_Local_API** is a lightweight desktop application (Python + PySide6) to chat with:

* **Local models** served by **[Ollama](https://ollama.com/)** (`localhost:11434`)
* **OpenAI models** (if `OPENAI_API_KEY` is set)

It features a clean GUI, multi-conversation, model favourites, visible chain-of-thought, and real-time resource usage.

### Key features

* Multi-conversation sidebar with auto-save to `%APPDATA%/OllamaChats`
* Toggle assistant chain-of-thought (`<think>…</think>`) with a single click
* Model favourites ⭐ and instant switch
* Token statistics (total & tok/s) + CPU/RAM monitor
* Automatic checks for missing Python dependencies & VC++ runtime on Windows
* Automatic detection and launch of Ollama server if not running
* OpenAI model support if API key is set

### Prerequisites

| Requirement      | Notes                                                        |
| ---------------- | ------------------------------------------------------------ |
| Python **3.10+** | Windows / macOS / Linux                                      |
| **Ollama**       | Needed only for local models – must run on `localhost:11434` |
| `OPENAI_API_KEY` | Optional – enables OpenAI models                             |

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
```

### Running

```bash
python main.py
```

On first launch, the app will propose to install missing Python packages, VC++ redistributable, or start **Ollama** if not detected.

### Environment variables

* `OPENAI_API_KEY` – OpenAI key (optional)
* `LOCALAPPDATA`  – Overrides default data directory on Windows

### File structure

```
main.py                  # Main application
config.py                # Configuration
ollama_client.py         # Ollama API client
chat_window.py           # Main GUI window
models.py                # Message dataclass
utils.py                 # Utilities (logging, checks, etc.)
requirements.txt         # Dependencies
%APPDATA%/OllamaChats/   # Auto-saved chats & settings
  ├── model_favorites.json
  └── <uuid>.json        # One file per conversation
```

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

**ChatGUI_AI_Local_API** est une application de bureau légère (Python + PySide6) qui permet de discuter :

* avec des **modèles locaux** servis par **[Ollama](https://ollama.com/)** (`localhost:11434`)
* avec des **modèles OpenAI** (si la variable `OPENAI_API_KEY` est définie)

Elle propose une interface soignée, la gestion de plusieurs conversations, des modèles favoris, l’affichage des pensées de l’IA et la surveillance des ressources système.

### Fonctionnalités clés

* Barre latérale multi-conversations avec sauvegarde automatique dans `%APPDATA%/OllamaChats`
* Affichage/masquage des pensées de l’IA (`<think>…</think>`) en un clic
* Favoris de modèles ⭐ et changement instantané
* Statistiques de tokens (total & tok/s) + moniteur CPU/RAM en temps réel
* Vérification automatique des dépendances Python et du runtime VC++ sous Windows
* Détection et lancement automatique du serveur Ollama si nécessaire
* Prise en charge des modèles OpenAI si la clé API est définie

### Prérequis

| Pré-requis       | Notes                                                                              |
| ---------------- | ---------------------------------------------------------------------------------- |
| Python **3.10+** | Windows / macOS / Linux                                                            |
| **Ollama**       | Nécessaire uniquement pour les modèles locaux – doit tourner sur `localhost:11434` |
| `OPENAI_API_KEY` | Optionnel – active les modèles OpenAI                                              |

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
```

### Lancement

```bash
python main.py
```

Au premier démarrage, l’application propose d’installer les paquets Python manquants, le runtime VC++ ou de lancer **Ollama** s’il n’est pas détecté.

### Variables d’environnement

* `OPENAI_API_KEY` – Clé OpenAI (optionnel)
* `LOCALAPPDATA`  – Redéfinit le répertoire de données sous Windows

### Arborescence

```
main.py                  # Application principale
config.py                # Configuration
ollama_client.py         # Client API Ollama
chat_window.py           # Fenêtre principale
models.py                # Dataclass Message
utils.py                 # Utilitaires (log, vérifications, etc.)
requirements.txt         # Dépendances
%APPDATA%/OllamaChats/   # Conversations et paramètres sauvegardés
  ├── model_favorites.json
  └── <uuid>.json        # Une conversation par fichier
```

### Création d’un exécutable Windows

```bash
pip install pyinstaller
pyinstaller --onefile --noconsole main.py
```

### Contribuer

Les pull requests et issues sont les bienvenus !

### Licence

MIT
