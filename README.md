# ChatGUI\_AI\_Local\_API

> **English 🇬🇧 | Français 🇫🇷**
> A simple bilingual README for a simple local AI chat GUI.

---

## English 🇬🇧

### Overview

**ChatGUI\_AI\_Local\_API** is a lightweight desktop application written in Python + PySide6 that lets you chat with:

* **Local models** served by **[Ollama](https://ollama.com/)** (`localhost:11434`)
* **OpenAI models** (when `OPENAI_API_KEY` is defined)

It ships with a clean GUI that supports multiple conversations, model favourites, visible chain‑of‑thought, and real‑time resource usage.

### Key features

* Multi‑conversation sidebar with auto‑save to `%APPDATA%/OllamaChats`
* Toggle assistant chain‑of‑thought (`<think>…</think>`) with a single click
* Model favourites ⭐ and instant switch
* Token statistics (total & tok/s) + CPU/RAM monitor
* Automatic checks for missing Python deps & VC++ runtime on Windows

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

# 3. Install deps
$ pip install -r requirements.txt
```

### Running

```bash
python ollama_chat_gui3.py
```

On first launch the app proposes to install missing Python packages, VC++ redistributable, or start **Ollama** if it is not detected.

### Environment variables

* `OPENAI_API_KEY` – OpenAI key (optional)
* `LOCALAPPDATA`  – Overrides default data directory on Windows

### File structure

```
ollama_chat_gui3.py        # Main application
requirements.txt          # Dependencies
%APPDATA%/OllamaChats/    # Auto‑saved chats & settings
  ├── model_favorites.json
  └── <uuid>.json         # One file per conversation
```

### Packaging (Windows)

```bash
pip install pyinstaller
pyinstaller --onefile --noconsole ollama_chat_gui3.py
```

### Contributing

Pull requests and issues are welcome!

### License

MIT

---

## Français 🇫🇷

### Aperçu

**ChatGUI\_AI\_Local\_API** est une application de bureau légère (Python + PySide6) qui permet de discuter :

* avec des **modèles locaux** servis par **[Ollama](https://ollama.com/)** (`localhost:11434`)
* avec des **modèles OpenAI** (si la variable `OPENAI_API_KEY` est définie)

Elle propose une interface soignée, la gestion de plusieurs conversations, des modèles favoris, l’affichage des pensées de l’IA et la surveillance des ressources système.

### Fonctionnalités clés

* Barre latérale multi‑conversations avec sauvegarde automatique dans `%APPDATA%/OllamaChats`
* Affichage/masquage des pensées de l’IA (`<think>…</think>`) en un clic
* Favoris de modèles ⭐ et changement instantané
* Statistiques de tokens (total & tok/s) + moniteur CPU/RAM en temps réel
* Vérifications automatiques des dépendances Python et du runtime VC++ sous Windows

### Prérequis

| Pré‑requis       | Notes                                                                              |
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
python ollama_chat_gui3.py
```

Au premier démarrage, l’application propose d’installer les paquets Python manquants, le runtime VC++ ou de lancer **Ollama** s’il n’est pas détecté.

### Variables d’environnement

* `OPENAI_API_KEY` – Clé OpenAI (optionnel)
* `LOCALAPPDATA`  – Redéfinit le répertoire de données sous Windows

### Arborescence

```
ollama_chat_gui3.py        # Application principale
requirements.txt          # Dépendances
%APPDATA%/OllamaChats/    # Conversations et paramètres sauvegardés
  ├── model_favorites.json
  └── <uuid>.json         # Une conversation par fichier
```

### Création d’un exécutable Windows

```bash
pip install pyinstaller
pyinstaller --onefile --noconsole ollama_chat_gui3.py
```

### Contribuer

Les pull requests et issues sont les bienvenus !

### Licence

MIT
