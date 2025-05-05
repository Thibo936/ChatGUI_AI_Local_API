# ChatGUI_AI_Local_API

## English

### Description

ChatGUI_AI_Local_API is a graphical user interface (GUI) application that allows you to chat with various AI models. It supports both local models via Ollama and cloud-based models via the OpenAI API. The application provides a simple interface to manage multiple conversations, select different AI models, and monitor resource usage.

### Features

*   **Multi-Model Support:** Interact with local models (via Ollama) and OpenAI models (GPT series).
*   **Conversation Management:** Create, switch between, and delete multiple chat conversations.
*   **Local Storage:** Conversations are saved locally in JSON format.
*   **Model Favorites:** Mark preferred models as favorites for quick access.
*   **Resource Monitoring:** Displays CPU and RAM usage.
*   **Markdown Rendering:** Basic Markdown support for assistant messages (bold, italics, code blocks).
*   **Dependency Checks:** Checks for necessary dependencies like VC++ Runtime and Python packages on startup.
*   **Cross-Platform:** Built with PySide6, aiming for cross-platform compatibility (primarily tested on Windows).

### Requirements

*   Python 3.x
*   Ollama (for local models): [https://ollama.com/](https://ollama.com/)
*   Required Python packages (see `requirements.txt` - installation attempted automatically):
    *   `PySide6`
    *   `requests`
    *   `psutil`
    *   `python-dotenv`
    *   `openai`
    *   `httpx`
*   (Windows) Microsoft Visual C++ Redistributable for Visual Studio 2015-2022 (x64). Installation prompted if missing.
*   (Optional) OpenAI API Key for using OpenAI models.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ChatGUI_AI_Local_API
    ```
2.  **Install Ollama:** Download and install Ollama from [https://ollama.com/](https://ollama.com/). Ensure the Ollama server is running.
3.  **Install Python dependencies:** The application attempts to install missing dependencies automatically. You can also install them manually:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file might need to be created based on the imports in the script if not already present).*
4.  **(Optional) Set up OpenAI API Key:** Create a `.env` file in the project directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY='your_api_key_here'
    ```

### Usage

1.  Run the Python script:
    ```bash
    python ollama_chat_gui3.py
    ```
2.  The application will start. If Ollama is not running, it might attempt to start it (depending on system configuration).
3.  Select an available AI model from the dropdown list (local models fetched from Ollama, OpenAI models if the API key is configured).
4.  Start chatting! Use the "➕ Nouvelle conversation" button to create new chats.
5.  Conversations are saved automatically.

---

## Français

### Description

ChatGUI_AI_Local_API est une application d'interface graphique (GUI) qui vous permet de discuter avec divers modèles d'IA. Elle prend en charge à la fois les modèles locaux via Ollama et les modèles basés sur le cloud via l'API OpenAI. L'application fournit une interface simple pour gérer plusieurs conversations, sélectionner différents modèles d'IA et surveiller l'utilisation des ressources.

### Fonctionnalités

*   **Support Multi-Modèles :** Interagissez avec des modèles locaux (via Ollama) et des modèles OpenAI (série GPT).
*   **Gestion des Conversations :** Créez, basculez entre et supprimez plusieurs conversations de chat.
*   **Stockage Local :** Les conversations sont sauvegardées localement au format JSON.
*   **Modèles Favoris :** Marquez les modèles préférés comme favoris pour un accès rapide.
*   **Surveillance des Ressources :** Affiche l'utilisation du CPU et de la RAM.
*   **Rendu Markdown :** Prise en charge basique du Markdown pour les messages de l'assistant (gras, italique, blocs de code).
*   **Vérification des Dépendances :** Vérifie les dépendances nécessaires comme le Runtime VC++ et les paquets Python au démarrage.
*   **Multiplateforme :** Construit avec PySide6, visant la compatibilité multiplateforme (principalement testé sous Windows).

### Prérequis

*   Python 3.x
*   Ollama (pour les modèles locaux) : [https://ollama.com/](https://ollama.com/)
*   Paquets Python requis (voir `requirements.txt` - installation tentée automatiquement) :
    *   `PySide6`
    *   `requests`
    *   `psutil`
    *   `python-dotenv`
    *   `openai`
    *   `httpx`
*   (Windows) Microsoft Visual C++ Redistributable pour Visual Studio 2015-2022 (x64). L'installation est proposée si manquant.
*   (Optionnel) Clé API OpenAI pour utiliser les modèles OpenAI.

### Installation

1.  **Cloner le dépôt :**
    ```bash
    git clone <url_du_depot>
    cd ChatGUI_AI_Local_API
    ```
2.  **Installer Ollama :** Téléchargez et installez Ollama depuis [https://ollama.com/](https://ollama.com/). Assurez-vous que le serveur Ollama est en cours d'exécution.
3.  **Installer les dépendances Python :** L'application tente d'installer automatiquement les dépendances manquantes. Vous pouvez aussi les installer manuellement :
    ```bash
    pip install -r requirements.txt
    ```
    *(Note : Un fichier `requirements.txt` pourrait devoir être créé basé sur les imports du script s'il n'est pas déjà présent).*
4.  **(Optionnel) Configurer la clé API OpenAI :** Créez un fichier `.env` dans le répertoire du projet et ajoutez votre clé API OpenAI :
    ```env
    OPENAI_API_KEY='votre_cle_api_ici'
    ```

### Utilisation

1.  Exécutez le script Python :
    ```bash
    python ollama_chat_gui3.py
    ```
2.  L'application va démarrer. Si Ollama n'est pas en cours d'exécution, elle pourrait tenter de le démarrer (selon la configuration système).
3.  Sélectionnez un modèle d'IA disponible dans la liste déroulante (modèles locaux récupérés depuis Ollama, modèles OpenAI si la clé API est configurée).
4.  Commencez à discuter ! Utilisez le bouton "➕ Nouvelle conversation" pour créer de nouveaux chats.
5.  Les conversations sont sauvegardées automatiquement.
