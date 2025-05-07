import os
import dotenv
from pathlib import Path

dotenv.load_dotenv()

# Chemins des répertoires
local_appdata = os.getenv("LOCALAPPDATA")
DATA_DIR = Path(local_appdata) / "ChatGUI_AI_Local_API"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Configuration Ollama
OLLAMA_URL = "http://localhost:11434"
SAVE_DIR = Path(os.getenv("APPDATA", ".")) / "OllamaChats"
SAVE_DIR.mkdir(exist_ok=True)

# Configuration OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuration du logging
LOG_DIR = DATA_DIR
LOG_FILE = LOG_DIR / "chatgui.log"
LOG_FORMATTER = "%(asctime)s %(levelname)-8s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"