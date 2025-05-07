import sys
import os
from PySide6.QtWidgets import QApplication

from utils import setup_logging, check_vc_redist, check_dependencies, prompt_install
from chat_window import ChatWindow

def main():
    setup_logging()

    if os.name == "nt":
        if not check_vc_redist():
            prompt_install(
                "VC++ Runtime manquant",
                "Le runtime VC++ 2022 n'est pas installé. Voulez-vous lancer le téléchargement ?",
                url="https://aka.ms/vs/17/release/vc_redist.x64.exe"
            )

    missing = check_dependencies()
    if missing:
        prompt_install(
            "Dépendances manquantes",
            "Certaines dépendances Python sont absentes :\n- " + "\n- ".join(missing) + "\nVoulez-vous tenter une installation automatique ?",
        )
        import subprocess
        subprocess.call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    app = QApplication(sys.argv)
    win = ChatWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()