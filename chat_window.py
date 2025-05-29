from dataclasses import asdict
from pathlib import Path
import os
import json
import re
import time
import uuid
import html
import logging
import io
import base64
import mimetypes
import platform
import subprocess
import shutil
import requests
import psutil
import hashlib
from PySide6.QtCore import Qt, QTimer, QUrl, QRunnable, QThreadPool, Signal, Slot, QObject
from PySide6.QtGui import QAction, QTextCursor, QTextOption, QDesktopServices, QIcon, QDragEnterEvent, QDropEvent, QCursor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QComboBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QMenu,
    QFileDialog,
    QSizePolicy,
    QDialog,
    QCheckBox,
    QScrollArea,
    QToolTip,
)

from config import SAVE_DIR
from models import Message, ModelCaps
from ollama_client import OllamaClient, is_ollama_running, start_ollama_server
from utils import log_critical_error
from file_utils import get_file_type, read_text_file, extract_text_from_pdf, resize_and_encode_image, ocr_image

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# Modèles par défaut si aucun n'est disponible
DEFAULT_MODELS = ["llama2", "mistral", "phi2", "gemma:2b"]

class WorkerSignals(QObject):
    finished = Signal(object) # Pourrait être tuple(role, content, tokens, tok_s, model)
    error = Signal(str)
    stats_updated = Signal(str)

class ApiWorker(QRunnable):
    def __init__(self, client, model_name_full, api_payload, is_openai):
        super().__init__()
        self.client = client # Peut être OllamaClient ou le client OpenAI
        self.model_name_full = model_name_full
        self.api_payload = api_payload
        self.is_openai = is_openai
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            if self.is_openai:
                model_name = self.model_name_full[len("OpenAI: "):]
                start = time.time()
                # Créer le client OpenAI ici pour l'isolation des threads
                import openai
                oai_client = openai.OpenAI()
                resp_obj = oai_client.chat.completions.create(
                    model=model_name,
                    messages=self.api_payload,
                )
                duration = max(time.time() - start, 1e-6)
                resp_content = resp_obj.choices[0].message.content
                total_tokens = resp_obj.usage.total_tokens
                tok_s = total_tokens / duration
                result = ("assistant", resp_content, total_tokens, tok_s, self.model_name_full)
            else: # Ollama
                # Le client Ollama est supposé être thread-safe pour les requêtes
                resp_content, total_tokens, tok_s = self.client.chat_custom_payload(self.api_payload)
                result = ("assistant", resp_content, total_tokens, tok_s, self.model_name_full)
            
            self.signals.finished.emit(result)
            self.signals.stats_updated.emit(f"Tokens: {total_tokens} – {tok_s:.1f} tok/s")

        except Exception as e:
            log_critical_error(f"Erreur API Worker ({self.model_name_full})", e) 
            self.signals.error.emit(str(e))

class ModelVisibilityDialog(QDialog):
    def __init__(self, all_models, visible_models, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gérer la visibilité des modèles")
        self.all_models = all_models
        self.visible_models = visible_models

        self.layout = QVBoxLayout(self)
        self.checkboxes = []

        for model in all_models:
            checkbox = QCheckBox(model, self)
            checkbox.setChecked(model in visible_models)
            self.layout.addWidget(checkbox)
            self.checkboxes.append(checkbox)

        self.save_button = QPushButton("Sauvegarder", self)
        self.save_button.clicked.connect(self.accept)
        self.layout.addWidget(self.save_button)

    def get_selected_models(self):
        return [checkbox.text() for checkbox in self.checkboxes if checkbox.isChecked()]

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ollama_available = is_ollama_running()
        self.setWindowTitle("ChatGUI AI Local API")
        self.resize(960, 640)

        if not self.ollama_available:
            reply = QMessageBox.question(
                self,
                "Ollama non disponible",
                "Ollama n'est pas en cours d'exécution. Voulez-vous le démarrer?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                start_ollama_server()
                # Attendre que le serveur démarre
                for _ in range(5):  # Essayer 5 fois
                    time.sleep(1)
                    if is_ollama_running():
                        self.ollama_available = True
                        break

        self.client = OllamaClient()
        self.current_model: str | None = None
        self.conversations: dict[str, list[Message]] = {}
        self.current_conv_id: str | None = None
        self.reason_states: dict[str, bool] = {}
        self.model_capabilities: dict[str, ModelCaps] = {}
        self.pending_attachments: list[dict] = [] 
        self.file_content_map = {}
        self.code_block_contents: dict[str, str] = {}

        self.model_visibility_config_file = SAVE_DIR / "model_visibility.json"
        self.visible_models: list[str] | None = self._load_model_visibility_config() # Charger au démarrage

        self.attachments_panel_widget = None
        self.toggle_files_panel_btn = None
        self.model_settings_btn = None

        self._setup_ui()
        self._setup_connections()
        self._start_stats_timer()
        self.threadpool = QThreadPool()
        logging.info(f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads")

        # Remplir la liste des modèles
        self._populate_model_box()
        
        # Sélectionner un modèle s'il y en a
        if self.model_box.count() > 0:
            self.change_model(self.model_box.currentText())
        else:
            QMessageBox.warning(self, "Aucun modèle", "Aucun modèle n'a pu être chargé. Vérifiez votre connexion et la configuration d'Ollama/OpenAI.")
            # Garder les boutons désactivés si aucun modèle

        self.load_conversations()
        if not self.conversations:
            self.new_conversation()
        else:
            first_cid = list(self.conversations.keys())[0]
            for i in range(self.conv_list.count()):
                item = self.conv_list.item(i)
                if item.data(Qt.UserRole) == first_cid:
                    self.conv_list.setCurrentItem(item)
                    self.switch_conversation(item)
                    break

    def _load_model_visibility_config(self) -> list[str] | None:
        if not self.model_visibility_config_file.exists():
            return None  # Aucun fichier de config, afficher tous les modèles
        try:
            data = json.loads(self.model_visibility_config_file.read_text(encoding='utf-8'))
            if isinstance(data, list):
                return data
        except Exception as e:
            logging.error(f"Erreur lors du chargement de la configuration de visibilité des modèles: {e}", exc_info=True)
        return None # En cas d'erreur ou de format incorrect, afficher tout

    def _save_model_visibility_config(self):
        if self.visible_models is None: # Si None, ne pas créer de fichier pour que tout reste visible
            if self.model_visibility_config_file.exists():
                 # Si l'utilisateur veut à nouveau tout voir, on pourrait supprimer le fichier
                 # ou y écrire une liste vide (selon la convention choisie)
                 # Pour l'instant, si visible_models devient None, on ne fait rien pour garder l'état "tout afficher par défaut"
                 logging.info("Aucune configuration de visibilité des modèles à sauvegarder (tout est visible par défaut).")
            return
        try:
            self.model_visibility_config_file.write_text(json.dumps(self.visible_models, ensure_ascii=False, indent=2), encoding='utf-8')
            logging.info(f"Configuration de visibilité des modèles sauvegardée dans {self.model_visibility_config_file}")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de la configuration de visibilité des modèles: {e}", exc_info=True)

    def _open_model_visibility_settings(self):
        # Récupérer tous les modèles uniques, qu'ils soient actuellement visibles ou non.
        # Cela nécessite de faire une passe de découverte similaire à _populate_model_box,
        # mais sans appliquer le filtre de visibilité initialement.
        all_discovered_models = self._get_all_discovered_models()

        if not all_discovered_models:
            QMessageBox.information(self, "Aucun modèle trouvé", "Impossible de récupérer la liste des modèles disponibles pour configurer leur visibilité.")
            return

        # Si self.visible_models est None, cela signifie que tous les modèles sont actuellement considérés comme visibles.
        # Pour la dialogue, nous avons besoin d'une liste explicite.
        current_visible_models_for_dialog = self.visible_models
        if current_visible_models_for_dialog is None:
            current_visible_models_for_dialog = list(all_discovered_models)


        dialog = ModelVisibilityDialog(all_discovered_models, current_visible_models_for_dialog, self)
        if dialog.exec(): # exec() est bloquant et retourne QDialog.Accepted ou QDialog.Rejected
            self.visible_models = dialog.get_selected_models()
            self._save_model_visibility_config()
            self._populate_model_box() # Rafraîchir la combobox
            # S'assurer qu'un modèle valide est sélectionné si possible
            if self.model_box.count() > 0:
                current_selection = self.model_box.currentData()
                if not current_selection or current_selection not in self.visible_models:
                    self.model_box.setCurrentIndex(0) # Sélectionner le premier visible
                self.change_model(self.model_box.currentText())
            else: # Aucun modèle n'est visible ou disponible
                self.current_model = None
                self.send_btn.setEnabled(False)
                self.attachment_btn.setEnabled(False)
                QMessageBox.warning(self, "Aucun modèle visible", "Aucun modèle n'est actuellement sélectionné pour être affiché. Veuillez en choisir dans les paramètres.")


    def _get_all_discovered_models(self) -> list[str]:
        """
        Récupère tous les modèles disponibles d'Ollama et OpenAI.
        Similaire au début de _populate_model_box mais sans le filtrage de visibilité.
        """
        models = []
        temp_model_capabilities = {} # Utiliser un temporaire pour ne pas affecter self.model_capabilities

        # Ollama
        if is_ollama_running():
            try:
                ollama_models = self.client.list_models()
                models.extend(ollama_models)
                for model_name in ollama_models:
                    # (Logique de capabilities existante...)
                    known_ollama_vision_models = ["gemma3:4b-it-qat", "gemma3:12b-it-qat","gpt-4.1"]
                    is_vision_model = ("llava" in model_name.lower() or model_name.lower() in [m.lower() for m in known_ollama_vision_models])
                    temp_model_capabilities[model_name] = ModelCaps(name=model_name, supports_images=is_vision_model)
            except Exception as e:
                logging.error(f"Erreur (get_all): Récupération modèles Ollama: {e}")
        elif self.ollama_available: # Si ollama était dispo mais ne l'est plus
            models.extend(DEFAULT_MODELS) # Fallback aux defauts si ollama ne repond plus
            for model_name in DEFAULT_MODELS:
                 temp_model_capabilities[model_name] = ModelCaps(name=model_name)


        # OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                oai_client = OpenAI()
                response = oai_client.models.list()
                openai_models = [model.id for model in response.data if "gpt" in model.id.lower() or "o3" in model.id.lower()]
                for model_id in openai_models:
                    full_model_name = f"OpenAI: {model_id}"
                    models.append(full_model_name)
                    # (Logique de capabilities existante...)
                    supports_vision = "vision" in model_id.lower()
                    is_turbo = "turbo" in model_id.lower()
                    max_tokens = 128000 if "gpt-4" in model_id.lower() and is_turbo else (16385 if "gpt-3.5-turbo-16k" in model_id.lower() else 8192 if "gpt-4" in model_id.lower() else 4096)
                    temp_model_capabilities[full_model_name] = ModelCaps(name=full_model_name, supports_images=supports_vision, max_tokens=max_tokens, supports_general_files=True)
            except Exception as e:
                logging.error(f"Erreur (get_all): Récupération modèles OpenAI: {e}")
        
        return sorted(list(dict.fromkeys(models)))

    def _setup_ui(self):
        # Création des widgets
        self.new_conv_btn = QPushButton("➕ Nouvelle conversation")
        self.conv_list = QListWidget()
        self.chat_view = QTextBrowser()
        self.msg_edit = QTextEdit()
        self.send_btn = QPushButton("Envoyer (Ctrl+Enter)")
        self.attachment_btn = QPushButton()
        self.model_box = QComboBox()
        self.stats_label = QLabel("Tokens: 0 – 0 tok/s")
        self.res_label = QLabel("CPU: 0%  RAM: 0%")
        self.model_settings_btn = QPushButton("⚙️") # Ou QPushButton(QIcon.fromTheme("preferences-system"), "")
        self.model_settings_btn.setToolTip("Paramètres d'affichage des modèles")
        self.model_settings_btn.setFixedSize(36, 36) # Même taille que attachment_btn


        # Nouveaux widgets pour les pièces jointes
        self.attached_files_label = QLabel("Fichiers joints :")
        self.attached_files_list = QListWidget()
        self.attached_files_list.setMaximumHeight(100) # Hauteur limitée pour la liste
        self.remove_attachment_btn = QPushButton("Supprimer la sélection")

        # Configuration des widgets
        self.chat_view.setOpenExternalLinks(False)
        self.chat_view.setOpenLinks(False)
        self.chat_view.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.msg_edit.setMaximumHeight(150)
        self.model_box.setContextMenuPolicy(Qt.CustomContextMenu)

        # Configurer le bouton de pièce jointe
        attachment_icon = QIcon.fromTheme("document-open", QIcon(":/qt-project.org/styles/commonstyle/images/standardbutton-open-16.png"))
        if attachment_icon.isNull():
            self.attachment_btn.setText("📎")
        else:
            self.attachment_btn.setIcon(attachment_icon)
        self.attachment_btn.setToolTip("Joindre un fichier")
        self.attachment_btn.setFixedSize(36, 36)
        self.attachment_btn.setStyleSheet("QPushButton { border: none; background-color: transparent; }")

        # Mise en page
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.new_conv_btn)
        left_layout.addWidget(self.conv_list, 1)

        message_input_layout = QHBoxLayout()
        message_input_layout.addWidget(self.msg_edit, 1)
        message_input_layout.addWidget(self.attachment_btn)
        message_input_layout.addWidget(self.send_btn)

        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(self.model_box)
        bottom_bar.addWidget(self.model_settings_btn) # AJOUT DU BOUTON
        bottom_bar.addWidget(self.stats_label)
        bottom_bar.addStretch(1)
        bottom_bar.addWidget(self.res_label)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.addWidget(self.chat_view, 1)
        right_layout.addLayout(message_input_layout)
        
        # Ajout des nouveaux widgets à la mise en page droite
        self.toggle_files_panel_btn = QPushButton("Fichiers Attachés")
        self.toggle_files_panel_btn.setCheckable(True)
        self.toggle_files_panel_btn.setChecked(False)
        right_layout.addWidget(self.toggle_files_panel_btn)

        self.attachments_panel_widget = QWidget()
        attachments_panel_layout = QVBoxLayout(self.attachments_panel_widget)
        attachments_panel_layout.addWidget(self.attached_files_label)
        attachments_panel_layout.addWidget(self.attached_files_list)
        attachments_panel_layout.addWidget(self.remove_attachment_btn)
        self.attachments_panel_widget.setVisible(False)
        right_layout.addWidget(self.attachments_panel_widget)
        
        right_layout.addLayout(bottom_bar)

        splitter = QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 4)
        self.setCentralWidget(splitter)

        # Activer le drag & drop sur la fenêtre principale
        self.setAcceptDrops(True)
        # Activer le drag & drop sur le widget QTextEdit
        self.msg_edit.setAcceptDrops(True)
        # Remplacer les méthodes pour gérer le drag & drop
        self.msg_edit.dragEnterEvent = self._msg_edit_drag_enter
        self.msg_edit.dropEvent = self._msg_edit_drop

        # Style CSS
        self.chat_view.document().setDefaultStyleSheet("""
            body { 
                font-family: "Segoe UI", Tahoma, sans-serif;
                line-height: 1.6; 
                background-color: #1e1e1e;
                color: #e4e4e4;
                margin: 0;
                padding: 10px;
            }
            .message-container { 
                margin-bottom: 20px;
                max-width: 85%;
            }
            .message-user {
                margin-left: auto;
                margin-right: 0;
            }
            .message-assistant {
                margin-left: 0;
                margin-right: auto;
            }
            .role-user { 
                font-weight: 600; 
                color: #4fc3f7;
                font-size: 0.85em;
                margin-bottom: 6px;
                display: block;
            }
            .role-assistant { 
                font-weight: 600; 
                color: #66bb6a;
                font-size: 0.85em;
                margin-bottom: 6px;
                display: block;
            }
            .message-body {
                padding: 12px 16px;
                border-radius: 18px;
                display: block;
                white-space: pre-wrap;
                word-wrap: break-word;
                box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                position: relative;
            }
            .message-user .message-body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #ffffff;
                border-bottom-right-radius: 6px;
            }
            .message-assistant .message-body {
                background-color: #2d2d2d;
                color: #e4e4e4;
                border-bottom-left-radius: 6px;
            }
            .code-block-wrapper {
                position: relative; 
                margin: 12px 0;
                border-radius: 8px; 
                background-color: #2a2a2a; 
            }
            .code-block-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 6px 10px;
            }
            .code-lang {
                font-size: 0.85em;
                color: #ccc;
                font-family: "Segoe UI", Tahoma, sans-serif;
            }
            .copy-btn {
                text-decoration: none;
                color: #aaa;
                font-size: 1.1em; /* Ajuster pour la taille de l'icône/texte */
                padding: 2px 5px;
                border-radius: 4px;
                cursor: pointer;
            }
            .copy-btn:hover {
                color: #fff;
                background-color: #444;
            }
            .code-block {
                background-color: #1a1a1a; /* Fond spécifique au bloc de code */
                color: #f8f8f2;
                border-top-left-radius: 0px; 
                border-top-right-radius: 0px;
                border-bottom-left-radius: 7px; /* Ajuster pour s'aligner avec le wrapper */
                border-bottom-right-radius: 7px; /* Ajuster pour s'aligner avec le wrapper */
                padding: 14px;
                margin: 0; /* Le wrapper gère la marge extérieure */
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 0.9em;
                overflow-x: auto;
                white-space: pre; /* Important pour conserver les espaces et sauts de ligne du code */
            }
            .think-header { 
                background-color: #2a2a2a;
                padding: 8px 12px;
                border-radius: 6px;
                margin: 8px 0 4px 0;
                border-left: 3px solid #ffa726;
            }
            .think-header a { 
                text-decoration: none; 
                color: #ffa726;
                font-weight: bold;
                margin-right: 8px;
            }
            .think-header span { 
                font-style: italic; 
                color: #ffa726;
                font-size: 0.9em;
            }
            .think-block {
                background-color: #252525;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 12px;
                margin: 4px 0 8px 20px;
                font-size: 0.9em;
                color: #cccccc;
            }
            .file-header { 
                background-color: #2a2a2a;
                padding: 8px 12px;
                border-radius: 6px;
                margin: 8px 0 4px 0;
                border-left: 3px solid #26c6da;
            }
            .file-header a { 
                text-decoration: none; 
                color: #26c6da;
                font-weight: bold;
                margin-right: 8px;
            }
            .file-header span { 
                font-style: italic; 
                color: #26c6da;
                font-size: 0.9em;
            }
            .file-block {
                background-color: #252525;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 12px;
                margin: 4px 0 8px 20px;
                font-size: 0.85em;
                color: #cccccc;
                max-height: 300px;
                overflow-y: auto;
            }
            hr { 
                display: none;
            }
            b { color: #ffffff; font-weight: 600; }
            i { color: #b0b0b0; }
        """)

    def _setup_connections(self):
        self.new_conv_btn.clicked.connect(self.new_conversation)
        self.conv_list.itemClicked.connect(self.switch_conversation)
        self.conv_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.conv_list.customContextMenuRequested.connect(self._show_conv_context_menu)
        self.del_conv_action = QAction("Supprimer conversation", self)
        self.del_conv_action.triggered.connect(self.delete_conversation)
        self.msg_edit.textChanged.connect(self._auto_resize)
        self.send_btn.clicked.connect(self.send_message)
        self.attachment_btn.clicked.connect(self._handle_attachment)
        self.model_box.currentTextChanged.connect(self.change_model)
        self.model_box.customContextMenuRequested.connect(self._show_model_context_menu)
        self.model_settings_btn.clicked.connect(self._open_model_visibility_settings)
        self.chat_view.anchorClicked.connect(self._anchor_clicked)
        self.msg_edit.keyPressEvent = self._key_press_override
        self.toggle_files_panel_btn.clicked.connect(self._toggle_attachments_panel)
        self.remove_attachment_btn.clicked.connect(self._remove_selected_attachment)

    def _key_press_override(self, event):
        # Ctrl+Entrée pour envoyer, sinon comportement normal
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and (event.modifiers() & Qt.ControlModifier):
            self.send_message()
        else:
            QTextEdit.keyPressEvent(self.msg_edit, event)

    def _show_model_context_menu(self, pos):
        menu = QMenu(self)
        # Exemple : ajouter une action pour rafraîchir la liste des modèles
        refresh_action = QAction("Rafraîchir la liste des modèles", self)
        refresh_action.triggered.connect(self._populate_model_box)
        menu.addAction(refresh_action)
        menu.exec(self.model_box.mapToGlobal(pos))

    def _auto_resize(self):
        doc = self.msg_edit.document()
        doc_height = doc.size().height()
        min_height = 40
        max_height = 150
        new_height = min(max(doc_height + 10, min_height), max_height)
        self.msg_edit.setFixedHeight(new_height)

    def _msg_edit_drag_enter(self, event: QDragEnterEvent):
        """Acceptation si au moins un fichier local est glissé."""
        if event.mimeData().hasUrls():
            # Accepter si au moins un fichier local est présent
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    event.acceptProposedAction()
                    return
        # Si c'est du texte standard, on laisse le comportement par défaut
        elif event.mimeData().hasText():
            QTextEdit.dragEnterEvent(self.msg_edit, event)
            return
        event.ignore()

    def _msg_edit_drop(self, event: QDropEvent):
        """Traiter chaque fichier glissé comme pièce jointe."""
        if event.mimeData().hasUrls():
            # Parcourir tous les fichiers glissés
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = Path(url.toLocalFile())
                    if file_path.exists() and file_path.is_file():
                        self._process_file_attachment(file_path)
            event.acceptProposedAction()
            return
        # Si c'est du texte standard, on laisse le comportement par défaut
        elif event.mimeData().hasText():
            QTextEdit.dropEvent(self.msg_edit, event)
            return
        event.ignore()

    def _start_stats_timer(self):
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._update_resource_stats)
        self.stats_timer.start(1000)

    def _update_resource_stats(self):
        self.res_label.setText(f"CPU: {psutil.cpu_percent():.0f}%  RAM: {psutil.virtual_memory().percent:.0f}%")

    def _load_model_favorites(self) -> list[str]:
        try:
            if not hasattr(self, 'fav_file'):
                # Correction : définir le chemin du fichier favoris si absent
                self.fav_file = SAVE_DIR / "model_favorites.json"
            if not self.fav_file.exists():
                return []
            data = json.loads((self.fav_file).read_text(encoding='utf-8'))
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _save_model_favorites(self):
        try:
            if not hasattr(self, 'fav_file'):
                # Correction : définir le chemin du fichier favoris si absent
                self.fav_file = SAVE_DIR / "model_favorites.json"
            (self.fav_file).write_text(json.dumps(self.favorites, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            logging.error(f"Erreur sauvegarde favoris : {e}", exc_info=True)

    def _populate_model_box(self):
        self.favorites = self._load_model_favorites()
        models_discovered = []
        self.model_capabilities.clear()
        
        if not is_ollama_running():
            self.ollama_available = False
        else:
            self.ollama_available = True
            
        if self.ollama_available:
            try:
                ollama_models_list = self.client.list_models()
                models_discovered.extend(ollama_models_list)
            except Exception:
                ollama_models_list = []
                
            if not ollama_models_list:
                models_discovered.extend(DEFAULT_MODELS)
                ollama_models_list = list(DEFAULT_MODELS) 
            
            for model_name in ollama_models_list:
                known_ollama_vision_models = ["gemma3:4b-it-qat", "gemma3:12b-it-qat","gpt-4.1"]
                is_vision_model = (
                    "llava" in model_name.lower() or 
                    model_name.lower() in [m.lower() for m in known_ollama_vision_models]
                )
                self.model_capabilities[model_name] = ModelCaps(
                    name=model_name,
                    supports_images=is_vision_model,
                    supports_general_files=False,
                    max_tokens=4096
                )

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                oai_client = OpenAI()
                response = oai_client.models.list()
                openai_models_ids = [model.id for model in response.data 
                               if "gpt" in model.id.lower() or "o3" in model.id.lower()]
                               
                if openai_models_ids:
                    for model_id in openai_models_ids:
                        supports_vision = "vision" in model_id.lower()
                        is_turbo = "turbo" in model_id.lower()
                        max_tokens = 128000 if "gpt-4" in model_id.lower() and is_turbo else (16385 if "gpt-3.5-turbo-16k" in model_id.lower() else 8192 if "gpt-4" in model_id.lower() else 4096)
                        full_model_name = f"OpenAI: {model_id}"
                        self.model_capabilities[full_model_name] = ModelCaps(
                            name=full_model_name,
                            supports_images=supports_vision,
                            supports_general_files=True,
                            max_tokens=max_tokens
                        )
                        models_discovered.append(full_model_name)
            except Exception:
                pass
        
        all_models_available_after_discovery = sorted(list(dict.fromkeys(models_discovered)))

        models_to_list_in_box = []
        if self.visible_models is None:
            models_to_list_in_box = list(all_models_available_after_discovery)
        else:
            models_to_list_in_box = [m for m in all_models_available_after_discovery if m in self.visible_models]

        if not all_models_available_after_discovery:
            QMessageBox.warning(
                self,
                "Aucun modèle disponible",
                "Aucun modèle n'a été trouvé. Assurez-vous qu'Ollama est en cours d'exécution ou que votre clé API OpenAI est valide."
            )
        elif not models_to_list_in_box and all_models_available_after_discovery:
             QMessageBox.information(self, "Modèles filtrés", "Tous les modèles disponibles sont actuellement masqués par vos paramètres d'affichage.")
            
        favs_in_visible = [m for m in self.favorites if m in models_to_list_in_box]
        oth_in_visible = sorted([m for m in models_to_list_in_box if m not in favs_in_visible])
        
        prev = self.current_model
        
        self.model_box.clear()
        for m in favs_in_visible:
            self.model_box.addItem(f"★ {m}", m)
        for m in oth_in_visible:
            self.model_box.addItem(m, m)
            
        if prev and prev in models_to_list_in_box:
            i = self.model_box.findData(prev)
            if i >= 0:
                self.model_box.setCurrentIndex(i)
        elif self.model_box.count() > 0:
            self.model_box.setCurrentIndex(0)

        if self.model_box.count() > 0:
             self.change_model(self.model_box.currentText())
        else:
             self.change_model(None)

    def new_conversation(self):
        cid = str(uuid.uuid4())
        self.conversations[cid] = []
        self.current_conv_id = cid
        item = QListWidgetItem("Nouvelle conversation")
        item.setData(Qt.UserRole, cid)
        self.conv_list.addItem(item)
        self.conv_list.setCurrentItem(item)
        self.chat_view.clear()
        self.reason_states.clear()
        self.pending_attachments.clear()
        self.attached_files_list.clear() # Vider aussi la liste UI

    def switch_conversation(self, item: QListWidgetItem):
        self.current_conv_id = item.data(Qt.UserRole)
        self.reason_states.clear()
        self.pending_attachments.clear()
        self.attached_files_list.clear() # Vider aussi la liste UI
        self.render_conversation() # Ajouter cette ligne pour rafraîchir l'affichage

    def change_model(self, _):
        idx = self.model_box.currentIndex()
        if idx < 0 or self.model_box.itemData(idx) is None or self.model_box.itemData(idx) == "Aucun modèle disponible":
            self.current_model = None
            self.send_btn.setEnabled(False)
            self.attachment_btn.setEnabled(False)
        else:
            model_name = self.model_box.itemData(idx)
            self.current_model = model_name
            self.send_btn.setEnabled(True)
            self.attachment_btn.setEnabled(True)

    def send_message(self):
        if not self.current_model:
            QMessageBox.warning(self, "Aucun modèle", "Veuillez sélectionner un modèle avant d'envoyer un message.")
            return

        txt = self.msg_edit.toPlainText().strip()
        if not txt and not self.pending_attachments:
            QMessageBox.warning(self, "Message vide", "Veuillez écrire un message ou joindre un fichier.")
            return
        
        # Nettoyer le placeholder du fichier dans le texte si on envoie vraiment quelque chose
        user_text_input = re.sub(r"\n\[(Image|PDF|Code|Texte).*?:.*?\s*\]", "", txt).strip()
        self.msg_edit.clear()

        conv = self.conversations[self.current_conv_id]
        
        # Construction du contenu du message utilisateur
        user_message_parts = []
        attachments_description_for_history = ""

        # D'abord, le contenu des fichiers joints si présents
        if self.pending_attachments:
            for pa_index, pa in enumerate(self.pending_attachments):
                attachments_description_for_history += f" (Fichier joint: {pa.get('original_filename', 'inconnu')})"
                if pa['type'] == 'text_content':
                    # Générer un id unique pour chaque fichier (par exemple hash du nom+contenu)
                    file_id = hashlib.sha1((pa.get('original_filename','') + pa['content']).encode('utf-8')).hexdigest()[:8]
                    user_message_parts.append(
                        f"<filedata id=\"{file_id}\" name=\"{pa.get('original_filename', 'Fichier sans nom')}\">{pa['content']}</filedata>"
                    )

        # Ensuite, le texte tapé par l'utilisateur
        if user_text_input:
            if user_message_parts: # S'il y avait déjà des fichiers
                user_message_parts.append(f"\nQuestion de l'utilisateur concernant les fichiers ci-dessus et/ou autre sujet :\n{user_text_input}")
            else: # Juste le texte de l'utilisateur
                user_message_parts.append(user_text_input)
        elif not user_message_parts: # Ni texte, ni contenu de fichier pertinent
            QMessageBox.warning(self, "Message vide", "Veuillez écrire un message ou joindre un fichier avec du contenu textuel.")
            self.send_btn.setEnabled(True)
            self.msg_edit.setReadOnly(False)
            return
        elif user_message_parts and not user_text_input: # Fichiers, mais pas de texte utilisateur explicite
            user_message_parts.append("\n\nExpliquez le contenu des fichiers fournis ci-dessus.")

        final_user_content = "\n\n".join(user_message_parts).strip()

        conv.append(Message("user", final_user_content)) # Utiliser le contenu final formaté
        
        if len(conv) == 1:
            title_basis = user_text_input if user_text_input else final_user_content
            self.conv_list.currentItem().setText(title_basis.split("\n", 1)[0][:40])
        self.render_conversation()
        self.send_btn.setEnabled(False) # Désactiver pendant la réponse
        self.msg_edit.setReadOnly(True) # Rendre msg_edit non modifiable pendant la réponse

        # Le payload API utilisera directement les messages de `conv` qui inclut maintenant le message utilisateur complet
        api_messages_payload_list = [{"role": m.role, "content": m.content} for m in conv]

        # Préparer le payload final pour Ollama si besoin
        final_api_payload_for_worker = api_messages_payload_list
        is_openai_call = self.current_model.startswith("OpenAI:")

        if not is_openai_call: # Ollama
            ollama_specific_payload = {
                "model": self.current_model,
                "messages": api_messages_payload_list, # Contient déjà le message utilisateur complet
                "stream": False
            }
            final_api_payload_for_worker = ollama_specific_payload
        
        worker_client = self.client if not is_openai_call else None 
        worker = ApiWorker(worker_client, self.current_model, final_api_payload_for_worker, is_openai_call)
        
        worker.signals.finished.connect(self._handle_api_response)
        worker.signals.error.connect(self._handle_api_error)
        
        self.pending_attachments.clear()
        self.attached_files_list.clear()

        self.threadpool.start(worker)

    def _handle_api_response(self, result):
        role, content, total_tokens, tok_s, model_used = result
        conv = self.conversations[self.current_conv_id]
        conv.append(Message(role, content, total_tokens, tok_s, model=model_used))
        self.stats_label.setText(f"Tokens: {total_tokens} – {tok_s:.1f} tok/s")
        self.render_conversation()
        self._save()
        self.send_btn.setEnabled(True) # Réactiver après réponse
        self.msg_edit.setReadOnly(False) # Rendre msg_edit modifiable

    def _handle_api_error(self, error_message):
        QMessageBox.critical(self, "Erreur API", str(error_message))
        self.send_btn.setEnabled(True) # Réactiver même en cas d'erreur
        self.msg_edit.setReadOnly(False) # Rendre msg_edit modifiable
    
    def _update_stats_label(self, stats_text):
        self.stats_label.setText(stats_text)

    def _format_assistant(self, txt: str) -> str:
        def _repl(m: re.Match) -> str:
            content = m.group(1).strip()
            rid = str(uuid.uuid5(uuid.NAMESPACE_OID, content))[:8]
            expanded = self.reason_states.get(rid, False)
            self.reason_states.setdefault(rid, False)
            arrow = "▼" if expanded else "▶"
            header = f'<div class="think-header"><a href="reason:{rid}">{arrow}</a> <span>Thoughts</span></div>'
            if not expanded:
                return header
            safe = html.escape(content)
            box = f'<div class="think-block"><pre>{safe}</pre></div>'
            return header + box

        txt = THINK_RE.sub(_repl, txt)

        def code_repl(m: re.Match) -> str:
            code_id = f"codeblock_{uuid.uuid4().hex[:8]}"
            lang = m.group(1)
            code_content = m.group(2).rstrip('\n') 
            
            self.code_block_contents[code_id] = code_content

            escaped_code_content = html.escape(code_content)
            lang_display = html.escape(lang if lang else "code")

            return (
                f'<div class="code-block-wrapper">'
                f'  <div class="code-block-header">'
                f'    <span class="code-lang">{lang_display}</span>'
                f'    <a href="copycode:{code_id}" class="copy-btn" title="Copier le code">📄</a>'
                f'  </div>'
                f'  <pre id="{code_id}" class="code-block">{escaped_code_content}</pre>'
                f'</div>'
            )

        txt = re.sub(
            r"```(\w*)?\n(.*?)\n?```", 
            code_repl,
            txt,
            flags=re.DOTALL,
        )

        txt = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", txt)
        txt = re.sub(r"\*(.*?)\*", r"<i>\1</i>", txt)

        # Extraction des contenus de fichiers cachés
        file_data_map = {}
        for m in re.finditer(r"<filedata id=\"(.*?)\">([\s\S]*?)</filedata>", txt):
            file_data_map[m.group(1)] = m.group(2)
        return txt

    def render_conversation(self):
        if not self.current_conv_id:
            return
        conv = self.conversations.get(self.current_conv_id, [])
        html_content = "<body>"
        for m in conv:
            role_class = f"role-{m.role}"
            role_text = "Utilisateur" if m.role == "user" else "Assistant"
            message_class = "message-user" if m.role == "user" else "message-assistant"
            
            body_raw = m.content

            if m.role == "user":
                final_html_parts = []
                last_idx = 0
                
                for match in re.finditer(r"<filedata id=\"(.*?)\"(?: name=\"([^\"]+)\")?>([\s\S]*?)</filedata>", body_raw):
                    pre_match_text = body_raw[last_idx:match.start()]
                    final_html_parts.append(html.escape(pre_match_text).replace("\n", "<br>"))
                    
                    file_id = match.group(1)
                    file_name = match.group(2) if match.group(2) else "Fichier joint"
                    raw_file_content = match.group(3)

                    expanded = self.reason_states.get(f"userfile_{file_id}", False)
                    arrow = "▼" if expanded else "▶"
                    
                    safe_file_name = html.escape(file_name)
                    header_html = f'<div class="file-header"><a href="userfile:{file_id}">{arrow}</a> <span>{safe_file_name} (cliquer pour afficher/masquer)</span></div>'
                    final_html_parts.append(header_html)
                    
                    if expanded:
                        escaped_file_content = html.escape(raw_file_content)
                        file_block_html = f'<div class="file-block"><pre>{escaped_file_content}</pre></div>'
                        final_html_parts.append(file_block_html)
                    
                    last_idx = match.end()
                
                post_match_text = body_raw[last_idx:]
                final_html_parts.append(html.escape(post_match_text).replace("\n", "<br>"))
                
                body_formatted = "".join(final_html_parts)
            else:
                body_formatted = self._format_assistant(body_raw)

            html_content += (
                f'<div class="message-container {message_class}">'
                f'<span class="{role_class}">{role_text}:</span>'
                f'<div class="message-body">{body_formatted}</div>'
                f'</div>'
            )

        html_content += "</body>"
        self.chat_view.setHtml(html_content)
        self.chat_view.moveCursor(QTextCursor.End)

    def _anchor_clicked(self, url: QUrl):
        scheme = url.scheme()
        if scheme == "reason":
            rid = url.path() or url.opaque()
            if rid:
                self.reason_states[rid] = not self.reason_states.get(rid, False)
                self.render_conversation()
            else:
                logging.warning(f"Could not extract reason ID from URL: {url.toString()}")
        elif scheme == "file":
            file_id = url.path() or url.opaque()
            if file_id:
                key = f"file_{file_id}"
                self.reason_states[key] = not self.reason_states.get(key, False)
                self.render_conversation()
        elif scheme == "userfile":
            file_id = url.path() or url.opaque()
            if file_id:
                key = f"userfile_{file_id}"
                self.reason_states[key] = not self.reason_states.get(key, False)
                self.render_conversation()
        elif scheme == "copycode":
            code_block_id = url.path() or url.opaque()
            if code_block_id and code_block_id in self.code_block_contents:
                text_to_copy = self.code_block_contents[code_block_id]
                clipboard = QApplication.instance().clipboard()
                if clipboard:
                    clipboard.setText(text_to_copy)
                    QToolTip.showText(QCursor.pos(), "Code copié !", self.chat_view, self.chat_view.rect(), 2000)
                else:
                    logging.error("Impossible d'accéder au presse-papiers.")
            elif code_block_id:
                logging.warning(f"Contenu du bloc de code non trouvé pour ID: {code_block_id}")

        elif scheme in ["http", "https"]:
            QDesktopServices.openUrl(url)

    def _save(self):
        if not self.current_conv_id:
            return
        try:
            path = SAVE_DIR / f"{self.current_conv_id}.json"
            conv_data = [asdict(m) for m in self.conversations.get(self.current_conv_id, [])]
            path.write_text(json.dumps(conv_data, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            logging.error(f"Error saving conversation {self.current_conv_id}: {e}", exc_info=True)

    def load_conversations(self):
        self.conversations = {}
        self.conv_list.clear()
        loaded_items = []
        for file_path in SAVE_DIR.glob("*.json"):
            if file_path.name == "model_favorites.json" or file_path.name == "model_visibility.json": # MODIFIÉ ICI
                continue
            try:
                cid = file_path.stem
                content = file_path.read_text(encoding='utf-8')
                conv_data = json.loads(content)
                messages = [Message(**m_data) for m_data in conv_data]
                self.conversations[cid] = messages

                title = "Conversation"
                if messages and messages[0].role == 'user':
                    title = messages[0].content.split('\n', 1)[0][:40]
                elif messages and messages[0].role == 'assistant' and len(messages) > 1 and messages[1].role == 'user': # Correction: len(messages) > 1
                     title = messages[1].content.split('\n', 1)[0][:40]

                item = QListWidgetItem(title)
                item.setData(Qt.UserRole, cid)
                loaded_items.append(item)
            except Exception as e:
                logging.error(f"Error loading conversation {file_path.name}: {e}", exc_info=True)

        loaded_items.sort(key=lambda x: x.text())
        for item in loaded_items:
            self.conv_list.addItem(item)

    def _show_conv_context_menu(self, pos):
        item = self.conv_list.itemAt(pos)
        if item:
            menu = QMenu(self)
            menu.addAction(self.del_conv_action)
            menu.exec(self.conv_list.mapToGlobal(pos))

    def delete_conversation(self):
        item = self.conv_list.currentItem()
        if not item:
            return
        cid = item.data(Qt.UserRole)
        reply = QMessageBox.question(
            self,
            "Supprimer",
            "Supprimer cette conversation ?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                (SAVE_DIR / f"{cid}.json").unlink()
            except Exception:
                pass
            self.conversations.pop(cid, None)
            row = self.conv_list.row(item)
            self.conv_list.takeItem(row)
            if self.current_conv_id == cid:
                if self.conv_list.count():
                    self.conv_list.setCurrentRow(0)
                    self.switch_conversation(self.conv_list.currentItem())
                else:
                    self.new_conversation()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = Path(url.toLocalFile())
                    if file_path.exists() and file_path.is_file():
                        self._process_file_attachment(file_path)
            event.acceptProposedAction()
            return
        event.ignore()

    def _handle_attachment(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            for file_path in selected_files:
                self._process_file_attachment(Path(file_path))

    def _toggle_attachments_panel(self):
        visible = self.toggle_files_panel_btn.isChecked()
        self.attachments_panel_widget.setVisible(visible)

    def _remove_selected_attachment(self):
        selected_items = self.attached_files_list.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            row = self.attached_files_list.row(item)
            file_display_name = item.text()
            
            original_filename_to_remove = None
            match = re.match(r"^(.*?)\s+\(", file_display_name)
            if match:
                original_filename_to_remove = match.group(1)

            if original_filename_to_remove:
                for i, att in enumerate(self.pending_attachments):
                    if att.get('original_filename') == original_filename_to_remove:
                        del self.pending_attachments[i]
                        break 
            
            self.attached_files_list.takeItem(row)
        
        self._update_msg_edit_with_attachments()

    def _process_file_attachment(self, file_path: Path):
        mime_type, _ = mimetypes.guess_type(str(file_path))
        original_filename = file_path.name
        file_info = {
            "original_filename": original_filename,
            "type": None,
            "content": None,
            "path": str(file_path)
        }

        if mime_type and mime_type.startswith("text"):
            try:
                content = read_text_file(file_path)
                file_info["type"] = "text_content"
                file_info["content"] = content
            except Exception as e:
                QMessageBox.warning(self, "Erreur fichier", f"Impossible de lire le fichier texte : {e}")
                return
        elif mime_type and "pdf" in mime_type:
            try:
                content = extract_text_from_pdf(file_path)
                file_info["type"] = "text_content"
                file_info["content"] = content
            except Exception as e:
                QMessageBox.warning(self, "Erreur PDF", f"Impossible de lire le PDF : {e}")
                return
        elif mime_type and mime_type.startswith("image"):
            try:
                content = ocr_image(file_path)
                file_info["type"] = "text_content"
                file_info["content"] = content
            except Exception as e:
                QMessageBox.warning(self, "Erreur image", f"Impossible de traiter l'image : {e}")
                return
        else:
            QMessageBox.warning(self, "Type non supporté", "Seuls les fichiers texte, PDF et images sont supportés.")
            return

        self.pending_attachments.append(file_info)
        size_kb = file_path.stat().st_size // 1024 + 1
        item_text = f"{file_info['original_filename']} ({size_kb} KB)"
        self.attached_files_list.addItem(item_text)