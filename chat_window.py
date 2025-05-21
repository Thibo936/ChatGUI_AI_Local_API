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
from PySide6.QtGui import QAction, QTextCursor, QTextOption, QDesktopServices, QIcon, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
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
        self.pending_attachments: list[dict] = [] # MODIFIÉ: anciennement self.pending_attachment
        self.file_content_map = {}  # Nouveau : {file_id: (nom, contenu)}

        self.fav_file = SAVE_DIR / "model_favorites.json"
        self.favorites = self._load_model_favorites()

        self.attachments_panel_widget = None
        self.toggle_files_panel_btn = None

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
                font-family: "Segoe UI", sans-serif; /* Police plus moderne */
                line-height: 1.5; 
                background-color: #2E3440; /* Fond sombre */
                color: #D8DEE9; /* Texte clair */
            }
            .message-container { 
                margin-bottom: 12px; 
            }
            .role-user { 
                font-weight: bold; 
                color: #88C0D0; /* Bleu clair pour l'utilisateur */
                font-size: 0.9em;
                margin-left: 5px;
            }
            .role-assistant { 
                font-weight: bold; 
                color: #A3BE8C; /* Vert clair pour l'assistant */
                font-size: 0.9em;
                margin-left: 5px;
            }
            .message-body {
                padding: 10px 15px;
                border-radius: 15px; /* Coins plus arrondis */
                margin-top: 4px;
                display: inline-block;
                max-width: 90%; /* Légère réduction pour l'esthétique */
                white-space: pre-wrap;
                word-wrap: break-word;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* Ombre subtile */
            }
            /* Style spécifique pour les messages utilisateur */
            .message-user .message-body {
                background-color: #3B4252; /* Fond légèrement différent pour l'utilisateur */
                border-bottom-right-radius: 5px; /* Style "bulle" */
            }
            /* Style spécifique pour les messages assistant */
            .message-assistant .message-body {
                background-color: #434C5E; /* Fond pour l'assistant */
                border-bottom-left-radius: 5px; /* Style "bulle" */
            }
            .code-block {
                background-color: #23272e; /* Fond plus sombre pour le code */
                color: #D8DEE9; /* Texte clair pour le code */
                border: 1px solid #4C566A; /* Bordure discrète */
                padding: 12px;
                margin: 10px 0;
            }
            .think-header a { text-decoration: none; color: #EBCB8B; } /* Jaune pour les liens "think" */
            .think-header span { font-style: italic; color: #EBCB8B; }
            .think-block {
                border: 1px dashed #4C566A; /* Bordure discrète */
                background-color: #3B4252; /* Fond pour les pensées */
                padding: 10px;
                margin: 6px 0 6px 25px; /* Marge ajustée */
            }
            .file-header a { text-decoration: none; color: #8FBCBB; } /* Sarcelle pour les liens fichiers */
            .file-header span { font-style: italic; color: #8FBCBB; }
            .file-block {
                border: 1px dashed #4C566A; /* Bordure discrète */
                background-color: #3B4252; /* Fond pour les blocs fichiers */
                padding: 10px;
                margin: 6px 0 6px 25px; /* Marge ajustée */
            }
            hr { border: 0; height: 1px; background-color: #4C566A; margin: 18px 0; } /* Séparateur plus discret */
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
        self.chat_view.anchorClicked.connect(self._anchor_clicked)
        self.msg_edit.keyPressEvent = self._key_press_override
        self.toggle_files_panel_btn.clicked.connect(self._toggle_attachments_panel)
        self.remove_attachment_btn.clicked.connect(self._remove_selected_attachment)

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
            data = json.loads((self.fav_file).read_text(encoding='utf-8'))
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _save_model_favorites(self):
        try:
            (self.fav_file).write_text(json.dumps(self.favorites, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            logging.error(f"Erreur sauvegarde favoris : {e}", exc_info=True)

    def _populate_model_box(self):
        self.favorites = self._load_model_favorites()
        models = []
        self.model_capabilities.clear()
        
        # Vérifier si Ollama est disponible
        if not is_ollama_running():
            self.ollama_available = False
            logging.warning("Ollama n'est pas en cours d'exécution")
        else:
            self.ollama_available = True
            
        if self.ollama_available:
            try:
                models = self.client.list_models()
                logging.info(f"Modèles Ollama récupérés: {models}")
            except Exception as e:
                models = []
                logging.error(f"Erreur lors de la récupération des modèles Ollama: {e}", exc_info=True)
                
        # Si aucun modèle n'a été récupéré, ajouter les modèles par défaut
        if not models and self.ollama_available:
            models = DEFAULT_MODELS
            logging.info(f"Utilisation des modèles par défaut: {models}")
                
        # Remplir les capacités pour les modèles Ollama (valeurs par défaut pour l'instant)
        for model_name in models:
            # Heuristique améliorée pour les modèles vision Ollama
            known_ollama_vision_models = ["gemma3:4b-it-qat", "gemma3:12b-it-qat","gpt-4.1"] # Modèles explicitement connus pour supporter les images
            is_vision_model = (
                "llava" in model_name.lower() or 
                model_name.lower() in [m.lower() for m in known_ollama_vision_models]
            )
            self.model_capabilities[model_name] = ModelCaps(
                name=model_name,
                supports_images=is_vision_model,
                supports_general_files=False, # À affiner
                max_tokens=4096 # À affiner, peut-être via ollama show
            )

        # Essayer de récupérer les modèles OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                logging.info("Tentative de récupération des modèles OpenAI...")
                
                # Créer un client avec l'API key
                client = OpenAI()  # La clé est déjà dans les variables d'environnement
                
                # Récupérer la liste des modèles
                response = client.models.list()
                logging.info(f"Réponse brute d'OpenAI: {response}")
                
                # Filtrer les modèles pour ne garder que ceux qui contiennent "gpt" ou "o3"
                openai_models = [model.id for model in response.data 
                               if "gpt" in model.id.lower() or "o3" in model.id.lower()]
                               
                if openai_models:
                    logging.info(f"Modèles OpenAI récupérés: {openai_models}")
                    # Remplir les capacités pour les modèles OpenAI
                    for model_id in openai_models:
                        # Heuristiques pour les modèles GPT vision et contextes longs
                        supports_vision = "vision" in model_id.lower()
                        # GPT-4 Turbo a généralement un contexte plus large
                        is_turbo = "turbo" in model_id.lower()
                        max_tokens = 128000 if "gpt-4" in model_id.lower() and is_turbo else (16385 if "gpt-3.5-turbo-16k" in model_id.lower() else 8192 if "gpt-4" in model_id.lower() else 4096)

                        full_model_name = f"OpenAI: {model_id}"
                        self.model_capabilities[full_model_name] = ModelCaps(
                            name=full_model_name,
                            supports_images=supports_vision,
                            supports_general_files=True, # Les modèles GPT peuvent gérer du texte de fichiers
                            max_tokens=max_tokens
                        )
                        models.append(full_model_name) # Ajouter avec le préfixe
                else:
                    logging.warning("Aucun modèle OpenAI correspondant trouvé dans la réponse")
            except Exception as e:
                logging.error(f"Erreur lors de la récupération des modèles OpenAI: {e}", exc_info=True)
        else:
            logging.warning("Pas de clé API OpenAI trouvée dans les variables d'environnement")
                
        # Si toujours aucun modèle disponible, afficher un message
        if not models:
            QMessageBox.warning(
                self,
                "Aucun modèle disponible",
                "Aucun modèle n'a été trouvé. Assurez-vous qu'Ollama est en cours d'exécution ou que votre clé API OpenAI est valide."
            )
            # Ajouter un modèle factice pour éviter les erreurs
            models = ["Aucun modèle disponible"]
            
        # Dédupliquer et trier les modèles
        models = list(dict.fromkeys(models))
        favs = [m for m in self.favorites if m in models]
        oth = sorted([m for m in models if m not in self.favorites])
        
        # Sauvegarder le modèle courant
        prev = self.current_model
        
        # Remplir la combobox
        self.model_box.clear()
        for m in favs:
            self.model_box.addItem(f"★ {m}", m)
        for m in oth:
            self.model_box.addItem(m, m)
            
        # Restaurer le modèle précédent s'il existe encore
        if prev:
            i = self.model_box.findData(prev)
            if i >= 0:
                self.model_box.setCurrentIndex(i)
        elif self.model_box.count() > 0:
            # Sélectionner le premier modèle par défaut
            self.model_box.setCurrentIndex(0)

        # Désactiver le bouton Envoyer initialement, sera activé si un modèle est chargé
        self.send_btn.setEnabled(False)
        self.attachment_btn.setEnabled(False)

    def _show_model_context_menu(self, pos):
        idx = self.model_box.currentIndex()
        if idx < 0:
            return
        model_name = self.model_box.itemData(idx)
        if not model_name:
            return
        menu = QMenu(self)
        if model_name in self.favorites:
            action = menu.addAction("Retirer des favoris")
        else:
            action = menu.addAction("Ajouter aux favoris")
        action.triggered.connect(lambda: self._toggle_favorite(model_name))
        menu.exec(self.model_box.mapToGlobal(pos))

    def _toggle_favorite(self, model_name: str):
        if model_name in self.favorites:
            self.favorites.remove(model_name)
        else:
            self.favorites.append(model_name)
        self._save_model_favorites()
        prev = self.model_box.currentText()
        self._populate_model_box()
        idx = self.model_box.findText(prev)
        if idx >= 0:
            self.model_box.setCurrentIndex(idx)

    def _auto_resize(self):
        h = self.msg_edit.document().size().height() + 10
        self.msg_edit.setFixedHeight(min(int(h), 150))

    def _key_press_override(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter) and e.modifiers() & Qt.ControlModifier:
            self.send_message()
        else:
            QTextEdit.keyPressEvent(self.msg_edit, e)

    def _handle_attachment(self):
        file_path_str, _ = QFileDialog.getOpenFileName(self, "Sélectionner un fichier", "", "Tous les fichiers (*);;Images (*.png *.jpg *.jpeg *.bmp *.gif);;Documents PDF (*.pdf);;Fichiers Texte (*.txt *.md);;Code (*.py *.js *.html *.css *.java *.c *.cpp)")
        if file_path_str:
            file_path = Path(file_path_str)
            self._process_file_attachment(file_path)

    def _process_file_attachment(self, file_path: Path):
        if not self.current_model:
            QMessageBox.warning(self, "Aucun modèle sélectionné", "Veuillez sélectionner un modèle avant de joindre un fichier.")
            return

        file_type = get_file_type(file_path)
        model_caps = self.model_capabilities.get(self.current_model)

        if not model_caps:
            QMessageBox.warning(self, "Capacités du modèle inconnues", f"Impossible de déterminer les capacités pour le modèle {self.current_model}.")
            return

        # Cas PDF
        if file_type == "pdf":
            text_content = extract_text_from_pdf(file_path)
            if text_content:
                attachment_data = {"type": "text_content", "content": text_content, "original_filename": file_path.name}
                if model_caps.supports_general_files or len(text_content) < model_caps.max_tokens * 2:
                    self.pending_attachments.append(attachment_data) # MODIFIÉ
                    self.attached_files_list.addItem(f"PDF: {file_path.name}") # Ajout à la liste UI
                else:
                    QMessageBox.information(self, "Contenu PDF", "Le contenu du PDF a été extrait, mais pourrait être trop long pour le modèle. Un résumé serait idéal ici.")
                    attachment_data['content'] = text_content[:model_caps.max_tokens*2] # MODIFIÉ
                    self.pending_attachments.append(attachment_data) # MODIFIÉ
                    self.attached_files_list.addItem(f"PDF: {file_path.name}") # Ajout à la liste UI
            else:
                QMessageBox.warning(self, "Erreur PDF", f"Impossible d'extraire le texte du PDF {file_path.name}.")

        # Cas Code ou Texte
        elif file_type in ["code", "text"]:
            text_content = read_text_file(file_path)
            if text_content:
                attachment_data = {"type": "text_content", "content": text_content, "original_filename": file_path.name}
                if model_caps.supports_general_files or len(text_content) < model_caps.max_tokens * 3:
                    self.pending_attachments.append(attachment_data) # MODIFIÉ
                    self.attached_files_list.addItem(f"{file_type.capitalize()}: {file_path.name}") # Ajout à la liste UI
                else:
                    QMessageBox.information(self, f"Contenu {file_type.capitalize()}", f"Le contenu du fichier {file_type} a été lu, mais pourrait être trop long. Un résumé/troncature serait idéal ici.")
                    attachment_data['content'] = text_content[:model_caps.max_tokens*3] # MODIFIÉ
                    self.pending_attachments.append(attachment_data) # MODIFIÉ
                    self.attached_files_list.addItem(f"{file_type.capitalize()}: {file_path.name}") # Ajout à la liste UI
            else:
                QMessageBox.warning(self, f"Erreur Fichier {file_type.capitalize()}", f"Impossible de lire le fichier {file_path.name}.")
        # Cas non géré
        else:
            QMessageBox.information(self, "Type de fichier non géré", f"Le fichier {file_path.name} de type '{file_type}' n'est pas encore géré ou est inconnu.")
            return

        if self.pending_attachments and not self.attachments_panel_widget.isVisible():
            self.toggle_files_panel_btn.setChecked(True)
            self.attachments_panel_widget.setVisible(True)

    def _toggle_attachments_panel(self):
        is_checked = self.toggle_files_panel_btn.isChecked()
        self.attachments_panel_widget.setVisible(is_checked)

    def _remove_selected_attachment(self):
        current_item = self.attached_files_list.currentItem()
        if current_item:
            row = self.attached_files_list.row(current_item)
            self.attached_files_list.takeItem(row)
            if 0 <= row < len(self.pending_attachments):
                del self.pending_attachments[row]
            if self.attached_files_list.count() == 0 and self.attachments_panel_widget.isVisible():
                self.toggle_files_panel_btn.setChecked(False)
                self.attachments_panel_widget.setVisible(False)

    def add_file_placeholder_to_message(self, file_path: Path, contextual_prefix: str):
        pass # Ne fait plus rien pour l'instant, géré par la liste

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
                    #file_intro = f"Contenu du fichier '{pa.get('original_filename', 'Fichier sans nom')}' :"
                    ## Encapsuler le contenu dans un tag filedata masqué par défaut
                    #user_message_parts.append(
                    #    f"{file_intro}\n<filedata id=\"{file_id}\">{pa['content']}</filedata>"
                    #)
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
            # La gestion des images pour Ollama (si c'était pour des fichiers binaires) est séparée.
            # Pour le contenu textuel des fichiers, il est déjà intégré dans `api_messages_payload_list`.
            final_api_payload_for_worker = ollama_specific_payload
        # Pour OpenAI, api_messages_payload_list est déjà au bon format si le modèle supporte les messages multiples.
        # Si le modèle OpenAI est multimodal (ex: vision), la structure du contenu du message utilisateur peut nécessiter un formatage spécifique (liste de dictionnaires type/text, type/image_url), ce qui n'est pas le cas ici car on ne traite que du texte.
        
        # Créer et démarrer le worker
        worker_client = self.client if not is_openai_call else None 
        worker = ApiWorker(worker_client, self.current_model, final_api_payload_for_worker, is_openai_call)
        
        worker.signals.finished.connect(self._handle_api_response)
        worker.signals.error.connect(self._handle_api_error)
        # worker.signals.stats_updated.connect(self._update_stats_label) # Déjà géré dans _handle_api_response
        
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
            code = m.group(2).rstrip()
            escaped_code = html.escape(code)
            return f'<pre class="code-block">{escaped_code}</pre>'

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
        # Ajout du rendu pour les fichiers joints
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
                
                # Regex pour trouver les tags <filedata id="..." name="...">...</filedata>
                # Le contenu du tag (group 3) est brut.
                for match in re.finditer(r"<filedata id=\"(.*?)\"(?: name=\"([^\"]+)\")?>([\s\S]*?)</filedata>", body_raw):
                    # Partie avant le tag <filedata>: échapper et remplacer \n par <br>
                    pre_match_text = body_raw[last_idx:match.start()]
                    final_html_parts.append(html.escape(pre_match_text).replace("\n", "<br>"))
                    
                    # Traiter le tag <filedata>
                    file_id = match.group(1)
                    file_name = match.group(2) if match.group(2) else "Fichier joint" # Nom depuis l'attribut name
                    raw_file_content = match.group(3) # Contenu brut du fichier

                    expanded = self.reason_states.get(f"userfile_{file_id}", False)
                    arrow = "▼" if expanded else "▶"
                    
                    # Échapper le nom du fichier pour l'affichage dans le header
                    safe_file_name = html.escape(file_name)
                    header_html = f'<div class="file-header"><a href="userfile:{file_id}">{arrow}</a> <span>{safe_file_name} (cliquer pour afficher/masquer)</span></div>'
                    final_html_parts.append(header_html)
                    
                    if expanded:
                        # Échapper le contenu du fichier pour l'affichage dans <pre>
                        escaped_file_content = html.escape(raw_file_content)
                        # <pre> gère les \n nativement, donc pas de .replace("\n", "<br>") ici.
                        file_block_html = f'<div class="file-block"><pre>{escaped_file_content}</pre></div>'
                        final_html_parts.append(file_block_html)
                    
                    last_idx = match.end()
                
                # Partie après le dernier tag <filedata> (ou tout le message si aucun tag)
                post_match_text = body_raw[last_idx:]
                final_html_parts.append(html.escape(post_match_text).replace("\n", "<br>"))
                
                body_formatted = "".join(final_html_parts)
            else: # Assistant
                body_formatted = self._format_assistant(body_raw) # _format_assistant gère déjà les <think> et ```code```

            html_content += (
                f'<div class="message-container {message_class}">'
                f'<span class="{role_class}">{role_text}:</span>'
                f'<div class="message-body">{body_formatted}</div>'
                f'</div>'
                f'<hr>'
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
            if file_path.name == "model_favorites.json":
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

    # --- Début des méthodes pour Drag & Drop ---
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            # Vérifier si c'est un seul fichier (on ne gère pas le multi-drop pour l'instant)
            if len(event.mimeData().urls()) == 1:
                url = event.mimeData().urls()[0]
                if url.isLocalFile():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                file_path = Path(url.toLocalFile())
                if file_path.exists() and file_path.is_file():
                    self._process_file_attachment(file_path)
                    event.acceptProposedAction()
                    return
        event.ignore()
    # --- Fin des méthodes pour Drag & Drop ---
