import os
import json
import re
import time
import uuid
import html
import logging
from dataclasses import asdict
from pathlib import Path

import psutil
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QTextCursor, QTextOption, QDesktopServices
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
)

from config import SAVE_DIR
from models import Message
from ollama_client import OllamaClient, is_ollama_running, start_ollama_server
from utils import log_critical_error

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# Modèles par défaut si aucun n'est disponible
DEFAULT_MODELS = ["llama2", "mistral", "phi2", "gemma:2b"]

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ollama_available = is_ollama_running()
        self.setWindowTitle("ChatGUI_AI_Local_API")
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

        self.fav_file = SAVE_DIR / "model_favorites.json"
        self.favorites = self._load_model_favorites()

        self._setup_ui()
        self._setup_connections()
        self._start_stats_timer()

        # Remplir la liste des modèles
        self._populate_model_box()
        
        # Sélectionner un modèle s'il y en a
        if self.model_box.count() > 0:
            self.change_model(self.model_box.currentText())
        
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
        self.model_box = QComboBox()
        self.stats_label = QLabel("Tokens: 0 – 0 tok/s")
        self.res_label = QLabel("CPU: 0%  RAM: 0%")

        # Configuration des widgets
        self.chat_view.setOpenExternalLinks(False)
        self.chat_view.setOpenLinks(False)
        self.chat_view.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.msg_edit.setMaximumHeight(150)
        self.model_box.setContextMenuPolicy(Qt.CustomContextMenu)

        # Mise en page
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.new_conv_btn)
        left_layout.addWidget(self.conv_list, 1)

        editor_bar = QHBoxLayout()
        editor_bar.addWidget(self.msg_edit, 1)
        editor_bar.addWidget(self.send_btn)

        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(self.model_box)
        bottom_bar.addWidget(self.stats_label)
        bottom_bar.addStretch(1)
        bottom_bar.addWidget(self.res_label)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.addWidget(self.chat_view, 1)
        right_layout.addLayout(editor_bar)
        right_layout.addLayout(bottom_bar)

        splitter = QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 4)
        self.setCentralWidget(splitter)

        # Style CSS
        self.chat_view.document().setDefaultStyleSheet("""
            body { font-family: sans-serif; line-height: 1.4; }
            .message-container { margin-bottom: 10px; }
            .role-user { font-weight: bold; color: #0055cc; }
            .role-assistant { font-weight: bold; color: #008000; }
            .message-body {
                padding: 8px 12px;
                border-radius: 10px;
                margin-top: 2px;
                display: inline-block;
                max-width: 95%;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .code-block {
                background-color: #282c34;
                color: #abb2bf;
                border: 1px solid #ccc;
                padding: 10px;
                margin: 8px 0;
                font-family: monospace;
                white-space: pre-wrap;
                word-wrap: break-word;
                display: block;
                border-radius: 4px;
                font-size: 0.9em;
            }
            .think-header a { text-decoration: none; color: #c5fac5; }
            .think-header span { font-style: italic; color: #c5fac5; }
            .think-block {
                border: 1px dashed #999;
                background-color: #454c59;
                padding: 8px;
                margin: 4px 0 4px 20px;
                font-family: monospace;
                white-space: pre-wrap;
                word-wrap: break-word;
                border-radius: 4px;
                font-size: 0.9em;
            }
            hr { border: 0; height: 1px; background-color: #ddd; margin: 15px 0; }
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
        self.model_box.currentTextChanged.connect(self.change_model)
        self.model_box.customContextMenuRequested.connect(self._show_model_context_menu)
        self.chat_view.anchorClicked.connect(self._anchor_clicked)
        self.msg_edit.keyPressEvent = self._key_press_override

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
                
        # Essayer de récupérer les modèles OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                
                # Créer un client avec l'API key
                client = OpenAI(api_key=api_key)
                
                # Récupérer la liste des modèles
                response = client.models.list()
                
                # Filtrer les modèles pour ne garder que ceux qui contiennent "gpt" ou "o3"
                openai_models = [model.id for model in response.data 
                               if "gpt" in model.id.lower() or "o3" in model.id.lower()]
                               
                if openai_models:
                    logging.info(f"Modèles OpenAI récupérés: {openai_models}")
                    models += [f"OpenAI: {m}" for m in openai_models]
                else:
                    logging.warning("Aucun modèle OpenAI correspondant trouvé")
            except Exception as e:
                logging.error(f"Erreur lors de la récupération des modèles OpenAI: {e}", exc_info=True)
                
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
        if e.key() in (Qt.Key_Return, Qt.Key_Enter) and not (e.modifiers() & Qt.ShiftModifier):
            self.send_message()
        else:
            QTextEdit.keyPressEvent(self.msg_edit, e)

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

    def switch_conversation(self, item: QListWidgetItem):
        self.current_conv_id = item.data(Qt.UserRole)
        self.reason_states.clear()
        self.render_conversation()

    def change_model(self, _):
        idx = self.model_box.currentIndex()
        model_name = self.model_box.itemData(idx)
        self.current_model = model_name

    def send_message(self):
        txt = self.msg_edit.toPlainText().strip()
        if not txt:
            return
        self.msg_edit.clear()

        conv = self.conversations[self.current_conv_id]
        conv.append(Message("user", txt))
        if len(conv) == 1:
            self.conv_list.currentItem().setText(txt.split("\n", 1)[0][:40])
        self.render_conversation()

        payload = [{"role": m.role, "content": m.content} for m in conv]
        try:
            if self.current_model and self.current_model.startswith("OpenAI: "):
                model_name = self.current_model[len("OpenAI: "):]
                from openai import OpenAI
                client = OpenAI()
                start = time.time()
                resp_obj = client.chat.completions.create(
                    model=model_name,
                    messages=payload,
                )
                duration = max(time.time() - start, 1e-6)
                resp = resp_obj.choices[0].message.content
                total_tokens = resp_obj.usage.total_tokens
                tok_s = total_tokens / duration
                conv.append(Message("assistant", resp, total_tokens, tok_s, model=self.current_model))
            else:
                resp, total_tokens, tok_s = self.client.chat(self.current_model, payload)
                conv.append(Message("assistant", resp, total_tokens, tok_s, model=self.current_model))

            self.stats_label.setText(f"Tokens: {total_tokens} – {tok_s:.1f} tok/s")
            self.render_conversation()
            self._save()
        except Exception as err:
            log_critical_error("Erreur lors de l'envoi du message", err)
            QMessageBox.critical(self, "Erreur", str(err))

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

        return txt.replace("\n", "<br>")

    def render_conversation(self):
        self.chat_view.clear()
        html_content = ""
        for m in self.conversations.get(self.current_conv_id, []):
            role_class = "role-user" if m.role == "user" else "role-assistant"
            message_class = "user-message" if m.role == "user" else "assistant-message"
            if m.role == "user":
                role_text = "Vous"
            else:
                model_name = m.model if getattr(m, "model", None) else "IA"
                role_text = f"IA ({model_name})"

            body_raw = m.content
            if m.role == "user":
                body_formatted = html.escape(body_raw).replace("\n", "<br>")
            else:
                body_formatted = self._format_assistant(body_raw)

            html_content += (
                f'<div class="message-container {message_class}">'
                f'<span class="{role_class}">{role_text}:</span>'
                f'<div class="message-body">{body_formatted}</div>'
                f'</div>'
                f'<hr>'
            )

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
                elif messages and messages[0].role == 'assistant' and len(messages) > 1 and messages[1].role == 'user':
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