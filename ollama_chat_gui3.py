# chat_ollama.py
"""
Chat GUI for local Ollama models (Windows)
-----------------------------------------
• **Nouveau** bouton (➕) pour démarrer un fil. Titres auto‑générés.  
• Vue chat Markdown : blocs de code rendus, et **<think></think>** affichés façon *LM Studio* (encadré avec flèche ▶/▼ pour ouvrir‑fermer).  
• Éditeur auto‑redimensionnable, **Ctrl+Entrée** pour envoyer.  
• Sélecteur de modèle (dynamique via `ollama list`).  
• Stats : tokens & tok/s + CPU/RAM.  
• Sauvegarde JSON dans `%APPDATA%/OllamaChats`.  

TODO : édition/suppression message, GPU NVML, thèmes et raccourcis supplémentaires.

> Installation : `pip install pyside6 requests psutil rich python-dotenv`
> pyinstaller --onefile --windowed ollama_chat_gui3.py
> $Env:OPENAI_API_KEY = "sk-svcacce..." ;
"""
from __future__ import annotations

import os
import dotenv  # ← nouveau
dotenv.load_dotenv()  # lit .env à la racine du projet
import httpx

import html
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
import requests
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QTextCursor, QTextOption, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QMenu,  # ← ajouté
)

# Mise à jour pour le nouveau SDK OpenAI
from openai import OpenAI  # Nouveau SDK 1.0.0
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None
if OPENAI_API_KEY:
    # Configuration du proxy si nécessaire via l'environnement httpx
    #import httpx
    #proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    #try:
    #    # httpx <0.18 attend un dict pour `proxies`
    #    kwargs = {}
    #    if proxy:
    #        kwargs["proxies"] = {"http": proxy, "https": proxy}
    #    http_client = httpx.Client(**kwargs)
    #except TypeError:
    #    # fallback si `proxies` n’est pas supporté
    #    http_client = httpx.Client()

    # laisse httpx lire automatiquement HTTP_PROXY/HTTPS_PROXY
    http_client = httpx.Client(trust_env=True)
        
    # Initialisation du client OpenAI
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        http_client=http_client
    )
    
    # Fonction pour lister les modèles OpenAI disponibles
    def list_openai_models():
        try:
            models = client.models.list()
            return [model.id for model in models.data 
                   if "gpt" in model.id.lower() or "o3" in model.id.lower()]
        except Exception as e:
            print(f"Erreur lors de la récupération des modèles OpenAI: {e}")
            return []

    # Fonction pour utiliser l'API OpenAI chat completion
    def chat_completion(model: str, messages: list[dict]):
        """
        Appelle l'API OpenAI chat completion
        Args:
            model: Nom du modèle (gpt-4.1-2025-04-14, o3-2025-04-16 etc.)
            messages: Liste de messages au format [{"role": "user", "content": "..."}]
        Returns:
            Objet de réponse de l'API OpenAI
        """
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                #temperature=0.7,
            )
            return response
        except Exception as e:
            print(f"Erreur lors de l'appel à OpenAI: {e}")
            raise

# ------------------------------- Config -------------------------------
OLLAMA_URL = "http://localhost:11434"
SAVE_DIR = Path(os.getenv("APPDATA", ".")) / "OllamaChats"
SAVE_DIR.mkdir(exist_ok=True)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# ------------------------------- Data -------------------------------
@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    tokens: int = 0
    tok_s: float = 0.0

# --------------------------- Ollama client ---------------------------
class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL):
        self.base = base_url.rstrip("/")

    def list_models(self) -> list[str]:
        r = requests.get(f"{self.base}/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json()["models"]]

    def chat(self, model: str, messages: list[dict]) -> tuple[str, int, float]:
        payload = {"model": model, "messages": messages, "stream": False}
        start = time.time()
        r = requests.post(f"{self.base}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        duration = max(time.time() - start, 1e-6)
        data = r.json()
        total_tokens = data.get("usage", {}).get("total_tokens", 0)
        return data["message"]["content"], total_tokens, total_tokens / duration

# ---------------------------- Main window ----------------------------
class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ollama Chat")
        self.resize(960, 640)

        # STATE -------------------------------------------------------
        self.client = OllamaClient()
        self.current_model: str | None = None
        self.conversations: dict[str, list[Message]] = {}
        self.current_conv_id: str | None = None
        self.reason_states: dict[str, bool] = {}

        # — Initialisation des favoris —
        self.fav_file = SAVE_DIR / "model_favorites.json"
        self.favorites = self._load_model_favorites()

        # LEFT PANEL --------------------------------------------------
        self.new_conv_btn = QPushButton("➕ Nouvelle conversation")
        self.new_conv_btn.clicked.connect(self.new_conversation)
        self.conv_list = QListWidget()
        self.conv_list.itemClicked.connect(self.switch_conversation)

        # --- AJOUT suppression via menu contextuel ---
        self.conv_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.conv_list.customContextMenuRequested.connect(self._show_conv_context_menu)
        self.del_conv_action = QAction("Supprimer conversation", self)
        self.del_conv_action.triggered.connect(self.delete_conversation)
        # ------------------------------------------------

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.new_conv_btn)
        left_layout.addWidget(self.conv_list, 1)

        # RIGHT PANEL -------------------------------------------------
        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(False)
        self.chat_view.setOpenLinks(False)
        self.chat_view.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.chat_view.anchorClicked.connect(self._anchor_clicked)
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
                white-space: pre-wrap; /* Ensure wrapping */
                word-wrap: break-word; /* Break long words */
            }
            .code-block {
                background-color: #282c34; /* Dark background for code */
                color: #abb2bf; /* Light text for code */
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

        self.msg_edit = QTextEdit()
        self.msg_edit.textChanged.connect(self._auto_resize)
        self.msg_edit.setMaximumHeight(150)

        self.send_btn = QPushButton("Envoyer (Ctrl+Enter)")
        self.send_btn.clicked.connect(self.send_message)

        self.model_box = QComboBox()
        # menu contextuel pour ajouter/retirer un favori
        self.model_box.setContextMenuPolicy(Qt.CustomContextMenu)
        self.model_box.customContextMenuRequested.connect(self._show_model_context_menu)

        # remplissage initial
        self._populate_model_box()
        self.model_box.currentTextChanged.connect(self.change_model)

        self.stats_label = QLabel("Tokens: 0 – 0 tok/s")
        self.res_label = QLabel("CPU: 0%  RAM: 0%")

        # Layouts -----------------------------------------------------
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

        # Shortcuts / timers -----------------------------------------
        self.msg_edit.keyPressEvent = self._key_press_override
        QTimer.singleShot(0, self._start_stats_timer)

        # Init --------------------------------------------------------
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

    # -------------------------- Timers -----------------------------
    def _start_stats_timer(self):
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._update_resource_stats)
        self.stats_timer.start(1000)

    def _update_resource_stats(self):
        self.res_label.setText(f"CPU: {psutil.cpu_percent():.0f}%  RAM: {psutil.virtual_memory().percent:.0f}%")

    # ---------------------- Favoris modèles ------------------------
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
            print(f"Erreur sauvegarde favoris : {e}")

    def _populate_model_box(self):
        # récupère tous les modèles
        models: list[str] = []
        try:
            models = self.client.list_models()
        except requests.exceptions.RequestException as e:
            QMessageBox.warning(self, "Erreur Ollama", f"Impossible de lister les modèles : {e}")
        if OPENAI_API_KEY:
            try:
                openai_models = list_openai_models()
                models += [f"OpenAI: {m}" for m in openai_models]
            except Exception as e:
                QMessageBox.warning(self, "Erreur OpenAI", f"Impossible de lister OpenAI : {e}")
        # supprime les doublons
        models = list(dict.fromkeys(models))
        # priorise les favoris
        favs = [m for m in self.favorites if m in models]
        others = sorted([m for m in models if m not in self.favorites])
        # mise à jour de la combo avec étoile pour les favoris
        prev_model = self.current_model
        self.model_box.clear()
        for m in favs:
            self.model_box.addItem(f"★ {m}", m)
        for m in others:
            self.model_box.addItem(m, m)
        # restaure la sélection
        if prev_model:
            for i in range(self.model_box.count()):
                if self.model_box.itemData(i) == prev_model:
                    self.model_box.setCurrentIndex(i)
                    break

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
        # on recrée la liste, on restaure la sélection
        prev = self.model_box.currentText()
        self._populate_model_box()
        idx = self.model_box.findText(prev)
        if idx >= 0:
            self.model_box.setCurrentIndex(idx)

    # ---------------------- UI helpers ------------------------------
    def _auto_resize(self):
        h = self.msg_edit.document().size().height() + 10
        self.msg_edit.setFixedHeight(min(int(h), 150))

    def _key_press_override(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter) and e.modifiers() & Qt.ControlModifier:
            self.send_message()
        else:
            QTextEdit.keyPressEvent(self.msg_edit, e)

    # ------------------ Conversation management ---------------------
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

    # ----------------------- Model change ---------------------------
    def change_model(self, _):
        # récupère le model réel stocké en userData
        idx = self.model_box.currentIndex()
        model_name = self.model_box.itemData(idx)
        self.current_model = model_name

    # -------------------------- Send -------------------------------
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
                # appel OpenAI
                model_name = self.current_model[len("OpenAI: "):]
                start = time.time()
                resp_obj = chat_completion(model=model_name, messages=payload)
                duration = max(time.time() - start, 1e-6)
                resp = resp_obj.choices[0].message.content
                total_tokens = resp_obj.usage.total_tokens
                tok_s = total_tokens / duration
            else:
                # appel Ollama existant
                resp, total_tokens, tok_s = self.client.chat(self.current_model, payload)

            conv.append(Message("assistant", resp, total_tokens, tok_s))
            self.stats_label.setText(f"Tokens: {total_tokens} – {tok_s:.1f} tok/s")
            self.render_conversation()
            self._save()
        except Exception as err:
            QMessageBox.critical(self, "Erreur", str(err))

    # --------------------- Rendering engine -------------------------
    def _format_assistant(self, txt: str) -> str:
        """Convert assistant raw text to HTML with collapsible reasoning box and styled code blocks."""
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
            role_text = "Vous" if m.role == "user" else "IA"

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

    # ----------------------- Toggle reasoning -----------------------
    def _anchor_clicked(self, url: QUrl):
        scheme = url.scheme()
        if scheme == "reason":
            rid = url.path() or url.opaque()
            if rid:
                self.reason_states[rid] = not self.reason_states.get(rid, False)
                self.render_conversation()
            else:
                print(f"Warning: Could not extract reason ID from URL: {url.toString()}")
        elif scheme in ["http", "https"]:
            QDesktopServices.openUrl(url)

    # -------------------- Persistence helper ------------------------
    def _save(self):
        if not self.current_conv_id:
            return
        try:
            path = SAVE_DIR / f"{self.current_conv_id}.json"
            conv_data = [asdict(m) for m in self.conversations.get(self.current_conv_id, [])]
            path.write_text(json.dumps(conv_data, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"Error saving conversation {self.current_conv_id}: {e}")

    def load_conversations(self):
        self.conversations = {}
        self.conv_list.clear()
        loaded_items = []
        for file_path in SAVE_DIR.glob("*.json"):
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
                print(f"Error loading conversation {file_path.name}: {e}")

        loaded_items.sort(key=lambda x: x.text())
        for item in loaded_items:
            self.conv_list.addItem(item)

    # ------------------ menu contextuel ---------------------
    def _show_conv_context_menu(self, pos):
        item = self.conv_list.itemAt(pos)
        if item:
            menu = QMenu(self)
            menu.addAction(self.del_conv_action)
            menu.exec(self.conv_list.mapToGlobal(pos))

    # ---------------- suppression conversation --------------
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
            # supprime fichier JSON
            try:
                (SAVE_DIR / f"{cid}.json").unlink()
            except Exception:
                pass
            # supprime de la mémoire et de l'UI
            self.conversations.pop(cid, None)
            row = self.conv_list.row(item)
            self.conv_list.takeItem(row)
            # réinitialise l'affichage
            if self.current_conv_id == cid:
                if self.conv_list.count():
                    self.conv_list.setCurrentRow(0)
                    self.switch_conversation(self.conv_list.currentItem())
                else:
                    self.new_conversation()

# ============================== run ================================
if __name__ == "__main__":
    app = QApplication([])
    win = ChatWindow()
    win.show()
    app.exec()
