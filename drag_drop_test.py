# fichier de test pour démontrer la solution au problème de drag & drop
from pathlib import Path
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QLabel

class DragDropWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test Drag & Drop")
        self.resize(600, 400)
        
        # Activer le drag & drop sur la fenêtre principale
        self.setAcceptDrops(True)
        
        # Interface de base
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Label explicatif
        self.info_label = QLabel("Glissez un fichier sur la zone de texte ci-dessous")
        self.layout.addWidget(self.info_label)
        
        # Zone de texte avec drag & drop personnalisé
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Glissez un fichier ici...")
        self.layout.addWidget(self.text_edit)
        
        # Zone pour afficher le résultat
        self.result_label = QLabel("Résultat:")
        self.layout.addWidget(self.result_label)
        
        # Remplacer les méthodes de gestion du drag & drop pour le QTextEdit
        self.text_edit.setAcceptDrops(True)
        self.text_edit.dragEnterEvent = self._textEdit_dragEnterEvent
        self.text_edit.dropEvent = self._textEdit_dropEvent
    
    def _textEdit_dragEnterEvent(self, event: QDragEnterEvent):
        """Gestion personnalisée du dragEnterEvent pour le QTextEdit."""
        if event.mimeData().hasUrls():
            # Vérifier si c'est un seul fichier
            if len(event.mimeData().urls()) == 1:
                url = event.mimeData().urls()[0]
                if url.isLocalFile():
                    event.acceptProposedAction()
                    return
        # Si c'est du texte standard, on laisse le comportement par défaut
        elif event.mimeData().hasText():
            QTextEdit.dragEnterEvent(self.text_edit, event)
            return
        event.ignore()
    
    def _textEdit_dropEvent(self, event: QDropEvent):
        """Gestion personnalisée du dropEvent pour le QTextEdit."""
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                file_path = Path(url.toLocalFile())
                if file_path.exists() and file_path.is_file():
                    # Au lieu d'ajouter l'URL au texte, on traite le fichier comme une pièce jointe
                    self.result_label.setText(f"Fichier joint: {file_path}")
                    # Simuler l'ajout d'un placeholder dans la zone de texte
                    self.text_edit.setPlainText(f"[Fichier joint: {file_path.name}]")
                    event.acceptProposedAction()
                    return
        # Si c'est du texte standard, on laisse le comportement par défaut
        elif event.mimeData().hasText():
            QTextEdit.dropEvent(self.text_edit, event)
            return
        event.ignore()
    
    # Ces méthodes sont pour le drag & drop sur la fenêtre principale
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                file_path = Path(url.toLocalFile())
                self.result_label.setText(f"Fichier déposé sur la fenêtre: {file_path}")
                event.acceptProposedAction()
        else:
            event.ignore()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = DragDropWindow()
    window.show()
    sys.exit(app.exec())
