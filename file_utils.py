import mimetypes
from pathlib import Path

# Ajout de types MIME courants pour le code, au cas où mimetypes ne les reconnaîtrait pas bien
mimetypes.add_type("text/x-python", ".py")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/x-c", ".c")
mimetypes.add_type("text/x-cpp", ".cpp")
mimetypes.add_type("text/x-java-source", ".java")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("text/css", ".css")


def get_file_type(file_path: Path) -> str:
    """Détermine le type d'un fichier en se basant sur son type MIME et son extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    extension = file_path.suffix.lower()

    if mime_type:
        primary_type = mime_type.split('/')[0]
        if primary_type == "image":
            return "image"
        if mime_type == "application/pdf":
            return "pdf"
        if primary_type == "text" or "script" in mime_type or "xml" in mime_type or "json" in mime_type:
            # Considérer comme 'code' si l'extension est commune pour le code, sinon 'text'
            # Ceci est une heuristique et pourrait être affinée
            code_extensions = [".py", ".js", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".html", ".css", ".ts", ".tsx", ".jsx", ".vue"]
            if extension in code_extensions or "x-" in mime_type: # ex: text/x-python
                return "code"
            return "text"
    
    # Si le type MIME n'est pas concluant, se baser sur l'extension
    if extension in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"]:
        return "image"
    if extension == ".pdf":
        return "pdf"
    if extension in [".txt", ".md", ".rtf", ".log"]:
        return "text"
    # Extensions de code plus explicites
    code_extensions = [".py", ".js", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".html", ".css", ".ts", ".tsx", ".jsx", ".vue", ".json", ".xml", ".yaml", ".sh"]
    if extension in code_extensions:
        return "code"

    return "unknown" # Type de fichier inconnu ou non géré

def read_text_file(file_path: Path) -> str | None:
    """Lit le contenu d'un fichier texte (ou code)."""
    try:
        return file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier texte {file_path}: {e}")
        return None

def extract_text_from_pdf(file_path: Path) -> str | None:
    """Extrait le contenu textuel d'un fichier PDF."""
    # Nécessite PyPDF2: pip install PyPDF2
    try:
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
        return text
    except ImportError:
        print("La bibliothèque PyPDF2 n'est pas installée. Impossible d'extraire le texte des PDF.")
        return None
    except Exception as e:
        print(f"Erreur lors de l'extraction du texte du PDF {file_path}: {e}")
        return None

def resize_and_encode_image(file_path: Path, max_size_kb: int = 500, target_format: str = 'JPEG') -> str | None:
    """Redimensionne une image si elle est trop grande, l'encode en base64."""
    # Nécessite Pillow: pip install Pillow
    try:
        from PIL import Image
        import base64
        import io

        img = Image.open(file_path)
        
        # Convertir en RGB si RGBA ou P pour éviter les problèmes avec JPEG
        if img.mode == 'RGBA' or img.mode == 'P':
            img = img.convert('RGB')

        # Redimensionnement si nécessaire (très basique, pourrait être amélioré pour garder le ratio)
        # Ceci est un placeholder, une vraie logique de redimensionnement en gardant le ratio serait mieux
        # Par exemple, réduire proportionnellement si la taille en octets de l'image encodée dépasse max_size_kb
        # Pour l'instant, nous allons juste l'encoder.
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=target_format, quality=85) # quality pour JPEG
        img_byte_arr = img_byte_arr.getvalue()

        # Vérifier la taille après une première compression
        if len(img_byte_arr) > max_size_kb * 1024:
            # Si trop gros, essayer de réduire la qualité ou la taille davantage
            # Cette partie nécessiterait une logique plus avancée de redimensionnement/compression itérative
            print(f"L'image {file_path} est volumineuse ({len(img_byte_arr)//1024} KB), un redimensionnement plus agressif serait nécessaire.")
            # Pour l'instant, on la retourne telle quelle si la simple compression ne suffit pas
            # pour ne pas bloquer, mais idéalement on réduirait la résolution ici.

        encoded_string = base64.b64encode(img_byte_arr).decode('utf-8')
        mime_type = f"image/{target_format.lower()}"
        return f"data:{mime_type};base64,{encoded_string}"

    except ImportError:
        print("La bibliothèque Pillow n'est pas installée. Impossible de traiter les images.")
        return None
    except Exception as e:
        print(f"Erreur lors du traitement de l'image {file_path}: {e}")
        return None

def ocr_image(file_path: Path) -> str | None:
    """Effectue l'OCR sur une image pour en extraire le texte."""
    # Nécessite Pillow et pytesseract: pip install Pillow pytesseract
    # Nécessite également l'installation de Tesseract OCR sur le système:
    # https://github.com/tesseract-ocr/tesseract#installing-tesseract
    try:
        from PIL import Image
        import pytesseract
        
        # Si vous êtes sous Windows, vous devrez peut-être configurer le chemin vers tesseract.exe
        # Exemple: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        text = pytesseract.image_to_string(Image.open(file_path))
        return text.strip()
    except ImportError:
        print("Pillow ou Pytesseract ne sont pas installés. OCR impossible.")
        return None
    except Exception as e: # Capturer aussi les erreurs de Tesseract (ex: non trouvé)
        print(f"Erreur lors de l'OCR de l'image {file_path}: {e}")
        print("Assurez-vous que Tesseract OCR est installé et accessible (dans le PATH ou via tesseract_cmd).")
        return None

def count_tokens_for_openai(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Compte le nombre de tokens pour un texte donné en utilisant tiktoken, spécifique à OpenAI."""
    # Nécessite tiktoken: pip install tiktoken
    try:
        import tiktoken
        # Gérer le cas où le nom du modèle vient avec le préfixe "OpenAI: "
        if model_name.startswith("OpenAI: "):
            actual_model_name = model_name[len("OpenAI: "):]
        else:
            actual_model_name = model_name
        
        try:
            encoding = tiktoken.encoding_for_model(actual_model_name)
        except KeyError:
            print(f"Avertissement: Modèle {actual_model_name} non trouvé pour tiktoken. Utilisation de cl100k_base.")
            encoding = tiktoken.get_encoding("cl100k_base") # Fallback pour les modèles inconnus
        
        num_tokens = len(encoding.encode(text))
        return num_tokens
    except ImportError:
        print("Tiktoken n'est pas installé. Impossible de compter les tokens pour OpenAI.")
        return -1 # Retourner -1 ou lever une exception pour indiquer l'échec
    except Exception as e:
        print(f"Erreur lors du comptage des tokens avec tiktoken: {e}")
        return -1

if __name__ == '__main__':
    # Exemples de test
    test_files = {
        Path("image.png"): "image",
        Path("document.pdf"): "pdf",
        Path("script.py"): "code",
        Path("readme.txt"): "text",
        Path("archive.zip"): "unknown",
        Path("webpage.html"): "code", # ou text selon la granularité souhaitée
        Path("styles.css"): "code",
        Path("data.json"): "code",
        Path("note"): "unknown" # Pas d'extension
    }

    for test_file, expected_type in test_files.items():
        # Créer des fichiers factices pour le test si mimetypes.guess_type en a besoin
        # (pour ce test simple, le nom suffit)
        # if not test_file.exists() and expected_type != "unknown":
        #     try:
        #         test_file.touch()
        #     except OSError: # Nom de fichier invalide sur certains OS pour le test
        #         pass 
        
        file_type = get_file_type(test_file)
        print(f"File: {test_file}, Guessed Type: {file_type}, Expected: {expected_type}, Match: {file_type == expected_type}")
        
        # Supprimer les fichiers factices après le test
        # if test_file.exists() and expected_type != "unknown":
        #     try:
        #         test_file.unlink()
        #     except OSError:
        #         pass 