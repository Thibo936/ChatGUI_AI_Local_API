from dataclasses import dataclass

@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    tokens: int = 0
    tok_s: float = 0.0
    model: str = ""

@dataclass
class ModelCaps:
    name: str
    supports_images: bool = False
    supports_general_files: bool = False  # Pour PDF, Code, etc.
    max_tokens: int = 4096  # Une valeur par défaut, à ajuster par modèle
    can_stream: bool = True # La plupart des modèles supportent le streaming