from dataclasses import dataclass

@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    tokens: int = 0
    tok_s: float = 0.0
    model: str = ""