import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Paths:
    ROOT: Path = _PROJECT_ROOT
    DATA: Path = _PROJECT_ROOT / "data"
    SEED: Path = _PROJECT_ROOT / "data" / "seed"
    CHROMA_DB: Path = _PROJECT_ROOT / "data" / "chroma_db"
    DEMO_CACHE: Path = _PROJECT_ROOT / "data" / "demo_cache"


@dataclass(frozen=True)
class Models:
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    NLI_MODEL: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    NLI_LABELS: tuple = ("contradiction", "neutral", "entailment")

    GEMINI_MODEL: str = "gemini-2.0-flash"
    GROQ_MODEL_PRIMARY: str = "llama-3.3-70b-versatile"
    GROQ_MODEL_SECONDARY: str = "llama-3.1-8b-instant"
    GROQ_MODEL_TERTIARY: str = "gemma2-9b-it"
    GROQ_MODEL_QUATERNARY: str = "mixtral-8x7b-32768"

    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 2048


@dataclass(frozen=True)
class APIKeys:
    GEMINI: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    GROQ: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    def has_gemini(self) -> bool:
        return bool(self.GEMINI) and self.GEMINI != "your_gemini_key_here"

    def has_groq(self) -> bool:
        return bool(self.GROQ) and self.GROQ != "your_groq_key_here"


@dataclass
class RetrievalConfig:
    initial_k: int = 5
    lambda_1: float = 0.7
    lambda_2: float = 0.3
    chunk_size_min: int = 400
    chunk_size_max: int = 600
    chunk_overlap_sentences: int = 2

    def get_lambdas(self) -> tuple[float, float]:
        return (self.lambda_1, self.lambda_2)


@dataclass
class CalibrationConfig:
    confidence_threshold: float = 0.70
    max_cycles: int = 2
    k_increment_multiplier: float = 3.0
    lambda_adjust_step: float = 0.05
    kb_admission_threshold: float = 0.85
    redundancy_min_similarity: float = 0.3

    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3


@dataclass
class VerifierConfig:
    verified_threshold: float = 0.7
    contradicted_threshold: float = 0.3


@dataclass(frozen=True)
class VectorStoreConfig:
    collection_name: str = "verifai_kb"
    distance_metric: str = "cosine"


EXAMPLE_QUERIES = [
    "What is the transformer architecture and how does self-attention work?",
    "Explain retrieval-augmented generation and its limitations",
    "How does RLHF work in training large language models?",
    "What is a vector database and how does similarity search work?",
]

SEED_WIKIPEDIA_TOPICS = [
    "Transformer (deep learning architecture)",
    "Retrieval-augmented generation",
    "Large language model",
    "BERT (language model)",
    "Attention (machine learning)",
    "Vector database",
    "Natural language processing",
    "Prompt engineering",
]

FACT_CHECK_EXAMPLES = {
    "accurate": (
        "The Transformer architecture was introduced in the 2017 paper "
        "'Attention Is All You Need' by Vaswani et al. It relies entirely "
        "on self-attention mechanisms and does not use recurrence or convolutions. "
        "The model uses multi-head attention to attend to different representation "
        "subspaces at different positions."
    ),
    "mixed": (
        "BERT is a language model developed by OpenAI in 2018. It uses "
        "bidirectional training of Transformer encoders. BERT was pre-trained "
        "on BookCorpus and English Wikipedia. The base model has 110 million "
        "parameters and was the first model to use attention mechanisms."
    ),
    "inaccurate": (
        "GPT-4 is an open-source model with 1 trillion parameters released in 2022. "
        "It was trained exclusively on Reddit data using supervised learning only. "
        "GPT-4 does not support multimodal inputs and can only process English text."
    ),
}

XRAY_QUERIES = [
    "What is CRISPR and how does it work?",
    "Explain how retrieval-augmented generation reduces hallucinations",
    "How do modern gene editing techniques compare to traditional methods?",
]


paths = Paths()
models = Models()
api_keys = APIKeys()
retrieval = RetrievalConfig()
calibration = CalibrationConfig()
verifier = VerifierConfig()
vectorstore_config = VectorStoreConfig()
