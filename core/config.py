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


@dataclass
class Models:
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    NLI_MODEL: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    NLI_LABELS: tuple = ("contradiction", "neutral", "entailment")

    GEMINI_MODEL: str = "gemini-2.0-flash"
    GROQ_MODEL_PRIMARY: str = "llama-3.3-70b-versatile"
    GROQ_MODEL_SECONDARY: str = "llama-3.1-8b-instant"
    GROQ_MODEL_TERTIARY: str = "openai/gpt-oss-20b"
    GROQ_MODEL_QUATERNARY: str = "meta-llama/llama-4-scout-17b-16e-instruct"

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
    max_cycles: int = 4
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
    "What are the top cyber threats identified in the ENISA Threat Landscape 2025?",
    "How do cybersecurity investment priorities differ across critical infrastructure sectors?",
    "What role does ransomware play in the current EU threat landscape?",
    "What are ENISA's key recommendations for critical infrastructure protection?",
]

SEED_WIKIPEDIA_TOPICS: list[str] = []

FACT_CHECK_EXAMPLES = {
    "accurate": (
        "Ransomware remains one of the most significant cybersecurity threats "
        "to organizations in the European Union. ENISA identified it as a top "
        "threat in their 2025 Threat Landscape report, with critical infrastructure "
        "sectors being particularly targeted."
    ),
    "mixed": (
        "DDoS attacks and supply chain compromises have been identified as growing "
        "threats to EU critical infrastructure by ENISA. Law enforcement operations "
        "successfully disrupted several major ransomware groups in 2024, but the "
        "overall number of cyberattacks against EU institutions decreased by 50% as a result."
    ),
    "inaccurate": (
        "Nuclear power plants are the most frequently targeted critical infrastructure "
        "by cyberattacks, accounting for over 40% of all incidents in 2025. ENISA "
        "reported that the EU has no cybersecurity regulations for the energy sector."
    ),
}

XRAY_QUERIES = [
    "What specific cybersecurity measures does NIS2 require for energy sector operators?",
    "How has the ransomware threat evolved for EU critical infrastructure?",
    "Compare cybersecurity risks facing renewable energy vs nuclear power facilities",
]


paths = Paths()
models = Models()
api_keys = APIKeys()
retrieval = RetrievalConfig()
calibration = CalibrationConfig()
verifier = VerifierConfig()
vectorstore_config = VectorStoreConfig()
