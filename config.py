import os
from pathlib import Path
from typing import Tuple

DOMAIN_MAPPING = {
    1: ("Health checkups", "health_checkups", "knowledge/health_checkups"),
    2: ("General outpatient visits", "routine_outpatient", "knowledge/routine_outpatient"),
    3: ("Surgical procedures", "surgery", "knowledge/surgery"),
    4: ("Hospitalization management", "hospitalization", "knowledge/hospitalization"),
    5: ("Customized domain-specific features", "general", "knowledge/general"),
}


class Config:
    """Centralized configuration for the SCT Pipeline."""

    def __init__(
        self,
        input_path: str,
        output_dir: str,
        domain: int = 1,
        model_name: str = "gpt-4o",
        planner_model: str = "gpt-4o",
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.0,
        chunk_size: int = 50,
    ):
        self.api_key = api_key or os.environ.get("SCT_API_KEY", "YOUR_API_KEY_HERE")
        self.base_url = base_url or os.environ.get("SCT_BASE_URL", "YOUR_BASE_URL_HERE")
        self.model_name = model_name
        self.planner_model = planner_model

        self.temperature = temperature
        self.seed = 42

        self.chunk_size = chunk_size
        self.segment_min_chars = 2000
        self.segment_max_chars = 3000

        self.input_path = os.path.abspath(input_path)
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.project_root = Path(__file__).parent
        self._set_domain(domain)

    def _set_domain(self, domain: int):
        """Select medical domain and resolve knowledge base path."""
        if domain not in DOMAIN_MAPPING:
            raise ValueError(f"Invalid domain index: {domain}. Must be 1-5.")

        self.domain_index = domain
        self.domain_name, self.domain_key, rel_path = DOMAIN_MAPPING[domain]
        self.knowledge_base = (self.project_root / rel_path).resolve()

        if not self.knowledge_base.exists():
            raise FileNotFoundError(f"Knowledge base not found: {self.knowledge_base}")

        self.sct_file = self.knowledge_base / "SCT.txt"
        if not self.sct_file.exists():
            raise FileNotFoundError(f"SCT.txt not found in {self.knowledge_base}")

    def __repr__(self):
        return (
            f"Config(model={self.model_name}, domain={self.domain_name}, "
            f"input={self.input_path}, output={self.output_dir})"
        )
