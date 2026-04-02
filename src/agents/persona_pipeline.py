"""
Agentic LLM pipeline for transforming ML segments into named personas.
Author: Priyanka Sinha
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PersonaPipeline:
    """
    Agentic pipeline that orchestrates:
    1. Segment profile extraction
    2. LLM-based persona naming and description
    3. Optional image generation via Gemini AI
    """

    def __init__(self, llm_provider: str = "anthropic", generate_images: bool = False):
        self.llm_provider = llm_provider
        self.generate_images = generate_images
        self.results_: Optional[Dict] = None

    def _call_llm(self, prompt: str) -> str:
        """
        Call the configured LLM API.
        Set ANTHROPIC_API_KEY or GEMINI_API_KEY as environment variable.
        """
        if self.llm_provider == "anthropic":
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text
            except Exception as e:
                logger.warning(f"LLM call failed: {e}. Using mock response.")
                return json.dumps({"name": "Premium Petra", "description": "Mock persona.", "key_traits": ["loyal", "high-value", "frequent"]})
        return "{}"

    def _build_prompt(self, segment_profile: Dict) -> str:
        return f"""
You are a senior marketing strategist at a B2B travel company.
Based on the customer segment profile below, create a memorable persona.

Segment Profile:
{json.dumps(segment_profile, indent=2)}

Return ONLY valid JSON with these exact keys:
- "name": A memorable first name + adjective (e.g. "Premium Petra")
- "description": 2 sentences describing this customer type
- "key_traits": List of exactly 3 trait strings
- "value_tier": One of "high", "medium", "low"
"""

    def run(self, segment_profiles: pd.DataFrame) -> Dict:
        """
        Run the full persona generation pipeline.

        Args:
            segment_profiles: DataFrame with one row per segment,
                              containing feature means and segment metadata

        Returns:
            Dict mapping segment_id -> persona dict
        """
        personas = {}
        for idx, row in segment_profiles.iterrows():
            profile = row.to_dict()
            prompt = self._build_prompt(profile)
            raw_response = self._call_llm(prompt)

            try:
                persona = json.loads(raw_response.strip())
            except json.JSONDecodeError:
                persona = {
                    "name": f"Segment {idx} Customer",
                    "description": "Profile generation pending.",
                    "key_traits": [],
                    "value_tier": "medium"
                }

            persona["segment_id"] = idx
            personas[idx] = persona
            logger.info(f"Generated persona for segment {idx}: {persona.get('name')}")

        self.results_ = personas
        return personas

    def to_dataframe(self) -> pd.DataFrame:
        if self.results_ is None:
            raise ValueError("Run run() first.")
        return pd.DataFrame(self.results_).T.reset_index(drop=True)

    def save(self, path: str = "personas.json") -> None:
        with open(path, "w") as f:
            json.dump(self.results_, f, indent=2)
        logger.info(f"Personas saved to {path}")
