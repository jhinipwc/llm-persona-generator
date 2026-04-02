# LLM Persona Generator

> Transforming abstract ML customer segments into named, photorealistic personas using LLMs and Gemini AI — making data science outputs intuitive for business stakeholders.

## Overview

This project demonstrates an Agentic AI workflow that takes customer segmentation outputs and generates human-named personas with descriptive profiles and AI-generated portrait imagery. Built to bridge the gap between technical ML outputs and business-ready customer intelligence.

**The problem it solves:** Data science teams often produce excellent segmentation models that business stakeholders struggle to act on. Abstract "Cluster 3" means nothing. "Premium Petra — a high-value corporate traveller who books 15+ nights per year and prioritises sustainability" is immediately actionable.

## Tech Stack
`Python` `LLM API` `Gemini AI` `Agentic AI` `prompt engineering` `pandas` `Pillow`

## Project Structure
```
llm-persona-generator/
├── src/
│   ├── agents/
│   │   ├── persona_naming_agent.py
│   │   ├── persona_description_agent.py
│   │   └── image_generation_agent.py
│   ├── prompts/
│   │   ├── naming_prompts.py
│   │   └── description_prompts.py
│   └── pipeline/
│       └── persona_pipeline.py
├── notebooks/
│   ├── 01_segment_to_persona.ipynb
│   └── 02_agentic_workflow.ipynb
├── examples/
│   └── sample_personas.json
├── requirements.txt
└── README.md
```

## Quick Start
```bash
git clone https://github.com/priyankasinhabhu/llm-persona-generator.git
cd llm-persona-generator
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
python src/pipeline/persona_pipeline.py --segments data/segments.json
```

## Example Output
```python
from src.pipeline.persona_pipeline import PersonaPipeline

pipeline = PersonaPipeline(llm_provider='gemini')
personas = pipeline.run(segment_profiles)

# Output:
# {
#   'segment_0': {
#     'name': 'Premium Petra',
#     'description': 'Senior executive, 35-50, books premium hotels 15+ nights/year...',
#     'image_url': 'generated_portrait.png',
#     'key_traits': ['quality-focused', 'loyalty-driven', 'sustainability-conscious']
#   }
# }
```

## Business Impact
At HRS Group, this approach transformed how commercial and product teams understood customer segments — increasing adoption of ML-driven recommendations and contributing to €100M+ YoY revenue impact.

---
**Priyanka Sinha** | [LinkedIn](https://linkedin.com/in/priyanka-sinha) | [Email](mailto:priyankasinhabhu@gmail.com)
