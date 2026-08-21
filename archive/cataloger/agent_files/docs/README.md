# Discogs Record Classification - Evaluation Package

## Quick Start

```bash
pip install -r requirements.txt
python evaluate.py
```

## Documentation

- **EVALUATION_FILES.md** - Technical reference for data files and scripts
- **GRADER_GUIDANCE.md** - Guidance for agentic grader on evaluating correctness

## Files

```
eval/
├── data/
│   ├── test_set.json          # 25k ground truth
│   ├── lp_catalog.json        # 2.9M catalog
│   └── enriched_training.json # 60k training data
├── evaluate.py                # Run this
├── test_api_client.py
├── metrics.py
├── requirements.txt
├── EVALUATION_FILES.md        # Technical docs
├── GRADER_GUIDANCE.md         # Grader instructions
└── README.md                  # This file
```
