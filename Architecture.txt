MOLECULAR PROPERTY PREDICTION WEB SERVICE
=========================================

┌─────────────────┐    ┌─────────────────────────────────┐
│   Streamlit     │◄──►│      Flask API (main.py)        │
│     Frontend    │    │  • LLM Natural Language Layer   │
│   (pages/)      │    │  • Request Routing              │
└─────────────────┘    └──────────────┬──────────────────┘
                                      │
               ┌──────────────────────┼──────────────────────┐
               ▼                      ▼                      ▼
┌─────────────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Model Predictor       │ │  SHAP Analyzer  │ │   RDKit Utils   │
│  • model_prediction.py  │ │  (shapley.py)   │ │ (formula_mol.py)│
│  • model_sav_prediction │ │                 │ │                 │
└─────────────┬───────────┘ └─────────────────┘ └─────────────────┘
              │
              ▼
    ┌──────────────────────┐
    │   Ensemble Models    │
    │  • BBBP              │
    │  • ClinTox           │
    │  • ESOL              │
    │  • FreeSolvSAMPL     │
    │  • HIV               │
    │  • Lipophilicity     │
    │  • base/             │
    └──────────────────────┘

=========================================

DATA FLOW:
User → SMILES → RDKit → Models → SHAP → LLM → Results

INFRASTRUCTURE:
┌─────────────────────────┐
│    Docker Container     │
│  • compose/api/         │
│  • requirements.txt     │
│  • environment.yml      │
│  • entrypoint.sh        │
└─────────────────────────┘

TECH: Python, scikit-learn, SHAP, RDKit, Flask, Docker
