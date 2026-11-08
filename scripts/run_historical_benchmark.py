"""CLI script to run and verify the historical AI_Biomed benchmark.

Reproduces the reported metrics:
- Dataset size: 1,659 samples
- Clinical features: 37
- BioBERT embedding dimensions: 768
- Logistic Regression accuracy: 75.30%
- Random Forest accuracy: 93.20%
- XGBoost accuracy: 96.00%
- XGBoost F1-score: 0.9498
- XGBoost ROC-AUC: 0.9982
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from ai_biomed.config import PathConfig
from ai_biomed.evaluation.benchmark import run_historical_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def main():
    path_cfg = PathConfig()
    metrics_df, comp_df = run_historical_benchmark(path_config=path_cfg, verbose=True)
    
    # Save results to disk
    out_dir = path_cfg.results_dir
    metrics_df.to_csv(out_dir / "historical_reproduction_metrics.csv", index=False)
    comp_df.to_csv(out_dir / "historical_verification_table.csv", index=False)
    print(f"[OK] Historical benchmark results saved to: {out_dir}")

if __name__ == "__main__":
    main()
