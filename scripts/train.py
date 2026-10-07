"""Train all artifacts: risk model + metrics, clusters, anomaly detector."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.anomaly import fit_anomaly, save_anomaly
from src.clustering import fit_clusters, save_cluster
from src.credit_risk import save_metrics, save_model, train_and_select
from src.data_loader import load_data, validate_data
from src.logging_utils import get_logger

log = get_logger("train")


def main() -> None:
    df = load_data()
    problems = validate_data(df)
    if problems:
        log.warning("Data quality notes: %s", problems)
    best, metrics, info = train_and_select(df)
    save_model(best)
    save_metrics(metrics, info)
    cpipe, _, cinfo = fit_clusters(df)
    save_cluster(cpipe, cinfo["profiles"], cinfo["metrics"])
    apipe, _ = fit_anomaly(df)
    save_anomaly(apipe)
    log.info("Selected: %s | metrics -> %s", info, config.RISK_METRICS_PATH)


if __name__ == "__main__":
    main()
