"""
Agent 6 — Anomaly Detection Agent
Surface statistically unusual properties for human review — never auto-remove.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from core.metrics import anomalies_detected_total
from core.schemas import (AgentStatus, AnomalyDetail, AnomalyOutput,
                          AnomalyRecord, AnomalyReport, AnomalySeverity)
from scipy import stats
from sklearn.ensemble import IsolationForest

ANOMALY_FEATURES = [
    "log_Gr Liv Area",
    "Total Bsmt SF",
    "log_Lot Area",
    "log_SalePrice",
    "log_TotalSF",
    "Overall Qual",
]
# Configuration thresholds (now configurable via environment variables)
ZSCORE_THRESHOLD = float(os.getenv("ANOMALY_ZSCORE_THRESHOLD", "3.5"))
ISO_SCORE_HIGH_SEVERITY = float(os.getenv("ANOMALY_ISO_SCORE_HIGH", "-0.3"))
ISO_SCORE_MEDIUM_SEVERITY = float(os.getenv("ANOMALY_ISO_SCORE_MEDIUM", "-0.1"))


class AnomalyAgent(BaseAgent):
    name = "anomaly_agent"
    version = "1.0.0"

    async def execute(self, input_data: Dict[str, Any], df: pd.DataFrame = None) -> tuple[AnomalyOutput, pd.DataFrame]:
        if df is None:
            raise ValueError("No DataFrame provided to AnomalyAgent")
            
        await self.emit(
            AgentStatus.PROGRESS, f"Running anomaly detection on {len(df):,} properties"
        )

        contamination = float(os.getenv("ANOMALY_CONTAMINATION", "0.02"))
        features_present = [f for f in ANOMALY_FEATURES if f in df.columns]
        X = df[features_present].fillna(0).values

        # Isolation Forest
        iforest = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        iforest_preds = iforest.fit_predict(X)
        iforest_scores = iforest.decision_function(X)
        iforest_flags = set(df.index[iforest_preds == -1])

        await self.emit(
            AgentStatus.PROGRESS,
            f"Isolation Forest (contamination={contamination}): {len(iforest_flags)} properties flagged",
        )

        # Z-score flagging
        z_scores = np.abs(stats.zscore(X, nan_policy="omit"))
        zscore_flag_mask = (z_scores > ZSCORE_THRESHOLD).any(axis=1)
        zscore_flags = set(df.index[zscore_flag_mask])

        await self.emit(
            AgentStatus.PROGRESS,
            f"Z-score analysis (|z|>{ZSCORE_THRESHOLD}): {len(zscore_flags)} properties flagged",
        )

        # Combine results
        all_flagged = iforest_flags | zscore_flags
        both_flagged = iforest_flags & zscore_flags

        await self.emit(
            AgentStatus.PROGRESS,
            f"Intersection: {len(both_flagged)} flagged by both methods (HIGH severity)",
        )

        # Build anomaly records
        flagged_records = []
        for idx in all_flagged:
            row = df.loc[idx]
            methods = []
            if idx in iforest_flags:
                methods.append("isolation_forest")
            if idx in zscore_flags:
                methods.append("z_score")

            iso_score = float(iforest_scores[df.index.get_loc(idx)])

            # Severity classification (using configurable thresholds)
            if idx in both_flagged and iso_score < ISO_SCORE_HIGH_SEVERITY:
                severity = AnomalySeverity.HIGH
            elif len(methods) == 1 or ISO_SCORE_HIGH_SEVERITY <= iso_score < ISO_SCORE_MEDIUM_SEVERITY:
                severity = AnomalySeverity.MEDIUM
            else:
                severity = AnomalySeverity.LOW

            anomalous_features = {}
            row_idx = df.index.get_loc(idx)
            for i, feat in enumerate(features_present):
                z = float(z_scores[row_idx, i]) if row_idx < len(z_scores) else 0.0
                anomalous_features[feat] = AnomalyDetail(
                    feature=feat,
                    value=float(row[feat]),
                    z_score=round(z, 3),
                    is_outlier_zscore=z > ZSCORE_THRESHOLD,
                    is_outlier_iforest=idx in iforest_flags,
                )

            pid_val = str(row.get("PID_orig", row.get("PID", idx)))
            neighborhood = str(row.get("Neighborhood_orig", row.get("Neighborhood", "Unknown")))

            flagged_records.append(
                AnomalyRecord(
                    pid=pid_val,
                    neighborhood=neighborhood,
                    methods=methods,
                    anomalous_features=anomalous_features,
                    isolation_score=round(iso_score, 4),
                    overall_severity=severity,
                )
            )

        # Update Prometheus gauge
        anomalies_detected_total.set(len(flagged_records))

        # Attach flags to DataFrame (never remove)
        df["anomaly_flagged"] = False
        df["anomaly_severity"] = "NONE"

        for idx, record in zip(all_flagged, flagged_records):
            if idx in df.index:
                df.at[idx, "anomaly_flagged"] = True
                df.at[idx, "anomaly_severity"] = record.overall_severity.value

        pct = (len(flagged_records) / len(df)) * 100

        report = AnomalyReport(
            total_flagged=len(flagged_records),
            pct_of_dataset=round(pct, 2),
            isolation_forest_flags=len(iforest_flags),
            zscore_flags=len(zscore_flags),
            both_methods_flags=len(both_flagged),
            flagged_records=flagged_records,
        )

        output = AnomalyOutput(anomaly_report=report, row_count=len(df))
        return output, df
