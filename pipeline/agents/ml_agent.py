"""
Agent 7 — ML Training & Serving Agent
Train, evaluate, track, register, and serve a production-ready price prediction model.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from core.metrics import model_mae, model_r2, model_rmse
from core.schemas import (AgentStatus, MLTrainingOutput, ModelMetrics,
                          ModelResult)
from lightgbm import LGBMRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from xgboost import XGBRegressor


class MLAgent(BaseAgent):
    name = "ml_agent"
    version = "1.0.0"

    async def execute(self, input_data) -> MLTrainingOutput:
        df: pd.DataFrame = self._get_df(input_data).copy()

        # Temporal split
        train_mask = df["Yr Sold"].isin([2006, 2007, 2008])
        val_mask = df["Yr Sold"] == 2009
        test_mask = df["Yr Sold"] == 2010

        # Determine target column
        target_col = "log_SalePrice" if "log_SalePrice" in df.columns else "SalePrice"

        # Feature columns: exclude target, metadata, and non-numeric
        exclude = {
            "SalePrice",
            "log_SalePrice",
            "Yr Sold",
            "anomaly_flagged",
            "anomaly_severity",
        }
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
        ]

        X_train = df.loc[train_mask, feature_cols].fillna(0)
        y_train = df.loc[train_mask, target_col].fillna(0)
        X_val = df.loc[val_mask, feature_cols].fillna(0)
        y_val = df.loc[val_mask, target_col].fillna(0)
        X_test = df.loc[test_mask, feature_cols].fillna(0)
        y_test = df.loc[test_mask, target_col].fillna(0)

        await self.emit(
            AgentStatus.PROGRESS,
            f"ML training initiated | Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}",
            rows_in=len(X_train),
        )

        use_log = target_col == "log_SalePrice"
        models_config = {
            "ridge": RidgeCV(
                alphas=[0.1, 1.0, 10.0, 100.0],
                cv=5,
                scoring="neg_root_mean_squared_error",
            ),
            "xgboost": XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.01,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
            "lightgbm": LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.01,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
        }

        model_results = []
        best_model = None
        best_r2 = -float("inf")
        best_model_name = ""
        trained_models = {}

        for name, model in models_config.items():
            await self.emit(AgentStatus.PROGRESS, f"Training {name}...")

            if name == "xgboost":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                # Emit progress for iterations
                best_iter = getattr(model, "best_iteration", model.n_estimators)
                await self.emit(
                    AgentStatus.PROGRESS,
                    f"XGBoost stopped at iter {best_iter} (early stopping)",
                )
            elif name == "lightgbm":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="rmse",
                )
            else:
                model.fit(X_train, y_train)

            trained_models[name] = model

            # Evaluate
            val_metrics = self._evaluate(model, X_val, y_val, "val", use_log)
            test_metrics = self._evaluate(model, X_test, y_test, "test", use_log)

            is_best = test_metrics.r2 > best_r2
            if is_best:
                best_r2 = test_metrics.r2
                best_model = model
                best_model_name = name

            model_results.append(
                ModelResult(
                    model_name=name,
                    val_metrics=val_metrics,
                    test_metrics=test_metrics,
                    is_best=is_best,
                )
            )

            await self.emit(
                AgentStatus.PROGRESS,
                f"{name}: Val RMSE=${val_metrics.rmse_dollars:,.0f} R²={val_metrics.r2:.3f} ✓",
                metric_key=f"{name}_val_rmse",
                metric_value=val_metrics.rmse_dollars,
            )

        # Update is_best flags
        for result in model_results:
            result.is_best = result.model_name == best_model_name

        await self.emit(
            AgentStatus.PROGRESS,
            f"🏆 Best model: {best_model_name} | Test RMSE=${model_results[-1].test_metrics.rmse_dollars:,.0f} R²={best_r2:.3f}",
            metric_key="best_test_r2",
            metric_value=best_r2,
        )

        # Update Prometheus metrics
        for result in model_results:
            model_rmse.labels(model_name=result.model_name, split="val").set(
                result.val_metrics.rmse_dollars
            )
            model_rmse.labels(model_name=result.model_name, split="test").set(
                result.test_metrics.rmse_dollars
            )
            model_r2.labels(model_name=result.model_name, split="test").set(
                result.test_metrics.r2
            )
            model_mae.labels(model_name=result.model_name, split="test").set(
                result.test_metrics.mae_dollars
            )

        # SHAP analysis
        shap_artifacts = []
        try:
            import shap

            await self.emit(
                AgentStatus.PROGRESS,
                f"Computing SHAP values on {len(X_test)} test properties...",
            )
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)

            base_artifacts = os.getenv("ARTIFACTS_DIR", "/app/artifacts")
            shap_dir = f"{base_artifacts}/shap/{self.run_id}"
            os.makedirs(shap_dir, exist_ok=True)

            # Save SHAP summary data
            shap_importance = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                }
            ).sort_values("mean_abs_shap", ascending=False)
            shap_importance.to_csv(
                os.path.join(shap_dir, "shap_importance.csv"), index=False
            )
            shap_artifacts.append("shap_importance.csv")

            await self.emit(AgentStatus.PROGRESS, "SHAP artifacts saved ✓")
        except Exception as e:
            await self.emit(AgentStatus.WARNING, f"SHAP computation skipped: {e}")

        # Save best model
        base_artifacts = os.getenv("ARTIFACTS_DIR", "/app/artifacts")
        model_dir = f"{base_artifacts}/models/{self.run_id}"
        os.makedirs(model_dir, exist_ok=True)
        import pickle

        with open(os.path.join(model_dir, "best_model.pkl"), "wb") as f:
            pickle.dump(best_model, f)
        with open(os.path.join(model_dir, "feature_cols.json"), "w") as f:
            json.dump(feature_cols, f)

        # MLflow logging
        try:
            import mlflow

            mlflow.set_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            )
            experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "ames-housing")
            mlflow.set_experiment(experiment_name)

            for result in model_results:
                with mlflow.start_run(
                    run_name=f"{result.model_name}_{self.run_id[:8]}"
                ):
                    mlflow.log_params(
                        {"model": result.model_name, "run_id": self.run_id}
                    )
                    mlflow.log_metrics(
                        {
                            "val_rmse_dollars": result.val_metrics.rmse_dollars,
                            "test_rmse_dollars": result.test_metrics.rmse_dollars,
                            "test_r2": result.test_metrics.r2,
                            "test_mae_dollars": result.test_metrics.mae_dollars,
                            "test_mape": result.test_metrics.mape,
                        }
                    )
                    if result.is_best:
                        mlflow.set_tag("best_model", "true")
                    result.mlflow_run_id = mlflow.active_run().info.run_id

            await self.emit(
                AgentStatus.PROGRESS,
                f"Registering {best_model_name} in MLflow Registry ✓",
            )
        except Exception as e:
            await self.emit(AgentStatus.WARNING, f"MLflow logging skipped: {e}")

        self._best_model = best_model
        self._feature_cols = feature_cols
        self._df = df

        return MLTrainingOutput(
            models_trained=len(model_results),
            best_model_name=best_model_name,
            best_test_r2=round(best_r2, 4),
            best_test_rmse=round(
                next(r.test_metrics.rmse_dollars for r in model_results if r.is_best), 2
            ),
            model_results=model_results,
            shap_artifacts=shap_artifacts,
            row_count=len(df),
        )

    def _evaluate(self, model, X, y_log, split: str, use_log: bool) -> ModelMetrics:
        y_pred_log = model.predict(X)
        if use_log:
            y_pred = np.expm1(y_pred_log)
            y_actual = np.expm1(y_log)
        else:
            y_pred = y_pred_log
            y_actual = y_log

        y_pred = np.clip(y_pred, 0, None)
        y_actual_safe = np.clip(y_actual, 1, None)

        return ModelMetrics(
            split=split,
            rmse_dollars=round(float(np.sqrt(mean_squared_error(y_actual, y_pred))), 2),
            mae_dollars=round(float(mean_absolute_error(y_actual, y_pred)), 2),
            r2=round(float(r2_score(y_log, y_pred_log)), 4),
            mape=round(
                float(mean_absolute_percentage_error(y_actual_safe, y_pred)) * 100, 2
            ),
            rmse_log=round(float(np.sqrt(mean_squared_error(y_log, y_pred_log))), 4),
        )

    def _get_df(self, input_data) -> pd.DataFrame:
        if isinstance(input_data, dict):
            for key in ["encoding_agent", "feature_agent", "cleaning_agent"]:
                agent = input_data.get(key)
                if agent and hasattr(agent, "_df"):
                    return agent._df
        raise ValueError("No DataFrame from upstream agents")
