"""
Pydantic schemas — all data contracts for the platform.
Every agent input/output, API payload, and event type is defined here.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ── Enums ────────────────────────────────────────────────────────────────────


class AgentStatus(str, Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    PROGRESS = "PROGRESS"
    WARNING = "WARNING"
    RETRYING = "RETRYING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ColumnType(str, Enum):
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    ORDINAL_CATEGORICAL = "ordinal_categorical"
    NOMINAL_CATEGORICAL = "nominal_categorical"


class AnomalySeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ── Agent Events (WebSocket payload) ─────────────────────────────────────────


class AgentEvent(BaseModel):
    run_id: str
    agent: str
    status: AgentStatus
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    rows_in: Optional[int] = None
    rows_out: Optional[int] = None
    duration_ms: Optional[int] = None
    progress_pct: Optional[float] = None
    metric_key: Optional[str] = None
    metric_value: Optional[float] = None
    traceback: Optional[str] = None


# ── Agent 1: Ingestion ───────────────────────────────────────────────────────


class IngestionInput(BaseModel):
    csv_path: str = "/app/data/AmesHousing.csv"
    expected_rows: int = 2930
    expected_cols: int = 82


class IngestionOutput(BaseModel):
    row_count: int
    col_count: int
    dataset_hash: str
    is_rerun: bool = False
    file_size_bytes: int
    encoding_detected: str
    ingestion_ts: datetime = Field(default_factory=datetime.utcnow)
    columns: List[str] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ── Agent 2: Schema Validation ───────────────────────────────────────────────


class ColumnInfo(BaseModel):
    name: str
    matched_name: str
    data_type: ColumnType
    null_rate: float
    unique_count: int
    is_structural_na: bool = False


class SchemaReport(BaseModel):
    total_columns: int
    numeric_continuous: int
    numeric_discrete: int
    ordinal_categorical: int
    nominal_categorical: int
    columns: List[ColumnInfo] = []
    sale_price_valid: bool = True


class SchemaOutput(BaseModel):
    schema_report: SchemaReport
    null_rates: Dict[str, float] = {}
    column_type_map: Dict[str, ColumnType] = {}
    schema_confidence_score: float = 1.0
    structural_na_candidates: List[str] = []
    column_name_map: Dict[str, str] = {}  # original -> standardized
    row_count: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)  #


# ── Agent 3: Cleaning ────────────────────────────────────────────────────────


class ImputationRecord(BaseModel):
    column: str
    method: str
    rows_affected: int
    fill_value: Optional[str] = None


class ArtifactFlag(BaseModel):
    index: int
    pid: Optional[str] = None
    reason: str
    gr_liv_area: float
    sale_price: float


class CleaningReport(BaseModel):
    rows_in: int
    rows_out: int
    rows_dropped: int
    structural_na_fills: Dict[str, int] = {}
    imputed_cols: Dict[str, ImputationRecord] = {}
    string_fixes: int = 0
    artifact_flags: List[ArtifactFlag] = []
    post_clean_null_rate: float = 0.0


class CleaningOutput(BaseModel):
    cleaning_report: CleaningReport
    row_count: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)  #


# ── Agent 4: Feature Engineering ─────────────────────────────────────────────


class FeatureManifestEntry(BaseModel):
    name: str
    formula: str
    rationale: str
    version: str = "1.0"
    pearson_correlation: float = 0.0
    created_by: str = "feature_agent_v1.0"


class FeatureOutput(BaseModel):
    features_created: int
    feature_manifest: List[FeatureManifestEntry] = []
    total_columns: int = 0
    row_count: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)  #


# ── Agent 5: Encoding & Scaling ──────────────────────────────────────────────


class EncodingOutput(BaseModel):
    final_shape: List[int] = []  # [rows, cols]
    ordinal_cols_encoded: int = 0
    target_encoded_cols: int = 0
    ohe_cols_encoded: int = 0
    ohe_features_created: int = 0
    log_transformed_cols: List[str] = []
    scaled_cols: int = 0
    artifacts_saved: List[str] = []
    row_count: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)  #


# ── Agent 6: Anomaly Detection ───────────────────────────────────────────────


class AnomalyDetail(BaseModel):
    feature: str
    value: float
    z_score: Optional[float] = None
    is_outlier_zscore: bool = False
    is_outlier_iforest: bool = False


class AnomalyRecord(BaseModel):
    pid: str
    neighborhood: str
    methods: List[str] = []
    anomalous_features: Dict[str, AnomalyDetail] = {}
    isolation_score: float = 0.0
    overall_severity: AnomalySeverity = AnomalySeverity.LOW


class AnomalyReport(BaseModel):
    total_flagged: int = 0
    pct_of_dataset: float = 0.0
    isolation_forest_flags: int = 0
    zscore_flags: int = 0
    both_methods_flags: int = 0
    flagged_records: List[AnomalyRecord] = []


class AnomalyOutput(BaseModel):
    anomaly_report: AnomalyReport
    row_count: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)  #


# ── Agent 7: ML Training ────────────────────────────────────────────────────


class ModelMetrics(BaseModel):
    split: str
    rmse_dollars: float
    mae_dollars: float
    r2: float
    mape: float
    rmse_log: float


class ModelResult(BaseModel):
    model_name: str
    val_metrics: ModelMetrics
    test_metrics: ModelMetrics
    is_best: bool = False
    mlflow_run_id: Optional[str] = None
    hyperparameters: Dict[str, Any] = {}


class MLTrainingOutput(BaseModel):
    models_trained: int
    best_model_name: str
    best_test_r2: float
    best_test_rmse: float
    model_results: List[ModelResult] = []
    shap_artifacts: List[str] = []
    row_count: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)  #


# ── Agent 8: Orchestration ───────────────────────────────────────────────────


class PipelineResult(BaseModel):
    run_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    agents_completed: List[str] = []
    agents_failed: List[str] = []
    best_model: Optional[str] = None
    best_r2: Optional[float] = None
    best_rmse: Optional[float] = None
    rows_processed: Optional[int] = None
    features_count: Optional[int] = None
    anomalies_count: Optional[int] = None
    knowledge_chunks: Optional[int] = None


# ── Processed Property (Final Output) ────────────────────────────────────────


class QualityFlags(BaseModel):
    missing_imputed: bool = False
    anomaly_flagged: bool = False
    anomaly_severity: Optional[AnomalySeverity] = None
    schema_valid: bool = True
    confidence_score: float = 1.0
    artifact_flag: bool = False


class LineageMetadata(BaseModel):
    dataset_hash: str
    pipeline_run_id: str
    agent_versions: Dict[str, str] = {}
    transformation_steps: List[str] = []
    encoder_artifact_path: str = ""


class ProcessedProperty(BaseModel):
    property_id: str
    sale_price: Optional[float] = None
    numerical_features: List[float] = []
    categorical_features: List[int] = []
    engineered_features: List[float] = []
    quality_flags: QualityFlags = QualityFlags()
    lineage_metadata: Optional[LineageMetadata] = None
    timestamp_processed: datetime = Field(default_factory=datetime.utcnow)


# ── API Schemas ──────────────────────────────────────────────────────────────


class RunPipelineResponse(BaseModel):
    run_id: str
    message: str = "Pipeline started"


class PipelineStatusResponse(BaseModel):
    run_id: str
    status: str
    progress_pct: float = 0.0
    agents: Dict[str, str] = {}
    metrics: Dict[str, Any] = {}
    started_at: Optional[datetime] = None
    duration_ms: Optional[int] = None


class PredictRequest(BaseModel):
    overall_qual: int = Field(ge=1, le=10)
    gr_liv_area: float = Field(gt=0)
    year_built: int = Field(ge=1800, le=2030)
    total_bathrooms: float = Field(ge=0)
    neighborhood: str = "NAmes"
    garage_area: float = 0
    total_bsmt_sf: float = 0
    first_flr_sf: float = 0
    full_bath: int = 1
    fireplace_qu: str = "None"


class PredictResponse(BaseModel):
    predicted_price: float
    confidence_interval: List[float] = []
    shap_top_features: Dict[str, float] = {}
    similar_properties: List[Dict[str, Any]] = []


class BatchPredictRequest(BaseModel):
    properties: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]


class KnowledgeBaseStatus(BaseModel):
    chunk_count: int = 0
    document_count: int = 0
    last_updated: Optional[datetime] = None
    documents: List[str] = []
