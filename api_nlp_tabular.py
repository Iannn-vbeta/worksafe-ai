"""

WorkSafe AI - FastAPI untuk 2 model sekaligus:
1. Model NLP: input title, description, top_skills, top_core_tasks, numeric_features
2. Model Tabular: input features berisi nilai fitur numerik tabular

Struktur folder:
project/
├── api_nlp_tabular.py
└── models/
    ├── nlp_model/
    │   ├── worksafe_risk_model_best.keras
    │   ├── worksafe_risk_model.keras              # opsional fallback
    │   └── worksafe_artifacts.json
    ├── tabular_model/
    │   ├── worksafe_model_v1.keras
    │   ├── scaler.pkl
    │   ├── imputer.pkl
    │   └── feature_cols.pkl

Install:
bikin venv lalu
pip install fastapi uvicorn tensorflow numpy pandas scikit-learn openrouter

Jalankan:
uvicorn api_nlp_tabular:app --reload --host 0.0.0.0 --port 8000

Docs:
http://127.0.0.1:8000/docs
"""

import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow import keras

# CUSTOM OBJECTS

@tf.keras.utils.register_keras_serializable(package="WorkSafeAI", name="AttentionPooling")
class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.score_dense = tf.keras.layers.Dense(units, activation="tanh")
        self.attention_score = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        scores = self.attention_score(self.score_dense(inputs))
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(inputs * weights, axis=1)
        return context

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


@tf.keras.utils.register_keras_serializable(package="WorkSafeAI", name="FocalLoss")
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name="focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_oh = tf.cast(tf.one_hot(y_true, depth=3), tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        ce = -y_true_oh * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        loss = self.alpha * focal_weight * tf.reduce_sum(ce, axis=-1)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


# Compatibility layer untuk model .keras yang disimpan dengan Keras lebih baru.
# Beberapa versi Keras menyimpan argumen `quantization_config` pada Embedding,
# sedangkan environment lama belum mengenalinya. Class ini mengabaikan argumen itu
# agar model tetap bisa diload untuk inference.
@tf.keras.utils.register_keras_serializable(package="keras.layers", name="Embedding")
class CompatibleEmbedding(tf.keras.layers.Embedding):
    def __init__(self, *args, quantization_config=None, **kwargs):
        self.quantization_config = quantization_config
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        # Jangan masukkan lagi ke config agar kompatibel dengan versi lama.
        config.pop("quantization_config", None)
        return config

# PATH CONFIG

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# NLP model paths
NLP_EXPORT_DIR = MODELS_DIR / "nlp_model"
NLP_MODEL_PATH = NLP_EXPORT_DIR / "worksafe_risk_model_best.keras"
NLP_FALLBACK_MODEL_PATH = NLP_EXPORT_DIR / "worksafe_risk_model.keras"
NLP_ARTIFACT_PATH = NLP_EXPORT_DIR / "worksafe_artifacts.json"

# Tabular model paths
TABULAR_EXPORT_DIR = MODELS_DIR / "tabular_model"
TABULAR_MODEL_PATH = TABULAR_EXPORT_DIR / "worksafe_model_v1.keras"
TABULAR_SCALER_PATH = TABULAR_EXPORT_DIR / "scaler.pkl"
TABULAR_IMPUTER_PATH = TABULAR_EXPORT_DIR / "imputer.pkl"
TABULAR_FEATURES_PATH = TABULAR_EXPORT_DIR / "feature_cols.pkl"

# GLOBAL OBJECTS

app = FastAPI(
    title="WorkSafe AI - NLP + Tabular Prediction API",
    description=(
        "API gabungan untuk menjalankan model NLP dan model tabular dalam satu server. "
        "Gunakan /predict untuk NLP saja, /predict-tabular untuk tabular saja, "
        "dan /predicts untuk menjalankan dua model sekaligus."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk production, ubah ke domain frontend kamu.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NLP_MODEL = None
NLP_ARTIFACTS: Optional[Dict[str, Any]] = None

TABULAR_MODEL = None
TABULAR_SCALER = None
TABULAR_IMPUTER = None
TABULAR_FEATURE_COLS = None
TABULAR_MEDIAN_VALUES: Dict[str, float] = {}

TABULAR_LABEL_MAP = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
}

# SCHEMAS

class NLPPredictRequest(BaseModel):
    title: str = Field(..., example="Warehouse Administration Staff")
    description: str = Field(
        default="",
        example="Responsible for data entry, inventory documentation, warehouse reports, and repetitive administrative tasks.",
    )
    top_skills: str = Field(
        default="",
        example="data entry, inventory management, spreadsheet, reporting",
    )
    top_core_tasks: str = Field(
        default="",
        example="record inventory movement, prepare reports, check warehouse documents",
    )
    numeric_features: Optional[Dict[str, float]] = Field(
        default=None,
        example={},
        description="Opsional untuk model NLP. Jika kosong, memakai median dari worksafe_artifacts.json.",
    )
    generate_reskilling: bool = Field(
        default=True,
        description="Jika true, API akan mencoba membuat rekomendasi reskilling via OpenRouter.",
    )
    openrouter_model: str = Field(
        default="meta-llama/llama-3.3-70b-instruct:free",
        description="Nama model OpenRouter.",
    )


class TabularPredictRequest(BaseModel):
    features: Optional[Dict[str, float]] = Field(
        default=None,
        example={
            "act_working_with_computers": 4,
            "skl_critical_thinking": 4,
            "skl_complex_problem_solving": 3,
        },
        description=(
            "Fitur numerik untuk model tabular. "
            "Boleh hanya mengirim sebagian fitur; fitur yang kosong akan diisi median training."
        ),
    )


class CombinedPredictRequest(NLPPredictRequest):
    features: Optional[Dict[str, float]] = Field(
        default=None,
        example={
            "act_working_with_computers": 4,
            "skl_critical_thinking": 4,
            "skl_complex_problem_solving": 3,
        },
        description=(
            "Fitur tabular. Ini berbeda dari numeric_features. "
            "features digunakan model tabular, numeric_features digunakan model NLP."
        ),
    )
    tabular_features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Alias opsional untuk features. Jika features kosong, nilai ini akan dipakai.",
    )


class HealthResponse(BaseModel):
    status: str
    nlp_model_loaded: bool
    nlp_artifacts_loaded: bool
    tabular_model_loaded: bool
    tabular_artifacts_loaded: bool
    paths: Dict[str, str]

# LOADERS

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_nlp_model():
    selected_model_path = NLP_MODEL_PATH

    if not selected_model_path.exists() and NLP_FALLBACK_MODEL_PATH.exists():
        selected_model_path = NLP_FALLBACK_MODEL_PATH

    if not selected_model_path.exists():
        raise FileNotFoundError(
            f"Model NLP tidak ditemukan. Dicari di: {NLP_MODEL_PATH} atau {NLP_FALLBACK_MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        selected_model_path,
        custom_objects={
            "AttentionPooling": AttentionPooling,
            "WorkSafeAI>AttentionPooling": AttentionPooling,
            "Embedding": CompatibleEmbedding,
            "keras.layers.Embedding": CompatibleEmbedding,
            "keras.src.layers.core.embedding.Embedding": CompatibleEmbedding,
        },
        compile=False,
    )


def load_tabular_artifacts():
    missing_paths = [
        path for path in [
            TABULAR_MODEL_PATH,
            TABULAR_SCALER_PATH,
            TABULAR_IMPUTER_PATH,
            TABULAR_FEATURES_PATH,
        ]
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Artifact tabular belum lengkap: "
            + ", ".join(str(path) for path in missing_paths)
        )

    tabular_model = keras.models.load_model(
        TABULAR_MODEL_PATH,
        custom_objects={
            "FocalLoss": FocalLoss,
            "WorkSafeAI>FocalLoss": FocalLoss,
            "Embedding": CompatibleEmbedding,
            "keras.layers.Embedding": CompatibleEmbedding,
            "keras.src.layers.core.embedding.Embedding": CompatibleEmbedding,
        },
        compile=False,
    )

    with open(TABULAR_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(TABULAR_IMPUTER_PATH, "rb") as f:
        imputer = pickle.load(f)

    with open(TABULAR_FEATURES_PATH, "rb") as f:
        feature_cols = pickle.load(f)

    return tabular_model, scaler, imputer, list(feature_cols)


@app.on_event("startup")
def startup_event():
    global NLP_MODEL, NLP_ARTIFACTS
    global TABULAR_MODEL, TABULAR_SCALER, TABULAR_IMPUTER
    global TABULAR_FEATURE_COLS, TABULAR_MEDIAN_VALUES

    print("Load NLP artifacts...")
    NLP_ARTIFACTS = load_json(NLP_ARTIFACT_PATH)

    print("Load NLP model...")
    NLP_MODEL = load_nlp_model()

    print("Load tabular model and artifacts...")
    (
        TABULAR_MODEL,
        TABULAR_SCALER,
        TABULAR_IMPUTER,
        TABULAR_FEATURE_COLS,
    ) = load_tabular_artifacts()

    TABULAR_MEDIAN_VALUES = {
        col: float(TABULAR_IMPUTER.statistics_[i])
        for i, col in enumerate(TABULAR_FEATURE_COLS)
    }

    print("Semua model dan artifact berhasil diload.")
    print("NLP input:", [inp.name for inp in NLP_MODEL.inputs])
    print("NLP output:", [out.name for out in NLP_MODEL.outputs])
    print(f"Jumlah fitur tabular: {len(TABULAR_FEATURE_COLS)}")

# NLP HELPERS

def build_job_text(
    title: str,
    description: str = "",
    top_skills: str = "",
    top_core_tasks: str = "",
) -> str:
    return (
        f"Job Title: {title}. "
        f"Description: {description}. "
        f"Top Skills: {top_skills}. "
        f"Core Tasks: {top_core_tasks}."
    )


def build_nlp_numeric_array(
    artifacts: Dict[str, Any],
    numeric_features: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    numeric_features = numeric_features or {}

    numeric_cols = artifacts.get("numeric_cols", [])
    numeric_medians = artifacts.get("numeric_medians", {})

    values = []
    for col in numeric_cols:
        value = numeric_features.get(col, numeric_medians.get(col, 0.0))
        values.append(float(value))

    return np.array([values], dtype="float32")


def extract_nlp_outputs(prediction):
    """
    Output model NLP kadang berupa list/tuple, kadang dict.
    Fungsi ini mencari:
    - pred_label: output klasifikasi dengan dimensi terakhir > 1
    - pred_score: output regresi skor risiko dengan dimensi terakhir == 1
    """
    if isinstance(prediction, dict):
        values = list(prediction.values())
    elif isinstance(prediction, (list, tuple)):
        values = list(prediction)
    else:
        raise ValueError("Output model NLP tidak dikenali.")

    if len(values) < 2:
        raise ValueError("Model NLP harus memiliki minimal 2 output: label dan score.")

    arrays = [np.asarray(v) for v in values]

    label_candidates = [
        arr for arr in arrays
        if arr.ndim >= 2 and arr.shape[-1] > 1
    ]
    score_candidates = [
        arr for arr in arrays
        if arr.ndim >= 2 and arr.shape[-1] == 1
    ]

    pred_label = label_candidates[0] if label_candidates else arrays[0]
    pred_score = score_candidates[0] if score_candidates else arrays[1]

    return pred_label, pred_score


def parse_ai_json(text: str) -> Dict[str, Any]:
    if not text:
        return {
            "error": "AI response kosong.",
            "raw_response": text,
        }

    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "error": "Response AI bukan JSON valid.",
        "raw_response": text,
    }


def generate_reskilling_with_openrouter(
    title: str,
    description: str,
    top_skills: str,
    top_core_tasks: str,
    risk_label: str,
    confidence: float,
    automation_risk_score: float,
    model_name: str = "meta-llama/llama-3.3-70b-instruct:free",
) -> Dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY", "")

    if not api_key:
        return {
            "recommendation_source": "openrouter_gen_ai",
            "openrouter_status": "failed",
            "error_type": "MissingAPIKey",
            "message": "OPENROUTER_API_KEY belum di-set di environment.",
        }

    try:
        from openrouter import OpenRouter
    except ImportError:
        return {
            "recommendation_source": "openrouter_gen_ai",
            "openrouter_status": "failed",
            "error_type": "ImportError",
            "message": "Package openrouter belum terinstall. Install dengan: pip install openrouter",
        }

    risk_percent = round(float(automation_risk_score) * 100, 2)

    input_payload = {
        "job_input": {
            "title": title,
            "description": description,
            "top_skills": top_skills,
            "top_core_tasks": top_core_tasks,
        },
        "model_prediction": {
            "risk_label": risk_label,
            "confidence": round(float(confidence), 4),
            "automation_risk_score": round(float(automation_risk_score), 4),
            "risk_percent": risk_percent,
        },
    }

    system_prompt = """
Kamu adalah AI career coach untuk aplikasi WorkSafe AI.
Tugasmu membuat rekomendasi reskilling pekerjaan berdasarkan input pekerjaan user dan hasil prediksi model NLP.

Balas HANYA dalam format JSON valid.
Jangan memakai markdown.
Jangan menambahkan penjelasan di luar JSON.
Gunakan bahasa Indonesia yang jelas, realistis, dan praktis untuk pekerja Indonesia.
Rekomendasi harus dibuat dinamis dari konteks input user.
""".strip()

    user_prompt = f"""
Buat rekomendasi reskilling dinamis berdasarkan data berikut:

{json.dumps(input_payload, indent=2, ensure_ascii=False)}

Format JSON yang wajib dikembalikan:

{{
  "recommendation_source": "openrouter_gen_ai",
  "risk_interpretation": "penjelasan singkat tentang arti level risiko pekerjaan ini",
  "main_reskilling_goal": "tujuan utama reskilling untuk user",
  "recommended_skills": [
    {{
      "skill": "nama skill",
      "reason": "alasan skill ini penting",
      "priority": "High/Medium/Low"
    }}
  ],
  "learning_roadmap": [
    {{
      "phase": "Minggu 1-2",
      "focus": "fokus belajar",
      "activities": ["aktivitas 1", "aktivitas 2"]
    }}
  ],
  "tools_to_learn": ["tool 1", "tool 2"],
  "mini_project_ideas": ["ide project 1", "ide project 2"],
  "career_transition_options": ["opsi karier 1", "opsi karier 2"],
  "estimated_learning_duration": "estimasi durasi belajar"
}}

Aturan:
- recommended_skills minimal 5 item.
- learning_roadmap minimal 3 fase.
- tools_to_learn minimal 4 item.
- mini_project_ideas minimal 3 item.
- career_transition_options minimal 3 item.
- Sesuaikan rekomendasi dengan risk_label dan automation_risk_score.
- Jangan memberi rekomendasi yang terlalu umum.
- Jawab hanya JSON valid.
""".strip()

    try:
        with OpenRouter(api_key=api_key) as client:
            response = client.chat.send(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

        message = response.choices[0].message
        content = message.content if hasattr(message, "content") else message["content"]

        parsed = parse_ai_json(content)
        if isinstance(parsed, dict):
            parsed.setdefault("recommendation_source", "openrouter_gen_ai")
            parsed.setdefault("openrouter_status", "success")
            parsed.setdefault("openrouter_model", model_name)

        return parsed

    except Exception as e:
        return {
            "recommendation_source": "openrouter_gen_ai",
            "openrouter_status": "failed",
            "openrouter_model": model_name,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "message": (
                "Prediksi risiko berhasil, tetapi rekomendasi reskilling gagal dibuat "
                "karena request OpenRouter gagal."
            ),
        }


def predict_nlp_core(payload: NLPPredictRequest) -> Dict[str, Any]:
    if NLP_MODEL is None or NLP_ARTIFACTS is None:
        raise HTTPException(
            status_code=503,
            detail="Model NLP atau artifact NLP belum berhasil diload.",
        )

    label_classes = NLP_ARTIFACTS.get("label_classes", [])

    job_text = build_job_text(
        title=payload.title,
        description=payload.description,
        top_skills=payload.top_skills,
        top_core_tasks=payload.top_core_tasks,
    )

    text_arr = np.array([job_text], dtype=object)
    numeric_arr = build_nlp_numeric_array(NLP_ARTIFACTS, payload.numeric_features)

    try:
        raw_prediction = NLP_MODEL.predict(
            {
                "job_text": text_arr,
                "numeric_features": numeric_arr,
            },
            verbose=0,
        )
    except Exception:
        raw_prediction = NLP_MODEL.predict(
            [text_arr, numeric_arr],
            verbose=0,
        )

    pred_label, pred_score = extract_nlp_outputs(raw_prediction)

    label_idx = int(np.argmax(pred_label[0]))
    risk_label = label_classes[label_idx] if label_classes else str(label_idx)
    confidence = float(pred_label[0][label_idx])
    risk_score = float(pred_score[0][0])

    result = {
        "status": "success",
        "model_type": "nlp",
        "input": {
            "title": payload.title,
            "description": payload.description,
            "top_skills": payload.top_skills,
            "top_core_tasks": payload.top_core_tasks,
        },
        "prediction": {
            "risk_label": risk_label,
            "confidence": round(confidence, 4),
            "automation_risk_score": round(risk_score, 4),
            "risk_percent": round(risk_score * 100, 2),
        },
    }

    if payload.generate_reskilling:
        result["reskilling_recommendation"] = generate_reskilling_with_openrouter(
            title=payload.title,
            description=payload.description,
            top_skills=payload.top_skills,
            top_core_tasks=payload.top_core_tasks,
            risk_label=risk_label,
            confidence=confidence,
            automation_risk_score=risk_score,
            model_name=payload.openrouter_model,
        )

    return result

# TABULAR HELPERS

def preprocess_tabular_input(input_dict: Dict[str, float]) -> np.ndarray:
    if TABULAR_SCALER is None or TABULAR_IMPUTER is None or TABULAR_FEATURE_COLS is None:
        raise HTTPException(
            status_code=503,
            detail="Artifact tabular belum berhasil diload.",
        )

    df = pd.DataFrame([input_dict])

    for col in TABULAR_FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[TABULAR_FEATURE_COLS]

    df_imp = pd.DataFrame(
        TABULAR_IMPUTER.transform(df),
        columns=TABULAR_FEATURE_COLS,
    )

    df_scaled = pd.DataFrame(
        TABULAR_SCALER.transform(df_imp),
        columns=TABULAR_FEATURE_COLS,
    )

    return df_scaled.values.astype(np.float32)


def predict_tabular_core(features: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    if TABULAR_MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model tabular belum berhasil diload.",
        )

    features = features or {}

    full_input = {
        col: TABULAR_MEDIAN_VALUES[col]
        for col in TABULAR_FEATURE_COLS
    }
    full_input.update(features)

    X = preprocess_tabular_input(full_input)
    proba = TABULAR_MODEL.predict(X, verbose=0)[0]
    kelas = int(np.argmax(proba))
    risk_label = TABULAR_LABEL_MAP.get(kelas, str(kelas))

    return {
        "status": "success",
        "model_type": "tabular",
        "input": {
            "provided_feature_count": len(features),
            "total_feature_count": len(TABULAR_FEATURE_COLS),
            "missing_features_filled_with_median": len(TABULAR_FEATURE_COLS) - len(features),
        },
        "prediction": {
            "risk_label": risk_label,
            "risk_class": kelas,
            "confidence": round(float(proba[kelas]) * 100, 2),
            "probabilities": {
                "Low Risk": round(float(proba[0]) * 100, 2),
                "Medium Risk": round(float(proba[1]) * 100, 2),
                "High Risk": round(float(proba[2]) * 100, 2),
            },
        },
    }

# COMBINED HELPERS

def normalize_risk_label(label: str) -> str:
    label_lower = str(label).lower()

    if "low" in label_lower or "rendah" in label_lower:
        return "Low"
    if "medium" in label_lower or "sedang" in label_lower:
        return "Medium"
    if "high" in label_lower or "tinggi" in label_lower:
        return "High"

    return str(label)


def build_combined_prediction(nlp_result: Dict[str, Any], tabular_result: Dict[str, Any]) -> Dict[str, Any]:
    nlp_label = normalize_risk_label(nlp_result["prediction"]["risk_label"])
    tabular_label = normalize_risk_label(tabular_result["prediction"]["risk_label"])

    risk_rank = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
    }

    nlp_rank = risk_rank.get(nlp_label, -1)
    tabular_rank = risk_rank.get(tabular_label, -1)

    if nlp_rank >= 0 and tabular_rank >= 0:
        final_label = nlp_label if nlp_rank >= tabular_rank else tabular_label
        agreement = nlp_label == tabular_label
    else:
        final_label = "Review Needed"
        agreement = False

    return {
        "merge_strategy": "conservative_max_risk",
        "agreement": agreement,
        "nlp_normalized_label": nlp_label,
        "tabular_normalized_label": tabular_label,
        "final_risk_label": final_label,
        "message": (
            "Hasil akhir memakai strategi konservatif: jika dua model berbeda, "
            "API mengambil level risiko yang lebih tinggi."
        ),
    }


# ROUTES

@app.get("/")
def root():
    return {
        "message": "WorkSafe AI NLP + Tabular API aktif",
        "docs": "/docs",
        "health": "/health",
        "nlp_predict": "/predict",
        "tabular_predict": "/predict-tabular",
        "combined_predict": "/predicts",
        "tabular_features": "/features",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "nlp_model_loaded": NLP_MODEL is not None,
        "nlp_artifacts_loaded": NLP_ARTIFACTS is not None,
        "tabular_model_loaded": TABULAR_MODEL is not None,
        "tabular_artifacts_loaded": all([
            TABULAR_SCALER is not None,
            TABULAR_IMPUTER is not None,
            TABULAR_FEATURE_COLS is not None,
        ]),
        "paths": {
            "nlp_model": str(NLP_MODEL_PATH if NLP_MODEL_PATH.exists() else NLP_FALLBACK_MODEL_PATH),
            "nlp_artifact": str(NLP_ARTIFACT_PATH),
            "tabular_model": str(TABULAR_MODEL_PATH),
            "tabular_scaler": str(TABULAR_SCALER_PATH),
            "tabular_imputer": str(TABULAR_IMPUTER_PATH),
            "tabular_features": str(TABULAR_FEATURES_PATH),
        },
    }


@app.get("/features")
def get_tabular_features():
    if TABULAR_FEATURE_COLS is None:
        raise HTTPException(
            status_code=503,
            detail="Daftar fitur tabular belum berhasil diload.",
        )

    return {
        "total_features": len(TABULAR_FEATURE_COLS),
        "feature_list": list(TABULAR_FEATURE_COLS),
        "example_payload": {
            "features": {
                col: 3
                for col in list(TABULAR_FEATURE_COLS)[:5]
            }
        },
    }


@app.post("/predict")
def predict_nlp(payload: NLPPredictRequest):
    """
    Endpoint lama untuk model NLP saja.
    """
    return predict_nlp_core(payload)


@app.post("/predict-tabular")
def predict_tabular(payload: TabularPredictRequest):
    """
    Endpoint untuk model tabular saja.
    """
    return predict_tabular_core(payload.features)


@app.post("/predicts")
def predict_both_models(payload: CombinedPredictRequest):
    """
    Endpoint gabungan.
    Satu request akan menjalankan:
    1. Model NLP
    2. Model Tabular
    3. Summary gabungan konservatif
    """
    nlp_payload = NLPPredictRequest(
        title=payload.title,
        description=payload.description,
        top_skills=payload.top_skills,
        top_core_tasks=payload.top_core_tasks,
        numeric_features=payload.numeric_features,
        generate_reskilling=payload.generate_reskilling,
        openrouter_model=payload.openrouter_model,
    )

    tabular_features = payload.features if payload.features is not None else payload.tabular_features

    nlp_result = predict_nlp_core(nlp_payload)
    tabular_result = predict_tabular_core(tabular_features)
    combined_prediction = build_combined_prediction(nlp_result, tabular_result)

    return {
        "status": "success",
        "endpoint": "/predicts",
        "nlp_result": nlp_result,
        "tabular_result": tabular_result,
        "combined_prediction": combined_prediction,
    }


# RUN DIRECTLY

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_nlp_tabular:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
