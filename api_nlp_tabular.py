"""
WorkSafe AI - Combined NLP + Tabular Prediction API

Struktur folder pake:

project/
├── api_nlp_tabular.py
└── models/
    ├── nlp_model/
    │   ├── worksafe_risk_model_best.keras           
    │   └── worksafe_artifacts.json
    └── tabular_model/
        ├── worksafe_model_v1.keras
        ├── scaler.pkl
        ├── imputer.pkl
        └── feature_cols.pkl

Install dependency ini yak:
pake venv
pip install fastapi uvicorn tensorflow numpy pandas scikit-learn openrouter

Jalankan:
uvicorn api_nlp_tabular_combined:app --reload --host 0.0.0.0 --port 8000

Dok:
http://127.0.0.1:8000/docs
"""

import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Custom Objects

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
class FocalLoss(tf.keras.losses.Loss):
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


CUSTOM_OBJECTS = {
    "AttentionPooling": AttentionPooling,
    "WorkSafeAI>AttentionPooling": AttentionPooling,
    "FocalLoss": FocalLoss,
    "WorkSafeAI>FocalLoss": FocalLoss,
}

# Path Model EyAi


BASE_DIR = Path(__file__).resolve().parent

NLP_DIR = BASE_DIR / "models" / "nlp_model"
NLP_MODEL_PATH = NLP_DIR / "worksafe_risk_model_best.keras"
NLP_FALLBACK_MODEL_PATH = NLP_DIR / "worksafe_risk_model.keras"
NLP_ARTIFACT_PATH = NLP_DIR / "worksafe_artifacts.json"

TABULAR_DIR = BASE_DIR / "models" / "tabular_model"
TABULAR_MODEL_PATH = TABULAR_DIR / "worksafe_model_v1.keras"
TABULAR_SCALER_PATH = TABULAR_DIR / "scaler.pkl"
TABULAR_IMPUTER_PATH = TABULAR_DIR / "imputer.pkl"
TABULAR_FEATURE_COLS_PATH = TABULAR_DIR / "feature_cols.pkl"

# FastAPI App

app = FastAPI(
    title="WorkSafe AI Combined Prediction API",
    description=(
        "API gabungan untuk prediksi risiko otomasi pekerjaan menggunakan "
        "model NLP dan model tabular, lalu membuat rekomendasi reskilling dengan OpenRouter."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk production, ubah ke domain frontend untuk cors.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Objects

NLP_MODEL = None
NLP_ARTIFACTS: Optional[Dict[str, Any]] = None

TABULAR_MODEL = None
TABULAR_SCALER = None
TABULAR_IMPUTER = None
TABULAR_FEATURE_COLS = None

LOAD_ERRORS: Dict[str, str] = {}

TABULAR_LABEL_MAP = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
}

# Schemas

class PredictRequest(BaseModel):
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

    # Dipertahankan agar kompatibel dengan API NLP lama.
    # Nilai ini dipakai untuk input numeric model NLP.
    # Jika tabular_features kosong, nilai ini juga dipakai sebagai input tabular.
    numeric_features: Optional[Dict[str, float]] = Field(
        default=None,
        example={},
        description=(
            "Opsional. Dipakai untuk numeric input model NLP. "
            "Jika tabular_features kosong, field ini juga dipakai untuk model tabular."
        ),
    )

    # Field baru khusus model tabular.
    tabular_features: Optional[Dict[str, float]] = Field(
        default=None,
        example={
            "act_repairing_and_maintaining_mechanical_equipment": 2,
            "act_operating_vehicles,_mechanized_devices,_or_equipment": 3,
            "act_handling_and_moving_objects": 4,
            "skl_troubleshooting": 3,
            "skl_critical_thinking": 4,
            "skl_complex_problem_solving": 4,
            "act_working_with_computers": 5,
            "skl_coordination": 4,
        },
        description=(
            "Opsional. Input fitur tabular sesuai feature_cols.pkl. "
            "Fitur yang tidak dikirim akan diisi oleh imputer/median training."
        ),
    )

    nlp_weight: float = Field(
        default=0.6,
        description="Bobot model NLP untuk final ensemble prediction.",
    )
    tabular_weight: float = Field(
        default=0.4,
        description="Bobot model tabular untuk final ensemble prediction.",
    )

    generate_reskilling: bool = Field(
        default=True,
        description="Jika true, API akan memanggil OpenRouter untuk membuat rekomendasi reskilling.",
    )
    openrouter_model: str = Field(
        default="deepseek/deepseek-v4-flash:free",
        description="Model OpenRouter. Bisa diganti sesuai model yang tersedia di akunmu.",
    )


class HealthResponse(BaseModel):
    status: str
    nlp_model_loaded: bool
    nlp_artifacts_loaded: bool
    tabular_model_loaded: bool
    tabular_artifacts_loaded: bool
    nlp_model_path: str
    nlp_artifact_path: str
    tabular_model_path: str
    tabular_scaler_path: str
    tabular_imputer_path: str
    tabular_feature_cols_path: str
    load_errors: Dict[str, str]

# Loaders

def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    with open(path, "rb") as f:
        obj = pickle.load(f)

    return obj


def repair_simple_imputer_compatibility(imputer):
    """
    Helper aman untuk mengurangi risiko error kompatibilitas pickle SimpleImputer
    antar versi scikit-learn. Jika atribut internal tidak ada, kita tambahkan
    secara defensif tanpa mengubah nilai statistics_ :).
    """
    if imputer is None:
        return imputer

    try:
        if not hasattr(imputer, "_fit_dtype"):
            imputer._fit_dtype = np.dtype("float64")
    except Exception:
        pass

    try:
        if not hasattr(imputer, "_fill_dtype"):
            imputer._fill_dtype = np.dtype("float64")
    except Exception:
        pass

    return imputer


def load_nlp_artifacts() -> Dict[str, Any]:
    return _load_json(NLP_ARTIFACT_PATH)


def load_nlp_model():
    selected_model_path = NLP_MODEL_PATH

    if not selected_model_path.exists() and NLP_FALLBACK_MODEL_PATH.exists():
        selected_model_path = NLP_FALLBACK_MODEL_PATH

    if not selected_model_path.exists():
        raise FileNotFoundError(
            f"File model NLP tidak ditemukan. Dicari di: {NLP_MODEL_PATH} atau {NLP_FALLBACK_MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        selected_model_path,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )


def load_tabular_resources() -> Tuple[Any, Any, Any, list]:
    if not TABULAR_MODEL_PATH.exists():
        raise FileNotFoundError(f"File model tabular tidak ditemukan: {TABULAR_MODEL_PATH}")

    model = tf.keras.models.load_model(
        TABULAR_MODEL_PATH,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )

    scaler = _load_pickle(TABULAR_SCALER_PATH)
    imputer = repair_simple_imputer_compatibility(_load_pickle(TABULAR_IMPUTER_PATH))
    feature_cols = _load_pickle(TABULAR_FEATURE_COLS_PATH)

    if not isinstance(feature_cols, (list, tuple)):
        raise TypeError("feature_cols.pkl harus berisi list/tuple nama fitur.")

    feature_cols = list(feature_cols)

    return model, scaler, imputer, feature_cols


@app.on_event("startup")
def startup_event():
    global NLP_MODEL, NLP_ARTIFACTS
    global TABULAR_MODEL, TABULAR_SCALER, TABULAR_IMPUTER, TABULAR_FEATURE_COLS
    global LOAD_ERRORS

    LOAD_ERRORS = {}

    print("Load NLP artifacts...")
    try:
        NLP_ARTIFACTS = load_nlp_artifacts()
        print("NLP artifacts loaded.")
    except Exception as e:
        LOAD_ERRORS["nlp_artifacts"] = f"{type(e).__name__}: {e}"
        print("Gagal load NLP artifacts:", LOAD_ERRORS["nlp_artifacts"])

    print("Load NLP model...")
    try:
        NLP_MODEL = load_nlp_model()
        print("NLP model loaded.")
        print("NLP input model:", [inp.name for inp in NLP_MODEL.inputs])
        print("NLP output model:", [out.name for out in NLP_MODEL.outputs])
    except Exception as e:
        LOAD_ERRORS["nlp_model"] = f"{type(e).__name__}: {e}"
        print("Gagal load NLP model:", LOAD_ERRORS["nlp_model"])

    print("Load tabular model & artifacts...")
    try:
        (
            TABULAR_MODEL,
            TABULAR_SCALER,
            TABULAR_IMPUTER,
            TABULAR_FEATURE_COLS,
        ) = load_tabular_resources()
        print("Tabular model & artifacts loaded.")
        print(f"Jumlah fitur tabular: {len(TABULAR_FEATURE_COLS)}")
    except Exception as e:
        LOAD_ERRORS["tabular"] = f"{type(e).__name__}: {e}"
        print("Gagal load tabular resources:", LOAD_ERRORS["tabular"])

# Helper NLP

def build_job_text(title: str, description: str = "", top_skills: str = "", top_core_tasks: str = "") -> str:
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


def unpack_nlp_prediction(raw_prediction):
    """
    Mendukung output model berbentuk list/tuple atau dict.
    Target:
    - pred_label: probabilitas klasifikasi
    - pred_score: skor risiko regresi
    """
    if isinstance(raw_prediction, dict):
        pred_label = None
        for key in ["risk_label", "risk_label_output", "classification"]:
            if key in raw_prediction:
                pred_label = raw_prediction[key]
                break

        pred_score = None
        for key in ["risk_score", "automation_risk_score", "score"]:
            if key in raw_prediction:
                pred_score = raw_prediction[key]
                break

        if pred_label is None:
            first_key = list(raw_prediction.keys())[0]
            pred_label = raw_prediction[first_key]

        if pred_score is None:
            keys = list(raw_prediction.keys())
            pred_score = raw_prediction[keys[1]] if len(keys) > 1 else None

        return pred_label, pred_score

    if isinstance(raw_prediction, (list, tuple)):
        if len(raw_prediction) == 1:
            return raw_prediction[0], None
        return raw_prediction[0], raw_prediction[1]

    return raw_prediction, None


def predict_nlp_core(payload: PredictRequest) -> Dict[str, Any]:
    if NLP_MODEL is None or NLP_ARTIFACTS is None:
        return {
            "status": "failed",
            "error": "Model NLP atau artifact NLP belum berhasil diload.",
            "load_errors": LOAD_ERRORS,
        }

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

    pred_label, pred_score = unpack_nlp_prediction(raw_prediction)

    label_idx = int(np.argmax(pred_label[0]))
    risk_label = label_classes[label_idx] if label_classes else TABULAR_LABEL_MAP.get(label_idx, str(label_idx))
    confidence = float(pred_label[0][label_idx])

    probabilities = {}
    for idx, prob in enumerate(pred_label[0]):
        label_name = label_classes[idx] if idx < len(label_classes) else TABULAR_LABEL_MAP.get(idx, str(idx))
        probabilities[str(label_name)] = round(float(prob) * 100, 2)

    if pred_score is not None:
        risk_score = float(np.ravel(pred_score)[0])
    else:
        risk_score = score_from_probabilities(probabilities)

    risk_score = clamp01(risk_score)

    return {
        "status": "success",
        "source": "nlp_model",
        "risk_label": str(risk_label),
        "risk_class": label_idx,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 2),
        "automation_risk_score": round(risk_score, 4),
        "risk_percent": round(risk_score * 100, 2),
        "probabilities": probabilities,
    }

# Helper Tabular

def preprocess_tabular_input(input_dict: Optional[Dict[str, float]]) -> np.ndarray:
    if TABULAR_FEATURE_COLS is None or TABULAR_IMPUTER is None or TABULAR_SCALER is None:
        raise RuntimeError("Artifact tabular belum berhasil diload.")

    input_dict = input_dict or {}
    df = pd.DataFrame([input_dict])

    for col in TABULAR_FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[TABULAR_FEATURE_COLS]

    # Pastikan semua nilai numerik. Nilai yang gagal dikonversi menjadi NaN,
    # lalu akan ditangani oleh imputer.
    for col in TABULAR_FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_imp = pd.DataFrame(
        TABULAR_IMPUTER.transform(df),
        columns=TABULAR_FEATURE_COLS,
    )

    df_scaled = pd.DataFrame(
        TABULAR_SCALER.transform(df_imp),
        columns=TABULAR_FEATURE_COLS,
    )

    return df_scaled.values.astype(np.float32)


def predict_tabular_core(payload: PredictRequest) -> Dict[str, Any]:
    if TABULAR_MODEL is None:
        return {
            "status": "failed",
            "error": "Model tabular belum berhasil diload.",
            "load_errors": LOAD_ERRORS,
        }

    # Jika tabular_features tidak dikirim, gunakan numeric_features sebagai fallback.
    input_features = payload.tabular_features
    if input_features is None:
        input_features = payload.numeric_features

    try:
        X = preprocess_tabular_input(input_features)
        proba = TABULAR_MODEL.predict(X, verbose=0)[0]
    except Exception as e:
        return {
            "status": "failed",
            "source": "tabular_model",
            "error_type": type(e).__name__,
            "error_message": str(e),
        }

    risk_class = int(np.argmax(proba))
    risk_label = TABULAR_LABEL_MAP.get(risk_class, str(risk_class))
    confidence = float(proba[risk_class])

    probabilities = {
        TABULAR_LABEL_MAP.get(i, str(i)): round(float(proba[i]) * 100, 2)
        for i in range(len(proba))
    }

    # Model tabular hanya klasifikasi. Agar bisa di-ensemble dengan NLP,
    # score dihitung dari probabilitas kelas:
    # Low=0.0, Medium=0.5, High=1.0.
    tabular_score = 0.0
    if len(proba) > 0:
        tabular_score += float(proba[0]) * 0.0
    if len(proba) > 1:
        tabular_score += float(proba[1]) * 0.5
    if len(proba) > 2:
        tabular_score += float(proba[2]) * 1.0

    tabular_score = clamp01(tabular_score)

    return {
        "status": "success",
        "source": "tabular_model",
        "risk_label": risk_label,
        "risk_class": risk_class,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 2),
        "automation_risk_score": round(tabular_score, 4),
        "risk_percent": round(tabular_score * 100, 2),
        "probabilities": probabilities,
        "features_received": sorted(list((input_features or {}).keys())),
        "missing_features_filled_by_imputer": [
            col for col in (TABULAR_FEATURE_COLS or []) if col not in (input_features or {})
        ],
    }

# Helper Ensemble

def clamp01(value: float) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0

    return max(0.0, min(1.0, value))


def normalize_risk_label(label: Any) -> str:
    label = str(label).lower()

    if "high" in label or "tinggi" in label:
        return "High Risk"

    if "medium" in label or "sedang" in label:
        return "Medium Risk"

    if "low" in label or "rendah" in label or "aman" in label:
        return "Low Risk"

    return str(label)


def label_from_score(score: float) -> str:
    score = clamp01(score)

    if score < 0.34:
        return "Low Risk"

    if score < 0.67:
        return "Medium Risk"

    return "High Risk"


def score_from_probabilities(probabilities: Dict[str, float]) -> float:
    """
    probabilities berisi persen, contoh:
    {
      "Low Risk": 20.0,
      "Medium Risk": 30.0,
      "High Risk": 50.0
    }
    """
    score = 0.0

    for label, prob_percent in probabilities.items():
        normalized = normalize_risk_label(label)
        prob = float(prob_percent) / 100.0

        if normalized == "Low Risk":
            score += prob * 0.0
        elif normalized == "Medium Risk":
            score += prob * 0.5
        elif normalized == "High Risk":
            score += prob * 1.0

    return clamp01(score)


def ensemble_predictions(
    nlp_prediction: Dict[str, Any],
    tabular_prediction: Dict[str, Any],
    nlp_weight: float = 0.6,
    tabular_weight: float = 0.4,
) -> Dict[str, Any]:
    nlp_ok = nlp_prediction.get("status") == "success"
    tabular_ok = tabular_prediction.get("status") == "success"

    if not nlp_ok and not tabular_ok:
        return {
            "status": "failed",
            "error": "Prediksi NLP dan tabular sama-sama gagal.",
        }

    available_scores = []
    total_weight = 0.0

    if nlp_ok:
        w = max(0.0, float(nlp_weight))
        available_scores.append((float(nlp_prediction["automation_risk_score"]), w, "nlp_model"))
        total_weight += w

    if tabular_ok:
        w = max(0.0, float(tabular_weight))
        available_scores.append((float(tabular_prediction["automation_risk_score"]), w, "tabular_model"))
        total_weight += w

    if total_weight <= 0:
        # Fallback jika user mengirim bobot 0 semua.
        total_weight = float(len(available_scores))
        available_scores = [(score, 1.0, source) for score, _, source in available_scores]

    final_score = sum(score * weight for score, weight, _ in available_scores) / total_weight
    final_score = clamp01(final_score)
    final_label = label_from_score(final_score)

    nlp_label = normalize_risk_label(nlp_prediction.get("risk_label")) if nlp_ok else None
    tabular_label = normalize_risk_label(tabular_prediction.get("risk_label")) if tabular_ok else None

    return {
        "status": "success",
        "source": "ensemble_nlp_tabular",
        "risk_label": final_label,
        "automation_risk_score": round(final_score, 4),
        "risk_percent": round(final_score * 100, 2),
        "weights": {
            "nlp_weight": nlp_weight if nlp_ok else 0.0,
            "tabular_weight": tabular_weight if tabular_ok else 0.0,
            "effective_total_weight": round(total_weight, 4),
        },
        "models_used": [source for _, _, source in available_scores],
        "model_agreement": bool(nlp_label == tabular_label) if (nlp_ok and tabular_ok) else None,
        "model_labels": {
            "nlp_model": nlp_label,
            "tabular_model": tabular_label,
        },
        "note": (
            "Final prediction dihitung dari weighted average automation_risk_score NLP dan tabular. "
            "Jika salah satu model gagal load/predict, final prediction memakai model yang berhasil."
        ),
    }

# Helper JSON AI

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

# OpenRouter Reskilling Generator

def generate_reskilling_with_openrouter(
    title: str,
    description: str,
    top_skills: str,
    top_core_tasks: str,
    final_prediction: Dict[str, Any],
    nlp_prediction: Dict[str, Any],
    tabular_prediction: Dict[str, Any],
    model_name: str = "deepseek/deepseek-v4-flash:free",
) -> Dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY")

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

    input_payload = {
        "job_input": {
            "title": title,
            "description": description,
            "top_skills": top_skills,
            "top_core_tasks": top_core_tasks,
        },
        "final_prediction": final_prediction,
        "model_predictions": {
            "nlp_prediction": nlp_prediction,
            "tabular_prediction": tabular_prediction,
        },
    }

    system_prompt = """
Kamu adalah AI career coach untuk aplikasi WorkSafe AI.
Tugasmu membuat rekomendasi reskilling pekerjaan berdasarkan input pekerjaan user dan hasil prediksi gabungan model NLP + tabular.

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
  "risk_interpretation": "penjelasan singkat tentang arti level risiko pekerjaan ini berdasarkan final_prediction",
  "result_model_percentage": "mengembalikan nilai presentase final gabungan kedua model",
  "result_model_risk_label": "mengembalikan label risiko combine kedua model",
  "main_reskilling_goal": "tujuan utama reskilling untuk user",
  "recommended_skills": [
    {{
      "skill": "nama skill",
      "reason": "alasan skill ini penting untuk pekerjaan user",
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
- Sesuaikan rekomendasi dengan final_prediction.risk_label dan final_prediction.automation_risk_score.
- Pertimbangkan juga input pekerjaan, top_skills, dan top_core_tasks.
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
                "karena request OpenRouter gagal. Coba ulang beberapa menit lagi, "
                "gunakan model lain, atau pastikan akun OpenRouter memiliki limit/credit."
            ),
        }

# Main Prediction

def predict_combined_core(payload: PredictRequest) -> Dict[str, Any]:
    nlp_prediction = predict_nlp_core(payload)
    tabular_prediction = predict_tabular_core(payload)

    final_prediction = ensemble_predictions(
        nlp_prediction=nlp_prediction,
        tabular_prediction=tabular_prediction,
        nlp_weight=payload.nlp_weight,
        tabular_weight=payload.tabular_weight,
    )

    result = {
        "status": "success" if final_prediction.get("status") == "success" else "failed",
        "input": {
            "title": payload.title,
            "description": payload.description,
            "top_skills": payload.top_skills,
            "top_core_tasks": payload.top_core_tasks,
            "numeric_features_received": sorted(list((payload.numeric_features or {}).keys())),
            "tabular_features_received": sorted(list((payload.tabular_features or {}).keys())),
        },
        "prediction": {
            "final": final_prediction,
            "nlp_model": nlp_prediction,
            "tabular_model": tabular_prediction,
        },
    }

    if payload.generate_reskilling and final_prediction.get("status") == "success":
        result["reskilling_recommendation"] = generate_reskilling_with_openrouter(
            title=payload.title,
            description=payload.description,
            top_skills=payload.top_skills,
            top_core_tasks=payload.top_core_tasks,
            final_prediction=final_prediction,
            nlp_prediction=nlp_prediction,
            tabular_prediction=tabular_prediction,
            model_name=payload.openrouter_model,
        )

    return result

# Endpoints

@app.get("/")
def root():
    return {
        "message": "WorkSafe AI Combined NLP + Tabular Risk Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok" if not LOAD_ERRORS else "warning",
        "nlp_model_loaded": NLP_MODEL is not None,
        "nlp_artifacts_loaded": NLP_ARTIFACTS is not None,
        "tabular_model_loaded": TABULAR_MODEL is not None,
        "tabular_artifacts_loaded": all(
            item is not None
            for item in [TABULAR_SCALER, TABULAR_IMPUTER, TABULAR_FEATURE_COLS]
        ),
        "nlp_model_path": str(NLP_MODEL_PATH if NLP_MODEL_PATH.exists() else NLP_FALLBACK_MODEL_PATH),
        "nlp_artifact_path": str(NLP_ARTIFACT_PATH),
        "tabular_model_path": str(TABULAR_MODEL_PATH),
        "tabular_scaler_path": str(TABULAR_SCALER_PATH),
        "tabular_imputer_path": str(TABULAR_IMPUTER_PATH),
        "tabular_feature_cols_path": str(TABULAR_FEATURE_COLS_PATH),
        "load_errors": LOAD_ERRORS,
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    return predict_combined_core(payload)

# Bisa pake langsung python nama_file.py
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_nlp_tabular:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
