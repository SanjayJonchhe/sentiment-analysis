from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import pickle

try:
    import joblib
except Exception:
    joblib = None
try:
    import numpy as np
except Exception:
    np = None

app = FastAPI(title="Sentiment Analysis Dashboard")
templates = Jinja2Templates(directory="templates")


def _load_any(filenames):
    """Try multiple filenames and load with joblib if available, fallback to pickle."""
    for p in filenames:
        if os.path.exists(p):
            try:
                if joblib:
                    return joblib.load(p)
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception:
                try:
                    with open(p, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    continue
    return None


# Accepted filenames (put your joblib/pickle files in project root):
# User-provided bundle filenames (combined model+vectorizer)
USER_BUNDLES = [
    "SVM",
    "sentiment_Logistic_regression",
    "sentiment_Naive_bayes",
]

# Build candidate filenames for each bundle (joblib/pkl/sav)
COMBINED_FILENAMES = []
for base in USER_BUNDLES:
    for ext in (".joblib", ".pkl", ".sav"):
        COMBINED_FILENAMES.append(f"{base}{ext}")

MODEL_FILENAMES = ["model.joblib", "model.pkl", "model.sav"]
VECTORIZER_FILENAMES = ["vectorizer.joblib", "vectorizer.pkl", "vectorizer.sav"]


def _extract_bundle(obj):
    """Return (model, vectorizer) if present in obj, else (None, None)."""
    if obj is None:
        return None, None
    if isinstance(obj, dict):
        m = obj.get("model") or obj.get("clf")
        v = obj.get("vectorizer") or obj.get("vect")
        return m, v
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return obj[0], obj[1]
    # try attributes
    m = getattr(obj, "model", None) or getattr(obj, "clf", None)
    v = getattr(obj, "vectorizer", None) or getattr(obj, "vect", None)
    return m, v


# Load all bundles found in the project root and present them by name
bundles = {}
for path in COMBINED_FILENAMES:
    if os.path.exists(path):
            obj = _load_any([path])
            m, v = _extract_bundle(obj)
            if m is not None and v is not None:
                # derive nice name from filename (strip extension)
                name = os.path.splitext(os.path.basename(path))[0]
                bundles[name] = {"model": m, "vectorizer": v}

# Fallback to separate model/vectorizer files if no bundles found
if not bundles:
    model = _load_any(MODEL_FILENAMES)
    vectorizer = _load_any(VECTORIZER_FILENAMES)
    if model is not None and vectorizer is not None:
        bundles["default"] = {"model": model, "vectorizer": vectorizer}

# Default selected model name
default_model_name = next(iter(bundles), None)


def interpret_prediction(pred, model_obj=None):
    """Normalize model prediction to one of: Positive, Negative, Neutral, or a readable fallback."""
    # unwrap arrays or lists
    try:
        if hasattr(pred, "__len__") and not isinstance(pred, (str, bytes)):
            if len(pred) > 0:
                pred = pred[0]
    except Exception:
        pass

    if isinstance(pred, bytes):
        try:
            pred = pred.decode()
        except Exception:
            pred = str(pred)

    if np is not None and isinstance(pred, (np.ndarray,)):
        try:
            pred = pred.item()
        except Exception:
            pass

    # string labels
    if isinstance(pred, str):
        lower = pred.lower()
        if "pos" in lower:
            return "Positive"
        if "neg" in lower:
            return "Negative"
        if "neu" in lower or "neutral" in lower:
            return "Neutral"
        return pred.capitalize()

    # numeric labels - common conventions
    try:
        i = int(pred)
        if i == 1:
            return "Positive"
        if i == 0:
            return "Negative"
        if i == 2:
            return "Neutral"
    except Exception:
        pass

    # try to infer from model classes if available
    try:
        if model_obj is not None and hasattr(model_obj, "classes_"):
            for c in list(model_obj.classes_):
                lc = str(c).lower()
                if "pos" in lc:
                    return "Positive"
                if "neg" in lc:
                    return "Negative"
                if "neu" in lc or "neutral" in lc:
                    return "Neutral"
    except Exception:
        pass

    return str(pred)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "models": list(bundles.keys()),
            "selected": default_model_name,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...), model_name: str = Form(None)):
    sel = model_name or default_model_name
    if sel not in bundles:
        sentiment = f"Model '{sel}' not found. Available: {', '.join(bundles.keys())}"
        return templates.TemplateResponse(
            "dashboard.html",
            {"request": request, "text": text, "sentiment": sentiment, "models": list(bundles.keys()), "selected": default_model_name},
        )

    model = bundles[sel]["model"]
    vectorizer = bundles[sel]["vectorizer"]

    X = None
    if vectorizer is not None:
        try:
            X = vectorizer.transform([text])
        except Exception:
            try:
                X = vectorizer([text])
            except Exception:
                sentiment = "Vectorizer failed to transform input"
                return templates.TemplateResponse(
                    "dashboard.html",
                    {"request": request, "text": text, "sentiment": sentiment, "models": list(bundles.keys()), "selected": sel},
                )
    else:
        # No vectorizer available — try to let the model accept raw input
        # Many scikit-learn models require a vectorizer; but if this model was trained
        # to accept raw text or is a pipeline that already includes preprocessing,
        # calling predict on a single-item list may work.
        try:
            X = [text]
        except Exception:
            sentiment = "Model requires a vectorizer but none found"
            return templates.TemplateResponse(
                "dashboard.html",
                {"request": request, "text": text, "sentiment": sentiment, "models": list(bundles.keys()), "selected": sel},
            )

    try:
        pred = model.predict(X)
        sentiment = interpret_prediction(pred, model_obj=model)
    except Exception:
        sentiment = "Model prediction failed"

    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "text": text, "sentiment": sentiment, "models": list(bundles.keys()), "selected": sel},
    )