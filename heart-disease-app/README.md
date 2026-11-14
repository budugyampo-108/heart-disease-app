# CardioScope: heart disease risk screening

Machine-learning web app for Assignment 2 (Hospital Management & Heart Disease Classification).
A scikit-learn model is trained on 500 hospital records, served by FastAPI, and explored through a responsive dashboard.

```
heart-disease-app/
├── data/hospital_synthetic_dataset.xlsx   your dataset (used by train.py)
├── train.py                               train, evaluate, save model.pkl + metrics.json
├── model.pkl                              trained pipeline (created by train.py)
├── metrics.json                           metrics, confusion matrix, ROC curve, form schema
├── backend/main.py                        FastAPI app (/api/health, /api/metrics, /api/predict)
├── web/index.html                         dashboard (HTML + CSS + JavaScript, no build step)
├── requirements.txt                       runtime dependencies (used by Vercel)
├── requirements-dev.txt                   + openpyxl, uvicorn, httpx for local work
└── vercel.json                            routes /api/* to FastAPI, everything else to web/
```

## 1. Run it locally

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python train.py                      # creates model.pkl and metrics.json
uvicorn backend.main:app --reload    # dashboard + API on one server
```

Open http://localhost:8000 for the dashboard and http://localhost:8000/api/docs for interactive API docs.

## 2. API

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | `{"status": "ok", "model_loaded": true, ...}` |
| GET | `/api/metrics` | accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve, feature importance |
| POST | `/api/predict` | prediction, class probabilities, risk level |

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age":58,"sex":"Male","blood_pressure":145,"cholesterol":240,"bmi":31.5,
       "smoker":"Yes","diabetes":"No","physical_activity":1,"family_history":"Yes","stress_level":7}'
```

## 3. Deploy to Vercel

1. Push the project to GitHub. **Commit `model.pkl` and `metrics.json`**: the deployed API loads them.
2. In Vercel choose **Add New, Project**, import the repository and deploy. No settings are needed because `vercel.json` defines the build.
3. Open the deployment URL. The dashboard is served from `/`, the API from `/api/*`.

If you retrain with a different scikit-learn version, update the pinned `scikit-learn==` line in `requirements.txt` to the `sklearn_version` shown in `metrics.json`.

## How the model works

- **Encoding:** `sex`, `smoker`, `diabetes` and `family_history` are one-hot encoded inside a single scikit-learn `Pipeline`, so the API passes raw values straight to the model.
- **Model choice:** Random Forest and Gradient Boosting are tuned with 5-fold cross-validated grid search on an 80% training split. The one with the higher ROC-AUC is kept.
- **Evaluation:** metrics are computed on the 20% test split that was never used for training or tuning.
- **Disclaimer:** educational project, not a medical device.
