import os
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS


# ==============================
# 1. CARGA DE DATOS Y FEATURES
# ==============================

BASE_PATH = './eda/data'

train_path     = os.path.join(BASE_PATH, 'train_format1.csv')
user_info_path = os.path.join(BASE_PATH, 'user_info_format1.csv')
user_log_path  = os.path.join(BASE_PATH, 'user_log_format1.csv')
test_path      = os.path.join(BASE_PATH, 'test_format1.csv')

print("Leyendo data...")
train = pd.read_csv(train_path)
user_info = pd.read_csv(user_info_path)
user_log = pd.read_csv(user_log_path)

if user_log.shape[1] == 1:
    print("user_log con 1 columna -> separando por comas...")
    col0 = user_log.columns[0]
    user_log = user_log[col0].astype(str).str.strip().str.split(',', expand=True)
    user_log.columns = ['user_id', 'item_id', 'cat_id', 'seller_id',
                        'brand_id', 'time_stamp', 'action_type']
else:
    if set(['user_id', 'item_id', 'cat_id', 'seller_id',
            'brand_id', 'time_stamp', 'action_type']).issubset(user_log.columns):
        user_log = user_log[['user_id', 'item_id', 'cat_id', 'seller_id',
                             'brand_id', 'time_stamp', 'action_type']]
    else:
        user_log.columns = ['user_id', 'item_id', 'cat_id', 'seller_id',
                            'brand_id', 'time_stamp', 'action_type']

for col in ['user_id', 'item_id', 'cat_id', 'seller_id',
            'brand_id', 'time_stamp', 'action_type']:
    user_log[col] = pd.to_numeric(user_log[col], errors='coerce')

print("Data cargada.")

# -------------------------------
# Construir features (mismo que tu script)
# -------------------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score, brier_score_loss


# Filtrar logs a usuarios de train
train_user_ids = set(train['user_id'].unique())
user_log_train = user_log[user_log['user_id'].isin(train_user_ids)].copy()

print("user_log original:", user_log.shape)
print("user_log filtrado:", user_log_train.shape)

logs_counts = (
    user_log_train
      .groupby(['user_id', 'seller_id', 'action_type'])
      .size()
      .unstack(fill_value=0)
      .reset_index()
)

logs_counts.rename(columns={
    'seller_id': 'merchant_id',
    0: 'n_clicks',
    1: 'n_cart',
    2: 'n_purchase',
    3: 'n_fav'
}, inplace=True)

# Estadísticas de tiempo
logs_time = (
    user_log_train
      .groupby(['user_id', 'seller_id'])['time_stamp']
      .agg(['min', 'max', 'mean'])
      .reset_index()
      .rename(columns={
          'seller_id': 'merchant_id',
          'min': 't_min',
          'max': 't_max',
          'mean': 't_mean'
      })
)

logs_pair = logs_counts.merge(
    logs_time,
    on=['user_id', 'merchant_id'],
    how='left'
)

logs_pair['total_actions'] = (
    logs_pair['n_clicks'] +
    logs_pair['n_cart'] +
    logs_pair['n_purchase'] +
    logs_pair['n_fav']
)

logs_pair['purchase_rate'] = np.where(
    logs_pair['total_actions'] > 0,
    logs_pair['n_purchase'] / logs_pair['total_actions'],
    0.0
)

print("logs_pair shape:", logs_pair.shape)

# Tabla final de entrenamiento
data = (
    train
    .merge(user_info, on='user_id', how='left')
    .merge(logs_pair, on=['user_id', 'merchant_id'], how='left')
)

behav_cols = [
    'n_clicks', 'n_cart', 'n_purchase', 'n_fav',
    'total_actions', 'purchase_rate',
    't_min', 't_max', 't_mean'
]

for c in behav_cols:
    data[c] = data[c].fillna(0)

data['age_range'] = data['age_range'].fillna(-1)
data['gender']    = data['gender'].fillna(-1)

feature_cols = ['age_range', 'gender'] + behav_cols
X = data[feature_cols]
y = data['label']

print("X shape:", X.shape)
print("Positivos:", y.sum(), "de", len(y), "(", y.mean(), ")")

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape, "Valid size:", X_valid.shape)

# =====================================
# 2. ENTRENAR MODELOS (como en tu script)
# =====================================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import lightgbm as lgb
from catboost import CatBoostClassifier

def get_metrics_dict(name, y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "name": name,
        "auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_proba))
    }


def eval_model(name, y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    auc  = roc_auc_score(y_true, y_pred_proba)
    ll   = log_loss(y_true, y_pred_proba)
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print("AUC       :", round(auc, 4))
    print("LogLoss   :", round(ll, 4))
    print("Accuracy  :", round(acc, 4))
    print("F1 (thr=0.5):", round(f1, 4))

print("\nEntrenando Regresión Logística...")
logreg_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        n_jobs=-1
    ))
])

logreg_param_grid = { 'logreg__C': [0.1, 1.0, 10.0] }

logreg_grid = GridSearchCV(
    estimator=logreg_pipe,
    param_grid=logreg_param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

logreg_grid.fit(X_train, y_train)
best_logreg = logreg_grid.best_estimator_
y_valid_proba_logreg = best_logreg.predict_proba(X_valid)[:, 1]
eval_model("Regresión Logística", y_valid, y_valid_proba_logreg)

print("\nEntrenando LightGBM...")
lgb_base = lgb.LGBMClassifier(
    objective='binary',
    class_weight='balanced',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lgb_param_dist = {
    'num_leaves': [31, 63, 127],
    'max_depth': [-1, 7, 10],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9]
}

lgb_rand = RandomizedSearchCV(
    estimator=lgb_base,
    param_distributions=lgb_param_dist,
    n_iter=10,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0,
    random_state=42
)

lgb_rand.fit(X_train, y_train)
best_lgb = lgb_rand.best_estimator_
y_valid_proba_lgb = best_lgb.predict_proba(X_valid)[:, 1]
eval_model("LightGBM", y_valid, y_valid_proba_lgb)

print("\nEntrenando CatBoost...")
pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

cb_model = CatBoostClassifier(
    loss_function='Logloss',
    eval_metric='AUC',
    depth=8,
    learning_rate=0.05,
    l2_leaf_reg=6,
    iterations=1000,
    random_seed=42,
    verbose=100,
    scale_pos_weight=float(pos_weight)
)

cb_model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    use_best_model=True,
    early_stopping_rounds=50
)

y_valid_proba_cb = cb_model.predict_proba(X_valid)[:, 1]
eval_model("CatBoost", y_valid, y_valid_proba_cb)

print("\nModelos entrenados. Listo para servir predicciones.")


# ==============================
#  MÉTRICAS Y FEATURE IMPORTANCE PARA API
# ==============================
from sklearn.metrics import brier_score_loss

FEATURES_FOR_API = ['n_clicks', 'n_cart', 'n_purchase', 'n_fav']
FEATURE_NAME_MAP = {
    'n_clicks': 'clicks',
    'n_cart': 'carts',
    'n_purchase': 'purchases',
    'n_fav': 'favorites',
}

def get_metrics_dict(name, y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "name": name,
        "auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_proba))
    }

MODEL_METRICS = [
    get_metrics_dict("Logistic Regression", y_valid, y_valid_proba_logreg),
    get_metrics_dict("LightGBM",           y_valid, y_valid_proba_lgb),
    get_metrics_dict("CatBoost",           y_valid, y_valid_proba_cb),
]
print("\n=== MÉTRICAS PARA API ===")
for m in MODEL_METRICS:
    print(m)

def get_feature_importance_logreg(pipe):
    """Importancia relativa (normalizada) para FEATURES_FOR_API usando |coef|."""
    lr = pipe.named_steps['logreg']
    coefs = np.abs(lr.coef_[0])
    imp = {}
    for f in FEATURES_FOR_API:
        idx = feature_cols.index(f)
        imp[f] = coefs[idx]
    s = sum(imp.values()) or 1.0
    for k in imp:
        imp[k] = float(imp[k] / s)
    return imp

def get_feature_importance_tree(model):
    """Importancia relativa (normalizada) para FEATURES_FOR_API en modelos tipo árbol."""
    importances = model.feature_importances_
    feat_to_imp = dict(zip(feature_cols, importances))
    imp = {f: float(feat_to_imp.get(f, 0.0)) for f in FEATURES_FOR_API}
    s = sum(imp.values()) or 1.0
    for k in imp:
        imp[k] = float(imp[k] / s)
    return imp

fi_logreg = get_feature_importance_logreg(best_logreg)
fi_lgb    = get_feature_importance_tree(best_lgb)
fi_cb     = get_feature_importance_tree(cb_model)

FEATURE_IMPORTANCE_API = []
for f in FEATURES_FOR_API:
    FEATURE_IMPORTANCE_API.append({
        "feature": FEATURE_NAME_MAP[f],
        "Logistic": fi_logreg[f],
        "LightGBM": fi_lgb[f],
        "CatBoost": fi_cb[f],
    })

print("\n=== FEATURE IMPORTANCE PARA API ===")
for row in FEATURE_IMPORTANCE_API:
    print(row)


# =====================================
# 3. API FLASK
# =====================================

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Espera un JSON:
    {
      "features": {
        "age_range": 5,
        "gender": 0,
        "clicks": 10,
        "carts": 1,
        "purchases": 0,
        "favorites": 0
      }
    }
    """
    data_json = request.get_json(force=True)
    feats = data_json.get('features', {})

    age_range = feats.get('age_range', -1)
    gender    = feats.get('gender', -1)
    clicks    = feats.get('clicks', 0)
    carts     = feats.get('carts', 0)
    purchases = feats.get('purchases', 0)
    favorites = feats.get('favorites', 0)

    total_actions = clicks + carts + purchases + favorites
    purchase_rate = purchases / total_actions if total_actions > 0 else 0.0

    row = {
        'age_range': age_range,
        'gender': gender,
        'n_clicks': clicks,
        'n_cart': carts,
        'n_purchase': purchases,
        'n_fav': favorites,
        'total_actions': total_actions,
        'purchase_rate': purchase_rate,
        't_min': 0,
        't_max': 0,
        't_mean': 0
    }

    df = pd.DataFrame([row])[feature_cols]

    proba_logreg = float(best_logreg.predict_proba(df)[0, 1])
    proba_lgb    = float(best_lgb.predict_proba(df)[0, 1])
    proba_cb     = float(cb_model.predict_proba(df)[0, 1])

    print("REQUEST FEATURES:", feats)
    print("PROB LOGREG:", proba_logreg,
          "PROB LGB:", proba_lgb,
          "PROB CB:", proba_cb)
    
    return jsonify({
        "modelPredictions": [
            {"model": "Logistic Regression", "prob": proba_logreg},
            {"model": "LightGBM",           "prob": proba_lgb},
            {"model": "CatBoost",           "prob": proba_cb}
        ]
    })

@app.route('/models/metrics', methods=['GET'])
def models_metrics():
    """
    Devuelve las métricas de cada modelo en validación para que el front
    pueda construir las barras de comparación.
    """
    # Si quieres también mandar el tipo y una descripción corta:
    meta = {
        "Logistic Regression": {
            "type": "Lineal",
            "description": "Modelo interpretable usado como baseline."
        },
        "LightGBM": {
            "type": "Gradient Boosting",
            "description": "Modelo de árboles eficiente para datos tabulares."
        },
        "CatBoost": {
            "type": "Gradient Boosting",
            "description": "Optimizado para variables categóricas."
        }
    }

    models_response = []
    for m in MODEL_METRICS:
        extra = meta.get(m["name"], {})
        models_response.append({
            "name": m["name"],
            "type": extra.get("type", ""),
            "description": extra.get("description", ""),
            "auc": m["auc"],
            "f1": m["f1"],
            "brier": m["brier"],
        })

    return jsonify({"models": models_response})

@app.route('/models/feature-importance', methods=['GET'])
def feature_importance():
    """
    Devuelve una lista con la importancia de características por modelo:
    [
      { "feature": "clicks", "Logistic": 0.30, "LightGBM": 0.25, "CatBoost": 0.28 },
      ...
    ]
    """
    return jsonify({"importance": FEATURE_IMPORTANCE_API})




@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "ok", "message": "API de modelos funcionando"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
