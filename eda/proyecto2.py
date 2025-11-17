import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 6)

# ------------------------------------------------
# Rutas y carga de datos
# ------------------------------------------------

BASE_PATH = './eda/data'
FIG_PATH  = './eda/figures'

os.makedirs(FIG_PATH, exist_ok=True)

train_path     = os.path.join(BASE_PATH, 'train_format1.csv')
user_info_path = os.path.join(BASE_PATH, 'user_info_format1.csv')
user_log_path  = os.path.join(BASE_PATH, 'user_log_format1.csv')
test_path      = os.path.join(BASE_PATH, 'test_format1.csv')

print("Archivos que se van a leer:")
for p in [train_path, user_info_path, user_log_path, test_path]:
    print(" -", p, "->", "OK" if os.path.exists(p) else "NO ENCONTRADO")

train = pd.read_csv(train_path)
user_info = pd.read_csv(user_info_path)

user_log = pd.read_csv(user_log_path)

print("\nuser_log columnas crudas:", list(user_log.columns))
print("shape user_log:", user_log.shape)

if user_log.shape[1] == 1:
    print("user_log solo tiene 1 columna, separando por comas...")
    col0 = user_log.columns[0]
    user_log = user_log[col0].astype(str).str.strip().str.split(',', expand=True)
    user_log.columns = ['user_id', 'item_id', 'cat_id', 'seller_id', 'brand_id', 'time_stamp', 'action_type']
else:
    if set(['user_id', 'item_id', 'cat_id', 'seller_id', 'brand_id', 'time_stamp', 'action_type']).issubset(user_log.columns):
        user_log = user_log[['user_id', 'item_id', 'cat_id', 'seller_id', 'brand_id', 'time_stamp', 'action_type']]
    else:
        user_log.columns = ['user_id', 'item_id', 'cat_id', 'seller_id', 'brand_id', 'time_stamp', 'action_type']

for col in ['user_id', 'item_id', 'cat_id', 'seller_id', 'brand_id', 'time_stamp', 'action_type']:
    user_log[col] = pd.to_numeric(user_log[col], errors='coerce')

test = pd.read_csv(test_path)
print("\ntest columnas crudas:", list(test.columns))
print("shape test:", test.shape)

if test.shape[1] == 1:
    print("test solo tiene 1 columna, separando por comas...")
    col0 = test.columns[0]
    test = test[col0].astype(str).str.strip().str.split(',', expand=True)
    test.columns = ['user_id', 'merchant_id', 'prob']

test['user_id'] = pd.to_numeric(test['user_id'], errors='coerce')
test['merchant_id'] = pd.to_numeric(test['merchant_id'], errors='coerce')

print("\n=== Carga completada ===\n")

# ------------------------------------------------
# 1. Descripción básica
# ------------------------------------------------

print(">>> SHAPE Y TIPOS DE DATOS\n")

print("train shape:", train.shape)
print(train.dtypes, "\n")

print("user_info shape:", user_info.shape)
print(user_info.dtypes, "\n")

print("user_log shape:", user_log.shape)
print(user_log.dtypes, "\n")

print("test shape:", test.shape)
print(test.dtypes, "\n")

# ------------------------------------------------
# 2. Estadística descriptiva
# ------------------------------------------------

print("\n>>> DESCRIPTIVE STATS - TRAIN\n")
print(train.describe())

print("\n>>> DESCRIPTIVE STATS - USER_INFO\n")
print(user_info.describe(include='all'))

print("\n>>> DESCRIPTIVE STATS - USER_LOG\n")
print(user_log.describe())

print("\n>>> DESCRIPTIVE STATS - TEST\n")
print(test.describe())

# ------------------------------------------------
# 3. Frecuencias categóricas
# ------------------------------------------------

print("\n>>> FRECUENCIAS CATEGÓRICAS\n")

if 'label' in train.columns:
    print("\nFrecuencia de label (train['label']):")
    print(train['label'].value_counts())

if 'age_range' in user_info.columns:
    print("\nFrecuencia de age_range (user_info['age_range']):")
    print(user_info['age_range'].value_counts().sort_index())

if 'gender' in user_info.columns:
    print("\nFrecuencia de gender (user_info['gender']):")
    print(user_info['gender'].value_counts(dropna=False))

if 'action_type' in user_log.columns:
    print("\nFrecuencia de action_type (user_log['action_type']):")
    print(user_log['action_type'].value_counts().sort_index())

# ------------------------------------------------
# 4. Cruces entre variables clave
# ------------------------------------------------

print("\n>>> CRUCES ENTRE VARIABLES CLAVE\n")

# train + user_info
if 'user_id' in train.columns and 'user_id' in user_info.columns:
    train_ui = pd.merge(train, user_info, on='user_id', how='left')

    if 'gender' in train_ui.columns:
        print("\nTasa de conversión (label) por gender:")
        print(train_ui.groupby('gender')['label'].mean())

    if 'age_range' in train_ui.columns:
        print("\nTasa de conversión (label) por age_range:")
        print(train_ui.groupby('age_range')['label'].mean())

# Historial de acciones
if 'user_id' in user_log.columns and 'action_type' in user_log.columns:
    MAX_ROWS = 1_000_000
    if len(user_log) > MAX_ROWS:
        user_log_sample = user_log.sample(MAX_ROWS, random_state=42)
        print(f"\nuser_log grande ({len(user_log)} filas). Se usa muestra de {MAX_ROWS}.")
    else:
        user_log_sample = user_log

    user_actions = (
        user_log_sample
        .groupby('user_id')['action_type']
        .value_counts()
        .unstack(fill_value=0)
    )
    user_actions.columns = [f'action_{int(c)}' for c in user_actions.columns]

    train_beh = pd.merge(train, user_actions, on='user_id', how='left')
    for c in user_actions.columns:
        train_beh[c] = train_beh[c].fillna(0)

    if 'action_2' in train_beh.columns:
        print("\nNúmero de compras previas (action_2) por valor de label:")
        print(train_beh.groupby('label')['action_2'].describe())

# ------------------------------------------------
# 5. Gráficos exploratorios
# ------------------------------------------------

print("\n>>> GRAFICOS EXPLORATORIOS (SE GUARDAN EN 'figures/')\n")

def save_fig(name):
    path = os.path.join(FIG_PATH, name)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada: {path}")

# Distribución de la variable objetivo (label)
if 'label' in train.columns:
    plt.figure()
    train['label'].hist(bins=3)
    plt.xlabel('label')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de la variable objetivo (label)')
    save_fig('hist_label.png')

# Histograma de age_range
if 'age_range' in user_info.columns:
    plt.figure()
    user_info['age_range'].hist()
    plt.xlabel('age_range')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de rango de edad')
    save_fig('hist_age_range.png')

# Barras de gender
if 'gender' in user_info.columns:
    plt.figure()
    user_info['gender'].value_counts(dropna=False).plot(kind='bar')
    plt.xlabel('gender')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de género (incluye NaN)')
    save_fig('bar_gender.png')

# Barras de action_type
if 'action_type' in user_log.columns:
    plt.figure()
    user_log['action_type'].value_counts().sort_index().plot(kind='bar')
    plt.xlabel('action_type')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de tipos de acción en user_log')
    save_fig('bar_action_type.png')

# Histograma del número de logs por usuario
if 'user_id' in user_log.columns:
    logs_per_user = user_log.groupby('user_id').size()
    plt.figure()
    logs_per_user.hist()
    plt.xlabel('Número de logs por usuario')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de eventos por usuario en user_log')
    save_fig('hist_logs_per_user.png')

print("\n=== EDA TERMINADO ===")
# ================================================
# 6. MODELADO
# ================================================
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score

# Filtrar logs solo a usuarios que están en train (para reducir tamaño)
train_user_ids = set(train['user_id'].unique())
user_log_train = user_log[user_log['user_id'].isin(train_user_ids)].copy()

print("user_log original:", user_log.shape)
print("user_log filtrado (solo usuarios de train):", user_log_train.shape)

# Agregados de comportamiento por (user_id, seller_id)
logs_counts = (
    user_log_train
      .groupby(['user_id', 'seller_id', 'action_type'])
      .size()
      .unstack(fill_value=0)
      .reset_index()
)

# renombrar columnas para que sean legibles
logs_counts.rename(columns={
    'seller_id': 'merchant_id',
    0: 'n_clicks',
    1: 'n_cart',
    2: 'n_purchase',
    3: 'n_fav'
}, inplace=True)

# Estadísticas de tiempo por (user_id, seller_id)
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

# Unir ambos bloques de features de logs
logs_pair = logs_counts.merge(
    logs_time,
    on=['user_id', 'merchant_id'],
    how='left'
)

# Features derivados
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

# Construir la tabla final de entrenamiento:
data = (
    train
    .merge(user_info, on='user_id', how='left')
    .merge(logs_pair, on=['user_id', 'merchant_id'], how='left')
)

# Manejo de NaN en features numéricas de comportamiento
behav_cols = [
    'n_clicks', 'n_cart', 'n_purchase', 'n_fav',
    'total_actions', 'purchase_rate',
    't_min', 't_max', 't_mean'
]

for c in behav_cols:
    data[c] = data[c].fillna(0)

# Manejo de NaN en demográficas
data['age_range'] = data['age_range'].fillna(-1)
data['gender']    = data['gender'].fillna(-1)

feature_cols = ['age_range', 'gender'] + behav_cols
X = data[feature_cols]
y = data['label']

print("X shape:", X.shape)
print("y positivos:", y.sum(), "de", len(y), "(", y.mean(), ")")


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape, "Valid size:", X_valid.shape)

# ================================================
# 7. FUNCIÓN DE EVALUACIÓN
# ================================================
def eval_model(name, y_true, y_pred_proba, threshold=0.5):
    """
    Imprime métricas básicas de evaluación para un modelo binario.
    """
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

# ================================================
# 8. MODELO 1: REGRESIÓN LOGÍSTICA
# ================================================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

logreg_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',   
        n_jobs=-1
    ))
])

logreg_param_grid = {
    'logreg__C': [0.1, 1.0, 10.0]
}

logreg_grid = GridSearchCV(
    estimator=logreg_pipe,
    param_grid=logreg_param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

print("\nEntrenando Regresión Logística (GridSearch)...")
logreg_grid.fit(X_train, y_train)
best_logreg = logreg_grid.best_estimator_
print("Mejores parámetros LogReg:", logreg_grid.best_params_)

y_valid_proba_logreg = best_logreg.predict_proba(X_valid)[:, 1]
eval_model("Regresión Logística", y_valid, y_valid_proba_logreg)

# ================================================
# 9. MODELO 2: LIGHTGBM
# ================================================
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV

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
    verbose=2,
    random_state=42
)

print("\nEntrenando LightGBM (RandomizedSearch)...")
lgb_rand.fit(X_train, y_train)
best_lgb = lgb_rand.best_estimator_
print("Mejores parámetros LightGBM:", lgb_rand.best_params_)

y_valid_proba_lgb = best_lgb.predict_proba(X_valid)[:, 1]
eval_model("LightGBM", y_valid, y_valid_proba_lgb)

# ================================================
# 10. MODELO 3: CATBOOST
# ================================================
from catboost import CatBoostClassifier

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

print("\nEntrenando CatBoost...")
cb_model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    use_best_model=True,
    early_stopping_rounds=50
)

y_valid_proba_cb = cb_model.predict_proba(X_valid)[:, 1]
eval_model("CatBoost", y_valid, y_valid_proba_cb)



if 'prob' in test.columns:
    test_base = test[['user_id', 'merchant_id']].copy()
else:
    test_base = test.copy()

test_feat = (
    test_base
    .merge(user_info, on='user_id', how='left')
    .merge(logs_pair, on=['user_id', 'merchant_id'], how='left')
)

for c in behav_cols:
    test_feat[c] = test_feat[c].fillna(0)

test_feat['age_range'] = test_feat['age_range'].fillna(-1)
test_feat['gender']    = test_feat['gender'].fillna(-1)

X_test_final = test_feat[feature_cols]

best_model = cb_model

best_model.fit(X, y)  
test_pred = best_model.predict_proba(X_test_final)[:, 1]

submission = pd.DataFrame({
    'user_id': test_base['user_id'],
    'merchant_id': test_base['merchant_id'],
    'prob': test_pred
})

submission.to_csv('prediction_lightgbm.csv', index=False)
print("\nArchivo de submission guardado como 'prediction_lightgbm.csv'")

from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score, f1_score,
    roc_curve, precision_recall_curve
)
import pandas as pd
import matplotlib.pyplot as plt

def get_metrics(name, y_true, y_proba, thr=0.5):
    y_pred = (y_proba >= thr).astype(int)
    return {
        'Modelo': name,
        'AUC': roc_auc_score(y_true, y_proba),
        'LogLoss': log_loss(y_true, y_proba),
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }

results = []
results.append(get_metrics("Regresión Logística", y_valid, y_valid_proba_logreg))
results.append(get_metrics("LightGBM", y_valid, y_valid_proba_lgb))
results.append(get_metrics("CatBoost", y_valid, y_valid_proba_cb))

metrics_df = pd.DataFrame(results).set_index('Modelo')
print("\n=== MÉTRICAS EN VALIDACIÓN ===")
print(metrics_df)

import os
FIG_PATH  = './eda/figures'
os.makedirs(FIG_PATH, exist_ok=True)

def save_fig(name):
    path = os.path.join(FIG_PATH, name)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada: {path}")

# Gráfico de barras AUC y F1 
plt.figure()
metrics_df[['AUC', 'F1']].plot(kind='bar')
plt.title('Comparación de modelos: AUC y F1')
plt.ylabel('Valor de la métrica')
plt.xticks(rotation=0)
plt.legend(loc='best')
save_fig('model_metrics_bar.png')

# Curvas ROC
fpr_log, tpr_log, _ = roc_curve(y_valid, y_valid_proba_logreg)
fpr_lgb, tpr_lgb, _ = roc_curve(y_valid, y_valid_proba_lgb)
fpr_cb,  tpr_cb,  _ = roc_curve(y_valid, y_valid_proba_cb)

plt.figure()
plt.plot(fpr_log, tpr_log, label='LogReg')
plt.plot(fpr_lgb, tpr_lgb, label='LightGBM')
plt.plot(fpr_cb,  tpr_cb,  label='CatBoost')
plt.plot([0, 1], [0, 1], linestyle='--', label='Azar')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curvas ROC en conjunto de validación')
plt.legend(loc='lower right')
save_fig('roc_curves.png')

plt.figure()
plt.hist(
    y_valid_proba_cb[y_valid == 0],
    bins=50, alpha=0.5, label='No recurrente (label=0)'
)
plt.hist(
    y_valid_proba_cb[y_valid == 1],
    bins=50, alpha=0.5, label='Recurrente (label=1)'
)
plt.xlabel('Probabilidad predicha')
plt.ylabel('Frecuencia')
plt.title('Distribución de probabilidades – CatBoost')
plt.legend(loc='best')
save_fig('proba_hist_catboost.png')
