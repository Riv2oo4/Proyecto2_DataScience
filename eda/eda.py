import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 6)

# ------------------------------------------------
# 0. Rutas y carga de datos
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
    print("⚠️ test solo tiene 1 columna, separando por comas...")
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

# 4.1 train + user_info
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

# Helper para guardar y cerrar
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
