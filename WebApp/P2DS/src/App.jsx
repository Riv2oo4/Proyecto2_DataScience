import React, { useState } from 'react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid
} from 'recharts'


// Cambia esto si ya tienes un backend con endpoint /predict
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

const INITIAL_MODELS = [
  {
    name: 'Logistic Regression',
    type: 'Lineal',
    auc: 0.78,
    f1: 0.42,
    brier: 0.19,
    description: 'Modelo interpretable, útil como baseline para comparar.'
  },
  {
    name: 'LightGBM',
    type: 'Gradient Boosting',
    auc: 0.86,
    f1: 0.51,
    brier: 0.16,
    description: 'Modelo de árboles rápido con buen rendimiento en datos tabulares.'
  },
  {
    name: 'CatBoost',
    type: 'Gradient Boosting',
    auc: 0.88,
    f1: 0.53,
    brier: 0.15,
    description: 'Optimizado para variables categóricas, suele mejorar el baseline.'
  }
]
// Importancia de características (puedes ajustar con tus resultados reales)
const FEATURE_IMPORTANCE = [
  {
    feature: 'clicks',
    Logistic: 0.30,
    LightGBM: 0.25,
    CatBoost: 0.28
  },
  {
    feature: 'carts',
    Logistic: 0.20,
    LightGBM: 0.22,
    CatBoost: 0.24
  },
  {
    feature: 'purchases',
    Logistic: 0.35,
    LightGBM: 0.40,
    CatBoost: 0.38
  },
  {
    feature: 'favorites',
    Logistic: 0.15,
    LightGBM: 0.13,
    CatBoost: 0.10
  }
]


// Componente para seleccionar la métrica que se va a comparar visualmente
function MetricSelector({ metric, onChange }) {
  return (
    <div className="metric-selector">
      <label htmlFor="metric">Métrica a comparar:</label>
      <select
        id="metric"
        value={metric}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="auc">ROC AUC</option>
        <option value="f1">F1-score</option>
        <option value="brier">Brier score</option>
      </select>
    </div>
  )
}

// Componente para mostrar tabla de métricas
function ModelsTable({ models }) {
  return (
    <div className="card">
      <h2>Resumen de modelos</h2>
      <p className="card-subtitle">
        Comparación numérica de las métricas principales obtenidas en validación.
      </p>
      <div className="table-wrapper">
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Modelo</th>
              <th>Tipo</th>
              <th>ROC AUC</th>
              <th>F1-score</th>
              <th>Brier score</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.name}>
                <td>{m.name}</td>
                <td>{m.type}</td>
                <td>{m.auc.toFixed(3)}</td>
                <td>{m.f1.toFixed(3)}</td>
                <td>{m.brier.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// Gráfico tipo “barra horizontal” hecho con CSS para comparar métricas
function MetricBars({ models, metric }) {
  const metricLabel =
    metric === 'auc' ? 'ROC AUC' :
    metric === 'f1' ? 'F1-score' :
    'Brier score'

  const values = models.map((m) => m[metric])
  const maxVal = metric === 'brier'
    ? Math.max(...values.map((v) => 1 - v)) || 1 // invertimos Brier para visualizar “mejor es más grande”
    : Math.max(...values) || 1

  return (
    <div className="card">
      <h2>Comparación visual de modelos</h2>
      <p className="card-subtitle">
        Barras normalizadas de la métrica seleccionada ({metricLabel}). 
        Mejor desempeño = barra más larga.
      </p>
      <div className="bars-container">
        {models.map((m) => {
          const rawValue = m[metric]
          const valueForBar = metric === 'brier' ? 1 - rawValue : rawValue
          const width = (valueForBar / maxVal) * 100
          return (
            <div key={m.name} className="bar-row">
              <div className="bar-label">
                <strong>{m.name}</strong>
                <span className="bar-metric">
                  {metricLabel}: {rawValue.toFixed(3)}
                </span>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${width}%` }} />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// Panel de detalle del modelo seleccionado
function ModelDetails({ model }) {
  if (!model) return null
  return (
    <div className="card">
      <h2>Detalle del modelo seleccionado</h2>
      <p className="model-name">{model.name}</p>
      <ul className="model-details-list">
        <li><strong>Tipo:</strong> {model.type}</li>
        <li><strong>ROC AUC:</strong> {model.auc.toFixed(3)}</li>
        <li><strong>F1-score:</strong> {model.f1.toFixed(3)}</li>
        <li><strong>Brier score:</strong> {model.brier.toFixed(3)}</li>
      </ul>
      <p className="model-description">{model.description}</p>
    </div>
  )
}

// Formulario para ingresar nuevos datos y obtener predicción
function PredictionForm({ onSubmit, loading }) {
  const [form, setForm] = useState({
    age_range: '5',
    gender: '0',
    clicks: '10',
    carts: '1',
    purchases: '0',
    favorites: '0'
  })

  const handleChange = (e) => {
    const { name, value } = e.target
    setForm((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    onSubmit({
      age_range: Number(form.age_range),
      gender: Number(form.gender),
      clicks: Number(form.clicks),
      carts: Number(form.carts),
      purchases: Number(form.purchases),
      favorites: Number(form.favorites)
    })
  }

  return (
    <div className="card">
      <h2>Predicción con nuevos datos</h2>
      <p className="card-subtitle">
        Ingresa características simplificadas de un usuario y el dashboard consultará
        a los modelos para estimar la probabilidad de compra.
      </p>

      <form className="form-grid" onSubmit={handleSubmit}>
        <div className="form-field">
          <label>Rango de edad</label>
          <select
            name="age_range"
            value={form.age_range}
            onChange={handleChange}
          >
            <option value="0">Desconocido</option>
            <option value="3">Joven</option>
            <option value="4">Adulto joven</option>
            <option value="5">Adulto</option>
            <option value="6">Adulto mayor</option>
            <option value="7">Senior</option>
          </select>
        </div>

        <div className="form-field">
          <label>Género</label>
          <select
            name="gender"
            value={form.gender}
            onChange={handleChange}
          >
            <option value="0">Femenino</option>
            <option value="1">Masculino</option>
            <option value="2">Otro / No especifica</option>
          </select>
        </div>

        <div className="form-field">
          <label>Número de clicks</label>
          <input
            type="number"
            min="0"
            name="clicks"
            value={form.clicks}
            onChange={handleChange}
          />
        </div>

        <div className="form-field">
          <label>Agregar al carrito</label>
          <input
            type="number"
            min="0"
            name="carts"
            value={form.carts}
            onChange={handleChange}
          />
        </div>

        <div className="form-field">
          <label>Compras previas</label>
          <input
            type="number"
            min="0"
            name="purchases"
            value={form.purchases}
            onChange={handleChange}
          />
        </div>

        <div className="form-field">
          <label>Favoritos</label>
          <input
            type="number"
            min="0"
            name="favorites"
            value={form.favorites}
            onChange={handleChange}
          />
        </div>

        <div className="form-actions">
          <button type="submit" disabled={loading}>
            {loading ? 'Calculando...' : 'Obtener predicciones'}
          </button>
        </div>
      </form>
    </div>
  )
}

// Panel para mostrar el resultado de las predicciones
function PredictionResults({ results }) {
  if (!results) return null

  return (
    <div className="card">
      <h2>Resultados de predicción</h2>
      <p className="card-subtitle">
        Probabilidad estimada de compra según cada modelo.
      </p>
      <div className="table-wrapper">
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Modelo</th>
              <th>Probabilidad de compra</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={r.model}>
                <td>{r.model}</td>
                <td>{(r.prob * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small-note">
        * Estos valores dependen del endpoint de predicción que implementes en tu backend.
      </p>
    </div>
  )
}

// Gráfico comparativo de rendimiento de modelos (interactivo con Tooltips)
function ModelPerformanceChart({ models }) {
  const data = models.map((m) => ({
    name: m.name,
    auc: m.auc,
    f1: m.f1,
    brier: m.brier
  }))

  return (
    <div className="card">
      <h2>Rendimiento de modelos (interactivo)</h2>
      <p className="card-subtitle">
        Compara simultáneamente AUC, F1 y Brier score para cada modelo. 
        Pasa el cursor sobre las barras para ver los valores exactos.
      </p>
      <div style={{ width: '100%', height: 320 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            {/* AQUÍ CAMBIAMOS */}
            <Bar dataKey="brier" name="Brier score" fill="#ef4444" />      {/* rojo */}
            <Bar dataKey="f1"    name="F1-score"    fill="#22c55e" />      {/* verde */}
            <Bar dataKey="auc"   name="ROC AUC"     fill="#3b82f6" />      {/* azul */}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}


// Gráfico de importancia de características por modelo
function FeatureImportanceChart({ data }) {
  return (
    <div className="card">
      <h2>Importancia de características</h2>
      <p className="card-subtitle">
        Visualización comparativa de la importancia relativa de cada característica
        en los diferentes modelos.
      </p>
      <div style={{ width: '100%', height: 320 }}>
        <ResponsiveContainer>
          <BarChart
            data={data}
            margin={{ top: 10, right: 20, bottom: 40, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="feature" angle={-20} textAnchor="end" />
            <YAxis />
            <Tooltip />
            <Legend />
            {/* AQUÍ CAMBIAMOS */}
            <Bar dataKey="Logistic" name="Logistic Regression" fill="#6366f1" /> {/* morado */}
            <Bar dataKey="LightGBM" name="LightGBM"            fill="#0ea5e9" /> {/* celeste */}
            <Bar dataKey="CatBoost"  name="CatBoost"           fill="#f97316" /> {/* naranja */}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}


export default function App() {
  const [models] = useState(INITIAL_MODELS)
  const [selectedModelName, setSelectedModelName] = useState(models[0].name)
  const [metric, setMetric] = useState('auc')
  const [predictionResults, setPredictionResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)

  const selectedModel = models.find((m) => m.name === selectedModelName)

  // Llamada al backend para predecir con nuevos datos
  const handlePredict = async (features) => {
    setLoading(true)
    setErrorMsg(null)
    setPredictionResults(null)

    try {
      // Esperando que tengas un endpoint POST /predict
      // que reciba { features } y responda { modelPredictions: [...] }
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features })
      })

      if (!response.ok) {
        throw new Error(`Error en la API: ${response.status}`)
      }

      const data = await response.json()

      // Si el backend aún no está listo, puedes simular:
      // const data = {
      //   modelPredictions: models.map((m) => ({
      //     model: m.name,
      //     prob: Math.random()
      //   }))
      // }

      setPredictionResults(data.modelPredictions)
    } catch (err) {
      console.error(err)
      setErrorMsg('No se pudo obtener la predicción. Revisa el backend o la URL de la API.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-root">
      <header className="app-header">
        <h1>Dashboard de Modelos – Proyecto Data Science</h1>
        <p className="app-header-subtitle">
          Visualización y comparación de modelos de predicción de compra,
          con soporte para predicciones interactivas.
        </p>
      </header>

      <main className="app-main">
        {/* Panel lado izquierdo: métricas y comparación */}
        <section className="layout-two-columns">
          <div className="left-column">
            <MetricSelector metric={metric} onChange={setMetric} />
            <MetricBars models={models} metric={metric} />
            
            {/* NUEVO: gráfico comparativo interactivo de rendimiento */}
            <ModelPerformanceChart models={models} />

            {/* NUEVO: gráfico de importancia de características */}
            <FeatureImportanceChart data={FEATURE_IMPORTANCE} />

            <ModelsTable models={models} />
          </div>


          {/* Panel lado derecho: detalle + predicción */}
          <div className="right-column">
            <div className="card">
              <h2>Seleccionar modelo principal</h2>
              <p className="card-subtitle">
                El modelo seleccionado se muestra con más detalle y se puede usar
                para análisis cualitativo en la presentación.
              </p>
              <select
                value={selectedModelName}
                onChange={(e) => setSelectedModelName(e.target.value)}
              >
                {models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>

            <ModelDetails model={selectedModel} />

            <PredictionForm onSubmit={handlePredict} loading={loading} />
            {errorMsg && <p className="error-msg">{errorMsg}</p>}

            <PredictionResults results={predictionResults} />
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <p>
          Proyecto 2 – CC3084 Data Science &middot; 
          Dashboard interactivo para visualización y experimentación con modelos.
        </p>
      </footer>
    </div>
  )
}
