# ============================================================
# STREAMLIT APP v2.0
# Predicción directa, inversa y optimización multicriterio
# Resistencia a flexión y costo en piezas ABS impresas por FDM
# Modelo: Redes neuronales MLP
# ============================================================

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy.optimize import differential_evolution
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(page_title="Predicción ABS FDM", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.main-title {font-size: 42px; font-weight: 800; color: #0B2341; line-height: 1.1;}
.subtitle {font-size: 18px; color: #4A4A4A; margin-top: 8px;}
.box-blue {background-color: #F4F7FB; padding: 18px; border-radius: 14px; border-left: 6px solid #0B2341; margin-bottom: 12px;}
.box-yellow {background-color: #FFF8E6; padding: 16px; border-radius: 14px; border-left: 6px solid #E69500; margin-bottom: 12px;}
.small-note {font-size: 14px; color: #555555;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# ENCABEZADO
# ============================================================

col_logo1, col_title, col_logo2 = st.columns([1, 5, 1])

with col_logo1:
    st.markdown("### LOGO EPN")
    logo_epn = st.file_uploader("Cargar logo EPN", type=["png", "jpg", "jpeg"], key="logo_epn", label_visibility="collapsed")
    if logo_epn:
        st.image(logo_epn, use_container_width=True)

with col_title:
    st.markdown("<div class='main-title'>🧠 Predicción directa e inversa en piezas ABS impresas por FDM</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Aplicación basada en redes neuronales MLP para estimar resistencia a flexión y costo total de fabricación.</div>", unsafe_allow_html=True)

with col_logo2:
    st.markdown("### LOGO FCA")
    logo_fca = st.file_uploader("Cargar logo FCA", type=["png", "jpg", "jpeg"], key="logo_fca", label_visibility="collapsed")
    if logo_fca:
        st.image(logo_fca, use_container_width=True)

st.write("")

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    nombre_estudiante = st.text_input("Nombre del estudiante / autora", value="Nombre del estudiante")
with col_info2:
    nombre_director = st.text_input("Director/a", value="Nombre del director/a")
with col_info3:
    nombre_proyecto = st.text_input("Nombre del proyecto", value="Trabajo de Integración Curricular - ABS FDM")

st.markdown("""
<div class='box-yellow'>
<b>Nota académica:</b> El modelo de flexión presenta capacidad predictiva moderada. El modelo de costo presenta alta capacidad predictiva. Las condiciones recomendadas deben validarse experimentalmente antes de su aplicación real.
</div>
""", unsafe_allow_html=True)

# ============================================================
# DATOS
# ============================================================

@st.cache_data
def cargar_datos():
    data = [
        ("1.1",260,20,0,29.14,0.431),("1.2",260,20,0,29.15,0.431),("1.3",260,20,0,28.82,0.431),("1.4",260,20,0,30.88,0.431),("1.5",260,20,0,29.13,0.431),
        ("2.1",260,60,45,20.85,1.782),("2.2",260,60,45,24.94,1.782),("2.3",260,60,45,26.19,1.782),("2.4",260,60,45,20.22,1.782),("2.5",260,60,45,20.85,1.782),
        ("3.1",260,100,90,18.92,0.791),("3.2",260,100,90,18.35,0.791),("3.3",260,100,90,16.14,0.791),("3.4",260,100,90,19.32,0.791),("3.5",260,100,90,18.71,0.791),
        ("4.1",270,20,45,19.15,1.710),("4.2",270,20,45,18.71,1.710),("4.3",270,20,45,19.94,1.710),("4.4",270,20,45,26.95,1.710),("4.5",270,20,45,19.85,1.710),
        ("5.1",270,60,90,25.40,0.593),("5.2",270,60,90,20.59,0.593),("5.3",270,60,90,22.14,0.593),("5.4",270,60,90,27.13,0.593),("5.5",270,60,90,21.04,0.593),
        ("6.1",270,100,0,35.53,0.699),("6.2",270,100,0,33.77,0.699),("6.3",270,100,0,33.59,0.699),("6.4",270,100,0,41.61,0.699),("6.5",270,100,0,34.19,0.699),
        ("7.1",280,20,90,24.65,0.593),("7.2",280,20,90,25.62,0.593),("7.3",280,20,90,24.05,0.593),("7.4",280,20,90,19.86,0.593),("7.5",280,20,90,20.46,0.593),
        ("8.1",280,60,0,30.52,0.568),("8.2",280,60,0,40.12,0.568),("8.3",280,60,0,40.18,0.568),("8.4",280,60,0,31.95,0.568),("8.5",280,60,0,39.66,0.568),
        ("9.1",280,100,45,32.77,1.854),("9.2",280,100,45,29.01,1.854),("9.3",280,100,45,33.37,1.854),("9.4",280,100,45,32.01,1.854),("9.5",280,100,45,33.82,1.854),
        ("10",262,30,15,19.92,0.241),("11",266,30,30,23.24,0.346),("12",270,30,45,28.81,0.352),("13",274,30,60,30.11,0.329),("14",278,30,75,23.32,0.250),
        ("15",262,45,30,24.21,0.352),("16",266,45,45,27.49,0.341),("17",270,45,60,33.30,0.334),("18",274,45,75,27.13,0.257),("19",278,45,15,14.48,0.248),
        ("20",262,60,45,21.85,0.345),("21",266,60,60,34.17,0.339),("22",270,60,75,26.86,0.264),("23",274,60,15,17.22,0.242),("24",278,60,30,28.29,0.342),
        ("25",262,75,60,32.39,0.344),("26",266,75,75,28.15,0.270),("27",270,75,15,23.33,0.247),("28",274,75,30,28.87,0.346),("29",278,75,45,28.07,0.350),
    ]
    return pd.DataFrame(data, columns=["Ensayo", "Temperatura", "Densidad", "Orientacion", "Flexion_MPa", "Costo_USD"])

df = cargar_datos()
X = df[["Temperatura", "Densidad", "Orientacion"]]
y_flexion = df["Flexion_MPa"]
y_costo = df["Costo_USD"]

TEMP_MIN, TEMP_MAX = 260.0, 280.0
DENS_MIN, DENS_MAX = 20.0, 100.0
ORIENT_MIN, ORIENT_MAX = 0.0, 90.0
costo_min_real = float(y_costo.min())
costo_max_real = float(y_costo.max())

# ============================================================
# MODELOS
# ============================================================

@st.cache_resource
def entrenar_modelos():
    modelo_flexion = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs", alpha=0.01, max_iter=10000, random_state=42))
    ])
    modelo_costo = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(5, 5), activation="tanh", solver="lbfgs", alpha=0.01, max_iter=10000, random_state=42))
    ])
    modelo_flexion.fit(X, y_flexion)
    modelo_costo.fit(X, y_costo)
    return modelo_flexion, modelo_costo

modelo_flexion, modelo_costo = entrenar_modelos()

def limitar_costo(costo):
    return max(costo_min_real, min(float(costo), costo_max_real))

def predecir_directo(temperatura, densidad, orientacion, limitar=True):
    entrada = pd.DataFrame({"Temperatura": [temperatura], "Densidad": [densidad], "Orientacion": [orientacion]})
    flexion = float(modelo_flexion.predict(entrada)[0])
    costo = float(modelo_costo.predict(entrada)[0])
    if limitar:
        costo = limitar_costo(costo)
    return flexion, costo

def prediccion_inversa_resistencia_min_costo(flexion_objetivo):
    def funcion_objetivo(parametros):
        temperatura, densidad, orientacion = parametros
        flexion_predicha, costo_predicho = predecir_directo(temperatura, densidad, orientacion, limitar=True)
        error_flexion = abs(flexion_objetivo - flexion_predicha)
        return error_flexion + 0.35 * costo_predicho
    resultado = differential_evolution(funcion_objetivo, bounds=[(TEMP_MIN, TEMP_MAX), (DENS_MIN, DENS_MAX), (ORIENT_MIN, ORIENT_MAX)], maxiter=500, popsize=12, tol=1e-6, seed=42, polish=True)
    T, D, O = resultado.x
    flexion_predicha, costo_predicho = predecir_directo(T, D, O, limitar=True)
    return T, D, O, flexion_predicha, costo_predicho

def prediccion_inversa_doble_objetivo(flexion_objetivo, costo_objetivo):
    def funcion_objetivo(parametros):
        temperatura, densidad, orientacion = parametros
        flexion_predicha, costo_predicho = predecir_directo(temperatura, densidad, orientacion, limitar=True)
        error_flexion = abs(flexion_objetivo - flexion_predicha)
        error_costo = abs(costo_objetivo - costo_predicho)
        return error_flexion + 10 * error_costo
    resultado = differential_evolution(funcion_objetivo, bounds=[(TEMP_MIN, TEMP_MAX), (DENS_MIN, DENS_MAX), (ORIENT_MIN, ORIENT_MAX)], maxiter=500, popsize=12, tol=1e-6, seed=42, polish=True)
    T, D, O = resultado.x
    flexion_predicha, costo_predicho = predecir_directo(T, D, O, limitar=True)
    return T, D, O, flexion_predicha, costo_predicho

def optimizacion_multicriterio(peso_flexion=1.0, peso_costo=1.0):
    flex_min, flex_max = float(y_flexion.min()), float(y_flexion.max())
    def funcion_objetivo(parametros):
        temperatura, densidad, orientacion = parametros
        flexion_predicha, costo_predicho = predecir_directo(temperatura, densidad, orientacion, limitar=True)
        flex_norm = (flexion_predicha - flex_min) / (flex_max - flex_min)
        costo_norm = (costo_predicho - costo_min_real) / (costo_max_real - costo_min_real)
        return -peso_flexion * flex_norm + peso_costo * costo_norm
    resultado = differential_evolution(funcion_objetivo, bounds=[(TEMP_MIN, TEMP_MAX), (DENS_MIN, DENS_MAX), (ORIENT_MIN, ORIENT_MAX)], maxiter=500, popsize=12, tol=1e-6, seed=42, polish=True)
    T, D, O = resultado.x
    flexion_predicha, costo_predicho = predecir_directo(T, D, O, limitar=True)
    return T, D, O, flexion_predicha, costo_predicho

# ============================================================
# MÉTRICAS
# ============================================================

pred_flexion = modelo_flexion.predict(X)
pred_costo_raw = modelo_costo.predict(X)
pred_costo = np.array([limitar_costo(c) for c in pred_costo_raw])

r2_flexion = r2_score(y_flexion, pred_flexion)
rmse_flexion = np.sqrt(mean_squared_error(y_flexion, pred_flexion))
mae_flexion = mean_absolute_error(y_flexion, pred_flexion)
r2_costo = r2_score(y_costo, pred_costo)
rmse_costo = np.sqrt(mean_squared_error(y_costo, pred_costo))
mae_costo = mean_absolute_error(y_costo, pred_costo)
cv_flexion = float(y_flexion.std() / y_flexion.mean() * 100)
cv_costo = float(y_costo.std() / y_costo.mean() * 100)

def nivel_modelo(r2):
    if r2 >= 0.95:
        return "Excelente"
    if r2 >= 0.80:
        return "Bueno / moderado-alto"
    if r2 >= 0.60:
        return "Moderado"
    return "Bajo"

# ============================================================
# GRÁFICOS
# ============================================================

def fig_exp_vs_pred(y_real, y_pred, titulo, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_real, y_pred)
    min_val = min(float(np.min(y_real)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_real)), float(np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], "--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def fig_resistencia_costo():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_costo, y_flexion)
    ax.set_xlabel("Costo experimental (USD)")
    ax.set_ylabel("Flexión experimental (MPa)")
    ax.set_title("Relación resistencia-costo")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def fig_importancia(modelo, y, titulo):
    perm = permutation_importance(modelo, X, y, n_repeats=30, random_state=42)
    imp = pd.DataFrame({"Variable": ["Temperatura", "Densidad", "Orientación"], "Importancia": perm.importances_mean}).sort_values("Importancia", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(imp["Variable"], imp["Importancia"])
    ax.set_ylabel("Importancia por permutación")
    ax.set_title(titulo)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, imp

def generar_superficie(variable_fija, valor_fijo, respuesta):
    n = 60
    if variable_fija == "Orientación":
        x_label, y_label = "Densidad de relleno (%)", "Temperatura (°C)"
        x_vals = np.linspace(DENS_MIN, DENS_MAX, n)
        y_vals = np.linspace(TEMP_MIN, TEMP_MAX, n)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X_grid)
        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                flex, cost = predecir_directo(Y_grid[i, j], X_grid[i, j], valor_fijo, limitar=True)
                Z[i, j] = flex if respuesta == "Flexión" else cost
    elif variable_fija == "Temperatura":
        x_label, y_label = "Densidad de relleno (%)", "Orientación (°)"
        x_vals = np.linspace(DENS_MIN, DENS_MAX, n)
        y_vals = np.linspace(ORIENT_MIN, ORIENT_MAX, n)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X_grid)
        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                flex, cost = predecir_directo(valor_fijo, X_grid[i, j], Y_grid[i, j], limitar=True)
                Z[i, j] = flex if respuesta == "Flexión" else cost
    else:
        x_label, y_label = "Temperatura (°C)", "Orientación (°)"
        x_vals = np.linspace(TEMP_MIN, TEMP_MAX, n)
        y_vals = np.linspace(ORIENT_MIN, ORIENT_MAX, n)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X_grid)
        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                flex, cost = predecir_directo(X_grid[i, j], valor_fijo, Y_grid[i, j], limitar=True)
                Z[i, j] = flex if respuesta == "Flexión" else cost
    fig, ax = plt.subplots(figsize=(7, 5))
    contour = ax.contourf(X_grid, Y_grid, Z, levels=20)
    unit = "MPa" if respuesta == "Flexión" else "USD"
    fig.colorbar(contour, ax=ax, label=f"{respuesta} predicha ({unit})")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Superficie de respuesta - {respuesta}\n{variable_fija} fija = {valor_fijo}")
    fig.tight_layout()
    return fig

def descargar_figura(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    return buffer

# ============================================================
# PANEL SUPERIOR
# ============================================================

st.write("")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Observaciones", f"{len(df)}")
m2.metric("Material", "ABS")
m3.metric("R² flexión", f"{r2_flexion:.3f}")
m4.metric("R² costo", f"{r2_costo:.3f}")

st.markdown(f"""
<div class='box-blue'>
<b>Proyecto:</b> {nombre_proyecto}<br>
<b>Estudiante:</b> {nombre_estudiante}<br>
<b>Director/a:</b> {nombre_director}<br>
<b>Dominio experimental:</b> Temperatura {TEMP_MIN:.0f}–{TEMP_MAX:.0f} °C, densidad {DENS_MIN:.0f}–{DENS_MAX:.0f} %, orientación {ORIENT_MIN:.0f}–{ORIENT_MAX:.0f}°.
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📌 Predicción directa", "🎯 Predicción inversa", "⚙️ Optimización", "📊 Estadística del modelo", "🗺️ Superficies de respuesta", "📁 Datos"])

with tab1:
    st.header("Predicción directa")
    st.markdown("<div class='box-blue'>Modifique los parámetros de impresión. La predicción se actualiza automáticamente.</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        temperatura = st.slider("Temperatura (°C)", min_value=TEMP_MIN, max_value=TEMP_MAX, value=270.0, step=0.5, key="slider_temperatura_directa")
    with col2:
        densidad = st.slider("Densidad de relleno (%)", min_value=DENS_MIN, max_value=DENS_MAX, value=60.0, step=1.0, key="slider_densidad_directa")
    with col3:
        orientacion = st.slider("Orientación de impresión (°)", min_value=ORIENT_MIN, max_value=ORIENT_MAX, value=45.0, step=1.0, key="slider_orientacion_directa")
    flexion_predicha, costo_predicho = predecir_directo(temperatura, densidad, orientacion, limitar=True)
    st.subheader("Resultados de predicción")
    c1, c2 = st.columns(2)
    c1.metric("Resistencia a flexión predicha", f"{flexion_predicha:.2f} MPa")
    c2.metric("Costo total predicho", f"{costo_predicho:.3f} USD")
    st.caption(f"Entrada evaluada: T = {temperatura:.1f} °C | Densidad = {densidad:.1f} % | Orientación = {orientacion:.1f}°")

with tab2:
    st.header("Predicción inversa")
    st.markdown("<div class='box-blue'>La predicción inversa recomienda parámetros de impresión a partir de una variable respuesta objetivo.</div>", unsafe_allow_html=True)
    st.subheader("Opción 1: resistencia objetivo con menor costo estimado")
    flexion_objetivo = st.number_input("Resistencia a flexión objetivo (MPa)", min_value=14.0, max_value=42.0, value=30.0, step=0.5, key="obj_flexion_simple")
    if st.button("Buscar parámetros recomendados", key="boton_inv1"):
        T, D, O, flexion_predicha, costo_predicho = prediccion_inversa_resistencia_min_costo(flexion_objetivo)
        st.success("Parámetros recomendados")
        a, b, c = st.columns(3)
        a.metric("Temperatura", f"{T:.2f} °C")
        b.metric("Densidad", f"{D:.2f} %")
        c.metric("Orientación", f"{O:.2f} °")
        d, e, f = st.columns(3)
        d.metric("Flexión predicha", f"{flexion_predicha:.2f} MPa")
        e.metric("Costo predicho", f"{costo_predicho:.3f} USD")
        f.metric("Error", f"{abs(flexion_objetivo - flexion_predicha):.2f} MPa")
    st.divider()
    st.subheader("Opción 2: resistencia objetivo y costo objetivo")
    a, b = st.columns(2)
    with a:
        flexion_objetivo_2 = st.number_input("Resistencia objetivo (MPa)", min_value=14.0, max_value=42.0, value=30.0, step=0.5, key="obj_flexion_doble")
    with b:
        costo_objetivo = st.number_input("Costo objetivo (USD)", min_value=costo_min_real, max_value=costo_max_real, value=0.350, step=0.01, key="obj_costo_doble")
    if st.button("Buscar solución doble objetivo", key="boton_inv2"):
        T, D, O, flexion_predicha, costo_predicho = prediccion_inversa_doble_objetivo(flexion_objetivo_2, costo_objetivo)
        st.success("Parámetros recomendados")
        a, b, c = st.columns(3)
        a.metric("Temperatura", f"{T:.2f} °C")
        b.metric("Densidad", f"{D:.2f} %")
        c.metric("Orientación", f"{O:.2f} °")
        d, e, f, g = st.columns(4)
        d.metric("Flexión predicha", f"{flexion_predicha:.2f} MPa")
        e.metric("Costo predicho", f"{costo_predicho:.3f} USD")
        f.metric("Error flexión", f"{abs(flexion_objetivo_2 - flexion_predicha):.2f} MPa")
        g.metric("Error costo", f"{abs(costo_objetivo - costo_predicho):.3f} USD")

with tab3:
    st.header("Optimización multicriterio")
    st.markdown("<div class='box-blue'>Esta opción busca una configuración que combine alta resistencia a flexión y bajo costo.</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        peso_flexion = st.slider("Peso de la resistencia", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    with col2:
        peso_costo = st.slider("Peso del costo", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    if st.button("Optimizar resistencia-costo", key="boton_multi"):
        T, D, O, flexion_predicha, costo_predicho = optimizacion_multicriterio(peso_flexion=peso_flexion, peso_costo=peso_costo)
        st.success("Configuración multicriterio recomendada")
        a, b, c = st.columns(3)
        a.metric("Temperatura", f"{T:.2f} °C")
        b.metric("Densidad", f"{D:.2f} %")
        c.metric("Orientación", f"{O:.2f} °")
        d, e = st.columns(2)
        d.metric("Flexión estimada", f"{flexion_predicha:.2f} MPa")
        e.metric("Costo estimado", f"{costo_predicho:.3f} USD")

with tab4:
    st.header("Estadística del modelo")
    resumen = pd.DataFrame({"Salida": ["Flexión", "Costo"], "R² ajuste": [r2_flexion, r2_costo], "RMSE": [rmse_flexion, rmse_costo], "MAE": [mae_flexion, mae_costo], "CV experimental (%)": [cv_flexion, cv_costo], "Arquitectura MLP": ["3-8-1", "3-5-5-1"], "Nivel": [nivel_modelo(r2_flexion), nivel_modelo(r2_costo)]})
    st.dataframe(resumen, use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.metric("R² flexión", f"{r2_flexion:.3f}")
        st.metric("RMSE flexión", f"{rmse_flexion:.3f} MPa")
        st.metric("MAE flexión", f"{mae_flexion:.3f} MPa")
    with b:
        st.metric("R² costo", f"{r2_costo:.3f}")
        st.metric("RMSE costo", f"{rmse_costo:.4f} USD")
        st.metric("MAE costo", f"{mae_costo:.4f} USD")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        fig1 = fig_exp_vs_pred(y_flexion, pred_flexion, "Experimental vs predicho - Flexión", "Flexión experimental (MPa)", "Flexión predicha (MPa)")
        st.pyplot(fig1)
        st.download_button("Descargar gráfico flexión", data=descargar_figura(fig1), file_name="experimental_vs_predicho_flexion.png", mime="image/png")
    with c2:
        fig2 = fig_exp_vs_pred(y_costo, pred_costo, "Experimental vs predicho - Costo", "Costo experimental (USD)", "Costo predicho (USD)")
        st.pyplot(fig2)
        st.download_button("Descargar gráfico costo", data=descargar_figura(fig2), file_name="experimental_vs_predicho_costo.png", mime="image/png")
    fig3 = fig_resistencia_costo()
    st.pyplot(fig3)
    st.download_button("Descargar gráfico resistencia-costo", data=descargar_figura(fig3), file_name="relacion_resistencia_costo.png", mime="image/png")
    st.divider()
    st.subheader("Importancia de variables")
    c1, c2 = st.columns(2)
    with c1:
        fig4, imp_flex = fig_importancia(modelo_flexion, y_flexion, "Importancia de variables - Flexión")
        st.pyplot(fig4)
        st.dataframe(imp_flex, use_container_width=True)
    with c2:
        fig5, imp_costo = fig_importancia(modelo_costo, y_costo, "Importancia de variables - Costo")
        st.pyplot(fig5)
        st.dataframe(imp_costo, use_container_width=True)

with tab5:
    st.header("Superficies de respuesta")
    st.markdown("<div class='box-blue'>Las superficies muestran la respuesta predicha al variar dos factores y mantener fijo el tercero.</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        variable_fija = st.selectbox("Variable fija", ["Orientación", "Temperatura", "Densidad"])
    with col2:
        if variable_fija == "Orientación":
            valor_fijo = st.slider("Valor fijo de orientación (°)", 0, 90, 45, 5)
        elif variable_fija == "Temperatura":
            valor_fijo = st.slider("Valor fijo de temperatura (°C)", 260, 280, 270, 1)
        else:
            valor_fijo = st.slider("Valor fijo de densidad (%)", 20, 100, 60, 5)
    c1, c2 = st.columns(2)
    with c1:
        fig_flex = generar_superficie(variable_fija, valor_fijo, "Flexión")
        st.pyplot(fig_flex)
        st.download_button("Descargar superficie flexión", data=descargar_figura(fig_flex), file_name="superficie_flexion.png", mime="image/png")
    with c2:
        fig_cost = generar_superficie(variable_fija, valor_fijo, "Costo")
        st.pyplot(fig_cost)
        st.download_button("Descargar superficie costo", data=descargar_figura(fig_cost), file_name="superficie_costo.png", mime="image/png")

with tab6:
    st.header("Base de datos experimental")
    st.dataframe(df, use_container_width=True)
    st.download_button(label="Descargar base de datos CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="base_datos_abs_fdm.csv", mime="text/csv")
    st.subheader("Resumen descriptivo")
    st.dataframe(df[["Temperatura", "Densidad", "Orientacion", "Flexion_MPa", "Costo_USD"]].describe(), use_container_width=True)
