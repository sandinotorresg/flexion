# ============================================================
# STREAMLIT APP
# Predicción directa e inversa de resistencia a flexión y costo
# ABS - FDM - Red neuronal MLP
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Predicción ABS FDM",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #0B2341;
}
.subtitle {
    font-size: 18px;
    color: #4A4A4A;
}
.info-box {
    background-color: #F4F7FB;
    padding: 18px;
    border-radius: 12px;
    border-left: 6px solid #0B2341;
}
.warning-box {
    background-color: #FFF8E6;
    padding: 16px;
    border-radius: 10px;
    border-left: 6px solid #E69500;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='main-title'>🧠 Predicción directa e inversa en piezas ABS impresas por FDM</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Modelo basado en redes neuronales MLP para estimar resistencia a flexión y costo total de fabricación.</div>",
    unsafe_allow_html=True
)

st.write("")

nombre_proyecto = st.text_input(
    "Nombre del estudiante / proyecto",
    value="Trabajo de Integración Curricular - ABS FDM"
)

st.markdown(
    """
    <div class='warning-box'>
    <b>Nota académica:</b> El modelo de flexión presenta capacidad predictiva moderada.
    Las condiciones recomendadas deben validarse experimentalmente antes de su aplicación real.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATOS
# ============================================================

@st.cache_data
def cargar_datos():
    data = [
        ("1.1",260,20,0,29.14,0.431), ("1.2",260,20,0,29.15,0.431),
        ("1.3",260,20,0,28.82,0.431), ("1.4",260,20,0,30.88,0.431),
        ("1.5",260,20,0,29.13,0.431),

        ("2.1",260,60,45,20.85,1.782), ("2.2",260,60,45,24.94,1.782),
        ("2.3",260,60,45,26.19,1.782), ("2.4",260,60,45,20.22,1.782),
        ("2.5",260,60,45,20.85,1.782),

        ("3.1",260,100,90,18.92,0.791), ("3.2",260,100,90,18.35,0.791),
        ("3.3",260,100,90,16.14,0.791), ("3.4",260,100,90,19.32,0.791),
        ("3.5",260,100,90,18.71,0.791),

        ("4.1",270,20,45,19.15,1.710), ("4.2",270,20,45,18.71,1.710),
        ("4.3",270,20,45,19.94,1.710), ("4.4",270,20,45,26.95,1.710),
        ("4.5",270,20,45,19.85,1.710),

        ("5.1",270,60,90,25.40,0.593), ("5.2",270,60,90,20.59,0.593),
        ("5.3",270,60,90,22.14,0.593), ("5.4",270,60,90,27.13,0.593),
        ("5.5",270,60,90,21.04,0.593),

        ("6.1",270,100,0,35.53,0.699), ("6.2",270,100,0,33.77,0.699),
        ("6.3",270,100,0,33.59,0.699), ("6.4",270,100,0,41.61,0.699),
        ("6.5",270,100,0,34.19,0.699),

        ("7.1",280,20,90,24.65,0.593), ("7.2",280,20,90,25.62,0.593),
        ("7.3",280,20,90,24.05,0.593), ("7.4",280,20,90,19.86,0.593),
        ("7.5",280,20,90,20.46,0.593),

        ("8.1",280,60,0,30.52,0.568), ("8.2",280,60,0,40.12,0.568),
        ("8.3",280,60,0,40.18,0.568), ("8.4",280,60,0,31.95,0.568),
        ("8.5",280,60,0,39.66,0.568),

        ("9.1",280,100,45,32.77,1.854), ("9.2",280,100,45,29.01,1.854),
        ("9.3",280,100,45,33.37,1.854), ("9.4",280,100,45,32.01,1.854),
        ("9.5",280,100,45,33.82,1.854),

        ("10",262,30,15,19.92,0.241), ("11",266,30,30,23.24,0.346),
        ("12",270,30,45,28.81,0.352), ("13",274,30,60,30.11,0.329),
        ("14",278,30,75,23.32,0.250),

        ("15",262,45,30,24.21,0.352), ("16",266,45,45,27.49,0.341),
        ("17",270,45,60,33.30,0.334), ("18",274,45,75,27.13,0.257),
        ("19",278,45,15,14.48,0.248),

        ("20",262,60,45,21.85,0.345), ("21",266,60,60,34.17,0.339),
        ("22",270,60,75,26.86,0.264), ("23",274,60,15,17.22,0.242),
        ("24",278,60,30,28.29,0.342),

        ("25",262,75,60,32.39,0.344), ("26",266,75,75,28.15,0.270),
        ("27",270,75,15,23.33,0.247), ("28",274,75,30,28.87,0.346),
        ("29",278,75,45,28.07,0.350),
    ]

    return pd.DataFrame(
        data,
        columns=[
            "Ensayo",
            "Temperatura",
            "Densidad",
            "Orientacion",
            "Flexion_MPa",
            "Costo_USD"
        ]
    )


df = cargar_datos()

X = df[["Temperatura", "Densidad", "Orientacion"]]
y_flexion = df["Flexion_MPa"]
y_costo = df["Costo_USD"]

costo_min_real = float(y_costo.min())
costo_max_real = float(y_costo.max())


# ============================================================
# MODELOS
# ============================================================

@st.cache_resource
def entrenar_modelos():
    modelo_flexion = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(8,),
            activation="tanh",
            solver="lbfgs",
            alpha=0.01,
            max_iter=10000,
            random_state=42
        ))
    ])

    modelo_costo = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(5, 5),
            activation="tanh",
            solver="lbfgs",
            alpha=0.01,
            max_iter=10000,
            random_state=42
        ))
    ])

    modelo_flexion.fit(X, y_flexion)
    modelo_costo.fit(X, y_costo)

    return modelo_flexion, modelo_costo


modelo_flexion, modelo_costo = entrenar_modelos()


def limitar_costo(costo):
    return max(costo_min_real, min(costo, costo_max_real))


def predecir_directo(temperatura, densidad, orientacion):
    entrada = pd.DataFrame({
        "Temperatura": [temperatura],
        "Densidad": [densidad],
        "Orientacion": [orientacion]
    })

    flexion = float(modelo_flexion.predict(entrada)[0])
    costo = float(modelo_costo.predict(entrada)[0])
    costo = limitar_costo(costo)

    return flexion, costo


def prediccion_inversa_resistencia_min_costo(flexion_objetivo):
    def funcion_objetivo(parametros):
        temperatura, densidad, orientacion = parametros
        flexion_predicha, costo_predicho = predecir_directo(
            temperatura, densidad, orientacion
        )

        error_flexion = abs(flexion_objetivo - flexion_predicha)

        return error_flexion + 0.35 * costo_predicho

    resultado = differential_evolution(
        funcion_objetivo,
        bounds=[(260, 280), (20, 100), (0, 90)],
        maxiter=600,
        popsize=15,
        tol=1e-6,
        seed=42,
        polish=True
    )

    T, D, O = resultado.x
    flexion_predicha, costo_predicho = predecir_directo(T, D, O)

    return T, D, O, flexion_predicha, costo_predicho


def prediccion_inversa_doble_objetivo(flexion_objetivo, costo_objetivo):
    def funcion_objetivo(parametros):
        temperatura, densidad, orientacion = parametros
        flexion_predicha, costo_predicho = predecir_directo(
            temperatura, densidad, orientacion
        )

        error_flexion = abs(flexion_objetivo - flexion_predicha)
        error_costo = abs(costo_objetivo - costo_predicho)

        return error_flexion + 10 * error_costo

    resultado = differential_evolution(
        funcion_objetivo,
        bounds=[(260, 280), (20, 100), (0, 90)],
        maxiter=600,
        popsize=15,
        tol=1e-6,
        seed=42,
        polish=True
    )

    T, D, O = resultado.x
    flexion_predicha, costo_predicho = predecir_directo(T, D, O)

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


# ============================================================
# INTERFAZ
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Predicción directa",
    "🎯 Predicción inversa",
    "📊 Estadística del modelo",
    "🗺️ Superficies de respuesta",
    "📁 Datos"
])


# ============================================================
# TAB 1
# ============================================================

with tab1:
    st.header("Predicción directa")

    st.markdown(
        """
        <div class='info-box'>
        Ingrese los parámetros de impresión dentro del dominio experimental:
        <b>260–280 °C</b>, <b>20–100 %</b> de densidad y <b>0–90°</b> de orientación.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        temperatura = st.slider("Temperatura (°C)", 260.0, 280.0, 270.0, 0.5)

    with col2:
        densidad = st.slider("Densidad de relleno (%)", 20.0, 100.0, 60.0, 1.0)

    with col3:
        orientacion = st.slider("Orientación de impresión (°)", 0.0, 90.0, 45.0, 1.0)

    flexion_predicha, costo_predicho = predecir_directo(
        temperatura, densidad, orientacion
    )

    st.write("")

    col4, col5 = st.columns(2)

    col4.metric(
        "Resistencia a flexión predicha",
        f"{flexion_predicha:.2f} MPa"
    )

    col5.metric(
        "Costo total predicho",
        f"{costo_predicho:.3f} USD"
    )


# ============================================================
# TAB 2
# ============================================================

with tab2:
    st.header("Predicción inversa")

    st.subheader("Opción 1: resistencia objetivo con menor costo")

    flexion_objetivo = st.number_input(
        "Resistencia a flexión objetivo (MPa)",
        min_value=14.0,
        max_value=42.0,
        value=30.0,
        step=0.5
    )

    if st.button("Buscar parámetros recomendados", key="inv1"):
        T, D, O, flexion_predicha, costo_predicho = prediccion_inversa_resistencia_min_costo(
            flexion_objetivo
        )

        st.success("Parámetros recomendados")

        c1, c2, c3 = st.columns(3)
        c1.metric("Temperatura", f"{T:.2f} °C")
        c2.metric("Densidad", f"{D:.2f} %")
        c3.metric("Orientación", f"{O:.2f} °")

        c4, c5, c6 = st.columns(3)
        c4.metric("Flexión predicha", f"{flexion_predicha:.2f} MPa")
        c5.metric("Costo predicho", f"{costo_predicho:.3f} USD")
        c6.metric("Error", f"{abs(flexion_objetivo - flexion_predicha):.2f} MPa")

    st.divider()

    st.subheader("Opción 2: resistencia objetivo y costo objetivo")

    c1, c2 = st.columns(2)

    with c1:
        flexion_objetivo_2 = st.number_input(
            "Resistencia objetivo (MPa)",
            min_value=14.0,
            max_value=42.0,
            value=30.0,
            step=0.5,
            key="flexion2"
        )

    with c2:
        costo_objetivo = st.number_input(
            "Costo objetivo (USD)",
            min_value=costo_min_real,
            max_value=costo_max_real,
            value=0.350,
            step=0.01
        )

    if st.button("Buscar solución doble objetivo", key="inv2"):
        T, D, O, flexion_predicha, costo_predicho = prediccion_inversa_doble_objetivo(
            flexion_objetivo_2, costo_objetivo
        )

        st.success("Parámetros recomendados")

        c1, c2, c3 = st.columns(3)
        c1.metric("Temperatura", f"{T:.2f} °C")
        c2.metric("Densidad", f"{D:.2f} %")
        c3.metric("Orientación", f"{O:.2f} °")

        c4, c5 = st.columns(2)
        c4.metric("Flexión predicha", f"{flexion_predicha:.2f} MPa")
        c5.metric("Costo predicho", f"{costo_predicho:.3f} USD")


# ============================================================
# TAB 3
# ============================================================

with tab3:
    st.header("Estadística del modelo")

    st.subheader(nombre_proyecto)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Modelo de flexión")
        st.metric("R² ajuste", f"{r2_flexion:.3f}")
        st.metric("RMSE", f"{rmse_flexion:.3f} MPa")
        st.metric("MAE", f"{mae_flexion:.3f} MPa")
        st.caption("Arquitectura MLP: 3 entradas – 8 neuronas ocultas – 1 salida")

    with col2:
        st.markdown("### Modelo de costo")
        st.metric("R² ajuste", f"{r2_costo:.3f}")
        st.metric("RMSE", f"{rmse_costo:.4f} USD")
        st.metric("MAE", f"{mae_costo:.4f} USD")
        st.caption("Arquitectura MLP: 3 entradas – 5 – 5 – 1")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.scatter(y_flexion, pred_flexion)
        min_val = min(y_flexion.min(), pred_flexion.min())
        max_val = max(y_flexion.max(), pred_flexion.max())
        ax1.plot([min_val, max_val], [min_val, max_val], "--")
        ax1.set_xlabel("Flexión experimental (MPa)")
        ax1.set_ylabel("Flexión predicha (MPa)")
        ax1.set_title("Experimental vs predicho - Flexión")
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(y_costo, pred_costo)
        min_val = min(y_costo.min(), pred_costo.min())
        max_val = max(y_costo.max(), pred_costo.max())
        ax2.plot([min_val, max_val], [min_val, max_val], "--")
        ax2.set_xlabel("Costo experimental (USD)")
        ax2.set_ylabel("Costo predicho (USD)")
        ax2.set_title("Experimental vs predicho - Costo")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.scatter(y_costo, y_flexion)
    ax3.set_xlabel("Costo experimental (USD)")
    ax3.set_ylabel("Flexión experimental (MPa)")
    ax3.set_title("Relación resistencia-costo")
    ax3.grid(alpha=0.3)
    st.pyplot(fig3)


# ============================================================
# TAB 4
# ============================================================

with tab4:
    st.header("Superficies de respuesta")

    orientacion_fija = st.slider(
        "Orientación fija para la superficie (°)",
        min_value=0,
        max_value=90,
        value=45,
        step=5
    )

    temp_range = np.linspace(260, 280, 60)
    dens_range = np.linspace(20, 100, 60)

    T_grid, D_grid = np.meshgrid(temp_range, dens_range)

    Z_flexion = np.zeros_like(T_grid)
    Z_costo = np.zeros_like(T_grid)

    for i in range(T_grid.shape[0]):
        for j in range(T_grid.shape[1]):
            flex, cost = predecir_directo(
                T_grid[i, j],
                D_grid[i, j],
                orientacion_fija
            )

            Z_flexion[i, j] = flex
            Z_costo[i, j] = cost

    c1, c2 = st.columns(2)

    with c1:
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        contour1 = ax4.contourf(D_grid, T_grid, Z_flexion, levels=20)
        fig4.colorbar(contour1, ax=ax4, label="Flexión predicha (MPa)")
        ax4.set_xlabel("Densidad de relleno (%)")
        ax4.set_ylabel("Temperatura (°C)")
        ax4.set_title(f"Flexión predicha | Orientación = {orientacion_fija}°")
        st.pyplot(fig4)

    with c2:
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        contour2 = ax5.contourf(D_grid, T_grid, Z_costo, levels=20)
        fig5.colorbar(contour2, ax=ax5, label="Costo predicho (USD)")
        ax5.set_xlabel("Densidad de relleno (%)")
        ax5.set_ylabel("Temperatura (°C)")
        ax5.set_title(f"Costo predicho | Orientación = {orientacion_fija}°")
        st.pyplot(fig5)


# ============================================================
# TAB 5
# ============================================================

with tab5:
    st.header("Base de datos experimental")

    st.dataframe(df, use_container_width=True)

    st.download_button(
        label="Descargar base de datos CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="base_datos_abs_fdm.csv",
        mime="text/csv"
    )