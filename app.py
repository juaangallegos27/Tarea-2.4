import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt

# Configuración de la app
st.set_page_config(page_title="K-Means con PCA y Comparativa", layout="wide")
st.title("🎯 Clustering Interactivo con K-Means y PCA (Comparación Antes/Después)")
st.write("""
Sube tus datos, aplica **K-Means**, y observa cómo el algoritmo agrupa los puntos en un espacio reducido con **PCA (2D o 3D)**.  
También puedes comparar la distribución **antes y después** del clustering.
""")

# --- Subir archivo ---
st.sidebar.header("📂 Subir datos")
uploaded_file = st.sidebar.file_uploader("Selecciona tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Archivo cargado correctamente.")

    st.write("### Vista previa de los datos:")
    # FIX: mostrar todo el DataFrame (scrollable) y el total de filas/columnas
    st.dataframe(data, use_container_width=True)
    st.caption(f"{data.shape[0]} filas × {data.shape[1]} columnas")

    # Filtrar columnas numéricas
    numeric_cols = data.select_dtypes(
        include=['float64', 'int64', 'float32', 'int32']
    ).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("⚠️ El archivo debe contener al menos dos columnas numéricas.")
    else:
        st.sidebar.header("⚙️ Configuración del modelo")

        # Seleccionar columnas a usar
        selected_cols = st.sidebar.multiselect(
            "Selecciona las columnas numéricas para el clustering:",
            numeric_cols,
            default=numeric_cols
        )

        # Parámetros de clustering
        k = st.sidebar.slider("Número de clusters (k):", 1, 10, 3)

        # Parámetros seleccionables (lo que pediste)
        init = st.sidebar.selectbox("init (método de inicialización)", ["k-means++", "random"], index=0)
        max_iter = st.sidebar.number_input("max_iter (iteraciones máximas)", min_value=10, max_value=1000, value=300, step=10)
        n_init_option = st.sidebar.selectbox("n_init (repeticiones de inicialización)", options=["auto"] + list(range(1, 31)), index=1)
        n_init = n_init_option  # puede ser 'auto' o int

        use_random_state = st.sidebar.checkbox("Fijar random_state", value=True)
        random_state = st.sidebar.number_input("random_state", min_value=0, max_value=10000, value=0, step=1) if use_random_state else None

        n_components = st.sidebar.radio("Visualización PCA:", [2, 3], index=0)

        # --- Datos y modelo ---
        X = data[selected_cols]
        kmeans = KMeans(
            n_clusters=k,
            init=init,
            max_iter=int(max_iter),
            n_init=n_init,
            random_state=random_state
        )
        kmeans.fit(X)
        data['Cluster'] = kmeans.labels_

        # --- PCA ---
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        pca_cols = [f'PCA{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        pca_df['Cluster'] = data['Cluster']

        # --- Visualización antes del clustering ---
        st.subheader("📊 Distribución original (antes de K-Means)")
        if n_components == 2:
            fig_before = px.scatter(
                pca_df, x='PCA1', y='PCA2',
                title="Datos originales proyectados con PCA (sin agrupar)",
                color_discrete_sequence=["gray"]
            )
        else:
            fig_before = px.scatter_3d(
                pca_df, x='PCA1', y='PCA2', z='PCA3',
                title="Datos originales proyectados con PCA (sin agrupar)",
                color_discrete_sequence=["gray"]
            )
        st.plotly_chart(fig_before, use_container_width=True)

        # --- Visualización después del clustering ---
        st.subheader(f"🎯 Datos agrupados con K-Means (k = {k})")
        if n_components == 2:
            fig_after = px.scatter(
                pca_df, x='PCA1', y='PCA2',
                color=pca_df['Cluster'].astype(str),
                title="Clusters visualizados en 2D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        else:
            fig_after = px.scatter_3d(
                pca_df, x='PCA1', y='PCA2', z='PCA3',
                color=pca_df['Cluster'].astype(str),
                title="Clusters visualizados en 3D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        st.plotly_chart(fig_after, use_container_width=True)

        # --- Centroides ---
        st.subheader("📍 Centroides de los clusters (en espacio PCA)")
        centroides_pca = pd.DataFrame(pca.transform(kmeans.cluster_centers_), columns=pca_cols)
        st.dataframe(centroides_pca)

        # --- Método del Codo ---
        st.subheader("📈 Método del Codo (Elbow Method)")
        if st.button("Calcular número óptimo de clusters"):
            inertias = []
            K = range(1, 11)
            for i in K:
                km = KMeans(
                    n_clusters=i,
                    init=init,
                    max_iter=int(max_iter),
                    n_init=n_init,
                    random_state=random_state
                )
                km.fit(X)
                inertias.append(km.inertia_)

            fig2, _ = plt.subplots(figsize=(8, 6))
            plt.plot(list(K), inertias, 'bo-')
            plt.title('Método del Codo')
            plt.xlabel('Número de Clusters (k)')
            plt.ylabel('Inercia (SSE)')
            plt.grid(True)
            st.pyplot(fig2)

        # --- Descarga de resultados ---
        st.subheader("💾 Descargar datos con clusters asignados")
        buffer = BytesIO()
        data.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="⬇️ Descargar CSV con Clusters",
            data=buffer,
            file_name="datos_clusterizados.csv",
            mime="text/csv"
        )

else:
    st.info("👉 Carga un archivo CSV en la barra lateral para comenzar.")
    st.write("""
    **Ejemplo de formato:**
    | Ingreso_Anual | Gasto_Tienda | Edad |
    |----------------|--------------|------|
    | 45000 | 350 | 28 |
    | 72000 | 680 | 35 |
    | 28000 | 210 | 22 |
    """)