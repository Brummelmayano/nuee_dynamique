import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import load_iris, load_wine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from nuees_dynamiques import (
    NuéesDynamique,
    compute_silhouette,
    compute_davies_bouldin,
    generate_synthetic_data,
)


st.set_page_config(page_title="Nuées Dynamiques - Interactive Demo", page_icon="🚀", layout="wide")

st.title("🚀 Nuées Dynamiques - Interactive Clustering Demo")
st.markdown(
    "Ce petit démonstrateur interactif permet d'explorer la méthode des Nuées Dynamiques en comparaison avec K-Means."
)

# Sidebar controls
st.sidebar.header("Configuration")

# Algorithm selection (controls conditional UI)
algorithm = st.sidebar.radio("Algorithme", ["Nuées Dynamiques", "K-Means", "Both"])

# Conditional parameter selectors based on algorithm choice
if algorithm == "K-Means":
    st.sidebar.info("K-Means: Distance euclidienne fixée + méthode centroïde")
    distance_metric = "euclidean"
    etallon_method = "centroid"
    show_nd_params = False
elif algorithm == "Nuées Dynamiques":
    distance_metric = st.sidebar.selectbox("Métrique de distance", ["euclidean", "manhattan", "minkowski", "chebyshev"])
    etallon_method = st.sidebar.selectbox("Méthode d'étalon", ["centroid", "medoid", "median", "mode"])
    show_nd_params = True
else:  # "Both"
    st.sidebar.subheader("Paramètres Nuées Dynamiques")
    distance_metric = st.sidebar.selectbox("Métrique de distance", ["euclidean", "manhattan", "minkowski", "chebyshev"])
    etallon_method = st.sidebar.selectbox("Méthode d'étalon", ["centroid", "medoid", "median", "mode"])
    show_nd_params = True

# Visualization dimension selector
viz_dim = st.sidebar.selectbox("Dimension de visualisation", ["2D", "3D"])

# Dataset selection
st.sidebar.subheader("Source de données")
dataset_name = st.sidebar.selectbox("Dataset prédéfini", ["Iris", "Wine", "Synthétique"])

# File upload for custom dataset
uploaded_file = st.sidebar.file_uploader("Ou charger un CSV personnalisé (features numériques uniquement)", type=["csv"])

# Number of clusters - with input option
st.sidebar.subheader("Nombre de clusters")
n_clusters_input = st.sidebar.radio("Sélectionner une option", ["Utiliser le slider", "Entrer manuellement"])

if n_clusters_input == "Utiliser le slider":
    n_clusters = st.sidebar.slider("Nombre de clusters", min_value=2, max_value=20, value=3)
else:
    n_clusters = st.sidebar.number_input("Nombre de clusters", min_value=2, max_value=100, value=3, step=1)

run_button = st.sidebar.button("Lancer le clustering")


def load_dataset(name: str, n_clusters: int):
    """Charge un dataset selon le nom et retourne X, y_true, info dict."""
    info = {}
    if name == "Iris":
        X, y = load_iris(return_X_y=True)
        info["source"] = "Iris (sklearn)"
    elif name == "Wine":
        X, y = load_wine(return_X_y=True)
        info["source"] = "Wine (sklearn)"
    else:
        # Synthétique: use helper from the package
        X, y = generate_synthetic_data(n_samples=300, n_features=5, n_clusters=n_clusters, random_state=42)
        info["source"] = "Synthétique (generate_synthetic_data)"

    info["n_samples"] = X.shape[0]
    info["n_features"] = X.shape[1]
    info["n_clusters_expected"] = len(np.unique(y)) if y is not None else n_clusters
    return X, y, info


def load_custom_csv(uploaded_file):
    """Charge un fichier CSV personnalisé et retourne X, y_true, info dict."""
    df = pd.read_csv(uploaded_file)
    X = df.select_dtypes(include=["number"]).values
    
    info = {}
    info["source"] = f"Personnalisé (uploadé: {uploaded_file.name})"
    info["n_samples"] = X.shape[0]
    info["n_features"] = X.shape[1]
    info["n_clusters_expected"] = "Défini par l'utilisateur"
    
    return X, None, info


def safe_metric(func, X, labels):
    try:
        return func(X, labels)
    except Exception:
        return None


def run_clustering(X, n_clusters, distance_metric, etallon_method, algorithm_name):
    results = {}

    if algorithm_name in ("Nuées Dynamiques", "Both"):
        try:
            nd = NuéesDynamique(
                data=X,
                n_clusters=n_clusters,
                distance_metric=distance_metric,
                init_method="kmeans++",
                etallon_method=etallon_method,
                max_iterations=100,
                tolerance=1e-4,
                random_state=42,
            )
            nd.fit()

            labels = nd.labels_
            centers = getattr(nd, "etallons_", None)
            inertia = None
            try:
                inertia = nd.get_inertia()
            except Exception:
                inertia = None

            silhouette = safe_metric(compute_silhouette, X, labels)
            davies_bouldin = safe_metric(compute_davies_bouldin, X, labels)

            results["Nuées Dynamiques"] = {
                "labels_": labels,
                "etallons_": centers,
                "n_iter_": getattr(nd, "n_iter_", None),
                "inertia": inertia,
                "silhouette": silhouette,
                "davies_bouldin": davies_bouldin,
            }
        except Exception as exc:
            results["Nuées Dynamiques"] = {"error": str(exc)}

    if algorithm_name in ("K-Means", "Both"):
        try:
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            km.fit(X)
            labels = km.labels_
            centers = km.cluster_centers_
            inertia = getattr(km, "inertia_", None)
            silhouette = safe_metric(compute_silhouette, X, labels)
            davies_bouldin = safe_metric(compute_davies_bouldin, X, labels)

            results["K-Means"] = {
                "labels_": labels,
                "etallons_": centers,
                "n_iter_": getattr(km, "n_iter_", None),
                "inertia": inertia,
                "silhouette": silhouette,
                "davies_bouldin": davies_bouldin,
            }
        except Exception as exc:
            results["K-Means"] = {"error": str(exc)}

    return results


def visualize_results(X, result: dict, algorithm_name: str, viz_dim: str):
    """Retourne une figure matplotlib avec la projection PCA 2D/3D et centres projetés."""
    n_components = 2 if viz_dim == "2D" else 3
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()

    # Déterminer le label pour les centres
    center_label = "Étalons" if algorithm_name == "Nuées Dynamiques" else "Centroïdes"

    if viz_dim == "2D":
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = result.get("labels_")
        if labels is None:
            ax.text(0.5, 0.5, "No labels to display", ha="center")
            return fig

        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.6, s=40)

        centers = result.get("etallons_")
        if centers is not None:
            try:
                centers_pca = pca.transform(centers)
                ax.scatter(
                    centers_pca[:, 0],
                    centers_pca[:, 1],
                    c="red",
                    marker="X",
                    s=200,
                    edgecolors="black",
                    linewidths=2,
                    label=center_label,
                )
            except Exception:
                pass

        cbar = fig.colorbar(scatter, ax=ax)
        ax.set_title(f"{algorithm_name} — PCA 2D (var: {explained:.1%})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

    else:  # 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        labels = result.get("labels_")
        if labels is None:
            ax.text(0.5, 0.5, 0.5, "No labels to display")
            return fig

        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap="viridis", alpha=0.6, s=40)

        centers = result.get("etallons_")
        if centers is not None:
            try:
                centers_pca = pca.transform(centers)
                ax.scatter(
                    centers_pca[:, 0],
                    centers_pca[:, 1],
                    centers_pca[:, 2],
                    c="red",
                    marker="X",
                    s=200,
                    edgecolors="black",
                    linewidths=2,
                    label=center_label,
                )
            except Exception:
                pass

        fig.colorbar(scatter, ax=ax, shrink=0.5)
        ax.set_title(f"{algorithm_name} — PCA 3D (var: {explained:.1%})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()

    return fig


if run_button:
    # Load dataset
    with st.spinner("Chargement du dataset..."):
        try:
            if uploaded_file is not None:
                X, y_true, info = load_custom_csv(uploaded_file)
                dataset_name = "Personnalisé"
            else:
                X, y_true, info = load_dataset(dataset_name, n_clusters)
        except Exception as exc:
            st.error(f"Erreur lors du chargement du dataset: {exc}")
            st.stop()

    st.subheader("Informations du dataset")
    st.write(info)

    # Run clustering
    with st.spinner("Exécution du clustering..."):
        results = run_clustering(X, n_clusters, distance_metric, etallon_method, algorithm)

    # Display results
    if algorithm == "Both":
        col1, col2 = st.columns(2)
        names = ["Nuées Dynamiques", "K-Means"]
        for col, name in zip((col1, col2), names):
            col.header(name)
            res = results.get(name, {})
            if "error" in res:
                col.error(res["error"])
                continue

            inertia = res.get("inertia")
            silhouette = res.get("silhouette")
            davies_bouldin = res.get("davies_bouldin")
            n_iter = res.get("n_iter_")

            m1, m2, m3 = col.columns(3)
            m1.metric("Inertia", f"{inertia:.2f}" if inertia is not None else "N/A")
            m2.metric("Silhouette", f"{silhouette:.3f}" if silhouette is not None else "N/A")
            m3.metric("Davies-Bouldin", f"{davies_bouldin:.3f}" if davies_bouldin is not None else "N/A")

            col.info(f"Itérations : {n_iter}")
            fig = visualize_results(X, res, name, viz_dim)
            col.pyplot(fig)

    else:
        name = algorithm
        res = results.get(name, {})
        if "error" in res:
            st.error(res["error"])
        else:
            st.subheader(name)
            inertia = res.get("inertia")
            silhouette = res.get("silhouette")
            davies_bouldin = res.get("davies_bouldin")
            n_iter = res.get("n_iter_")

            c1, c2, c3 = st.columns(3)
            c1.metric("Inertia", f"{inertia:.2f}" if inertia is not None else "N/A")
            c2.metric("Silhouette", f"{silhouette:.3f}" if silhouette is not None else "N/A")
            c3.metric("Davies-Bouldin", f"{davies_bouldin:.3f}" if davies_bouldin is not None else "N/A")

            st.info(f"Itérations : {n_iter}")
            fig = visualize_results(X, res, name, viz_dim)
            st.pyplot(fig)

    # Download dataset with labels
    st.subheader("📥 Télécharger les résultats")
    if algorithm == "Both":
        # For "Both", download ND results by default
        name = "Nuées Dynamiques"
        res = results.get(name, {})
        labels = res.get("labels_")
    else:
        res = results.get(algorithm, {})
        labels = res.get("labels_")
    
    if labels is not None:
        df_download = pd.DataFrame(X)
        df_download[f"cluster_{algorithm}"] = labels
        if y_true is not None:
            df_download["true_label"] = y_true
        
        csv_data = df_download.to_csv(index=False)
        st.download_button(
            label="📊 Télécharger le dataset avec les labels de clustering (CSV)",
            data=csv_data,
            file_name=f"dataset_clustered_{algorithm.replace(' ', '_')}.csv",
            mime="text/csv"
        )


st.markdown("---")
st.subheader("Aperçu du dataset")
if run_button and 'X' in locals() and 'y_true' in locals():
    df_preview = pd.DataFrame(X)
    if y_true is not None:
        df_preview["true_label"] = y_true
    st.dataframe(df_preview.head(20))



st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.9em;'>"
    "<p><strong>Nuées Dynamiques - Démonstrateur Clustering Interactif</strong></p>"
    "<p>Université de Kinshasa</p>"
    "<p>Réalisé par Duasenge Mayano Brummel</p>"
    "</div>",
    unsafe_allow_html=True
)
