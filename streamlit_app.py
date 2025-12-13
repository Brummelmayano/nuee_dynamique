import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px

from sklearn.datasets import load_iris, load_wine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from nuees_dynamiques import (
    NuéesDynamique,
    compute_silhouette,
    compute_davies_bouldin,
    generate_synthetic_data,
    compute_distance_matrix,
)

# Mapping entre libellés français descriptifs et valeurs internes
INIT_METHOD_LABELS = {
    "Centroïde unique (similaire KMeans)": "kmeans++",
    "Ensemble de points aléatoires": "random",
    "Distribution probabiliste (GMM)": "gmm",
    "Axes factoriels (ACP)": "pca"
}

def compute_pca_cached(_X: np.ndarray, n_components: int):
    """PCA for etalon evolution."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(_X)
    return X_pca, pca

st.set_page_config(page_title="Nuées Dynamiques - Interactive Demo", page_icon="🚀", layout="wide")

st.title("🚀 Nuées Dynamiques - Interactive Clustering Demo")
st.markdown(    
    "Ce petit démonstrateur interactif permet d'explorer la méthode des Nuées Dynamiques [Diday, E.(1971).](https://www.numdam.org/item/RSA_1971__19_2_19_0/)"
    )

st.markdown(
    """
    ### À propos des noyaux multi-étalons

    Le paramètre **« Nombre d'étalons par cluster »** contrôle le nombre de points représentatifs par cluster :
    - **1 étalon** : chaque cluster est représenté par un seul point central (centroïde, médoïde, etc.).
    - **Plusieurs étalons** : chaque cluster est représenté par un **noyau** de plusieurs points représentatifs, permettant de capturer des formes de clusters allongées ou irrégulières.

    Lorsque **plusieurs étalons** sont utilisés, les visualisations affichent tous les points du noyau sous forme de **marqueurs distinctifs**.

    Cette réalisation part du principe que l’on ne dispose pas d’indications sur la typologie probable, les étalons sont donc initialement positionnés au hasard.

    """
    )

with st.expander("ℹ️ Guide des méthodes d'initialisation"):
    st.markdown("""
    ### Méthodes d'initialisation disponibles

    | Méthode | Type | Avantages | Cas d'usage | Référence |
    |---------|------|-----------|-------------|-----------|
    | **Centroïde unique (similaire KMeans)** | Probabiliste | Convergence améliorée, 1 étalon/cluster | Clusters sphériques, comportement K-Means | Arthur & Vassilvitskii 2007 |
    | **Ensemble de points aléatoires** | Stochastique | Rapide, flexible (multi-étalons) | Exploration, formes complexes | Diday 1971 |
    | **Distribution probabiliste (GMM)** | Probabiliste | Capture gaussiennes, haute précision | Clusters elliptiques, chevauchements | EM algorithm |
    | **Axes factoriels (ACP)** | Déterministe | Exploite variance, reproductible | Données structurées, haute dimensionnalité | Su & Dy 2007 |

    **Recommandations scientifiques** :
    - **GMM** : Validé pour les clusters avec chevauchement et formes elliptiques (supérieur à K-means avec Jaccard 0.745 vs 0.652)
    - **ACP** : Méthode PCA-Part (Su & Dy, 2007) génère des clusters avec SSE proche du minimum global
    - **Centroïde unique** : Initialisation standard pour K-means, améliore la convergence
    - **Ensemble aléatoire** : Baseline simple, nécessite plusieurs exécutions pour robustesse

    **Distinction importante** :
    - **Initialisation** (`init_method`) : Choix des étalons au DÉBUT de l'algorithme
    - **Calcul** (`etallon_method`) : Mise à jour des étalons à CHAQUE ITÉRATION (centroïde, médoïde, médiane, mode)
    """)

# Sidebar controls
st.sidebar.header("⚙️ Paramètres du clustering")


# Algorithm selection (controls conditional UI)
algorithm = st.sidebar.radio("Algorithme", ["Nuées Dynamiques", "Comparer Nuées Dynamiques avec K-Means"])


# --- Dataset inputs (needed early to compute max_ni for the slider) ---
dataset_name = st.sidebar.selectbox("Dataset prédéfini", ["Iris", "Wine", "Synthétique"])
uploaded_file = st.sidebar.file_uploader("Importer un dataset (CSV)", type=["csv"], accept_multiple_files=False, help="Le fichier CSV doit contenir uniquement des données numériques.")


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


# Load a lightweight preview of the dataset early so we can compute a sensible max for the
# `n_etalons_per_cluster` slider (max_ni = n_samples // n_clusters). This allows validating
# the slider preventively and keeps backward compatibility when ni=1.
try:
    if uploaded_file is not None:
        X_preview, y_preview, info_preview = load_custom_csv(uploaded_file)
        # Reset file pointer for subsequent read in run_clustering block
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        dataset_name_preview = "Personnalisé"
    else:
        X_preview, y_preview, info_preview = load_dataset(dataset_name, 3)
except Exception:
    X_preview, y_preview, info_preview = None, None, None



# Number of clusters - with input option
st.sidebar.subheader("Paramètres de base")

n_clusters = st.sidebar.number_input("Nombre de clusters", min_value=2, max_value=100, value=3, step=1, help="Nombre de groupes à identifier dans les données")

st.sidebar.subheader("Méthode d'initialisation")
# Disable PCA if no preview available (to avoid invalid pca_n_components)
available_methods = ["random", "kmeans++", "gmm"]
if info_preview is not None:
    available_methods.append("pca")

# Create available labels based on available methods
available_labels = [label for label, method in INIT_METHOD_LABELS.items() if method in available_methods]

selected_label = st.sidebar.selectbox(
    "Méthode d'initialisation des étalons",
    available_labels,
    index=0,
    help="""
    **Méthode d'initialisation des étalons** : Détermine comment les étalons initiaux sont choisis au début de l'algorithme.

    • Centroïde unique (similaire KMeans) : Initialisation probabiliste avec kmeans++ (Arthur & Vassilvitskii 2007), force 1 étalon/cluster
    • Ensemble de points aléatoires : Sélection aléatoire de k ensembles de points (Diday 1971, baseline)
    • Distribution probabiliste (GMM) : Modèle de mélange gaussien, capture les structures elliptiques
    • Axes factoriels (ACP) : Projection PCA et sélection des points extrêmes (Su & Dy 2007, déterministe)

    ⚠️ À ne pas confondre avec la "Méthode de calcul de l'étalon" qui définit comment les étalons sont MIS À JOUR à chaque itération (centroïde, médoïde, etc.).
    """ + (" (ACP indisponible sans aperçu du dataset)" if "pca" not in available_methods else "")
)

# Map the selected label to the internal method
init_method = INIT_METHOD_LABELS[selected_label]

# Paramètre ni : nombre d'étalons par cluster (champ numérique dans la sidebar)
#noyaux multi-étalons
if selected_label == "Centroïde unique (similaire KMeans)":
    n_etalons_per_cluster = 1
    st.sidebar.info("Mode centroïde unique activé : chaque cluster est représenté par un seul point central (comme K-Means).")
else:
    if info_preview is not None:
        max_ni = max(1, info_preview.get("n_samples", 1) // max(1, 3))
        default_ni = min(30, max_ni)
    else:
        max_ni = 100
        default_ni = 30

    n_etalons_per_cluster = st.sidebar.number_input(
        "Étalons par cluster",
        min_value=1,
        max_value=max_ni,
        value=min(default_ni, max_ni),
        step=1,
        help="Nombre de points représentatifs par cluster. 1 étalon = forme simple, plusieurs étalons = formes complexes ou allongées"
    )

# Méthode d'étalon visible uniquement lorsque ni == 1
if n_etalons_per_cluster == 1:
    etallon_method = st.sidebar.selectbox(
        "Méthode de calcul de l'étalon",
        ["centroid", "medoid", "median", "mode"],
        help="Méthode de CALCUL de l'étalon : définit comment l'étalon est mis à jour à chaque itération (centroïde=moyenne, médoïde=point le plus central, etc.). Différent de l'initialisation."
    )
else:
    # Par défaut, centroid est utilisé pour composer les noyaux multi-étalons
    etallon_method = "centroid"

# Paramètres conditionnels pour GMM
gmm_init_mode = "means"  # Valeur par défaut
if init_method == "gmm":
    gmm_init_mode = st.sidebar.radio(
        "Mode d'initialisation GMM",
        ["means", "sample"],
        index=0,
        help="""
        Mode d'initialisation pour le modèle de mélange gaussien :
        • means : Utilise les moyennes des composantes gaussiennes (stable, déterministe)
        • sample : Échantillonne des points depuis les distributions gaussiennes (variable, stochastique)
        """
    )

# Paramètres conditionnels pour PCA
pca_n_components = None  # Valeur par défaut (sera égal à n_clusters)
if init_method == "pca" and info_preview is not None:
    # Calculer le nombre maximum de composantes (nombre de features)
    if info_preview is not None:
        max_components = info_preview.get("n_features", n_clusters)
    else:
        max_components = 10  # Valeur par défaut si pas de preview

    pca_n_components = st.sidebar.slider(
        "Nombre de composantes principales",
        min_value=1,
        max_value=max_components,
        value=min(n_clusters, max_components),
        step=1,
        help="""
        Nombre de composantes principales à utiliser pour l'initialisation PCA.
        Plus de composantes = plus de points extrêmes disponibles pour l'initialisation.
        Par défaut : égal au nombre de clusters.
        Recommandation : au moins n_clusters/2 pour éviter l'initialisation aléatoire de secours.
        """
    )

# Conditional parameter selectors based on algorithm choice
if algorithm == "Nuées Dynamiques":
    distance_metric = st.sidebar.selectbox("Métrique de distance", ["euclidean", "manhattan", "minkowski", "chebyshev", "chi2", "sebestyen"])
    show_nd_params = True
else:  # Comparer Nuées Dynamiques avec K-Means
    st.sidebar.info("Affiche les résultats ND et K-Means pour comparaison")
    distance_metric = st.sidebar.selectbox("Métrique de distance", ["euclidean", "manhattan", "minkowski", "chebyshev", "chi2", "sebestyen"])
    show_nd_params = True

# Visualization dimension selector
viz_dim = st.sidebar.selectbox("Dimension de visualisation", ["2D", "3D"])

show_etalon_evolution = st.sidebar.checkbox(
    "Afficher l'évolution des étalons",
    value=False,
    help="Visualise la position des étalons à chaque itération de l'algorithme."
)


# Run button remains at the end of the sidebar
run_button = st.sidebar.button("▶️ Lancer l'analyse", type="primary")

def safe_metric(func, X, labels):
    try:
        return func(X, labels)
    except Exception:
        return None


def run_clustering(X, n_clusters, distance_metric, etallon_method, algorithm_name,
                   n_etalons_per_cluster=1, init_method="random",
                   gmm_init_mode="means", pca_n_components=None):
    results = {}

    if algorithm_name in ("Nuées Dynamiques", "Comparer Nuées Dynamiques avec K-Means"):
        try:
            nd = NuéesDynamique(
                data=X,
                n_clusters=n_clusters,
                distance_metric=distance_metric,
                init_method=init_method,  # Utiliser le paramètre au lieu de "random"
                gmm_init_mode=gmm_init_mode,  # Nouveau paramètre
                pca_n_components=pca_n_components,  # Nouveau paramètre
                etallon_method=etallon_method,
                max_iterations=100,
                tolerance=1e-4,
                random_state=42,
                n_etalons_per_cluster=n_etalons_per_cluster,
            )
            nd.fit()

            # Après la ligne 199 (nd.fit())
            cluster_sizes = nd.get_cluster_sizes()
            homogeneite = nd.get_homogeneite_per_cluster()
            S_total = nd.get_S_total()
            rep_noyaux = nd.get_representative_noyaux()

            labels = nd.labels_
            centers = getattr(nd, "etallons_", None)
            inertia = None
            try:
                inertia = nd.get_inertia()
            except Exception:
                inertia = None

            silhouette = safe_metric(lambda X_arg, labels_arg: compute_silhouette(X_arg, labels_arg, metric=distance_metric), X, labels)
            davies_bouldin = safe_metric(compute_davies_bouldin, X, labels)

            results["Nuées Dynamiques"] = {
                "labels_": labels,
                "etallons_": centers,
                "n_iter_": getattr(nd, "n_iter_", None),
                "inertia": inertia,
                "silhouette": silhouette,
                "davies_bouldin": davies_bouldin,
                "nd": nd,
                # Nouvelles métriques de l'article
                "cluster_sizes": cluster_sizes,
                "homogeneite_per_cluster": homogeneite,
                "S_total": S_total,
                "representative_noyaux": rep_noyaux,
            }
        except Exception as exc:
            results["Nuées Dynamiques"] = {"error": str(exc)}

    if algorithm_name in ("Comparer Nuées Dynamiques avec K-Means",):
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


def visualize_etalon_evolution(X_pca, pca, etallons_at_iter, labels, iteration, viz_dim):
    """Visualise les étalons à une itération donnée."""
    explained = pca.explained_variance_ratio_.sum()

    if etallons_at_iter is None:
        st.warning(f"⚠️ Étalons à l'itération {iteration} : Aucun historique disponible pour cette itération.")
        return None

    # Vérifier la forme des étalons
    expected_features = pca.n_features_in_
    if etallons_at_iter.ndim == 2:
        if etallons_at_iter.shape[1] != expected_features:
            st.warning(f"⚠️ Forme inattendue à l'itération {iteration} : {etallons_at_iter.shape}, attendu (n_clusters, {expected_features})")
    elif etallons_at_iter.ndim == 3:
        if etallons_at_iter.shape[2] != expected_features:
            st.warning(f"⚠️ Forme inattendue à l'itération {iteration} : {etallons_at_iter.shape}, attendu (n_clusters, ni, {expected_features})")
    else:
        st.warning(f"⚠️ Dimension inattendue à l'itération {iteration} : {etallons_at_iter.ndim}D (attendu 2D ou 3D)")

    # Debug info (optionnel, à commenter après résolution) :
    # st.write(f"🔍 Debug iter {iteration} : shape={etallons_at_iter.shape}, ndim={etallons_at_iter.ndim}")

    center_label = "Étalons"

    if viz_dim == "2D":
        fig, ax = plt.subplots(figsize=(10, 6))
        if labels is None:
            ax.text(0.5, 0.5, "No labels to display", ha="center")
            return fig

        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.6, s=40)

        if etallons_at_iter is not None:
            try:
                if getattr(etallons_at_iter, "ndim", 2) == 2:
                    centers_pca = pca.transform(etallons_at_iter)
                    n_clusters_viz = etallons_at_iter.shape[0]
                    for k in range(n_clusters_viz):
                        color = plt.cm.viridis(k / max(1, n_clusters_viz - 1))
                        darker_color = tuple(c * 0.7 for c in color[:3])
                        ax.scatter(
                            centers_pca[k, 0],
                            centers_pca[k, 1],
                            color=darker_color,
                            marker="X",
                            s=200,
                            edgecolors="black",
                            linewidths=2,
                            label=center_label if k == 0 else "",
                        )
                elif getattr(etallons_at_iter, "ndim", 2) == 3:
                    n_clusters_viz, ni, n_features = etallons_at_iter.shape
                    flat_centers = etallons_at_iter.reshape(-1, n_features)
                    centers_pca = pca.transform(flat_centers)
                    for k in range(n_clusters_viz):
                        start = k * ni
                        end = (k + 1) * ni
                        color = plt.cm.viridis(k / max(1, n_clusters_viz - 1))
                        darker_color = tuple(c * 0.7 for c in color[:3])
                        ax.scatter(
                            centers_pca[start:end, 0],
                            centers_pca[start:end, 1],
                            color=darker_color,
                            marker="*",
                            s=100,
                            edgecolors="black",
                            linewidths=1,
                            label=f"{center_label} (noyaux, ni={ni})" if k == 0 else "",
                            alpha=0.8,
                        )
            except Exception as e:
                st.error(f"⚠️ Erreur affichage étalons 2D (iter {iteration}) : {e}")

        cbar = fig.colorbar(scatter, ax=ax)
        ax.set_title(f"Évolution des étalons — Itération {iteration} — PCA 2D (var: {explained:.1%})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

    else:  # 3D
        fig = go.Figure()

        if labels is None:
            fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='text', text=["No labels to display"]))
            return fig

        # Points de données
        fig.add_trace(go.Scatter3d(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            z=X_pca[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=labels,
                colorscale='Viridis',
                opacity=0.6,
                showscale=True,
                colorbar=dict(
                    title="Cluster",
                    x=1.1,  # Position à droite de la figure
                    y=0.5,  # Centré verticalement
                    len=0.6  # Longueur réduite pour éviter le chevauchement
                )
            ),
            name="Données"
        ))

        if etallons_at_iter is not None:
            try:
                if getattr(etallons_at_iter, "ndim", 2) == 2:
                    centers_pca = pca.transform(etallons_at_iter)
                    n_clusters_viz = etallons_at_iter.shape[0]
                    colors = []
                    for k in range(n_clusters_viz):
                        color = plt.cm.viridis(k / max(1, n_clusters_viz - 1))
                        darker = tuple(int(c * 0.7 * 255) for c in color[:3])
                        colors.append(f'rgb{darker}')
                    fig.add_trace(go.Scatter3d(
                        x=centers_pca[:, 0],
                        y=centers_pca[:, 1],
                        z=centers_pca[:, 2],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=colors,
                            symbol='x',
                            line=dict(width=2, color='black')
                        ),
                        name=center_label
                    ))
                elif getattr(etallons_at_iter, "ndim", 2) == 3:
                    n_clusters_viz, ni, n_features = etallons_at_iter.shape
                    flat_centers = etallons_at_iter.reshape(-1, n_features)
                    centers_pca = pca.transform(flat_centers)
                    colors = []
                    for k in range(n_clusters_viz):
                        color = plt.cm.viridis(k / max(1, n_clusters_viz - 1))
                        darker = tuple(int(c * 0.7 * 255) for c in color[:3])
                        for _ in range(ni):
                            colors.append(f'rgb{darker}')
                    fig.add_trace(go.Scatter3d(
                        x=centers_pca[:, 0],
                        y=centers_pca[:, 1],
                        z=centers_pca[:, 2],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=colors,
                            symbol='diamond',
                            line=dict(width=1, color='black'),
                            opacity=0.8
                        ),
                        name=f"{center_label} (noyaux, ni={ni})"
                    ))
            except Exception as e:
                st.error(f"⚠️ Erreur affichage étalons 3D (iter {iteration}) : {e}")

        fig.update_layout(
            title=f"Évolution des étalons — Itération {iteration} — PCA 3D (var: {explained:.1%})",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3"
            ),
            showlegend=True
        )

    return fig


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
            if centers.size == 0:
                st.info(f"ℹ️ Aucun centre à afficher pour {algorithm_name}")
                centers = None  # Skip affichage
            else:
                try:
                    # Support both historical 2D etallons_ (n_clusters, n_features)
                    # and multi-noyaux 3D etallons_ (n_clusters, ni, n_features)
                    if getattr(centers, "ndim", 2) == 2:
                        centers_pca = pca.transform(centers)
                        n_clusters_viz = centers.shape[0]
                        for k in range(n_clusters_viz):
                            color = plt.cm.viridis(k / max(1, n_clusters_viz - 1))
                            darker_color = tuple(c * 0.7 for c in color[:3])
                            ax.scatter(
                                centers_pca[k, 0],
                                centers_pca[k, 1],
                                color=darker_color,
                                marker="X",
                                s=200,
                                edgecolors="black",
                                linewidths=2,
                                label=center_label if k == 0 else "",
                            )
                    elif getattr(centers, "ndim", 2) == 3:
                        n_clusters_viz, ni, n_features = centers.shape
                        flat_centers = centers.reshape(-1, n_features)
                        centers_pca = pca.transform(flat_centers)
                        for k in range(n_clusters_viz):
                            start = k * ni
                            end = (k + 1) * ni
                            color = plt.cm.viridis(k / max(1, n_clusters_viz - 1))
                            darker_color = tuple(c * 0.7 for c in color[:3])
                            ax.scatter(
                                centers_pca[start:end, 0],
                                centers_pca[start:end, 1],
                                color=darker_color,
                                marker="*",
                                s=100,
                                edgecolors="black",
                                linewidths=1,
                                label=f"{center_label} (noyaux, ni={ni})" if k == 0 else "",
                                alpha=0.8,
                            )
                except Exception as e:
                    st.warning(f"⚠️ Erreur affichage centres 2D ({algorithm_name}) : {e}")

        cbar = fig.colorbar(scatter, ax=ax)
        ax.set_title(f"{algorithm_name} — PCA 2D (var: {explained:.1%})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

    else:  # 3D
        fig = go.Figure()

        labels = result.get("labels_")
        if labels is None:
            fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='text', text=["No labels to display"]))
            return fig

        # Points de données
        fig.add_trace(go.Scatter3d(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            z=X_pca[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=labels,
                colorscale='Viridis',
                opacity=0.6,
                showscale=True,
                colorbar=dict(
                    title="Cluster",
                    x=1.1,  # Position à droite de la figure
                    y=0.5,  # Centré verticalement
                    len=0.6  # Longueur réduite pour éviter le chevauchement
                )
            ),
            name="Données"
        ))

        centers = result.get("etallons_")
        if centers is not None:
            if centers.size == 0:
                st.info(f"ℹ️ Aucun centre à afficher pour {algorithm_name}")
                centers = None  # Skip affichage
            else:
                try:
                    if getattr(centers, "ndim", 2) == 2:
                        centers_pca = pca.transform(centers)
                        n_clusters_viz = centers.shape[0]
                        colors = []
                        for k in range(n_clusters_viz):
                            color = plt.cm.viridis(k / max(1, n_clusters_viz - 1))
                            darker = tuple(int(c * 0.7 * 255) for c in color[:3])
                            colors.append(f'rgb{darker}')
                        fig.add_trace(go.Scatter3d(
                            x=centers_pca[:, 0],
                            y=centers_pca[:, 1],
                            z=centers_pca[:, 2],
                            mode='markers',
                            marker=dict(
                                size=10,
                                color=colors,
                                symbol='x',
                                line=dict(width=2, color='black'),
                                showscale=False  # Désactiver la colorbar pour les étalons
                            ),
                            name=center_label
                        ))
                    elif getattr(centers, "ndim", 2) == 3:
                        n_clusters_viz, ni, n_features = centers.shape
                        flat_centers = centers.reshape(-1, n_features)
                        centers_pca = pca.transform(flat_centers)
                        colors = []
                        for k in range(n_clusters_viz):
                            color = plt.cm.viridis(k / max(1, n_clusters_viz - 1))
                            darker = tuple(int(c * 0.7 * 255) for c in color[:3])
                            for _ in range(ni):
                                colors.append(f'rgb{darker}')
                        fig.add_trace(go.Scatter3d(
                            x=centers_pca[:, 0],
                            y=centers_pca[:, 1],
                            z=centers_pca[:, 2],
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=colors,
                                symbol='diamond',
                                line=dict(width=1, color='black'),
                                opacity=0.8
                            ),
                            name=f"{center_label} (noyaux, ni={ni})"
                        ))
                except Exception as e:
                    st.warning(f"⚠️ Erreur affichage centres 3D ({algorithm_name}) : {e}")

        fig.update_layout(
            title=f"{algorithm_name} — PCA 3D (var: {explained:.1%})",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3"
            ),
            showlegend=True
        )

    return fig


def display_article_metrics(X, res, n_clusters, distance_metric, silhouette=None, davies_bouldin=None):
    """Affiche les métriques conformes à l'article pour Nuées Dynamiques."""
    cluster_sizes = res.get("cluster_sizes")
    homogeneite = res.get("homogeneite_per_cluster")
    S_total = res.get("S_total")
    rep_noyaux = res.get("representative_noyaux")
    inertia = res.get("inertia")
    
    if cluster_sizes is None or homogeneite is None or S_total is None or rep_noyaux is None:
        st.info("ℹ️ Métriques de l'article non disponibles (modèle non entraîné ou erreur).")
        return
    
    st.subheader('📊 Métriques de qualité du clustering (nuées dynamiques)')
    
    k = len(cluster_sizes)

    # Vérifier cohérence avec les autres métriques
    if len(homogeneite) != k:
        st.warning(
            f"⚠️ Incohérence des longueurs : cluster_sizes={k}, "
            f"homogeneite={len(homogeneite)}. "
            "Troncature aux longueurs communes."
        )
        min_len = min(k, len(homogeneite))
        k = min_len
        cluster_sizes = cluster_sizes[:min_len]
        homogeneite = homogeneite[:min_len]

    # Récupérer l'instance nd pour accéder aux étalons complets
    nd = res.get("nd")
    if nd is None or not hasattr(nd, 'etallons_'):
        st.warning("⚠️ Instance NuéesDynamique non disponible pour afficher les étalons.")
        return

    etallons = nd.etallons_

    # Étape supplémentaire pour aligner k avec le nombre d'étalons disponibles
    n_clusters_etallons = etallons.shape[0]
    if n_clusters_etallons < k:
        k = n_clusters_etallons
        cluster_sizes = cluster_sizes[:k]
        homogeneite = homogeneite[:k]
        st.warning(f"⚠️ Nombre d'étalons effectifs ({n_clusters_etallons}) inférieur à k ({k}). Troncature appliquée.")

    # Créer une colonne résumé pour les étalons
    etalons_summary = []
    for i in range(k):
        if etallons.ndim == 2:  # ni=1, shape (n_clusters, n_features)
            n_etalons = 1
            coord_sample = etallons[i][:2] if etallons.shape[1] >= 2 else etallons[i]
        else:  # ni>1, shape (n_clusters, ni, n_features)
            n_etalons = etallons.shape[1]
            coord_sample = etallons[i, 0, :2] if etallons.shape[2] >= 2 else etallons[i, 0]

        # Gérer les cas selon la longueur de coord_sample
        coord_len = len(coord_sample)
        if coord_len >= 2:
            summary = f"{n_etalons} pt(s) | Ex: [{coord_sample[0]:.2f}, {coord_sample[1]:.2f}...]"
        elif coord_len == 1:
            summary = f"{n_etalons} pt(s) | Ex: [{coord_sample[0]:.2f}...]"
        else:
            summary = f"{n_etalons} pt(s) | Ex: n/a"
        etalons_summary.append(summary)

    df_article = pd.DataFrame({
        'Cluster': range(1, k+1),  # Numérotation 1-based
        'Taille (nombre de points)': cluster_sizes,
        "L'homogénéité de chacune des classes obtenues.": [f'{s:.3f}' for s in homogeneite],  # Précision à 3 décimales
        'Étalons du noyau': etalons_summary
    })
    st.table(df_article)
    st.caption("**L'homogénéité** : mesure la compacité des points autour des étalons (valeur faible = cluster compact)")


    # Afficher toutes les métriques sur 4 colonnes
    st.markdown("**Indicateurs de performance :**")
    col1, col3, col4 = st.columns(3)
    col1.metric("La valeur de la partition obtenue.", f'{S_total:.3f}', help='Somme des dispersions intra-cluster (valeur faible = clustering compact)')

    col3.metric('Score de Silhouette', f'{silhouette:.3f}' if silhouette is not None else 'N/A', help='Cohésion interne et séparation des clusters (proche de 1 = excellent)')
    col4.metric('Indice Davies-Bouldin', f'{davies_bouldin:.3f}' if davies_bouldin is not None else 'N/A', help='Compacité relative et séparation (valeur faible = bon clustering)')
    st.caption('**Interprétation** : Valeur de la partition (Inertie) faible  + Silhouette élevée + Davies-Bouldin faible = clustering optimal')
    


    # Ajouter des expanders pour afficher les étalons complets par classe
    st.markdown("**Détail des étalons par cluster :**")
    for i in range(k):
        if etallons.ndim == 2:  # ni=1
            etalons_i = etallons[i].reshape(1, -1)  # Reshape en (1, n_features) pour cohérence
        else:  # ni>1
            etalons_i = etallons[i]  # Shape (ni, n_features)
        
        with st.expander(f"Cluster {i+1} : Détails du noyau ({etalons_i.shape[0]} étalons, {cluster_sizes[i]} points)"):
            df_etalons = pd.DataFrame(
                etalons_i,
                columns=[f'Dimension {j+1}' for j in range(etalons_i.shape[1])]
            )
            df_etalons.index = [f'Étalon {j+1}' for j in range(etalons_i.shape[0])]
            st.dataframe(df_etalons.style.format("{:.4f}"))
            st.caption("Étalons : points représentatifs centraux du cluster")
    
    # Expander optionnel pour la matrice R(x,i,L) sample
    with st.expander("Le degré de similarité de chaque individu à chaque classe."):
        st.markdown(
            "**Distance minimale** de chaque observation aux étalons de chaque cluster. "
            "Pour chaque point, on calcule la distance à tous les étalons du noyau d'un cluster, "
            "puis on retient la plus petite distance."
        )
        
        nd = res.get("nd")
        if nd is None or not hasattr(nd, 'etallons_'):
            st.warning("⚠️ Instance NuéesDynamique non disponible pour calculer R.")
            return
        
        n_samples_display = min(10, X.shape[0])

        # Validation de cohérence entre n_clusters paramètre et n_clusters du modèle
        n_clusters_model = getattr(nd.etallons_, 'shape', (0,))[0]
        if n_clusters_model != n_clusters:
            n_clusters_eff = min(n_clusters, n_clusters_model)
            st.warning(f"⚠️ Incohérence entre n_clusters paramètre ({n_clusters}) et n_clusters du modèle ({n_clusters_model}). Utilisation de {n_clusters_eff} clusters effectifs (min({n_clusters}, {n_clusters_model})).")
        else:
            n_clusters_eff = n_clusters

        if n_clusters_eff == 0:
            st.warning("Aucun cluster valide pour matrice R.")
            return

        R_sample = np.zeros((n_samples_display, n_clusters_eff))

        for i in range(n_clusters_eff):
            noyau_i = nd.etallons_[i]  # shape: (ni, n_features) ou (n_features,) si ni=1
            # Gérer le cas où noyau_i est 1D (ni=1)
            if noyau_i.ndim == 1:
                noyau_i = noyau_i.reshape(1, -1)
            
            # Calculer distances entre les 10 premiers points et le noyau i
            dists_i = compute_distance_matrix(X[:n_samples_display], noyau_i, distance_metric)
            # Prendre le minimum sur l'axe des étalons (axis=1)
            R_sample[:, i] = np.min(dists_i, axis=1)
        
        df_R = pd.DataFrame(R_sample, columns=[f'Distance au cluster {i+1}' for i in range(n_clusters_eff)])
        df_R.index = [f'Observation {i+1}' for i in range(n_samples_display)]
        st.dataframe(df_R.style.format("{:.4f}"))


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

    # Check for invalid dataset
    if X.size == 0:
        st.error("Le dataset ne contient pas de données numériques valides.")
        st.stop()
    if np.any(np.isnan(X)):
        st.error("Le dataset contient des valeurs NaN. Veuillez nettoyer les données.")
        st.stop()

    # Avertissement pour chi2 avec valeurs négatives
    if distance_metric == "chi2" and np.any(X < 0):
        st.warning(
            "⚠️ **Distance χ² sensible aux valeurs négatives** : Votre dataset contient des valeurs négatives. "
            "La distance χ² est optimale pour des données non-négatives (ex: histogrammes, fréquences). "
            "Considérez normaliser vos données (min-max scaling vers [0,1]) ou choisir une autre métrique."
        )

    # Run clustering
    with st.spinner("Exécution du clustering..."):
        results = run_clustering(
            X, n_clusters, distance_metric, etallon_method, algorithm,
            n_etalons_per_cluster=n_etalons_per_cluster,
            init_method=init_method,  # Nouveau paramètre
            gmm_init_mode=gmm_init_mode,  # Nouveau paramètre
            pca_n_components=pca_n_components  # Nouveau paramètre
        )

    # Display results
    if algorithm == "Comparer Nuées Dynamiques avec K-Means":
        st.info(
            "⚠️ **Note sur les métriques d'évaluation** : Le score de Silhouette pour Nuées Dynamiques "
            "utilise la métrique de distance sélectionnée. Cependant, K-Means et l'indice Davies-Bouldin "
            "utilisent toujours la distance Euclidienne (limitation de scikit-learn)."
        )
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

            col.info(f"Itérations : {n_iter}")
            fig = visualize_results(X, res, name, viz_dim)
            if viz_dim == "3D":
                col.plotly_chart(fig, use_container_width=True)
            else:
                col.pyplot(fig)


            # Afficher les métriques uniquement pour K-Means (Nuées Dynamiques les affiche dans display_article_metrics)
            if name == "K-Means":
                m1, m2, m3 = col.columns(3)
                m1.metric("Inertia", f"{inertia:.2f}" if inertia is not None else "N/A")
                m2.metric("Silhouette", f"{silhouette:.3f}" if silhouette is not None else "N/A")
                m3.metric("Davies-Bouldin", f"{davies_bouldin:.3f}" if davies_bouldin is not None else "N/A")

            # Après l'affichage de la visualisation (ligne 653)
            if name == "Nuées Dynamiques":
                display_article_metrics(X, res, n_clusters, distance_metric, silhouette, davies_bouldin)

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

            # Les métriques seront affichées dans display_article_metrics pour Nuées Dynamiques
            # (pas d'affichage ici pour éviter la duplication)

            st.info(f"Itérations : {n_iter}")
            fig = visualize_results(X, res, name, viz_dim)
            if viz_dim == "3D":
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.pyplot(fig)
            # Après la ligne 677 (st.pyplot(fig))
            if name == "Nuées Dynamiques":
                display_article_metrics(X, res, n_clusters, distance_metric, silhouette, davies_bouldin)

    # Persister les résultats pour accès hors du bloc run_button
    st.session_state['clustering_results'] = {
        'results': results,
        'X': X,
        'y_true': y_true,
        'viz_dim': viz_dim,
        'n_clusters': n_clusters,
        'distance_metric': distance_metric,
        'dataset_name': dataset_name,
        'algorithm': algorithm
    }

# Section évolution des étalons (persistante, hors du bloc run_button)
if 'clustering_results' in st.session_state and show_etalon_evolution:
    data = st.session_state['clustering_results']
    results = data['results']
    X = data['X']
    viz_dim = data['viz_dim']
    st.caption("ℹ️ La dimension de visualisation est liée au dernier run de clustering.")
    
    if "Nuées Dynamiques" in results:
        nd = results.get("Nuées Dynamiques", {}).get("nd")
        if nd and hasattr(nd, 'etallons_history_') and len(nd.etallons_history_) > 1:
            st.info(f"📊 Historique disponible : {len(nd.etallons_history_)} états (itérations 0 à {len(nd.etallons_history_)-1})")
            
            # Vérifier cohérence avec n_iter_
            if nd.n_iter_ is not None and isinstance(nd.n_iter_, int):
                expected_len = nd.n_iter_ + 1
                if len(nd.etallons_history_) != expected_len:
                    st.warning(f"⚠️ Incohérence : historique a {len(nd.etallons_history_)} états, attendu {expected_len} (n_iter_+1)")
            else:
                st.info("ℹ️ n_iter_ non disponible, vérification de cohérence ignorée.")
            
            # PCA computation (no caching to handle varying shapes)
            n_components = 2 if viz_dim == "2D" else 3
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)
            
            st.subheader("📈 Évolution des étalons à travers les itérations")
            iteration = st.slider(
                "Itération de l'algorithme",
                min_value=0,
                max_value=len(nd.etallons_history_) - 1,
                value=0,
                step=1,
                key="etalon_iteration_slider"  # Clé stable
            )
            
            if iteration >= len(nd.etallons_history_):
                st.error(f"❌ Index {iteration} hors limites (historique : {len(nd.etallons_history_)} états)")
            else:
                etallons_at_iter = nd.etallons_history_[iteration]
                labels = results["Nuées Dynamiques"]["labels_"]
                fig = visualize_etalon_evolution(X_pca, pca, etallons_at_iter, labels, iteration, viz_dim)
                
                if fig is None:
                    st.info("ℹ️ Aucune figure à afficher pour cette itération.")
                else:
                    if viz_dim == "3D":
                        st.plotly_chart(fig, use_container_width=True, key="etalon_evolution")  # Clé stable sans {iteration}
                    else:
                        st.pyplot(fig)
        else:
            st.info("L'historique des étalons n'est pas disponible ou l'algorithme a convergé immédiatement.")


# Utiliser session_state si disponible (après clustering), sinon skip
if 'clustering_results' in st.session_state:
    # Download dataset with labels
    st.subheader("📥 Télécharger les résultats")
    data = st.session_state['clustering_results']
    results = data['results']
    X = data['X']
    y_true = data['y_true']
    algorithm = data['algorithm']
    distance_metric = data['distance_metric']
    
    if algorithm == "Comparer Nuées Dynamiques avec K-Means":
        # For the compare option, download ND results by default
        name = "Nuées Dynamiques"
        res = results.get(name, {})
        labels = res.get("labels_")
    else:
        res = results.get(algorithm, {})
        labels = res.get("labels_")
    
    if labels is not None:
        df_download = pd.DataFrame(X)
        df_download[f"cluster_{algorithm}"] = labels
        # If the Nuées Dynamiques result contains multi-noyaux (3D etallons_), compute the
        # index (0..ni-1) of the closest point within the assigned kernel for each sample.
        res_et = res.get("etallons_")
        if res_et is not None and getattr(res_et, "ndim", 2) == 3:
            n_samples_dl = X.shape[0]
            n_clusters_dl, ni_dl, n_features_dl = res_et.shape
            closest_indices = np.zeros(n_samples_dl, dtype=int)
            # Vectorized computation: group samples by cluster_id, compute distances once per cluster
            unique_clusters = np.unique(labels)
            for cluster_id in unique_clusters:
                cluster_mask = labels == cluster_id
                X_cluster = X[cluster_mask]  # samples in this cluster
                noyau = res_et[int(cluster_id)]  # shape (ni, n_features)
                # compute distances between cluster samples and noyau points
                dists = compute_distance_matrix(X_cluster, noyau, metric=distance_metric)
                # find argmin for each sample within the noyau
                closest_per_sample = np.argmin(dists, axis=1)
                closest_indices[cluster_mask] = closest_per_sample
            df_download["closest_noyau_idx"] = closest_indices

        if y_true is not None:
            df_download["true_label"] = y_true

        csv_data = df_download.to_csv(index=False)
        st.download_button(
            label="📊 Télécharger le dataset avec les labels de clustering (CSV)",
            data=csv_data,
            file_name=f"dataset_clustered_{algorithm.replace(' ', '_')}.csv",
            mime="text/csv",
        )


if 'clustering_results' in st.session_state:
    st.markdown("---")
    st.subheader("Aperçu du dataset")
    data = st.session_state['clustering_results']
    X = data['X']
    y_true = data['y_true']
    df_preview = pd.DataFrame(X)
    if y_true is not None:
        df_preview["true_label"] = y_true
    st.dataframe(df_preview.head(20))



st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.9em;'>"
    "<p><strong>Nuées Dynamiques - Démonstrateur Clustering Interactif</strong></p>"
    "<p>Université de Kinshasa</p>"
    "<p>Réalisé par Brummel Duasenge Mayano</p>"
    "</div>",
    unsafe_allow_html=True
)
