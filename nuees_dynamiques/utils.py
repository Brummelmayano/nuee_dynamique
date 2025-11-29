"""
Module utilitaire pour visualisation, évaluation et génération de données
pour la méthode des nuées dynamiques.

Ce module fournit des fonctions pour :
- Générer des données synthétiques (blobs) pour le test d'algorithmes
- Visualiser les clusters en 2D et 3D
- Évaluer la qualité du clustering via silhouette et Davies-Bouldin
"""

import numpy as np

# Imports optionnels avec gestion d'erreur
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def generate_synthetic_data(
    n_samples: int,
    n_features: int = 2,
    n_clusters: int = 3,
    cluster_std: float = 1.0,
    random_state: int = None,
):
    """Génère des données synthétiques pour tester le clustering.

    Utilise `sklearn.datasets.make_blobs` pour créer des clusters bien séparés
    avec une distribution gaussienne.

    Parameters
    ----------
    n_samples : int
        Nombre total d'échantillons à générer.
    n_features : int, optional
        Nombre de caractéristiques (défaut=2).
    n_clusters : int, optional
        Nombre de clusters à générer (défaut=3).
    cluster_std : float, optional
        Écart-type de chaque cluster (défaut=1.0).
    random_state : int, optional
        Graine pour la reproductibilité (défaut=None).

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Données générées.
    y_true : ndarray, shape (n_samples,)
        Labels vrais des clusters (de 0 à n_clusters-1).

    Raises
    ------
    ImportError
        Si scikit-learn n'est pas installé.

    Examples
    --------
    >>> X, y = generate_synthetic_data(300, n_features=2, n_clusters=3)
    >>> X.shape
    (300, 2)
    >>> np.unique(y)
    array([0, 1, 2])

    Notes
    -----
    Les données générées suivent une distribution gaussienne pour chaque cluster,
    ce qui peut simplifier le problème de clustering comparé à des données réelles.
    """
    if not _HAS_SKLEARN:
        raise ImportError(
            "Le module `generate_synthetic_data` nécessite scikit-learn. "
            "Installez-le via : pip install scikit-learn"
        )

    # Générer les blobs en utilisant make_blobs
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return X, y


def plot_clusters(
    X,
    labels,
    centers=None,
    title: str = "Clusters",
    figsize: tuple = (10, 8),
    show_centers: bool = True,
):
    """Visualise les clusters en 2D ou 3D.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Données à visualiser. Doit avoir 2 ou 3 features.
    labels : ndarray, shape (n_samples,)
        Assignations de cluster pour chaque point.
    centers : ndarray, shape (n_clusters, n_features), optional
        Positions des étalons/centres (optionnel).
    title : str, optional
        Titre de la figure (défaut='Clusters').
    figsize : tuple, optional
        Taille de la figure en pouces (largeur, hauteur) (défaut=(10, 8)).
    show_centers : bool, optional
        Afficher les centres/étalons si True (défaut=True).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Objet figure matplotlib.

    Raises
    ------
    ImportError
        Si matplotlib n'est pas installé.
    ValueError
        Si X a plus de 3 features (visualisation non supportée).

    Notes
    -----
    - Pour 2 features : visualisation 2D via scatter plot.
    - Pour 3 features : visualisation 3D via scatter 3D.
    - Pour > 3 features : levée d'une ValueError (considérer une réduction dimensionnelle via PCA).
    - Les centres sont affichés avec des marqueurs 'X' rouges.

    Examples
    --------
    >>> from nuees_dynamiques import NuéesDynamique, generate_synthetic_data
    >>> X, y = generate_synthetic_data(100, n_features=2, n_clusters=3)
    >>> nd = NuéesDynamique(X, n_clusters=3)
    >>> nd.fit()
    >>> fig = plot_clusters(X, nd.labels_, nd.etallons_)
    >>> plt.show()
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "Le module `plot_clusters` nécessite matplotlib. "
            "Installez-le via : pip install matplotlib"
        )

    X = np.asarray(X)
    labels = np.asarray(labels)

    # Validation de la dimension de X
    if X.ndim != 2:
        raise ValueError(
            f"`X` doit être un tableau 2D de forme (n_samples, n_features), "
            f"mais a {X.ndim} dimension(s)"
        )

    # Validation de la correspondance entre X et labels
    if len(labels) != X.shape[0]:
        raise ValueError(
            f"La longueur de `labels` ({len(labels)}) ne correspond pas au nombre "
            f"de lignes dans `X` ({X.shape[0]})"
        )

    # Validation des dimensions
    n_features = X.shape[1]
    if n_features > 3:
        raise ValueError(
            f"Visualisation supportée seulement pour 2 ou 3 features, "
            f"mais X a {n_features} features. "
            f"Considérez une réduction dimensionnelle (ex: PCA) avant visualisation."
        )

    # Créer la figure
    fig = plt.figure(figsize=figsize)

    if n_features == 2:
        # Visualisation 2D
        ax = fig.add_subplot(111)
        # Scatter plot des points colorés par cluster
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.6, s=50)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, alpha=0.3)
        # Afficher les centres si fournis
        if centers is not None and show_centers:
            centers = np.asarray(centers)
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                c="red",
                marker="X",
                s=200,
                edgecolors="black",
                linewidths=2,
                label="Étalons",
            )
            ax.legend()

    elif n_features == 3:
        # Visualisation 3D
        ax = fig.add_subplot(111, projection="3d")
        # Scatter plot 3D
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap="viridis", alpha=0.6, s=50)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        ax.grid(True, alpha=0.3)
        # Afficher les centres si fournis
        if centers is not None and show_centers:
            centers = np.asarray(centers)
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                centers[:, 2],
                c="red",
                marker="X",
                s=200,
                edgecolors="black",
                linewidths=2,
                label="Étalons",
            )
            ax.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig


def compute_silhouette(X, labels, metric: str = "euclidean"):
    """Calcule le score de silhouette pour évaluer la qualité du clustering.

    Le score de silhouette mesure la séparation et la cohésion des clusters.
    Les valeurs varient de -1 à 1, où des valeurs plus élevées indiquent
    une meilleure séparation des clusters.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Données utilisées pour le clustering.
    labels : ndarray, shape (n_samples,)
        Assignations de cluster pour chaque point.
    metric : str, optional
        Métrique de distance utilisée (défaut='euclidean').
        Voir sklearn.metrics.silhouette_score pour les métriques supportées.

    Returns
    -------
    float
        Score de silhouette entre -1 et 1.

    Raises
    ------
    ImportError
        Si scikit-learn n'est pas installé.
    ValueError
        Si moins de 2 clusters ou autres conditions invalides.

    Notes
    -----
    Formule conceptuelle :
        s_i = (b_i - a_i) / max(a_i, b_i)

    où:
    - a_i : distance intra-cluster moyenne (moyenne des distances de i à tous les autres points du cluster)
    - b_i : distance au cluster voisin le plus proche (minimum sur les autres clusters)

    Interprétation des résultats :
    - Score ≈ 1  : clustering excellent (clusters bien séparés)
    - Score ≈ 0  : clusters se chevauchent
    - Score < 0  : mauvaise assignation (points plus proches d'autres clusters)

    Examples
    --------
    >>> from nuees_dynamiques import NuéesDynamique, generate_synthetic_data, compute_silhouette
    >>> X, y = generate_synthetic_data(200, n_features=2, n_clusters=3)
    >>> nd = NuéesDynamique(X, n_clusters=3)
    >>> nd.fit()
    >>> score = compute_silhouette(X, nd.labels_)
    >>> print(f"Silhouette score : {score:.3f}")
    """
    if not _HAS_SKLEARN:
        raise ImportError(
            "Le module `compute_silhouette` nécessite scikit-learn. "
            "Installez-le via : pip install scikit-learn"
        )

    X = np.asarray(X)
    labels = np.asarray(labels)

    # Valider que nous avons au moins 2 clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Au moins 2 clusters distincts sont nécessaires pour calculer la silhouette")

    # Calculer et retourner le score
    score = silhouette_score(X, labels, metric=metric)
    return score


def compute_davies_bouldin(X, labels):
    """Calcule l'indice Davies-Bouldin pour évaluer la qualité du clustering.

    L'indice Davies-Bouldin mesure le ratio moyen de similarité intra-cluster
    sur la dissimilarité inter-cluster. Les valeurs plus faibles indiquent
    une meilleure séparation des clusters.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Données utilisées pour le clustering.
    labels : ndarray, shape (n_samples,)
        Assignations de cluster pour chaque point.

    Returns
    -------
    float
        Indice Davies-Bouldin (>= 0).

    Raises
    ------
    ImportError
        Si scikit-learn n'est pas installé.
    ValueError
        Si moins de 2 clusters ou autres conditions invalides.

    Notes
    -----
    L'indice Davies-Bouldin est défini comme :
        DB = (1/n_clusters) * sum_i max_j(R_ij)

    où R_ij mesure la similarité relative entre les clusters i et j.

    Interprétation des résultats :
    - Score = 0  : clustering parfait (clusters bien séparés)
    - Score bas  : bonne séparation
    - Score élevé : clusters se chevauchent ou mal séparés

    Examples
    --------
    >>> from nuees_dynamiques import NuéesDynamique, generate_synthetic_data, compute_davies_bouldin
    >>> X, y = generate_synthetic_data(200, n_features=2, n_clusters=3)
    >>> nd = NuéesDynamique(X, n_clusters=3)
    >>> nd.fit()
    >>> db_score = compute_davies_bouldin(X, nd.labels_)
    >>> print(f"Davies-Bouldin score : {db_score:.3f}")
    """
    if not _HAS_SKLEARN:
        raise ImportError(
            "Le module `compute_davies_bouldin` nécessite scikit-learn. "
            "Installez-le via : pip install scikit-learn"
        )

    X = np.asarray(X)
    labels = np.asarray(labels)

    # Valider que nous avons au moins 2 clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Au moins 2 clusters distincts sont nécessaires pour calculer Davies-Bouldin")

    # Calculer et retourner le score
    score = davies_bouldin_score(X, labels)
    return score


# Exports publics
__all__ = [
    "generate_synthetic_data",
    "plot_clusters",
    "compute_silhouette",
    "compute_davies_bouldin",
]
