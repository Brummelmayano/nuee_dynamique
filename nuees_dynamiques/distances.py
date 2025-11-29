"""
Module de calcul des distances pour la méthode des Nuées Dynamiques.

Ce module fournit des fonctions vectorisées et optimisées pour calculer
les différentes métriques de distance utilisées par l'algorithme :
- euclidienne (L2)
- Manhattan (L1)
- Minkowski (Lp)
- Chebyshev (L\u221E)

Il propose aussi des fonctions en batch : `compute_distance_matrix` et
`pairwise_distances` pour les calculs entre ensembles de points et centroïdes.

Toutes les docstrings et messages d'erreur sont en français.
"""

from typing import Optional
import numpy as np

try:
    # Utiliser scipy si disponible pour pairwise optimisé
    from scipy.spatial.distance import pdist, squareform
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _validate_input(x, y=None):
    """Valide et convertit les entrées en arrays numpy.

    Si `y` est fourni, vérifie que `x` et `y` ont des formes compatibles.

    Parameters
    ----------
    x : array-like
        Vecteur ou tableau convertible en `numpy.ndarray`.
    y : array-like, optional
        Deuxième vecteur à comparer.

    Returns
    -------
    x_arr, y_arr
        Les entrées converties en `np.ndarray` (y_arr peut être None si y non fourni).

    Raises
    ------
    ValueError
        Si les formes ne sont pas compatibles ou si la conversion échoue.
    """
    x_arr = np.asarray(x)
    if y is None:
        return x_arr, None

    y_arr = np.asarray(y)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"Formes incompatibles : {x_arr.shape} vs {y_arr.shape}")

    # Contraintes : on attend des vecteurs 1D pour les fonctions bas niveau.
    # Cela permet de rester explicite sur le contrat d'entrée des fonctions
    # comme `euclidean_distance`, `manhattan_distance`, etc.
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("Les entrées doivent être des vecteurs 1D (tableaux à une dimension)")
    return x_arr, y_arr


def euclidean_distance(x, y):
    """Calcule la distance euclidienne (L2) entre deux vecteurs.

    Formula : sqrt(sum((x_i - y_i)^2)).

    Parameters
    ----------
    x, y : array-like
        Vecteurs de même dimension.

    Returns
    -------
    float
        Distance euclidienne entre `x` et `y`.

    Examples
    --------
    >>> euclidean_distance([0,0], [3,4])
    5.0
    """
    x_arr, y_arr = _validate_input(x, y)
    # Utilisation de la norme L2 de numpy
    return float(np.linalg.norm(x_arr - y_arr))


def manhattan_distance(x, y):
    """Calcule la distance de Manhattan (L1) entre deux vecteurs.

    Formula : sum(|x_i - y_i|).

    Parameters
    ----------
    x, y : array-like
        Vecteurs de même dimension.

    Returns
    -------
    float
        Distance de Manhattan.
    """
    x_arr, y_arr = _validate_input(x, y)
    return float(np.sum(np.abs(x_arr - y_arr)))

    # Examples:
    # >>> manhattan_distance([0,0], [1,2])
    # 3.0


def minkowski_distance(x, y, p: float = 2.0):
    """Calcule la distance de Minkowski d'ordre `p` entre deux vecteurs.

    Formula : (sum(|x_i - y_i|^p))^(1/p).

    Parameters
    ----------
    x, y : array-like
        Vecteurs de même dimension.
    p : float
        Ordre de la norme (p >= 1). p=1 -> Manhattan, p=2 -> Euclidienne.

    Returns
    -------
    float
        Distance de Minkowski.

    Raises
    ------
    ValueError
        Si `p < 1`.
    """
    if p < 1:
        raise ValueError("L'ordre 'p' pour Minkowski doit être >= 1")
    x_arr, y_arr = _validate_input(x, y)
    if np.isinf(p):
        # p = inf correspond à Chebyshev
        return chebyshev_distance(x_arr, y_arr)
    return float(np.sum(np.abs(x_arr - y_arr) ** p) ** (1.0 / p))

    # Examples:
    # >>> minkowski_distance([0,0], [1,1], p=1)
    # 2.0
    # >>> minkowski_distance([0,0], [3,4], p=2)
    # 5.0
    # >>> minkowski_distance([0,0], [2,5], p=float('inf'))
    # 5.0  # équivalent à Chebyshev


def chebyshev_distance(x, y):
    """Calcule la distance de Chebyshev (L\u221E) entre deux vecteurs.

    Formula : max(|x_i - y_i|).

    Parameters
    ----------
    x, y : array-like
        Vecteurs de même dimension.

    Returns
    -------
    float
        Distance de Chebyshev.
    """
    x_arr, y_arr = _validate_input(x, y)
    return float(np.max(np.abs(x_arr - y_arr)))

    # Examples:
    # >>> chebyshev_distance([0,0], [3,1])
    # 3.0


def compute_distance(x, y, metric: str = "euclidean", **kwargs):
    """Fonction générique pour calculer la distance selon une métrique choisie.

    Parameters
    ----------
    x, y : array-like
        Vecteurs à comparer.
    metric : str
        Nom de la métrique : 'euclidean', 'manhattan', 'minkowski', 'chebyshev'.
    **kwargs : dict
        Paramètres additionnels (ex: p pour Minkowski).

    Returns
    -------
    float
        Distance calculée.

    Raises
    ------
    ValueError
        Si la métrique n'est pas reconnue.
    """
    metric = metric.lower()
    dispatch = {
        "euclidean": euclidean_distance,
        "l2": euclidean_distance,
        "manhattan": manhattan_distance,
        "l1": manhattan_distance,
        "minkowski": minkowski_distance,
        "chebyshev": chebyshev_distance,
        "linf": chebyshev_distance,
    }
    if metric not in dispatch:
        raise ValueError(f"Métrique inconnue '{metric}'. Métriques supportées : {list(dispatch.keys())}")
    func = dispatch[metric]
    return func(x, y, **kwargs) if metric == "minkowski" else func(x, y)

# Exemple:
# >>> compute_distance([0,0], [3,4], metric='euclidean')
# 5.0


def compute_distance_matrix(X: np.ndarray, centroids: np.ndarray, metric: str = "euclidean", **kwargs) -> np.ndarray:
    """Calcule la matrice de distances entre les points de X et les centroïdes.

    La sortie est de forme (n_samples, n_centroids) où l'entrée [i, j] correspond
    à la distance entre X[i] et centroids[j].

    Implementations optimisées :
    - Pour 'euclidean' et 'manhattan' on utilise le broadcasting numpy.
    - Pour d'autres métriques, on effectue une boucle vectorisée sur les centroïdes.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    centroids : ndarray, shape (n_centroids, n_features)
    metric : str
        Nom de la métrique (voir `compute_distance`).
    **kwargs : dict
        Paramètres additionnels pour certaines métriques.

    Returns
    -------
    ndarray, shape (n_samples, n_centroids)
        Matrice des distances.
    """
    X_arr = np.asarray(X)
    C_arr = np.asarray(centroids)
    if X_arr.ndim != 2 or C_arr.ndim != 2:
        raise ValueError("X et centroids doivent être des tableaux 2D (n_samples, n_features)")
    if X_arr.shape[1] != C_arr.shape[1]:
        raise ValueError("Le nombre de caractéristiques (features) doit être identique pour X et centroids")

    metric = metric.lower()
    if metric in ("euclidean", "l2"):
        # broadcasting: (n_samples, 1, n_features) - (1, n_centroids, n_features)
        # Ici, `X_arr[:, np.newaxis, :]` étend X en (n_samples, 1, n_features)
        # et `C_arr[np.newaxis, :, :]` étend les centroïdes en (1, n_centroids, n_features).
        # La soustraction donne un tableau (n_samples, n_centroids, n_features)
        # puis on somme sur l'axe des features pour obtenir (n_samples, n_centroids).
        diff = X_arr[:, np.newaxis, :] - C_arr[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    if metric in ("manhattan", "l1"):
        # Même principe de broadcasting que pour Euclidienne, mais on somme
        # les valeurs absolues des différences pour obtenir la matrice (n_samples, n_centroids).
        diff = np.abs(X_arr[:, np.newaxis, :] - C_arr[np.newaxis, :, :])
        return np.sum(diff, axis=2)

    if metric in ("chebyshev", "linf"):
        # Broadcasting puis maximum sur l'axe des features -> (n_samples, n_centroids)
        diff = np.abs(X_arr[:, np.newaxis, :] - C_arr[np.newaxis, :, :])
        return np.max(diff, axis=2)

    # Pour minkowski ou métriques personnalisées
    if metric == "minkowski":
        p = kwargs.get("p", 2.0)
        if p < 1:
            raise ValueError("L'ordre 'p' pour Minkowski doit être >= 1")
        if np.isinf(p):
            # Chebyshev
            diff = np.abs(X_arr[:, np.newaxis, :] - C_arr[np.newaxis, :, :])
            return np.max(diff, axis=2)
        # calcul vectorisé pour p générique
        return np.sum(np.abs(X_arr[:, np.newaxis, :] - C_arr[np.newaxis, :, :]) ** p, axis=2) ** (1.0 / p)

    # Cas général : boucle sur les centroïdes en appelant compute_distance
    n_samples = X_arr.shape[0]
    n_centroids = C_arr.shape[0]
    D = np.empty((n_samples, n_centroids), dtype=float)
    for j in range(n_centroids):
        # vectorisé sur les échantillons
        D[:, j] = np.array([compute_distance(X_arr[i], C_arr[j], metric=metric, **kwargs) for i in range(n_samples)])
    return D


def pairwise_distances(X: np.ndarray, metric: str = "euclidean", **kwargs) -> np.ndarray:
    """Calcule la matrice symétrique des distances pairwise entre les points de X.

    Si SciPy est disponible, on utilise `pdist` + `squareform` pour de meilleures performances.
    Sinon on retombe sur une implémentation basée sur broadcasting.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    metric : str
        Nom de la métrique ('euclidean', 'manhattan', 'minkowski', 'chebyshev').
    **kwargs : dict
        Paramètres additionnels (ex: p pour Minkowski).

    Returns
    -------
    ndarray, shape (n_samples, n_samples)
        Matrice symétrique des distances.
    """
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError("X doit être un tableau 2D (n_samples, n_features)")

    metric = metric.lower()
    if _HAS_SCIPY:
        # mapping des noms pour scipy si nécessaire
        scipy_metric = metric
        if metric in ("euclidean", "l2"):
            scipy_metric = "euclidean"
        elif metric in ("manhattan", "l1"):
            scipy_metric = "cityblock"
        elif metric in ("chebyshev", "linf"):
            scipy_metric = "chebyshev"
        elif metric == "minkowski":
            p = kwargs.get("p", 2.0)
            # Validation cohérente avec minkowski_distance
            if p < 1:
                raise ValueError("L'ordre 'p' pour Minkowski doit être >= 1")
            # Si p est infini, on calcule Chebyshev via broadcasting (éviter pdist)
            if np.isinf(p):
                # Reprendre l'implémentation broadcasting utilisée dans la voie sans SciPy
                diff = np.abs(X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :])
                # (n_samples, n_samples)
                return np.max(diff, axis=2)
            return squareform(pdist(X_arr, metric="minkowski", p=p))
        return squareform(pdist(X_arr, metric=scipy_metric))

    # Implémentation via broadcasting (moins efficace en mémoire pour grands jeux de données)
    if metric in ("euclidean", "l2"):
        # Broadcasting : X_arr[:, np.newaxis, :] -> (n_samples, 1, n_features)
        #                X_arr[np.newaxis, :, :] -> (1, n_samples, n_features)
        # La soustraction donne un tableau (n_samples, n_samples, n_features)
        # puis on réduit sur l'axe des features pour obtenir (n_samples, n_samples).
        diff = X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))
    if metric in ("manhattan", "l1"):
        # Même principe : somme des valeurs absolues des différences -> (n_samples, n_samples)
        diff = np.abs(X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :])
        return np.sum(diff, axis=2)
    if metric in ("chebyshev", "linf"):
        # Maximum sur l'axe des features -> matrice (n_samples, n_samples)
        diff = np.abs(X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :])
        return np.max(diff, axis=2)
    if metric == "minkowski":
        p = kwargs.get("p", 2.0)
        # Validation cohérente avec les autres fonctions Minkowski
        if p < 1:
            raise ValueError("L'ordre 'p' pour Minkowski doit être >= 1")
        if np.isinf(p):
            diff = np.abs(X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :])
            return np.max(diff, axis=2)
        return np.sum(np.abs(X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :]) ** p, axis=2) ** (1.0 / p)

    # Cas général : utiliser compute_distance de façon pairwise
    n = X_arr.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_distance(X_arr[i], X_arr[j], metric=metric, **kwargs)
            D[i, j] = d
            D[j, i] = d
    return D


# Exports publics (le package principal ajoutera NuéesDynamique ultérieurement)
__all__ = [
    "euclidean_distance",
    "manhattan_distance",
    "minkowski_distance",
    "chebyshev_distance",
    "compute_distance",
    "compute_distance_matrix",
    "pairwise_distances",
]
