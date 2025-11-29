"""
Tests complets pour le module nuees_dynamiques/distances.py.

Ce module teste toutes les fonctions de calcul de distances :
- Fonctions unitaires : euclidean_distance, manhattan_distance, minkowski_distance, chebyshev_distance
- Fonction de dispatch : compute_distance
- Fonctions batch : compute_distance_matrix, pairwise_distances
"""

import pytest
import numpy as np
import numpy.testing as npt
from nuees_dynamiques.distances import (
    euclidean_distance,
    manhattan_distance,
    minkowski_distance,
    chebyshev_distance,
    compute_distance,
    compute_distance_matrix,
    pairwise_distances,
)


# ============================================================================
# Tests pour euclidean_distance (distance L2)
# ============================================================================


def test_euclidean_distance_known_values():
    """
    Teste euclidean_distance avec des cas connus.
    
    Vérifie que les distances euclidiennes calculées correspondent
    aux résultats attendus pour des cas simples.
    """
    # Cas 1 : triangle 3-4-5
    x = np.array([0.0, 0.0])
    y = np.array([3.0, 4.0])
    result = euclidean_distance(x, y)
    assert result == 5.0, "Distance entre [0,0] et [3,4] doit être 5.0"

    # Cas 2 : vecteurs identiques → distance = 0
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    result = euclidean_distance(x, y)
    assert result == 0.0, "Distance entre vecteurs identiques doit être 0"

    # Cas 3 : vecteurs simples en 3D
    x = np.array([0.0, 0.0, 0.0])
    y = np.array([1.0, 0.0, 0.0])
    result = euclidean_distance(x, y)
    assert result == 1.0, "Distance entre [0,0,0] et [1,0,0] doit être 1.0"


def test_euclidean_distance_symmetry():
    """
    Teste la propriété de symétrie : d(x,y) = d(y,x).
    
    La distance euclidienne doit être symétrique.
    """
    x = np.array([1.5, 2.7, 3.2])
    y = np.array([4.1, 5.3, 6.8])
    
    d_xy = euclidean_distance(x, y)
    d_yx = euclidean_distance(y, x)
    
    npt.assert_almost_equal(d_xy, d_yx)


def test_euclidean_distance_invalid_shapes():
    """
    Teste la validation des entrées : vecteurs de tailles différentes.
    
    Deux vecteurs de dimensions différentes doivent lever ValueError.
    """
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError):
        euclidean_distance(x, y)


def test_euclidean_distance_non_1d():
    """
    Teste la validation des entrées : tableaux 2D.
    
    Les entrées doivent être des vecteurs 1D, pas des matrices.
    """
    x = np.array([[1.0, 2.0]])  # Shape (1, 2) au lieu de (2,)
    y = np.array([1.0, 2.0])
    
    with pytest.raises(ValueError):
        euclidean_distance(x, y)


# ============================================================================
# Tests pour manhattan_distance (distance L1)
# ============================================================================


def test_manhattan_distance_known_values():
    """
    Teste manhattan_distance avec des cas connus.
    
    Distance L1 = somme des valeurs absolues des différences.
    """
    # Cas 1 : |1-0| + |2-0| = 3
    x = np.array([0.0, 0.0])
    y = np.array([1.0, 2.0])
    result = manhattan_distance(x, y)
    assert result == 3.0, "Distance Manhattan entre [0,0] et [1,2] doit être 3.0"

    # Cas 2 : vecteurs identiques
    x = np.array([5.0, 10.0, 15.0])
    y = np.array([5.0, 10.0, 15.0])
    result = manhattan_distance(x, y)
    assert result == 0.0, "Distance Manhattan entre vecteurs identiques doit être 0"

    # Cas 3 : |2-(-1)| + |4-3| = 3 + 1 = 4
    x = np.array([-1.0, 3.0])
    y = np.array([2.0, 4.0])
    result = manhattan_distance(x, y)
    assert result == 4.0, "Distance Manhattan doit être 4.0"


def test_manhattan_distance_symmetry():
    """
    Teste la propriété de symétrie pour Manhattan.
    """
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([5.0, 6.0, 7.0])
    
    d_xy = manhattan_distance(x, y)
    d_yx = manhattan_distance(y, x)
    
    npt.assert_equal(d_xy, d_yx)


def test_manhattan_distance_invalid_inputs():
    """
    Teste la validation des entrées pour Manhattan.
    """
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError):
        manhattan_distance(x, y)


# ============================================================================
# Tests pour minkowski_distance (distance Lp généralisée)
# ============================================================================


def test_minkowski_distance_p1_equals_manhattan():
    """
    Teste que minkowski_distance avec p=1 donne Manhattan.
    
    Pour p=1, la distance de Minkowski est identique à Manhattan.
    """
    x = np.array([0.0, 0.0])
    y = np.array([3.0, 4.0])
    
    # Manhattan : |3-0| + |4-0| = 7
    manhattan = manhattan_distance(x, y)
    minkowski_p1 = minkowski_distance(x, y, p=1)
    
    npt.assert_almost_equal(minkowski_p1, manhattan)
    npt.assert_almost_equal(minkowski_p1, 7.0)


def test_minkowski_distance_p2_equals_euclidean():
    """
    Teste que minkowski_distance avec p=2 donne Euclidienne.
    
    Pour p=2, la distance de Minkowski est identique à Euclidienne.
    """
    x = np.array([0.0, 0.0])
    y = np.array([3.0, 4.0])
    
    euclidean = euclidean_distance(x, y)
    minkowski_p2 = minkowski_distance(x, y, p=2)
    
    npt.assert_almost_equal(minkowski_p2, euclidean)
    npt.assert_almost_equal(minkowski_p2, 5.0)


def test_minkowski_distance_pinf_equals_chebyshev():
    """
    Teste que minkowski_distance avec p=inf donne Chebyshev.
    
    Pour p=inf, la distance de Minkowski est identique à Chebyshev.
    """
    x = np.array([0.0, 0.0])
    y = np.array([3.0, 4.0])
    
    chebyshev = chebyshev_distance(x, y)
    minkowski_pinf = minkowski_distance(x, y, p=np.inf)
    
    npt.assert_almost_equal(minkowski_pinf, chebyshev)
    npt.assert_almost_equal(minkowski_pinf, 4.0)


def test_minkowski_distance_invalid_p():
    """
    Teste que p < 1 lève ValueError.
    
    La distance de Minkowski est définie pour p >= 1.
    """
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    
    with pytest.raises(ValueError):
        minkowski_distance(x, y, p=0.5)


def test_minkowski_distance_known_values():
    """
    Teste minkowski_distance avec des valeurs p spécifiques.
    
    Vérifie le calcul pour p=3 et p=4.
    """
    # Sous-cas 1 : p=3 avec vecteurs [0,0] et [1,1]
    x = np.array([0.0, 0.0])
    y = np.array([1.0, 1.0])
    
    # Pour p=3 : (1^3 + 1^3)^(1/3) = (2)^(1/3) ≈ 1.26
    minkowski_p3 = minkowski_distance(x, y, p=3)
    expected_p3 = 2.0 ** (1/3)
    npt.assert_almost_equal(minkowski_p3, expected_p3)
    
    # Sous-cas 2 : p=4 avec vecteurs [0,0] et [1,2]
    x = np.array([0.0, 0.0])
    y = np.array([1.0, 2.0])
    
    # Pour p=4 : (|1-0|^4 + |2-0|^4)^(1/4) = (1 + 16)^(1/4) = 17^(1/4)
    minkowski_p4 = minkowski_distance(x, y, p=4)
    expected_p4 = (1.0**4 + 2.0**4) ** (1/4)
    npt.assert_almost_equal(minkowski_p4, expected_p4)


# ============================================================================
# Tests pour chebyshev_distance (distance L∞)
# ============================================================================


def test_chebyshev_distance_known_values():
    """
    Teste chebyshev_distance avec des cas connus.
    
    Distance L∞ = maximum de la valeur absolue des différences.
    """
    # Cas 1 : max(|3-0|, |1-0|) = max(3, 1) = 3
    x = np.array([0.0, 0.0])
    y = np.array([3.0, 1.0])
    result = chebyshev_distance(x, y)
    assert result == 3.0, "Distance Chebyshev doit être 3.0"

    # Cas 2 : vecteurs identiques
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    result = chebyshev_distance(x, y)
    assert result == 0.0, "Distance Chebyshev entre vecteurs identiques doit être 0"

    # Cas 3 : max(|5-1|, |8-2|, |3-7|) = max(4, 6, 4) = 6
    x = np.array([1.0, 2.0, 7.0])
    y = np.array([5.0, 8.0, 3.0])
    result = chebyshev_distance(x, y)
    assert result == 6.0, "Distance Chebyshev doit être 6.0"


def test_chebyshev_distance_symmetry():
    """
    Teste la propriété de symétrie pour Chebyshev.
    """
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([5.0, 6.0, 7.0])
    
    d_xy = chebyshev_distance(x, y)
    d_yx = chebyshev_distance(y, x)
    
    npt.assert_equal(d_xy, d_yx)


# ============================================================================
# Tests pour compute_distance (fonction de dispatch)
# ============================================================================


def test_compute_distance_euclidean():
    """
    Teste compute_distance avec metric='euclidean'.
    
    Doit dispatcher vers euclidean_distance.
    """
    x = np.array([0.0, 0.0])
    y = np.array([3.0, 4.0])
    
    result = compute_distance(x, y, metric="euclidean")
    assert result == 5.0


def test_compute_distance_manhattan():
    """
    Teste compute_distance avec metric='manhattan'.
    """
    x = np.array([0.0, 0.0])
    y = np.array([1.0, 2.0])
    
    result = compute_distance(x, y, metric="manhattan")
    assert result == 3.0


def test_compute_distance_l2_alias():
    """
    Teste compute_distance avec alias 'l2' pour euclidean.
    """
    x = np.array([0.0, 0.0])
    y = np.array([3.0, 4.0])
    
    result = compute_distance(x, y, metric="l2")
    assert result == 5.0


def test_compute_distance_l1_alias():
    """
    Teste compute_distance avec alias 'l1' pour manhattan.
    """
    x = np.array([0.0, 0.0])
    y = np.array([1.0, 2.0])
    
    result = compute_distance(x, y, metric="l1")
    assert result == 3.0


def test_compute_distance_linf_alias():
    """
    Teste compute_distance avec alias 'linf' pour chebyshev.
    """
    x = np.array([0.0, 0.0])
    y = np.array([3.0, 1.0])
    
    result = compute_distance(x, y, metric="linf")
    assert result == 3.0


def test_compute_distance_minkowski_with_p():
    """
    Teste compute_distance avec metric='minkowski' et passage de p.
    """
    x = np.array([0.0, 0.0])
    y = np.array([1.0, 1.0])
    
    result = compute_distance(x, y, metric="minkowski", p=3)
    expected = minkowski_distance(x, y, p=3)
    npt.assert_almost_equal(result, expected)


def test_compute_distance_unknown_metric():
    """
    Teste compute_distance avec une métrique inconnue.
    
    Une métrique invalide doit lever ValueError.
    """
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    
    with pytest.raises(ValueError):
        compute_distance(x, y, metric="unknown_metric")


# ============================================================================
# Tests pour compute_distance_matrix (calcul batch)
# ============================================================================


def test_compute_distance_matrix_shape():
    """
    Teste que compute_distance_matrix retourne une matrice de bonne forme.
    
    Forme attendue : (n_samples, n_centroids)
    """
    X = np.random.RandomState(42).randn(10, 5)  # 10 points, 5 features
    centroids = np.random.RandomState(42).randn(3, 5)  # 3 centroides
    
    result = compute_distance_matrix(X, centroids, metric="euclidean")
    
    assert result.shape == (10, 3), f"Forme attendue (10, 3), obtenu {result.shape}"


def test_compute_distance_matrix_euclidean():
    """
    Teste compute_distance_matrix avec Euclidienne.
    
    Compare les résultats avec calcul manuel pour chaque paire.
    """
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    centroids = np.array([[0.0, 0.0], [1.0, 1.0]])
    
    result = compute_distance_matrix(X, centroids, metric="euclidean")
    
    # Vérifier chaque distance manuellement
    npt.assert_almost_equal(result[0, 0], 0.0)  # d([0,0], [0,0]) = 0
    npt.assert_almost_equal(result[0, 1], np.sqrt(2))  # d([0,0], [1,1]) = sqrt(2)
    npt.assert_almost_equal(result[1, 0], 5.0)  # d([3,4], [0,0]) = 5
    npt.assert_almost_equal(result[1, 1], np.sqrt(13))  # d([3,4], [1,1]) = sqrt(13)


def test_compute_distance_matrix_manhattan():
    """
    Teste compute_distance_matrix avec Manhattan.
    """
    X = np.array([[0.0, 0.0], [1.0, 2.0]])
    centroids = np.array([[0.0, 0.0]])
    
    result = compute_distance_matrix(X, centroids, metric="manhattan")
    
    # Manhattan : |0-0| + |0-0| = 0, |1-0| + |2-0| = 3
    npt.assert_array_equal(result[:, 0], [0.0, 3.0])


def test_compute_distance_matrix_minkowski():
    """
    Teste compute_distance_matrix avec Minkowski (p=3).
    """
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    centroids = np.array([[0.0, 0.0]])
    
    result = compute_distance_matrix(X, centroids, metric="minkowski", p=3)
    
    # Minkowski p=3 : (|0-0|^3 + |0-0|^3)^(1/3) = 0, (|1-0|^3 + |1-0|^3)^(1/3) = 2^(1/3)
    npt.assert_almost_equal(result[0, 0], 0.0)
    npt.assert_almost_equal(result[1, 0], 2.0 ** (1/3))


def test_compute_distance_matrix_invalid_features():
    """
    Teste que X et centroids avec features différents lèvent ValueError.
    """
    X = np.random.RandomState(42).randn(10, 5)
    centroids = np.random.RandomState(42).randn(3, 7)  # Features différent
    
    with pytest.raises(ValueError):
        compute_distance_matrix(X, centroids, metric="euclidean")


def test_compute_distance_matrix_non_2d():
    """
    Teste que les entrées doivent être 2D.
    """
    X = np.array([1.0, 2.0, 3.0])  # 1D au lieu de 2D
    centroids = np.array([[1.0, 2.0]])
    
    with pytest.raises(ValueError):
        compute_distance_matrix(X, centroids, metric="euclidean")


# ============================================================================
# Tests pour pairwise_distances (matrice tous-tous)
# ============================================================================


def test_pairwise_distances_symmetry():
    """
    Teste que pairwise_distances retourne une matrice symétrique.
    
    La matrice de distance entre tous les points doit être symétrique.
    """
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    
    result = pairwise_distances(X, metric="euclidean")
    
    # Vérifier symétrie
    npt.assert_array_almost_equal(result, result.T)


def test_pairwise_distances_diagonal_zero():
    """
    Teste que la diagonale de la matrice de distance est zéro.
    
    La distance d'un point à lui-même est 0.
    """
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    result = pairwise_distances(X, metric="euclidean")
    
    npt.assert_array_almost_equal(np.diag(result), [0.0, 0.0, 0.0])


def test_pairwise_distances_euclidean():
    """
    Teste pairwise_distances avec Euclidienne.
    
    Compare les résultats avec calcul manuel.
    """
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    
    result = pairwise_distances(X, metric="euclidean")
    
    expected = np.array([[0.0, 5.0], [5.0, 0.0]])
    npt.assert_array_almost_equal(result, expected)


def test_pairwise_distances_manhattan():
    """
    Teste pairwise_distances avec Manhattan.
    """
    X = np.array([[0.0, 0.0], [1.0, 2.0]])
    
    result = pairwise_distances(X, metric="manhattan")
    
    expected = np.array([[0.0, 3.0], [3.0, 0.0]])
    npt.assert_array_equal(result, expected)


def test_pairwise_distances_minkowski_pinf():
    """
    Teste pairwise_distances avec Minkowski p=inf (Chebyshev).
    """
    X = np.array([[0.0, 0.0], [3.0, 1.0]])
    
    result = pairwise_distances(X, metric="minkowski", p=np.inf)
    
    # Chebyshev : max(|0-3|, |0-1|) = 3
    expected = np.array([[0.0, 3.0], [3.0, 0.0]])
    npt.assert_array_almost_equal(result, expected)


def test_pairwise_distances_invalid_p():
    """
    Teste que p < 1 lève ValueError avec Minkowski.
    """
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    with pytest.raises(ValueError):
        pairwise_distances(X, metric="minkowski", p=0.5)


def test_pairwise_distances_non_2d():
    """
    Teste que X doit être 2D.
    """
    X = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError):
        pairwise_distances(X, metric="euclidean")


def test_pairwise_distances_shape():
    """
    Teste que pairwise_distances retourne une matrice carrée (n, n).
    """
    X = np.random.RandomState(42).randn(5, 3)
    
    result = pairwise_distances(X, metric="euclidean")
    
    assert result.shape == (5, 5)


# ============================================================================
# Tests de cohérence inter-fonctions
# ============================================================================


def test_distance_metrics_consistency():
    """
    Teste la cohérence entre fonctions unitaires et batch.
    
    Les résultats de compute_distance_matrix doivent correspondre
    à des appels individuels à compute_distance.
    """
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    
    # Calcul batch
    batch_result = compute_distance_matrix(X, centroids, metric="euclidean")
    
    # Calcul unitaire
    for i in range(X.shape[0]):
        for j in range(centroids.shape[0]):
            expected = euclidean_distance(X[i], centroids[j])
            npt.assert_almost_equal(batch_result[i, j], expected)
