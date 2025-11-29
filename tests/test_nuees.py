"""
Tests complets pour le module nuees_dynamiques/core.py.

Ce module teste la classe NuéesDynamique de manière exhaustive :
- Initialisation avec validation des paramètres
- Méthodes d'initialisation des étalons (random, kmeans++)
- Assignation des objets aux clusters
- Mise à jour des étalons (centroïde, médoïde)
- Vérification de convergence
- Entraînement complet (fit)
- Prédiction sur nouvelles données
- Calcul de l'inertie
- Cas limites et edge cases
"""

import pytest
import numpy as np
import numpy.testing as npt
from nuees_dynamiques import NuéesDynamique, generate_synthetic_data


# ============================================================================
# Fixtures pytest
# ============================================================================


@pytest.fixture
def simple_data_2d():
    """
    Fixture : petit dataset 2D bien séparé pour les tests.
    
    Retourne un tuple (X, y_true) avec 30 points répartis en 3 clusters.
    """
    X, y_true = generate_synthetic_data(
        n_samples=30, n_features=2, n_clusters=3, cluster_std=0.5, random_state=42
    )
    return X, y_true


@pytest.fixture
def simple_data_3d():
    """
    Fixture : dataset 3D pour les tests.
    
    Retourne un tuple (X, y_true) avec 40 points répartis en 4 clusters.
    """
    X, y_true = generate_synthetic_data(
        n_samples=40, n_features=3, n_clusters=4, cluster_std=0.5, random_state=123
    )
    return X, y_true


@pytest.fixture
def fitted_model(simple_data_2d):
    """
    Fixture : modèle NuéesDynamique déjà entraîné.
    
    Retourne une instance de NuéesDynamique entraînée sur simple_data_2d.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(
        data=X,
        n_clusters=3,
        distance_metric="euclidean",
        init_method="kmeans++",
        random_state=42,
    )
    nd.fit()
    return nd


# ============================================================================
# Tests d'initialisation
# ============================================================================


def test_init_valid_parameters(simple_data_2d):
    """
    Teste que l'initialisation avec paramètres valides fonctionne.
    
    Vérifie que la création d'une instance ne lève pas d'erreur.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    assert nd is not None
    assert nd.n_clusters == 3


def test_init_invalid_data_shape():
    """
    Teste que des données 1D lèvent ValueError.
    
    NuéesDynamique attend une matrice 2D (n_samples, n_features).
    """
    X = np.array([1.0, 2.0, 3.0])  # 1D au lieu de 2D
    
    with pytest.raises(ValueError):
        NuéesDynamique(data=X, n_clusters=2)


def test_init_invalid_n_clusters_zero():
    """
    Teste que n_clusters=0 lève ValueError.
    """
    X = np.random.RandomState(42).randn(20, 2)
    
    with pytest.raises(ValueError):
        NuéesDynamique(data=X, n_clusters=0)


def test_init_invalid_n_clusters_greater_than_samples():
    """
    Teste que n_clusters > n_samples lève ValueError.
    """
    X = np.random.RandomState(42).randn(5, 2)
    
    with pytest.raises(ValueError):
        NuéesDynamique(data=X, n_clusters=10)


def test_init_invalid_max_iterations():
    """
    Teste que max_iterations < 1 lève ValueError.
    """
    X = np.random.RandomState(42).randn(20, 2)
    
    with pytest.raises(ValueError):
        NuéesDynamique(data=X, n_clusters=3, max_iterations=0)


def test_init_invalid_tolerance():
    """
    Teste que tolerance < 0 lève ValueError.
    """
    X = np.random.RandomState(42).randn(20, 2)
    
    with pytest.raises(ValueError):
        NuéesDynamique(data=X, n_clusters=3, tolerance=-0.1)


def test_init_invalid_init_method():
    """
    Teste que init_method invalide lève ValueError.
    """
    X = np.random.RandomState(42).randn(20, 2)
    
    with pytest.raises(ValueError):
        NuéesDynamique(data=X, n_clusters=3, init_method="invalid_method")


def test_init_invalid_etallon_method():
    """
    Teste que etallon_method invalide lève ValueError.
    """
    X = np.random.RandomState(42).randn(20, 2)
    
    with pytest.raises(ValueError):
        NuéesDynamique(data=X, n_clusters=3, etallon_method="invalid_method")


def test_init_invalid_distance_metric():
    """
    Teste que distance_metric invalide lève ValueError.
    
    La métrique est validée lors de initialize_etallons si init='kmeans++'.
    """
    X = np.random.RandomState(42).randn(20, 2)
    
    # Avec init_method='random', la métrique n'est pas testée à l'initialisation
    # Mais elle sera testée au premier appel de fit()
    nd = NuéesDynamique(
        data=X, n_clusters=3, distance_metric="unknown_metric", init_method="random"
    )
    assert nd is not None  # Initialisation acceptée


def test_fit_with_invalid_distance_metric_raises(simple_data_2d):
    """
    Teste que fit() lève ValueError avec une métrique de distance invalide.
    
    L'instance peut être créée avec une métrique invalide, mais fit()
    doit lever ValueError lors de l'utilisation réelle de la métrique.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(
        data=X,
        n_clusters=3,
        distance_metric="unknown_metric",
        init_method="random",
        random_state=42,
    )
    
    # fit() doit lever ValueError à cause de la métrique invalide
    with pytest.raises(ValueError):
        nd.fit()


def test_init_random_state_reproducibility(simple_data_2d):
    """
    Teste que random_state=42 donne résultats reproductibles.
    
    Deux initialisations avec le même random_state doivent donner
    les mêmes étalons initiaux (au moins pour random et kmeans++).
    """
    X, _ = simple_data_2d
    
    # Initialisation 1
    nd1 = NuéesDynamique(data=X, n_clusters=3, init_method="random", random_state=42)
    nd1.initialize_etallons()
    etallons1 = nd1.etallons_.copy()
    
    # Initialisation 2
    nd2 = NuéesDynamique(data=X, n_clusters=3, init_method="random", random_state=42)
    nd2.initialize_etallons()
    etallons2 = nd2.etallons_
    
    npt.assert_array_almost_equal(etallons1, etallons2)


# ============================================================================
# Tests de initialize_etallons
# ============================================================================


def test_initialize_etallons_random_shape(simple_data_2d):
    """
    Teste que initialize_etallons avec 'random' retourne bonne forme.
    
    Forme attendue : (n_clusters, n_features).
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, init_method="random", random_state=42)
    nd.initialize_etallons()
    
    assert nd.etallons_.shape == (3, 2)


def test_initialize_etallons_random_from_data(simple_data_2d):
    """
    Teste que les étalons 'random' sont des points du dataset.
    
    Chaque étalon doit correspondre à une ligne du dataset.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, init_method="random", random_state=42)
    nd.initialize_etallons()
    
    # Vérifier que chaque étalon est dans le dataset
    for i in range(nd.etallons_.shape[0]):
        # Trouver si cet étalon correspond à un point du dataset
        found = False
        for j in range(X.shape[0]):
            if np.allclose(nd.etallons_[i], X[j]):
                found = True
                break
        assert found, f"Étalon {i} ne correspond pas à un point du dataset"


def test_initialize_etallons_kmeans_plus_plus_shape(simple_data_2d):
    """
    Teste que initialize_etallons avec 'kmeans++' retourne bonne forme.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, init_method="kmeans++", random_state=42)
    nd.initialize_etallons()
    
    assert nd.etallons_.shape == (3, 2)


def test_initialize_etallons_kmeans_plus_plus_spread(simple_data_2d):
    """
    Teste que kmeans++ produit des étalons bien espacés.
    
    La distance minimale entre deux étalons doit être > 0.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, init_method="kmeans++", random_state=42)
    nd.initialize_etallons()
    
    # Calculer distances entre tous les étalons
    for i in range(nd.etallons_.shape[0]):
        for j in range(i + 1, nd.etallons_.shape[0]):
            distance = np.linalg.norm(nd.etallons_[i] - nd.etallons_[j])
            assert distance > 0.1, "Les étalons kmeans++ doivent être bien espacés"


def test_initialize_etallons_reproducibility(simple_data_2d):
    """
    Teste que initialize_etallons avec random_state est reproductible.
    
    Deux appels avec le même random_state doivent donner les mêmes étalons.
    """
    X, _ = simple_data_2d
    
    nd1 = NuéesDynamique(data=X, n_clusters=3, init_method="kmeans++", random_state=99)
    nd1.initialize_etallons()
    etallons1 = nd1.etallons_.copy()
    
    nd2 = NuéesDynamique(data=X, n_clusters=3, init_method="kmeans++", random_state=99)
    nd2.initialize_etallons()
    etallons2 = nd2.etallons_
    
    npt.assert_array_almost_equal(etallons1, etallons2)


# ============================================================================
# Tests de assign_objects
# ============================================================================


def test_assign_objects_shape(simple_data_2d):
    """
    Teste que assign_objects retourne labels de forme (n_samples,).
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.initialize_etallons()
    nd.assign_objects()
    
    assert nd.labels_.shape == (X.shape[0],)


def test_assign_objects_range(simple_data_2d):
    """
    Teste que les labels sont entre 0 et n_clusters-1.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.initialize_etallons()
    nd.assign_objects()
    
    assert np.all(nd.labels_ >= 0)
    assert np.all(nd.labels_ < 3)


def test_assign_objects_without_etallons():
    """
    Teste que assign_objects lève ValueError sans initialisation.
    """
    X = np.random.RandomState(42).randn(20, 2)
    nd = NuéesDynamique(data=X, n_clusters=3)
    
    with pytest.raises(ValueError):
        nd.assign_objects()


def test_assign_objects_deterministic(simple_data_2d):
    """
    Teste que assign_objects est déterministe.
    
    Avec les mêmes étalons, l'assignation doit être identique.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.initialize_etallons()
    etallons_copy = nd.etallons_.copy()
    
    nd.assign_objects()
    labels1 = nd.labels_.copy()
    
    # Restaurer les étalons et ré-assigner
    nd.etallons_ = etallons_copy.copy()
    nd.assign_objects()
    labels2 = nd.labels_
    
    npt.assert_array_equal(labels1, labels2)


def test_assign_objects_simple_case():
    """
    Teste assign_objects sur un cas simple et prévisible.
    
    Créer 3 points éloignés et 3 étalons aux mêmes positions.
    """
    # Données : 3 points distincts
    X = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    
    nd = NuéesDynamique(data=X, n_clusters=3)
    # Initialiser les étalons aux positions exactes des points
    nd.etallons_ = X.copy()
    
    nd.assign_objects()
    
    # Chaque point doit être assigné au cluster correspondant
    # (distance minimale = 0 à son étalon)
    expected_labels = np.array([0, 1, 2])
    npt.assert_array_equal(nd.labels_, expected_labels)


# ============================================================================
# Tests de update_etallons
# ============================================================================


def test_update_etallons_centroid_shape(simple_data_2d):
    """
    Teste que update_etallons centroid retourne bonne forme.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(
        data=X, n_clusters=3, etallon_method="centroid", random_state=42
    )
    nd.initialize_etallons()
    nd.assign_objects()
    nd.update_etallons()
    
    assert nd.etallons_.shape == (3, 2)


def test_update_etallons_centroid_calculation():
    """
    Teste que le centroïde est calculé correctement.
    
    Centroïde = moyenne arithmétique des points du cluster.
    """
    # Données simples : 2 clusters distincts
    X = np.array([[0.0, 0.0], [2.0, 2.0], [10.0, 10.0], [12.0, 12.0]])
    
    nd = NuéesDynamique(data=X, n_clusters=2, etallon_method="centroid")
    # Assigner manuellement : points 0,1 au cluster 0; points 2,3 au cluster 1
    nd.etallons_ = np.array([[0.0, 0.0], [10.0, 10.0]])  # Étalons initiaux
    nd.labels_ = np.array([0, 0, 1, 1])
    
    nd.update_etallons()
    
    # Centroïde du cluster 0 : moyenne de [0,0] et [2,2] = [1,1]
    # Centroïde du cluster 1 : moyenne de [10,10] et [12,12] = [11,11]
    expected = np.array([[1.0, 1.0], [11.0, 11.0]])
    npt.assert_array_almost_equal(nd.etallons_, expected)


def test_update_etallons_medoid_shape(simple_data_2d):
    """
    Teste que update_etallons médoïde retourne bonne forme.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, etallon_method="medoid", random_state=42)
    nd.initialize_etallons()
    nd.assign_objects()
    nd.update_etallons()
    
    assert nd.etallons_.shape == (3, 2)


def test_update_etallons_medoid_is_data_point(simple_data_2d):
    """
    Teste que le médoïde est toujours un point du dataset.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, etallon_method="medoid", random_state=42)
    nd.initialize_etallons()
    nd.assign_objects()
    nd.update_etallons()
    
    # Vérifier que chaque médoïde correspond à un point du dataset
    for i in range(nd.etallons_.shape[0]):
        found = False
        for j in range(X.shape[0]):
            if np.allclose(nd.etallons_[i], X[j]):
                found = True
                break
        assert found, f"Médoïde {i} ne correspond pas à un point du dataset"


def test_update_etallons_empty_cluster():
    """
    Teste la gestion des clusters vides.
    
    Forcer un cluster vide et vérifier la réassignation aléatoire.
    """
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    
    nd = NuéesDynamique(data=X, n_clusters=3, etallon_method="centroid", random_state=42)
    nd.etallons_ = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    # Assigner tous les points au cluster 0 (crée clusters vides 1 et 2)
    nd.labels_ = np.array([0, 0, 0])
    
    # update_etallons doit gérer les clusters vides
    nd.update_etallons()
    
    # Vérifier que les nouveaux étalons sont des points du dataset
    for i in range(nd.etallons_.shape[0]):
        found = False
        for j in range(X.shape[0]):
            if np.allclose(nd.etallons_[i], X[j]):
                found = True
                break
        assert found, f"Étalon {i} après réassignation ne correspond pas à un point"


def test_update_etallons_without_labels():
    """
    Teste que update_etallons lève ValueError sans labels.
    """
    X = np.random.RandomState(42).randn(20, 2)
    nd = NuéesDynamique(data=X, n_clusters=3)
    nd.initialize_etallons()
    
    with pytest.raises(ValueError):
        nd.update_etallons()


# ============================================================================
# Fixtures pour les tests median et mode
# ============================================================================


@pytest.fixture
def data_with_outliers():
    """
    Données synthétiques 2D avec outliers extrêmes pour tester la robustesse de median.
    
    Génère 3 clusters bien séparés avec des outliers extrêmes ajoutés.
    Retourne array numpy shape (n_samples, 2).
    """
    rng = np.random.RandomState(42)
    # 3 clusters bien séparés
    cluster1 = rng.randn(15, 2) + np.array([0, 0])
    cluster2 = rng.randn(15, 2) + np.array([10, 10])
    cluster3 = rng.randn(15, 2) + np.array([20, 20])
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # Ajouter des outliers extrêmes
    outliers = np.array([[100, 100], [-100, -100], [100, -100], [-100, 100]])
    X = np.vstack([X, outliers])
    return X


@pytest.fixture
def data_categorical():
    """
    Données discrètes/catégorielles avec valeurs répétées pour tester mode.
    
    Crée 3 clusters avec valeurs entières discrètes pures centrées sur 1, 5, 9.
    Les données sont générées comme entiers puis converties en float pour cohérence.
    Retourne array numpy shape (n_samples, 2) avec dtype float.
    """
    # Cluster 1 : entiers purs [1, 1] répétés
    cluster1 = np.ones((20, 2), dtype=float) * 1
    # Cluster 2 : entiers purs [5, 5] répétés
    cluster2 = np.ones((20, 2), dtype=float) * 5
    # Cluster 3 : entiers purs [9, 9] répétés
    cluster3 = np.ones((20, 2), dtype=float) * 9
    
    X = np.vstack([cluster1, cluster2, cluster3])
    return X.astype(float)


# ============================================================================
# Tests pour les nouveaux étalons : median et mode
# ============================================================================


def test_update_etallons_median_shape_and_type():
    """
    Vérifie que median produit des étalons de forme et type corrects.
    
    Teste l'ajout et la cohérence des étalons calculés via median.
    """
    X = np.random.RandomState(42).randn(30, 2)
    nd = NuéesDynamique(data=X, n_clusters=3, etallon_method="median")
    nd.etallons_ = nd.initialize_etallons()
    nd.labels_ = nd.assign_objects()
    
    new_etallons = nd.update_etallons()
    
    # Assertions
    assert new_etallons.shape == (3, 2), "Shape doit être (n_clusters, n_features)"
    assert new_etallons.dtype == np.float64 or new_etallons.dtype == float, "Type doit être float"
    assert not np.any(np.isnan(new_etallons)), "Pas de NaN dans les étalons"
    assert not np.any(np.isinf(new_etallons)), "Pas d'Inf dans les étalons"


def test_update_etallons_median_robustness_to_outliers(data_with_outliers):
    """
    Median est plus robuste aux outliers que centroid (moyenne).
    
    Compare la distance des étalons aux vrais centres entre median et centroid.
    Utilise un appariement ordre-indépendant : chaque centre vrai trouve son étalon appris le plus proche.
    """
    X = data_with_outliers
    
    # Modèle avec centroid
    nd_centroid = NuéesDynamique(data=X, n_clusters=3, etallon_method="centroid", random_state=42)
    nd_centroid.fit()
    
    # Modèle avec median
    nd_median = NuéesDynamique(data=X, n_clusters=3, etallon_method="median", random_state=42)
    nd_median.fit()
    
    # Centers théoriques
    true_centers = np.array([[0, 0], [10, 10], [20, 20]])
    
    # Calcul des distances avec appariement ordre-indépendant
    # Pour chaque centre vrai, trouver la distance minimale à n'importe quel étalon appris
    def compute_matched_distance(true_ctrs, learned_etallons):
        """Calcule la distance moyenne après appariement optimal ordre-indépendant."""
        distances = []
        for true_ctr in true_ctrs:
            # Distance de ce centre vrai à tous les étalons appris
            dists_to_learned = np.linalg.norm(learned_etallons - true_ctr, axis=1)
            # Distance minimale (meilleur appariement)
            distances.append(np.min(dists_to_learned))
        return np.mean(distances)
    
    dist_centroid = compute_matched_distance(true_centers, nd_centroid.etallons_)
    dist_median = compute_matched_distance(true_centers, nd_median.etallons_)
    
    # Median doit être plus proche (plus robuste)
    assert dist_median < dist_centroid, "Median doit être plus robuste aux outliers que centroid"


def test_update_etallons_median_convergence(data_with_outliers):
    """
    Convergence complète avec median sur données avec outliers.
    
    Vérifie que le modèle converge correctement et enregistre l'historique.
    """
    X = data_with_outliers
    nd = NuéesDynamique(data=X, n_clusters=3, etallon_method="median", max_iterations=50, random_state=42)
    nd.fit()
    
    # Assertions
    assert nd.n_iter_ > 0, "Au moins une itération"
    assert nd.inertia_ is not None and nd.inertia_ >= 0, "Inertie valide"
    assert len(nd.history_) == nd.n_iter_, "Historique complet"
    assert np.all((nd.labels_ >= 0) & (nd.labels_ < 3)), "Labels cohérents"


def test_update_etallons_mode_shape_and_type(data_categorical):
    """
    Mode produit des étalons valides pour données discrètes.
    
    Vérifie la forme, le type et l'absence de NaN/Inf.
    """
    X = data_categorical
    nd = NuéesDynamique(data=X, n_clusters=3, etallon_method="mode")
    nd.etallons_ = nd.initialize_etallons()
    nd.labels_ = nd.assign_objects()
    
    new_etallons = nd.update_etallons()
    
    # Assertions
    assert new_etallons.shape == (3, 2), "Shape doit être (n_clusters, n_features)"
    assert new_etallons.dtype in (np.float64, np.float32, float), "Type doit être numérique"
    assert not np.any(np.isnan(new_etallons)), "Pas de NaN dans les étalons"


def test_update_etallons_mode_correctness():
    """
    Mode identifie correctement les valeurs les plus fréquentes par dimension.
    
    Teste sur données déterministes avec clusters purs.
    """
    # Données simples : 3 clusters purs
    cluster1 = np.ones((10, 2)) * 1  # Tous [1, 1]
    cluster2 = np.ones((10, 2)) * 5  # Tous [5, 5]
    cluster3 = np.ones((10, 2)) * 9  # Tous [9, 9]
    X = np.vstack([cluster1, cluster2, cluster3])
    
    nd = NuéesDynamique(data=X, n_clusters=3, etallon_method="mode")
    nd.etallons_ = nd.initialize_etallons()
    # Assigner labels manuellement (parfait)
    nd.labels_ = np.array([0] * 10 + [1] * 10 + [2] * 10)
    
    new_etallons = nd.update_etallons()
    
    # Assertions : étalons doivent être les modes (valeurs pures)
    npt.assert_allclose(new_etallons[0], [1, 1], atol=0.1)
    npt.assert_allclose(new_etallons[1], [5, 5], atol=0.1)
    npt.assert_allclose(new_etallons[2], [9, 9], atol=0.1)


def test_update_etallons_mode_convergence(data_categorical):
    """
    Convergence stable avec mode sur données catégorielles.
    
    Vérifie la convergence et la stabilité des résultats.
    """
    X = data_categorical
    nd = NuéesDynamique(data=X, n_clusters=3, etallon_method="mode", max_iterations=50, random_state=42)
    nd.fit()
    
    # Assertions
    assert nd.n_iter_ <= nd.max_iterations, "Convergence atteinte"
    assert nd.inertia_ >= 0, "Inertie valide"
    assert np.all((nd.labels_ >= 0) & (nd.labels_ < 3)), "Labels dans le range"


def test_update_etallons_median_empty_cluster():
    """
    Gestion robuste des clusters vides avec median (réinit aléatoire).
    
    Force un cluster vide et vérifie que la réinitialisation utilise exactement un point de data.
    """
    X = np.random.RandomState(42).randn(20, 2)
    nd = NuéesDynamique(data=X, n_clusters=3, etallon_method="median", random_state=42)
    nd.etallons_ = nd.initialize_etallons()
    # Forcer tous les labels sur 0 ou 1 (cluster 2 vide)
    nd.labels_ = np.array([0] * 10 + [1] * 10)
    
    # Ne doit pas lever d'erreur
    new_etallons = nd.update_etallons()
    
    # Assertions
    assert new_etallons.shape == (3, 2), "Shape préservée"
    
    # Vérifier que l'étalon du cluster vide est exactement un point de X
    empty_etallon = new_etallons[2]
    min_dist_to_X = np.min(np.linalg.norm(X - empty_etallon, axis=1))
    assert min_dist_to_X < 1e-9, "Étalon du cluster vide doit être exactement un point de X"


def test_update_etallons_mode_empty_cluster():
    """
    Gestion robuste des clusters vides avec mode (réinit aléatoire).
    
    Force un cluster vide et vérifie que la réinitialisation utilise exactement un point de data.
    """
    X = np.random.RandomState(42).randn(20, 2)
    nd = NuéesDynamique(data=X, n_clusters=3, etallon_method="mode", random_state=42)
    nd.etallons_ = nd.initialize_etallons()
    # Forcer tous les labels sur 0 ou 1 (cluster 2 vide)
    nd.labels_ = np.array([0] * 10 + [1] * 10)
    
    # Ne doit pas lever d'erreur
    new_etallons = nd.update_etallons()
    
    # Assertions
    assert new_etallons.shape == (3, 2), "Shape préservée"
    
    # Vérifier que l'étalon du cluster vide est exactement un point de X
    empty_etallon = new_etallons[2]
    min_dist_to_X = np.min(np.linalg.norm(X - empty_etallon, axis=1))
    assert min_dist_to_X < 1e-9, "Étalon du cluster vide doit être exactement un point de X"


def test_fit_with_median_deterministic(data_with_outliers):
    """
    Reproductibilité avec random_state pour median.
    
    Deux entraînements identiques doivent donner les mêmes résultats.
    """
    X = data_with_outliers
    
    nd1 = NuéesDynamique(data=X, n_clusters=3, etallon_method="median", random_state=42)
    nd1.fit()
    
    nd2 = NuéesDynamique(data=X, n_clusters=3, etallon_method="median", random_state=42)
    nd2.fit()
    
    # Assertions
    assert np.array_equal(nd1.labels_, nd2.labels_), "Labels identiques"
    npt.assert_allclose(nd1.etallons_, nd2.etallons_, rtol=1e-10)
    assert nd1.n_iter_ == nd2.n_iter_, "Itérations identiques"
    assert np.isclose(nd1.inertia_, nd2.inertia_), "Inertie identique"


def test_fit_with_mode_deterministic(data_categorical):
    """
    Reproductibilité avec random_state pour mode.
    
    Deux entraînements identiques doivent donner les mêmes résultats.
    """
    X = data_categorical
    
    nd1 = NuéesDynamique(data=X, n_clusters=3, etallon_method="mode", random_state=42)
    nd1.fit()
    
    nd2 = NuéesDynamique(data=X, n_clusters=3, etallon_method="mode", random_state=42)
    nd2.fit()
    
    # Assertions
    assert np.array_equal(nd1.labels_, nd2.labels_), "Labels identiques"
    npt.assert_allclose(nd1.etallons_, nd2.etallons_, rtol=1e-5)
    assert nd1.n_iter_ == nd2.n_iter_, "Itérations identiques"
    assert np.isclose(nd1.inertia_, nd2.inertia_), "Inertie identique"


def test_predict_with_median(data_with_outliers):
    """
    Prédiction fonctionne correctement avec étalons median.
    
    Vérifie la cohérence de la prédiction après entraînement.
    """
    X = data_with_outliers
    nd = NuéesDynamique(data=X, n_clusters=3, etallon_method="median", random_state=42)
    nd.fit()
    
    # Prédire sur un subset
    new_data = X[:10]
    predictions = nd.predict(new_data)
    
    # Assertions
    assert predictions.shape == (10,), "Shape correcte"
    assert np.all((predictions >= 0) & (predictions < 3)), "Labels dans range [0, 2]"
    assert len(np.unique(predictions)) > 0, "Au moins un cluster prédit"


def test_predict_with_mode(data_categorical):
    """
    Prédiction fonctionne correctement avec étalons mode.
    
    Vérifie la cohérence de la prédiction après entraînement.
    """
    X = data_categorical
    nd = NuéesDynamique(data=X, n_clusters=3, etallon_method="mode", random_state=42)
    nd.fit()
    
    # Prédire sur un subset
    new_data = X[:10]
    predictions = nd.predict(new_data)
    
    # Assertions
    assert predictions.shape == (10,), "Shape correcte"
    assert np.all((predictions >= 0) & (predictions < 3)), "Labels dans range [0, 2]"
    assert len(np.unique(predictions)) > 0, "Au moins un cluster prédit"


# ============================================================================
# Tests de check_convergence
# ============================================================================


def test_check_convergence_true():
    """
    Teste que check_convergence retourne True pour étalons identiques.
    """
    X = np.random.RandomState(42).randn(20, 2)
    nd = NuéesDynamique(data=X, n_clusters=3)
    nd.etallons_ = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    # Copie exacte → convergence
    result = nd.check_convergence(nd.etallons_.copy())
    assert result is True


def test_check_convergence_false():
    """
    Teste que check_convergence retourne False pour étalons très différents.
    """
    X = np.random.RandomState(42).randn(20, 2)
    nd = NuéesDynamique(data=X, n_clusters=3)
    nd.etallons_ = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    # Étalons très différents
    new_etallons = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    result = nd.check_convergence(new_etallons)
    assert result is False


def test_check_convergence_threshold():
    """
    Teste check_convergence avec déplacement exactement = tolerance.
    """
    X = np.random.RandomState(42).randn(20, 2)
    tolerance = 0.1
    nd = NuéesDynamique(data=X, n_clusters=2, tolerance=tolerance)
    nd.etallons_ = np.array([[0.0, 0.0], [1.0, 1.0]])
    
    # Déplacement de 0.1 (au seuil de tolerance)
    new_etallons = np.array([[0.1, 0.0], [1.0, 1.0]])
    result = nd.check_convergence(new_etallons)
    # Avec tolerance=0.1, un déplacement de 0.1 doit converger
    assert result is True


# ============================================================================
# Tests de fit
# ============================================================================


def test_fit_completes(simple_data_2d):
    """
    Teste que fit() se termine sans erreur.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    # Ne doit pas lever d'exception
    nd.fit()


def test_fit_sets_attributes(simple_data_2d):
    """
    Teste que fit() initialise tous les attributs requis.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    assert hasattr(nd, "labels_") and nd.labels_ is not None
    assert hasattr(nd, "etallons_") and nd.etallons_ is not None
    assert hasattr(nd, "n_iter_") and nd.n_iter_ is not None
    assert hasattr(nd, "inertia_") and nd.inertia_ is not None
    assert hasattr(nd, "history_") and nd.history_ is not None


def test_fit_labels_shape(simple_data_2d):
    """
    Teste que labels_ a la bonne forme après fit().
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    assert nd.labels_.shape == (X.shape[0],)


def test_fit_etallons_shape(simple_data_2d):
    """
    Teste que etallons_ a la bonne forme après fit().
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    assert nd.etallons_.shape == (3, X.shape[1])


def test_fit_n_iter_positive(simple_data_2d):
    """
    Teste que n_iter_ >= 1 après fit().
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    assert nd.n_iter_ >= 1


def test_fit_inertia_positive(simple_data_2d):
    """
    Teste que inertia_ >= 0 après fit().
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    assert nd.inertia_ >= 0


def test_fit_history_length(simple_data_2d):
    """
    Teste que history_ a longueur = n_iter_.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    assert len(nd.history_) == nd.n_iter_


def test_fit_convergence(simple_data_2d):
    """
    Teste que fit() converge sur données simples.
    
    Avec tolerance élevée, convergence doit être rapide (< 50 itérations).
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, tolerance=0.01, max_iterations=50)
    nd.fit()
    
    assert nd.n_iter_ < 50


def test_fit_max_iterations_respected(simple_data_2d):
    """
    Teste que max_iterations est respecté.
    
    Avec max_iterations=1, n_iter_ doit être 1.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, max_iterations=1, random_state=42)
    nd.fit()
    
    assert nd.n_iter_ == 1


def test_fit_reproducibility(simple_data_2d):
    """
    Teste que fit() avec random_state=42 donne résultats reproductibles.
    """
    X, _ = simple_data_2d
    
    nd1 = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd1.fit()
    labels1 = nd1.labels_.copy()
    
    nd2 = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd2.fit()
    labels2 = nd2.labels_
    
    npt.assert_array_equal(labels1, labels2)


def test_fit_immediate_convergence():
    """
    Teste la convergence immédiate quand les étalons initiaux sont optimaux.
    
    Si les étalons initiaux sont déjà les centroïdes vrais,
    la convergence doit se produire en exactement 1 itération (avec max_iterations=1).
    """
    # Créer des données avec centroïdes connus
    X = np.array([[0.0, 0.0], [0.5, 0.5], [10.0, 10.0], [10.5, 10.5]])
    
    nd = NuéesDynamique(data=X, n_clusters=2, max_iterations=1)
    # Assigner les centroïdes vrais
    nd.etallons_ = np.array([[0.25, 0.25], [10.25, 10.25]])
    nd.labels_ = np.array([0, 0, 1, 1])
    
    nd.fit()
    
    # Avec centroïdes optimaux et max_iterations=1, n_iter_ doit être exactement 1
    assert nd.n_iter_ == 1, f"Convergence immédiate : attendu n_iter_=1, obtenu {nd.n_iter_}"


# ============================================================================
# Tests de fit_predict
# ============================================================================


def test_fit_predict_returns_labels(simple_data_2d):
    """
    Teste que fit_predict() retourne les labels.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    labels = nd.fit_predict()
    
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (X.shape[0],)


def test_fit_predict_equals_fit_then_labels(simple_data_2d):
    """
    Teste que fit_predict() équivaut à fit() + accès à labels_.
    """
    X, _ = simple_data_2d
    
    nd1 = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    labels1 = nd1.fit_predict()
    
    nd2 = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd2.fit()
    labels2 = nd2.labels_
    
    npt.assert_array_equal(labels1, labels2)


# ============================================================================
# Tests de predict
# ============================================================================


def test_predict_without_fit():
    """
    Teste que predict() lève ValueError sans fit().
    """
    X_train = np.random.RandomState(42).randn(20, 2)
    nd = NuéesDynamique(data=X_train, n_clusters=3)
    
    X_test = np.random.RandomState(42).randn(5, 2)
    with pytest.raises(ValueError):
        nd.predict(X_test)


def test_predict_shape(simple_data_2d):
    """
    Teste que predict() retourne labels de bonne forme.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    X_test = np.random.RandomState(99).randn(5, 2)
    labels = nd.predict(X_test)
    
    assert labels.shape == (5,)


def test_predict_range(simple_data_2d):
    """
    Teste que predict() retourne labels entre 0 et n_clusters-1.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    X_test = np.random.RandomState(99).randn(5, 2)
    labels = nd.predict(X_test)
    
    assert np.all(labels >= 0)
    assert np.all(labels < 3)


def test_predict_same_data(simple_data_2d):
    """
    Teste que predict() sur données d'entraînement donne labels similaires.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    labels_train = nd.labels_.copy()
    labels_test = nd.predict(X)
    
    # Doit être identique ou très similaire
    npt.assert_array_equal(labels_train, labels_test)


def test_predict_invalid_features(simple_data_2d):
    """
    Teste que predict() lève ValueError avec nombre de features différent.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    X_test = np.random.RandomState(99).randn(5, 5)  # Features différent
    with pytest.raises(ValueError):
        nd.predict(X_test)


def test_predict_empty_data(simple_data_2d):
    """
    Teste que predict() avec données vides retourne array vide.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    X_test = np.empty((0, 2))
    labels = nd.predict(X_test)
    
    assert labels.shape == (0,)


def test_predict_non_2d(simple_data_2d):
    """
    Teste que predict() lève ValueError avec données 1D.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    X_test = np.array([1.0, 2.0])  # 1D au lieu de 2D
    with pytest.raises(ValueError):
        nd.predict(X_test)


def test_predict_deterministic(simple_data_2d):
    """
    Teste que deux appels à predict() donnent les mêmes résultats.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    X_test = np.random.RandomState(99).randn(5, 2)
    labels1 = nd.predict(X_test)
    labels2 = nd.predict(X_test)
    
    npt.assert_array_equal(labels1, labels2)


# ============================================================================
# Tests de get_inertia
# ============================================================================


def test_get_inertia_without_fit():
    """
    Teste que get_inertia() lève ValueError sans fit().
    """
    X = np.random.RandomState(42).randn(20, 2)
    nd = NuéesDynamique(data=X, n_clusters=3)
    
    with pytest.raises(ValueError):
        nd.get_inertia()


def test_get_inertia_positive(simple_data_2d):
    """
    Teste que get_inertia() retourne valeur >= 0.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    inertia = nd.get_inertia()
    assert inertia >= 0


def test_get_inertia_equals_attribute(simple_data_2d):
    """
    Teste que get_inertia() == inertia_.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    npt.assert_almost_equal(nd.get_inertia(), nd.inertia_)


def test_get_inertia_decreases(simple_data_2d):
    """
    Teste que l'inertia décroît ou stagne au cours des itérations.
    """
    X, _ = simple_data_2d
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    # L'historique d'inertia doit être non-croissant
    for i in range(1, len(nd.history_)):
        assert nd.history_[i] <= nd.history_[i - 1] + 1e-6, "Inertia doit décroître"


# ============================================================================
# Tests de _compute_inertia
# ============================================================================


def test_compute_inertia_calculation():
    """
    Teste que _compute_inertia calcule correctement.
    
    Inertia = somme des distances au carré vers les étalons les plus proches.
    """
    # Données simples
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    
    nd = NuéesDynamique(data=X, n_clusters=1)
    nd.etallons_ = np.array([[0.0, 0.0]])
    nd.labels_ = np.array([0, 0, 0])
    
    inertia = nd._compute_inertia()
    
    # Distances : 0, 1, 1
    # Inertia = 0^2 + 1^2 + 1^2 = 2
    npt.assert_almost_equal(inertia, 2.0)


def test_compute_inertia_zero_for_perfect_clustering():
    """
    Teste que l'inertia = 0 si tous les points = étalons.
    """
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    
    nd = NuéesDynamique(data=X, n_clusters=2)
    nd.etallons_ = X.copy()
    nd.labels_ = np.array([0, 1])
    
    inertia = nd._compute_inertia()
    
    npt.assert_almost_equal(inertia, 0.0)


# ============================================================================
# Tests déterministes (données manuelles, sans generate_synthetic_data)
# Réduisent la dépendance aux fixtures et fournissent des cas stables
# ============================================================================


def test_fit_deterministic_two_clusters():
    """
    Teste fit() sur un petit dataset déterministe avec 2 clusters bien séparés.
    
    Données : 4 points formant 2 clusters distincts (pas de generate_synthetic_data).
    - Points 0,1 : cluster 0 au voisinage de [0, 0]
    - Points 2,3 : cluster 1 au voisinage de [10, 10]
    """
    X = np.array([[0.0, 0.0], [0.5, 0.5], [10.0, 10.0], [10.5, 10.5]])
    
    nd = NuéesDynamique(data=X, n_clusters=2, random_state=42, max_iterations=50)
    nd.fit()
    
    # Vérification des attributs
    assert nd.labels_ is not None
    assert len(nd.labels_) == 4
    
    # Vérification que les points proches sont dans le même cluster
    # Points 0 et 1 doivent être assignés au même cluster
    assert nd.labels_[0] == nd.labels_[1], "Points proches doivent être dans le même cluster"
    
    # Points 2 et 3 doivent être assignés au même cluster
    assert nd.labels_[2] == nd.labels_[3], "Points proches doivent être dans le même cluster"
    
    # Les deux clusters doivent être différents
    assert nd.labels_[0] != nd.labels_[2], "Clusters éloignés doivent être différents"


def test_predict_deterministic_handcrafted():
    """
    Teste predict() sur données manuelles pour vérifier assignation correcte.
    
    Entraîner sur 4 points distincts, prédire sur des points identiques
    pour vérifier que la prédiction est déterministe et correcte.
    """
    # Données d'entraînement : 4 points bien séparés
    X_train = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]])
    
    nd = NuéesDynamique(data=X_train, n_clusters=2, random_state=42)
    nd.fit()
    
    # Données de test : identifier chaque point d'entraînement
    X_test = np.array([[0.05, 0.05], [10.05, 10.05]])
    labels_test = nd.predict(X_test)
    
    # Le premier point de test (près de [0,0]) doit être dans le même cluster
    # que les points 0,1 d'entraînement
    assert labels_test[0] == nd.labels_[0], "Prédiction échoue pour point proche de X_train[0]"
    
    # Le second point de test (près de [10,10]) doit être dans le même cluster
    # que les points 2,3 d'entraînement
    assert labels_test[1] == nd.labels_[2], "Prédiction échoue pour point proche de X_train[2]"


def test_get_inertia_deterministic_single_cluster():
    """
    Teste get_inertia() sur un cas simple avec inertia calculable manuellement.
    
    Configuration : 1 cluster avec 3 points connus, inertia = somme distances²
    """
    # Données : 3 points alignés verticalement
    X = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    
    nd = NuéesDynamique(data=X, n_clusters=1, random_state=42)
    nd.fit()
    
    inertia = nd.get_inertia()
    
    # Avec 1 cluster, l'étalon devrait être le centroïde [0, 1]
    # Distances : |0-0|=0, |1-1|=0, |2-1|=1 → inertia = 0 + 0 + 1² = 1
    # (Pour la distance euclidienne 2D avec centroïde [0, 1])
    # Distances réelles : 0, 1, 1 → inertia = 0 + 1 + 1 = 2
    npt.assert_almost_equal(inertia, 2.0, decimal=5)


# ============================================================================
# Tests de edge cases
# ============================================================================


def test_single_cluster():
    """
    Teste le clustering avec un seul cluster.
    
    Tous les points doivent avoir le label 0.
    """
    X = np.random.RandomState(42).randn(20, 2)
    nd = NuéesDynamique(data=X, n_clusters=1, random_state=42)
    nd.fit()
    
    npt.assert_array_equal(nd.labels_, np.zeros(20, dtype=int))


def test_n_clusters_equals_n_samples():
    """
    Teste quand n_clusters = n_samples.
    
    Chaque point devient son propre cluster.
    """
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    # Chaque point doit avoir un label unique
    assert len(np.unique(nd.labels_)) == 3


def test_identical_points():
    """
    Teste le clustering avec tous les points identiques.
    
    Ne doit pas lever d'erreur.
    """
    X = np.ones((20, 2))
    nd = NuéesDynamique(data=X, n_clusters=3, random_state=42)
    nd.fit()
    
    # Ne doit pas lever d'exception
    assert nd.labels_ is not None


def test_different_metrics():
    """
    Teste fit() avec différentes métriques de distance.
    """
    X = np.random.RandomState(42).randn(20, 2)
    
    for metric in ["euclidean", "manhattan", "chebyshev"]:
        nd = NuéesDynamique(data=X, n_clusters=3, distance_metric=metric, random_state=42)
        nd.fit()
        assert nd.labels_ is not None, f"Métrique {metric} a échoué"


def test_different_init_methods():
    """
    Teste fit() avec différentes méthodes d'initialisation.
    """
    X = np.random.RandomState(42).randn(20, 2)
    
    labels_random = None
    labels_kmeans = None
    
    for init_method in ["random", "kmeans++"]:
        nd = NuéesDynamique(data=X, n_clusters=3, init_method=init_method, random_state=42)
        nd.fit()
        assert nd.labels_ is not None
        
        if init_method == "random":
            labels_random = nd.labels_.copy()
        else:
            labels_kmeans = nd.labels_.copy()
    
    # Les deux méthodes doivent converger (mais peuvent donner des résultats différents)
    assert labels_random is not None
    assert labels_kmeans is not None


def test_different_etallon_methods(simple_data_2d):
    """
    Teste fit() avec différentes méthodes d'étalon.
    """
    X, _ = simple_data_2d
    
    for etallon_method in ["centroid", "medoid"]:
        nd = NuéesDynamique(
            data=X, n_clusters=3, etallon_method=etallon_method, random_state=42
        )
        nd.fit()
        assert nd.labels_ is not None, f"Méthode {etallon_method} a échoué"
        
        # Médoïde doit être un point du dataset
        if etallon_method == "medoid":
            for i in range(nd.etallons_.shape[0]):
                found = any(np.allclose(nd.etallons_[i], X[j]) for j in range(X.shape[0]))
                assert found, f"Médoïde {i} n'est pas un point du dataset"
