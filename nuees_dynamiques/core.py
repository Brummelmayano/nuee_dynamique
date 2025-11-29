"""
Module principal implémentant la classe `NuéesDynamique` pour l'algorithme
des nuées dynamiques (Diday, 1971).

La classe fournit une implémentation modulaire et reproductible avec deux
stratégies d'initialisation (`random`, `kmeans++`) et deux méthodes de
recalcul des étalons (`centroid`, `medoid`).

Références
---------
Diday, E. (1971). Méthode des nuées dynamiques.
"""

from typing import Optional, Literal
import numpy as np
from scipy.stats import mode as scipy_mode

from .distances import (
    compute_distance,
    compute_distance_matrix,
    pairwise_distances,
)


class NuéesDynamique:
    """Classe implémentant l'algorithme des Nuées Dynamiques.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Jeu de données à partitionner. Doit être convertible en tableau 2D.
    n_clusters : int
        Nombre de clusters (K).
    distance_metric : str, optional
        Métrique de distance (par défaut 'euclidean'). Doit être supportée
        par `nuees_dynamiques.distances.compute_distance`.
    max_iterations : int, optional
        Nombre maximal d'itérations (défaut=100).
    tolerance : float, optional
        Seuil de convergence sur le déplacement maximal des étalons (défaut=1e-4).
    init_method : {'random','kmeans++'}, optional
        Méthode d'initialisation des étalons.
    etallon_method : {'centroid','medoid','median','mode'}, optional
        Façon de recalculer les étalons à chaque itération.
    random_state : int | None, optional
        Graine pour la reproductibilité.

    Attributes
    ----------
    labels_ : ndarray, shape (n_samples,)
        Étiquettes des clusters après `fit()`.
    etallons_ : ndarray, shape (n_clusters, n_features)
        Étallons (prototypes) finaux.
    n_iter_ : int
        Nombre d'itérations effectuées.
    inertia_ : float
        Inertie finale (somme des distances au carré aux étalons).
    history_ : list
        Historique des inerties par itération.

    Examples
    --------
    >>> nd = NuéesDynamique(data, n_clusters=3, random_state=0)
    >>> nd.fit()
    >>> labels = nd.labels_
    """

    def __init__(
        self,
        data,
        n_clusters: int,
        distance_metric: str = "euclidean",
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        init_method: Literal["random", "kmeans++"] = "random",
        etallon_method: Literal["centroid", "medoid", "median", "mode"] = "centroid",
        random_state: Optional[int] = None,
    ):
        # Validation et stockage des paramètres
        self.data = np.asarray(data)
        if self.data.ndim != 2:
            raise ValueError("`data` doit être un tableau 2D de forme (n_samples, n_features)")
        n_samples = self.data.shape[0]
        if not (isinstance(n_clusters, int) and 1 <= n_clusters <= n_samples):
            raise ValueError("`n_clusters` doit être un entier compris entre 1 et n_samples")
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.max_iterations = int(max_iterations)
        if self.max_iterations < 1:
            raise ValueError("`max_iterations` doit être un entier supérieur ou égal à 1")
        self.tolerance = float(tolerance)
        if self.tolerance < 0:
            raise ValueError("`tolerance` doit être supérieure ou égale à 0")
        if init_method not in ("random", "kmeans++"):
            raise ValueError("`init_method` doit être 'random' ou 'kmeans++'")
        self.init_method = init_method
        if etallon_method not in ("centroid", "medoid", "median", "mode"):
            raise ValueError("`etallon_method` doit être 'centroid', 'medoid', 'median' ou 'mode'")
        self.etallon_method = etallon_method
        self.random_state = random_state

        # Attributs de sortie initialisés
        self.labels_ = None
        self.etallons_ = None
        self.n_iter_ = 0
        self.inertia_ = None
        self.history_ = []
        # Cache interne pour la dernière matrice de distances calculée
        self._last_distance_matrix = None

        # Instance RNG
        self._rng = np.random.RandomState(self.random_state)

    def initialize_etallons(self) -> np.ndarray:
        """Initialise les étalons selon `init_method`.

        Returns
        -------
        etallons : ndarray, shape (n_clusters, n_features)
            Étallons initiaux.

        Notes
        -----
        - `random` : sélection aléatoire de `n_clusters` observations.
        - `kmeans++` : initialisation probabiliste (préférer les points éloignés).

        Examples
        --------
        >>> nd = NuéesDynamique(data, n_clusters=3, init_method='random', random_state=0)
        >>> nd.initialize_etallons().shape
        (3, data.shape[1])
        """
        n_samples, n_features = self.data.shape
        if self.init_method == "random":
            indices = self._rng.choice(n_samples, size=self.n_clusters, replace=False)
            self.etallons_ = self.data[indices].astype(float)
            return self.etallons_

        # kmeans++
        # Supporter la métrique configurée en utilisant compute_distance_matrix
        centers = np.empty((self.n_clusters, n_features), dtype=float)
        # Choisir le premier centre aléatoirement
        first = self._rng.randint(0, n_samples)
        centers[0] = self.data[first]
        # Distances minimales actuelles de chaque point aux centres choisis
        current_dist_sq = np.full(n_samples, np.inf)
        for i in range(1, self.n_clusters):
            # Calculer les distances de tous les points aux centres déjà choisis.
            # On construit un tableau temporaire de centres sélectionnés.
            chosen_centers = centers[:i]
            # compute_distance_matrix supporte la métrique configurée
            D = compute_distance_matrix(self.data, chosen_centers, metric=self.distance_metric)
            # distance minimale de chaque point à l'ensemble des centres choisis
            min_dist = np.min(D, axis=1)
            # Probabilité proportionnelle au carré des distances (comme k-means++)
            current_dist_sq = min_dist ** 2
            total = np.sum(current_dist_sq)
            if total == 0:
                # Tous les points sont identiques aux centres sélectionnés ; choisir aléatoirement
                probs = None
            else:
                probs = current_dist_sq / total
            # Tirage selon la distribution (ou uniforme si probs est None)
            if probs is None:
                next_idx = self._rng.choice(n_samples)
            else:
                next_idx = self._rng.choice(n_samples, p=probs)
            centers[i] = self.data[next_idx]
        self.etallons_ = centers
        return self.etallons_

    def assign_objects(self) -> np.ndarray:
        """Assigne chaque observation au cluster le plus proche.

        Retourne un tableau 1D d'étiquettes de forme `(n_samples,)`.

        Notes
        -----
        Utilise `compute_distance_matrix` pour calculer efficacement la matrice
        des distances entre `self.data` et `self.etallons_`.

        Examples
        --------
        >>> nd.etallons_ = nd.initialize_etallons()
        >>> labels = nd.assign_objects()
        >>> labels.shape
        (n_samples,)
        """
        # Vérifier l'état
        if self.etallons_ is None:
            raise ValueError("`etallons_` est None : initialiser les étalons ou appeler `fit()` avant `assign_objects()`")

        D = compute_distance_matrix(self.data, self.etallons_, metric=self.distance_metric)
        # Cacher la dernière matrice de distances pour réutilisation éventuelle
        self._last_distance_matrix = D
        self.labels_ = np.argmin(D, axis=1)
        return self.labels_

    def update_etallons(self) -> np.ndarray:
        """Recalcule les étalons selon `etallon_method` pour chaque cluster.

        Returns
        -------
        etallons : ndarray, shape (n_clusters, n_features)
            Nouveaux étalons calculés.

        Notes
        -----
        - `centroid` : moyenne des objets du cluster.
        - `median` : médiane par dimension des objets du cluster (robuste aux outliers).
        - `mode` : valeur la plus fréquente par dimension (utile pour variables discrètes/catégorielles).
        - `medoid` : point du cluster minimisant la somme des distances aux autres points du cluster.
        - Si un cluster est vide, on réinitialise son étalon en prenant un point aléatoire.
        """
        # Vérifier l'état
        if self.labels_ is None:
            raise ValueError("`labels_` est None : appeler `assign_objects()` ou `fit()` avant `update_etallons()`")
        if len(self.labels_) != self.data.shape[0]:
            raise ValueError("La longueur de `labels_` doit correspondre au nombre d'échantillons de `data`")

        n_samples, n_features = self.data.shape
        new_etallons = np.zeros((self.n_clusters, n_features), dtype=float)
        for k in range(self.n_clusters):
            mask = (self.labels_ == k)
            members = np.where(mask)[0]
            if members.size == 0:
                # Cluster vide : réinitialiser aléatoirement
                idx = self._rng.randint(0, n_samples)
                new_etallons[k] = self.data[idx]
                continue
            cluster_points = self.data[members]
            if self.etallon_method == "centroid":
                new_etallons[k] = np.mean(cluster_points, axis=0)
            elif self.etallon_method == "median":
                # Médiane par dimension
                new_etallons[k] = np.median(cluster_points, axis=0)
            elif self.etallon_method == "mode":
                # Mode par dimension. scipy.stats.mode peut renvoyer un ModeResult
                # ou un tuple selon la version de SciPy. Extraire robustement.
                mode_res = scipy_mode(cluster_points, axis=0)
                # ModeResult possède l'attribut `mode`; pour compatibilité,
                # essayer d'accéder puis tomber back sur l'index 0.
                try:
                    modes = mode_res.mode
                except Exception:
                    modes = mode_res[0]
                modes = np.asarray(modes).squeeze()
                new_etallons[k] = modes
            else:
                # medoid (par défaut historique) : point du cluster avec somme minimale des distances
                D = pairwise_distances(cluster_points, metric=self.distance_metric)
                sums = np.sum(D, axis=1)
                medoid_idx = np.argmin(sums)
                new_etallons[k] = cluster_points[medoid_idx]
        # Invalider le cache de distances car les étalons viennent de changer
        self._last_distance_matrix = None
        # Assigner les nouveaux étalons
        self.etallons_ = new_etallons
        return self.etallons_

    def check_convergence(self, old_etallons: np.ndarray) -> bool:
        """Vérifie la convergence en regardant le déplacement maximal des étalons.

        Returns
        -------
        bool
            True si la distance maximale entre anciens et nouveaux étalons est
            inférieure ou égale à `self.tolerance`.
        """
        shifts = np.linalg.norm(self.etallons_ - old_etallons, axis=1)
        max_shift = np.max(shifts)
        return bool(max_shift <= self.tolerance)

    def _compute_inertia(self) -> float:
        """Calcule l'inertie : somme des distances au carré des points à leur étalon.

        La formule utilisée est : sum_i d(x_i, c_{label_i})^2.
        """
        # Réutiliser la matrice des distances calculée lors de l'assignation si disponible
        D = getattr(self, "_last_distance_matrix", None)
        if D is not None and D.shape == (self.data.shape[0], self.etallons_.shape[0]):
            chosen = D[np.arange(self.data.shape[0]), self.labels_]
            return float(np.sum(chosen ** 2))

        # Sinon calculer explicitement
        D = compute_distance_matrix(self.data, self.etallons_, metric=self.distance_metric)
        chosen = D[np.arange(self.data.shape[0]), self.labels_]
        return float(np.sum(chosen ** 2))

    def fit(self):
        """Exécute l'algorithme des nuées dynamiques jusqu'à convergence.

        Returns
        -------
        self
            L'objet lui-même avec les attributs `labels_`, `etallons_`, `n_iter_`, `inertia_` mis à jour.
        """
        # Politique de reproductibilité : si `random_state` est fourni,
        # réinitialiser le générateur aléatoire au début de chaque appel à `fit()`
        # afin de garantir que plusieurs appels successifs à `fit()` sur la même
        # instance produisent des résultats identiques.
        if self.random_state is not None:
            self._rng = np.random.RandomState(self.random_state)

        # Initialisation
        self.initialize_etallons()
        self.history_ = []
        for it in range(self.max_iterations):
            # Assigner en premier et conserver la matrice de distances calculée
            self.assign_objects()

            # Calculer l'inertie à partir de la matrice de distances mise en cache
            inertia = self._compute_inertia()
            self.history_.append(inertia)

            # Sauvegarder les anciens étalons avant mise à jour
            old_etallons = self.etallons_.copy()
            # Mettre à jour les étalons
            self.etallons_ = self.update_etallons()

            self.n_iter_ = it + 1
            if self.check_convergence(old_etallons):
                break

        # Résultats finaux
        self.inertia_ = self.history_[-1] if len(self.history_) > 0 else None
        return self

    def fit_predict(self):
        """Pratique : exécute `fit()` puis renvoie `labels_`.

        Returns
        -------
        ndarray
            Étiquettes de cluster (n_samples,).
        """
        self.fit()
        return self.labels_

    def predict(self, new_data):
        """Prédit les labels de cluster pour de nouvelles observations.

        Après entraînement, cette méthode classifie chaque nouvelle observation
        au cluster dont l'étalon est le plus proche.

        Parameters
        ----------
        new_data : array-like, shape (n_samples, n_features)
            Nouvelles données à classer. Doit avoir le même nombre de features
            que les données d'entraînement.

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            Indice du cluster assigné à chaque nouvelle observation.

        Raises
        ------
        ValueError
            Si le modèle n'a pas été entraîné (etallons_ est None) ou si
            new_data n'a pas le même nombre de features que les données d'entraînement.

        Notes
        -----
        La prédiction utilise la métrique de distance configurée lors de la
        création de l'instance (`distance_metric`).

        Examples
        --------
        >>> nd = NuéesDynamique(X_train, n_clusters=3)
        >>> nd.fit()
        >>> y_pred = nd.predict(X_test)
        """
        # Vérifier que le modèle a été entraîné
        if self.etallons_ is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appeler `fit()` avant `predict()`.")

        # Convertir new_data en tableau numpy
        new_data = np.asarray(new_data)
        if new_data.ndim != 2:
            raise ValueError("`new_data` doit être un tableau 2D de forme (n_samples, n_features)")

        # Gestion du cas où new_data est vide (zéro échantillon)
        if new_data.shape[0] == 0:
            return np.array([], dtype=int)

        # Vérifier le nombre de features
        if new_data.shape[1] != self.data.shape[1]:
            raise ValueError(
                f"Le nombre de features {new_data.shape[1]} ne correspond pas "
                f"au nombre de features des données d'entraînement {self.data.shape[1]}"
            )

        # Calculer les distances entre les nouvelles données et les étalons
        D = compute_distance_matrix(new_data, self.etallons_, metric=self.distance_metric)

        # Assigner à l'étalon le plus proche
        predicted_labels = np.argmin(D, axis=1)

        return predicted_labels

    def get_inertia(self):
        """Retourne l'inertie du clustering entraîné.

        L'inertie est la somme des carrés des distances entre chaque point
        et son étalon assigné : sum_i d(x_i, c_{label_i})^2.

        Returns
        -------
        float
            Valeur de l'inertie calculée lors du dernier `fit()`.

        Raises
        ------
        ValueError
            Si le modèle n'a pas été entraîné (inertia_ est None).

        Notes
        -----
        L'inertie est calculée internement par la méthode privée `_compute_inertia()`
        à chaque itération de `fit()`. Plus la valeur est faible, plus cohésifs
        sont les clusters (mais ce critère peut être biaisé pour de nombreux clusters).

        Examples
        --------
        >>> nd = NuéesDynamique(X, n_clusters=3)
        >>> nd.fit()
        >>> inertia = nd.get_inertia()
        >>> print(f"Inertie : {inertia}")
        """
        # Vérifier que le modèle a été entraîné
        if self.inertia_ is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appeler `fit()` avant `get_inertia()`.")

        return self.inertia_
