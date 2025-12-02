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
    n_etalons_per_cluster : int, optional
        Nombre d'étalons (ni) par cluster. Si `> 1`, chaque cluster est
        représenté par un noyau de `ni` points (conforme à Diday IV.1).
    random_state : int | None, optional
        Graine pour la reproductibilité.

    Attributes
    ----------
    labels_ : ndarray, shape (n_samples,)
        Étiquettes des clusters après `fit()`.
        Correspond aux parties C1, C2, ..., CK du Programme IV.2 (Diday, 1971).
    etallons_ : ndarray, shape (n_clusters, n_features) or (n_clusters, n_etalons_per_cluster, n_features)
        Étallons (prototypes) finaux.

        Peut être soit :
        - un tableau 2D de forme `(n_clusters, n_features)` lorsque
            `n_etalons_per_cluster == 1` (cas historique),
        - soit un tableau 3D de forme `(n_clusters, n_etalons_per_cluster, n_features)`
            lorsque `n_etalons_per_cluster > 1`. Dans ce dernier cas chaque cluster
            est représenté par un « noyau » de `ni` points (notation Diday IV.1).
        
        Correspond aux noyaux E1, E2, ..., EK du Programme IV.2 (Diday, 1971).
    n_iter_ : int
        Nombre d'itérations effectuées.
    inertia_ : float
        Inertie finale (somme des distances au carré aux étalons).
        Fournit une mesure d'homogénéité globale de la partition (sortie IV.2, Diday).
    cluster_sizes_ : ndarray, shape (n_clusters,)
        Nombre de points dans chaque cluster (ni = |Ci|).
    homogeneite_per_cluster_ : ndarray, shape (n_clusters,)
        Homogénéité Si de chaque cluster (somme des distances carrées intra-cluster).
    S_total_ : float
        Somme totale des Si (S = ∑Si), équivalent à `inertia_`.
    representative_noyau_ : ndarray, shape (n_clusters, n_features)
        Centroïde de chaque noyau (ou étalon unique si ni=1).
    history_ : list
        Historique des inerties par itération.
    etallons_history_ : list of ndarray
        Historique des étalons par itération. Chaque élément est un snapshot
        de `etallons_` (forme 2D ou 3D selon `n_etalons_per_cluster`).

    Notes on Diday IV.2 Correspondence
    ----------------------------------
    Les attributs et méthodes de cette classe correspondent aux sorties du
    Programme IV.2 décrit dans Diday (1971) :
    
    - **labels_** : partition finale (C1, C2, ..., CK).
    - **etallons_** : noyaux finaux (E1, E2, ..., EK). Pour chaque cluster i,
      Ei est soit un seul étalon (ni=1) soit un ensemble de ni étalons (IV.1).
    - **inertia_** et **history_** : mesure d'homogénéité globale et historique.
      On peut aussi étendre par du code utilisateur pour calculer des mesures
      par classe (similarité individu/classe, etc.).
    - **compute_silhouette()**, **compute_davies_bouldin()** : fournissent une
      mesure qualitative de la partition (validité globale).

    Examples
    --------
        >>> nd = NuéesDynamique(data, n_clusters=3, random_state=0)
        >>> nd.fit()
        >>> labels = nd.labels_

        Notes
        -----
        - IV.1 : support des noyaux multi-étalons. Si `n_etalons_per_cluster > 1`,
            chaque slice `etallons_[k]` est le noyau `E_i` du cluster k (ensemble de
            `ni` points) conformément à Diday (1971, IV.1).

        Examples (multi-étalons)
        ------------------------
        >>> # Exemple Diday-like : 283 points, ~40 étalons par cluster
        >>> nd = NuéesDynamique(data, n_clusters=2, n_etalons_per_cluster=40)
        >>> nd.initialize_etallons()  # -> shape (2, 40, n_features) si possible
    
    Références
    ----------
    Diday, E. (1971). Méthode des nuées dynamiques — voir section IV.1 (noyaux multi-étalons)
    et section IV.2 (Programme principal et sorties).
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
        n_etalons_per_cluster: int = 1,
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

        # Nombre d'étalons par cluster (ni en notation Diday IV.1)
        if not (isinstance(n_etalons_per_cluster, int) and n_etalons_per_cluster >= 1):
            raise ValueError("`n_etalons_per_cluster` doit être un entier >= 1")
        # Eviter des noyaux trop grands par rapport à la taille moyenne d'un cluster
        # Note: on ne lève plus d'erreur si `n_etalons_per_cluster` est plus grand
        # que la taille moyenne d'un cluster ; la logique d'initialisation et de
        # mise à jour gère le padding lorsque le cluster contient moins de
        # points que `n_etalons_per_cluster`.
        self.n_etalons_per_cluster = n_etalons_per_cluster

        self.random_state = random_state
        self.square_distances = self.distance_metric not in ['sebestyen']  # Divergences use sum(R), not sum(R²)

        # Attributs de sortie initialisés
        self.labels_ = None
        self.etallons_ = None
        self.n_iter_ = 0
        self.inertia_ = None
        self.history_ = []
        self.etallons_history_ = []  # Historique des étalons par itération
        # Cache interne pour la dernière matrice de distances calculée
        self._last_distance_matrix = None

        # Instance RNG
        self._rng = np.random.RandomState(self.random_state)

    def initialize_etallons(self) -> np.ndarray:
        """Initialise les étalons selon `init_method`.

                Returns
                -------
                etallons : ndarray
                        Étallons initiaux.

                        - Si `n_etalons_per_cluster == 1` : tableau 2D de forme
                            `(n_clusters, n_features)`.
                        - Si `n_etalons_per_cluster > 1` : tableau 3D de forme
                            `(n_clusters, n_etalons_per_cluster, n_features)` où
                            `etallons_[k]` est le noyau `E_i` pour le cluster `k`. 

       Après exécution, `self.etallons_history_` contient l'état initial des étalons (premier élément). Utilisez `fit()` pour l'historique complet des itérations.

       Notes
        -----
        - `random` : sélection aléatoire de `n_clusters` observations.
        - `kmeans++` : initialisation probabiliste (préférer les points éloignés).
        
            Pour `n_etalons_per_cluster > 1`, la stratégie est la suivante :
            on commence par calculer des centres 2D (un point représentatif par
            cluster), on effectue une assignation préliminaire de chaque point à
            ces centres, puis on sélectionne `ni` points par cluster pour
            construire le noyau (avec un remplissage si le cluster est trop petit).

        Examples
        --------
        >>> nd = NuéesDynamique(data, n_clusters=3, init_method='random', random_state=0)
        >>> nd.initialize_etallons().shape
        (3, data.shape[1])  # si `n_etalons_per_cluster==1`
        # ou (3, ni, data.shape[1]) si `n_etalons_per_cluster>1`
        """
        n_samples, n_features = self.data.shape
        if self.init_method == "random":
            # Cas backward-compatible : un seul étalon par cluster
            if self.n_etalons_per_cluster == 1:
                indices = self._rng.choice(n_samples, size=self.n_clusters, replace=False)
                centers = self.data[indices].astype(float)
                self.etallons_ = centers
                self.etallons_history_ = []
                self.etallons_history_.append(self.etallons_.copy())
                return self.etallons_
            # Multi-étalons : sélection aléatoire pure comme dans l'article Diday
            # Sélectionner n_clusters * n_etalons_per_cluster points aléatoirement
            total_etalons_needed = self.n_clusters * self.n_etalons_per_cluster
            if total_etalons_needed <= n_samples:
                indices = self._rng.choice(n_samples, size=total_etalons_needed, replace=False)
            else:
                # Si pas assez de points, utiliser replace=True
                indices = self._rng.choice(n_samples, size=total_etalons_needed, replace=True)
            selected_points = self.data[indices].astype(float)
            # Réarranger en (n_clusters, n_etalons_per_cluster, n_features)
            self.etallons_ = selected_points.reshape(self.n_clusters, self.n_etalons_per_cluster, n_features)
            self.etallons_history_ = []
            self.etallons_history_.append(self.etallons_.copy())
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
        # Si ni == 1: comportement historique
        if self.n_etalons_per_cluster == 1:
            self.etallons_ = centers
            self.etallons_history_ = []
            self.etallons_history_.append(self.etallons_.copy())
            return self.etallons_

        # Multi-étalons : construire noyaux à partir d'une assignation préliminaire
        # Effectuer l'assignation préliminaire en utilisant les centres 2D
        # calculés ci-dessus (ne pas appeler `assign_objects()` ici).
        self.etallons_ = centers
        D_prelim = compute_distance_matrix(self.data, centers, metric=self.distance_metric)
        prelim_labels = np.argmin(D_prelim, axis=1)
        multi_etallons = []
        for k in range(self.n_clusters):
            members = np.where(prelim_labels == k)[0]
            if members.size == 0:
                idxs = self._rng.choice(n_samples, size=min(self.n_etalons_per_cluster, n_samples), replace=False)
                selected = self.data[idxs]
            else:
                # Sélectionner les points les plus proches du prototype (centroïde)
                prototype = np.mean(self.data[members], axis=0)
                proto_arr = prototype.reshape(1, -1)
                dists = compute_distance_matrix(self.data[members], proto_arr, metric=self.distance_metric).ravel()
                sorted_idx = np.argsort(dists)
                ni_actual = min(self.n_etalons_per_cluster, members.size)
                chosen_member_idxs = members[sorted_idx[:ni_actual]]
                selected = self.data[chosen_member_idxs]
                if ni_actual < self.n_etalons_per_cluster:
                    padding = np.tile(selected[-1], (self.n_etalons_per_cluster - ni_actual, 1))
                    selected = np.vstack([selected, padding])
            multi_etallons.append(selected.astype(float))
        self.etallons_ = np.array(multi_etallons)
        self.etallons_history_ = []
        self.etallons_history_.append(self.etallons_.copy())
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

        # Supporter deux formes d'étalons : 2D (n_clusters, n_features)
        # ou 3D (n_clusters, ni, n_features). La décision se fait sur
        # la dimension réelle de `self.etallons_` pour éviter des incohérences
        # entre la forme des étalons et la valeur de `self.n_etalons_per_cluster`.
        ndim = getattr(self.etallons_, "ndim", None)
        if ndim == 2:
            D = compute_distance_matrix(self.data, self.etallons_, metric=self.distance_metric)
            # Cacher la dernière matrice de distances calculée
            self._last_distance_matrix = D
            self.labels_ = np.argmin(D, axis=1)
            return self.labels_

        if ndim == 3:
            n_samples, n_features = self.data.shape
            n_clusters_et, ni = self.etallons_.shape[0], self.etallons_.shape[1]
            flat_etallons = self.etallons_.reshape(-1, n_features)
            D_flat = compute_distance_matrix(self.data, flat_etallons, metric=self.distance_metric)
            # Reshape en (n_samples, n_clusters, ni)
            D_reshaped = D_flat.reshape(n_samples, n_clusters_et, ni)
            # Distance minimale par noyau
            D_min_per_noyau = np.min(D_reshaped, axis=2)
            self._last_distance_matrix = D_min_per_noyau
            self.labels_ = np.argmin(D_min_per_noyau, axis=1)
            return self.labels_

        # Si la forme n'est pas attendue, lever une erreur explicite
        raise ValueError("`etallons_` doit être un tableau 2D (n_clusters, n_features) ou 3D (n_clusters, ni, n_features)")

    def update_etallons(self) -> np.ndarray:
        """Recalcule les étalons selon `etallon_method` pour chaque cluster.

                Returns
                -------
                etallons : ndarray
                        Nouveaux étalons calculés.

                        - Si `n_etalons_per_cluster == 1` : tableau 2D de forme
                            `(n_clusters, n_features)` (prototype unique par cluster).
                        - Si `n_etalons_per_cluster > 1` : tableau 3D de forme
                            `(n_clusters, n_etalons_per_cluster, n_features)` représentant
                            les noyaux `E_i` (un ensemble de `ni` points par cluster).

        Notes
        -----
        - `centroid` : moyenne des objets du cluster.
        - `median` : médiane par dimension des objets du cluster (robuste aux outliers).
        - `mode` : valeur la plus fréquente par dimension (utile pour variables discrètes/catégorielles).
        - `medoid` : point du cluster minimisant la somme des distances aux autres points du cluster.
        - Si un cluster est vide, on réinitialise son étalon en prenant un point aléatoire.
        
        Note
        ----
        Pour `n_etalons_per_cluster > 1`, la sélection des `ni` points par
        cluster se fait en choisissant les points les plus proches du prototype
        (centroid/medoid/median/mode selon `etallon_method`). Cette
        stratégie forme des noyaux compacts et voisins, où les étalons d'un
        même cluster sont regroupés autour du centre, conformément à
        l'approche des noyaux multi-étalons.
        """
        # Vérifier l'état
        if self.labels_ is None:
            raise ValueError("`labels_` est None : appeler `assign_objects()` ou `fit()` avant `update_etallons()`")
        if len(self.labels_) != self.data.shape[0]:
            raise ValueError("La longueur de `labels_` doit correspondre au nombre d'échantillons de `data`")

        n_samples, n_features = self.data.shape

        # Cas simple (historique) : un seul étalon par cluster
        if self.n_etalons_per_cluster == 1:
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
                    new_etallons[k] = np.median(cluster_points, axis=0)
                elif self.etallon_method == "mode":
                    mode_res = scipy_mode(cluster_points, axis=0)
                    try:
                        modes = mode_res.mode
                    except Exception:
                        modes = mode_res[0]
                    modes = np.asarray(modes).squeeze()
                    new_etallons[k] = modes
                else:
                    D = pairwise_distances(cluster_points, metric=self.distance_metric)
                    sums = np.sum(D, axis=1)
                    medoid_idx = np.argmin(sums)
                    new_etallons[k] = cluster_points[medoid_idx]
            # Invalider le cache de distances car les étalons viennent de changer
            self._last_distance_matrix = None
            # Assigner les nouveaux étalons
            self.etallons_ = new_etallons
            return self.etallons_

        # Multi-étalons : sélectionner ni points par cluster en choisissant
        # les points les plus proches du prototype (centroid/medoid/median/mode).
        # Cette stratégie forme des noyaux compacts et voisins autour du
        # prototype, conformément à l'approche des noyaux multi-étalons.
        new_etallons = []
        for k in range(self.n_clusters):
            mask = (self.labels_ == k)
            members = np.where(mask)[0]
            if members.size == 0:
                # Cluster vide : ni points aléatoires (ou remplissage si insuffisant)
                idxs = self._rng.choice(n_samples, size=min(self.n_etalons_per_cluster, n_samples), replace=False)
                noyau_k = self.data[idxs].astype(float)
                if noyau_k.shape[0] < self.n_etalons_per_cluster:
                    padding = np.tile(noyau_k[-1], (self.n_etalons_per_cluster - noyau_k.shape[0], 1))
                    noyau_k = np.vstack([noyau_k, padding])
                new_etallons.append(noyau_k)
                continue

            cluster_points = self.data[members]
            ni_actual = min(self.n_etalons_per_cluster, cluster_points.shape[0])
            # Calculer le prototype selon la méthode choisie
            if self.etallon_method == "centroid":
                prototype = np.mean(cluster_points, axis=0)
            elif self.etallon_method == "median":
                prototype = np.median(cluster_points, axis=0)
            elif self.etallon_method == "mode":
                mode_res = scipy_mode(cluster_points, axis=0)
                try:
                    modes = mode_res.mode
                except Exception:
                    modes = mode_res[0]
                prototype = np.asarray(modes).squeeze()
            else:
                # medoid : choisir le point du cluster minimisant la somme des distances
                D_cluster = pairwise_distances(cluster_points, metric=self.distance_metric)
                sums = np.sum(D_cluster, axis=1)
                medoid_idx = np.argmin(sums)
                prototype = cluster_points[medoid_idx]

            # Sélectionner les ni_actual points les plus proches du prototype
            proto_arr = np.asarray(prototype).reshape(1, -1)
            dists_to_proto = compute_distance_matrix(cluster_points, proto_arr, metric=self.distance_metric).ravel()
            sorted_idx = np.argsort(dists_to_proto)
            chosen = sorted_idx[:ni_actual]
            noyau_k = cluster_points[chosen]
            # Padding si nécessaire
            if noyau_k.shape[0] < self.n_etalons_per_cluster:
                padding = np.tile(noyau_k[-1], (self.n_etalons_per_cluster - noyau_k.shape[0], 1))
                noyau_k = np.vstack([noyau_k, padding])
            new_etallons.append(noyau_k.astype(float))

        # Finaliser la structure (n_clusters, ni, n_features)
        new_etallons = np.array(new_etallons)
        self._last_distance_matrix = None
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
        # Note: pour `etallons_` 3D (noyaux), la comparaison est effectuée
        # après un tri déterministe (lexsort) des points du noyau afin de
        # rendre la mesure de déplacement indépendante de l'ordre interne
        # des points dans chaque noyau.
        # Gérer forme 2D (n_clusters, n_features) et 3D (n_clusters, ni, n_features)
        ndim = getattr(self.etallons_, "ndim", 2)
        if ndim == 2:
            shifts = np.linalg.norm(self.etallons_ - old_etallons, axis=1)
            max_shift = np.max(shifts)
            return bool(max_shift <= self.tolerance)

        # Multi-étalons : trier de façon déterministe les points du noyau
        def sort_noyau(noyau):
            # noyau: (ni, n_features) -> trier lignes lexicographiquement
            if noyau.shape[0] == 0:
                return noyau
            keys = tuple(noyau[:, i] for i in range(noyau.shape[1]-1, -1, -1))
            idx = np.lexsort(keys)
            return noyau[idx]

        et_sorted = np.array([sort_noyau(self.etallons_[k]) for k in range(self.n_clusters)])
        old_sorted = np.array([sort_noyau(old_etallons[k]) for k in range(self.n_clusters)])
        # Flatten and compute shifts
        shifts = np.linalg.norm(et_sorted.reshape(-1, self.data.shape[1]) - old_sorted.reshape(-1, self.data.shape[1]), axis=1)
        max_shift = np.max(shifts)
        return bool(max_shift <= self.tolerance)

    def _compute_inertia(self) -> float:
        """Calcule l'inertie : somme des distances (au carré pour certaines métriques) des points à leur étalon.

        La formule utilisée est : sum_i d(x_i, c_{label_i})^2 pour métriques où le carré est approprié,
        sinon sum_i d(x_i, c_{label_i}) pour métriques de divergence.
        """
        # Note: dans le cas multi-étalons (noyaux), l'inertie est calculée en
        # prenant pour chaque point la distance minimale au noyau de son cluster
        # (i.e. distance au point le plus proche du noyau), puis en sommant
        # les carrés de ces distances (ou les distances directement selon la métrique).
        # Réutiliser la matrice des distances calculée lors de l'assignation si disponible
        D = getattr(self, "_last_distance_matrix", None)
        ndim = getattr(self.etallons_, "ndim", 2)
        # Cas 2D historique
        if ndim == 2:
            if D is not None and D.shape == (self.data.shape[0], self.etallons_.shape[0]):
                chosen = D[np.arange(self.data.shape[0]), self.labels_]
                return float(np.sum(chosen ** 2 if self.square_distances else chosen))
            # Sinon calculer explicitement
            D = compute_distance_matrix(self.data, self.etallons_, metric=self.distance_metric)
            chosen = D[np.arange(self.data.shape[0]), self.labels_]
            return float(np.sum(chosen ** 2 if self.square_distances else chosen))

        # Cas multi-étalons (3D) : D (cached) attendu en forme (n_samples, n_clusters)
        if D is not None and D.shape == (self.data.shape[0], self.n_clusters):
            chosen = D[np.arange(self.data.shape[0]), self.labels_]
            return float(np.sum(chosen ** 2 if self.square_distances else chosen))

        # Recalculer si cache invalide : aplatir les étalons et calculer min distance par noyau
        n_samples, n_features = self.data.shape
        n_clusters_et, ni = self.etallons_.shape[0], self.etallons_.shape[1]
        flat_etallons = self.etallons_.reshape(-1, n_features)
        D_flat = compute_distance_matrix(self.data, flat_etallons, metric=self.distance_metric)
        D_reshaped = D_flat.reshape(n_samples, n_clusters_et, ni)
        D_min = np.min(D_reshaped, axis=2)
        chosen = D_min[np.arange(self.data.shape[0]), self.labels_]
        return float(np.sum(chosen ** 2 if self.square_distances else chosen))

    def _compute_homogeneite_per_cluster(self) -> np.ndarray:
        """Calcule l'homogénéité Si pour chaque cluster i.

        Si est définie comme la somme des distances (au carré pour certaines métriques)
        minimales de chaque point x dans Ci à son étalon le plus proche dans le noyau Ei.
        Pour métriques de divergence, utilise sum(R) au lieu de sum(R²).

        Pour optimiser les performances, réutilise la matrice de distances
        mise en cache `self._last_distance_matrix` (de forme (n_samples, n_clusters))
        calculée lors de `assign_objects()`. Si le cache est invalide ou absent,
        retombe sur le calcul explicite via `compute_distance_matrix`.
        """
        D = getattr(self, '_last_distance_matrix', None)
        use_cache = D is not None and D.shape == (self.data.shape[0], self.n_clusters)
        si_list = []
        for k in range(self.n_clusters):
            mask = self.labels_ == k
            if not np.any(mask):
                si_list.append(0.0)
                continue
            if use_cache:
                dists_k = D[mask, k]
                Si = np.sum(dists_k ** 2 if self.square_distances else dists_k)
            else:
                X_k = self.data[mask]
                etalons_k = self.etallons_[k]
                if self.etallons_.ndim == 2:
                    etalons_k = etalons_k.reshape(1, -1)
                dists = compute_distance_matrix(X_k, etalons_k, self.distance_metric)
                min_dists = np.min(dists, axis=1)
                Si = np.sum(min_dists ** 2 if self.square_distances else min_dists)
            si_list.append(Si)
        return np.array(si_list)

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
            self.etallons_history_.append(self.etallons_.copy())

            self.n_iter_ = it + 1
            if self.check_convergence(old_etallons):
                break

        # Résultats finaux
        self.inertia_ = self.history_[-1] if len(self.history_) > 0 else None
        self.cluster_sizes_ = np.bincount(self.labels_, minlength=self.n_clusters)
        self.homogeneite_per_cluster_ = self._compute_homogeneite_per_cluster()
        self.S_total_ = np.sum(self.homogeneite_per_cluster_)
        self.representative_noyau_ = np.array([np.mean(self.etallons_[k], axis=0) if self.n_etalons_per_cluster > 1 else self.etallons_[k] for k in range(self.n_clusters)])
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
        
        Note
        ----
        Si `n_etalons_per_cluster > 1` (structure 3D des `etallons_`), la
        prédiction assigne chaque observation au cluster dont le noyau
        contient le point le plus proche (min sur les `ni` points du noyau),
        et non pas au prototype unique.

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
        ndim = getattr(self.etallons_, "ndim", 2)
        if ndim == 2:
            D = compute_distance_matrix(new_data, self.etallons_, metric=self.distance_metric)
            predicted_labels = np.argmin(D, axis=1)
            return predicted_labels

        # Multi-étalons : aplatir les étalons, calculer distances, puis min par noyau
        n_features = self.data.shape[1]
        n_clusters_et, ni = self.etallons_.shape[0], self.etallons_.shape[1]
        flat_etallons = self.etallons_.reshape(-1, n_features)
        D_flat = compute_distance_matrix(new_data, flat_etallons, metric=self.distance_metric)
        D_reshaped = D_flat.reshape(new_data.shape[0], n_clusters_et, ni)
        D_min = np.min(D_reshaped, axis=2)
        predicted_labels = np.argmin(D_min, axis=1)
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

    def get_cluster_sizes(self):
        """Retourne les tailles des clusters.

        Returns
        -------
        ndarray, shape (n_clusters,)
            Nombre de points dans chaque cluster.

        Raises
        ------
        ValueError
            Si le modèle n'a pas été entraîné.
        """
        if not hasattr(self, 'cluster_sizes_') or self.cluster_sizes_ is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appeler `fit()` avant `get_cluster_sizes()`.") 
        return self.cluster_sizes_

    def get_homogeneite_per_cluster(self):
        """Retourne l'homogénéité par cluster.

        Returns
        -------
        ndarray, shape (n_clusters,)
            Homogénéité Si pour chaque cluster.

        Raises
        ------
        ValueError
            Si le modèle n'a pas été entraîné.
        """
        if not hasattr(self, 'homogeneite_per_cluster_') or self.homogeneite_per_cluster_ is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appeler `fit()` avant `get_homogeneite_per_cluster()`.") 
        return self.homogeneite_per_cluster_

    def get_S_total(self):
        """Retourne la somme totale des homogénéités.

        Returns
        -------
        float
            Somme des Si, équivalent à l'inertie.

        Raises
        ------
        ValueError
            Si le modèle n'a pas été entraîné.
        """
        if not hasattr(self, 'S_total_') or self.S_total_ is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appeler `fit()` avant `get_S_total()`.") 
        return self.S_total_

    def get_representative_noyaux(self):
        """Retourne les noyaux représentatifs.

        Returns
        -------
        ndarray, shape (n_clusters, n_features)
            Centroïde de chaque noyau ou étalon unique.

        Raises
        ------
        ValueError
            Si le modèle n'a pas été entraîné.
        """
        if not hasattr(self, 'representative_noyau_') or self.representative_noyau_ is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appeler `fit()` avant `get_representative_noyaux()`.") 
        return self.representative_noyau_