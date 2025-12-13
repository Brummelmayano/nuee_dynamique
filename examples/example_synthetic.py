"""
Exemple complet : Utilisation de la méthode des Nuées Dynamiques sur données synthétiques.

Ce script démontre l'utilisation de la classe NuéesDynamique sur des données générées
(2D et 3D), avec visualisations, calcul de métriques de qualité (silhouette, Davies-Bouldin)
et comparaison avec l'algorithme K-means de scikit-learn.

Les exemples illustrent :
- Quatre méthodes d'initialisation : centroïde unique (kmeans++), ensemble aléatoire (random), distribution probabiliste (GMM), axes factoriels (PCA)
- Méthodes d'étalons (centroïde vs médoïde)
- Différentes métriques de distance
- Évaluation quantitative de la qualité du clustering
- Noyaux multi-étalons (Diday IV.1) pour clusters complexes
"""

import numpy as np
import matplotlib.pyplot as plt

# Imports depuis le package nuees_dynamiques
from nuees_dynamiques import (
    NuéesDynamique,
    generate_synthetic_data,
    plot_clusters,
    compute_silhouette,
    compute_davies_bouldin,
)

# Imports supplémentaires
from sklearn.cluster import KMeans


def example_2d_clustering():
    """
    Exemple 1 : Clustering 2D avec Nuées Dynamiques.
    
    Génère 300 points répartis en 3 clusters gaussiens et effectue le clustering
    avec la méthode des Nuées Dynamiques.
    """
    print("\n" + "=" * 70)
    print("EXEMPLE 1 : CLUSTERING 2D - DONNÉES SYNTHÉTIQUES")
    print("=" * 70)

    # Génération de 300 points répartis en 3 clusters gaussiens
    # random_state=42 pour reproductibilité
    X, y_true = generate_synthetic_data(
        n_samples=300,
        n_features=2,
        n_clusters=3,
        cluster_std=1.0,
        random_state=42,
    )
    print(f"\n✓ Données générées : {X.shape[0]} échantillons, {X.shape[1]} features, {len(np.unique(y_true))} clusters réels")

    # Création et entraînement du modèle
    # init_method='kmeans++' pour une initialisation de meilleure qualité
    nd = NuéesDynamique(
        data=X,
        n_clusters=3,
        distance_metric="euclidean",
        init_method="kmeans++",
        etallon_method="centroid",
        max_iterations=100,
        tolerance=1e-4,
        random_state=42,
    )

    nd.fit()
    print(f"\n✓ Clustering effectué en {nd.n_iter_} itérations")
    print(f"  - Inertie finale : {nd.get_inertia():.2f}")

    # Calcul des métriques de qualité
    silhouette = compute_silhouette(X, nd.labels_)
    davies_bouldin = compute_davies_bouldin(X, nd.labels_)

    print(f"\n✓ Métriques de qualité :")
    print(f"  - Silhouette score : {silhouette:.3f}")
    print(f"    → Interprétation : Score proche de 1 indique une très bonne séparation des clusters")
    print(f"  - Davies-Bouldin index : {davies_bouldin:.3f}")
    print(f"    → Interprétation : Indice faible (< 1) indique une bonne séparation")

    # Visualisation
    fig = plot_clusters(X, nd.labels_, nd.etallons_, title="Nuées Dynamiques - Données 2D")
    plt.tight_layout()
    plt.show()


def example_3d_clustering():
    """
    Exemple 2 : Clustering 3D avec Nuées Dynamiques (médoïde).
    
    Génère 400 points répartis en 4 clusters gaussiens 3D et effectue le clustering
    avec la méthode médoïde pour l'étalon.
    
    Note pédagogique : Cet exemple utilise init_method='random' et etallon_method='medoid'
    pour démontrer la variante médoïde (contraste avec l'Exemple 1 qui utilise centroïde).
    """
    print("\n" + "=" * 70)
    print("EXEMPLE 2 : CLUSTERING 3D - INITIALISATION MÉDOÏDE")
    print("=" * 70)

    # Génération de 400 points 3D répartis en 4 clusters
    X, y_true = generate_synthetic_data(
        n_samples=400,
        n_features=3,
        n_clusters=4,
        cluster_std=0.8,
        random_state=123,
    )
    print(f"\n✓ Données générées : {X.shape[0]} échantillons, {X.shape[1]} features, {len(np.unique(y_true))} clusters réels")

    # Création et entraînement avec méthode médoïde
    # CHOIX PÉDAGOGIQUE : init_method='random' et etallon_method='medoid'
    # Cette configuration différente de l'Exemple 1 illustre la flexibilité de Nuées Dynamiques
    # - init_method='random' : initialisation aléatoire simple
    # - etallon_method='medoid' : l'étalon de chaque cluster est le point le plus central (robustesse)
    nd = NuéesDynamique(
        data=X,
        n_clusters=4,
        distance_metric="euclidean",
        init_method="random",
        etallon_method="medoid",
        max_iterations=100,
        tolerance=1e-4,
        random_state=123,
    )

    nd.fit()
    print(f"\n✓ Clustering effectué en {nd.n_iter_} itérations")
    print(f"  - Inertie finale : {nd.get_inertia():.2f}")
    print(f"\n  Note : Médoïde vs Centroïde")
    print(f"  - Centroïde : moyenne arithmétique des points du cluster")
    print(f"  - Médoïde : point réel du cluster minimisant la somme des distances")
    print(f"  - Avantage du médoïde : moins sensible aux points aberrants (outliers)")

    # Calcul des métriques
    silhouette = compute_silhouette(X, nd.labels_)
    davies_bouldin = compute_davies_bouldin(X, nd.labels_)

    print(f"\n✓ Métriques de qualité :")
    print(f"  - Silhouette score : {silhouette:.3f}")
    print(f"  - Davies-Bouldin index : {davies_bouldin:.3f}")

    # Visualisation 3D
    fig = plot_clusters(X, nd.labels_, nd.etallons_, title="Nuées Dynamiques - Données 3D (Médoïde)")
    plt.tight_layout()
    plt.show()


def comparison_with_kmeans():
    """
    Exemple 3 : Comparaison Nuées Dynamiques vs K-means.
    
    Compare les résultats des deux algorithmes sur les mêmes données 2D.
    
    Note pédagogique : Cet exemple utilise init_method='kmeans++' pour Nuées Dynamiques
    pour refléter l'initialisation standard de KMeans, permettant une comparaison équitable.
    """
    print("\n" + "=" * 70)
    print("EXEMPLE 3 : COMPARAISON - NUÉES DYNAMIQUES vs K-MEANS")
    print("=" * 70)

    # Génération des données
    X, y_true = generate_synthetic_data(
        n_samples=300,
        n_features=2,
        n_clusters=3,
        cluster_std=1.0,
        random_state=42,
    )

    # Nuées Dynamiques
    # CHOIX PÉDAGOGIQUE : init_method='kmeans++' pour comparaison équitable
    # L'Exemple 1 utilise aussi 'kmeans++' (initialisation de haute qualité)
    # L'Exemple 2 utilise 'random' avec 'medoid' pour montrer une variante
    nd = NuéesDynamique(
        data=X,
        n_clusters=3,
        distance_metric="euclidean",
        init_method="kmeans++",
        random_state=42,
    )
    nd.fit()

    # K-means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)

    # Comparaison
    print(f"\n✓ Résultats Nuées Dynamiques :")
    print(f"  - Inertie : {nd.get_inertia():.2f}")
    print(f"  - Itérations : {nd.n_iter_}")
    print(f"  - Silhouette : {compute_silhouette(X, nd.labels_):.3f}")
    print(f"  - Davies-Bouldin : {compute_davies_bouldin(X, nd.labels_):.3f}")

    print(f"\n✓ Résultats K-means :")
    print(f"  - Inertie : {kmeans.inertia_:.2f}")
    print(f"  - Itérations : {kmeans.n_iter_}")
    print(f"  - Silhouette : {compute_silhouette(X, kmeans.labels_):.3f}")
    print(f"  - Davies-Bouldin : {compute_davies_bouldin(X, kmeans.labels_):.3f}")

    # Analyse comparative
    print(f"\n✓ Analyse comparative :")
    inertia_diff = abs(nd.get_inertia() - kmeans.inertia_)
    print(f"  - Différence d'inertie : {inertia_diff:.2f}")
    print(f"  - Similarité des résultats : Les deux algorithmes convergent vers des solutions similaires")
    print(f"  - Avantages Nuées Dynamiques :")
    print(f"    • Flexibilité : support de multiples métriques de distance (euclidienne, Manhattan, Minkowski, etc.)")
    print(f"    • Médoïdes : robustesse accrue face aux points aberrants")
    print(f"    • Algorithme itératif convergent avec critères de convergence personnalisables")
    print(f"  - Avantages K-means :")
    print(f"    • Simplicité et interprétabilité")
    print(f"    • Performance bien optimisée dans scikit-learn")

    # Visualisation côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Nuées Dynamiques
    ax = axes[0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=nd.labels_, cmap="viridis", alpha=0.6, s=50)
    ax.scatter(
        nd.etallons_[:, 0],
        nd.etallons_[:, 1],
        c="red",
        marker="X",
        s=200,
        edgecolors="black",
        linewidths=2,
        label="Étalons",
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Nuées Dynamiques")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # K-means
    ax = axes[1]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="viridis", alpha=0.6, s=50)
    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        c="red",
        marker="X",
        s=200,
        edgecolors="black",
        linewidths=2,
        label="Centroïdes",
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("K-means")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Comparaison : Nuées Dynamiques vs K-means", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def example_multinoyau_diday():
    """
    Exemple avancé : Démonstration des noyaux multi-étalons (Diday IV.1).
    
    Cet exemple illustre la différence entre un clustering avec ni=1 (prototype unique)
    et ni>1 (noyau multi-étalon) sur des clusters allongés.
    
    Les noyaux multi-étalons capturent mieux les structures complexes et non-sphériques,
    comme décrit dans Diday (1971, section IV.1).
    """
    print("\n" + "=" * 70)
    print("EXEMPLE AVANCÉ : NOYAUX MULTI-ÉTALONS (DIDAY IV.1)")
    print("=" * 70)

    # Génération de données synthétiques avec clusters allongés
    # On crée 2 clusters gaussiens puis on applique une transformation linéaire pour les étirer
    X_base, y_true = generate_synthetic_data(
        n_samples=200,
        n_features=2,
        n_clusters=2,
        cluster_std=0.8,
        random_state=42,
    )

    # Transformation linéaire pour étirer les clusters (anisotrope)
    # Matrice de rotation + scaling
    theta = np.pi / 4  # 45 degrés
    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    scaling = np.array([[3.0, 0], [0, 0.8]])  # Étirer horizontalement
    transform = rotation @ scaling

    X = X_base @ transform.T
    print(f"\n✓ Dataset généré : {X.shape[0]} points, {X.shape[1]} features")
    print(f"  - 2 clusters allongés (covariance anisotrope)")
    print(f"  - Clusters étirés pour montrer l'intérêt des noyaux multi-étalons")

    # --- MODÈLE 1 : ni = 1 (prototype unique) ---
    # Correspond au mode "Centroïde unique (similaire KMeans)" dans l'interface Streamlit
    nd_single = NuéesDynamique(
        data=X,
        n_clusters=2,
        distance_metric='euclidean',
        init_method='kmeans++',  # "Centroïde unique (similaire KMeans)"
        etallon_method='centroid',
        n_etalons_per_cluster=1,  # Prototype unique
        random_state=0,
    )
    nd_single.fit()
    silhouette_single = compute_silhouette(X, nd_single.labels_)
    db_single = compute_davies_bouldin(X, nd_single.labels_)

    print(f"\n✓ Modèle 1 : ni = 1 (prototype unique)")
    print(f"  - Étalons shape : {nd_single.etallons_.shape}")
    print(f"  - Itérations : {nd_single.n_iter_}")
    print(f"  - Silhouette : {silhouette_single:.3f}")
    print(f"  - Davies-Bouldin : {db_single:.3f}")

    # --- MODÈLE 2 : ni = 20 (noyau multi-étalon) ---
    # Calculer le max_ni possible
    max_ni = max(1, X.shape[0] // 2)  # 200 // 2 = 100
    ni_multi = min(20, max_ni)  # Utiliser min(20, 100) = 20

    nd_multi = NuéesDynamique(
        data=X,
        n_clusters=2,
        distance_metric='euclidean',
        init_method='kmeans++',
        etallon_method='centroid',
        n_etalons_per_cluster=ni_multi,  # Noyau multi-étalon
        random_state=0,
    )
    nd_multi.fit()
    silhouette_multi = compute_silhouette(X, nd_multi.labels_)
    db_multi = compute_davies_bouldin(X, nd_multi.labels_)

    print(f"\n✓ Modèle 2 : ni = {ni_multi} (noyau multi-étalon, Diday IV.1)")
    print(f"  - Étalons shape : {nd_multi.etallons_.shape}")
    print(f"  - Itérations : {nd_multi.n_iter_}")
    print(f"  - Silhouette : {silhouette_multi:.3f}")
    print(f"  - Davies-Bouldin : {db_multi:.3f}")

    # Comparaison
    print(f"\n✓ Comparaison :")
    print(f"  - Δ Silhouette (multi - single) : {silhouette_multi - silhouette_single:+.3f}")
    print(f"    → Positif : les noyaux multi-étalons donnent une meilleure cohésion/séparation")
    print(f"  - Δ Davies-Bouldin (multi - single) : {db_multi - db_single:+.3f}")
    print(f"    → Négatif : indice plus faible est préférable (clusters plus distincts)")

    # Visualisation en 2D
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Subplot 1 : ni = 1 ---
    ax = axes[0]
    scatter1 = ax.scatter(X[:, 0], X[:, 1], c=nd_single.labels_, cmap='viridis', alpha=0.6, s=40)
    # Afficher les étalons (croix rouge, unique par cluster)
    ax.scatter(
        nd_single.etallons_[:, 0],
        nd_single.etallons_[:, 1],
        c='red',
        marker='X',
        s=200,
        edgecolors='black',
        linewidths=2,
        label=f'Étalons (ni=1)',
    )
    ax.set_title(f'ni=1 : Prototype unique\n(Silhouette={silhouette_single:.3f}, DB={db_single:.3f})')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Subplot 2 : ni = multi ---
    ax = axes[1]
    scatter2 = ax.scatter(X[:, 0], X[:, 1], c=nd_multi.labels_, cmap='viridis', alpha=0.6, s=40)
    # Afficher les noyaux multi-étalons (étoiles rouges, multiples par cluster)
    # Aplatir les noyaux pour l'affichage
    flat_etallons = nd_multi.etallons_.reshape(-1, nd_multi.etallons_.shape[-1])
    ax.scatter(
        flat_etallons[:, 0],
        flat_etallons[:, 1],
        c='red',
        marker='*',
        s=150,
        edgecolors='black',
        linewidths=1,
        label=f'Noyaux multi-étalons (ni={ni_multi})',
        alpha=0.8,
    )
    ax.set_title(f'ni={ni_multi} : Noyau multi-étalon (Diday IV.1)\n(Silhouette={silhouette_multi:.3f}, DB={db_multi:.3f})')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n✓ Observations :")
    print(f"  - Les clusters sont allongés, ce qui pose problème au prototype unique (ni=1)")
    print(f"  - Le noyau multi-étalon (ni={ni_multi}) distribue ses points le long de l'allongement")
    print(f"  - Cette distribution capture mieux la structure géométrique des clusters")
    print(f"  - C'est exactement le phénomène décrit par Diday (1971, IV.1) : les noyaux multi-étalons")
    print(f"    permettent une meilleure approximation de clusters non-sphériques.")


def example_gmm_initialization():
    """
    Exemple : Initialisation par modèle de mélange gaussien (GMM).

    Cet exemple démontre l'utilisation de init_method='gmm' qui correspond
    au mode "Distribution probabiliste (GMM)" dans l'interface Streamlit.
    Cette méthode est particulièrement adaptée aux clusters elliptiques.
    """
    print("\n" + "=" * 70)
    print("EXEMPLE : INITIALISATION GMM - DISTRIBUTION PROBABILISTE")
    print("=" * 70)

    # Génération de données avec clusters elliptiques
    X, y_true = generate_synthetic_data(
        n_samples=300,
        n_features=2,
        n_clusters=3,
        cluster_std=1.5,
        random_state=42,
    )

    # Appliquer une transformation pour créer des clusters elliptiques
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Transformation affine pour créer des ellipses
    transform = np.array([[2.0, 0.5], [0.5, 0.8]])
    X_elliptical = X_scaled @ transform

    print(f"\n✓ Données générées : {X_elliptical.shape[0]} échantillons, {X_elliptical.shape[1]} features")
    print(f"  - Clusters elliptiques (covariance non-sphérique)")

    # Clustering avec initialisation GMM
    nd_gmm = NuéesDynamique(
        data=X_elliptical,
        n_clusters=3,
        distance_metric="euclidean",
        init_method="gmm",  # Correspond à "Distribution probabiliste (GMM)"
        gmm_init_mode="means",  # Utilise les moyennes des gaussiennes
        etallon_method="centroid",
        random_state=42,
    )
    nd_gmm.fit()

    print(f"\n✓ Clustering avec init_method='gmm' (Distribution probabiliste)")
    print(f"  - Itérations : {nd_gmm.n_iter_}")
    print(f"  - Inertie : {nd_gmm.get_inertia():.2f}")
    print(f"  - Silhouette : {compute_silhouette(X_elliptical, nd_gmm.labels_):.3f}")
    print(f"  - Davies-Bouldin : {compute_davies_bouldin(X_elliptical, nd_gmm.labels_):.3f}")

    # Comparaison avec kmeans++
    nd_kmeans = NuéesDynamique(
        data=X_elliptical,
        n_clusters=3,
        distance_metric="euclidean",
        init_method="kmeans++",  # Centroïde unique
        etallon_method="centroid",
        random_state=42,
    )
    nd_kmeans.fit()

    print(f"\n✓ Comparaison avec init_method='kmeans++' (Centroïde unique)")
    print(f"  - Itérations : {nd_kmeans.n_iter_}")
    print(f"  - Inertie : {nd_kmeans.get_inertia():.2f}")
    print(f"  - Silhouette : {compute_silhouette(X_elliptical, nd_kmeans.labels_):.3f}")
    print(f"  - Davies-Bouldin : {compute_davies_bouldin(X_elliptical, nd_kmeans.labels_):.3f}")

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # GMM
    ax = axes[0]
    scatter = ax.scatter(X_elliptical[:, 0], X_elliptical[:, 1], c=nd_gmm.labels_, cmap="viridis", alpha=0.6, s=50)
    ax.scatter(
        nd_gmm.etallons_[:, 0],
        nd_gmm.etallons_[:, 1],
        c="red",
        marker="X",
        s=200,
        edgecolors="black",
        linewidths=2,
        label="Étalons GMM",
    )
    ax.set_title("Initialisation GMM\n(Distribution probabiliste)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # K-means++
    ax = axes[1]
    scatter = ax.scatter(X_elliptical[:, 0], X_elliptical[:, 1], c=nd_kmeans.labels_, cmap="viridis", alpha=0.6, s=50)
    ax.scatter(
        nd_kmeans.etallons_[:, 0],
        nd_kmeans.etallons_[:, 1],
        c="red",
        marker="X",
        s=200,
        edgecolors="black",
        linewidths=2,
        label="Étalons K-means++",
    )
    ax.set_title("Initialisation K-means++\n(Centroïde unique)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Comparaison : Initialisation GMM vs K-means++ sur clusters elliptiques", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print(f"\n✓ Observations :")
    print(f"  - Les données ont des clusters elliptiques (covariance anisotrope)")
    print(f"  - L'initialisation GMM capture mieux la structure probabiliste des données")
    print(f"  - Cela correspond au mode 'Distribution probabiliste (GMM)' dans l'interface Streamlit")


def example_pca_initialization():
    """
    Exemple : Initialisation par axes factoriels (ACP).

    Cet exemple démontre l'utilisation de init_method='pca' qui correspond
    au mode "Axes factoriels (ACP)" dans l'interface Streamlit.
    Cette méthode est déterministe et exploite la variance maximale.
    """
    print("\n" + "=" * 70)
    print("EXEMPLE : INITIALISATION PCA - AXES FACTORIELS")
    print("=" * 70)

    # Génération de données avec structure linéaire
    X, y_true = generate_synthetic_data(
        n_samples=300,
        n_features=4,  # Données en haute dimension
        n_clusters=3,
        cluster_std=1.0,
        random_state=42,
    )

    print(f"\n✓ Données générées : {X.shape[0]} échantillons, {X.shape[1]} features")
    print(f"  - Données 4D avec structure sous-jacente")

    # Clustering avec initialisation PCA
    nd_pca = NuéesDynamique(
        data=X,
        n_clusters=3,
        distance_metric="euclidean",
        init_method="pca",  # Correspond à "Axes factoriels (ACP)"
        pca_n_components=2,  # Utiliser 2 composantes principales
        etallon_method="centroid",
        random_state=42,
    )
    nd_pca.fit()

    print(f"\n✓ Clustering avec init_method='pca' (Axes factoriels)")
    print(f"  - Composantes PCA utilisées : 2")
    print(f"  - Itérations : {nd_pca.n_iter_}")
    print(f"  - Inertie : {nd_pca.get_inertia():.2f}")
    print(f"  - Silhouette : {compute_silhouette(X, nd_pca.labels_):.3f}")
    print(f"  - Davies-Bouldin : {compute_davies_bouldin(X, nd_pca.labels_):.3f}")

    # Comparaison avec random
    nd_random = NuéesDynamique(
        data=X,
        n_clusters=3,
        distance_metric="euclidean",
        init_method="random",  # Ensemble de points aléatoires
        etallon_method="centroid",
        random_state=42,
    )
    nd_random.fit()

    print(f"\n✓ Comparaison avec init_method='random' (Ensemble aléatoire)")
    print(f"  - Itérations : {nd_random.n_iter_}")
    print(f"  - Inertie : {nd_random.get_inertia():.2f}")
    print(f"  - Silhouette : {compute_silhouette(X, nd_random.labels_):.3f}")
    print(f"  - Davies-Bouldin : {compute_davies_bouldin(X, nd_random.labels_):.3f}")

    # Visualisation en 2D (projection PCA)
    from sklearn.decomposition import PCA
    pca_viz = PCA(n_components=2)
    X_pca = pca_viz.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PCA
    ax = axes[0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=nd_pca.labels_, cmap="viridis", alpha=0.6, s=50)
    # Projeter les étalons dans l'espace PCA pour la visualisation
    etallons_pca = pca_viz.transform(nd_pca.etallons_)
    ax.scatter(
        etallons_pca[:, 0],
        etallons_pca[:, 1],
        c="red",
        marker="X",
        s=200,
        edgecolors="black",
        linewidths=2,
        label="Étalons PCA",
    )
    ax.set_title("Initialisation PCA\n(Axes factoriels)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Random
    ax = axes[1]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=nd_random.labels_, cmap="viridis", alpha=0.6, s=50)
    etallons_random_pca = pca_viz.transform(nd_random.etallons_)
    ax.scatter(
        etallons_random_pca[:, 0],
        etallons_random_pca[:, 1],
        c="red",
        marker="X",
        s=200,
        edgecolors="black",
        linewidths=2,
        label="Étalons aléatoires",
    )
    ax.set_title("Initialisation aléatoire\n(Ensemble de points)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Comparaison : Initialisation PCA vs Aléatoire sur données 4D", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print(f"\n✓ Observations :")
    print(f"  - Les données sont en 4D, visualisation en 2D via PCA")
    print(f"  - L'initialisation PCA exploite la structure linéaire des données")
    print(f"  - Méthode déterministe : résultats reproductibles")
    print(f"  - Cela correspond au mode 'Axes factoriels (ACP)' dans l'interface Streamlit")


def example_synthetic_multinoyau_3d():
    """
    Exemple avancé : Test multi-noyaux sur données synthétiques 3D natives.
    
    Cet exemple démontre l'utilisation des noyaux multi-étalons sur des données 3D générées
    synthétiquement, sans projection (visualisation native 3D). Il compare un clustering
    avec ni=1 (prototype unique) et ni=25 (noyau multi-étalon) pour illustrer comment
    les noyaux capturent la structure 3D des clusters.
    """
    print("\n" + "=" * 70)
    print("EXEMPLE AVANCÉ : MULTI-NOYAU SUR DONNÉES SYNTHÉTIQUES 3D NATIVES")
    print("=" * 70)

    # Génération de données synthétiques 3D
    X, y_true = generate_synthetic_data(
        n_samples=400,
        n_features=3,
        n_clusters=4,
        cluster_std=0.9,
        random_state=99,
    )
    print(f"\n✓ Données générées : {X.shape[0]} échantillons, {X.shape[1]} features, {len(np.unique(y_true))} clusters réels")
    print(f"  - Données 3D natives (pas de projection PCA)")

    # Calculer max_ni = 400 // 4 = 100, choisir ni=25
    max_ni = X.shape[0] // 4  # 100
    ni_multi = 25

    # --- MODÈLE 1 : ni = 1 (prototype unique) ---
    nd_single = NuéesDynamique(
        data=X,
        n_clusters=4,
        n_etalons_per_cluster=1,
        distance_metric='manhattan',
        init_method='random',
        etallon_method='centroid',
        random_state=99,
    )
    nd_single.fit()
    silhouette_single = compute_silhouette(X, nd_single.labels_)
    db_single = compute_davies_bouldin(X, nd_single.labels_)

    print(f"\n✓ Modèle 1 : ni = 1 (prototype unique)")
    print(f"  - Étalons shape : {nd_single.etallons_.shape}")
    print(f"  - Itérations : {nd_single.n_iter_}")
    print(f"  - Silhouette : {silhouette_single:.3f}")
    print(f"  - Davies-Bouldin : {db_single:.3f}")

    # --- MODÈLE 2 : ni = 25 (noyau multi-étalon) ---
    nd_multi = NuéesDynamique(
        data=X,
        n_clusters=4,
        n_etalons_per_cluster=ni_multi,
        distance_metric='manhattan',
        init_method='random',
        etallon_method='centroid',
        random_state=99,
    )
    nd_multi.fit()
    silhouette_multi = compute_silhouette(X, nd_multi.labels_)
    db_multi = compute_davies_bouldin(X, nd_multi.labels_)

    print(f"\n✓ Modèle 2 : ni = {ni_multi} (noyau multi-étalon)")
    print(f"  - Étalons shape : {nd_multi.etallons_.shape}")
    print(f"  - Itérations : {nd_multi.n_iter_}")
    print(f"  - Silhouette : {silhouette_multi:.3f}")
    print(f"  - Davies-Bouldin : {db_multi:.3f}")

    # Comparaison
    print(f"\n✓ Comparaison ni=25 vs ni=1 :")
    print(f"  - Δ Silhouette (multi - single) : {silhouette_multi - silhouette_single:+.3f}")
    print(f"  - Δ Davies-Bouldin (multi - single) : {db_multi - db_single:+.3f}")
    print(f"  - Interprétation : Les noyaux multi-étalons capturent mieux la structure 3D,")
    print(f"    permettant une meilleure approximation des clusters dans l'espace natif.")

    # Visualisation 3D native
    fig = plt.figure(figsize=(14, 6))
    colors = ['blue', 'green', 'red', 'purple']

    # --- Subplot 1 : ni = 1 ---
    ax1 = fig.add_subplot(121, projection='3d')
    for k in range(4):
        mask = nd_single.labels_ == k
        ax1.scatter(X[mask, 0], X[mask, 1], X[mask, 2], c=colors[k], alpha=0.6, s=40, label=f'Cluster {k}')
    # Étalons uniques
    ax1.scatter(
        nd_single.etallons_[:, 0],
        nd_single.etallons_[:, 1],
        nd_single.etallons_[:, 2],
        c='black',
        marker='X',
        s=200,
        edgecolors='white',
        linewidths=2,
        label='Étalons (ni=1)',
    )
    ax1.set_title(f'ni=1 : Prototype unique\n(Silhouette={silhouette_single:.3f}, DB={db_single:.3f})')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Feature 3')
    ax1.legend()

    # --- Subplot 2 : ni = 25 ---
    ax2 = fig.add_subplot(122, projection='3d')
    for k in range(4):
        mask = nd_multi.labels_ == k
        ax2.scatter(X[mask, 0], X[mask, 1], X[mask, 2], c=colors[k], alpha=0.6, s=40, label=f'Cluster {k}')
    # Noyaux multi-étalons : flatten 4×25×3 → 100×3
    flat_etallons = nd_multi.etallons_.reshape(-1, 3)
    # Assigner couleurs par cluster pour les étoiles
    etallon_colors = []
    for k in range(4):
        etallon_colors.extend([colors[k]] * ni_multi)
    ax2.scatter(
        flat_etallons[:, 0],
        flat_etallons[:, 1],
        flat_etallons[:, 2],
        c=etallon_colors,
        marker='*',
        s=150,
        edgecolors='black',
        linewidths=1,
        label=f'Noyaux multi-étalons (ni={ni_multi})',
        alpha=0.8,
    )
    ax2.set_title(f'ni={ni_multi} : Noyau multi-étalon\n(Silhouette={silhouette_multi:.3f}, DB={db_multi:.3f})')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_zlabel('Feature 3')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(f"\n✓ Observations :")
    print(f"  - Visualisation native 3D sans projection PCA.")
    print(f"  - Les noyaux multi-étalons (ni={ni_multi}) distribuent les points dans l'espace 3D,")
    print(f"    capturant la structure volumétrique des clusters.")
    print(f"  - Comparé à ni=1, ni={ni_multi} offre une meilleure représentation pour des données 3D complexes.")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DÉMONSTRATION DE LA MÉTHODE DES NUÉES DYNAMIQUES")
    print("Données synthétiques avec visualisations et comparaison K-means")
    print("=" * 70)

    # Exécuter les exemples
    example_2d_clustering()
    example_3d_clustering()
    comparison_with_kmeans()
    example_multinoyau_diday()  # Noyaux multi-étalons
    example_gmm_initialization()  # Initialisation GMM
    example_pca_initialization()  # Initialisation PCA
    example_synthetic_multinoyau_3d()  # Multi-noyau 3D

    print("\n" + "=" * 70)
    print("Démonstration terminée !")
    print("=" * 70 + "\n")