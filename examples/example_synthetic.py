"""
Exemple complet : Utilisation de la méthode des Nuées Dynamiques sur données synthétiques.

Ce script démontre l'utilisation de la classe NuéesDynamique sur des données générées
(2D et 3D), avec visualisations, calcul de métriques de qualité (silhouette, Davies-Bouldin)
et comparaison avec l'algorithme K-means de scikit-learn.

Les exemples illustrent :
- Initialisation (random vs kmeans++)
- Méthodes d'étalons (centroïde vs médoïde)
- Différentes métriques de distance
- Évaluation quantitative de la qualité du clustering
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


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DÉMONSTRATION DE LA MÉTHODE DES NUÉES DYNAMIQUES")
    print("Données synthétiques avec visualisations et comparaison K-means")
    print("=" * 70)

    # Exécuter les trois exemples
    example_2d_clustering()
    example_3d_clustering()
    comparison_with_kmeans()

    print("\n" + "=" * 70)
    print("Démonstration terminée !")
    print("=" * 70 + "\n")
