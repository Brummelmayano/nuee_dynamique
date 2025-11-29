"""
Exemple complet : Utilisation de la méthode des Nuées Dynamiques sur datasets réels.

Ce script démontre l'utilisation de la classe NuéesDynamique sur des datasets réels
(Iris et Wine), avec évaluation quantitative des résultats et comparaison avec K-means.

Les exemples illustrent :
- Application à des données réelles multi-dimensionnelles
- Impact du choix de la métrique de distance (euclidienne vs Manhattan)
- Utilisation de PCA pour visualisation en dimensions réduites
- Métriques d'évaluation (silhouette, Davies-Bouldin)
- Prédiction sur nouvelles données
"""



import numpy as np
import matplotlib.pyplot as plt

# Imports depuis le package nuees_dynamiques
from nuees_dynamiques import (
    NuéesDynamique,
    plot_clusters,
    compute_silhouette,
    compute_davies_bouldin,
)

# Imports supplémentaires
from sklearn.datasets import load_iris, load_wine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def example_iris_dataset():
    """
    Exemple 1 : Clustering du dataset Iris.
    
    Dataset Iris : 150 échantillons, 4 features (dimensions de sépales et pétales),
    3 classes réelles (setosa, versicolor, virginica).
    """
    print("\n" + "=" * 70)
    print("EXEMPLE 1 : DATASET IRIS - 4 FEATURES, 3 CLASSES")
    print("=" * 70)

    # Chargement du dataset
    X, y_true = load_iris(return_X_y=True)
    print(f"\n✓ Dataset chargé : {X.shape[0]} échantillons, {X.shape[1]} features")
    print(f"  - Classes réelles : {len(np.unique(y_true))} (setosa, versicolor, virginica)")
    print(f"  - Dimensions : longueur sépale, largeur sépale, longueur pétale, largeur pétale")

    # Création et entraînement du modèle
    # distance_metric='euclidean' : métrique L2, standard pour Iris
    # init_method='kmeans++' : initialisation de haute qualité
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
    print(f"\n✓ Clustering effectué (Nuées Dynamiques) :")
    print(f"  - Itérations : {nd.n_iter_}")
    print(f"  - Inertie finale : {nd.get_inertia():.2f}")

    # Calcul des métriques
    silhouette = compute_silhouette(X, nd.labels_)
    davies_bouldin = compute_davies_bouldin(X, nd.labels_)

    print(f"\n✓ Métriques de qualité :")
    print(f"  - Silhouette score : {silhouette:.3f}")
    print(f"    → Interprétation : Score élevé (>0.5) indique une bonne cohésion et séparation")
    print(f"  - Davies-Bouldin index : {davies_bouldin:.3f}")
    print(f"    → Interprétation : Indice faible suggère des clusters distincts")

    # Comparaison avec K-means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)

    print(f"\n✓ Clustering effectué (K-means scikit-learn) :")
    print(f"  - Itérations : {kmeans.n_iter_}")
    print(f"  - Inertie : {kmeans.inertia_:.2f}")
    print(f"  - Silhouette : {compute_silhouette(X, kmeans.labels_):.3f}")
    print(f"  - Davies-Bouldin : {compute_davies_bouldin(X, kmeans.labels_):.3f}")

    # Visualisation en 2D (via PCA)
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X)

    # Projetter les étalons dans le même espace PCA 2D que les données
    etallons_pca = pca_2d.transform(nd.etallons_)

    print(f"\n✓ Visualisation en 2D via PCA :")
    print(f"  - Variance expliquée : {pca_2d.explained_variance_ratio_.sum():.1%}")
    print(f"  - Note : PCA est utilisée UNIQUEMENT pour visualisation")
    print(f"  - Le clustering est effectué sur les 4 features originales")

    fig = plot_clusters(X_pca_2d, nd.labels_, etallons_pca, title="Iris - Nuées Dynamiques (Projection PCA 2D)")
    plt.tight_layout()
    plt.show()


def example_wine_dataset():
    """
    Exemple 2 : Clustering du dataset Wine - DÉMONSTRATION DE LA VARIANTE MÉDOÏDE.
    
    Dataset Wine : 178 échantillons, 13 features (composition chimique du vin),
    3 classes réelles (3 cultivars).
    
    Note pédagogique : Cet exemple utilise distance_metric='manhattan' et etallon_method='medoid'
    pour démontrer explicitement la VARIANTE MÉDOÏDE sur données réelles.
    - C'est ici que la flexibilité de Nuées Dynamiques (médoïde + métrique alternative) est illustrée
    - La fonction comparative_analysis() se limite volontairement à la configuration BASELINE
      (Euclidienne + centroïde) pour une comparaison chiffrée équitable avec K-means
    """
    print("\n" + "=" * 70)
    print("EXEMPLE 2 : DATASET WINE - DÉMONSTRATION MÉDOÏDE (13 FEATURES, 3 CLASSES)")
    print("=" * 70)

    # Chargement du dataset
    X, y_true = load_wine(return_X_y=True)
    print(f"\n✓ Dataset chargé : {X.shape[0]} échantillons, {X.shape[1]} features")
    print(f"  - Classes réelles : {len(np.unique(y_true))} cultivars de vin")
    print(f"  - Features : alcool, acide malique, acide ascorbique, etc.")

    # Création et entraînement avec métrique Manhattan (L1)
    # CHOIX PÉDAGOGIQUE : distance_metric='manhattan' et etallon_method='medoid'
    # ➜ DÉMONSTRATION EXPLICITE DE LA VARIANTE MÉDOÏDE sur données réelles (13D)
    # Ces paramètres (différents de comparative_analysis()) illustrent la flexibilité
    # - distance_metric='manhattan' (L1) : plus robuste aux outliers que euclidienne
    # - etallon_method='medoid' : point réel du dataset, résistant aux valeurs extrêmes
    # Contrairement à comparative_analysis() qui privilégie la configuration baseline pour comparaison,
    # cet exemple met en avant les capacités avancées de Nuées Dynamiques
    nd = NuéesDynamique(
        data=X,
        n_clusters=3,
        distance_metric="manhattan",
        init_method="kmeans++",
        etallon_method="medoid",
        max_iterations=100,
        tolerance=1e-4,
        random_state=42,
    )

    nd.fit()
    print(f"\n✓ Clustering effectué (Nuées Dynamiques avec distance Manhattan) :")
    print(f"  - Itérations : {nd.n_iter_}")
    print(f"  - Inertie finale : {nd.get_inertia():.2f}")
    print(f"\n  ★ VARIANTE MÉDOÏDE - Démonstration de flexibilité")
    print(f"  - Distance Manhattan (L1) : plus robuste aux outliers que euclidienne")
    print(f"  - Médoïde : étalon est un point RÉEL du dataset, moins affecté par les extrêmes")
    print(f"  - Application : performance supérieure sur données avec bruit ou valeurs aberrantes")
    print(f"\n  ⚠ NOTE MÉTHODOLOGIQUE :")
    print(f"  - Cette démonstration montre la flexibilité de Nuées Dynamiques")
    print(f"  - Les résultats ne sont PAS directement comparables avec comparative_analysis()")
    print(f"  - comparative_analysis() utilise VOLONTAIREMENT une configuration baseline")
    print(f"    (Euclidienne + centroïde) pour une comparaison numérique équitable avec K-means")

    # Calcul des métriques
    silhouette = compute_silhouette(X, nd.labels_)
    davies_bouldin = compute_davies_bouldin(X, nd.labels_)

    print(f"\n✓ Métriques de qualité :")
    print(f"  - Silhouette score : {silhouette:.3f}")
    print(f"  - Davies-Bouldin index : {davies_bouldin:.3f}")
    print(f"    → Interprétation : Un indice modéré reflète la complexité du dataset réel")

    # Comparaison avec K-means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)

    print(f"\n✓ Clustering effectué (K-means scikit-learn) :")
    print(f"  - Itérations : {kmeans.n_iter_}")
    print(f"  - Inertie : {kmeans.inertia_:.2f}")
    print(f"  - Silhouette : {compute_silhouette(X, kmeans.labels_):.3f}")
    print(f"  - Davies-Bouldin : {compute_davies_bouldin(X, kmeans.labels_):.3f}")

    # Visualisation en 3D (via PCA)
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X)

    # Projetter les étalons dans le même espace PCA 3D que les données
    etallons_pca = pca_3d.transform(nd.etallons_)

    print(f"\n✓ Visualisation en 3D via PCA :")
    print(f"  - Variance expliquée : {pca_3d.explained_variance_ratio_.sum():.1%}")
    print(f"  - Note : PCA réduit les 13 dimensions à 3 pour visualisation")
    print(f"  - Le clustering reste basé sur les 13 features originales")

    fig = plot_clusters(X_pca_3d, nd.labels_, etallons_pca, title="Wine - Nuées Dynamiques (Projection PCA 3D)")
    plt.tight_layout()
    plt.show()


def comparative_analysis():
    """
    Exemple 3 : Analyse comparative détaillée.
    
    Crée un tableau récapitulatif comparant les résultats sur Iris et Wine
    pour Nuées Dynamiques et K-means.
    
    Note pédagogique : Cet exemple utilise une CONFIGURATION BASELINE COMMUNE :
    - distance_metric='euclidean' : standard pour comparaison avec K-means
    - etallon_method='centroid' : permet une comparaison équitable
    - init_method='kmeans++' : initialisation de haute qualité
    Cette configuration diffère de example_wine_dataset() qui utilise Manhattan+medoid.
    """
    print("\n" + "=" * 70)
    print("EXEMPLE 3 : ANALYSE COMPARATIVE DÉTAILLÉE")
    print("=" * 70)

    # Chargement des datasets
    X_iris, y_iris = load_iris(return_X_y=True)
    X_wine, y_wine = load_wine(return_X_y=True)

    datasets = [
        ("Iris", X_iris, y_iris),
        ("Wine", X_wine, y_wine),
    ]

    print(f"\n✓ Configuration BASELINE COMMUNE pour comparaison équitable :")
    print(f"  - Distance : Euclidienne (standard, compatible avec K-means)")
    print(f"  - Étalon : Centroïde (permet comparaison directe avec K-means)")
    print(f"  - Initialisation : kmeans++ (haute qualité)")
    print(f"  Note : Cette configuration diffère de example_wine_dataset() qui utilise Manhattan+medoid")
    print(f"\n{'Dataset':<10} {'Méthode':<20} {'Inertie':<12} {'Silhouette':<12} {'Davies-Bouldin':<15} {'Itérations':<10}")
    print("-" * 80)

    for dataset_name, X, y in datasets:
        # Nuées Dynamiques
        # Configuration BASELINE (euclidienne + centroïde) pour comparaison équitable
        nd = NuéesDynamique(
            data=X,
            n_clusters=3,
            distance_metric="euclidean",
            init_method="kmeans++",
            etallon_method="centroid",  # Configuration baseline
            max_iterations=100,
            tolerance=1e-4,
            random_state=42,
        )
        nd.fit()

        sil_nd = compute_silhouette(X, nd.labels_)
        db_nd = compute_davies_bouldin(X, nd.labels_)

        print(
            f"{dataset_name:<10} {'Nuées Dynamiques':<20} {nd.get_inertia():<12.2f} {sil_nd:<12.3f} {db_nd:<15.3f} {nd.n_iter_:<10}"
        )

        # K-means
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)

        sil_km = compute_silhouette(X, kmeans.labels_)
        db_km = compute_davies_bouldin(X, kmeans.labels_)

        print(
            f"{'':<10} {'K-means':<20} {kmeans.inertia_:<12.2f} {sil_km:<12.3f} {db_km:<15.3f} {kmeans.n_iter_:<10}"
        )
        print("-" * 80)

    print(f"\n✓ Synthèse des observations :")
    print(f"  • Iris : Les deux méthodes convergent vers des solutions similaires")
    print(f"    - Structure bien définie avec clusters gaussiens distincts")
    print(f"    - Silhouette élevée indique une bonne séparation")
    print(f"  • Wine : Résultats comparables, données plus complexes")
    print(f"    - Dataset réel avec chevauchement entre classes")
    print(f"    - Nuées Dynamiques offre plus de flexibilité (métriques, médoïdes)")
    print(f"\n  Avantages de Nuées Dynamiques :")
    print(f"  - Support de multiples métriques de distance")
    print(f"  - Utilisation de médoïdes pour robustesse")
    print(f"  - Contrôle fin de la convergence (tolerance)")
    print(f"  - Initialisation flexible (random vs kmeans++)")


def prediction_example():
    """
    Exemple 4 : Prédiction sur nouvelles données.
    
    Démontre l'utilisation de la méthode predict() pour classifier
    de nouvelles observations sur un modèle entraîné.
    """
    print("\n" + "=" * 70)
    print("EXEMPLE 4 : PRÉDICTION SUR NOUVELLES DONNÉES")
    print("=" * 70)

    # Chargement du dataset Iris
    X, y_true = load_iris(return_X_y=True)

    # Séparation train/test (80/20)
    split_idx = 120
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_test = y_true[split_idx:]

    print(f"\n✓ Séparation des données :")
    print(f"  - Train : {X_train.shape[0]} échantillons")
    print(f"  - Test : {X_test.shape[0]} échantillons")

    # Entraînement du modèle sur les données d'entraînement
    nd = NuéesDynamique(
        data=X_train,
        n_clusters=3,
        distance_metric="euclidean",
        init_method="kmeans++",
        max_iterations=100,
        tolerance=1e-4,
        random_state=42,
    )

    nd.fit()
    print(f"\n✓ Modèle entraîné :")
    print(f"  - Itérations : {nd.n_iter_}")
    print(f"  - Inertie (train) : {nd.get_inertia():.2f}")

    # Prédiction sur les données de test
    y_pred = nd.predict(X_test)

    print(f"\n✓ Prédictions sur les {X_test.shape[0]} échantillons de test :")
    print(f"\n{'Index':<8} {'Prédiction':<15} {'Classe réelle':<15}")
    print("-" * 40)

    for i in range(min(10, len(y_pred))):  # Afficher les 10 premiers
        print(f"{split_idx + i:<8} {y_pred[i]:<15} {y_test[i]:<15}")

    if len(y_pred) > 10:
        print(f"... ({len(y_pred) - 10} autres)")

    # Interprétation
    print(f"\n✓ Interprétation :")
    print(f"  - La méthode predict() assigne chaque nouveau point à l'étalon le plus proche")
    print(f"  - Distance utilisée : {nd.distance_metric}")
    print(f"  - Nombre de clusters : {nd.n_clusters}")
    print(f"  - Les prédictions permettent de classifier de nouvelles observations")
    print(f"    sans re-entraîner le modèle")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DÉMONSTRATION DE LA MÉTHODE DES NUÉES DYNAMIQUES")
    print("Application à des datasets réels avec évaluation quantitative")
    print("=" * 70)

    # Exécuter les quatre exemples
    example_iris_dataset()
    example_wine_dataset()
    comparative_analysis()
    prediction_example()

    print("\n" + "=" * 70)
    print("Démonstration terminée !")
    print("=" * 70 + "\n")
