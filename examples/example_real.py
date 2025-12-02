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


def example_iris_multinoyau():
    """
    Exemple avancé : Iris avec noyaux multi-étalons (Diday IV.1).
    
    Cet exemple applique le mode multi-noyaux au dataset Iris pour montrer
    comment les noyaux multi-étalons capturent mieux la structure interne
    des classes dans des espaces multi-dimensionnels.
    
    Le dataset Iris contient 150 échantillons et 4 features. Avec ni=20,
    chaque classe est représentée par un noyau de 20 points dans l'espace 4D.
    """
    print("\n" + "=" * 70)
    print("EXEMPLE AVANCÉ : IRIS AVEC NOYAUX MULTI-ÉTALONS (DIDAY IV.1)")
    print("=" * 70)

    # Chargement du dataset
    X, y_true = load_iris(return_X_y=True)
    print(f"\n✓ Dataset Iris chargé : {X.shape[0]} échantillons, {X.shape[1]} features")
    print(f"  - Classes réelles : {len(np.unique(y_true))} (setosa, versicolor, virginica)")

    # Calcul du ni maximal possible
    n_samples = X.shape[0]
    n_clusters = 3
    max_ni = max(1, n_samples // n_clusters)  # 150 // 3 = 50
    ni_chosen = min(20, max_ni)  # min(20, 50) = 20

    # --- Création du modèle multi-noyaux ---
    nd = NuéesDynamique(
        data=X,
        n_clusters=n_clusters,
        distance_metric='euclidean',
        init_method='kmeans++',
        etallon_method='centroid',
        n_etalons_per_cluster=ni_chosen,  # Noyau multi-étalon (Diday IV.1)
        max_iterations=100,
        tolerance=1e-4,
        random_state=42,
    )

    nd.fit()

    print(f"\n✓ Clustering effectué (Nuées Dynamiques, noyaux multi-étalons) :")
    print(f"  - Nombre d'étalons par classe (ni) : {ni_chosen}")
    print(f"  - Forme des noyaux : {nd.etallons_.shape}")
    print(f"    → {n_clusters} classes × {ni_chosen} étalons × {X.shape[1]} features")
    print(f"  - Itérations : {nd.n_iter_}")
    print(f"  - Inertie finale : {nd.get_inertia():.2f}")

    # Calcul des métriques de qualité
    silhouette = compute_silhouette(X, nd.labels_)
    davies_bouldin = compute_davies_bouldin(X, nd.labels_)

    print(f"\n✓ Métriques de qualité :")
    print(f"  - Silhouette score : {silhouette:.3f}")
    print(f"    → Score élevé (> 0.5) indique une bonne cohésion et séparation")
    print(f"  - Davies-Bouldin index : {davies_bouldin:.3f}")
    print(f"    → Indice faible suggère des clusters bien distincts")

    # Visualisation en 2D via PCA
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X)

    # Aplatir les noyaux multi-étalons pour l'affichage en 2D
    flat_etallons = nd.etallons_.reshape(-1, nd.etallons_.shape[-1])
    etallons_pca = pca_2d.transform(flat_etallons)

    print(f"\n✓ Visualisation en 2D via PCA :")
    print(f"  - Variance expliquée : {pca_2d.explained_variance_ratio_.sum():.1%}")
    print(f"  - Note : PCA est utilisée UNIQUEMENT pour visualisation")
    print(f"  - Le clustering est effectué sur les 4 features originales (espace réel)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Points de données colorés par classe
    scatter = ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=nd.labels_, cmap='viridis', alpha=0.6, s=50)

    # Noyaux multi-étalons (affichés sous forme d'étoiles rouges)
    ax.scatter(
        etallons_pca[:, 0],
        etallons_pca[:, 1],
        c='red',
        marker='*',
        s=150,
        edgecolors='black',
        linewidths=1,
        label=f'Noyaux multi-étalons (ni={ni_chosen})',
        alpha=0.8,
    )

    ax.set_title(
        f"Iris - Noyaux Multi-Étalons (Diday IV.1)\n"
        f"PCA 2D (var={pca_2d.explained_variance_ratio_.sum():.1%}) | "
        f"Silhouette={silhouette:.3f} | DB={davies_bouldin:.3f}"
    )
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n✓ Interprétation :")
    print(f"  - L'intérêt des noyaux multi-étalons devient visible en 4D :")
    print(f"    Avec ni=1 (prototype unique), chaque classe est réduite à son centroïde.")
    print(f"    Avec ni={ni_chosen}, chaque classe est décrite par {ni_chosen} points qui")
    print(f"    capturent la structure et l'étendue interne de la classe dans l'espace 4D.")
    print(f"  - Cela permet une meilleure compréhension géométrique des classes")
    print(f"  - et une représentation plus fidèle pour la prédiction et l'analyse.")
    print(f"  - Conformément à Diday (1971, IV.1), ce mode est particulièrement utile")
    print(f"    pour les clusters de formes complexes ou allongées.")


def example_wine_multinoyau_3d():
    """
    Exemple avancé : Wine avec noyaux multi-étalons en visualisation 3D.
    
    Cet exemple applique le mode multi-noyaux au dataset Wine pour démontrer
    comment les noyaux multi-étalons fonctionnent sur des données réelles
    haute-dimension (13D) avec visualisation PCA 3D interactive.
    
    Le dataset Wine contient 178 échantillons et 13 features. Avec ni=15,
    chaque classe est représentée par un noyau de 15 points dans l'espace 13D,
    projeté en 3D pour visualisation.
    """
    print("\n" + "=" * 70)
    print("EXEMPLE AVANCÉ : WINE AVEC NOYAUX MULTI-ÉTALONS EN 3D (DIDAY IV.1)")
    print("=" * 70)

    # Chargement du dataset
    X, y_true = load_wine(return_X_y=True)
    print(f"\n✓ Dataset Wine chargé : {X.shape[0]} échantillons, {X.shape[1]} features")
    print(f"  - Classes réelles : {len(np.unique(y_true))} cultivars de vin")
    print(f"  - Features : alcool, acide malique, etc. (composition chimique)")

    # Calcul du ni maximal possible
    n_samples = X.shape[0]
    n_clusters = 3
    max_ni = max(1, n_samples // n_clusters)  # 178 // 3 = 59
    ni_chosen = 15  # Choisi comme demandé

    # --- Création du modèle multi-noyaux ---
    nd = NuéesDynamique(
        data=X,
        n_clusters=n_clusters,
        n_etalons_per_cluster=ni_chosen,  # Noyau multi-étalon (Diday IV.1)
        distance_metric='euclidean',
        init_method='random',
        etallon_method='centroid',
        max_iterations=100,
        tolerance=1e-4,
        random_state=42,
    )

    nd.fit()

    print(f"\n✓ Clustering effectué (Nuées Dynamiques, noyaux multi-étalons) :")
    print(f"  - Nombre d'étalons par classe (ni) : {ni_chosen}")
    print(f"  - Forme des noyaux : {nd.etallons_.shape}")
    print(f"    → {n_clusters} classes × {ni_chosen} étalons × {X.shape[1]} features")
    print(f"  - Itérations : {nd.n_iter_}")
    print(f"  - Inertie finale : {nd.get_inertia():.2f}")

    # Calcul des métriques de qualité
    silhouette = compute_silhouette(X, nd.labels_)
    davies_bouldin = compute_davies_bouldin(X, nd.labels_)

    print(f"\n✓ Métriques de qualité :")
    print(f"  - Silhouette score : {silhouette:.3f}")
    print(f"    → Score modéré reflète la complexité des données réelles haute-dim")
    print(f"  - Davies-Bouldin index : {davies_bouldin:.3f}")
    print(f"    → Indice suggère des clusters avec chevauchement partiel")

    # Visualisation en 3D via PCA
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X)

    # Aplatir les noyaux multi-étalons pour l'affichage en 3D
    flat_etallons = nd.etallons_.reshape(-1, nd.etallons_.shape[-1])
    etallons_pca = pca_3d.transform(flat_etallons)

    print(f"\n✓ Visualisation en 3D via PCA :")
    print(f"  - Variance expliquée : {pca_3d.explained_variance_ratio_.sum():.1%}")
    print(f"  - Note : PCA réduit les 13 dimensions à 3 pour visualisation")
    print(f"  - Le clustering est effectué sur les 13 features originales (espace réel)")

    # Plot 3D avec matplotlib
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Points de données colorés par classe
    scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=nd.labels_, cmap='viridis', alpha=0.6, s=50)

    # Noyaux multi-étalons (affichés sous forme d'étoiles rouges)
    ax.scatter(
        etallons_pca[:, 0],
        etallons_pca[:, 1],
        etallons_pca[:, 2],
        c='red',
        marker='*',
        s=150,
        edgecolors='black',
        linewidths=1,
        label=f'Noyaux multi-étalons (ni={ni_chosen})',
        alpha=0.8,
    )

    ax.set_title(
        f"Wine - Noyaux Multi-Étalons 3D (Diday IV.1)\n"
        f"PCA 3D (var={pca_3d.explained_variance_ratio_.sum():.1%}) | "
        f"Silhouette={silhouette:.3f} | DB={davies_bouldin:.3f}"
    )
    ax.set_xlabel(f"PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})")
    ax.set_zlabel(f"PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n✓ Interprétation :")
    print(f"  - Test multi-noyaux sur données réelles haute-dimension (13D→3D PCA) :")
    print(f"    Avec ni={ni_chosen}, chaque classe est décrite par {ni_chosen} points qui")
    print(f"    capturent la structure interne dans l'espace 13D des compositions chimiques.")
    print(f"  - La visualisation 3D permet une exploration interactive (rotation) pour")
    print(f"    comprendre la séparation des cultivars de vin dans un espace réduit.")
    print(f"  - Cela démontre la flexibilité de Nuées Dynamiques pour des données complexes,")
    print(f"    où les noyaux multi-étalons offrent une représentation plus riche que les prototypes uniques.")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DÉMONSTRATION DE LA MÉTHODE DES NUÉES DYNAMIQUES")
    print("Application à des datasets réels avec évaluation quantitative")
    print("=" * 70)

    # Exécuter les exemples
    example_iris_dataset()
    example_wine_dataset()
    comparative_analysis()
    prediction_example()
    example_iris_multinoyau()  # Nouvel exemple
    example_wine_multinoyau_3d()  # Nouvel exemple

    print("\n" + "=" * 70)
    print("Démonstration terminée !")
    print("=" * 70 + "\n")