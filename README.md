# Méthode des Nuées Dynamiques (Diday, 1971)

[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Description
-----------
Implementation légère et modulaire de la méthode des Nuées Dynamiques de Diday (1971). Le package fournit une API inspirée de scikit-learn pour partitionner des jeux de données en clusters, avec plusieurs méthodes d'étalons et métriques configurables.

Principe de l'algorithme
------------------------
1. Initialisation : choisir K étalons (prototypes).
2. Assignation : affecter chaque point à l'étalon le plus proche selon une métrique.
3. Mise à jour des étalons : recalculer les étalons du cluster.
4. Convergence : répéter jusqu'à stabilisation ou nombre maximum d'itérations.
5. Résultat : partition finale et étalons.

Différences avec K-means
------------------------
- 4 méthodes d'étalons : `centroid`, `medoid`, `median`, `mode`.
- Métriques multiples : `euclidean`, `manhattan`, `minkowski`, `chebyshev`, etc.
- Initialisations : `random`, `kmeans++`.
- API : `fit()`, `predict()`, `fit_predict()`, `get_inertia()`.

Fonctionnalités clés ✨
- `centroid` (moyenne), `medoid` (point réel), `median` (médiane robuste), `mode` (valeur fréquente).
- Support de plusieurs métriques de distance.
- Réinitialisation automatique des clusters vides.
- Outils d'évaluation (silhouette, Davies-Bouldin) dans `nuees_dynamiques.utils`.

Installation 🚀

Depuis la racine du projet :

```powershell
cd c:/Users/brummel/OneDrive/Desktop/nuee_dynamique
pip install .
```

Ou installer les dépendances :

```powershell
pip install -r requirements.txt
```

Compatibilité et notes de versions
---------------------------------

Ce projet a été testé avec Python 3.11 et scikit-learn 1.2.x / 1.3.x. Sur
certaines plateformes, des versions plus anciennes de la dépendance
`threadpoolctl` peuvent produire des logs d'AttributeError (souvent bénins)
lors de l'exécution de `sklearn.cluster.KMeans` — ces messages sont
généralement sans impact sur la sortie mais peuvent polluer la console.

Pour éviter cela, le fichier `requirements.txt` pinne `scikit-learn` dans une
fourchette récente et ajoute `threadpoolctl>=3.1.0`. Si vous rencontrez des
messages d'erreur liés à `threadpoolctl`, réinstallez les dépendances dans un
environnement propre :

```bash
# depuis la racine du projet
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Alternativement, l'installation en mode editable est pratique pour le développement :

```bash
pip install -e .
```

Utilisation rapide 📖

Voici des exemples d'utilisation :

```python
import numpy as np
from nuees_dynamiques import NuéesDynamique

X = np.random.RandomState(0).randn(200, 2)

# 1) Centroid (défaut)
nd = NuéesDynamique(data=X, n_clusters=3, etallon_method='centroid', random_state=0)
nd.fit()
print('Centroid labels:', np.unique(nd.labels_))

# 2) Medoid (point réel, utile pour formes non-convexes)
nd_medo = NuéesDynamique(data=X, n_clusters=3, etallon_method='medoid', random_state=0)
nd_medo.fit()

# 3) Median (robuste aux outliers)
nd_med = NuéesDynamique(data=X, n_clusters=3, etallon_method='median', random_state=0)
nd_med.fit()

# 4) Mode (pour données discrètes / catégorielles)
# Assurez-vous que les données sont encodées numériquement si nécessaire
X_int = (X * 10).astype(int)
nd_mode = NuéesDynamique(data=X_int, n_clusters=3, etallon_method='mode', random_state=0)
nd_mode.fit()

# Prédiction
X_new = np.random.RandomState(1).randn(5, 2)
labels_new = nd.predict(X_new)
print('Predictions:', labels_new)
```

Noyaux Multi-Étalons (Diday 1971, section IV.1) 🎯
---------------------------------------------------

Diday a proposé dans la section IV.1 de son article une extension permettant à chaque cluster d'être représenté non pas par un seul étalon, mais par un **noyau de plusieurs étalons** (`ni = card(Ei)`). Cette généralisation permet de capturer des structures de clusters plus complexes, notamment les formes allongées ou irrégulières.

### Algorithme IV.1 – Entrées et sorties

**Entrées :**
- `data` : jeu de données à partitionner (n_samples, n_features).
- `K` : nombre de classes/clusters.
- `ni` : nombre d'étalons par classe (cardinal de chaque noyau Ei).
- Optionnel : initialisation des noyaux Ei (ou choix aléatoire/kmeans++).

**Sorties :**
- `C1, C2, ..., CK` : partition finale en K classes.
- `E1, E2, ..., EK` : noyaux finaux (chaque Ei contient ni points).
- Mesure d'homogénéité globale (inertie, silhouette, Davies-Bouldin).
- Mesure locale par classe (en option).

### Exemple de code – Mode multi-noyaux

Pour utiliser le mode multi-noyaux, il suffit de passer le paramètre `n_etalons_per_cluster` :

```python
import numpy as np
from nuees_dynamiques import NuéesDynamique

# Jeu de données avec clusters allongés
X = np.random.RandomState(0).randn(300, 2)

# Mode multi-noyaux (Diday IV.1)
nd = NuéesDynamique(
    data=X,
    n_clusters=2,
    n_etalons_per_cluster=40,  # 40 étalons par cluster (au lieu de 1)
    etallon_method='centroid',
    random_state=0
)
nd.fit()

# Résultat : etallons_ est maintenant de forme (2, 40, 2)
print(f"Forme des noyaux : {nd.etallons_.shape}")
# Sortie : Forme des noyaux : (2, 40, 2)

# Chaque cluster est représenté par un noyau de 40 points
print(f"Noyau du cluster 0 : {nd.etallons_[0].shape}")  # (40, 2)
```

### Avantages du mode multi-noyaux

- **Clusters allongés** : Un seul étalon ne peut pas bien représenter un cluster de forme elliptique. Plusieurs étalons, disposés le long de l'allongement, offrent une meilleure approximation.
- **Structures irrégulières** : Pour des clusters non-convexes, un noyau multi-étalon peut s'adapter à la géométrie locale.
- **Robustesse** : L'ensemble de points dans le noyau fournit une caractérisation plus riche qu'un centroïde unique.
- **Exemple article** : Dans l'article de Diday (1971, IV.1), des expériences sur 283 ou 180 points montrent que ~40 étalons par classe capturent bien les structures des partitions.

**Note** : Le mode multi-noyaux est complètement transparent ; assignation et prédiction fonctionnent de la même façon qu'avec `ni=1`.

Exemples fournis 📊
- `examples/example_synthetic.py` — démonstrations sur données synthétiques (2D/3D) et comparaison avec `sklearn.cluster.KMeans`.
- `examples/example_real.py` — exemples sur jeux réels (Iris, Wine), PCA et prédiction.

Application interactive (Streamlit)
---------------------------------

Un démonstrateur interactif a été ajouté : `streamlit_app.py` (interface en français) qui permet de :

- Choisir un jeu de données (Iris, Wine, Synthétique ou uploader un CSV).
- Comparer `Nuées Dynamiques` et `K-Means` côte à côte.
- Sélectionner la métrique, la méthode d'étalon, le nombre de clusters (slider ou saisie manuelle).
- Visualiser en 2D/3D (projection PCA) et télécharger le jeu avec labels.

Exécution locale
-----------------

1. Installez les dépendances (Streamlit et pandas sont inclus dans `requirements.txt`) :

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Lancez l'application Streamlit localement :

```bash
streamlit run streamlit_app.py
```

Utiliser l'application déployée
-------------------------------

L'application Streamlit a été déployée en ligne. Pour l'exploiter :

1. Ouvrez l'URL fournie pour l'application (`https://nuees-dynamiques.streamlit.app/`). 

2. Interface générale :
	- La barre latérale (gauche) contient tous les contrôles : choix de l'algorithme (`Nuées Dynamiques`, `K-Means` ou `Both`), métrique de distance, méthode d'étalon, jeu de données ou upload CSV, dimension de visualisation (2D/3D) et le nombre de clusters (slider ou saisie manuelle).
	- Cliquez sur « Lancer le clustering » pour exécuter l'algorithme avec les paramètres choisis.

3. Visualisation et interprétation :
	- La visualisation utilise une projection PCA. Le titre affiche « PCA 2D (var: XX%) » ou « PCA 3D (var: XX%) » : `var` est le pourcentage de variance expliquée par les composantes principales affichées. Plus cette valeur est élevée, mieux la projection conserve l'information originale.
	- En mode `Both`, les résultats ND et K-Means sont affichés côte à côte pour comparaison.
	- Les centres sont étiquetés « Étalons » pour Nuées Dynamiques et « Centroïdes » pour K-Means.

4. Upload CSV et format attendu :
	- Le bouton d'upload accepte les fichiers CSV.
	- L'application utilisera uniquement les colonnes numériques (toutes les colonnes non numériques seront ignorées).
	- Le CSV peut contenir une ligne d'entête ; l'encodage UTF-8 est recommandé.
	- Pour de meilleurs résultats, fournissez un jeu de caractéristiques normalisées ou standardisées si vos features ont des échelles très différentes.

5. Télécharger les résultats :
	- Après exécution, utilisez le bouton de téléchargement pour récupérer un CSV contenant les features originales, la colonne `cluster_<algorithme>` et (si disponible) `true_label`.

6. Conseils performance et limites :
	- L'application effectue des calculs côté serveur : pour de très grands jeux de données (dizaines de milliers d'échantillons), l'exécution peut être lente ou atteindre les limites de la plateforme hébergeant l'app.
	- Si vous traitez de gros volumes, pré-traitez et réduisez la dimensionnalité (PCA) localement avant upload.

7. Dépannage rapide :
	- Pas de données affichées après upload : vérifiez que le CSV contient des colonnes numériques et qu'il n'est pas vide.
	- Erreurs liées à scikit-learn ou threadpoolctl : assurez-vous d'utiliser les versions listées dans `requirements.txt` 


Tests rapides
-------------

Après installation, vous pouvez exécuter la suite de tests :

```bash
pytest -q
```


Références
----------
- Diday, E. (1971). Méthode des nuées dynamiques.

Licence
-------
MIT
