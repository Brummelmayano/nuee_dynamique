# Méthode des Nuées Dynamiques (Diday, 1971)

[![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue)](https://www.python.org/)
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

1. Ouvrez l'URL fournie pour l'application (ex. `https://share.streamlit.io/<utilisateur>/<repo>/<chemin>`). Remplacez ceci par l'URL réelle de votre instance déployée.

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
	- Erreurs liées à scikit-learn ou threadpoolctl : assurez-vous d'utiliser les versions listées dans `requirements.txt` (notamment `scikit-learn>=1.2.2,<1.4` et `threadpoolctl>=3.1.0`).
	- Si l'app renvoie une erreur inattendue, récupérez la trace d'erreur (console du serveur) et ouvrez une issue avec un exemple de CSV minimal.

Si vous voulez, je peux ajouter dans le README le lien direct vers l'application déployée (badge ou URL) — fournissez simplement l'URL et je l'insère proprement.

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
