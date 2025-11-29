"""
Méthode des Nuées Dynamiques (Diday, 1971)

Ce package fournit une implémentation complète de la méthode des nuées dynamiques,
un algorithme de clustering non-hiérarchique introduit par E. Diday en 1971.

Le package contient les modules principaux :
- `core` : classe NuéesDynamique pour l'entraînement et la prédiction
- `distances` : métriques de distance optimisées (euclidienne, Manhattan, Minkowski, Chebyshev)
- `utils` : utilitaires pour visualisation, évaluation (silhouette, Davies-Bouldin) et génération de données synthétiques

Les fonctions utilitaires (visualization, evaluation, synthetic data generation) nécessitent matplotlib et scikit-learn.
"""

__version__ = "0.1.0"
__author__ = "Votre Nom"
__description__ = "Implémentation de la méthode des nuées dynamiques (Diday, 1971)."

# Imports principaux : distances et classe NuéesDynamique
from .distances import (
	euclidean_distance,
	manhattan_distance,
	minkowski_distance,
	chebyshev_distance,
	compute_distance,
	compute_distance_matrix,
	pairwise_distances,
)

from .core import NuéesDynamique

# Imports des utilitaires
# Les fonctions utilitaires (visualisation, évaluation, génération de données) dépendent
# de matplotlib et scikit-learn. Ces dépendances sont gérées directement dans utils.py
# via les drapeaux _HAS_MATPLOTLIB et _HAS_SKLEARN. Le module utils.py lève des
# ImportError explicites avec messages en français si les dépendances manquent.
# En important directement ici sans try-except, on évite de masquer les erreurs réelles
# du module utils (ex: bugs de syntaxe, problèmes internes).
from .utils import (
	plot_clusters,
	compute_silhouette,
	compute_davies_bouldin,
	generate_synthetic_data,
)

# Contrôle des exports publics du package
__all__ = [
	# Fonctions de distance
	"euclidean_distance",
	"manhattan_distance",
	"minkowski_distance",
	"chebyshev_distance",
	"compute_distance",
	"compute_distance_matrix",
	"pairwise_distances",
	# Classe principale
	"NuéesDynamique",
	# Utilitaires
	"plot_clusters",
	"compute_silhouette",
	"compute_davies_bouldin",
	"generate_synthetic_data",
]

