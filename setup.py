"""Configuration et installation du package `nuees-dynamiques`.

Ce fichier permet d'installer le package depuis les sources.
"""

from setuptools import setup, find_packages

# Lire la description longue depuis README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Lire les dépendances depuis requirements.txt (ignorer les commentaires)
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.strip().startswith('#')]

setup(
    # Métadonnées du package
    name="nuees-dynamiques",
    version="0.1.0",
    author="Brummel Mayano",
    author_email="brummelmayano@gmail.com",

    # Descriptions
    description="Implémentation de la méthode des nuées dynamiques (Diday, 1971).",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # Lien vers le repository (à personnaliser si nécessaire)
    url="https://github.com/votre-repo/nuees-dynamiques",

    # Découverte automatique des packages
    packages=find_packages(),

    # Métadonnées de classement
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    # Versions Python supportées
    python_requires=">=3.8",

    # Dépendances
    install_requires=requirements,

    # Mots-clés
    keywords="clustering, nuées dynamiques, Diday, machine learning",
)
