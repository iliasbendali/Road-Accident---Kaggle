# Road-Accident---Kaggle
Prédiction du Risque d'Accident de la Route - Challenge Kaggle d'octobre 2025

## Objectif

L'objectif était de développer un modèle pour prédire un score de risque d'accident.

## Démarche et Modèle

J'ai importé deux fichiers : l'EDA et la modélisation. On retrouve dans l'EDA tous les graphiques qui m'ont servi par la suite. Pour la modélisation, j'ai essayé plein de choses : target encoding, binary encoding, passage en bins pour les variables numériques, K-means, PCA... Par soucis de clarté, je n'ai gardé que ce qui a bien fonctionné.

### 1. Feature Engineering

* **Encodage par fréquence** : Les variables catégorielles (`road_type`, `weather`, etc.) ont été encodées en utilisant la fréquence de leurs occurrences.
* **Création de features composites** : Ajout de features booléennes (`is_night_foggy`, `is_high_accident_zone`).
* **Feature linéaire personnalisée** : Création d'une feature `y` basée sur une régression linéaire pondérée manuellement (combinant `curvature`, `lighting`, `weather`, etc.). Cette feature a ensuite été "clippée" à l'aide d'une fonction sigmoïde personnalisée (`scipy.stats.norm`) pour la borner entre 0 et 1.

### 2. Modélisation

* **Modèle** : `XGBoost Regressor`.
* **Validation** : Utilisation d'une Cross-Validation (5-fold) pour optimiser les hyperparamètres (max_depth, learning_rate, etc.) et trouver le nombre optimal d'itérations (`early_stopping_rounds`).

## Technologies Utilisées

* Python
* Pandas & Numpy
* **XGBoost**
* Scikit-learn
* Scipy
