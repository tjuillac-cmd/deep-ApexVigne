# deep-ApexVigne

Modélisation du stress hydrique/phénologique de la vigne à l'échelle régionale, par deep learning et géostatistique. Projet réalisé dans le cadre
d'un stage de fin d'études à l'Institut Agro Montpellier au sien de l'UMR ITAP.

## Objectif

Produire des cartes hebdomadaires de probabilité de stress hydrique et
l'incertitude associée, à partir des observations terrain crowdsourcées
**APEX-Vigne** (indice iC-APEX) et de covariables environnementales
(topographie, climat SICLIMA/SAFRAN, réserve utile des sols, précipitations
cumulées).

Deux variantes de modèle sont comparées chaque semaine :
- **classique** — toutes les covariables disponibles
- **degrade** — seulement les covariables basiques, accessibles facilement en
  temps réel (exclut les réanalyses climatiques)

## Environnement

Environnement (Python 3.11) comprenant les dépendances :
`numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`,
`optuna`, `geopandas`, `rasterio`, `shapely`, `cartopy`, `matplotlib`,
`papermill`, `openpyxl`, plus les modules locaux `src.model` et `src.losses`.

## Structure du dépôt

```
deepApex-Vigne.ipynb          Notebook principal (entraînement + inférence spatiale)
pipeline_hebdo.py          Exécute le notebook pour chaque semaine et chaque
                            modèle (classique/degrade) via papermill
calcul_bornes_couleurs.py  Calcule les bornes de couleur globales (min/max/q99)
                            à partir des résultats hebdomadaires, séparément
                            par modèle
agreger_resultats.py       Agrège les résultats hebdomadaires dans un fichier
                            Excel (resultats_hebdo.xlsx)
src/model.py                Architecture du modèle (MLP) et prédiction MC-Dropout
src/losses.py                Loss DeepMaxEnt personnalisée
data/cov/rasters/            Covariables raster (alt, pente, exp, pluvio, cwb...)
data/cov/cov_siclima_*.csv    Covariables climatiques SICLIMA/SAFRAN par cellule
results/                    Résultats hebdomadaires exportés ({tag}_{date}.pkl)
executed_notebooks/         Copies exécutées du notebook (traçabilité, générées
                            par pipeline_hebdo.py)
bornes_couleurs_reference.json  Bornes de couleur globales par modèle, utilisées
                            pour des cartes comparables entre semaines
```

## Notebook principal (`deepApex-Vigne.ipynb`)

Le notebook est paramétré par une cellule taguée `parameters` contenant :

```python
MODEL_TAG = "classique"      # ou "degrade"
date_debut_str = "2025-06-16"
```

Pipeline : chargement des covariables (raster + SICLIMA) → K-Fold aléatoire
(non spatial, vu le faible nombre d'observations) → entraînement MLP avec loss
DeepMaxEnt → sélection du meilleur fold (Kendall tau) → inférence spatiale sur
toute la grille avec quantification d'incertitude par MC-Dropout → export des
résultats dans `results/{MODEL_TAG}_{date_debut_str}.pkl`.

## Lancer le pipeline hebdomadaire

```bash
conda activate apex
python pipeline_hebdo.py
```

Exécute `model_apex8.ipynb` pour chaque semaine du 16 juin au 17 août 2025,
pour les deux variantes de modèle. Produit un notebook exécuté par run dans
`executed_notebooks/` et un `.pkl` de résultats par run dans `results/`.

## Calculer les bornes de couleur globales

Une fois le pipeline exécuté sur toutes les semaines voulues :

```bash
python3 calcul_bornes_couleurs.py
```

Produit `bornes_couleurs_reference.json`, avec des bornes (min, max, 99e
percentile) séparées par tag de modèle — à charger dans les cellules de plot
du notebook pour que les cartes hebdomadaires restent comparables entre elles
dans le temps (au lieu de recalculer une échelle locale à chaque semaine).

## Agréger les résultats dans un fichier Excel

```bash
python agreger_resultats.py
```

Produit `resultats_hebdo.xlsx`, avec une feuille par modèle (`modele` /
`modele_degrade`), résumant pour chaque semaine : nombre d'observations,
nombre de cellules observées, AUC-ROC et Kendall tau (top 1/2/3 + moyenne/
écart-type sur les 3 meilleurs folds), et les statistiques iC-APEX terrain.