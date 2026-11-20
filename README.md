# Leaffliction

Classification de maladies de feuilles par vision par ordinateur (projet 42).
Le jeu de données (Apple et Grape, 8 classes, issu de PlantVillage) n'est
**pas** dans le dépôt.

## Installation

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install plantcv opencv-python-headless matplotlib scikit-learn torch flake8
```

Les images doivent être rangées une classe par dossier :

```
images/Apple/apple_healthy, apple_apple_scab, apple_black_rot, apple_cedar_apple_rust
images/Grape/Grape_healthy, Grape_Black_rot, Grape_Esca, Grape_spot
```

## Programmes

| Programme | Rôle |
|---|---|
| `Distribution.py` | Pie chart et bar chart par type de plante |
| `Augmentation.py` | 6 augmentations d'une image, ou équilibrage d'un jeu de données |
| `Transformation.py` | Transformations PlantCV d'une image ou d'un dossier |
| `train.py` | Entraînement, produit `learnings.zip` |
| `predict.py` | Prédiction de la maladie d'une feuille |
| `leaf_utils.py`, `leaf_model.py` | Fonctions partagées (augmentation, masque, modèle) |

### Distribution

```sh
./Distribution.py images/Apple          # ou ./images pour toutes les plantes
./Distribution.py images -o charts      # sauvegarde aussi les PNG
```

### Augmentation

```sh
./Augmentation.py "images/Apple/apple_healthy/image (1).JPG"
# crée image (1)_Flip.JPG, _Rotate, _Skew, _Shear, _Crop, _Distortion
./Augmentation.py images -o augmented_directory   # équilibre toutes les classes
```

Sans `-o`, l'équilibrage se fait dans le dossier existant.

### Transformation

```sh
./Transformation.py "images/Apple/apple_healthy/image (1).JPG"   # affichage
./Transformation.py -src images/Apple/apple_healthy -dst dst_directory
./Transformation.py -src images/Apple/apple_healthy -dst dst_directory -mask
```

Transformations : Gaussian blur, Mask, Roi objects, Analyze object,
Pseudolandmarks, Color histogram. `-mask` ne sauvegarde que les images
masquées (feuille sur fond blanc). `--original` copie aussi l'originale.
Le masque vient du canal Lab « a » (Otsu), puis remplissage des trous.

### Classification

```sh
./train.py images -o learnings.zip   # ou ./train.py images/Apple
./predict.py "images/Apple/apple_healthy/image (1).JPG"
./predict.py images/Apple             # précision sur un dossier
./predict.py --validate               # précision sur la validation du zip
```

`train.py` :
1. découpe chaque classe en 80 % entraînement / 20 % validation (stratifié,
   graine fixe), avant toute augmentation ;
2. équilibre le jeu d'entraînement par augmentation ;
3. isole la feuille (suppression du fond) à 128x128 ;
4. entraîne un petit CNN (30 époques, GPU si disponible) ;
5. écrit le zip : modèle, noms de classes, images d'entraînement
   augmentées, images de validation et leur liste.

Résultat : 99,93 % de précision sur 1447 images de validation.

## Rendu

`signature.txt` contient le SHA1 de `learnings.zip` :

```sh
sha1sum learnings.zip > signature.txt
```

Ne plus régénérer le zip après avoir calculé la signature. Le jeu de données
et le zip ne sont pas versionnés.

## Norme

```sh
flake8 *.py
```
