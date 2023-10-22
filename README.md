# Emergent Language in Multi-Step Bidirectional Reference Game

## Description

## Project Set-Up

1. Clone the repository with ``git clone [repository URL]``.
2. Large ``data/`` and ``results/`` folders are pushed with ``.gitignore``. They will appear once you clone the repository.
3. Setting up the environment. 
    * Recreate the environment: ``conda env create --name myenv --file environment.yml``
    * Activate conda env: ``conda activate --name``

## Reposetory Strucure

<img src="images/repo-structure.png" width="300" height="500">

## Project Execution
- The ``data_loader.py`` loads ``coco_captions`` dataset, extracts features with ``ResNet50``, and saves ``train``, ``test`` and ``val`` datasets as ``{ds}.tfrecord`` files. You may skip this step; extracted features are already saved in ``data`` folder for further usage.
- Run ``main.py`` to execute the rest of the project:
    - ``python3 main.py`` will execute training with all hyperparamters one by one (see ``args.py``)

    You can specify which hyperparameters combination to check, e.g.:
    - ``python3 main.py --learning-rates 0.01 --batch-sizes 64 --epochs 50``
    - ``python3 main.py --learning-rates 0.001 0.0001 --batch-sizes 64 128 --epochs 100 200``