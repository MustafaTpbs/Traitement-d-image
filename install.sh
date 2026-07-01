#!/bin/bash
set -e

echo "--- Étape 1 : Installation des dépendances système ---"
sudo apt update
sudo apt install -y python3-venv python3-libcamera libcap-dev
#activation de l'i2c sur la raspberry
sudo raspi-config nonint do_i2c 0

echo "--- Étape 2 : Création de l'environnement virtuel ---"
# Elle permet au venv de voir les paquets installés par 'apt' (comme libcamera)
python3 -m venv --system-site-packages .venv

echo "--- Étape 3 : Installation des dépendances Python ---"
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo "--- Installation terminée ---"