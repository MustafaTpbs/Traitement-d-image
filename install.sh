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



echo "--- Étape 4 : Configuration du lancement automatique ---"
chmod +x "$(pwd)/run.sh" #Rend le fichier de lancement de programme executable
# On crée le fichier de service
cat <<EOF | sudo tee /etc/systemd/system/mon_service.service
[Unit]
Description= Lancement automatique du script de traitement d'image de l'indicateur
After=network.target

[Service]
Type=simple
ExecStart=/bin/bash $(pwd)/run.sh
WorkingDirectory=$(pwd)
User=$(whoami)
Restart=on-failure
RestartSec=2s

[Install]
WantedBy=multi-user.target
EOF

# Active le service
sudo systemctl daemon-reload
sudo systemctl enable mon_service.service
sudo systemctl start mon_service.service

echo "--- Installation et activation du lancement automatique terminées ---"



echo "--- Installation terminée ---"