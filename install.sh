#!/bin/bash
set -e


PROJECT_DIR="$(pwd)"


echo "--- Étape 1 : Installation des dépendances système ---"
sudo apt update
sudo apt install -y python3-venv python3-libcamera libcap-dev
#activation de l'i2c sur la raspberry
sudo raspi-config nonint do_i2c 0



echo "--- Étape 1bis : Installation Witty Pi 4 (RTC + alimentation) ---" 
sudo apt install -y git build-essential i2c-tools 

# wiringPi (retiré des dépôts apt depuis Bookworm, fork officiel maintenu) 
if [ ! -d "/home/raspberry/WiringPi" ]; then 
	cd /home/raspberry 
	git clone https://github.com/WiringPi/WiringPi.git 
	cd WiringPi 
	./build 
else 
	echo  "WiringPi déjà présent, on passe." 
	
fi 
gpio -v 

# Witty Pi 4 (installe dans /home/raspberry/wittypi) 
if [ ! -d "/home/raspberry/wittypi" ]; then 
	cd /home/raspberry 
	wget https://www.uugear.com/repo/WittyPi4/install.sh -O wittypi_install.sh 
	yes | sudo bash wittypi_install.sh 
else 
	echo "Witty Pi déjà installée, on passe." 
fi 

# Synchronise l'heure système actuelle (bonne, car on est encore en ligne ici) vers la RTC 
cd /home/raspberry/wittypi 
. ./utilities.sh
system_to_rtc 
rtc_to_system 
date



echo "--- Étape 1ter : Marge de sécurité + réveil quotidien Witty Pi ---"
cd /home/raspberry/wittypi
. ./utilities.sh
i2c_write ${I2C_BUS} $I2C_MC_ADDRESS $I2C_CONF_POWER_CUT_DELAY 250

WAKE_TIME="10:20"
WORK_MINUTES=5
OFF_MINUTES=$((24*60 - WORK_MINUTES))
OFF_H=$((OFF_MINUTES / 60))
OFF_M=$((OFF_MINUTES % 60))
BEGIN_DATE=$(date -d "yesterday" +%Y-%m-%d)

cat > /home/raspberry/wittypi/schedule.wpi <<EOF
BEGIN ${BEGIN_DATE} ${WAKE_TIME}:00
END 2035-12-31 23:59:59
ON M${WORK_MINUTES}
OFF H${OFF_H} M${OFF_M}
EOF

./runScript.sh

# Correction du décalage de sécurité de Witty Pi (documenté) :
# runScript.sh peut repousser le premier réveil d'un cycle complet
# si le script tourne pendant que le Pi est déjà allumé hors de la fenêtre ON.
# On force donc explicitement la vraie prochaine occurrence de WAKE_TIME.
NOW_SEC=$(date +%s)
TODAY_TARGET=$(date -d "today ${WAKE_TIME}" +%s)
if [ "$TODAY_TARGET" -gt "$NOW_SEC" ]; then
    TARGET_TS=$TODAY_TARGET
else
    TARGET_TS=$(date -d "tomorrow ${WAKE_TIME}" +%s)
fi
TARGET_DAY=$(date -d @$TARGET_TS +%d)
TARGET_HOUR=$(date -d @$TARGET_TS +%H)
TARGET_MIN=$(date -d @$TARGET_TS +%M)
set_startup_time $TARGET_DAY $TARGET_HOUR $TARGET_MIN 0
echo "Prochain réveil forcé à : jour ${TARGET_DAY}, ${TARGET_HOUR}:${TARGET_MIN}"

cd "$PROJECT_DIR"


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


echo "--- Étape 5 (optionnelle) : Environnement de debug avec fenêtres graphiques ---"
sudo apt install -y python3-opencv python3-numpy python3-picamera2
python3 -m venv --system-site-packages .venv-debug
./.venv-debug/bin/pip install --upgrade pip
./.venv-debug/bin/pip install picamzero==1.0.2 --no-deps
./.venv-debug/bin/pip install adafruit-circuitpython-mcp4725
echo "--- Environnement de debug prêt (.venv-debug) ---"


echo "--- Installation terminée ---"
