# Guide d'installation

---

## 1. Prérequis

* Raspberry Pi sous Raspberry Pi OS
* Carte Witty Pi 4 (RTC + gestion d'alimentation) branchée sur les GPIO.
* Module caméra (v2, Module 3, etc.) branché sur le port CSI.
* Convertisseur DAC MCP4725 branché en I2C.
* Connexion Internet (uniquement pour l'installation).

---

## 2. Installation

Ouvrir un terminal et copier-coller : 

```bash
sudo apt update && sudo apt install -y git
git clone https://github.com/MustafaTpbs/Traitement-d-image.git
cd Traitement-d-image
chmod +x install.sh && ./install.sh
```

### Ce que `install.sh` fait automatiquement :
* Active le bus I2C.
* Installe WiringPi, l'utilitaire Witty Pi 4 et synchronise la RTC.
* Programme le réveil automatique quotidien à 10h20 (durée : 5 min).
* Crée l'environnement virtuel `.venv` et installe les dépendances Python.
* Configure et active le service systemd (`mon_service.service`) au démarrage.
* Crée l'environnement de debug (`.venv-debug`) avec OpenCV et Picamera2.

---

## 3. Vérification de l'installation

### Matériel & I2C

```bash
sudo i2cdetect -y 1
```
**Résultat attendu :**
* `0x08` présent (Horloge RTC / Witty Pi 4)
* `0x60` présent (DAC MCP4725).

```bash
sudo timedatectl
```
**Résultat attendu :** L'heure et la date actuelles s'affichent.

---

### Caméra

```bash
rpicam-hello --list-cameras
```
**Résultat attendu :** La caméra est détectée à l'index 0.

```bash
# 1. Stopper temporairement le service de démarrage automatique
sudo systemctl stop mon_service.service

# 2. Prendre la photo de test
rpicam-still -o test.jpg

# 3. Relancer le service automatique une fois le test fini
sudo systemctl start mon_service.service
```
**Résultat attendu :** Une photo `test.jpg` est créée sans erreur (dossier du terminal).

---

### Réveil automatique Witty Pi 4

```bash
cat /home/raspberry/wittypi/schedule.wpi
```
**Résultat attendu :**
```text
BEGIN 2026-09-22 10:20:00
END 2035-12-31 23:59:59
ON M5
OFF H23 M55
```

---

### Service de démarrage automatique

```bash
sudo systemctl status mon_service.service
```
**Résultat attendu :** Statut `active (running)` ou `active (exited)`.

Pour voir les logs du script principal :
```bash
sudo journalctl -u mon_service.service -e --no-pager
```
**Résultat attendu :** Affichage des mesures d'angle et de la tension envoyée au DAC.

---

### Test manuel avec affichage graphique (Debug)

```bash
# Mettre la caméra devant l’indicateur

# 1. Stopper d'abord le service autonome qui occupe la caméra
sudo systemctl stop mon_service.service

# 2. Se placer dans le dossier du projet (CRUCIAL)
cd /home/raspberry/Traitement-d-image

# 3. Lancer l'exécution en mode debug
./.venv-debug/bin/python main.py
./.venv-debug/bin/python main.py

# 1. Re activer le service autonome qui occupe la caméra
sudo systemctl start mon_service.service
```
**Résultat attendu :** Les fenêtres de rendu OpenCV (Redressement, Contours, Détection Aiguille) s'ouvrent à l'écran.
