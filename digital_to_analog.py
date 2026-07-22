'''
Ce fichier est une bibliothèque de fonction écrite dans la cadre du projet de traitement d'image d'un indicateur de niveau d'huile. 
Ces fonctions sont utilisés dans le programme principale 'main_pro.py'
'''

#Vérifier : sudo raspi-config PUIS interface options PUIS I2C PUIS enable
#Vérifier : i2cdetect -y 1 DOIT AFFICHER 60 ou 61


# ─── LIBRAIRIES ─────────────────────────────────────────────────────────────────────
# Installation : pip install adafruit-circuitpython-mcp4725
import board                    # Fournit les références aux broches I2C du Pi
import busio                    # Gère la communication I2C bas niveau
import adafruit_mcp4725         # Driver spécifique au composant MCP4725
#─────────────────────────────────────────────────────────────────────────────────────


#─── Constante importante ──────────────────────────────────────────────────────────
DAC_MAX = 4095		#Résolution du DAC 12 bits : 2^12 - 1 = 4095 
#─────────────────────────────────────────────────────────────────────────────────────


# ─── Conversion de la valeur en donnée traitable ────────────────────────────────────
def valeur_vers_dac(valeur: float, VALEUR_MIN: float, VALEUR_MAX: float)->int:
    """
    Convertit une valeur physique (float) en code DAC 12 bits (int 0–4095)
    évite tout débordement hors plage
    """
    # Limite la valeur dans la plage connue pour éviter un code hors [0, 4095]
    valeur_clampee = max(VALEUR_MIN, min(VALEUR_MAX, valeur))

    # Normalisation linéaire : mappe [VALEUR_MIN, VALEUR_MAX] vers [0, DAC_MAX]
    ratio = (valeur_clampee - VALEUR_MIN) / (VALEUR_MAX - VALEUR_MIN)

    # Calcule le code entier correspondant et l'arrondit
    code_dac = int(round(ratio * DAC_MAX))
    
    return code_dac
#─────────────────────────────────────────────────────────────────────────────────────


# ─── Envoie de la valeur vers le DAC => NV de tension  ──────────────────────────────
def envoyer_au_dac(valeur: float, VALEUR_MIN : float, VALEUR_MAX : float) -> None:
    """
    Initialise le bus I2C, instancie le DAC et envoie la valeur convertie.
    """
    # Crée le bus I2C en utilisant les broches SCL/SDA par défaut du Pi (GPIO3/GPIO2)
    i2c = busio.I2C(board.SCL, board.SDA)

    # Instancie le driver MCP4725 sur l'adresse I2C 0x60
    dac = adafruit_mcp4725.MCP4725(i2c, address=0x60)

    # Convertit la valeur physique en code DAC 12 bits
    code = valeur_vers_dac(valeur,VALEUR_MIN, VALEUR_MAX)

    # Envoie le code au registre du DAC via I2C
    # Le composant convertit immédiatement ce code en tension sur VOUT
    dac.raw_value = code

    # Calcul de la tension théorique en sortie pour le log (VCC = 3.3 V)
    tension_sortie = (code / DAC_MAX) * 3.3

    print(f"Valeur : {valeur} → code DAC : {code} → tension VOUT ≈ {tension_sortie:.3f} V")
#─────────────────────────────────────────────────────────────────────────────────────
