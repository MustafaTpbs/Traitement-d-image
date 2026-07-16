# ─── Bibliotheques ──────────────────────────────────────────────────────────────────
import cv2	#Traitement d'image en temps réel.
import numpy as np	#Fonctions mathématiques sur des matrices/tableaux de données.
from redressement import detecter_pastilles, trier_losange, redimensionner_si_trop_grande	#Deuxième fichier .py du projet.
from picamzero import Camera	#Fonctions de la caméra de la raspberry.
from digital_to_analog import valeur_vers_dac, envoyer_au_dac
from pathlib import Path
#─────────────────────────────────────────────────────────────────────────────────────


#─── Chemin du dossier ───────────────────────────────────────────────────────────────
chemin = Path(__file__).resolve().parent
#─────────────────────────────────────────────────────────────────────────────────────


#─── PRISE DE PHOTO ──────────────────────────────────────────────────────────────────
cam = Camera()	#Initialisation de ma variable caméra.
cam.still_size = (1280, 960)	#Résolution plus élevée pour plus de détails.

#Paramètres anti-reflet
cam.brightness = 0.0	#Pas de surexposition artificielle à la luminosité [-1 => 1]
cam.white_balance = "auto"	#Peut être 'daylight' si la photo est prise en journée. 'Auto' pour plus de robustesse.

cam.take_photo(chemin/"image.jpg")	#Prend la photo et enrigistre à l'adresse voulu

#Post-traitement anti-reflet
img_brute = cv2.imread(chemin/"image.jpg")	#Lit l'image à traiter et la stocke dans une variable

gamma   = 1.3	#Correction : assombrit les hautes lumières sans toucher les foncés
lut     = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)		#Création d'une 'colormap' personalisé pour traiter les hautes lumières
img_brute = cv2.LUT(img_brute, lut) 	#voir : wirelog.net/posts/2022-04-11-custom-colormap-opencv/ 

lab  = cv2.cvtColor(img_brute, cv2.COLOR_BGR2Lab)	#Conversion de couleur pré-traitement voir : caterbum.com/blog/python-opencv-color-conversion-library-explorer
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) 	#voir : geeksforgeeks.org/python/clahe-histogram-eqalization-opencv/
lab[:, :, 0] = clahe.apply(lab[:, :, 0]) 	#Correction : améliore le contraste local
img_brute = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR) 	#Re-conversion des couleurs post-traitement

cv2.imwrite(chemin/"image.jpg", img_brute)	#Sauvegarde l'image traité
#──────────────────────────────────────────────────────────────────────────────────────


#─── Variables importantes ────────────────────────────────────────────────────────────
IMAGE = "image.jpg" 	#Photographie qui va être utilisé par la suite
#IMAGE = chemin/"tests pastilles sur feuille/test_vif2.png" #RAPPORT

VALEUR_MIN = -35    # Valeur gravee sur le repere MIN du cadran [°C]
VALEUR_MAX = 150    # Valeur MAX du cadran [°C]

#La trigo en info : le 'x' est à sa position conventionnel (à droite) MAIS le 'y' est à la position du '-y' conventionnel. SENS HORAIRE
ANGLE_MIN_DEG = 240.0 	# MIN est en bas => ~240 deg 
ANGLE_MAX_DEG = 120.0 	#MAX est en haut => ~120 deg
#──────────────────────────────────────────────────────────────────────────────────────


# ─── CHARGEMENT ──────────────────────────────────────────────────────────────────────
#Vérifie si l'image existe bien SINON exit programme
img = cv2.imread(IMAGE)
if img is None:
    print(f"Erreur : impossible de charger '{IMAGE}'.")
    exit(1)

#Utilisation de la fonction écrite dans le fichier redressement.py
img = redimensionner_si_trop_grande(img, largeur_max=500) 	#Redimensionner permet de faciliter le traitement car sinon image trop lourde à traiter
#──────────────────────────────────────────────────────────────────────────────────────


# ─── REDRESSEMENT ────────────────────────────────────────────────────────────────────
#img_ref = cv2.imread("indicateur1_pastilles.jpg") 		#Image de reference pour la position des pastilles en position de face (dessus)
img_ref = cv2.imread(chemin/"tests pastilles sur feuille/ref_vrf.png") 		#Image de reference pour la position des pastilles en position de face (sur feuille)
img_ref = redimensionner_si_trop_grande(img_ref, largeur_max=500) 		#Redimensionner car sinon image trop lourde à traiter

# Détection des points avec vos fonctions importées (fonction de redressement.py)
pts_test, _ = detecter_pastilles(img, "test")
pts_ref, _ = detecter_pastilles(img_ref, "reference")

# Calcul de la transformation 
src = trier_losange(pts_test) 	#Trouve respectivement la pastille en haut,droite,bas et gauche (fonction de redressement.py) 
dst = trier_losange(pts_ref)  	#Trouve respectivement la pastille en haut,droite,bas et gauche (fonction de redressement.py) )
M = cv2.getPerspectiveTransform(src, dst) 	#Calcule la matrice de transformation

# Application du redressement sur 'img'
h, w = img_ref.shape[:2] 	#Récupère la forme de l'image référence, cela permet de 'zoomer' indirectement sur l'iamge principale
img = cv2.warpPerspective(img, M, (w, h)) 	#Applique la matrice calculé sur toute l'image, voir : theailearner.com/tag/cv2-getperspectivetransform/

# Copie propre sur laquelle les annotations seront fait à la fin
img_resultat = img.copy()
#──────────────────────────────────────────────────────────────────────────────────────


# ─── DETECTION DU CERCLE ─────────────────────────────────────────────────────────────
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Conversion en niveaux de gris, necessaire pour HoughCircles car la méthode ne fonctionne pas s'il y a des couleurs
gris_floute = cv2.GaussianBlur(gris, (9, 9), 0) # Floute légèrement pour que la détection de cercle soit plus robuste, enlève le bruit autours des contours (fait une moyenne de la couleur des 9 pixels voisin pour chaque pixel)

'''     --Transformee de Hough--
==> Detecte les cercles dans l'image
Méthode utilisé : Chaque pixel de contour (changement brusque d'intensité lumineuse) vote pour tous ces "centres possibles"
dans la direction du gradient, l'endroit qui a le plus de vote est élu centre du cercle et le rayon constant est enregistré '''
cercles = cv2.HoughCircles(gris_floute, cv2.HOUGH_GRADIENT,
    dp=1,             # Résolution de l'accumulateur. 1 => même résolution que image. 2 => deux fois plus flou.
    minDist=100,      # Distance minimale entre deux centres cercles détectés
    param1=50,        # Seuil pour la sensibilité de la détection de contours => 50 à 150
    param2=20,        # seuil d'accumulation, minimum à attaindre pour être considéré comme un cercle (plus bas = plus permissif) ==> 20 à 100 
    minRadius=0, maxRadius=300)
    
if cercles is None:
    print("Aucun cercle trouve. Ajuster les parametres de HoughCircles.")
    exit(1)

cx, cy, rayon = np.round(cercles[0][0]).astype(int) 	#cercles[0][0] = meilleur cercle detecte [x_centre, y_centre, rayon]
print(f"Cadran detecte => centre : ({cx}, {cy}), rayon : {rayon}px")

'''Creation  d'un masque circulaire : image noire avec un disque blanc a l'emplacement du cadran
==> Sert a ignorer tout ce qui est en dehors du cadran dans les etapes suivantes'''
masque_cercle = np.zeros(img.shape[:2], dtype=np.uint8)     # Image noire de meme taille que la photo
cv2.circle(masque_cercle, (cx, cy), rayon, 255, -1)         # -1 = remplissage complet par du blanc du disque
#──────────────────────────────────────────────────────────────────────────────────────


# ─── MASQUE DE LA ZONE UTILE ─────────────────────────────────────────────────────────
# Conversion des angles en radians pour les calculs trigonometriques
angle_min_rad = np.deg2rad(ANGLE_MIN_DEG)
angle_max_rad = np.deg2rad(ANGLE_MAX_DEG)

# Generation de 300 points regulierement espaces
angles_arc = np.linspace(angle_min_rad, angle_max_rad, 300)

# Calcule les coordonnees de chaque point sur le bord du cercle
masque_secteur = np.zeros(img.shape[:2], dtype=np.uint8)
points_arc = np.array([
    [cx + rayon * np.cos(a), cy - rayon * np.sin(a)]	 # x = cx + r*cos(a),  y = cy - r*sin(a)  (le - compense l axe Y inverse de l image)
    for a in angles_arc
], dtype=np.int32)

# Construit le polygone du secteur = centre + tous les points de l arc
cv2.fillPoly(masque_secteur, [np.vstack([[cx, cy], points_arc])], 255) 		# fillPoly remplit ce polygone en blanc dans le masque
#──────────────────────────────────────────────────────────────────────────────────────


# ─── DETECTION DU ROUGE ET DU BLANC DANS LA ZONE UTILE ───────────────────────────────
#Définition des variables locales
lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab) 	 #Conversion des couleurs afin d'avoir un équivalent du rouge plus prononcé
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   #HSV fait ressortir le rouge
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #le kernel est une forme géométrique simple pour dilater/boucher plus tard les zones utiles

# Zone rouge/rose : canal a* élevé (rouge dans LAB)
masque_rouge = cv2.inRange(lab, np.array([20, 133, 0]), np.array([255, 255, 255]))	#Définition de l'équivalent du "rouge" dans LAB 
masque_rouge = cv2.bitwise_and(masque_rouge, masque_secteur) 	#On garde le rouge de la zone utile avec le masque_secteur
masque_rouge = cv2.morphologyEx(masque_rouge, cv2.MORPH_OPEN,  kernel, iterations=1) 	#Utilisation du kernel pour enlever le petit bruit
masque_rouge = cv2.morphologyEx(masque_rouge, cv2.MORPH_CLOSE, kernel, iterations=3) 	#Utilisation du kernel pour enlever le petit bruit
#voir : geeksforgeeks.org/python/python-opencv-morphological-operations/

# Zone blanche = tout le secteur MOINS la zone rouge, pas de détection explicite du blanc
masque_blanc = cv2.bitwise_and(
    cv2.bitwise_not(masque_rouge),
    masque_secteur)
    
masque_blanc = cv2.morphologyEx(masque_blanc, cv2.MORPH_OPEN, kernel, iterations=1) #Utilisation du kernel pour enlever le petit bruit

#Vérification de la quantité de rouge détectée (facultatif)
nb_r = cv2.countNonZero(masque_rouge)
nb_b = cv2.countNonZero(masque_blanc)
print(f"[DIAG] masque_rouge={nb_r}px  masque_blanc={nb_b}px")
if nb_r < 200:
    print("[DIAG] ATTENTION : rouge trop peu détecté — vérifier la photo")
#──────────────────────────────────────────────────────────────────────────────────────


# ─── CONTOUR UNIQUEMENT A LA JONCTION ROUGE/BLANC ────────────────────────────────────
'''Canny detecte tous les bords (rouge/noir, blanc/noir, rouge/blanc).
Pour isoler uniquement la frontiere rouge/blanc :
On dilate les deux masques pour les faire se chevaucher,
Donc : l intersection des zones dilatees = uniquement la zone de contact rouge/blanc.
Canny applique dans cette zone ne retient que le bord qui interesse. '''
bord_rouge = cv2.dilate(masque_rouge, kernel, iterations=5)     # Agrandit la zone rouge
bord_blanc = cv2.dilate(masque_blanc, kernel, iterations=3)      # Agrandit la zone blanche 
jonction   = cv2.bitwise_and(bord_rouge, bord_blanc)             # Zone de chevauchement = frontiere

# Detecte les bords dans l image en gris par gradient d intensite
contours_img = cv2.Canny(gris, 30, 100) 	#30 = seuil bas (bords faibles acceptes), 100 = seuil haut (bords forts certains)
#voir : geeksforgeeks.org/python/python-opencv-canny-function/

# Garde uniquement les bords situes dans la zone de jonction rouge/blanc
contour_frontiere = cv2.bitwise_and(contours_img, jonction)
#──────────────────────────────────────────────────────────────────────────────────────


# ─── HOUGH LINEAIRE SUR LE CONTOUR DE LA FRONTIERE ───────────────────────────────────
'''     --Transformee de Hough--
==> Detecte les lignes dans l'image par vote. Ici l'image c'est le contour rouge/blanc'''
segments = cv2.HoughLinesP(
    contour_frontiere,
    rho=1, 					     # Resolution en pixels de l accumulateur de distance, cherche à 1 pixel près
    theta=np.pi/180, 			 # Resolution angulaire : 1 degre par pas
    threshold=15,			     # Nombre minimum de votes (pixel alignés) pour valider un segment
    minLineLength=rayon * 0.2,	 # Longueur minimale acceptee : 20% du rayon du cadran de l'indicateur
    maxLineGap=15)				 # Ecart maximal entre deux morceaux d un meme segment
 
if segments is None:
    print("Aucun segment trouve. Verifier les masques rouge et blanc.")
    exit(1)
#──────────────────────────────────────────────────────────────────────────────────────


# ── DETERMINER LE MEILLEUR SEGMENT ───────────────────────────────────────────────────
def score_segment(seg, cx, cy):
    x1, y1, x2, y2 = seg		#Définition des coordonnées 
    dx, dy = x2-x1, y2-y1		#Calcul de la longueur de l'axe des x puis de y 
    longueur = np.sqrt(dx**2 + dy**2) + 1e-6 	#Calcul de la longueur du segment
    
    dist_centre = abs(dy*cx - dx*cy + x2*y1 - y2*x1) / longueur 	#formule de distance d'un point à une droite, distance perpendiculaire
    
    d1 = (x1-cx)**2 + (y1-cy)**2	#carrés des distances (plus rapide que la racine) 
    d2 = (x2-cx)**2 + (y2-cy)**2	#carrés des distances (plus rapide que la racine) 
    px_l, py_l = (x2,y2) if d2>d1 else (x1,y1) 	#identifie l'extrémité du segment la plus éloignée du centre
    a_seg = np.rad2deg(np.arctan2(-(py_l-cy), px_l-cx)) % 360 	 #convertit les coordonnées cartésiennes en angle polaire
    
    # Score : distance au centre, pénalité si court
    score = dist_centre - longueur * 0.3 #les segments longs et proches du centre ont les scores les plus bas, ils apparaissent en tête de liste lors du tri.
    
    return score, a_seg, longueur, dist_centre


MARGE_BORD = 15.0    

# Séparer les segments "bord de masque" (angle ≈ MIN ou MAX) des vrais candidats
candidats_normaux = []
candidats_bord    = []

for seg in segments:
    score, a_seg, longueur, dist_centre = score_segment(seg[0], cx, cy)
    # Hors secteur : ignorer
    if not (ANGLE_MAX_DEG - MARGE_BORD <= a_seg <= ANGLE_MIN_DEG + MARGE_BORD):
        continue
    est_bord_masque = (
        abs(a_seg - ANGLE_MIN_DEG) < MARGE_BORD or
        abs(a_seg - ANGLE_MAX_DEG) < MARGE_BORD
    )
    if est_bord_masque:
        candidats_bord.append((score, seg[0], a_seg))
    else:
        candidats_normaux.append((score, seg[0], a_seg))

#Priorité 1 : meilleur candidat normal (frontière réelle dans le secteur)
if candidats_normaux:
    candidats_normaux.sort(key=lambda x: x[0])
    meilleur_segment = candidats_normaux[0][1]
    print(f"Frontière normale à {candidats_normaux[0][2]:.0f}°") 
    
#Priorité 2 : si aucun candidat normal → l'aiguille est vraiment à MIN ou MAX
elif candidats_bord:
    candidats_bord.sort(key=lambda x: x[0])
    meilleur_segment = candidats_bord[0][1]
    print(f"Frontière bord MIN/MAX : aiguille à l'extrémité ({candidats_bord[0][2]:.0f}°)") 
else:
#Dernier recours : segment le plus proche du centre sans aucun filtre
    meilleur_segment = min(segments, key=lambda s: score_segment(s[0], cx, cy)[0])[0]
    print("Frontière global") 

x1, y1, x2, y2 = meilleur_segment
print(f"Segment frontiere : ({x1},{y1}) -> ({x2},{y2})")	
#──────────────────────────────────────────────────────────────────────────────────────


# ─── CALCUL DE L'ANGLE DE LA FRONTIERE ───────────────────────────────────────────────
# Le segment a deux extremites. On prend la plus eloignee du centre : c'est elle qui donne la vraie direction vers le bord du cadran.
d1 = (x1 - cx)**2 + (y1 - cy)**2   # Distance au carre de l extremite 1 au centre
d2 = (x2 - cx)**2 + (y2 - cy)**2   # Distance au carre de l extremite 2 au centre
px_loin, py_loin = (x2, y2) if d2 > d1 else (x1, y1)

angle_frontiere_rad = np.arctan2(-(py_loin - cy), px_loin - cx) 	# arctan2 calcule l angle de la direction centre -> point_loin
angle_frontiere_deg = np.rad2deg(angle_frontiere_rad) % 360 	#Conversion en degré de l'angle
print(f"Angle frontiere : {angle_frontiere_deg:.1f} deg") 	
#──────────────────────────────────────────────────────────────────────────────────────


# ─── CALCUL DE LA VALEUR ─────────────────────────────────────────────────────────────
# Mappage de l angle mesure sur l echelle physique MIN -> MAX
plage_deg  = ANGLE_MIN_DEG - ANGLE_MAX_DEG 		#amplitude totale entre le repere MIN et le repere MAX
mesure_deg = ANGLE_MIN_DEG - angle_frontiere_deg 		#position de la frontiere depuis le repere MIN
ratio      = max(0.0, min(1.0, mesure_deg / plage_deg)) 		#ratio = position normalisee entre 0.0 (MIN) et 1.0 (MAX)
valeur_numerisee = VALEUR_MIN + ratio * (VALEUR_MAX - VALEUR_MIN)
print(f"Ratio : {ratio:.2%}  =>  Valeur : {valeur_numerisee:.1f}") 	
#──────────────────────────────────────────────────────────────────────────────────────


# ─── Envoie en niveau de tenstion ────────────────────────────────────────────────────
#envoyer_au_dac(valeur_numerisee, VALEUR_MIN, VALEUR_MAX)
#──────────────────────────────────────────────────────────────────────────────────────
