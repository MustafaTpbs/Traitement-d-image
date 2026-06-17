# ─── Bibliotheques ────────────────────────────────────────────────────────────
import cv2
import numpy as np
import sys
from redressement import detecter_pastilles, trier_losange, redimensionner_si_trop_grande
from picamzero import Camera
import time
import os

#─── PRISE DE PHOTO ───────────────────────────────────────────────────────────
cam = Camera()
cam.still_size = (1280, 960)  # Résolution plus élevée pour plus de détails
cam.preview_size = (640, 480)
#Paramètres anti-reflet / stabilisation exposure
cam.brightness = 0.0              # Pas de surexposition artificielle
cam.white_balance = "auto"
#Laisser l'AEC/AGC converger avant la capture
cam.start_preview()
time.sleep(2.0) # 2 s de stabilisation = moins de reflets "chauds"
cam.take_photo("/home/raspberry/Projet Indicateur EDF/image.jpg")
cam.stop_preview()
#Post-traitement anti-reflet : correction gamma + CLAHE
img_brute = cv2.imread("/home/raspberry/Projet Indicateur EDF/image.jpg")
#Correction gamma 1.3 : assombrit les hautes lumières sans toucher les foncés
gamma   = 1.3
lut     = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
img_brute = cv2.LUT(img_brute, lut)
#CLAHE sur le canal L (Lab) : améliore le contraste local, réduit l'effet voile des reflets
lab  = cv2.cvtColor(img_brute, cv2.COLOR_BGR2Lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lab[:, :, 0] = clahe.apply(lab[:, :, 0])
img_brute = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
cv2.imwrite("/home/raspberry/Projet Indicateur EDF/image.jpg", img_brute)

#IMAGE = "indicateur1_angle1_pastilles.jpg"    # image en perspective avec pastilles
#IMAGE = "indicateur1_pastilles.jpg"
#IMAGE = "indicateur2.png"

IMAGE = "image.jpg"

#IMAGE = "/home/raspberry/Projet Indicateur EDF/tests/base.png"
#IMAGE = "/home/raspberry/Projet Indicateur EDF/tests/basse.png"
#IMAGE = "/home/raspberry/Projet Indicateur EDF/tests/blanc.png"
#IMAGE = "/home/raspberry/Projet Indicateur EDF/tests/rouge.png"
#IMAGE = "/home/raspberry/Projet Indicateur EDF/tests/max.png"
#IMAGE = "/home/raspberry/Projet Indicateur EDF/tests/min.png"
#IMAGE = "/home/raspberry/Projet Indicateur EDF/tests/nuit.png"
#IMAGE = "/home/raspberry/Projet Indicateur EDF/tests/reflet.png"
#IMAGE = "/home/raspberry/Projet Indicateur EDF/tests/vrai1.png"
#IMAGE = "/home/raspberry/Projet Indicateur EDF/tests/vrai2.png"

VALEUR_MIN = -35    # Valeur gravee sur le repere MIN du cadran
VALEUR_MAX = 150    # Valeur gravee sur le repere MAX du cadran

# Convention trigo : 0 deg = droite, sens antihoraire, Y image inverse.
# MIN est en bas => ~240 deg   MAX est en haut => ~120 deg
ANGLE_MIN_DEG = 240.0
ANGLE_MAX_DEG = 120.0

# ─── CHARGEMENT ───────────────────────────────────────────────────────────────
# Charge l image depuis le dossier courant
img = cv2.imread(IMAGE)
if img is None:
    print(f"Erreur : impossible de charger '{IMAGE}'.")
    exit()
    
img = redimensionner_si_trop_grande(img, largeur_max=500)

cv2.imshow("Etape 1 : image de base", img)  # a effacer

# ─── REDRESSEMENT ───────────────────────────────────────────────────────────────
img_ref = cv2.imread("indicateur1_pastilles.jpg")
img_ref = redimensionner_si_trop_grande(img_ref, largeur_max=500)

# Détection des points avec vos fonctions importées
pts_test, _ = detecter_pastilles(img, "test")
pts_ref, _ = detecter_pastilles(img_ref, "reference")

# Calcul de la transformation
src = trier_losange(pts_test)
dst = trier_losange(pts_ref)
M = cv2.getPerspectiveTransform(src, dst)

# Application du redressement sur 'img'
h, w = img_ref.shape[:2]
img = cv2.warpPerspective(img, M, (w, h))

# Copie propre sur laquelle on va dessiner les annotations a la fin
img_resultat = img.copy()

# ─── DETECTION DU CERCLE ──────────────────────────────────────────────────────
# Conversion en niveaux de gris, necessaire pour HoughCircles car la méthode ne fonctionne pas s'il y a des couleurs
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Floute légèrement pour que la détection de cercle soit plus robuste, enlève le bruit autours des contours
# La fonction fait une moyenne de la couleur des 9 pixels voisin pour chaque pixel
gris_floute = cv2.GaussianBlur(gris, (9, 9), 0)
cv2.imshow("Etape 2 : gris flou", gris_floute)  # a effacer

# Detecte les cercles dans l image par la transformee de Hough
# méthode utilisé : Chaque pixel de contour (changement brusque d'intensité lumineuse) vote pour tous ces "centres possibles"
# dans la direction du gradient, l'endroit qui a le plus de vote est élu centre du cercle et le rayon constant est enregistré
cercles = cv2.HoughCircles(
    gris_floute, cv2.HOUGH_GRADIENT,
    dp=1,             # résolution de l'accumulateur. 1 => même résolution que image. 2 => deux fois plus flou. valeur recommandé => 1.5
    minDist=100,      # distance minimale entre deux centres cercles détectés
    param1=50,        # seuil pour la senisibilité de la détection de contours => 50 à 150
    param2=20,        # seuil d'accumulation, minimum à attaindre pour être considéré comme un cercle (plus bas = plus permissif) ==> 20 à 100 
    minRadius=0, maxRadius=300
)
if cercles is None:
    print("Aucun cercle trouve. Ajuster les parametres de HoughCircles.")
    cv2.waitKey(0)
    exit()

# cercles[0][0] = meilleur cercle detecte : [x_centre, y_centre, rayon]
cx, cy, rayon = np.round(cercles[0][0]).astype(int)
print(f"Cadran detecte => centre : ({cx}, {cy}), rayon : {rayon}px")

# Cree un masque circulaire : image noire avec un disque blanc a l emplacement du cadran
# Sert a ignorer tout ce qui est en dehors du cadran dans les etapes suivantes
masque_cercle = np.zeros(img.shape[:2], dtype=np.uint8)     # Image noire de meme taille que la photo
cv2.circle(masque_cercle, (cx, cy), rayon, 255, -1)         # -1 = remplissage complet du disque
cv2.imshow("Etape 3 : masque cadran", masque_cercle)  # a effacer

# ─── MASQUE DE LA ZONE UTILE ──────────────────────────────────────────────────
# Conversion des angles en radians pour les calculs trigonometriques
angle_min_rad = np.deg2rad(ANGLE_MIN_DEG)
angle_max_rad = np.deg2rad(ANGLE_MAX_DEG)
# Generation de 300 points regulierement espaces sur l arc du secteur
angles_arc = np.linspace(angle_min_rad, angle_max_rad, 300)

# Calcule les coordonnees de chaque point sur le bord du cercle
masque_secteur = np.zeros(img.shape[:2], dtype=np.uint8)
points_arc = np.array([
    [cx + rayon * np.cos(a), cy - rayon * np.sin(a)] # x = cx + r*cos(a),  y = cy - r*sin(a)  (le - compense l axe Y inverse de l image)
    for a in angles_arc
], dtype=np.int32)
# Construit le polygone du secteur = centre + tous les points de l arc
# fillPoly remplit ce polygone en blanc dans le masque
cv2.fillPoly(masque_secteur, [np.vstack([[cx, cy], points_arc])], 255)
cv2.imshow("Etape 4 : masque zone utile", masque_secteur)  # a effacer
'''
# ─── DETECTION DU ROUGE ET DU BLANC DANS LA ZONE UTILE ───────────────────────
# Conversion en HSV (Teinte, Saturation, Valeur)
# Plus pratique que BGR pour isoler une couleur precise
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Noyau elliptique 5x5 utilise pour les operations morphologiques
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #le kernel est une forme géométrique simple pour dilater/boucher plus tard les zones utiles

# Le rouge occupe deux extremites du spectre HSV (0-10 et 160-180 deg)
# inRange cree un masque blanc la ou la couleur est dans la plage
# bitwise_or fusionne les deux masques en un seul
masque_rouge = cv2.bitwise_or(
    cv2.inRange(hsv, np.array([0,   80,  80]), np.array([10,  255, 255])),
    cv2.inRange(hsv, np.array([160, 80,  80]), np.array([180, 255, 255]))
)
# Limite la detection a la zone utile uniquement
masque_rouge = cv2.bitwise_and(masque_rouge, masque_secteur)
# MORPH_CLOSE = bouche les petits trous dans la zone rouge
masque_rouge = cv2.morphologyEx(masque_rouge, cv2.MORPH_CLOSE, kernel)

# Le blanc : saturation tres basse (S proche de 0) et luminosite tres haute (V proche de 255)
masque_blanc = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 80, 255]))
masque_blanc = cv2.bitwise_and(masque_blanc, masque_secteur)
masque_blanc = cv2.morphologyEx(masque_blanc, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Etape 5 : masque rouge", masque_rouge)  # a effacer
cv2.imshow("Etape 6 : masque blanc", masque_blanc)  # a effacer
'''

# ─── DETECTION DU ROUGE ET DU BLANC DANS LA ZONE UTILE ───────────────────────
lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   # conservé uniquement pour masque_blanc
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #le kernel est une forme géométrique simple pour dilater/boucher plus tard les zones utiles

# Zone rouge/rose : canal a* élevé (rouge dans LAB)
masque_rouge = cv2.inRange(lab, np.array([20, 133, 0]), np.array([255, 255, 255]))
masque_rouge = cv2.bitwise_and(masque_rouge, masque_secteur) # bitwise_or fusionne les deux masques en un seul
masque_rouge = cv2.morphologyEx(masque_rouge, cv2.MORPH_OPEN,  kernel, iterations=1)
masque_rouge = cv2.morphologyEx(masque_rouge, cv2.MORPH_CLOSE, kernel, iterations=3)

# Zone blanche = tout le secteur MOINS la zone rouge
# Plus besoin de détecter le blanc explicitement : c'est ce qui reste
masque_blanc = cv2.bitwise_and(
    cv2.bitwise_not(masque_rouge),
    masque_secteur
)
masque_blanc = cv2.morphologyEx(masque_blanc, cv2.MORPH_OPEN, kernel, iterations=1)

nb_r = cv2.countNonZero(masque_rouge)
nb_b = cv2.countNonZero(masque_blanc)
print(f"[DIAG] masque_rouge={nb_r}px  masque_blanc={nb_b}px")
if nb_r < 200:
    print("[DIAG] ATTENTION : rouge trop peu détecté — vérifier la photo")

cv2.imshow("Etape 5 : masque rouge", masque_rouge)
cv2.imshow("Etape 6 : masque blanc", masque_blanc)

# ─── CONTOUR UNIQUEMENT A LA JONCTION ROUGE/BLANC ────────────────────────────
# Canny detecte tous les bords (rouge/noir, blanc/noir, rouge/blanc).
# Pour isoler uniquement la frontiere rouge/blanc :
# On dilate les deux masques pour les faire se chevaucher,
# l intersection des zones dilatees = uniquement la zone de contact rouge/blanc.
# Canny applique dans cette zone ne retient que le bord qui interesse.
bord_rouge = cv2.dilate(masque_rouge, kernel, iterations=3)     # Agrandit la zone rouge
bord_blanc = cv2.dilate(masque_blanc, kernel, iterations=3)     # Agrandit la zone blanche #ici
jonction   = cv2.bitwise_and(bord_rouge, bord_blanc)            # Zone de chevauchement = frontiere

# Detecte les bords dans l image en gris par gradient d intensite
# 30 = seuil bas (bords faibles acceptes), 100 = seuil haut (bords forts certains)
contours_img = cv2.Canny(gris, 30, 100)
# Garde uniquement les bords situes dans la zone de jonction rouge/blanc
contour_frontiere = cv2.bitwise_and(contours_img, jonction)
cv2.imshow("Etape 7 : contour frontiere rouge/blanc", contour_frontiere)  # a effacer
'''
# ─── HOUGH LINEAIRE SUR LE CONTOUR DE LA FRONTIERE ───────────────────────────
# La frontiere rouge/blanc est une droite radiale partant du centre.
# HoughLinesP detecte des segments de droite dans une image binaire par vote.
segments = cv2.HoughLinesP(
    contour_frontiere,
    rho=1,                      # Resolution en pixels de l accumulateur de distance, cherche à 1 pixel près
    theta=np.pi/180,            # Resolution angulaire : 1 degre par pas
    threshold=15,               # Nombre minimum de votes (pixel alignés) pour valider un segment
    minLineLength=rayon * 0.2,  # Longueur minimale acceptee : 20% du rayon du cadran de l'indicateur
    maxLineGap=15               # Ecart maximal entre deux morceaux d un meme segment
)

if segments is None:
    print("Aucun segment trouve. Verifier les masques rouge et blanc.")
    cv2.waitKey(0)
    exit()

# Parmi tous les segments detectes, on garde celui dont la droite portante
# passe le plus pres du centre du cadran 
meilleur_segment = None
distance_min = float("inf")

for seg in segments:
    x1, y1, x2, y2 = seg[0]
    dx, dy = x2 - x1, y2 - y1
    # Formule distance d un point a une droite : |dy*cx - dx*cy + x2*y1 - y2*x1| / longueur
    dist = abs(dy * cx - dx * cy + x2 * y1 - y2 * x1) / (np.sqrt(dx**2 + dy**2) + 1e-6)
    if dist < distance_min:
        distance_min     = dist
        meilleur_segment = seg[0]

x1, y1, x2, y2 = meilleur_segment
print(f"Segment frontiere : ({x1},{y1}) -> ({x2},{y2}), distance au centre : {distance_min:.1f}px")
'''
# ─── HOUGH LINEAIRE SUR LE CONTOUR DE LA FRONTIERE ───────────────────────────
segments = cv2.HoughLinesP(
    contour_frontiere,
    rho=1,
    theta=np.pi/180,
    threshold=15,
    minLineLength=rayon * 0.2,
    maxLineGap=15
)

if segments is None:
    print("Aucun segment trouve. Verifier les masques rouge et blanc.")
    cv2.waitKey(0)
    exit()

# ── Calcul du score de chaque segment ─────────────────────────────────────────
def score_segment(seg, cx, cy):
    x1, y1, x2, y2 = seg
    dx, dy = x2-x1, y2-y1
    longueur = np.sqrt(dx**2 + dy**2) + 1e-6
    dist_centre = abs(dy*cx - dx*cy + x2*y1 - y2*x1) / longueur
    d1 = (x1-cx)**2 + (y1-cy)**2
    d2 = (x2-cx)**2 + (y2-cy)**2
    px_l, py_l = (x2,y2) if d2>d1 else (x1,y1)
    a_seg = np.rad2deg(np.arctan2(-(py_l-cy), px_l-cx)) % 360
    # Score : distance au centre, pénalité si court
    score = dist_centre - longueur * 0.3
    return score, a_seg, longueur, dist_centre

MARGE_BORD = 5.0   # °, pour détecter un faux bord sur le contour du masque

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

# Priorité 1 : meilleur candidat normal (frontière réelle dans le secteur)
# Priorité 2 : si aucun candidat normal → l'aiguille est vraiment à MIN ou MAX
if candidats_normaux:
    candidats_normaux.sort(key=lambda x: x[0])
    meilleur_segment = candidats_normaux[0][1]
    print(f"[DIAG] Frontière normale à {candidats_normaux[0][2]:.0f}°")
elif candidats_bord:
    candidats_bord.sort(key=lambda x: x[0])
    meilleur_segment = candidats_bord[0][1]
    print(f"[DIAG] Fallback bord MIN/MAX : aiguille à l'extrémité ({candidats_bord[0][2]:.0f}°)")
else:
    # Dernier recours : segment le plus proche du centre sans aucun filtre
    meilleur_segment = min(segments, key=lambda s: score_segment(s[0], cx, cy)[0])[0]
    print("[DIAG] Fallback global")

x1, y1, x2, y2 = meilleur_segment
print(f"Segment frontiere : ({x1},{y1}) -> ({x2},{y2})")


# ─── CALCUL DE L'ANGLE DE LA FRONTIERE ───────────────────────────────────────
# Le segment a deux extremites. On prend la plus eloignee du centre :
# c est elle qui donne la vraie direction vers le bord du cadran.
d1 = (x1 - cx)**2 + (y1 - cy)**2   # Distance au carre de l extremite 1 au centre
d2 = (x2 - cx)**2 + (y2 - cy)**2   # Distance au carre de l extremite 2 au centre
px_loin, py_loin = (x2, y2) if d2 > d1 else (x1, y1)

# arctan2 calcule l angle de la direction centre -> point_loin
# Le - devant (py_loin - cy) compense l axe Y qui pointe vers le bas en image
angle_frontiere_rad = np.arctan2(-(py_loin - cy), px_loin - cx)
angle_frontiere_deg = np.rad2deg(angle_frontiere_rad) % 360
print(f"Angle frontiere : {angle_frontiere_deg:.1f} deg")

# ─── CALCUL DE LA VALEUR ──────────────────────────────────────────────────────
# Mappage de l angle mesure sur l echelle physique MIN -> MAX
# plage_deg = amplitude totale entre le repere MIN et le repere MAX
# mesure_deg = position de la frontiere depuis le repere MIN
# ratio = position normalisee entre 0.0 (MIN) et 1.0 (MAX)
plage_deg        = ANGLE_MIN_DEG - ANGLE_MAX_DEG
mesure_deg       = ANGLE_MIN_DEG - angle_frontiere_deg
ratio            = max(0.0, min(1.0, mesure_deg / plage_deg))
valeur_numerisee = VALEUR_MIN + ratio * (VALEUR_MAX - VALEUR_MIN)
print(f"Ratio : {ratio:.2%}  =>  Valeur : {valeur_numerisee:.1f}")

# ─── DESSIN DES ANNOTATIONS ───────────────────────────────────────────────────

# Surbrillance semi-transparente sur les pixels detectes comme rouges
# addWeighted melange overlay (30%) et img_resultat (70%)
overlay = img_resultat.copy()
overlay[masque_rouge > 0] = [0, 0, 200]
img_resultat = cv2.addWeighted(overlay, 0.3, img_resultat, 0.7, 0)

# Cercle vert sur le contour du cadran detecte
cv2.circle(img_resultat, (cx, cy), rayon, (0, 200, 0), 1)

# Arc orange sur le bord du cercle pour visualiser le secteur de recherche
for i in range(len(angles_arc) - 1):
    p1 = (int(cx + rayon * np.cos(angles_arc[i])),   int(cy - rayon * np.sin(angles_arc[i])))
    p2 = (int(cx + rayon * np.cos(angles_arc[i+1])), int(cy - rayon * np.sin(angles_arc[i+1])))
    cv2.line(img_resultat, p1, p2, (0, 200, 255), 2)

# Graduations de 10 en 10 sur l arc de l echelle physique
valeurs_grad = range(int(VALEUR_MIN), int(VALEUR_MAX) + 1, 10)
for val in valeurs_grad:
    t     = (val - VALEUR_MIN) / (VALEUR_MAX - VALEUR_MIN)         # Position normalisee 0->1
    angle = angle_min_rad + t * (angle_max_rad - angle_min_rad)    # Angle correspondant en radians

    px_ext = int(cx + rayon        * np.cos(angle))    # Point sur le bord du cercle
    py_ext = int(cy - rayon        * np.sin(angle))
    px_int = int(cx + (rayon - 12) * np.cos(angle))    # Point en retrait de 12px pour le tiret
    py_int = int(cy - (rayon - 12) * np.sin(angle))
    px_lbl = int(cx + (rayon + 16) * np.cos(angle))    # Point a l exterieur pour l etiquette
    py_lbl = int(cy - (rayon + 16) * np.sin(angle))

    cv2.line(img_resultat, (px_int, py_int), (px_ext, py_ext), (255, 220, 0), 1)
    cv2.putText(img_resultat, f"{val:.0f}", (px_lbl - 10, py_lbl + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 0), 1)

# Etiquettes MIN et MAX aux extremites de l echelle physique
for label, angle_r in [("MIN", angle_min_rad), ("MAX", angle_max_rad)]:
    px = int(cx + (rayon + 22) * np.cos(angle_r))
    py = int(cy - (rayon + 22) * np.sin(angle_r))
    cv2.putText(img_resultat, label, (px - 12, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

# Ligne cyan du centre vers le bord representant la frontiere mesuree
px_bord = int(cx + rayon * np.cos(angle_frontiere_rad))
py_bord = int(cy - rayon * np.sin(angle_frontiere_rad))
cv2.line(img_resultat, (cx, cy), (px_bord, py_bord), (0, 220, 220), 2)

# Point blanc au centre du cadran
cv2.circle(img_resultat, (cx, cy), 3, (255, 255, 255), -1)

# Valeur numerisee en bas a gauche du cadran
cv2.putText(img_resultat, f"Niveau : {valeur_numerisee:.1f}",
            (cx - rayon + 5, cy + rayon + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# ─── SAUVEGARDE ET AFFICHAGE ──────────────────────────────────────────────────
cv2.imwrite("resultat_analyse.png", img_resultat)   # Sauvegarde l image annotee sur disque
print("Image sauvegardee : resultat_analyse.png")

cv2.imshow("Analyse niveau huile", img_resultat)
cv2.waitKey(0)          # Attend une touche clavier avant de fermer
cv2.destroyAllWindows() # Ferme toutes les fenetres OpenCV
