# ─── Bibliotheques ────────────────────────────────────────────────────────────
import cv2
import numpy as np
import sys
from redressement import detecter_pastilles, trier_losange, redimensionner_si_trop_grande
from picamzero import Camera
import os

# ─── PARAMETRES ───────────────────────────────────────────────────────────────
# Initialise la camera picamzero
cam = Camera()
cam.still_size = (396,378)     # Configuration de la resolution de la photo en pixels (largeur x hauteur)
cam.preview_size = (396,378)
cam.brightness = 0.05         # Luminosite (0.05 = tres legerement plus lumineux)
cam.white_balance = "auto"     # Balance des blancs automatique selon la temperature de couleur
cam.start_preview()
# Capture et sauvegarde la photo sur disque
cam.take_photo(f"/home/raspberry/Projet Indicateur EDF/image.jpg")
cam.stop_preview()

IMAGE = "indicateur1_angle1_pastilles.jpg"    # image en perspective avec pastilles
#IMAGE = "indicateur1.jpeg"
#IMAGE = "indicateur2.png"
#IMAGE = "image.jpg"

VALEUR_MIN = -20    # Valeur gravee sur le repere MIN du cadran
VALEUR_MAX = 100    # Valeur gravee sur le repere MAX du cadran

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

# ─── CONTOUR UNIQUEMENT A LA JONCTION ROUGE/BLANC ────────────────────────────
# Canny detecte tous les bords (rouge/noir, blanc/noir, rouge/blanc).
# Pour isoler uniquement la frontiere rouge/blanc :
# On dilate les deux masques pour les faire se chevaucher,
# l intersection des zones dilatees = uniquement la zone de contact rouge/blanc.
# Canny applique dans cette zone ne retient que le bord qui interesse.
bord_rouge = cv2.dilate(masque_rouge, kernel, iterations=3)     # Agrandit la zone rouge
bord_blanc = cv2.dilate(masque_blanc, kernel, iterations=3)     # Agrandit la zone blanche
jonction   = cv2.bitwise_and(bord_rouge, bord_blanc)            # Zone de chevauchement = frontiere

# Detecte les bords dans l image en gris par gradient d intensite
# 30 = seuil bas (bords faibles acceptes), 100 = seuil haut (bords forts certains)
contours_img = cv2.Canny(gris, 30, 100)
# Garde uniquement les bords situes dans la zone de jonction rouge/blanc
contour_frontiere = cv2.bitwise_and(contours_img, jonction)
cv2.imshow("Etape 7 : contour frontiere rouge/blanc", contour_frontiere)  # a effacer

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
