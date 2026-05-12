# ─── Bibliotheques ────────────────────────────────────────────────────────────
import cv2
import numpy as np

# ─── PARAMETRES ───────────────────────────────────────────────────────────────
IMAGE      = "indicateur1.jpeg"
VALEUR_MIN = -20
VALEUR_MAX = 100

# Convention trigo : 0 deg = droite, sens antihoraire, Y image inverse.
# MIN est en bas => ~240 deg    MAX est en haut => ~120 deg
ANGLE_MIN_DEG = 240.0
ANGLE_MAX_DEG = 120.0

# ─── CHARGEMENT ───────────────────────────────────────────────────────────────
img = cv2.imread(IMAGE)
if img is None:
    print(f"Erreur : impossible de charger '{IMAGE}'.")
    exit()

img_resultat = img.copy()
cv2.imshow("Etape 1 : image de base", img_resultat)  # a effacer

# ─── DETECTION DU CERCLE ──────────────────────────────────────────────────────
gris        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gris_floute = cv2.GaussianBlur(gris, (9, 9), 0)
cv2.imshow("Etape 2 : gris flou", gris_floute)  # a effacer

cercles = cv2.HoughCircles(
    gris_floute, cv2.HOUGH_GRADIENT,
    dp=1, minDist=100, param1=50, param2=20,
    minRadius=0, maxRadius=300
)
if cercles is None:
    print("Aucun cercle trouve. Ajuster les parametres de HoughCircles.")
    exit()

cx, cy, rayon = np.round(cercles[0][0]).astype(int)
print(f"Cadran detecte => centre : ({cx}, {cy}), rayon : {rayon}px")

masque_cercle = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.circle(masque_cercle, (cx, cy), rayon, 255, -1)
cv2.imshow("Etape 3 : masque cadran", masque_cercle)  # a effacer

# ─── MASQUE DE LA ZONE UTILE ──────────────────────────────────────────────────
angle_min_rad = np.deg2rad(ANGLE_MIN_DEG)
angle_max_rad = np.deg2rad(ANGLE_MAX_DEG)
angles_arc    = np.linspace(angle_min_rad, angle_max_rad, 300)

masque_secteur = np.zeros(img.shape[:2], dtype=np.uint8)
points_arc = np.array([
    [cx + rayon * np.cos(a), cy - rayon * np.sin(a)]
    for a in angles_arc
], dtype=np.int32)
cv2.fillPoly(masque_secteur, [np.vstack([[cx, cy], points_arc])], 255)
cv2.imshow("Etape 4 : masque zone utile", masque_secteur)  # a effacer

# ─── DETECTION DU ROUGE ET DU BLANC DANS LA ZONE UTILE ───────────────────────
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

masque_rouge = cv2.bitwise_or(
    cv2.inRange(hsv, np.array([0,   80,  80]), np.array([10,  255, 255])),
    cv2.inRange(hsv, np.array([160, 80,  80]), np.array([180, 255, 255]))
)
masque_rouge = cv2.bitwise_and(masque_rouge, masque_secteur)
masque_rouge = cv2.morphologyEx(masque_rouge, cv2.MORPH_CLOSE, kernel)

masque_blanc = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))
masque_blanc = cv2.bitwise_and(masque_blanc, masque_secteur)
masque_blanc = cv2.morphologyEx(masque_blanc, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Etape 5 : masque rouge", masque_rouge)  # a effacer
cv2.imshow("Etape 6 : masque blanc", masque_blanc)  # a effacer

# ─── CONTOUR UNIQUEMENT A LA JONCTION ROUGE/BLANC ────────────────────────────
# On dilate les deux masques pour qu'ils se chevauchent un peu,
# puis on prend leur intersection : seuls les pixels a la frontiere
# rouge/blanc sont dans les deux zones dilatees en meme temps.
# Canny sur l'image originale dans cette zone donne le contour net
# de la frontiere, sans les bords parasites rouge/noir ou blanc/noir.
bord_rouge = cv2.dilate(masque_rouge, kernel, iterations=3)
bord_blanc = cv2.dilate(masque_blanc, kernel, iterations=3)
jonction   = cv2.bitwise_and(bord_rouge, bord_blanc)

contours_img      = cv2.Canny(gris, 30, 100)
contour_frontiere = cv2.bitwise_and(contours_img, jonction)
cv2.imshow("Etape 7 : contour frontiere rouge/blanc", contour_frontiere)  # a effacer

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
    exit()

# On garde le segment dont la droite portante passe le plus pres du centre
meilleur_segment = None
distance_min     = float("inf")

for seg in segments:
    x1, y1, x2, y2 = seg[0]
    dx, dy = x2 - x1, y2 - y1
    dist = abs(dy * cx - dx * cy + x2 * y1 - y2 * x1) / (np.sqrt(dx**2 + dy**2) + 1e-6)
    if dist < distance_min:
        distance_min     = dist
        meilleur_segment = seg[0]

x1, y1, x2, y2 = meilleur_segment
print(f"Segment frontiere : ({x1},{y1}) -> ({x2},{y2}), distance au centre : {distance_min:.1f}px")

# ─── CALCUL DE L'ANGLE DE LA FRONTIERE ───────────────────────────────────────
d1 = (x1 - cx)**2 + (y1 - cy)**2
d2 = (x2 - cx)**2 + (y2 - cy)**2
px_loin, py_loin = (x2, y2) if d2 > d1 else (x1, y1)

angle_frontiere_rad = np.arctan2(-(py_loin - cy), px_loin - cx)
angle_frontiere_deg = np.rad2deg(angle_frontiere_rad) % 360
print(f"Angle frontiere : {angle_frontiere_deg:.1f} deg")

# ─── CALCUL DE LA VALEUR ──────────────────────────────────────────────────────
plage_deg        = ANGLE_MIN_DEG - ANGLE_MAX_DEG
mesure_deg       = ANGLE_MIN_DEG - angle_frontiere_deg
ratio            = max(0.0, min(1.0, mesure_deg / plage_deg))
valeur_numerisee = VALEUR_MIN + ratio * (VALEUR_MAX - VALEUR_MIN)
print(f"Ratio : {ratio:.2%}  =>  Valeur : {valeur_numerisee:.1f}")

# ─── DESSIN DES ANNOTATIONS ───────────────────────────────────────────────────

overlay = img_resultat.copy()
overlay[masque_rouge > 0] = [0, 0, 200]
img_resultat = cv2.addWeighted(overlay, 0.3, img_resultat, 0.7, 0)

cv2.circle(img_resultat, (cx, cy), rayon, (0, 200, 0), 1)

for i in range(len(angles_arc) - 1):
    p1 = (int(cx + rayon * np.cos(angles_arc[i])),   int(cy - rayon * np.sin(angles_arc[i])))
    p2 = (int(cx + rayon * np.cos(angles_arc[i+1])), int(cy - rayon * np.sin(angles_arc[i+1])))
    cv2.line(img_resultat, p1, p2, (0, 200, 255), 2)

# Graduations de 10 en 10 : de -20 a 100 => 13 graduations
valeurs_grad = range(int(VALEUR_MIN), int(VALEUR_MAX) + 1, 10)
for val in valeurs_grad:
    t     = (val - VALEUR_MIN) / (VALEUR_MAX - VALEUR_MIN)
    angle = angle_min_rad + t * (angle_max_rad - angle_min_rad)

    px_ext = int(cx + rayon        * np.cos(angle))
    py_ext = int(cy - rayon        * np.sin(angle))
    px_int = int(cx + (rayon - 12) * np.cos(angle))
    py_int = int(cy - (rayon - 12) * np.sin(angle))
    px_lbl = int(cx + (rayon + 16) * np.cos(angle))
    py_lbl = int(cy - (rayon + 16) * np.sin(angle))

    cv2.line(img_resultat, (px_int, py_int), (px_ext, py_ext), (255, 220, 0), 1)
    cv2.putText(img_resultat, f"{val:.0f}", (px_lbl - 10, py_lbl + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 0), 1)

for label, angle_r in [("MIN", angle_min_rad), ("MAX", angle_max_rad)]:
    px = int(cx + (rayon + 22) * np.cos(angle_r))
    py = int(cy - (rayon + 22) * np.sin(angle_r))
    cv2.putText(img_resultat, label, (px - 12, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

px_bord = int(cx + rayon * np.cos(angle_frontiere_rad))
py_bord = int(cy - rayon * np.sin(angle_frontiere_rad))
cv2.line(img_resultat, (cx, cy), (px_bord, py_bord), (0, 220, 220), 2)

cv2.circle(img_resultat, (cx, cy), 3, (255, 255, 255), -1)

cv2.putText(img_resultat, f"Niveau : {valeur_numerisee:.1f}",
            (cx - rayon + 5, cy + rayon + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# ─── SAUVEGARDE ET AFFICHAGE ──────────────────────────────────────────────────
cv2.imwrite("resultat_analyse.png", img_resultat)
print("Image sauvegardee : resultat_analyse.png")

cv2.imshow("Image originale",      img)
cv2.imshow("Analyse niveau huile", img_resultat)
cv2.waitKey(0)
cv2.destroyAllWindows()
