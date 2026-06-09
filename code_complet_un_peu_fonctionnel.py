import cv2
import numpy as np

# ─── PARAMETRES ───────────────────────────────────────────────────────────────
IMAGE_REF  = "indicateur1.jpeg"                  # image droite (reference)
IMAGE_TEST = "indicateur1_angle1_pastilles.png"  # image a redresser

VALEUR_MIN    = -20
VALEUR_MAX    = 100
ANGLE_MIN_DEG = 240.0
ANGLE_MAX_DEG = 120.0

# ─── FONCTIONS ────────────────────────────────────────────────────────────────
def detecter_pastilles(img, min_surface=50, max_surface=8000):
    """Detecte les pastilles vertes et retourne leurs centroïdes."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masque = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN,  kernel, iterations=2)
    masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in contours:
        if min_surface < cv2.contourArea(c) < max_surface:
            M = cv2.moments(c)
            if M["m00"] > 0:
                pts.append([M["m10"]/M["m00"], M["m01"]/M["m00"]])
    return np.array(pts, dtype=np.float32), masque


def trier_losange(pts):
    """Trie 4 points : haut (y min), droite (x max), bas (y max), gauche (x min)."""
    haut   = pts[np.argmin(pts[:, 1])]
    bas    = pts[np.argmax(pts[:, 1])]
    gauche = pts[np.argmin(pts[:, 0])]
    droite = pts[np.argmax(pts[:, 0])]
    return np.array([haut, droite, bas, gauche], dtype=np.float32)


def analyser_cadran(img, cx, cy, rayon, angle_min_rad, angle_max_rad):
    """Detecte la frontiere rouge/blanc, retourne l'angle en radians."""
    gris   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Masque du secteur utile
    angles_arc = np.linspace(angle_min_rad, angle_max_rad, 300)
    masque_secteur = np.zeros(img.shape[:2], dtype=np.uint8)
    points_arc = np.array([
        [cx + rayon * np.cos(a), cy - rayon * np.sin(a)]
        for a in angles_arc
    ], dtype=np.int32)
    cv2.fillPoly(masque_secteur, [np.vstack([[cx, cy], points_arc])], 255)

    # Masques rouge et blanc
    masque_rouge = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   80,  80]), np.array([10,  255, 255])),
        cv2.inRange(hsv, np.array([160, 80,  80]), np.array([180, 255, 255]))
    )
    masque_rouge = cv2.bitwise_and(masque_rouge, masque_secteur)
    masque_rouge = cv2.morphologyEx(masque_rouge, cv2.MORPH_CLOSE, kernel)

    masque_blanc = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))
    masque_blanc = cv2.bitwise_and(masque_blanc, masque_secteur)
    masque_blanc = cv2.morphologyEx(masque_blanc, cv2.MORPH_CLOSE, kernel)

    # Jonction rouge/blanc
    jonction = cv2.bitwise_and(
        cv2.dilate(masque_rouge, kernel, iterations=3),
        cv2.dilate(masque_blanc, kernel, iterations=3)
    )
    contour_frontiere = cv2.bitwise_and(cv2.Canny(gris, 30, 100), jonction)

    # Detection du segment
    segments = cv2.HoughLinesP(
        contour_frontiere, rho=1, theta=np.pi/180,
        threshold=15, minLineLength=rayon * 0.2, maxLineGap=15
    )
    if segments is None:
        return None, masque_rouge, angles_arc

    meilleur, dist_min = None, float("inf")
    for seg in segments:
        x1, y1, x2, y2 = seg[0]
        dx, dy = x2 - x1, y2 - y1
        dist = abs(dy*cx - dx*cy + x2*y1 - y2*x1) / (np.sqrt(dx**2 + dy**2) + 1e-6)
        if dist < dist_min:
            dist_min, meilleur = dist, seg[0]

    x1, y1, x2, y2 = meilleur
    px_loin, py_loin = (x2, y2) if (x2-cx)**2+(y2-cy)**2 > (x1-cx)**2+(y1-cy)**2 else (x1, y1)
    return np.arctan2(-(py_loin - cy), px_loin - cx), masque_rouge, angles_arc


# ─── CHARGEMENT ───────────────────────────────────────────────────────────────
img_ref  = cv2.imread(IMAGE_REF)
img_test = cv2.imread(IMAGE_TEST)

if img_ref is None:
    print(f"Erreur : impossible de charger '{IMAGE_REF}'.")
    exit()
if img_test is None:
    print(f"Erreur : impossible de charger '{IMAGE_TEST}'.")
    exit()

# ─── DETECTION AUTOMATIQUE DES PASTILLES ─────────────────────────────────────
pts_ref_brut,  _ = detecter_pastilles(img_ref)
pts_test_brut, _ = detecter_pastilles(img_test)

for nom, pts in [("reference", pts_ref_brut), ("test", pts_test_brut)]:
    if len(pts) != 4:
        print(f"ERREUR : {len(pts)} pastille(s) dans l'image {nom}, 4 attendues.")
        exit()

pts_dst = trier_losange(pts_ref_brut)   # positions cibles = image droite
pts_src = trier_losange(pts_test_brut)  # positions source = image en perspective

print("Pastilles reference :", pts_dst)
print("Pastilles test      :", pts_src)

# ─── REDRESSEMENT PAR HOMOGRAPHIE ────────────────────────────────────────────
h, w = img_ref.shape[:2]
M_homo        = cv2.getPerspectiveTransform(pts_src, pts_dst)
img_redressee = cv2.warpPerspective(img_test, M_homo, (w, h))

cv2.imshow("Image redressee", img_redressee)

# On travaille desormais sur l'image redressée
img          = img_redressee
img_resultat = img.copy()

# ─── DETECTION DU CERCLE ─────────────────────────────────────────────────────
gris        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gris_floute = cv2.GaussianBlur(gris, (9, 9), 0)

cercles = cv2.HoughCircles(
    gris_floute, cv2.HOUGH_GRADIENT,
    dp=1, minDist=100, param1=50, param2=20,
    minRadius=0, maxRadius=300
)
if cercles is None:
    print("Aucun cercle trouve apres redressement.")
    cv2.waitKey(0); cv2.destroyAllWindows(); exit()

cx, cy, rayon = np.round(cercles[0][0]).astype(int)
print(f"Cadran => centre : ({cx}, {cy}), rayon : {rayon}px")

# ─── ANALYSE DU CADRAN ───────────────────────────────────────────────────────
angle_min_rad = np.deg2rad(ANGLE_MIN_DEG)
angle_max_rad = np.deg2rad(ANGLE_MAX_DEG)

angle_rad, masque_rouge, angles_arc = analyser_cadran(
    img, cx, cy, rayon, angle_min_rad, angle_max_rad
)
if angle_rad is None:
    print("Frontiere rouge/blanc introuvable.")
    cv2.waitKey(0); cv2.destroyAllWindows(); exit()

angle_deg        = np.rad2deg(angle_rad) % 360
plage_deg        = ANGLE_MIN_DEG - ANGLE_MAX_DEG
mesure_deg       = ANGLE_MIN_DEG - angle_deg
ratio            = max(0.0, min(1.0, mesure_deg / plage_deg))
valeur_numerisee = VALEUR_MIN + ratio * (VALEUR_MAX - VALEUR_MIN)
print(f"Angle : {angle_deg:.1f}°  |  Ratio : {ratio:.2%}  |  Valeur : {valeur_numerisee:.1f}")

# ─── ANNOTATIONS ─────────────────────────────────────────────────────────────
overlay = img_resultat.copy()
overlay[masque_rouge > 0] = [0, 0, 200]
img_resultat = cv2.addWeighted(overlay, 0.3, img_resultat, 0.7, 0)

cv2.circle(img_resultat, (cx, cy), rayon, (0, 200, 0), 1)

for i in range(len(angles_arc) - 1):
    p1 = (int(cx + rayon * np.cos(angles_arc[i])),   int(cy - rayon * np.sin(angles_arc[i])))
    p2 = (int(cx + rayon * np.cos(angles_arc[i+1])), int(cy - rayon * np.sin(angles_arc[i+1])))
    cv2.line(img_resultat, p1, p2, (0, 200, 255), 2)

for val in range(int(VALEUR_MIN), int(VALEUR_MAX) + 1, 10):
    t      = (val - VALEUR_MIN) / (VALEUR_MAX - VALEUR_MIN)
    angle  = angle_min_rad + t * (angle_max_rad - angle_min_rad)
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

px_bord = int(cx + rayon * np.cos(angle_rad))
py_bord = int(cy - rayon * np.sin(angle_rad))
cv2.line(img_resultat, (cx, cy), (px_bord, py_bord), (0, 220, 220), 2)
cv2.circle(img_resultat, (cx, cy), 3, (255, 255, 255), -1)
cv2.putText(img_resultat, f"Niveau : {valeur_numerisee:.1f}",
            (cx - rayon + 5, cy + rayon + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# ─── SAUVEGARDE ───────────────────────────────────────────────────────────────
cv2.imwrite("resultat_analyse.png", img_resultat)
print("Sauvegarde : resultat_analyse.png")
cv2.imshow("Analyse niveau huile", img_resultat)
cv2.waitKey(0)
cv2.destroyAllWindows()