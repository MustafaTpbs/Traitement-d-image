# ─── Bibliothèques ────────────────────────────────────────────────────────────
import cv2
import numpy as np

# ─── PARAMÈTRES ───────────────────────────────────────────────────────────────

# Nom de l'image à analyser (doit être dans le même dossier que ce script)
IMAGE = "indicateur1.jpeg"

# Valeurs min et max de l'échelle de l'indicateur
VALEUR_MIN = -35
VALEUR_MAX = 100

# ─── CHARGEMENT DE L'IMAGE ────────────────────────────────────────────────────

# Charge l'image depuis le dossier courant
img = cv2.imread(IMAGE)
if img is None:
    print(f"Erreur : impossible de charger '{IMAGE}'. Vérifiez que le fichier est dans le même dossier.")
    exit()

# Fait une copie propre pour dessiner dessus à la fin
img_resultat = img.copy()
cv2.imshow("Etape 1 : image de base", img_resultat) #à effacer

# ─── ISOLATION DE LA ZONE CIRCULAIRE (LE CADRAN) ──────────────────────────────

# Convertie en niveaux de gris pour détecter le cercle
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Etape 2 : image en gris", gris) #à effacer

# Floute légèrement pour que la détection de cercle soit plus robuste, enlève le bruit autours des contours
gris_floute = cv2.GaussianBlur(gris, (9, 9), 0) #La fonction fait une moyenne de la couleur des 9 pixels voisin pour chaque pixel
cv2.imshow("Etape 3 : image en gris flou", gris_floute) #à effacer

# Cherche le cercle du cadran avec la transformée de Hough
cercles = cv2.HoughCircles(
    gris_floute,		# image utilisé
    cv2.HOUGH_GRADIENT, # méthode utilisé : Chaque pixel de contour (changement brusque d'intensité lumineuse) vote pour tous ces "centres possibles"
						# dans la direction du gradient, l'endroit qui a le plus de vote est élu centre du cercle et le rayon constant est enrigistré
    dp=1,             # résolution de l'accumulateur. 1 => même résolution que image. 2 => deux fois plus flou. valeur recommandé => 1.5
    minDist=100,      # distance minimale entre deux centres cercles détectés
    param1=50,        # seuil pour la senisibilité  de la détection de contours => 50 à 150
    param2=20,        # seuil d'accumulation, minimum à attaindre pour être considéré comme un cercle (plus bas = plus permissif) ==> 20 à 100 
    minRadius=0,
    maxRadius=300
)

if cercles is None:
    print("Aucun cercle trouvé. Ajuster les paramètres de HoughCircles")
    exit()

# Prend le premier cercle trouvé, cadran intérieur
cx, cy, rayon = np.round(cercles[0][0]).astype(int)
print(f"Cadran détecté → centre : ({cx}, {cy}), rayon : {rayon}px")

# Je crée un masque circulaire pour ignorer ce qui est en dehors du cadran
masque_cercle = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.circle(masque_cercle, (cx, cy), rayon, 255, -1)  # -1 = remplissage

# ─── DÉTECTION DE LA COULEUR ROSE ─────────────────────────────────────────────

# Je convertis en HSV, plus pratique pour isoler une couleur
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Plage de couleur pour le rose/rouge 
# Le rose de l'indicateur tombe dans ces valeurs HSV
rose_bas = np.array([140, 30, 150])
rose_haut = np.array([180, 255, 255])
masque_rose = cv2.inRange(hsv, rose_bas, rose_haut)

# Je complète avec le rouge côté bas du spectre HSV (0-10)
rouge_bas = np.array([0, 30, 150])
rouge_haut = np.array([10, 255, 255])
masque_rouge = cv2.inRange(hsv, rouge_bas, rouge_haut)

# Je combine les deux masques roses/rouges
masque_couleur = cv2.bitwise_or(masque_rose, masque_rouge)

# J'applique le masque circulaire : je garde uniquement l'intérieur du cadran
masque_final = cv2.bitwise_and(masque_couleur, masque_cercle)

# ─── CALCUL DU NIVEAU ─────────────────────────────────────────────────────────

# Je compte les pixels roses dans la moitié basse vs la moitié haute du cadran
# → le fond tourne, le rose en bas = plein, le blanc en bas = vide
hauteur_cadran = rayon * 2

# Je découpe le masque dans la zone du cadran
y1 = max(cy - rayon, 0)
y2 = min(cy + rayon, img.shape[0])
x1 = max(cx - rayon, 0)
x2 = min(cx + rayon, img.shape[1])

zone_masque = masque_final[y1:y2, x1:x2]

# Je calcule le ratio de pixels roses sur l'ensemble de la zone du cadran
pixels_roses = np.count_nonzero(zone_masque)
pixels_total = np.count_nonzero(masque_cercle[y1:y2, x1:x2])
ratio = pixels_roses / pixels_total if pixels_total > 0 else 0

# Je mappe ce ratio sur l'échelle min/max de l'indicateur
valeur_numerisee = VALEUR_MIN + ratio * (VALEUR_MAX - VALEUR_MIN)
print(f"Pixels roses : {pixels_roses} / {pixels_total} → ratio : {ratio:.2%}")
print(f"Valeur numérisée : {valeur_numerisee:.1f}")

# ─── DÉTECTION DE LA FRONTIÈRE ROSE / BLANC ───────────────────────────────────

# Je floute le masque pour avoir une frontière douce
masque_floute = cv2.GaussianBlur(masque_final, (21, 21), 0)

# Je cherche les contours de la zone rose (frontière avec le blanc)
contours, _ = cv2.findContours(masque_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ─── DESSIN DES ANNOTATIONS SUR L'IMAGE RÉSULTAT ──────────────────────────────

# Je dessine le cercle du cadran détecté
cv2.circle(img_resultat, (cx, cy), rayon, (0, 200, 0), 2)

# Je dessine les contours de la frontière rose/blanc en bleu
cv2.drawContours(img_resultat, contours, -1, (255, 100, 0), 2)

# Je marque le centre du cadran
cv2.circle(img_resultat, (cx, cy), 4, (0, 255, 255), -1)

# J'ajoute le texte MIN en bas du cadran
pos_min = (cx - rayon + 10, cy + rayon - 10)
cv2.putText(img_resultat, f"MIN {VALEUR_MIN}", pos_min,
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 2)

# J'ajoute le texte MAX en haut du cadran
pos_max = (cx - rayon + 10, cy - rayon + 20)
cv2.putText(img_resultat, f"MAX {VALEUR_MAX}", pos_max,
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 2)

# J'affiche la valeur numérisée bien visible au centre
label = f"Niveau : {valeur_numerisee:.1f}"
cv2.putText(img_resultat, label, (cx - 70, cy + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# ─── SUPERPOSITION DU MASQUE ROSE EN TRANSPARENCE ────────────────────────────

# Je crée une image colorée à partir du masque (rose semi-transparent)
overlay = img_resultat.copy()
overlay[masque_final > 0] = [180, 60, 180]  # je colorie les pixels roses détectés
img_resultat = cv2.addWeighted(overlay, 0.25, img_resultat, 0.75, 0)  # je mélange

# Je re-dessine les textes par-dessus la transparence (sinon ils sont écrasés)
cv2.drawContours(img_resultat, contours, -1, (255, 100, 0), 2)
cv2.putText(img_resultat, f"MIN {VALEUR_MIN}", pos_min,
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 2)
cv2.putText(img_resultat, f"MAX {VALEUR_MAX}", pos_max,
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 2)
cv2.putText(img_resultat, label, (cx - 70, cy + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# ─── SAUVEGARDE ET AFFICHAGE ──────────────────────────────────────────────────

# Je sauvegarde l'image annotée dans le même dossier
cv2.imwrite("resultat_analyse.png", img_resultat)
print("Image annotée sauvegardée : resultat_analyse.png")

# J'affiche les deux images côte à côte pour comparer
cv2.imshow("Image originale", img)
cv2.imshow("Analyse niveau huile", img_resultat)
cv2.waitKey(0)   # j'attends que l'utilisateur appuie sur une touche
cv2.destroyAllWindows()
