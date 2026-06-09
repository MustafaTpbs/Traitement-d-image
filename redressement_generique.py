# ─── LIBRAIRIES ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import sys

# ─── PARAMETRES ───────────────────────────────────────────────────────────────
IMAGE_REF  = "indicateur1_pastilles.jpg"    # image droite avec pastilles
IMAGE_TEST = "indicateur1_angle1_pastilles.jpg"    # image en perspective avec pastilles

# ─── DETECTION ROBUSTE DES PASTILLES VERTES ───────────────────────────────────
def detecter_pastilles(img, nom="image"):
    """
    Détecte automatiquement 4 pastilles vertes dans une image.
    S'adapte à la résolution et à la qualité (sombre, délavé, basse saturation).
    Retourne un array numpy (4, 2) ou lève une exception avec message debug.
    """
    h, w = img.shape[:2]
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    nb_pixels = w * h

    # Surface attendue : une pastille = entre 0.003% et 5% de l'image
    surf_min = nb_pixels * 0.00003
    surf_max = nb_pixels * 0.05

    # Marge de bord : ignorer les blobs collés au bord de l'image (artefacts JPEG)
    marge = max(30, int(min(w, h) * 0.01))

    # Taille de kernel morphologique selon résolution
    if   nb_pixels > 6_000_000:  tailles_kernel = [9, 11, 7]
    elif nb_pixels > 1_000_000:  tailles_kernel = [7,  5, 9]
    else:                         tailles_kernel = [5,  3, 7]

    # Plages HSV testées en cascade, de la plus stricte à la plus permissive
    plages_hsv = [
        ([40, 100, 100], [80, 255, 255]),   # vert vif
        ([35,  60,  80], [90, 255, 255]),   # vert standard
        ([35,  40,  80], [95, 255, 255]),   # vert sombre / délavé
        ([30,  25,  60], [100,255, 255]),   # vert très atténué
    ]

    meilleur_masque = None
    meilleurs_pts   = []

    for lo, hi in plages_hsv:
        for k in tailles_kernel:
            masque = cv2.inRange(hsv, np.array(lo), np.array(hi))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN,  kernel, iterations=2)
            masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, kernel, iterations=3)

            contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pts_candidats = []
            for c in contours:
                s = cv2.contourArea(c)
                if not (surf_min < s < surf_max):
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                x = M["m10"] / M["m00"]
                y = M["m01"] / M["m00"]
                if x < marge or x > w - marge or y < marge or y > h - marge:
                    continue
                pts_candidats.append([x, y])

            # On s'arrête dès qu'on trouve exactement 4 points
            if len(pts_candidats) == 4:
                print(f"[{nom}] Pastilles trouvées — HSV [{lo}~{hi}], kernel={k}x{k}")
                return np.array(pts_candidats, dtype=np.float32), masque

            # Mémoriser le meilleur résultat partiel pour le message d'erreur
            if abs(len(pts_candidats) - 4) < abs(len(meilleurs_pts) - 4):
                meilleurs_pts   = pts_candidats
                meilleur_masque = masque.copy()

    # Echec : afficher le debug et quitter
    print(f"[ERREUR] {len(meilleurs_pts)} pastille(s) trouvée(s) dans '{nom}' (4 attendues).")
    print("  → Vérifier l'éclairage, la couleur des pastilles ou les paramètres HSV.")
    if meilleur_masque is not None:
        debug = img.copy()
        for pt in meilleurs_pts:
            cv2.circle(debug, (int(pt[0]), int(pt[1])), 20, (0, 0, 255), -1)
        scale = 900 / max(w, h)
        cv2.imshow(f"Debug pastilles - {nom}", cv2.resize(debug, (int(w*scale), int(h*scale))))
        cv2.imshow(f"Debug masque   - {nom}", cv2.resize(meilleur_masque, (int(w*scale), int(h*scale))))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    sys.exit(1)


def trier_losange(pts):
    """
    Trie 4 points en losange : haut (y min), droite (x max), bas (y max), gauche (x min).
    Robuste à toute orientation de prise de vue.
    """
    return np.array([
        pts[np.argmin(pts[:, 1])],   # haut
        pts[np.argmax(pts[:, 0])],   # droite
        pts[np.argmax(pts[:, 1])],   # bas
        pts[np.argmin(pts[:, 0])],   # gauche
    ], dtype=np.float32)


def afficher_cote_a_cote(titre, img_gauche, img_droite, largeur_max=1400):
    """Affiche deux images côte à côte dans une seule fenêtre."""
    h1, w1 = img_gauche.shape[:2]
    h2, w2 = img_droite.shape[:2]

    # Même hauteur pour les deux
    h_cible = min(h1, h2, 900)
    img_g = cv2.resize(img_gauche, (int(w1 * h_cible / h1), h_cible))
    img_d = cv2.resize(img_droite, (int(w2 * h_cible / h2), h_cible))

    # Limiter la largeur totale
    largeur_totale = img_g.shape[1] + img_d.shape[1]
    if largeur_totale > largeur_max:
        scale = largeur_max / largeur_totale
        img_g = cv2.resize(img_g, (int(img_g.shape[1]*scale), int(img_g.shape[0]*scale)))
        img_d = cv2.resize(img_d, (int(img_d.shape[1]*scale), int(img_d.shape[0]*scale)))

    # Séparateur vertical blanc
    separateur = np.ones((img_g.shape[0], 4, 3), dtype=np.uint8) * 200

    # Texte au dessus de chaque image
    def ajouter_label(img, texte):
        bande = np.zeros((40, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(bande, texte, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return np.vstack([bande, img])

    img_g = ajouter_label(img_g, "ORIGINAL")
    img_d = ajouter_label(img_d, "REDRESSE")

    # S'assurer que les deux ont la même hauteur après label
    h_max = max(img_g.shape[0], img_d.shape[0])
    def pad_hauteur(img, h):
        diff = h - img.shape[0]
        return np.vstack([img, np.zeros((diff, img.shape[1], 3), dtype=np.uint8)]) if diff > 0 else img

    img_g = pad_hauteur(img_g, h_max)
    separateur = np.ones((h_max, 4, 3), dtype=np.uint8) * 200
    img_d = pad_hauteur(img_d, h_max)

    cv2.imshow(titre, np.hstack([img_g, separateur, img_d]))


# ─── PROGRAMME PRINCIPAL ──────────────────────────────────────────────────────
print("Chargement des images...")
img_ref  = cv2.imread(IMAGE_REF)
img_test = cv2.imread(IMAGE_TEST)

if img_ref is None:
    print(f"Erreur : impossible de charger '{IMAGE_REF}'"); sys.exit(1)
if img_test is None:
    print(f"Erreur : impossible de charger '{IMAGE_TEST}'"); sys.exit(1)

print(f"Référence : {img_ref.shape[1]}x{img_ref.shape[0]}px")
print(f"Test      : {img_test.shape[1]}x{img_test.shape[0]}px")

# Détection des pastilles
print("\nDétection des pastilles...")
pts_ref,  _ = detecter_pastilles(img_ref,  "reference")
pts_test, _ = detecter_pastilles(img_test, "test")

# Tri en losange (haut / droite / bas / gauche)
src = trier_losange(pts_test)
dst = trier_losange(pts_ref)

print(f"\nPoints source      : {src.astype(int).tolist()}")
print(f"Points destination : {dst.astype(int).tolist()}")

# Calcul de l'homographie et redressement
h, w     = img_ref.shape[:2]
M        = cv2.getPerspectiveTransform(src, dst)
redressee = cv2.warpPerspective(img_test, M, (w, h))

# Affichage côte à côte
afficher_cote_a_cote("Redressement par homographie", img_test, redressee)
print("\nAppuyer sur une touche pour fermer...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sauvegarde
cv2.imwrite("image_redressee.jpg", redressee)
print("Image sauvegardée : image_redressee.jpg")