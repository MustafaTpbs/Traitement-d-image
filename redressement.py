'''
Ce fichier est une bibliothèque de fonction écrite dans la cadre du projet de traitement d'image d'un indicateur de niveau d'huile. 
Ces fonctions sont utilisés dans le programme principale 'main.py'
'''
# ─── LIBRAIRIES ─────────────────────────────────────────────────────────────────────
import cv2  #Traitement d'image en temps réel.
import numpy as np  #Fonctions mathématiques sur des matrices/tableaux de données.
#─────────────────────────────────────────────────────────────────────────────────────


# ─── PARAMETRES ─────────────────────────────────────────────────────────────────────
IMAGE_REF  = "indicateur1_pastilles.jpg"    # image droite avec pastilles, c'est la référence pour le redressement
IMAGE_TEST = "indicateur1_angle1_pastilles.jpg"    # image en perspective avec pastilles
#─────────────────────────────────────────────────────────────────────────────────────


# ─── DETECTION  DES PASTILLES VERTES ────────────────────────────────────────────────
def detecter_pastilles(img, nom="image"):
    """
    Détecte automatiquement 4 pastilles vertes dans une image.
    S'adapte à la résolution et à la qualité (sombre, délavé, basse saturation).
    Retourne un array numpy (4, 2) ou lève une exception avec message debug.
    """
    h, w = img.shape[:2]    #Récupère les dimensions de l'image en paramètre
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)     #Change la mapping de couleur de l'image pour la détection du vert
    nb_pixels = w * h       #Récupère le nombre de pixels de l'image 

    # Surface attendue : une pastille = entre 0.003% et 5% de l'image
    surf_min = nb_pixels * 0.00003
    surf_max = nb_pixels * 0.05

    # Marge de bord : ignorer les blobs collés au bord de l'image (artefacts JPEG)
    marge = max(30, int(min(w, h) * 0.01))      #30 pixels ou 1% du bord

    # Taille de kernel morphologique selon résolution |  #le kernel est une forme géométrique simple pour dilater/boucher plus tard les zones utiles
    if   nb_pixels > 6_000_000:  tailles_kernel = [9, 11, 7]
    elif nb_pixels > 1_000_000:  tailles_kernel = [7,  5, 9]
    else:                        tailles_kernel = [5,  3, 7]

    # Plages HSV testées en cascade, de la plus stricte à la plus permissive
    plages_hsv = [
        ([40, 100, 100], [80, 255, 255]),   # vert vif
        ([35,  60,  80], [90, 255, 255]),   # vert standard
        ([35,  40,  80], [95, 255, 255]),   # vert sombre / délavé
        ([30,  25,  60], [100,255, 255]),   # vert très atténué
    ]

    meilleur_masque = None      #Simple déclaration
    meilleurs_pts   = []        #Simple déclaration

    for lo, hi in plages_hsv:
        for k in tailles_kernel:
            masque = cv2.inRange(hsv, np.array(lo), np.array(hi))       #isole en blanc (255) tous les pixels qui entrent dans la plage de vert actuelle, et met le reste en noir (0)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))       #génération d'une matrice de forme elliptique de taille k×k qui servira de pinceau pour nettoyer le masque binaire
            masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN,  kernel, iterations=2)
            masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, kernel, iterations=3)

            contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)      #extraction des contours extérieurs de toutes les formes blanches restantes dans le masque nettoyé
            pts_candidats = []
            for c in contours:
                s = cv2.contourArea(c)      #calcul de la surface réelle (en pixels) occupée par la forme géométrique délimitée par le contour
                if not (surf_min < s < surf_max):
                    continue
                M = cv2.moments(c)      #vérifie si la surface de la forme est nulle
                if M["m00"] == 0:
                    continue
                x = M["m10"] / M["m00"]     #calcul des coordonnées cartésiennes (x,y) du centre exact de la forme blanche
                y = M["m01"] / M["m00"]
                if x < marge or x > w - marge or y < marge or y > h - marge:    #rejet du point calculé si son centre se trouve trop près des bords de l'image
                    continue
                pts_candidats.append([x, y])        #ajout descoordonnées du centre de la pastille dans la liste des candidats de la tentative en cours

            # s'arrête dès qu'on trouve exactement 4 points
            if len(pts_candidats) == 4:
                print(f"[{nom}] Pastilles trouvées — HSV [{lo}~{hi}], kernel={k}x{k}")
                return np.array(pts_candidats, dtype=np.float32), masque

            # Mémoriser le meilleur résultat partiel pour le message d'erreur
            if abs(len(pts_candidats) - 4) < abs(len(meilleurs_pts) - 4):   #Fait la comparaison de l'essai actuel avec les essais précédents 
                meilleurs_pts   = pts_candidats
                meilleur_masque = masque.copy()

    # Echec : afficher le debug et quitter
    print(f"[ERREUR] {len(meilleurs_pts)} pastille(s) trouvée(s) dans '{nom}' (4 attendues).")
    print("  → Vérifier l'éclairage, la couleur des pastilles ou les paramètres HSV.")
    if meilleur_masque is not None:   #RAPPORT
        debug = img.copy()   #RAPPORT
        for pt in meilleurs_pts:   #RAPPORT
            cv2.circle(debug, (int(pt[0]), int(pt[1])), 20, (0, 0, 255), -1)   #RAPPORT
        scale = 900 / max(w, h)   #RAPPORT
        cv2.imshow(f"Debug pastilles - {nom}", cv2.resize(debug, (int(w*scale), int(h*scale))))     #RAPPORT
        cv2.imshow(f"Debug masque   - {nom}", cv2.resize(meilleur_masque, (int(w*scale), int(h*scale))))       #RAPPORT
        cv2.waitKey(0)      #RAPPORT
        cv2.destroyAllWindows()
    sys.exit(1)
#─────────────────────────────────────────────────────────────────────────────────────


# ─── Fonction de reconnaissance de position de pastille ─────────────────────────────
def trier_losange(pts):
    """
    Trie 4 points en losange : haut (y min), droite (x max), bas (y max), gauche (x min).
    """
    return np.array([
        pts[np.argmin(pts[:, 1])],   # haut
        pts[np.argmax(pts[:, 0])],   # droite
        pts[np.argmax(pts[:, 1])],   # bas
        pts[np.argmin(pts[:, 0])],   # gauche
    ], dtype=np.float32)
#─────────────────────────────────────────────────────────────────────────────────────


# ─── Réduction de la taille pour traitement efficace de l'img ────────────────────────
def redimensionner_si_trop_grande(img, largeur_max=800):
    """Réduit l'image si sa largeur dépasse largeur_max, tout en gardant les proportions."""
    h, w = img.shape[:2]
    if w > largeur_max:
        scale = largeur_max / w
        nouvelle_hauteur = int(h * scale)
        return cv2.resize(img, (largeur_max, nouvelle_hauteur), interpolation=cv2.INTER_AREA)
    return img
#─────────────────────────────────────────────────────────────────────────────────────
