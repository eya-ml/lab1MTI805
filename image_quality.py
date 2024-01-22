import cv2
import numpy as np
#Ajuste la luminosité de l'image. Luminosité: -255 à 255
def ajuster_luminosite(image, luminosite=0):


    if luminosite != 0:
        image = cv2.convertScaleAbs(image, alpha=1, beta=luminosite)
    return image
#Ajuste le contraste de l'image. Contraste: -127 à 127
def ajuster_contraste(image, contraste=0):

    if contraste != 0:
        f = 131 * (contraste + 127) / (127 * (131 - contraste))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    return image
#Ajoute du bruit gaussien à l'image. Niveau de bruit: 0 à 100
def ajouter_bruit(image, niveau_bruit=0):

    if niveau_bruit != 0:
        ligne, colonne, ch = image.shape
        moyenne = 0
        sigma = abs(niveau_bruit)
        gauss = np.random.normal(moyenne, sigma, (ligne, colonne, ch))
        gauss = gauss.reshape(ligne, colonne, ch)
        image_bruitee = image + gauss
        image_bruitee = np.clip(image_bruitee, 0, 255)
        image_bruitee = image_bruitee.astype(np.uint8)
        return image_bruitee
    return image
# Fonctions pour recadrer et redimensionner les images
def recadrer_et_redimensionner(image, x, y, w, h, taille_cible=224):
    image_recadree = image[y:y+h, x:x+w]
    image_redimensionnee = cv2.resize(image_recadree, (taille_cible, taille_cible))
    return image_redimensionnee