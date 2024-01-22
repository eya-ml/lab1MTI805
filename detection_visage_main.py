import cv2
import mediapipe as mp
from datetime import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from deepface import DeepFace
models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib"]

# Initialiser le modèle de suivi des mains de MediaPipe.
mp_mains = mp.solutions.hands
mp_dessin = mp.solutions.drawing_utils
mains = mp_mains.Hands()

# Charger le Haar Cascade pour la détection des visages.
cascade_visage = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Charger le modèle ResNet pré-entraîné pour la génération des embeddings
resnet_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Fonctions pour recadrer et redimensionner les images
def recadrer_et_redimensionner(image, x, y, w, h, taille_cible=224):
    image_recadree = image[y:y+h, x:x+w]
    image_redimensionnee = cv2.resize(image_recadree, (taille_cible, taille_cible))
    return image_redimensionnee

# Ajouter les fonctions de réglage de la luminosité, du contraste et du bruit.
def ajuster_luminosite(image, luminosite=0):
    """ Ajuste la luminosité de l'image. Luminosité: -255 à 255 """
    if luminosite != 0:
        image = cv2.convertScaleAbs(image, alpha=1, beta=luminosite)
    return image

def ajuster_contraste(image, contraste=0):
    """ Ajuste le contraste de l'image. Contraste: -127 à 127 """
    if contraste != 0:
        f = 131 * (contraste + 127) / (127 * (131 - contraste))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    return image

def ajouter_bruit(image, niveau_bruit=0):
    """ Ajoute du bruit gaussien à l'image. Niveau de bruit: 0 à 100 """
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

# Initialiser la webcam.
webcam = cv2.VideoCapture(0)

# Définir la luminosité, le contraste et le niveau de bruit initiaux
luminosite = 0  
contraste = 0  
niveau_bruit = 0

nom_fenetre = 'Detection'
cv2.namedWindow(nom_fenetre)

while True:
    ret, cadre = webcam.read()
    if not ret:
        break

    # Ajuster la luminosité, le contraste et le bruit séparément
    cadre = ajuster_luminosite(cadre, luminosite)
    cadre = ajuster_contraste(cadre, contraste)
    cadre = ajouter_bruit(cadre, niveau_bruit)

    # Convertir le cadre en RGB.
    cadre_rgb = cv2.cvtColor(cadre, cv2.COLOR_BGR2RGB)
    
    # Effectuer la détection des mains.
    resultats = mains.process(cadre_rgb)

    # Dessiner les repères des mains et les cadres englobants.
    if resultats.multi_hand_landmarks:
        for repere_mains in resultats.multi_hand_landmarks:
            # Dessiner les repères.
            mp_dessin.draw_landmarks(cadre, repere_mains, mp_mains.HAND_CONNECTIONS)

            # Calculer le cadre englobant.
            x_min = int(min([lm.x for lm in repere_mains.landmark]) * cadre.shape[1])
            x_max = int(max([lm.x for lm in repere_mains.landmark]) * cadre.shape[1])
            y_min = int(min([lm.y for lm in repere_mains.landmark]) * cadre.shape[0])
            y_max = int(max([lm.y for lm in repere_mains.landmark]) * cadre.shape[0])

            # Dessiner le cadre.
            cv2.rectangle(cadre, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Dessiner l'étiquette.
            cv2.putText(cadre, "Hand", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convertir le cadre en nuances de gris pour la détection des visages.
    gris = cv2.cvtColor(cadre, cv2.COLOR_BGR2GRAY)

    # Effectuer la détection des visages.
    visages = cascade_visage.detectMultiScale(gris, 1.1, 4)
    for (x, y, w, h) in visages:
        # Dessiner le cadre englobant.
        cv2.rectangle(cadre, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Dessiner l'étiquette.
        cv2.putText(cadre, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Recadrer et redimensionner l'image.
        visage_recadre = recadrer_et_redimensionner(cadre, x, y, w, h)

        # Sauvegarder dans fichier visage.jpg
        cv2.imwrite("visage.jpg", visage_recadre)

       # Reconnaissance du visage 
        for image in os.listdir("data"):
            try:
                obj = DeepFace.verify("visage.jpg", os.path.join("data", image), model_name = models[0])
                if obj["verified"] == True:
                    # Montrer nom de l'image (sans l'extension) au-dessus du cadre
                    cv2.putText(cadre, image.split(".")[0], (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except:
                    pass

    # Afficher le cadre avec les détections.
    cv2.imshow(nom_fenetre, cadre)

    touche = cv2.waitKey(1) & 0xFF
    if touche == ord('q'): # Appuyer sur 'q' pour quitter.
        break
    elif touche == ord('c'):  # Appuyer sur 'c' pour prendre un capture d'écran.
        nom_capture = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(os.path.join("captures", nom_capture), cadre)
        print(f"Capture saved as {nom_capture}")
    elif touche == ord('w'):
        luminosite += 10
    elif touche == ord('s'):
        luminosite -= 10
    elif touche == ord('a'):
        contraste -= 10
    elif touche == ord('d'):
        contraste += 10
    elif touche == ord('z'):
        niveau_bruit += 5
    elif touche == ord('x'):
        niveau_bruit -= 5

    # Vérifier si la fenêtre est fermée
    if cv2.getWindowProperty(nom_fenetre, cv2.WND_PROP_VISIBLE) < 1:
        break

# Libérer les ressources.
mains.close()
webcam.release()
cv2.destroyAllWindows()
