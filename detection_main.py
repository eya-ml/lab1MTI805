import cv2
import mediapipe as mp
from datetime import datetime

def detction_main(img):
# Initialiser le modèle de suivi des mains de MediaPipe.
    mp_mains = mp.solutions.hands
    mp_dessin = mp.solutions.drawing_utils
    mains = mp_mains.Hands()
    # Convertir le cadre en RGB.
    cadre_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Effectuer la détection des mains.
    resultats = mains.process(cadre_rgb)

    # Dessiner les repères des mains et les cadres englobants.
    if resultats.multi_hand_landmarks:
        for repere_mains in resultats.multi_hand_landmarks:
            # Dessiner les repères.
            mp_dessin.draw_landmarks(img, repere_mains, mp_mains.HAND_CONNECTIONS)

            # Calculer le cadre englobant.
            x_min = int(min([lm.x for lm in repere_mains.landmark]) * img.shape[1])
            x_max = int(max([lm.x for lm in repere_mains.landmark]) * img.shape[1])
            y_min = int(min([lm.y for lm in repere_mains.landmark]) * img.shape[0])
            y_max = int(max([lm.y for lm in repere_mains.landmark]) * img.shape[0])

            # Dessiner le cadre.
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Dessiner l'étiquette.
            cv2.putText(img, "Hand", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



    # Afficher le cadre avec les détections.
    cv2.imshow("detection main", img)


# Libérer les ressources.
    mains.close()

