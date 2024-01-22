import cv2
import time
def detection_visage_Haar_Cascade(img):
# Charger le Haar Cascade pour la détection des visages.

# Démarrer le chronomètre
    start_time = time.time()
    cascade_visage = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convertir le cadre en nuances de gris pour la détection des visages.
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Effectuer la détection des visages.
    visages = cascade_visage.detectMultiScale(gris, 1.1, 4)


# Arrêter le chronomètre
    end_time = time.time()

# Calculer le temps d'exécution
    execution_time = end_time - start_time
    print("Temps d'exécution du modéle haar cascade : ", execution_time, " secondes")
    for (x, y, w, h) in visages:
        # Dessiner le cadre englobant.
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Dessiner l'étiquette.
        cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Afficher le cadre avec les détections.
    #cv2.imshow("detction du visage", img)
    return visages


