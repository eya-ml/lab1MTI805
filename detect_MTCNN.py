import cv2
from mtcnn.mtcnn import MTCNN
import time
def detectMTCNN(img):
    visages=[]
# Démarrer le chronomètre
    start_time = time.time()
# crer un detecteur enutilisant le modele MTCNN
    detector = MTCNN()
# detecter les visages dans l'image
    faces = detector.detect_faces(img)
# Arrêter le chronomètre
    end_time = time.time()

# Calculer le temps d'exécution
    execution_time = end_time - start_time
    print("Temps d'exécution du modéle MTCNN : ", execution_time, " secondes")


# afficher les visages détecter

    for result in faces:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        cv2.rectangle(img, (x, y), (x + width, y + height),
                      (0, 0, 255), 2)
        visages.append([x, y, width, height])
    cv2.imshow('Detected Faces', img)
    return visages