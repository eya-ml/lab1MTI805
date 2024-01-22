import cv2
import numpy as np
from detection_visage_Haar_Cascade import detection_visage_Haar_Cascade
from extraire_primitive import primitive
#fonction pour calculer la distance euclidienne entre 2 vecteurs
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)
# Lire les images
image1 = cv2.imread("data/eya/eya luminosite.png")
image2 = cv2.imread("capture_eya.png")
image3 = cv2.imread("data/bianca/capture_bianca.png")

visages1 = detection_visage_Haar_Cascade(image1)
visages2 = detection_visage_Haar_Cascade(image2)
visages3 = detection_visage_Haar_Cascade(image3)
# Extraire primitive de visage en utilisantResNet50
face_encoding1 = primitive(visages1,image1)[0]
face_encoding2 = primitive(visages2,image2)[0]
face_encoding3 = primitive(visages3,image3)[0]

#  Calculer les distances entre meme classes
intra_distance = euclidean_distance(face_encoding1, face_encoding2)

# Calculer les distances entre different classes
inter_distance = euclidean_distance(face_encoding1, face_encoding3)

print("Intra-Class Distance:", intra_distance)
print("Inter-Class Distance:", inter_distance)