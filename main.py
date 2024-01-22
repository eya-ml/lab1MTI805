import os
import pickle
import cv2
from detect_MTCNN import detectMTCNN
from detection_visage_Haar_Cascade import detection_visage_Haar_Cascade
from extraire_primitive import primitive
from image_quality import ajuster_luminosite, ajuster_contraste, ajouter_bruit, recadrer_et_redimensionner
import numpy as np
import tensorflow as tf

from train_model import train
# charger le modèle de reconnaissance des visages avec l'encodeur d'étiquettes
recognizer = pickle.loads(open('output/model.pickle', "rb").read())
le = pickle.loads(open('output/label_encoder.pickle', "rb").read())


resnet_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3), pooling='avg')
webcam = cv2.VideoCapture(0)

# Définir la luminosité, le contraste et le niveau de bruit initiaux
luminosite = 0
contraste = 0
niveau_bruit = 0
while True:
    ret, cadre = webcam.read()
    if not ret:
        break

    # Ajuster la luminosité, le contraste et le bruit séparément
    cadre = ajuster_luminosite(cadre, luminosite)
    cadre = ajuster_contraste(cadre, contraste)
    cadre = ajouter_bruit(cadre, niveau_bruit)
    #detecter le visage de 2 façon
    visages1 = detectMTCNN(cadre)
    visages2 = detection_visage_Haar_Cascade(cadre)
    data = os.listdir("data")
    print(data)

    images = []
    #extraire les primitives de visage

    faces_embeddings=[]
    for name in data:
        for filename in os.listdir("data/" + name):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join("data/" + name, filename)
                image = cv2.imread(image_path)
                visages=detection_visage_Haar_Cascade(image)
                face_embeddings1 = primitive(visages,image)

                faces_embeddings.append(face_embeddings1)


    embeddings = list(zip(images, faces_embeddings))

    dataset = {"embeddings": faces_embeddings, "name": data}
    f = open("output/embeddings.pickle", "wb")
    f.write(pickle.dumps(dataset))
    f.close()

    x, y, w, h = visages2[0][0],visages2[0][1],visages2[0][2],visages2[0][3]
    visage_recadre = recadrer_et_redimensionner(cadre, x, y, w, h)
    visage_recadre = visage_recadre / 255.0  # Normalisation
    embedding_visage = resnet_model.predict(np.expand_dims(visage_recadre, axis=0))
    print("len = ",embedding_visage[0])
    #entrainer le modele et faire une prédiction
    train()
    preds = recognizer.predict_proba(embedding_visage)[0]

    if preds[0] > 0.5 or preds[1] > 0.5:
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]
    else:
        name = "eya"
    print("preds", preds)
    # afficher le visage et le nom de personne qu'on a reconnu


    y1 = y - 10 if y - 10 > 10 else y + 10
    cv2.rectangle(cadre, (x, y), (x+w, y+h),
                  (0, 0, 255), 2)
    cv2.putText(cadre, name, (x, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.imshow("result",cadre)
    touche = cv2.waitKey(1) & 0xFF
    print("luminosite", luminosite)
    print("contraste", contraste)
    print("niveau_bruit", niveau_bruit)

    # Ajuster la luminosité, le contraste et le bruit séparément
    if touche == ord('q'):  # Appuyer sur 'q' pour quitter.
        break
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


# Libérer les tout ressources.
webcam.release()
cv2.destroyAllWindows()