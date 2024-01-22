import os

import pandas as pd
import tensorflow as tf
import cv2
import numpy as np
from image_quality import recadrer_et_redimensionner




def primitive(visage,img):
    embedding_genere = False
    resnet_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3), pooling='avg')
    for (x, y, w, h) in visage:

    # Générer l'embedding du visage (seulement le premier visage).
        if not embedding_genere and len(visage) > 0:
            x, y, w, h = visage[0]

        # Recadrer et redimensionner le visage.
            visage_recadre = recadrer_et_redimensionner(img, x, y, w, h)
            visage_recadre = visage_recadre / 255.0  # Normalisation
            embedding_visage = resnet_model.predict(np.expand_dims(visage_recadre, axis=0))
            # Marquer que l'embedding a été généré
            embedding_genere = True

    return embedding_visage


