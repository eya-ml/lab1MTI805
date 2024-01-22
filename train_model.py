import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
def train():
    #lire données
    with open("output/embeddings.pickle", "rb") as file:
        data = pickle.load(file)
    print("data[embeddings] = ",data["embeddings"])
    print("names",data["name"])
    #entrainer le modele
    le = LabelEncoder()
    labels = le.fit_transform(data["name"])
    print(labels)
    recognizer = SVC(C=2, kernel="rbf", probability=True)
    embeddings = np.concatenate(data["embeddings"], axis=0)
    recognizer.fit(embeddings, labels)
    #engregistrer le modele
    f = open('output/model.pickle', "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    f = open('output/label_encoder.pickle', "wb")
    f.write(pickle.dumps(le))
    f.close()

