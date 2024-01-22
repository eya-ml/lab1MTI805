import cv2

from image_quality import ajuster_luminosite, ajuster_contraste, ajouter_bruit
def capture():
    webcam = cv2.VideoCapture(0)

# Définir la luminosité, le contraste et le niveau de bruit initiaux
    luminosite = 0
    contraste = 0
    niveau_bruit = 0

    nom_fenetre = 'capture'
    cv2.namedWindow(nom_fenetre)

    while True:
        ret, cadre = webcam.read()
        if not ret:
            break

    # Ajuster la luminosité, le contraste et le bruit séparément
        cadre = ajuster_luminosite(cadre, luminosite)
        cadre = ajuster_contraste(cadre, contraste)
        cadre = ajouter_bruit(cadre, niveau_bruit)

        touche = cv2.waitKey(1) & 0xFF
        print("luminosite", luminosite)
        print("contraste", contraste)
        print("niveau_bruit", niveau_bruit)

        if touche == ord('q'):  # Appuyer sur 'q' pour quitter.
            break
        elif touche == ord('c'):  # Appuyer sur 'c' pour prendre un capture d'écran.
            user = input("Entrez nom de candidat ")
            nom_capture = f"capture_{user}.png"
            cv2.imwrite(nom_capture, cadre)
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
    webcam.release()
    cv2.destroyAllWindows()
capture()