
## Face Recognition Toolkit
Cette boîte à outils de reconnaissance des visages fournit un assortiment de scripts Python qui couvrent différents aspects de la technologie de reconnaissance des visages, y compris la capture d'images, la détection des visages, la détection des mains, l'extraction des caractéristiques, et plus encore. 

## Description of Scripts
-** capture_image.py : **
Ce script s'interface avec une webcam pour capturer des images dans des conditions variables. Il est utile pour collecter des ensembles de données.
-** detecte_MTCNN.py: **
Utilise le modèle MTCNN (Multi-task Cascaded Convolutional Networks) pré-entraîné pour détecter les visages dans les images fournies. Il constitue l'étape initiale du processus de reconnaissance des visages.

-** detect_main.py: **
Utilise un modèle MediaPipe pré-entraîné pour reconnaître les gestes de la main dans les images ou les flux vidéo. Cette fonction est particulièrement utile dans les applications qui souhaitent intégrer des interactions main-geste.

-** detecte_visage_haar_cascade.py: **
Applique le classificateur Haar Cascade, une méthode bien établie pour la détection d'objets, afin d'identifier les visages dans les images, ce qui constitue une étape clé dans de nombreux processus de reconnaissance faciale.

-** extraire_primitive.py: **
Responsable de l'extraction des caractéristiques faciales, ou "primitives", des visages détectés. Ces caractéristiques peuvent ensuite être utilisées pour faire correspondre ou identifier des individus dans un système de reconnaissance.

-** image_quality.py : **
Offre une collection de fonctions permettant d'ajuster les paramètres de l'image tels que la taille, la luminosité, le contraste et d'introduire du bruit. Ces fonctions sont importantes pour le prétraitement des images afin d'assurer leur cohérence lorsqu'elles sont introduites dans les modèles de reconnaissance.
-** main.py: **
Joue le rôle de script pilote principal, orchestrant l'ensemble du processus de reconnaissance des visages en tirant parti des fonctionnalités fournies par les autres scripts de cette boîte à outils.

-** train_model.py : **
entraîner les  nouvelles données pour reconnaitre le visage

-** test_efficacite_primitive_visage.py : **
Teste l'efficacité du modèle ResNet-50 dans le contexte de la reconnaissance des visages

## Lien rapport :
https://etsmtl365-my.sharepoint.com/:w:/r/personal/eya_mlika_1_ens_etsmtl_ca/_layouts/15/Doc.aspx?sourcedoc=%7B1D42D5F1-AEAB-434A-96A6-8AF91AEF7EA7%7D&file=lab1.docx&action=default&mobileredirect=tr##
