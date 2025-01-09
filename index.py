import cv2
import os
from deepface import DeepFace
import numpy as np

# Charger les images de référence
def charger_images_references(dossier_images):
    base_donnees = []

    for fichier in os.listdir(dossier_images):
        if fichier.endswith('.jpg') or fichier.endswith('.png'):
            try:
                chemin_image = os.path.join(dossier_images, fichier)
                embedding = DeepFace.represent(img_path=chemin_image, model_name="Facenet")
                
                # Si DeepFace renvoie une liste, on prend l'élément approprié
                if isinstance(embedding, list):
                    embedding = np.array(embedding[0]['embedding'])
                else:
                    embedding = np.array(embedding)
                
                base_donnees.append({
                    "nom": os.path.splitext(fichier)[0],
                    "embedding": embedding
                })
                print(f"Encodage ajouté pour {fichier}.")
            except Exception as e:
                print(f"Erreur lors de l'encodage de {fichier}: {e}")

    return base_donnees


# Vérifier la correspondance avec la base de données
def trouver_correspondance(face_embedding, base_donnees, seuil=0.5):
    best_match = None
    best_distance = float('inf')  # On initialise avec une très grande valeur

    for personne in base_donnees:
        # S'assurer que face_embedding et personne["embedding"] sont bien des tableaux numpy
        if isinstance(personne["embedding"], np.ndarray) and isinstance(face_embedding, np.ndarray):
            distance = np.linalg.norm(personne["embedding"] - face_embedding)  # Calcul de la distance euclidienne
            if distance < best_distance:
                best_distance = distance
                best_match = personne["nom"]

    # Calcul du pourcentage de ressemblance
    if best_match is not None:
        max_distance = 10  # Vous pouvez ajuster cette valeur si nécessaire (maximiser ou minimiser la distance)
        similarity_percentage = max(0, min(100, (1 - best_distance / max_distance) * 100))
        return best_match, similarity_percentage
    else:
        return "Inconnu", 0


# Dossier contenant les images de référence
dossier_images = "images"
base_donnees = charger_images_references(dossier_images)

if not base_donnees:
    print("Aucune donnée de référence trouvée. Vérifiez le dossier 'images'.")
    exit()

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam")
    exit()

print("Appuyez sur 'q' pour quitter")

while True:
    try:
        # Capture de la webcam
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire l'image de la webcam")
            break

        # Détection des visages avec HaarCascade (OpenCV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        print(f"Nombre de visages détectés : {len(faces)}")  # Affichage du nombre de visages détectés

        for (x, y, w, h) in faces:
            # Extraire le visage détecté
            face_roi = frame[y:y+h, x:x+w]

            try:
                # Convertir l'image en RGB et redimensionner à 160x160 pour la compatibilité avec DeepFace
                face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                face_roi_resized = cv2.resize(face_roi_rgb, (160, 160))  # Redimensionner à 160x160

                # Obtenir l'encodage du visage avec DeepFace
                face_embedding = DeepFace.represent(img_path=face_roi_resized, model_name="Facenet", enforce_detection=False)

                # Si DeepFace renvoie une liste, prendre l'élément approprié
                if isinstance(face_embedding, list):
                    face_embedding = np.array(face_embedding[0]['embedding'])
                else:
                    face_embedding = np.array(face_embedding)

                # Trouver la correspondance dans la base de données
                nom, similarity_percentage = trouver_correspondance(face_embedding, base_donnees)

                # Dessiner un rectangle autour du visage
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Afficher le nom ou "Inconnu" avec le pourcentage de ressemblance
                label = f"{nom} ({similarity_percentage:.2f}%)" if nom != "Inconnu" else f"Inconnu ({similarity_percentage:.2f}%)"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            except Exception as e:
                print(f"Erreur lors du traitement du visage : {e}")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Erreur", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Afficher le flux vidéo avec les rectangles
        cv2.imshow("Reconnaissance de visage - DeepFace", frame)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Erreur dans la boucle principale : {e}")
        continue

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
