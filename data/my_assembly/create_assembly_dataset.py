import cv2
import os

# Directorio de la carpeta original y la nueva carpeta
original_folder = 'train_B'
new_folder = 'train_A'

# Crear la nueva carpeta si no existe
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# Recorrer las imágenes de la carpeta original
for filename in os.listdir(original_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Asegúrate de incluir otros formatos si es necesario
        # Leer la imagen
        image_path = os.path.join(original_folder, filename)
        image = cv2.imread(image_path, 0)

        # Aplicar el filtro de Canny
        edges = cv2.Canny(image, 20, 200)

        # Guardar la imagen procesada en la nueva carpeta
        new_image_path = os.path.join(new_folder, filename)
        cv2.imwrite(new_image_path, edges)


import json

# Diccionario para los prompts
prompts = {filename: "Assembly piece without defects" for filename in os.listdir(new_folder)}

# Guardar el diccionario en un archivo JSON
with open('train_prompts.json', 'w') as json_file:
    json.dump(prompts, json_file)
