import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import shutil

# Caminho do modelo treinado
model_path = r'C:\Users\Siraissi\Documents\GitHub\IdentifyPan\trainamento\modelo_classificacao_pan.keras'
model = load_model(model_path)

# Caminhos para salvar feedback
feedback_pan_dir = r'C:\Users\Siraissi\Documents\GitHub\IdentifyPan\feedback\pan'
feedback_not_pan_dir = r'C:\Users\Siraissi\Documents\GitHub\IdentifyPan\feedback\not_pan'
os.makedirs(feedback_pan_dir, exist_ok=True)
os.makedirs(feedback_not_pan_dir, exist_ok=True)

# Função para classificar imagem
def classify_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return prediction[0][0]

# Função para carregar nova imagem
def load_new_image():
    image_path = filedialog.askopenfilename()
    if image_path:
        prediction = classify_image(image_path)
        classification = "Pan" if prediction >= 0.5 else "Não Pan"
        img = Image.open(image_path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        image_label.configure(image=img_tk)
        image_label.image = img_tk
        classification_label.config(text=f'Classificação: {classification}')
        feedback_correct_button.config(state=tk.NORMAL)
        feedback_incorrect_button.config(state=tk.NORMAL)
        feedback_correct_button.image_path = image_path
        feedback_incorrect_button.image_path = image_path
        feedback_correct_button.prediction = prediction
        feedback_incorrect_button.prediction = prediction

# Função para coletar feedback do usuário
def get_user_feedback(correct):
    image_path = feedback_correct_button.image_path if correct else feedback_incorrect_button.image_path
    prediction = feedback_correct_button.prediction if correct else feedback_incorrect_button.prediction
    is_pan = prediction >= 0.5
    
    if correct:
        if is_pan:
            feedback_path = os.path.join(feedback_pan_dir, os.path.basename(image_path))
        else:
            feedback_path = os.path.join(feedback_not_pan_dir, os.path.basename(image_path))
    else:
        if is_pan:
            feedback_path = os.path.join(feedback_not_pan_dir, os.path.basename(image_path))
        else:
            feedback_path = os.path.join(feedback_pan_dir, os.path.basename(image_path))
    
    shutil.move(image_path, feedback_path)
    messagebox.showinfo("Feedback", f"Imagem movida para: {feedback_path}")
    image_label.configure(image='')
    classification_label.config(text='')
    feedback_correct_button.config(state=tk.DISABLED)
    feedback_incorrect_button.config(state=tk.DISABLED)
    
    # Re-treinamento do modelo (simplificado para fins de demonstração)
    data_gen = ImageDataGenerator(rescale=1./255)
    
    if correct:
        if is_pan:
            data_flow = data_gen.flow_from_directory(feedback_pan_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
        else:
            data_flow = data_gen.flow_from_directory(feedback_not_pan_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
    else:
        if is_pan:
            data_flow = data_gen.flow_from_directory(feedback_not_pan_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
        else:
            data_flow = data_gen.flow_from_directory(feedback_pan_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
    
    if len(data_flow) > 0:
        model.fit(data_flow, steps_per_epoch=10, epochs=1)

# Criar a janela principal
root = tk.Tk()
root.title("Classificação de Imagens")

# Configurar os widgets da interface
load_button = tk.Button(root, text="Carregar Nova Imagem", command=load_new_image)
load_button.pack()

image_label = tk.Label(root)
image_label.pack()

classification_label = tk.Label(root, text='')
classification_label.pack()

feedback_frame = tk.Frame(root)
feedback_frame.pack()

feedback_correct_button = tk.Button(feedback_frame, text="Correto", state=tk.DISABLED, command=lambda: get_user_feedback(True))
feedback_correct_button.pack(side=tk.LEFT)

feedback_incorrect_button = tk.Button(feedback_frame, text="Incorreto", state=tk.DISABLED, command=lambda: get_user_feedback(False))
feedback_incorrect_button.pack(side=tk.RIGHT)

# Executar a interface gráfica
root.mainloop()
