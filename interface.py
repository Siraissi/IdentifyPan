import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Função para carregar e pré-processar a imagem
def load_image(img_path):
    img_width, img_height = 150, 150  # Deve ser o mesmo tamanho usado no treinamento
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalização
    return img_array

# Função para carregar o modelo salvo
def load_model():
    return tf.keras.models.load_model(r'C:\Users\Siraissi\Documents\GitHub\IdentifyPan\trainamento\modelo_classificacao_pan.keras')

# Função para fazer a predição
def predict_image(img_array, model):
    prediction = model.predict(img_array)
    probability = prediction[0][0] * 100
    return probability, prediction[0] > 0.5

# Função para abrir o diálogo de seleção de arquivo
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        display_image(file_path)
        img_array = load_image(file_path)
        probability, is_pan = predict_image(img_array, model)
        result_text.set(f"Probabilidade de a imagem pertencer à classe 'pan': {probability:.2f}%")
        if is_pan:
            messagebox.showinfo("Resultado", "A imagem pertence à classe 'pan'.")
        else:
            messagebox.showinfo("Resultado", "A imagem não pertence à classe 'pan'.")

# Função para exibir a imagem selecionada
def display_image(file_path):
    img = Image.open(file_path)
    img = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

# Função para mudar de frame
def switch_frame(frame):
    frame.tkraise()

# Função para mostrar mensagem de área em desenvolvimento
def show_message():
    messagebox.showinfo("Em Desenvolvimento", "Esta área ainda não está pronta.")

# Criar a janela principal
root = tk.Tk()
root.title("Classificação de Imagens")

# Carregar o modelo
model = load_model()

# Configurar o container principal para os frames
container = tk.Frame(root)
container.pack(fill="both", expand=True)

# Inicializar os frames
frames = {}
class HomeFrame(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Bem-vindo!", font=("Helvetica", 16))
        label.pack(pady=10)
        train_button = tk.Button(self, text="Treinamento", command=show_message)
        train_button.pack(pady=10)
        classify_button = tk.Button(self, text="Classificação de Imagem", command=lambda: switch_frame(frames["ClassifyFrame"]))
        classify_button.pack(pady=10)

class TrainFrame(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Treinamento - Em Desenvolvimento", font=("Helvetica", 16))
        label.pack(pady=10)
        back_button = tk.Button(self, text="Voltar", command=lambda: switch_frame(frames["HomeFrame"]))
        back_button.pack(pady=10)

class ClassifyFrame(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        frame = tk.Frame(self)
        frame.pack(pady=10)
        btn_load = tk.Button(frame, text="Carregar Imagem", command=open_file)
        btn_load.pack(side=tk.LEFT, padx=10)
        global result_text
        result_text = tk.StringVar()
        result_label = tk.Label(self, textvariable=result_text)
        result_label.pack(pady=10)
        global image_label
        image_label = tk.Label(self)
        image_label.pack(pady=10)
        back_button = tk.Button(self, text="Voltar", command=lambda: switch_frame(frames["HomeFrame"]))
        back_button.pack(pady=10)

for F in (HomeFrame, TrainFrame, ClassifyFrame):
    page_name = F.__name__
    frame = F(parent=container, controller=root)
    frames[page_name] = frame
    frame.grid(row=0, column=0, sticky="nsew")

# Mostrar a página inicial
switch_frame(frames["HomeFrame"])

# Iniciar o loop principal da interface gráfica
root.mainloop()
