import tkinter as tk
from tkinter import filedialog
from tkinter import *

from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

import pandas as pd
from sklearn.cluster import KMeans
from ultralytics import YOLO
model = YOLO('yolov8s.pt')

top = tk.Tk()
top.geometry('1200x800')
top.title('Traffic Analysis')
top.configure(background = '#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def InitializeModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

def Detect(file_path):
    global label_packed
    image = cv2.imread(file_path)
    try:
        image = cv2.imread(file_path)
        results = model.predict(image)
        a = results[0].boxes.data
        b = a.detach().cpu().numpy()
        px = pd.DataFrame(b).astype("float")
        list = []
        p_list = []
        v_list = []

        for index,row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            if d == 2:
                list.append([x1,y1,x2,y2])
            elif d == 0:
                p_list.append([x1,y1,x2,y2])
            elif d == 1 or d == 3 or d == 5 or d == 7 :
                v_list.append([x1,y1,x2,y2])
        
        for vals in v_list:
            x,y,w,h = vals
            cv2.rectangle(image,(x,y),(w,h),(105,105,105),2)

        for vals in list:
            x,y,w,h = vals
            int_img = image[y:h,x:w]
            int_img = cv2.cvtColor(int_img, cv2.COLOR_BGR2RGB)
            pixels = int_img.reshape(-1, 3)
            kmeans = KMeans(n_clusters=1)
            kmeans.fit(pixels)
            dominant_color = kmeans.cluster_centers_[0]
            [R,G,B] = dominant_color
            [R,G,B] = [round(R),round(G),round(B)]
            cv2.rectangle(image,(x,y),(w,h),(R,G,B),2)

        for vals in p_list:
            x,y,z,w = vals
            int_img = image[y:w,x:z]
            int_img = cv2.cvtColor(int_img, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(int_img,(64,64))
            try:
                Genmodel = InitializeModel("model_gender1.json","gender_model_weights1.h5")
                GENDER_LIST = ["Female", "Male"]
                predicted_gender = GENDER_LIST[np.argmax(Genmodel.predict(roi[np.newaxis,:,:,np.newaxis]))]
                if predicted_gender == "Female":
                    color = (203,192,255)
                else:
                    color = (235,206,135)
            except:
                color = (255,255,255)
            cv2.rectangle(image,(x,y),(z,w),color,2)
        res = f"Detected {len(list)} car(s), {len(p_list)} people(s) and {len(v_list)} other vehicle(s).\n"
        print(res)
        label1.configure(foreground="#011638",text = res)
        reupload_image(image)
    except:
            label1.configure(foreground="#011638",text = "Unable to read image")

def show_detect_button(file_path):
    detect_b = Button(top,text=">>Detect<<", command=lambda:Detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground="white", font=('arial',10,'bold'))
    detect_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2),(top.winfo_height()/2)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_detect_button(file_path)
    except:
        pass
    
def reupload_image(image_update):
    try:
        image_update = Image.fromarray(cv2.cvtColor(image_update, cv2.COLOR_BGR2RGB))
        image_update.thumbnail(((top.winfo_width() / 2), (top.winfo_height() / 2)))
        im = ImageTk.PhotoImage(image_update)
        sign_image.configure(image=im)
        sign_image.image = im
    except:
        pass
    
upload = Button(top,text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom',expand='True')
label1.pack(side='bottom',expand='True')
heading = Label(top,text='Traffic Analysis', pady=20, font=('arial',25,'bold'))
heading.configure(background="#CDCDCD",foreground="#364156")
heading.pack()
top.mainloop()