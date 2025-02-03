import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import torch
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model = torch.hub.load('ultralytics/yolov5', 'custom', path='gundetection.pt')

def load_img():
    global original_img, img_tk
    file_path = filedialog.askopenfilename()
    if file_path:
        original_img = Image.open(file_path)
        img_tk = ImageTk.PhotoImage(original_img)
        img_label.config(image=img_tk)

def detect_obj():
    global original_img, img_tk_detected
    if 'original_img' in globals() and original_img:
        results = model(original_img)
        detections = results.pandas().xyxy[0]
        img_with_boxes = original_img.copy()
        draw = ImageDraw.Draw(img_with_boxes)

        for _, detect in detections.iterrows():
            x_min, y_min, x_max, y_max, confidence, class_id, class_name = detect.values
            label = f"{class_name} {confidence:.2f}"
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)
            text_position = (x_min, y_min - 10) if y_min > 20 else (x_min, y_min + 10)
            draw.text(text_position, label, fill="red")

        img_tk_detected = ImageTk.PhotoImage(img_with_boxes)
        img_label.config(image=img_tk_detected)
    else:
        tk.messagebox.showinfo("Info", "กรุณาโหลดรูปภาพก่อน")

root = tk.Tk()
root.title("Object Detection")

bg_color = '#f0f0f0'
root.configure(bg=bg_color)

font_style = ('Helvetica', 10)
title_font = ('Helvetica', 12, 'bold')

img_frame = tk.Frame(root, bg=bg_color)
img_frame.grid(row=0, column=0, padx=20, pady=20)

img_title = tk.Label(img_frame, text="รูปภาพ", font=title_font, bg=bg_color)
img_title.grid(row=0, column=0, padx=5, pady=(0, 5), sticky="w")

img_label = tk.Label(img_frame, relief=tk.SOLID, borderwidth=1, bg='white')
img_label.grid(row=1, column=0, padx=5, pady=5)

button_frame = tk.Frame(root, bg=bg_color)
button_frame.grid(row=1, column=0, pady=10)

load_button = tk.Button(button_frame, text="Load Image", command=load_img, font=font_style,
                         bg='#e0e0e0', fg='black', padx=10, pady=5, relief=tk.RAISED, bd=2)
load_button.grid(row=0, column=0, padx=10)

detect_button = tk.Button(button_frame, text="Detect Objects", command=detect_obj, font=font_style,
                          bg='#4CAF50', fg='white', padx=10, pady=5, relief=tk.RAISED, bd=2)
detect_button.grid(row=0, column=1, padx=10)

root.mainloop()