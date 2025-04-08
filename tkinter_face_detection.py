import cv2
import numpy as np
import face_recognition
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from datetime import datetime
import csv
import os

detected_faces = set()

def log_attendance(name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
    file_exists = os.path.isfile("Detection.csv")

    with open("Detection.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Timestamp"])
        writer.writerow([name, dt_string])

def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    try:
        image_files = ["Ganesh.jpg", "Rohit.jpg", "Avinash.jpg", "Dnyandip.jpg"]
        names = ["Ganesh", "Rohit", "Avinash", "Dnyandip"]

        for img_path, name in zip(image_files, names):
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(name)
            else:
                print(f"Warning: No face found in {img_path}")

    except Exception as e:
        print("Error loading images:", e)

    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces()

root = tk.Tk()
root.title("🎯 Face Recognition System 🎯")
root.configure(bg="#222831")

title_label = Label(root, text="Real-Time Face Recognition", font=("Helvetica", 20, "bold"), bg="#00ADB5", fg="white", pady=10)
title_label.pack(fill=tk.X)

video_label = Label(root, bg="#393E46")
video_label.pack(padx=10, pady=10)

info_label = Label(root, text="", font=("Helvetica", 12), bg="#222831", fg="white")
info_label.pack()

video_capture = cv2.VideoCapture(0)

def process_frame():
    ret, frame = video_capture.read()
    if not ret:
        return

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(distances) == 0:
            continue
        best_match_index = np.argmin(distances)
        name = "Unknown"

        # THRESHOLD set to 0.5 instead of 0.45
        if distances[best_match_index] < 0.5:
            name = known_face_names[best_match_index]

        face_names.append(name)

        if name != "Unknown" and name not in detected_faces:
            detected_faces.add(name)
            log_attendance(name)

    known_count = face_names.count("Unknown")
    unknown_count = len(face_names) - known_count

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Just show rectangle — no blur
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    info_label.config(text=f"Total Detected: {len(face_names)} | Known: {unknown_count} | Unknown: {known_count}")

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, process_frame)

def start_camera():
    process_frame()

def close_camera():
    video_capture.release()
    cv2.destroyAllWindows()

    print("\n--- Detected Faces During Session ---")
    if detected_faces:
        for name in detected_faces:
            print(name)
    else:
        print("No known faces detected.")

    root.quit()

button_style = {
    "font": ("Helvetica", 14, "bold"),
    "padx": 20,
    "pady": 10,
    "bd": 0,
    "fg": "white",
    "activeforeground": "white"
}

start_btn = Button(root, text="🚀 Start Camera", command=start_camera, bg="#4CAF50", activebackground="#45a049", **button_style)
start_btn.pack(pady=10)

exit_btn = Button(root, text="❌ Exit", command=close_camera, bg="#f44336", activebackground="#e53935", **button_style)
exit_btn.pack(pady=10)

root.mainloop()
