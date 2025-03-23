import os
import numpy as np 
import torch
import csv
import sqlite3
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import faiss
from flask import current_app
from datetime import datetime, date
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cdist
from src.modules.data_helper import add_update_student

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
yolo_path = r'C:\Python_Development\Attendance System Using Face Recognation\Backend\models\best.pt'
yolo_face_model = YOLO(yolo_path).to(device)

def extract_faces(image):
    image_array = np.array(image)
    results = yolo_face_model(image_array)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return None
    
    face = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    x1, y1, x2, y2 = map(int, face)
    cropped_face = image.crop((x1, y1, x2, y2))
    return cropped_face

def preprocess_face(image, target_size=(160,160)):
    image = image.resize(target_size).convert('RGB')
    face_image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return face_image.to(device)

def get_embeddings():
    conn = sqlite3.connect(current_app.config['DATABASE_PATH'])
    cursor = conn.cursor()
    cursor.execute("SELECT prnno, name, embedding FROM student")
    peoples = cursor.fetchall()
    conn.close()

    embedding_dict = {}
    for prnno, name, embedding_blob in peoples:
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        embedding_dict[prnno] = {"name": name, "embedding": embedding}
    
    return embedding_dict

def store_attendance(prnno, name):
    try:
        conn = sqlite3.connect(current_app.config['DATABASE_PATH'])
        cursor = conn.cursor()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO attendance (prnno, name, timestamp)
            VALUES (?, ?, ?)
        """, (prnno, name, timestamp))

        conn.commit()
        conn.close()

        print(f"Attendance Marked successfully: PRN : {prnno} Name: {name}")
    except sqlite3.Error as e:
        print(f"Error occurred while marking attendance: {e}")

def mark_attendance(image):
    faces = extract_faces(image)

    if faces is None:
        print("No Faces Detected")
        return []

    face_tensor = preprocess_face(faces)
    face_tensor = face_tensor.squeeze(0).to(device)
    face_tensor = torch.stack(face_tensor)

    with torch.no_grad():
        recognized_embedding = embedding_model(face_tensor.unsqueeze(0)).numpy()
    
    conn = sqlite3.connect(current_app.config['DATABASE_PATH'])
    cursor = conn.cursor()
    cursor.execute("SELECT prnno, name, embedding FROM student")
    people_records = cursor.fetchall()

    database_embeddings = []
    student_mapping = []
    
    for prnno, name, embedding_blob in people_records:
        database_embeddings.append(np.frombuffer(embedding_blob,dtype=np.float32))
        student_mapping.append((prnno,name))
    
    database_embeddings = np.vstack(database_embeddings)
    index = faiss.IndexFlatL2(database_embeddings.shape[1])
    index.add(database_embeddings)
    distances, indices = index.search(recognized_embedding,k=1)

    attendance_list = []
    for i, distance in enumerate(distances):
        if distance[0] < 0.7:
            prnno, name = student_mapping[indices[i][0]]
            attendance_list.append((prnno, name))

    csv_dir = "Attendance_record"
    os.makedirs(csv_dir,exist_ok=True)
    today = date.today()
    formatted_date = today.strftime("%Y-%m-%d")
    csv_path = os.path.join(csv_dir,f"attendance_{formatted_date}.csv")

    previous_data = set()
    if os.path.exists(csv_path) and os.stat(csv_path).st_size > 0:
        with open(csv_path, mode='r') as file:
            reader =  csv.reader(file)
            next(reader)
            for row in reader:
                previous_data.add(row[0],formatted_date)

    with open(csv_path,mode='a',newline="") as file:
        writter = csv.writer(file)
        if os.stat(csv_path).st_size == 0:
            writter.writerow(["PrnNo", "Name","Timestamp"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for prnno, name in attendance_list:
            if(prnno,formatted_date) not in previous_data:
                writter.writerow([prnno,name,timestamp])
                previous_data.add((prnno,formatted_date))
    
    print(f"Attendance Marked for {len(attendance_list)} students")























    
    # recognized_people = []
    # for prnno, data in database_embeddings.items():
    #     similarity = cdist([data['embedding']], recognized_embedding.reshape(1, -1), metric='cosine').flatten()[0]
    #     print(similarity)
    #     break

    #     # # Check if the similarity is above the threshold (90% match)
    #     # if similarity < 0.1:  # cosine similarity threshold for 90% match
    #     #     recognized_people.append((prnno, data['name']))
    #     #     store_attendance(prnno, data['name'])
            
    # return recognized_people