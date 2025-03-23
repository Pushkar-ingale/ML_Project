import os 
import cv2
import sqlite3
import numpy as np 
import torch
import base64
from io import BytesIO
from PIL import Image
from flask import current_app
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from src.modules.data_helper import add_update_student


    
yolo_path = r'C:\Python_Development\Attendance System Using Face Recognation\Backend\models\best.pt'
embedding_path = r'C:\Python_Development\Attendance System Using Face Recognation\Backend\models\embedding_model.keras'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

yolo_face_model = YOLO(yolo_path).to(device)
embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def extract_faces(image):
    image_array = np.array(image)
    results = yolo_face_model(image_array)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return None
    
    face = max(boxes, key = lambda box : (box[2] - box[0]) * (box[3] - box[1]))
    x1, y1, x2, y2 = map(int, face)
    cropped_face = image.crop((x1,y1,x2,y2))
    # cropped_face_cv = np.array(cropped_face)
    # cropped_face_cv = cv2.cvtColor(cropped_face_cv,cv2.COLOR_RGB2BGR)
    # cv2.imshow("Cropped Face", cropped_face_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropped_face

        
def preprocess_face(image,target_size=(160,160)):
    image = image.resize(target_size).convert('RGB')
    face_image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
    return face_image.to(device)

def get_faces(prnno,name,images):
    
    face_embeddings = []
    for image in images:
        image_data = Image.open(BytesIO(base64.b64decode(image.split(",")[1])))
        face = extract_faces(image_data)
        if face is None:
            print("Faces are not detected !!")
            continue

        face = preprocess_face(face)
        with torch.no_grad():
            embeddings = embedding_model(face).cpu().numpy().squeeze()
            face_embeddings.append(embeddings)

    if len(face_embeddings) == 0:
        return
    
    mean_face_embeddings = np.mean(face_embeddings,axis=0).tobytes()
    add_update_student(prnno,name,mean_face_embeddings)
        


def get_embeddings():
    conn = sqlite3.connect(current_app.config['DATABASE_PATH'])
    cursor = conn.cursor()
    cursor.execute("SELECT prnno, name, embedding FROM student")
    peoples = cursor.fetchall()
    conn.close()

    embedding_dict = {}
    for prnno, name , embedding_blob in peoples:
        embedding = np.frombuffer(embedding_blob,dtype=np.float32)
        embedding_dict[prnno] = {"name":name, "embedding":embedding}
    
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

        print(f"Attendance Marked sucessfully: PRN : {prnno} Name: {name}")
    except sqlite3.Error as e:
        print(f"Error occured while marking attendance: {e}")