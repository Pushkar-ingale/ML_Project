import os
import cv2
import logging
import base64
import torch
import traceback
import sqlite3
import numpy as np 
from io import BytesIO
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from flask import Flask, jsonify, request, redirect, url_for
from flask_cors import CORS
from config.config import Config
from werkzeug.utils import secure_filename
from src.modules.data_helper import create_database, attendance_database
from src.modules.database import get_faces, extract_faces
from src.modules.Attendance import mark_attendance


app = Flask(__name__)
cors = CORS(app, origins='*')
app.config.from_object(Config)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

with app.app_context():
    create_database()
    attendance_database()
    

# Checking file is valid or not
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_path = r'C:\Python_Development\Attendance System Using Face Recognation\Backend\models\best.pt'
yolo_model = YOLO(yolo_path).to(device)

facenet_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Function for to clean upload folder after marking attendance
def clean_upload_file(filepath):
    """ Delete file if exitsts """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logging.info(f"File {filepath} Deleted Sucessfully")
    except Exception as e:
        logging.error(f"Eror during Deletion: {e} ")


@app.route('/submit', methods = ['POST'])
def submit():
    """ Get and Add Employee Data"""
    try:
        data = request.get_json()
        
        prnno = data.get('prnno')
        name = data.get('name')
        images = data.get('images')

        if not prnno or not name or not images:
            return jsonify({"error":"Missing required fields"}),400
        
        # Capture Employee face
        get_faces(prnno,name,images)
        
        return jsonify({"message":"Data submitted successfully."}),200
    except ValueError as e:
        logging.error(f"ValueError in /submit: {e}")
        return jsonify({"error":str(e)}),400
    except Exception as e:
        logging.error(f"Unexpected happen in /submit: {traceback.format_exc()}")
        return jsonify({"error":"An Unexpected error occured"}),500
    
    

@app.route("/process_frame", methods=["POST"])
def process_student_image():
    try:
        data = request.json
        image_data = base64.b64decode(data['image'].split(",")[1])
        image = Image.open(BytesIO(image_data)).convert("RGB")

        recognized_people = mark_attendance(image)
        if recognized_people:
            print(f"Attendance Marked PRN : {recognized_people[-1][0]} Name : {recognized_people[-1][1]}")
            return jsonify({"success": True, "recognized": True, "name": recognized_people[-1][1], "prnno": recognized_people[-1][0]})
        else:
            return jsonify({"success": False, "error": "Face not recognized"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)