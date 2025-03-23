import os
import sqlite3
import logging
from flask import current_app

# Function to initialize database
def create_database():

    database_path = current_app.config['DATABASE_PATH']
    try:
        with sqlite3.connect(database_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS student (
                   prnno TEXT PRIMARY KEY,
                   name TEXT NOT NULL,
                   embeddings BLOB NOT NULL         
                )
            """)
            conn.commit()
        print(f" Database Created at {current_app.config['DATABASE_PATH']}")
    except Exception as e:
        logging.error(f"Error while creating Database: {e}")
        raise

def attendance_database():

    database_path = current_app.config['DATABASE_PATH']
    try:
        with sqlite3.connect(database_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                   prnno TEXT PRIMARY KEY,
                   name TEXT NOT NULL,
                   timestamp DATETIME DEFAULT CURRECT_TIMESTAMP         
                )
            """)
            conn.commit()
        print(f" Database Created at {current_app.config['DATABASE_PATH']}")
    except Exception as e:
        logging.error(f"Error while creating Database: {e}")
        raise


def add_update_student(prnno, name, mean_embeddings):
    database_path = current_app.config['DATABASE_PATH']
    try:
        with sqlite3.connect(database_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                  INSERT INTO student (prnno, name, embeddings) VALUES (?, ?, ?)
                  ON CONFLICT(prnno) DO UPDATE SET name = excluded.name, embeddings = excluded.embeddings""",
                  (prnno, name, mean_embeddings))
            conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Error while inserting or updating student {name} and PRN: {prnno} : {e}")
        raise

