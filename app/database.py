import sqlite3
import logging
from datetime import datetime
import numpy as np
import json
import csv
import threading
from app import config
import atexit

_conn = None
_db_lock = threading.Lock()

def get_connection():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.execute("PRAGMA busy_timeout=5000")
    return _conn

def close_connection():
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None

atexit.register(close_connection)

def init_tables():
    """Ensure that all required database tables exist."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS persons(
            person_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            profile_image_path TEXT,
            created_at TEXT
        )
        """)
        
        # Check for missing columns in persons table (migration)
        cur.execute("PRAGMA table_info(persons)")
        person_columns = {row[1] for row in cur.fetchall()}
        if "profile_image_path" not in person_columns:
            cur.execute("ALTER TABLE persons ADD COLUMN profile_image_path TEXT")
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS images(
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            image_path TEXT,
            cropped_path TEXT,
            pose TEXT,
            expression TEXT,
            lighting_condition TEXT,
            detector_name TEXT,
            session_id TEXT,
            captured_at TEXT,
            width INTEGER,
            height INTEGER,
            cropped_width INTEGER,
            cropped_height INTEGER,
            blur_score REAL,
            brightness_score REAL
        )
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings(
            embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            person_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            detector_name TEXT,
            session_id TEXT,
            embedding_dim INTEGER NOT NULL,
            embedding_vector BLOB NOT NULL,
            embedding_preview TEXT,
            embedding_norm REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(image_id) REFERENCES images(image_id),
            FOREIGN KEY(person_id) REFERENCES persons(person_id)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS recognition_events(
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            person_name TEXT,
            similarity_score REAL,
            threshold_used REAL,
            detector_name TEXT,
            recognizer_name TEXT,
            timestamp TEXT
        )
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance_logs(
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            person_name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            recognition_score REAL,
            model_name TEXT,
            detector_name TEXT,
            device TEXT,
            status TEXT DEFAULT 'IN',
            FOREIGN KEY(person_id) REFERENCES persons(person_id)
        )
        """)

        # Migration: Ensure status column exists in existing databases
        cur.execute("PRAGMA table_info(attendance_logs)")
        columns = [col[1] for col in cur.fetchall()]
        if "status" not in columns:
            cur.execute("ALTER TABLE attendance_logs ADD COLUMN status TEXT DEFAULT 'IN'")

        # Indexes for performance
        cur.execute("CREATE INDEX IF NOT EXISTS idx_images_person_id ON images(person_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_embeddings_model_person ON face_embeddings(model_name, person_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_recognition_events_person_id ON recognition_events(person_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_attendance_logs_person_id ON attendance_logs(person_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_attendance_logs_date ON attendance_logs(date)")
        
        # Deduplicate recognition_events to avoid IntegrityError if duplicates already exist
        cur.execute("""
        DELETE FROM recognition_events
        WHERE event_id NOT IN (
            SELECT MIN(event_id)
            FROM recognition_events
            GROUP BY person_id, substr(timestamp, 1, 10)
        )
        """)
        
        # Drop old uniqueness constraints to allow multiple daily check-ins
        cur.execute("DROP INDEX IF EXISTS idx_unique_daily_attendance")
        cur.execute("DROP INDEX IF EXISTS idx_unique_daily_attendance_logs")
        
        # Create non-unique indexes for performance
        cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_attendance ON recognition_events(person_id, substr(timestamp, 1, 10))")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_attendance_logs ON attendance_logs(person_id, date)")
        
        conn.commit()


def get_or_create_person(name):
    """Get or create a person in the database."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT person_id FROM persons WHERE name=?", (name,))
        row = cur.fetchone()

        if row:
            pid = row[0]
        else:
            cur.execute(
                "INSERT INTO persons(name,created_at) VALUES(?,?)",
                (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            pid = cur.lastrowid
        return pid

def delete_person(person_id: int):
    """Delete a person, their images, and their embeddings from the database."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT image_id FROM images WHERE person_id = ?", (person_id,))
        image_ids = [row[0] for row in cur.fetchall()]
        
        if image_ids:
            placeholders = ",".join("?" for _ in image_ids)
            cur.execute(f"DELETE FROM face_embeddings WHERE image_id IN ({placeholders})", image_ids)
            cur.execute("DELETE FROM images WHERE person_id = ?", (person_id,))
            
        cur.execute("DELETE FROM persons WHERE person_id = ?", (person_id,))
        conn.commit()

def set_profile_image_path(person_id, profile_image_path):
    """Set the representative profile image for a person."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE persons SET profile_image_path = ? WHERE person_id = ?",
            (profile_image_path, person_id)
        )
        conn.commit()

def save_record(
    person_id,
    image_path,
    cropped_path,
    pose,
    expression,
    lighting_condition,
    width,
    height,
    cropped_width,
    cropped_height,
    blur_score,
    brightness_score,
    detector_name=None,
    session_id=None
):
    """Save an image record to the database."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO images(
            person_id, image_path, cropped_path, pose, expression, lighting_condition,
            detector_name, session_id, captured_at, width, height, cropped_width, cropped_height,
            blur_score, brightness_score
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            person_id, image_path, cropped_path, pose, expression, lighting_condition,
            detector_name, session_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            width, height, cropped_width, cropped_height, blur_score, brightness_score
        ))
        conn.commit()
        image_id = cur.lastrowid
        return image_id

def get_images_for_person(person_id: int):
    """Retrieve all images associated with a person."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT image_id, image_path, cropped_path, blur_score, 
                   cropped_width, cropped_height
            FROM images
            WHERE person_id = ?
        """, (person_id,))
        return cur.fetchall()

def nullify_image_paths(image_id: int):
    """Set the image and cropped paths to NULL for a given image ID."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE images 
            SET image_path = NULL, cropped_path = NULL 
            WHERE image_id = ?
        """, (image_id,))
        conn.commit()

def save_embedding_record(
    image_id,
    person_id,
    model_name,
    embedding_vector,
    embedding_dim,
    detector_name=None,
    session_id=None,
):
    """Save an embedding record to the database with preview and norm calculation."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()

        # Convert to float32 numpy array
        emb_array = np.asarray(embedding_vector, dtype=np.float32)
        emb_bytes = emb_array.tobytes()
        
        # Compute L2 norm
        emb_norm = float(np.linalg.norm(emb_array))
        
        # Generate preview string
        preview_values = emb_array[:10]
        embedding_preview = "[" + ", ".join(f"{v:.3f}" for v in preview_values) + ", ...]"

        cur.execute("""
        INSERT INTO face_embeddings(
            image_id, person_id, model_name, detector_name, session_id,
            embedding_dim, embedding_vector, embedding_preview, embedding_norm, created_at
        )
        VALUES(?,?,?,?,?,?,?,?,?,?)
        """, (
            image_id, person_id, model_name, detector_name, session_id,
            embedding_dim, emb_bytes, embedding_preview, emb_norm,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        embedding_id = cur.lastrowid
        return embedding_id

def load_gallery_embeddings(model_name=config.RECOGNIZER_MODEL_NAME):
    """Load every embedding for model_name, joined with person name."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT fe.embedding_id, fe.image_id, fe.person_id, p.name,
                   fe.embedding_dim, fe.embedding_vector
            FROM face_embeddings fe
            JOIN persons p ON fe.person_id = p.person_id
            WHERE fe.model_name = ?
            ORDER BY fe.person_id, fe.image_id
        """, (model_name,))
        rows = cur.fetchall()

    results = []
    for r in rows:
        results.append({
            "embedding_id": r[0],
            "image_id": r[1],
            "person_id": r[2],
            "name": r[3],
            "embedding_dim": r[4],
            "embedding_vector": np.frombuffer(r[5], dtype=np.float32).copy(),
        })
    return results

def load_persons():
    """Return all persons as a list of dicts."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT person_id, name FROM persons ORDER BY person_id")
        rows = cur.fetchall()
    return [{"person_id": r[0], "name": r[1]} for r in rows]

def log_recognition_event(
    person_id,
    person_name,
    similarity_score,
    threshold_used,
    detector_name="MediaPipe",
    recognizer_name="FaceNet",
):
    """Log a recognition event to the database."""
    try:
        with _db_lock:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO recognition_events(
                person_id, person_name, similarity_score, threshold_used,
                detector_name, recognizer_name, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                person_id,
                person_name,
                float(similarity_score) if similarity_score is not None else None,
                float(threshold_used) if threshold_used is not None else None,
                detector_name,
                recognizer_name,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
    except sqlite3.IntegrityError:
        # Ignore duplicate daily attendance entries due to unique constraint
        pass
    except Exception as e:
        logging.warning(f" Failed to log recognition event to database: {e}")

def log_attendance(person_id: int, person_name: str, recognition_score: float = None, model_name: str = "FaceNet", detector_name: str = "MediaPipe", device: str = "RP5") -> str:
    """Log a successful recognition as an attendance entry with alternating status (IN/OUT)."""
    if person_id is None:
        return "IN"
    
    status = "IN"
    try:
        with _db_lock:
            conn = get_connection()
            cur = conn.cursor()
            
            # Alternate status per person globally
            cur.execute("""
                SELECT status FROM attendance_logs 
                WHERE person_id = ? 
                ORDER BY timestamp DESC LIMIT 1
            """, (person_id,))
            row = cur.fetchone()
            if row and row[0] == "IN":
                status = "OUT"
            
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            timestamp_str = now.isoformat()
            
            cur.execute("""
            INSERT INTO attendance_logs(
                person_id, person_name, date, time, timestamp,
                recognition_score, model_name, detector_name, device, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                person_id, person_name, date_str, time_str, timestamp_str,
                float(recognition_score) if recognition_score is not None else None,
                model_name, detector_name, device, status
            ))
            conn.commit()
    except sqlite3.IntegrityError:
        pass
    except Exception as e:
        logging.error(f" Failed to log attendance: {e}")
    return status

def get_today_logs():
    """Retrieve all attendance logs for today."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    return get_logs_by_date(date_str)

def get_logs_by_person(person_id: int):
    """Retrieve all attendance logs for a specific person."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM attendance_logs WHERE person_id = ? ORDER BY timestamp DESC", (person_id,))
        columns = [col[0] for col in cur.description]
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    return rows

def get_logs_by_date(date_str: str):
    """Retrieve all attendance logs for a specific date (YYYY-MM-DD)."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM attendance_logs WHERE date = ? ORDER BY timestamp DESC", (date_str,))
        columns = [col[0] for col in cur.description]
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    return rows
def get_all_logs(limit: int = 1000):
    """Retrieve all attendance logs ordered by timestamp DESC up to limit."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM attendance_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
        columns = [col[0] for col in cur.description]
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    return rows

def clear_attendance_logs():
    """Clear all attendance logs."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM attendance_logs")
        conn.commit()

def clear_recognition_events():
    """Clear all face recognition events."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM recognition_events")
        conn.commit()

def clear_all_logs():
    """Clear both attendance logs and recognition events."""
    clear_attendance_logs()
    clear_recognition_events()

def count_logs() -> int:
    """Return the total number of attendance logs."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM attendance_logs")
        count = cur.fetchone()[0]
    return count

def export_logs_csv(filepath: str):
    """Export all attendance logs to a CSV file."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM attendance_logs ORDER BY timestamp DESC")
        rows = cur.fetchall()
        
        if rows:
            columns = [col[0] for col in cur.description]
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerows(rows)

def export_logs_json(filepath: str):
    """Export all attendance logs to a JSON file."""
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM attendance_logs ORDER BY timestamp DESC")
        rows = cur.fetchall()
        
        data = []
        if rows:
            columns = [col[0] for col in cur.description]
            for row in rows:
                data.append(dict(zip(columns, row)))
                
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

# Table initialization has been moved to explicit entrypoints (main.py, validate_deployment.py)
# to prevent side-effects at import time.
