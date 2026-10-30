
import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "chest_xray.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as connection:
        # Create the table if it does not exist.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                patient_name TEXT,
                age INTEGER,
                gender TEXT,
                image_name TEXT NOT NULL,
                reference_fit TEXT,
                predicted_fit TEXT,
                reference_medical_label TEXT,
                predicted_medical_label TEXT,
                measurements TEXT,
                analysis_date TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Update existing databases without deleting records.
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(analyses)"
            ).fetchall()
        }

        if "patient_name" not in columns:
            connection.execute(
                "ALTER TABLE analyses ADD COLUMN patient_name TEXT"
            )

        # Recover names from older records when possible.
        rows = connection.execute("""
            SELECT id, measurements
            FROM analyses
            WHERE patient_name IS NULL OR patient_name = ''
        """).fetchall()

        for record_id, measurements in rows:
            try:
                data = json.loads(measurements or "{}")
                name = data.get("Patient Name")

                if name:
                    connection.execute("""
                        UPDATE analyses
                        SET patient_name = ?
                        WHERE id = ?
                    """, (name, record_id))

            except (ValueError, TypeError):
                pass


def save_analysis(record):
    with get_connection() as connection:
        connection.execute("""
            INSERT INTO analyses (
                case_id,
                patient_name,
                age,
                gender,
                image_name,
                reference_fit,
                predicted_fit,
                reference_medical_label,
                predicted_medical_label,
                measurements
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["case_id"],
            record["patient_name"],
            record["age"],
            record["gender"],
            record["image_name"],
            record["reference_fit"],
            record["predicted_fit"],
            record["reference_medical_label"],
            record["predicted_medical_label"],
            record["measurements"],
        ))


def get_analyses():
    with get_connection() as connection:
        return connection.execute("""
            SELECT
                id,
                case_id,
                patient_name,
                age,
                gender,
                image_name,
                reference_fit,
                predicted_fit,
                reference_medical_label,
                predicted_medical_label,
                measurements,
                analysis_date
            FROM analyses
            ORDER BY id DESC
        """).fetchall()