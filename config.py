from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MANUALS_DIR = BASE_DIR / "Data" / "Maintencie"
PROCESSED_DIR = BASE_DIR / "data_processed"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

MANUALS = {
    "sc6002xl": {
        "manufacturer": "Siemens",
        "device": "SC 6002XL Patient Monitor",
        "file": MANUALS_DIR / "Copy of Sc 6002Xl Patient Monitor.pdf",
    },

    "servo_ventilator": {
        "manufacturer": "Siemens",
        "device": "Servo Ventilator System",
        "file": MANUALS_DIR / "Copy of Siemens Servo 900 Ventilator.pdf",
    },
}