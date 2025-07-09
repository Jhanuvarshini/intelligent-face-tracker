import sqlite3
import os

# Ensure the logs folder exists
os.makedirs("logs", exist_ok=True)

# Path to your database
db_path = os.path.join("logs", "face_data.db")

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create or update the faces table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id TEXT,
        timestamp TEXT,
        image_path TEXT,
        event_type TEXT
    )
''')

conn.commit()
conn.close()
print("✅ Database table with event_type created.")
