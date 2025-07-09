import sqlite3

conn = sqlite3.connect("logs/face_data.db")
cursor = conn.cursor()

cursor.execute("SELECT DISTINCT id FROM faces")
unique_ids = cursor.fetchall()
print(f"🧍 Total Unique Visitors: {len(unique_ids)}")

conn.close()
