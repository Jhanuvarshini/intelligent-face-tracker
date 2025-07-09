from flask import Flask, render_template
import sqlite3

# ✅ Important: Set correct template folder path
app = Flask(__name__, template_folder="../templates")

def fetch_data():
    conn = sqlite3.connect('logs/face_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, event_type, image_path FROM faces ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

@app.route('/')
def index():
    rows = fetch_data()
    unique_ids = len(set([row[0] for row in rows]))
    return render_template("index.html", rows=rows, total=unique_ids)

if __name__ == "__main__":
    app.run(debug=True)
