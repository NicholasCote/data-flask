# app.py
from flask import Flask, request, jsonify, render_template
import psycopg2
import os

app = Flask(__name__)

def get_db_connection():
    conn = psycopg2.connect(
        host=os.environ.get('DB_HOST', 'postgres'),
        database=os.environ.get('DB_NAME', 'notesdb'),
        user=os.environ.get('DB_USER', 'postgres'),
        password=os.environ.get('DB_PASSWORD', 'mysecretpassword')
    )
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/notes', methods=['GET'])
def get_notes():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, content, created_at FROM notes ORDER BY created_at DESC;')
    notes = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify([{'id': note[0], 'content': note[1], 'created_at': note[2].isoformat()} for note in notes])

@app.route('/notes', methods=['POST'])
def create_note():
    content = request.json.get('content', '')
    if not content:
        return jsonify({'error': 'Content is required'}), 400
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('INSERT INTO notes (content) VALUES (%s) RETURNING id, content, created_at;', (content,))
    new_note = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    
    return jsonify({'id': new_note[0], 'content': new_note[1], 'created_at': new_note[2].isoformat()})

@app.route('/init-db')
def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Check if table exists before creating
    cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'notes');")
    table_exists = cur.fetchone()[0]
    
    if not table_exists:
        cur.execute('CREATE TABLE notes (id SERIAL PRIMARY KEY, content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);')
        conn.commit()
        message = "Database initialized successfully!"
    else:
        message = "Database already initialized."
    
    cur.close()
    conn.close()
    return jsonify({'message': message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)