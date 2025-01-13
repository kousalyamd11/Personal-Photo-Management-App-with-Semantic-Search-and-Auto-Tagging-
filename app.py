from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import os
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util
import json
import numpy as np
import sqlite3
import pickle
from PIL.ExifTags import TAGS
import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import jwt
from datetime import datetime, timedelta
import secrets
import random
import string
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
load_dotenv(find_dotenv())

MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB limit

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Add these configurations
DB_NAME = 'image_metadata.db'
model = SentenceTransformer('all-MiniLM-L6-v2')
secret_key = secrets.token_hex(32)  # 32 bytes = 64 characters
print(secret_key)
app.config['SECRET_KEY'] = 'secret_key'  # Change this to a secure secret key
JWT_EXPIRATION_DELTA = timedelta(days=7)
app.config['SMTP_SERVER'] = 'smtp.gmail.com'  # Replace with your SMTP server
app.config['SMTP_PORT'] = 587
app.config['SMTP_USERNAME'] = 'photomanagement80@gmail.com'  # Replace with your email
app.config['SMTP_PASSWORD'] = 'umss bznz pjkj bqsd'  # Replace with your email password
RESET_CODES = {}  # Store reset codes temporarily (consider using Redis in production)

# Define the token_required decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            token = token.split(' ')[1]  # Remove 'Bearer ' prefix
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user_id = data['user_id']
        except:
            return jsonify({'error': 'Token is invalid'}), 401
        
        return f(current_user_id, *args, **kwargs)
    
    return decorated

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize both pipelines
pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
tag_pipe = pipeline("image-classification", model="google/vit-base-patch16-224", top_k=20)

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Modify images table to include user_id
    c.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            caption TEXT NOT NULL,
            tags BLOB NOT NULL,
            embedding BLOB NOT NULL,
            created_date TEXT,
            file_format TEXT,
            file_size INTEGER,
            location TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()



def save_image_metadata(user_id, filename, filepath, caption, tags, embedding, metadata):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO images (
            user_id, filename, filepath, caption, tags, embedding,
            created_date, file_format, file_size, location
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        filename,
        filepath,
        caption,
        pickle.dumps(tags),
        pickle.dumps(embedding),
        metadata['created_date'],
        metadata['file_format'],
        metadata['file_size'],
        metadata['location']
    ))
    conn.commit()
    conn.close()

def get_user_images(user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        SELECT filename, filepath, caption, tags, embedding,
               created_date, file_format, file_size, location 
        FROM images
        WHERE user_id = ?
    ''', (user_id,))
    rows = c.fetchall()
    
    results = {
        'images': [],
        'embeddings': []
    }
    
    for row in rows:
        results['images'].append({
            'filename': row[0],
            'path': row[1],
            'caption': row[2],
            'tags': pickle.loads(row[3]),
            'metadata': {
                'created_date': row[5],
                'file_format': row[6],
                'file_size': row[7],
                'location': row[8]
            }
        })
        results['embeddings'].append(pickle.loads(row[4]))
    
    conn.close()
    return results

def extract_metadata(image_path):
    try:
        img = Image.open(image_path)
        
        # Get file stats
        file_stats = os.stat(image_path)
        created_date = datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        file_size = file_stats.st_size
        file_format = img.format
        
        # Initialize location as None (you might want to add GPS extraction if available)
        location = None
        
        # Try to extract GPS info if available
        exif = img._getexif()
        if exif:
            for tag_id in exif:
                tag = TAGS.get(tag_id, tag_id)
                data = exif.get(tag_id)
                if tag == 'GPSInfo':
                    # This is a simplified version. You might want to add proper GPS coordinate parsing
                    location = str(data)
        
        return {
            'created_date': created_date,
            'file_format': file_format,
            'file_size': file_size,
            'location': location
        }
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        return {
            'created_date': None,
            'file_format': None,
            'file_size': None,
            'location': None
        }

def image2text(image_path, user_id):
    try:
        print(f"Processing image: {image_path}")
        
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
        
        # Get image caption
        caption_result = pipe(img)
        
        # Get image tags with improved confidence handling
        tag_results = tag_pipe(img)
        
        # Improved tag filtering and processing
        tags = []
        seen_labels = set()
        
        for result in tag_results:
            if result["score"] > 0.15 and result["label"].lower() not in seen_labels:
                label = result["label"].replace("_", " ").title()
                seen_labels.add(label.lower())
                
                tags.append({
                    "label": label,
                    "confidence": round(result["score"], 3)
                })
                
                if len(tags) >= 10:
                    break
        
        tags = sorted(tags, key=lambda x: x["confidence"], reverse=True)
        
        img.close()
        
        # Extract metadata
        metadata = extract_metadata(image_path)

          # Create embedding for the caption
        caption_embedding = model.encode(caption_result[0]["generated_text"]).tolist()
        
        result = {
            "caption": caption_result[0]["generated_text"],
            "tags": tags,
            "filename": os.path.basename(image_path),
            "metadata": metadata,
            "embedding": caption_embedding 
        }
        
        print(f"Successfully processed image with result: {result}")
        
       
        
        # Store in SQLite database
        save_image_metadata(
            user_id=user_id,
            filename=os.path.basename(image_path),
            filepath=image_path,
            caption=result["caption"],
            tags=tags,
            embedding=caption_embedding,
            metadata=metadata
        )
        
        return result
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

@app.route('/upload', methods=['POST'])
@token_required
def upload_file(current_user_id):
    if 'file' not in request.files:
        return jsonify({'error': 'No file is selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        allowed_extensions = ', '.join(ALLOWED_EXTENSIONS)
        return jsonify({
            'error': f'Invalid file type. Only {allowed_extensions} files are allowed.'
        }), 400
    
    try:
        original_filename = secure_filename(file.filename)
        # Generate a unique filename
        unique_suffix = datetime.now().strftime('%Y%m%d%H%M%S') + "_" + str(uuid.uuid4().hex)
        filename = f"{unique_suffix}_{original_filename}"
        # Create user-specific upload directory
        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user_id))
        if not os.path.exists(user_upload_dir):
            os.makedirs(user_upload_dir)
            
        filepath = os.path.join(user_upload_dir, filename)
        file.save(filepath)
        
        result = image2text(filepath, current_user_id)
        
        return jsonify({'result': result})
    except Exception as e:
        # Clean up file if there's an error
        if os.path.exists(filepath):
            os.remove(filepath)
        print(f"Error in upload_file: {str(e)}")
        return jsonify({'error': f"Processing failed: {str(e)}"}), 500

# Add new route for semantic search
@app.route('/search', methods=['POST'])
@token_required
def semantic_search(current_user_id):
    try:
        query = request.json.get('query', '')
        date_filter = request.json.get('date', '')  # New date filter parameter

        if not query and not date_filter:
            return jsonify({'error': 'No query or date filter provided'}), 400
        
        # Fetch all images for the current user
        metadata = get_user_images(current_user_id)
        if not metadata['images']:
            return jsonify({'results': []})

        filtered_images = metadata['images']
        filtered_embeddings = metadata['embeddings']

        # Apply date filter if provided
        if date_filter:
            filtered_images = [
                image for image in metadata['images']
                if image['metadata']['created_date'][:10] == date_filter
            ]
            # Filter embeddings accordingly
            filtered_embeddings = [
                metadata['embeddings'][idx]
                for idx, image in enumerate(metadata['images'])
                if image['metadata']['created_date'][:10] == date_filter
            ]

        if not query and date_filter:
            # If only the date filter is applied, return filtered images
            for image in filtered_images:
                image['metadata']['user_id'] = current_user_id  # Add user_id to metadata
            return jsonify({'results': [
                {
                    'filename': image['filename'],
                    'caption': image['caption'],
                    'tags': image['tags'],
                    'metadata': image['metadata'],
                    'similarity': float(1.0)
                }
                for image in filtered_images
            ]})

        if query:
            # Perform semantic search on the filtered images
            query_embedding = model.encode(query, convert_to_numpy=True).astype(np.float32)
            stored_embeddings = np.array(filtered_embeddings, dtype=np.float32)

            similarities = util.cos_sim(query_embedding, stored_embeddings)[0]
            similarities = np.array(similarities)

            # Find top matches
            top_k = min(10, len(filtered_images))
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0.2:
                    image_info = filtered_images[idx]
                    image_info['metadata']['user_id'] = current_user_id
                    results.append({
                        'filename': image_info['filename'],
                        'caption': image_info['caption'],
                        'tags': image_info['tags'],
                        'metadata': image_info['metadata'],
                        'similarity': float(similarities[idx])
                    })

            return jsonify({'results': results})

        return jsonify({'error': 'No query or valid date filter provided'}), 400
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<int:user_id>/<path:filename>')
def serve_image(user_id, filename):
    user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
    file_path = os.path.join(user_upload_dir, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(user_upload_dir, filename)



# Add this new route to handle metadata updates
@app.route('/update-metadata', methods=['POST'])
@token_required
def update_metadata(current_user_id):
    try:
        data = request.json
        filename = data.get('filename')
        caption = data.get('caption')
        tags = data.get('tags')

        if not all([filename, caption, tags]):
            return jsonify({'error': 'Missing to add tags or caption'}), 400

        # Create embedding for the new caption
        caption_embedding = model.encode(caption).tolist()

        # Update the database
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''
            UPDATE images 
            SET caption = ?, tags = ?, embedding = ?
            WHERE filename = ? AND user_id = ?
        ''', (
            caption,
            pickle.dumps(tags),
            pickle.dumps(caption_embedding),
            filename,
            current_user_id
        ))
        conn.commit()
        conn.close()

        return jsonify({'success': True})
    except Exception as e:
        print(f"Error updating metadata: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add new routes for authentication
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    
    try:
        username = data['username']
        password = data['password']
        email = data['email']
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # Check if username or email already exists
        c.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        if c.fetchone():
            return jsonify({'error': 'Username or email already exists'}), 400
        
        # Create new user
        password_hash = generate_password_hash(password)
        c.execute('''
            INSERT INTO users (username, password_hash, email)
            VALUES (?, ?, ?)
        ''', (username, password_hash, email))
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'User registered successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    
    try:
        username = data['username']
        password = data['password']
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        
        if not user or not check_password_hash(user[1], password):
            return jsonify({'error': 'Invalid username or password'}), 401
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': user[0],
            'exp': datetime.utcnow() + JWT_EXPIRATION_DELTA
        }, app.config['SECRET_KEY'])
        
        return jsonify({'token': token})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def send_reset_email(email, reset_code):
    try:
        msg = MIMEMultipart()
        msg['From'] = app.config['SMTP_USERNAME']
        msg['To'] = email
        msg['Subject'] = 'Password Reset Code'
        
        body = f'Your password reset code is: {reset_code}\nThis code will expire in 15 minutes.'
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(app.config['SMTP_SERVER'], app.config['SMTP_PORT'])
        server.starttls()
        server.login(app.config['SMTP_USERNAME'], app.config['SMTP_PASSWORD'])
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

@app.route('/send-reset-code', methods=['POST'])
def send_reset_code():
    try:
        email = request.json.get('email')
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        # Check if email exists in database
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()
        
        if not user:
            return jsonify({'error': 'Email not found'}), 404
        
        # Generate reset code
        reset_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        RESET_CODES[email] = {
            'code': reset_code,
            'expires': datetime.utcnow() + timedelta(minutes=15)
        }
        
        # Send reset code via email
        if send_reset_email(email, reset_code):
            return jsonify({'message': 'Reset code sent successfully'})
        else:
            return jsonify({'error': 'Failed to send reset code'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset-password', methods=['POST'])
def reset_password():
    try:
        email = request.json.get('email')
        reset_code = request.json.get('reset_code')
        new_password = request.json.get('new_password')
        
        if not all([email, reset_code, new_password]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Verify reset code
        stored_data = RESET_CODES.get(email)
        if not stored_data or stored_data['code'] != reset_code:
            return jsonify({'error': 'Invalid reset code'}), 400
        
        if datetime.utcnow() > stored_data['expires']:
            del RESET_CODES[email]
            return jsonify({'error': 'Reset code has expired'}), 400
        
        # Update password in database
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''
            UPDATE users 
            SET password_hash = ? 
            WHERE email = ?
        ''', (generate_password_hash(new_password), email))
        conn.commit()
        conn.close()
        
        # Clean up reset code
        del RESET_CODES[email]
        
        return jsonify({'message': 'Password reset successful'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True) 