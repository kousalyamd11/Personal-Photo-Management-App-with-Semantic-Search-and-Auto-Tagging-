# Personal Photo Management System

## Description
The **Personal Photo Management System** is a comprehensive application designed to help users manage their photo collections. It leverages AI capabilities for image captioning, tagging, and semantic search. This project includes user authentication, secure file handling, and image metadata extraction.

---

## Features

### User Authentication
- **Registration**: Secure user account creation with hashed passwords.
- **Login**: Authenticate users using JSON Web Tokens (JWT).
- **Password Recovery**: Email-based password reset functionality.

### Photo Management
- **Image Uploads**: Drag-and-drop or file selection to upload images.
- **AI Captioning**: Automatically generates a descriptive caption for uploaded images.
- **Tagging**: Adds AI-generated tags to images for better categorization.
- **Search**: Perform semantic searches on image captions and metadata.
- **Metadata Updates**: Edit captions and tags directly in the UI.

### Secure and Scalable
- **SQLite Database**: Stores user data and image metadata.
- **Flask Backend**: Manages API routes and processes images.
- **CORS Support**: Enables safe cross-origin requests.

---

## Tech Stack

### Backend
- **Python** (Flask, SQLAlchemy, JWT, SentenceTransformer, Transformers, PIL)

### Frontend
- **HTML**
- **CSS** (TailwindCSS)
- **JavaScript**

### Database
- **SQLite**: For user and image metadata storage.

### AI Models
- **Image Captioning**: Salesforce's BLIP Image Captioning.
- **Generating tags**:GroqCloud

---

## Installation

### Prerequisites
1. Install [Python 3.8+](https://www.python.org/).
2. Clone the repository:

   ```bash
   git clone https://github.com/kousalyamd11/Personal-Photo-Management-App-with-Semantic-Search-and-Auto-Tagging-.git
   cd Personal-Photo-Management-App-with-Semantic-Search-and-Auto-Tagging-
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up in app.py file with the following:
   ```env
   API_KEY = "groq API key"
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=<your-email>
   SMTP_PASSWORD=<your-email-password>
   ```

7. Initialize the database:

   ```bash
   python app.py
   ```
---

## Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open the application in your browser at `http://localhost:5000`.

---

## Usage

1. **Register/Login**: Create an account and log in to start using the app.
2. **Upload Images**: Drag and drop or choose images to upload.
3. **View Results**: Captions and tags are generated automatically.
4. **Edit Metadata**: Update captions and tags as needed.
5. **Search**: Use the semantic search bar to find images by description.

---

## API Endpoints

### Authentication
- `POST /register`: Register a new user.
- `POST /login`: Log in and obtain a JWT.
- `POST /send-reset-code`: Request a password reset code.
- `POST /reset-password`: Reset your password.

### Image Management
- `POST /upload`: Upload an image for processing.
- `POST /search`: Perform a semantic search for images.
- `POST /update-metadata`: Update captions or tags for an image.
- `GET /uploads/<user_id>/<filename>`: Fetch a specific image.

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss your ideas.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

Kousalya


---

## Acknowledgments

- **Salesforce** for BLIP Image Captioning.
- **Google** for ViT Image Classification.
- **Flask** for providing a simple yet powerful framework.
