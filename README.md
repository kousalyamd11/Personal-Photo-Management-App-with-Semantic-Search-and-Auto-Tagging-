Personal Photo Management System
Overview
The Personal Photo Management System is a web-based application designed to help users efficiently manage, search, and annotate their photo collections. The system leverages AI capabilities to generate captions and tags for uploaded images, enabling semantic search and metadata editing. With a user-friendly interface and secure authentication, this application provides a seamless photo organization experience.

Features
AI-Powered Image Analysis:

Automatically generate captions for uploaded images.
Assign tags to photos using advanced image classification models.
Search Functionality:

Perform semantic searches to find relevant photos based on natural language queries.
Metadata Management:

Edit captions and tags for better organization.
View detailed metadata, including file format, size, and creation date.
User Authentication:

Secure login and registration system.
Password reset functionality with email-based verification.
Responsive Design:

A clean, intuitive interface designed using Tailwind CSS.
Mobile and desktop-friendly layouts.
Technologies Used
Frontend:

HTML, CSS (Tailwind), and JavaScript for a responsive and dynamic user interface.
Backend:

Flask for server-side application logic.
SQLite for efficient and lightweight data storage.
AI Models:

Salesforce/blip-image-captioning-large for image captioning.
google/vit-base-patch16-224 for image classification.
Security:

JWT-based authentication for secure access control.
Password hashing with werkzeug.security.
Email Integration:

SMTP for password reset email functionality.
Setup and Installation

Clone the Repository:
git clone <repository_url>
cd personal-photo-management-system
Install Dependencies: Ensure Python is installed. Install required packages:

pip install -r requirements.txt
Configure Environment Variables: Create a .env file with the following variables:


SMTP_SERVER=<your_smtp_server>
SMTP_PORT=<smtp_port>
SMTP_USERNAME=<your_email>
SMTP_PASSWORD=<your_email_password>
Initialize the Database: Run the following to create necessary tables:


python app.py
Start the Application:

bash
Copy code
flask run
Access the Application: Open your browser and navigate to http://localhost:5000.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your proposed changes.
