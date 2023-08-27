## TxtEase-Text-Documents-Organizer-Web-App
![txtEase](https://github.com/janahmorano/TxtEase-Text-Documents-Organizer-Web-App/assets/142562162/17430ca8-d9ac-4998-a7df-66698730acd1)

This Flask-based web application streamlines the organization of uploaded text documents through the power of K-Means clustering. Users can effortlessly upload DOCX, TXT, and PDF files, allowing the application's K-Means algorithm to intelligently group similar documents into clusters. The resulting clusters are available for easy download in a convenient ZIP file format.

## Features

- Allows Upload of any of your DOCX, TXT, and PDF files for clustering.
- Utilizes the TF-IDF algorithm for NLP-based text analysis.
- Applies K-Means clustering to group similar documents effectively.
- Organizes clustered documents and provides a ZIP file for download.

## Getting Started

Follow these steps to get the Text Documents Web App up and running on your local machine:

1. Clone this repository to your local machine.
2. Navigate to the project directory: `cd Text-Documents-Web-App`.
3. Create a virtual environment: `python3 -m venv venv`.
4. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS and Linux: `source venv/bin/activate`
5. Install the required dependencies: `pip install -r requirements.txt`.
6. Run the Flask app: `python app.py`.
7. Access the app in your web browser at `http://localhost:5000`.

## Usage

1. Open your web browser and go to `http://localhost:5000`.
2. Upload DOCX, TXT, or PDF files using the provided dropzone.
3. Click the "Cluster Documents" button to start the clustering process.
4. Once clustering is complete, you'll receive a link to download the organized documents in a ZIP file.

## Contributing

Contributions are welcome! If you find a bug or want to enhance the app, feel free to submit an issue or create a pull request.

------
Created by Janah Patricia Morano
