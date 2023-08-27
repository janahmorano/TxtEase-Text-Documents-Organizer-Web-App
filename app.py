#code by: JANAH PATRICIA MORANO
from flask import Flask, render_template, request
from flask import make_response
from flask import Flask, render_template, request, session, jsonify
from flask_session import Session 
import os
import numpy as np
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request
from sklearn.cluster import KMeans
import pdfplumber
import docx2txt
import shutil
import zipfile
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
import spacy
import nltk
nltk.download('stopwords')


# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Configure session options
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = 'filesystem'  
Session(app) 


UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'UPLOADED FILES')
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        uploaded_files = request.files.getlist('file')

        
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        uploaded_file_paths = []

        for file in uploaded_files:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            uploaded_file_paths.append(file_path)

        session['uploaded_file_paths'] = uploaded_file_paths

        return jsonify({"message": "Files uploaded and ready for clustering."})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        uploaded_file_paths = session.get('uploaded_file_paths')
        if not uploaded_file_paths:
            return jsonify({"error": "No uploaded files found."})

        uploaded_document_paths = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))

        # Preprocess functions
        def process_batch(batch_paths):
            batch_texts = []
            for path in batch_paths:
                if path.endswith('.pdf'):
                    with pdfplumber.open(path) as pdf:
                        text = ' '.join([page.extract_text() for page in pdf.pages])
                elif path.endswith('.docx'):
                    text = docx2txt.process(path)
                else:
                    with open(path, 'r', encoding='utf-8') as file:
                        text = file.read()
                batch_texts.append(text)
            return batch_texts
        

        def preprocess_text(text):
            lemmatized_text = nlp(text)
            tokens = [token.lemma_ for token in lemmatized_text if token.text.lower() not in set(stopwords.words('english'))]
            return ' '.join(tokens)
        

        batch_size = 50  # Adjust as needed
        num_documents = len(uploaded_document_paths) 
        num_batches = (num_documents + batch_size - 1) // batch_size

        preprocessed_documents = []

        for batch_index in range(num_batches):
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, num_documents)
            batch_paths = uploaded_document_paths[start_index:end_index]
            batch_texts = process_batch(batch_paths)
            batch_texts = [preprocess_text(text) for text in batch_texts]  
            preprocessed_documents.extend(batch_texts)

        # Convert preprocessed documents to TF-IDF vectors and determine the optimal number of clusters
        try:
            tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', 
                                                max_df=0.85, min_df=2)  # Adjust hyperparameters
            document_embeddings = tfidf_vectorizer.fit_transform(preprocessed_documents)

            # Determine the optimal number of clusters using silhouette score
            silhouette_scores = []
            for num_clusters in range(2, 21):  
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(document_embeddings)
                silhouette_scores.append(silhouette_score(document_embeddings, cluster_labels))

            optimal_clusters = np.argmax(silhouette_scores) + 2

            # Apply k-means clustering with the chosen number of clusters
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            cluster_assignments = kmeans.fit_predict(document_embeddings)

            # Get the most common terms in each cluster
            cluster_terms = {}
            for cluster_id in range(optimal_clusters):
                cluster_indices = np.where(cluster_assignments == cluster_id)[0]
                cluster_text = [preprocessed_documents[i] for i in cluster_indices]
                cluster_terms[cluster_id] = ' '.join(cluster_text)

            # Convert preprocessed cluster texts to TF-IDF vectors
            cluster_embeddings = tfidf_vectorizer.transform(cluster_terms.values())

            # Find the most common terms in each cluster
            most_common_terms = []
            feature_names = tfidf_vectorizer.get_feature_names_out()
            for cluster_id, cluster_embedding in enumerate(cluster_embeddings):
                cluster_embedding = cluster_embedding.toarray()[0]
                term_indices = cluster_embedding.argsort()[::-1][:5]  # Get top 5 terms
                terms = [feature_names[i] for i in term_indices]
                most_common_terms.append((cluster_id, ' '.join(terms)))

            # Create a base folder to store the clustering results
            clustering_output_folder = os.path.join(UPLOAD_FOLDER, 'clustering_results')
            os.makedirs(clustering_output_folder, exist_ok=True)

            # Create folders and organize documents with topic labels
            for cluster_id, common_terms in most_common_terms:
                topic_folder = os.path.join(clustering_output_folder, f'{common_terms.upper()}')
                os.makedirs(topic_folder, exist_ok=True)

            for doc_idx, cluster_id in enumerate(cluster_assignments):
                source_path = uploaded_document_paths[doc_idx]  
                file_name = os.path.basename(source_path)
                target_folder = os.path.join(clustering_output_folder, f'{most_common_terms[cluster_id][1].upper()}')
                target_path = os.path.join(target_folder, file_name)
                shutil.copy(source_path, target_path)  

            print("Documents organized into clusters!")

            zip_file_path = os.path.join(UPLOAD_FOLDER, 'organized_folders.zip')

            # Create the zip file 
            with zipfile.ZipFile(zip_file_path, 'w') as zipf:
                for root, dirs, files in os.walk(clustering_output_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, clustering_output_folder)
                        zipf.write(file_path, arcname=arcname)
 
            session['zip_file_path'] = zip_file_path 

            response = {
                "success": True,
                "download_link": zip_file_path
            }
            return jsonify(response)
            
                    
            return jsonify({"message": "Clustering complete."})
        except Exception as e:
            return jsonify({"error": str(e)})
        
        return jsonify({"message": "Clustering complete."})
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/download', methods=['GET'])
def download():
    try:
        zip_file_path = session.get('zip_file_path')
        if not zip_file_path:
            return jsonify({"error": "Zip file not found."})

        zip_file_name = 'organized_folders.zip'

        with open(zip_file_path, 'rb') as zip_file:
            zip_data = zip_file.read()

        shutil.rmtree(UPLOAD_FOLDER)

        response = make_response(zip_data)
        response.headers['Content-Type'] = 'application/zip'
        response.headers['Content-Disposition'] = f'attachment; filename={zip_file_name}'
        
        return response

    except Exception as e:
        return jsonify({"error": str(e)})

    
if __name__ == '__main__':
    app.run(debug=True)
