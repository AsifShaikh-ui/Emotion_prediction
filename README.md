<img width="301" height="370" alt="Screenshot 2025-10-26 124040" src="https://github.com/user-attachments/assets/4305760d-2cc2-44ed-b5cc-89e04b8086fc" />

🎭 Emotion Detection Using Machine Learning
This project is a Machine Learning
based Emotion Prediction Model that identifies human emotions from text data such as joy, sadness, anger, fear, love, and surprise.
It helps understand how people express emotions through words using Natural Language Processing (NLP).

📘 Objective
The main goal of this project is to predict the emotion behind a given text using machine learning.
It uses text preprocessing, feature extraction (TF-IDF), and a classification model to detect emotions with high accuracy.

⚙️ Methodology
Dataset Preparation:
The dataset train.txt contains two columns: Text and Emotions.
The data was analyzed and balanced to remove class imbalance problems.

Data Preprocessing:
Converted all text to lowercase.
Removed unnecessary characters and spaces.
Encoded emotions into numeric form using LabelEncoder.

Feature Extraction:
Used TF-IDF Vectorizer to convert text into numerical feature vectors.

Model Training:
Applied Logistic Regression classifier.
Trained the model on balanced data for better emotion prediction.

Model Evaluation:
Achieved around 88% accuracy.
Evaluated using metrics like Precision, Recall, and F1-score.

Model Deployment:
The model is deployed using Streamlit, a Python web framework for interactive apps.
It allows users to enter any text and view the predicted emotion instantly.

🧠 Algorithms & Tools Used
Category	Tools / Libraries
Programming Language	Python
Data Handling	Pandas, NumPy
Model Building	Scikit-learn (Logistic Regression)
Text Processing	TF-IDF Vectorizer
Web App	Streamlit
Model Saving	Joblib
