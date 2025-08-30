\# NLP Project: News Classification, Topic Modeling \& Conversational AI



This repository contains the code and report for an academic Natural Language Processing (NLP) project.

The project demonstrates a full NLP pipeline — from data collection and preprocessing to feature engineering, model building, and deployment — using both traditional ML and deep learning approaches. We also integrated Retrieval-Augmented Generation (RAG) into a conversational AI system with a Streamlit interface..



Important Note:

The app.py file is not provided in this repository. It is an academic Python script that activates a Streamlit application for the RAG-based conversational AI, which was developed and deployed on Hugging Face Spaces..



\## Dataset



HuffPost News Category Dataset from Kaggle: https://www.kaggle.com/datasets/rmisra/news-category-dataset.

The dataset is not included in this repository due to size. Please download it from Kaggle using the link.

For our tasks, we focused on the Top 5 most frequent categories: Politics, Wellness, Entertainment, Travel, Style \& Beauty.



\## Pipeline



1\. Data Preprocessing



-Removed duplicates, cleaned URLs, usernames, emails, emojis, and special characters



-Lowercasing, stopword removal (with exceptions like "no", "not", "never")



-Lemmatization and tokenization



2\. Feature Engineering \& Visualization



-TF-IDF (unigrams \& bigrams)



-Word2Vec embeddings (trained with Gensim)



-Sentence Transformers (all-MiniLM-L6-v2)



-Visualizations: PCA, t-SNE, cosine similarity heatmaps, word clouds



3\. Model Building



\# Unsupervised Learning:



-Topic modeling with LDA and NMF



-Evaluation using Coherence Score, Perplexity, pyLDAvis visualizations



\# Supervised Learning:



-Traditional ML: Logistic Regression, Linear SVM



-Deep Learning: DistilBERT fine-tuning (via Hugging Face Transformers)



-Explainable AI: LIME applied to BERT predictions



4\. Conversational AI (RAG)



-Built a Retrieval-Augmented Generation system



-Sentence embeddings for document retrieval



-Response generation with Mistral-7B-Instruct



-Integrated into an interactive Streamlit UI, deployed on Hugging Face Spaces



\## Deployment



-RAG chatbot deployed on Hugging Face Spaces with interactive UI elements:



-Query input \& response display



-Model selection



-Persona customization (Concise, Friendly, Technical, Storyteller)



-Clear chat history, debug panel, retrieval parameter controls



\## Key Results



-Topic Modeling: NMF outperformed LDA in interpretability (Coherence ≈ 0.67).



-Classification: DistilBERT achieved the best accuracy among all models.



-Explainability: LIME provided meaningful insights into model predictions.



-RAG System: Produced relevant and context-aware answers with a smooth UI.



\## Installation



Clone the repository:



git clone https://github.com/labropouloun/NLP-News-Classification-RAG-Chatbot.git

cd NLP-News-Classification-RAG-Chatbot

pip install -r requirements.txt



\## Note: 

For training and running deep learning models (e.g., DistilBERT, RAG system), it is highly recommended to use a GPU for faster performance.



\## Libraries Used



-pandas

-numpy

-nltk

-scikit-learn

-gensim

-plotly

-matplotlib

-seaborn

-wordcloud

-pyLDAvis

-sentence-transformers

-transformers

-datasets

-lime

-torch



\## Author

Nancy Labropoulou

GitHub: labropouloun

