# 📚 EPFL_Apple: Book Recommendation System
![EPFL_Apple](Images_for_the_report/overview.png)

## Project Overview

This project investigates and implements multiple recommendation system techniques for a large-scale book dataset. The primary objectives are to analyze user-book interactions, enrich and clean book metadata, and develop both collaborative and content-based recommendation models to improve book suggestions for users.

## Data Sources and Setup

### Datasets
- **interactions_train.csv**: User-book interaction logs (user id `u`, book id `i`, timestamp `t`).
- **items.csv**: Book metadata (title, author, ISBN, publisher, subjects, etc.).
- **books_complete.csv**: Enriched and cleaned book metadata (after merging with external sources).

### Libraries used

- **Data handling**: pandas, numpy
- **Visualization**:: matplotlib
- **NLP and text vectorization**: nltk, TfidfVectorizer
- **Dimensionality reduction**: TruncatedSVD
- **Similarity calculations**: cosine_similarity

## Exploratory Data Analysis (EDA)

The notebook EPFL_Apple_EDA begins with:
- Loading and previewing user interactions and book metadata.
- Counting elements and missing values in the dataset. 
👉 So-What?: This could affect content-based recommendations that rely on textual features like author similarity.
| Feature        | Missing Values |
|---------------|:--------------:|
| Title          | 0              |
| Author         | 2,653          |
| ISBN Valid     | 723            |
| Publisher      | 25             |
| Subjects       | 2,223          |
| i              | 0              |

- Analyzing user activity: number of books read per user, unique users, and interaction patterns over time.
👉 So-What?: There’s high variability in user behavior (std. dev = 16.4). Some users are extremely active, while others barely engage.
- Calculating statistics such as mean, standard deviation, and coefficient of variation for books read by each user.
- Visualizing distributions (e.g., standard deviation of book IDs, coefficient of variation, delta between last and max book IDs).
- Aggregating user activity over time and identifying highly active users.

### Data Enrichment and Cleaning

- Extracted and cleaned ISBNs from book metadata.
- Queried external APIs (e.g., ISBNdb) to enrich book information.
- Normalized and merged API results with the original dataset.
- Cleaned author and publisher fields by removing noise, standardizing names, and handling missing values.
- Combined and cleaned relevant features for use in content-based models.

## Collaborative Filtering Approaches

### Simple Collaborative Filtering

- Constructed a binary user-item interaction matrix.
- Calculated cosine similarity between users.
- Generated top-N recommendations for each user based on similar users' reading history.
- Exported recommendations to CSV for evaluation.

### Improved Collaborative Filtering with SVD

- Applied TruncatedSVD for dimensionality reduction on the interaction matrix.
- Reconstructed approximate interaction scores and generated recommendations based on latent factors.
- Compared results with the simple approach.

## Content-Based Recommendations (TFIDF)

- Created a `combined_features` column from book metadata (title, author, ISBN, publisher, pages).
- Used TFIDF vectorization (with French stop words) to represent books.
- For each user, computed similarity between books they read and all other books.
- Generated top-N recommendations per user, with optional naive filtering based on book ID ranges.
- Exported recommendations for further evaluation.

## Results Summary

- Collaborative filtering (user-based and SVD) and content-based (TFIDF) approaches were implemented and compared.
- The best-performing models achieved Precision@10 scores around 0.15.
- Data cleaning and enrichment significantly improved the quality of recommendations.
- Exported recommendation files are ready for submission or further analysis.

| Model               | Precision@10 | Recall@10 | Kaggle |
|---------------------|--------------|-----------|--------|
| User-User CF        | 0.056        | 0.29      |0.1515  |
| Item-Item CF        | 0.059        | 0.28      |0.1508  |
| User-User CF (rank) | 0.06         | 0.29      |0.1642  |
| TF-IDF (raw, naive) | ?????        | ????      |0.1560  |

## Best Model & Interpretation



## Additional Experiments

- Attempted to generate BERT embeddings for book titles and subjects (not fully implemented).
- Explored further collaborative filtering techniques and evaluation strategies.
- Used `GridSearchCV` to tune SVD hyperparameters.
- Split data into training and test sets.
- Evaluated models using Precision@10 metric.

---

## 🖥️ Interface (A Librarian-Facing App)

To bridge insights with real-life usage, we designed and developed an interactive web application using Streamlit. This Book Recommendations App provides users with personalized book recommendations and allows them to explore books based on various search criteria.

You can see our Book Recommendations App here: https://epflapple.streamlit.app/

🌟 Key Features:
- Personalized recommendations: suggests books based on user history and similarity scores.
- Search & discovery tools: allows filtering books by user, keywords, authors, or title.
- Visual interface: displays book covers, titles, and metadata with clean UI components.
- Interactive modals: clicking a book opens a detailed popup with descriptions, authorship, and availability.

---

## Dependencies

- Python 3.x
- pandas, numpy, matplotlib, scikit-learn, nltk, requests

## Acknowledgements

- Data provided by EPFL and external book APIs.
- Libraries: scikit-learn, NLTK, matplotlib, pandas, numpy, streamlit.
- In the process of the project the following AI sources were used: ChatGPT, Copilot, Gemini.

---
For detailed code and outputs, see the notebooks: `EPFL_Apple_EDA.ipynb`, `EPFL_Apple_successful models.ipynb`.