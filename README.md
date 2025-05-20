# EPFL_Apple: Book Recommendation System

## Project Overview

This project investigates and implements multiple recommendation system techniques for a large-scale book dataset. The primary objectives are to analyze user-book interactions, enrich and clean book metadata, and develop both collaborative and content-based recommendation models to improve book suggestions for users.

## Data Sources

- **interactions_train.csv**: User-book interaction logs (user id `u`, book id `i`, timestamp `t`).
- **items.csv**: Book metadata (title, author, ISBN, publisher, subjects, etc.).
- **books_complete.csv**: Enriched and cleaned book metadata (after merging with external sources).

## Exploratory Data Analysis (EDA)

The notebook begins with:
- Loading and previewing user interactions and book metadata.
- Counting elements and missing values in the dataset.
- Analyzing user activity: number of books read per user, unique users, and interaction patterns over time.
- Calculating statistics such as mean, standard deviation, and coefficient of variation for books read by each user.
- Visualizing distributions (e.g., standard deviation of book IDs, coefficient of variation, delta between last and max book IDs).
- Aggregating user activity over time and identifying highly active users.

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

## Data Enrichment and Cleaning

- Extracted and cleaned ISBNs from book metadata.
- Queried external APIs (e.g., ISBNdb) to enrich book information.
- Normalized and merged API results with the original dataset.
- Cleaned author and publisher fields by removing noise, standardizing names, and handling missing values.
- Combined and cleaned relevant features for use in content-based models.

## Additional Experiments

- Attempted to generate BERT embeddings for book titles and subjects (not fully implemented).
- Explored further collaborative filtering techniques and evaluation strategies.
- Used `GridSearchCV` to tune SVD hyperparameters.
- Split data into training and test sets.
- Evaluated models using Precision@10 metric.

## Results Summary

- Collaborative filtering (user-based and SVD) and content-based (TFIDF) approaches were implemented and compared.
- The best-performing models achieved Precision@10 scores around 0.15.
- Data cleaning and enrichment significantly improved the quality of recommendations.
- Exported recommendation files are ready for submission or further analysis.

## Usage

1. Place the required CSV files in the `kaggle_data/` directory.
2. Run the notebook `EPFL_Apple_EDA.ipynb` step by step.
3. Generated recommendation files will be saved in the project directory.

## Dependencies

- Python 3.x
- pandas, numpy, matplotlib, scikit-learn, nltk, requests

## Acknowledgements

- Data provided by EPFL and external book APIs.
- Libraries: scikit-learn, NLTK, matplotlib, pandas, numpy.

---
For detailed code and outputs, see the notebook: `EPFL_Apple_EDA.ipynb`.