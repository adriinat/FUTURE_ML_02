# AI-Powered Support Ticket Classification & Prioritization

An end-to-end Machine Learning project that automatically classifies IT support tickets into relevant categories and assigns priority levels using a combination of Natural Language Processing, Machine Learning, and rule-based logic.

## Project Overview

IT support teams receive numerous tickets every day. Manually categorizing and prioritizing these tickets can be time-consuming.

This project aims to automate that process by:
- Predicting the category of an IT support ticket using Machine Learning.
- Assigning High, Medium, or Low priority using rule-based logic.
- Supporting individual ticket classification and bulk CSV processing.
- Providing an interactive analytics dashboard.

## Features

- Single Ticket Classification
- Batch Ticket Classification through CSV upload
- Automatic category prediction
- Rule-based priority assignment
- CSV download of classified tickets
- Interactive ticket analytics dashboard
- Input validation and empty-ticket handling

## Machine Learning Approach

**Text preprocessing:** Regular expressions and text cleaning.

**Feature extraction:** TF-IDF Vectorization with unigram and bigram features.

**Models evaluated:**
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (Linear SVM)

**Final selected model:** Linear SVM

The model was selected based on its performance on the held-out test dataset.

## Model Performance

| Model | Accuracy | Macro F1 Score |
|---|---:|---:|
| Linear SVM | 85.43% | 85.67% |
| Logistic Regression | 85.18% | 85.24% |
| Multinomial Naive Bayes | 78.50% | 75.54% |

## Dataset

**Dataset:** IT Service Ticket Classification Dataset

**Source:** Kaggle

https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset

- Total records: 47,837
- Input feature: Document
- Target feature: Topic_group
- Number of categories: 8

Categories:
1. Hardware
2. Access
3. Miscellaneous
4. HR Support
5. Purchase
6. Administrative rights
7. Storage
8. Internal Project

## Priority Assignment

Priority assignment is implemented using a separate rule-based system.

- High: Critical incidents, security breaches, server outages, and major service disruptions.
- Medium: Access problems, application errors, and service issues.
- Low: Routine installation requests and general service requests.

**Note:** Priority is assigned using predefined rules, not a trained priority-classification model. The dataset does not contain ground-truth priority labels.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- TF-IDF

## Project Structure

```text
FUTURE_ML_02/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── linear_svm_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── Support Ticket Classification & Prioritization.ipynb
├── outputs/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/adriinat/FUTURE_ML_02
cd FUTURE_ML_02
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle and place it in the appropriate `data/raw/` directory if you intend to rerun the notebook.

Run the application:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## Testing

The application was tested for:
- Single ticket classification
- Batch classification using CSV files
- Invalid CSV file uploads
- Missing and empty ticket descriptions
- Batch processing of 100 tickets
- CSV download functionality
- Analytics dashboard updates

## Future Improvements

- Improve category classification with additional training data.
- Add confidence scores for predictions.
- Develop a supervised priority-classification model using labelled priority data.
- Integrate the application with an IT service management platform.

## Author

Developed as a Machine Learning project under FUTURE_ML_02.