# 📨 SMS Spam Classifier

Detect spam messages using natural language processing (NLP) and classic machine learning.

---

## 🎯 Project Overview
This project uses the **SMS Spam Collection dataset** (UCI / Kaggle) to build a text classifier that predicts whether a message is *spam* or *ham* (not spam).  

The workflow includes:
1. Data loading and cleaning  
2. Exploratory Data Analysis (EDA)  
3. Text feature extraction with **TF-IDF**  
4. Training Logistic Regression, SVM, and Naive Bayes models  
5. Evaluating performance with Accuracy, F1, ROC-AUC, and confusion matrices  

---

## 🧰 Tech Stack
- Python 3.9+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- TF-IDF Vectorizer
- Logistic Regression / Linear SVM / MultinomialNB

---

## 📊 Results
| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|-----------|------------|---------|----|------|
| Logistic Regression | ~98% | ~98% | ~97% | ~97% | ~0.99 |
| Linear SVM | ~98% | ~98% | ~98% | ~98% | ~0.99 |
| MultinomialNB | ~97% | ~97% | ~95% | ~96% | ~0.98 |

✅ **Top spam terms:** *free*, *win*, *txt*, *claim*, *call*  
✅ **Top ham terms:** *ok*, *love*, *home*, *see*, *later*

---

## 📦 Dataset
**Name:** SMS Spam Collection  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  
**Rows:** 5572 messages  
**Labels:** 0 = ham, 1 = spam

---

## ⚙️ How to Run
```bash
conda activate ds-fresh
jupyter notebook notebook/sms_spam_classifier.ipynb
