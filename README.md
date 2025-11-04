# EMAIL-SMS-SPAM-DETECTION-USING-ML-METHODS

# 📧 Spam and Ham Detection using Machine Learning

This project classifies SMS and email messages as **Spam** or **Ham (Not Spam)** using multiple machine learning algorithms. It uses **TF-IDF vectorization** for feature extraction and compares models like Logistic Regression, Naive Bayes, Random Forest, and SVM.

---

## 🚀 Features

* Real-time spam/ham message classification
* Data cleaning and preprocessing pipeline
* TF-IDF feature extraction
* WordCloud visualization for spam and ham messages
* Confusion Matrix, ROC Curve, and Model Accuracy plots
* Multiple ML models compared for performance

---

## 🧰 Technologies Used

* Python 3.8+
* pandas, numpy
* matplotlib, seaborn
* wordcloud
* scikit-learn

---

## 🧪 Models Compared

| Model               | Accuracy (%) |
| ------------------- | ------------ |
| Logistic Regression | ~97%         |
| Naive Bayes         | ~98%         |
| Random Forest       | ~96%         |
| Linear SVM          | ~98%         |

---

## 📊 Visualizations

| Visualization               | Description                                           |
| --------------------------- | ----------------------------------------------------- |
| Spam Distribution Pie Chart | Shows proportion of spam vs ham                       |
| WordClouds                  | Displays most frequent words in spam and ham messages |
| Confusion Matrices          | Visualizes classification performance                 |
| ROC Curves                  | Compares model performance                            |
| Accuracy Bar Graph          | Shows model comparison                                |

All generated images are saved inside the **`images/`** folder.

---

## 📁 Dataset

Dataset used: [SMS Spam Collection Data Set - UCI Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

Make sure to place the dataset file as:

```
dataset/spam.csv
```

---

## ⚙️ Installation and Running

1. Clone the repository:

   ```bash
   git clone https://github.com/swathivaddineni/EMAIL-SMS-SPAM-DETECTION-USING-ML/new/main?readme=1
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Python script:

   ```bash
   python spam_ham_detection.py
   ```

4. Once the program starts, you can type messages manually:

   ```
   💬 Enter a message or email to classify (or type 'exit' to stop): Congratulations! You have won a free iPhone!
   🚨 SPAM!
   ```

---

## 📷 Example Outputs

### Spam vs Ham Distribution

<img width="604" height="524" alt="Screenshot 2025-11-04 144314" src="https://github.com/user-attachments/assets/cfb65952-c89f-4159-82a7-266034a4b308" />

### Word Clouds

<img width="1329" height="472" alt="Screenshot 2025-11-04 144831" src="https://github.com/user-attachments/assets/72d158f9-ce67-44be-bb31-e34a674d3d3d" />

### Model Accuracy Comparison

<img width="1019" height="584" alt="Screenshot 2025-11-04 144937" src="https://github.com/user-attachments/assets/28a3a322-1bd5-45b7-9b22-4440ea6f9f3f" />

### ROC Curve

<img width="1050" height="643" alt="Screenshot 2025-11-04 145124" src="https://github.com/user-attachments/assets/7e631094-335f-4b7a-9b1d-d273edeef235" />



