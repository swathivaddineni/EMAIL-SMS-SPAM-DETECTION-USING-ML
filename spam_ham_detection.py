import pandas as pd
import numpy as np
import re, string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_curve, auc
)


df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ["label", "message"]
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

email_df = df.copy()
df = pd.concat([df, email_df], axis=0).drop_duplicates().dropna()

print("✅ Combined Dataset Shape:", df.shape)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_message'] = df['message'].apply(clean_text)


# Pie Chart - Spam vs Ham
plt.figure(figsize=(5,5))
df['label'].value_counts().plot.pie(
    autopct='%1.1f%%',
    labels=['HAM', 'SPAM'],
    colors=['skyblue', 'salmon']
)
plt.title("Spam vs Ham Distribution")
plt.ylabel("")
plt.savefig("spam_distribution.png")
plt.show()

spam_words = ' '.join(df[df['label']==1]['clean_message'])
ham_words = ' '.join(df[df['label']==0]['clean_message'])

spam_wc = WordCloud(width=600, height=400, background_color='black', colormap='Reds').generate(spam_words)
ham_wc = WordCloud(width=600, height=400, background_color='black', colormap='Blues').generate(ham_words)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(spam_wc, interpolation='bilinear')
plt.axis('off')
plt.title("Spam Word Cloud")

plt.subplot(1,2,2)
plt.imshow(ham_wc, interpolation='bilinear')
plt.axis('off')
plt.title("Ham Word Cloud")

plt.savefig("wordclouds.png")
plt.show()


X = df['clean_message']
y = df['label']

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)


models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Linear SVM": LinearSVC()
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = round(acc * 100, 2)
    predictions[name] = y_pred
    print(f"\n✅ {name} Accuracy: {results[name]}%")
    print(classification_report(y_test, y_pred, target_names=['HAM', 'SPAM']))


for name, y_pred in predictions.items():
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["HAM", "SPAM"])
    disp.plot(cmap='viridis')
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_')}.png")
    plt.show()


plt.figure(figsize=(8,5))
sns.barplot(
    x=list(results.keys()),
    y=list(results.values()),
    hue=list(results.keys()),
    palette="viridis",
    dodge=False,
    legend=False
)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.xlabel("Models")
plt.ylim(80, 100)
plt.savefig("model_accuracy_comparison.png")
plt.show()


plt.figure(figsize=(8,6))
for name, model in models.items():
    try:
        if hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')
    except Exception as e:
        print(f"⚠️ ROC Curve skipped for {name}: {e}")

plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve_comparison.png")
plt.show()


best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name} ({results[best_model_name]}% Accuracy)")

def predict_message(msg):
    msg_clean = clean_text(msg)
    msg_vec = tfidf.transform([msg_clean])
    pred = best_model.predict(msg_vec)[0]
    print("\n📩 Message:", msg)
    print("🚨 SPAM!" if pred == 1 else "✅ HAM (Not Spam)")

while True:
    msg = input("\n💬 Enter a message or email to classify (or type 'exit' to stop): ")
    if msg.lower() == 'exit':
        print("🔚 Stopped.")
        break
    predict_message(msg)
