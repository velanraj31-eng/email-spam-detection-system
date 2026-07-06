import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.neural_network import MLPClassifier  
from sklearn.metrics import accuracy_score, confusion_matrix  
import matplotlib.pyplot as plt  
import seaborn as sns  

# Load dataset
raw_mail_data = pd.read_csv(r"C:\Users\velan\Downloads\mail_data.csv", encoding="ISO-8859-1")  
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')  

# Encode labels: spam = 0, ham = 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0  
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1  

# Split data
X = mail_data['Message']  
Y = mail_data['Category']  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)  

# TF-IDF Vectorization
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)  
X_train_features = feature_extraction.fit_transform(X_train)  
X_test_features = feature_extraction.transform(X_test)  

# Convert labels to int
Y_train = Y_train.astype('int')  
Y_test = Y_test.astype('int')  

# Train MLP model
model = MLPClassifier() 
model.fit(X_train_features, Y_train)  

# Evaluate
train_pred = model.predict(X_train_features)  
test_pred = model.predict(X_test_features)  
print('Accuracy on training data :', accuracy_score(Y_train, train_pred))  
print('Accuracy on test data :', accuracy_score(Y_test, test_pred))  

# ----------  Confusion Matrix ----------
cm = confusion_matrix(Y_test, test_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Spam', 'Ham'], yticklabels=['Spam', 'Ham'])
plt.title('Confusion Matrix - Spam Detection')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ----------  Test multiple mails ----------
test_mails = [
    "dear friend, can we meet tomorrow evening?",
    "free vacation offer just for you",
    "you have won $5000 click on this link"
    "project meeting rescheduled to 10am"
]

input_features = feature_extraction.transform(test_mails)
predictions = model.predict(input_features)

print("\n Test Mail Predictions:")
for mail, label in zip(test_mails, predictions):
    result = "Ham mail" if label == 1 else "Spam mail"
    print(f" Mail: {mail}\n→ Prediction: {result}\n")

# ----------  Pie chart of results ----------
labels = ['Spam', 'Ham']
counts = [sum(predictions == 0), sum(predictions == 1)]

plt.figure(figsize=(5,5))
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
plt.title('Spam vs Ham Distribution in Test Mails')
plt.show()
