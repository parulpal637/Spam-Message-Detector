# Step 1: Sample messages
messages = [
    "Congratulations! You won a free gift",
    "Call me when you are free",
    "Win cash prize now",
    "Let's meet tomorrow",
    "Free entry in a contest"
]

# Step 2: Labels (Binary - BEST practice)
# 1 = Spam, 0 = Not Spam
labels = [1, 0, 1, 0, 1]

# Step 3: Convert text to numbers
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Step 4: Train the model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, labels)

# Step 5: Test with new message
new_message = ["You have won cash prize"]
new_X = vectorizer.transform(new_message)

result = model.predict(new_X)

# Step 6: Show result
if result[0] == 1:
    print("Spam Message")
else:
    print("Not Spam Message")
