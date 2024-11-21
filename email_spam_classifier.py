# Importing required libraries
import pandas as pd  # Library for data manipulation and analysis, used for reading the dataset.
from sklearn.feature_extraction.text import CountVectorizer  # Used to convert a collection of text documents into a matrix of token counts.
from sklearn.model_selection import train_test_split  # Function to split the dataset into training and testing sets.
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes classifier, often used for text classification tasks.
from sklearn.metrics import accuracy_score  # Function to calculate accuracy metrics for classification models.

# Step 1: Load the dataset
data = pd.read_csv("emails.csv")  # Reads the 'emails.csv' file into a pandas DataFrame.
data.head()  # Displays the first five rows of the dataset to get an overview.

# Step 2: Text Vectorization
vectorizer = CountVectorizer()  # Initialize the CountVectorizer which converts text data to numerical form.
X = vectorizer.fit_transform(data['text'])  # Transforms the 'text' column of the dataset into a sparse matrix of token counts.

# Step 3: Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, data['spam'], test_size=0.2, random_state=42)
# 'X_train' and 'Y_train' will be used for training the model,
# 'X_test' and 'Y_test' will be used to evaluate the model's performance.
# The 'test_size=0.2' parameter means 20% of the data will be used for testing, and 80% will be used for training.

# Step 4: Train the model
model = MultinomialNB()  # Initialize the Multinomial Naive Bayes model.
model.fit(X_train, Y_train)  # Fit the model to the training data (X_train) and labels (Y_train).

# Step 5: Make predictions
Y_pred = model.predict(X_test)  # Use the trained model to make predictions on the test data.

# Step 6: Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)  # Calculate the accuracy of the model by comparing predicted labels with actual labels.
print(f"Model accuracy: {accuracy * 100:.2f}%")  # Print the accuracy as a percentage.

# Step 7: Function to predict whether a message is spam or not
def predict_message(message):
    # Transform the input message into the same vector format as the training data.
    message_vector = vectorizer.transform([message])
    # Use the trained model to predict the label (spam or ham) for the input message.
    prediction = model.predict(message_vector)
    # Return the prediction: 'spam' if the predicted label is 1, 'ham' if it's 0.
    return 'spam' if prediction[0] == 1 else 'ham'

# Step 8: User Input and Prediction
# Ask the user to input a message to classify as spam or ham.
user_input = input('Enter a message: ')
# Call the 'predict_message' function to classify the user's message.
message_prediction = predict_message(user_input)
# Print the predicted label (spam or ham) for the user's message.
print(f'Your message is: {message_prediction}')

