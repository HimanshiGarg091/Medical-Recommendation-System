# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset containing diseases and their symptoms
df = pd.read_csv("DiseaseAndSymptoms.csv")

# Display the first few rows of the dataset
df.head()

# Calculate the percentage of missing values in each column
mis_per = (df.isnull().sum() / len(df)) * 100
print("Missing values percentage in each column:\n", mis_per)

# Fill missing values with '1' as a placeholder
df = df.fillna('1')

# Check the frequency of each disease in the dataset
df['Disease'].value_counts()

# Load additional datasets containing precautions, diets, medications, and descriptions
precaution_df = pd.read_csv("precautions_df.csv")
diet_df = pd.read_csv("diets.csv")
medicine_df = pd.read_csv("medications.csv")
description_df = pd.read_csv("description.csv")

# Check the structure of the medications dataset
medicine_df.info()

# Display the first few rows of the precautions dataset
precaution_df.head()

# Remove unnecessary columns from the precautions dataset
precaution_df.drop(['Unnamed: 0'], axis=1, inplace=True)

# Check the structure of the precautions dataset after dropping the column
precaution_df.info()

# Fill missing values in the precautions dataset with '1'
precaution_df.fillna('1')

# Count unique values in the precautions dataset
precaution_df.nunique()

# Merge all datasets on the 'Disease' column
merged_df = pd.merge(df, precaution_df, on='Disease', how='inner')
merged_df = pd.merge(merged_df, diet_df, on='Disease', how='inner')
merged_df = pd.merge(merged_df, medicine_df, on='Disease', how='inner')
merged_df = pd.merge(merged_df, description_df, on='Disease', how='inner')

# Display the first few rows of the merged dataset
merged_df.head()

# Check the structure of the merged dataset
merged_df.info()

# Encode disease names into numerical values for model training
lb = LabelEncoder()
merged_df['Disease_encoded'] = lb.fit_transform(merged_df['Disease'])

# Display a sample of original diseases and their encoded values
merged_df[['Disease', 'Disease_encoded']].sample(15)

# Create a sorted list of all unique symptoms
all_symptoms = sorted(set(merged_df['Symptom_1'].unique()) |
                      set(merged_df['Symptom_2'].unique()) |
                      set(merged_df['Symptom_3'].unique()) |
                      set(merged_df['Symptom_4'].unique()))

# Map symptoms to indices
symptom_dict = {symptom: idx for idx, symptom in enumerate(all_symptoms)}

# Function to convert symptoms into a binary vector
def symptoms_to_binary_vector(row):
    vector = np.zeros(len(symptom_dict), dtype=int)
    for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
        symptom = row[col]
        if pd.notna(symptom) and symptom in symptom_dict:
            vector[symptom_dict[symptom]] = 1
    return vector

# Apply the function to create symptom vectors for all rows
symptom_vectors = merged_df.apply(symptoms_to_binary_vector, axis=1)
symptom_vectors_df = pd.DataFrame(symptom_vectors.tolist(), columns=[f"Symptom_{i}" for i in range(len(symptom_dict))])

# Combine processed data with encoded symptoms
processed_data = pd.concat([merged_df[['Disease_encoded', 'Disease', 'Medication',
                                       'Precaution_1', 'Precaution_2', 'Precaution_3',
                                       'Precaution_4', 'Description', 'Diet']], symptom_vectors_df], axis=1)

# Display a sample of the processed data
processed_data.sample(5)

# Separate features (symptoms) and target (encoded diseases)
x = processed_data.drop(columns=['Disease', 'Precaution_1', 'Precaution_2',
                                  'Precaution_3', 'Precaution_4', 'Description',
                                  'Diet', 'Disease_encoded', 'Medication'])
y = processed_data['Disease_encoded']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize models
knn = KNeighborsClassifier(n_neighbors=5)  # K-Nearest Neighbors
model = MultinomialNB()  # Naive Bayes

# Train the models
model.fit(x_train, y_train)
knn.fit(x_train, y_train)

# Make predictions using KNN
y_pred = knn.predict(x_test)

# Evaluate the KNN model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy with KNN:", accuracy)

# Evaluate the Naive Bayes model's accuracy
y_pred_nb = model.predict(x_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Model Accuracy with Naive Bayes:", accuracy_nb)

# Experiment with different values of 'k' for KNN
accuracy_scores = []
k_values = range(1, 21)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Plot the accuracy of KNN for different values of 'k'
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o', color='b')
plt.title('KNN Model Accuracy for Different Values of k')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid()
plt.show()

# Visualize final accuracy of the KNN model
plt.figure(figsize=(6, 4))
plt.bar(['KNN Model'], [accuracy], color='skyblue')
plt.ylim(0, 1)  # Set y-axis range from 0 to 1 for clarity
plt.title(f'Final Accuracy of KNN Model (k=5)')
plt.ylabel('Accuracy')
plt.text(0, accuracy + 0.02, f'{accuracy:.2f}', ha='center', va='bottom', fontsize=12)
plt.show()

# Generate confusion matrix for KNN predictions
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Example usage: Predict disease for a new input
sample_input_vector = np.zeros(len(symptom_dict))
sample_input_vector[symptom_dict[' chills']] = 1
sample_input_vector[symptom_dict[' joint_pain']] = 1
sample_input_vector[symptom_dict[' stomach_pain']] = 1
sample_input_vector[symptom_dict[' acidity']] = 1
sample_input_vector[symptom_dict[' shivering']] = 1

# Reshape input vector for prediction
sample_input_vector = sample_input_vector.reshape(1, -1)

# Predict the disease and retrieve relevant information
predicted_disease_encoded = knn.predict(sample_input_vector)[0]
predicted_disease = lb.inverse_transform([predicted_disease_encoded])[0]

# Retrieve details of the predicted disease
disease_info = merged_df[merged_df['Disease'] == predicted_disease].iloc[0]
precautions = [disease_info['Precaution_1'], disease_info['Precaution_2'],
               disease_info['Precaution_3'], disease_info['Precaution_4']]
description = disease_info['Description']
diet = disease_info['Diet']

# Display the prediction and associated details
print("Predicted Disease:", predicted_disease)
print("Precautions:", precautions)
print("Description:", description)
print("Diet:", diet)
