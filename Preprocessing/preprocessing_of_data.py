# Install necessary libraries
!pip install transformers
!pip install nltk

# Import libraries
import zipfile
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.utils import resample

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Extract zip file
zip_file_path = '/content/metadata.csv.zip'
output_folder = '/content/'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(output_folder)

# Read metadata
meta_data = pd.read_csv('metadata.csv')

# Preprocess metadata
meta_data.columns = ['video_name','transcript','extracted']

# Load combined data
combined_data = pd.read_csv('train.csv')
df = combined_data.copy()

# Filter metadata based on pre-requisite
mask = meta_data['video_name'].eq(df['pre requisite'])
filtered_meta_data = meta_data[mask]

# Merge filtered metadata with combined data
result_df = pd.merge(df, filtered_meta_data[['video_name', 'transcript', 'extracted']], how='left', left_on='pre requisite', right_on='video_name')
result_df['merged_column'] = result_df['video_name'].astype(str) + ' ' + result_df['transcript'].astype(str) + ' ' + result_df['extracted'].astype(str)
result_df = result_df.drop(columns=['video_name', 'transcript', 'extracted'])

# Downsample majority class to balance data
majority_class = result_df[result_df['label'] == 0]
minority_class = result_df[result_df['label'] == 1]
downsampled_majority = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=42)
downsampled_data = pd.concat([downsampled_majority, minority_class])
downsampled_data = downsampled_data.sample(frac=1, random_state=42)

# Clean and lemmatize text
def clean_and_lemmatize(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = nltk.word_tokenize(cleaned_text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    cleaned_and_lemmatized_text = ' '.join(lemmatized_words)
    return cleaned_and_lemmatized_text

# Apply cleaning and lemmatization
downsampled_data['pre'] = downsampled_data['pre'].apply(clean_and_lemmatize)

# Calculate 75th percentile of lengths of 'pre' column
length_of_pre_column = downsampled_data['pre'].str.len()
percentile_75 = np.percentile(length_of_pre_column, 75)

# Print the 75th percentile
print(f"The 75th percentile of the lengths of 'pre' column is: {percentile_75}")

# Save cleaned and preprocessed data
downsampled_data.to_csv('cleaned_data.csv', index=False)
