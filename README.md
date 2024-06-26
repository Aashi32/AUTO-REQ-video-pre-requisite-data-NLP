# AUTO-REQ-video-pre-requisite-data-NLP

## Overview
This repository contains the code for the AUTO-REQ-video-pre-requisite-data project, which is focused on identifying pre-requisites between academic videos to enhance the learning experience on online platforms. The project uses Natural Language Processing (NLP) techniques to preprocess and analyze the video transcripts.

## Notebook
The main notebook for this project is located on Google Colab. You can access it "https://colab.research.google.com/drive/1_D00OOSqpFrAKUIXPr4DHSEu_x0NuCzl?usp=sharing".

## Preprocessing Techniques Used
The following preprocessing techniques were used on the video transcripts:

-> Cleaning: Removal of symbols, special characters, and numbers.

-> Tokenization: Splitting text into individual words.

-> Stopword Removal: Removal of common words that do not contribute to the meaning of the text.

-> Lemmatization: Converting words to their base or root form.

## Algorithm Used
The algorithm used in this project leverages the contextual embeddings from BERT and captures sequential information using the Bidirectional GRU (Gated Recurrent Unit) layer. The model is trained to classify text data into binary categories. The training loop updates the model parameters to minimize the loss, and the evaluation provides insights into the model's performance on the test set. Adjusting hyperparameters and experimenting with different pre-trained BERT models could further optimize the model's performance.

## Approach
1. Data Preprocessing: Cleaned the video transcripts using the mentioned techniques to prepare them for analysis.
2. Downsampling: Balanced the dataset by downsampling the majority class to match the number of samples in the minority class.
3. Data Merging: Merged the preprocessed data with the original dataset based on pre-requisite information.
4. Text Classification: Used the BERT + Bi-GRU model to classify the text data into binary categories.

## Usage
To run the notebook, open it in Google Colab and follow the instructions within the notebook.

## Note
This project does not require any specific environment setup or dependencies beyond the libraries imported in the notebook itself.
If you encounter any issues or have questions, feel free to open an issue in this repository.
