# Hate Speech Detection Web App

## Overview
The **Hate Speech Detection Web App** is a machine learning-based application built using **Streamlit** that detects whether a given text contains hate speech or not. The app leverages **Natural Language Processing (NLP)** and **Support Vector Machine (SVM)** classifier to classify text into two categories: Hate Speech or Non-Hate Speech.

## Features
- Simple and interactive UI
- Text preprocessing using NLTK
- Hate speech detection using SVM classifier
- Instant results with success or error messages
- Informative message alerts

## Technologies Used
- Python
- Streamlit
- Scikit-Learn
- NLTK
- Pandas
- Seaborn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Venusbits/Hate-Speech-Detection
   cd Hate-Speech-Detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If `seaborn` is missing, add it manually:
   ```bash
   pip install seaborn
   ```
3. Run the app locally:
   ```bash
   streamlit run hate.py
   ```

## How to Use
1. Enter your text into the input box.
2. Click the **Detect** button.
3. The app will clean the text and classify it into either **Hate Speech** or **Offensive** or **Non-Hate Speech**.
4. View the result on the app screen.

## Deployment
The app is deployed on **Streamlit Cloud** and can be accessed at:
👉 [Hate Speech Detection App](https://hate-speech-detection-ljkr8wz53fvzdd28pvmemy.streamlit.app/)

## Contributing
Contributions are welcome! If you'd like to improve the model or UI, feel free to fork the repository and create a pull request.

## Contact
For any queries or feedback, feel free to reach out to **Vishakha Sapkal**.

Let's build a positive digital space! 😊

