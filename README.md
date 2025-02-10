# content-response-prediction

Fast Sentiment Analysis Using Gradient Boosting

Author: Sierra Gordon

Description
This project predicts how people will react to content using advanced machine-learning techniques. It leverages Gradient Boosting to analyze patterns of sentiment, behavior, and engagement. This system is perfect for optimizing content and understanding user reactions.


Introduction
The purpose of this project is to build a fast and accurate sentiment prediction system that analyzes how people react to different content. By using Gradient Boosting models, we aim to uncover patterns in sentiment, behavior, and engagement. This system can be applied to optimize messaging, understand user reactions, and influence opinions through targeted content.

Setup Instructions

Clone the repository:

!git clone <repository-url>

Run the notebook: Ensure all cells are executed in order for the system to work correctly.

Note: Make sure to upload your dataset to the Colab environment. You can find the dataset on Kaggle here. https://www.kaggle.com/dataset-url

Install dependencies:
Importing Libraries
import logging
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
import nltk
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import random
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE

Data Sampling and Resampling
Data Sampling: Extracted 80,000 samples from a total of 800,000.

Training Split: Used 64,000 samples for training.

Resampling: Applied SMOTE to balance the classes, resulting in 64,122 samples with 1,167 features.
Model Training and Evaluation
Model Used: Gradient Boosting Classifier

##  metrics and their values
metrics = ['Precision', 'Recall', 'F1-Score']
class_0 = [0.72, 0.67, 0.69]
class_1 = [0.69, 0.73, 0.71]
accuracy = [0.70] * 3  # Accuracy for both classes and overall

# Create a bar plot
x = np.arange(len(metrics))
width = 0.2

fig, ax = plt.subplots()
bar1 = ax.bar(x - width, class_0, width, label='Class 0')
bar2 = ax.bar(x, class_1, width, label='Class 1')
bar3 = ax.bar(x + width, accuracy, width, label='Overall Accuracy')
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Model Performance Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Annotate bars with their values
def annotate_bars(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')

annotate_bars(bar1)
annotate_bars(bar2)
annotate_bars(bar3)

fig.tight_layout()
plt.show()
    Cool Features
Topic sentiment tracking

Behavior analysis

Text pattern detection

Engagement prediction

Challenges I Solved
Made single predictions work perfectly

Sped up processing time

Enhanced vocabulary handling

Built smart feature combinations

Balanced the dataset and selected key features

What I Learned
Ensemble methods significantly improve model performance

Gradient Boosting offers unique strengths

SMOTE and RFE are powerful tools for improving model accuracy

Clear visuals help show results

License
MIT License Copyright (c) 2025 Sierra Gordon Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Citation
If you use this dataset, please cite the following paper: Go, A., Bhayani, R. and Huang, L., 2009. Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(2009), p.12.

Built this to make content optimization faster and better!
