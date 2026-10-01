Content Response Prediction: Sentiment Analysis with Gradient Boosting
A sentiment prediction system that classifies how people respond to content, built with a Gradient Boosting model and a text-processing pipeline.

Author: Sierra Gordon

Pipeline map

Overview
This project builds a fast and accurate sentiment classifier that predicts reactions to text content. It uses a Gradient Boosting model together with text preprocessing, feature selection, and class balancing to uncover patterns in sentiment and engagement. The approach can support content optimization and a better understanding of how audiences respond to messaging.

Dataset
The model is trained on a large sentiment dataset available on Kaggle. To keep training efficient, a stratified sample is drawn from the full dataset.

Sampled 80,000 records from a total of 800,000.
Used 64,000 records for training.
Applied SMOTE to balance the classes, producing 64,122 samples across 1,167 features.
Methods
Text Preprocessing and Feature Engineering
Tokenization and stop-word removal with NLTK.
Text vectorization with CountVectorizer.
Feature standardization with StandardScaler.
Feature selection with Recursive Feature Elimination (RFE).
Class balancing with SMOTE.
Model
Classifier: GradientBoostingClassifier from scikit-learn.
Hyperparameter tuning with GridSearchCV.
Pipeline construction with scikit-learn Pipeline.
Results
Test set: 40,000 tweets.

Class	Precision	Recall	F1-Score
0	0.72	0.67	0.69
1	0.69	0.73	0.71
Overall accuracy: 0.70.

![Performance metrics][images/performance_metrics.png] 

### What drives the predictions

The strongest features were behavior metrics, not single words. @mentions (0.104) and average word length (0.098) came out on top, followed by exclamation marks and words like "miss," "sad," "good," and "sorry."

![Feature importance][images/feature_importance.png]

Key Takeaways
Ensemble methods such as Gradient Boosting meaningfully improve classification performance on text sentiment.
SMOTE and RFE are effective for handling class imbalance and reducing feature dimensionality.
Clear visualizations make model performance easier to interpret and communicate.
Repository Contents
content_response_predictions.ipynb — full preprocessing, training, and evaluation notebook
LICENSE — MIT License
README.md — project documentation
How to Run
Clone the repository.
Open content_response_predictions.ipynb in Jupyter or Google Colab.
Upload the dataset to the environment and run the cells in order. Install any missing dependencies with pip install (pandas, numpy, nltk, scikit-learn, imbalanced-learn, matplotlib, seaborn).
Citation
Go, A., Bhayani, R., and Huang, L., 2009. Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(2009), p.12.

License
This project is licensed under the MIT License. See the LICENSE file for details.

[images/feature_importance.png]: images/feature_importance.png [images/performance_metrics.png]: images/performance_metrics.png [images/topic_keywords.png]: images/topic_keywords.png
