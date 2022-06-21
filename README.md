<a><img alt = 'python' src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white"></a>
<a><img alt = 'spyder' src="https://img.shields.io/badge/Spyder%20Ide-FF0000?style=for-the-badge&logo=spyder%20ide&logoColor=white"></a>
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

<a><img alt='tf' src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"></a>


# Predictive Classification Model Using Machine Learning Model Support Vector Classification (SVC) for Heart Attack Analysis.
 Using machine learning model to train over 300  Heart Attack Analysis and Prediction dataset to categorize either the person has a chance to get heart attack or not.

## Description
1. The project's objective is to categorize the IMDB movies rating. 
2. The IMDB movie reviews contain enormous amount of data, which can be used to predict whether the movie review is a negative or positive review. 
3. The dataset contains anomalies such as HTML tags (removed using RegEx), lowercase/uppercase, and duplicates data.
4. The method used for the deep learning model are word embedding, LSTM and Bidirectional.
5. Several method can be used to improve the model such as lemmatization, stemming, CNN, n-grams, etc.

### Deep learning model images
![model_architecture](static/model.png)

## Results
Training loss & Validation loss:
![model_loss](static/loss.png)

Training accuracy & Validation accuracy:
![model_accuracy](static/accuracy.png)

Model score:
![model_score](static/score_sentiment.PNG)

## Discussion
1. The model achieved 84% accuracy during training. 
2. Both recall and f1 score report 85%. 
3. However, the model starts to overfit after 2nd epochs. Early stopping can be used to prevent overfitting. The dropout data can be increased to control overfitting.

## Credits:
Shout out to Ankit152 for the IMDB Dataset. Check out the dataset by clicking the link below. :smile:
### Dataset link
[IMDB-Sentiment-Analysis](https://github.com/Ankit152/IMDB-sentiment-analysis)
