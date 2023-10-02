# DEEP-LEARNING-
Comment Toxicity Detection Using NLP

The goal is to build a model that can accurately predict whether a given comment contains toxic content. Here are the key highlights:

1)The code begins by installing the necessary dependencies using pip. The required libraries, including TensorFlow, pandas, matplotlib, and scikit-learn, are installed.

2)Performed data pre-processing, model creation, training, evaluation, and visualization.

3)The code reads a CSV file named "train.csv" . The CSV file contains toxic comment data. This dataset will be used for training the toxic comment classification model.

4)Preprocessing steps are performed on the data to prepare it for model training. The text data is extracted from the DataFrame, and a Text Vectorization layer is created using TensorFlow's Text Vectorization class.The vectorization process helps in representing the textual data in a numerical format that can be utilized by the deep learning model.

5)A TensorFlow dataset is created from the vectorized text and corresponding labels.

6)Splitting ensures that the model is trained on a majority of the data, validated on a smaller portion, and tested on a separate portion to evaluate its performance.

7)A sequential model is created using the Keras API.The model includes an embedding layer, a bidirectional LSTM layer, fully connected layers for feature extraction, and a final dense layer with sigmoid activation for multi-label classification. The embedding layer learns to represent words in a continuous vector space, while the bidirectional LSTM layer processes the sequential information in both forward and backward directions. The fully connected layers serve as feature extractors, and the final dense layer with sigmoid activation produces the probability scores for each class label.

8)The model is compiled with the BinaryCrossentropy loss function and the Adam optimizer.

9)The model's performance is evaluated on the validation dataset during training to monitor its progress.

10)During the training process, the code captures the training history, including the changes in loss and accuracy over the epochs.

11)A threshold of 0.5 is applied to the predicted probabilities to convert them into binary labels, indicating whether a comment is toxic or not.

12)Evaluation metrics such as precision, recall, and accuracy are calculated using the TensorFlow metrics classes.

13)To enhance the user experience and demonstrate the model's functionality, the code includes the installation of the Gradio and Jinja2 libraries.

14)The function takes a comment as input, preprocesses it using the saved vectorizer, passes it through the loaded model, and generates the toxicity.

I greatly appreciate Nicholas Renotte for his informative youtube videos that helped me in learning the concepts of deep learning and building this
model!
