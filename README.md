# Text-Classification-using-a-CNN

An extension to sentiment analysis of identifying a positive or negative sentiment from a sentence. This model learns on 7 different emotions listed below:
1. Joy
2. Anger
3. Fear
4. Sadness
5. Disgust
6. Shame
7. Guilt

It is a multi class text classification problem. A CNN was used to build the model as it is faster to train and a better option for sentences of short length (not paragraphs) as they do not consist of long term depedancies. Google News word2vec word embeddings were also used to refine the model and imporve the accuracy of the model.

Here are the results of training and testing the model for the parameters recorded in parameters.json

Accuracy on dev set: 0.608163265306
Accuracy on train set is 0.986928104575 based on the best model
Accuracy on test set is 0.595689092097 based on the best model

To run the CNN Classifier: python training.py
