# cartruckclassifier
In order to successfully launch this code you will need

python 3.12.8
pytorch 2.6.0

Hi, so if you want to run the program there are two ways
the first is to run classifier.py and wait for some time until the model is trained, this can take some time,
the second way is to run trained_model.py which uses trained_model.pth already trained in classifier.py.

I have prepared two variants of the pth file with training at 20 epochs and at 30,
the variant can be changed in the code itself in the trained_model.py file

Here is classification report for 20 epochs(there is no big difference from 30 epochs):

accuracy: 77.68296685264194%

Detailed Classification Report:
              precision    recall  f1-score   support

       Truck       0.77      0.85      0.81      1698
         Car       0.78      0.69      0.73      1349

    accuracy                           0.78      3047
   macro avg       0.78      0.77      0.77      3047
weighted avg       0.78      0.78      0.77      3047
