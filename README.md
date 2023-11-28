### Receipts_Count_Prediction

**Task:** 
This project aims to help in the prediction of approximate number of receipts (think tickets/requests) to be expected in a specified period.

**My Approach:**

My approach to solving the problem was to treat it as a regression problem. I extracted features from the data
provided and built a linear regression model from scratch. The model was then trained and evaluated.
I used the parameters (weights and bias) from the trained model to run predictions in a Flask app.

The code for the model can be run by running the model.py file in the 'receipt_prediction' folder on a machine with the
package requirements outlined in the **documtation.txt** file. 

Please check **documentation.txt** for guide on how to run the Flask app to generate predictions.


