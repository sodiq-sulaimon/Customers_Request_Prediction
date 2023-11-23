### Receipts_Count_Prediction

**Task:** 
This project aims to help in the prediction of approximate number of receipts (think tickets/requests) to be expected in a given period.

**My Approach:**

My approach to solving the problem was to treat it as a regression problem. I extracted features from the data
provided and built a linear regression model from scratch. The model was then trained and evaluated.
I used the parameters (weights and bias) from the trained model to run predictions in the flask app.

The code for the model can be run by running the model.py file in the 'receipt_prediction' folder on a machine with the
package requirements outlined in the **documtation.txt** file. 

Please check **documtation.txt** for guide on how to run the app.

PS: The model is deployed on Heroku at https://receipts-9150bcabee7e.herokuapp.com but I am currently having issued with the account so the app may not be accessible.


