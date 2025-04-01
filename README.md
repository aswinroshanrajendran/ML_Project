# Student Performance Prediction Web Application

This project is a machine learning web application that predicts student performance based on their demographic data and test scores. The app is built using Flask for the web framework, with machine learning models trained to predict student scores using various regression techniques.


## Description

This application allows users to input student information and predict their academic performance based on historical data. The machine learning model used in this application is trained using different regression algorithms and the user’s input data is processed via the prediction pipeline.

## Features

- **Input Form**: Users can fill out a form with data about a student’s demographics and test scores.
- **Prediction**: After submitting the form, the system predicts the student’s performance.
- **Machine Learning Model**: The application uses a trained regression model to make predictions.
  
## Requirements

Before running the application, ensure that you have the following libraries installed:

1. **Python 3.x**
2. **Flask**
3. **scikit-learn**
4. **pandas**
5. **numpy**
6. **dill**
7. **catboost** (if using CatBoost in the model)

You can install the dependencies using the `requirements.txt` file:

pip install -r requirements.txt

## Setup Instructions
Clone the repository:
git clone https://github.com/your-username/ML_Project.git
cd ML_Project

Install dependencies:

Run the following command to install the necessary libraries:
pip install -r requirements.txt


Start the Flask application:

You can start the Flask application by running:

python app.py


he app will be available at http://127.0.0.1:5000/.

Model Training:

The model_trainer.py script handles the training of the model using various regression algorithms like Linear Regression, Random Forest, Decision Tree, etc.

The trained model is saved in the artifacts/ folder as model.pkl.

Prediction:

Users can enter details such as gender, ethnicity, parental education level, test preparation course, and scores.

The predict_pipeline.py is responsible for handling the data and making predictions.

How to Use
Open the web app: Once the app is running, open your browser and visit http://127.0.0.1:5000/.

Enter student details: On the homepage, you’ll see a form where you can enter the student’s information (gender, ethnicity, etc.).

Submit the form: Once you’ve entered the data, submit the form, and the model will predict the student’s performance.

View results: The predicted result will be displayed on the page.

Folder Details
src/: Contains all the Python scripts that handle the data ingestion, model training, prediction pipeline, and utility functions.

components/: Includes scripts like data_ingestion.py for processing input data, model_trainer.py for training models, and predict_pipeline.py for prediction logic.

exception.py: Handles custom exceptions used in the application.

logger.py: For logging important events, errors, and information during execution.

utils.py: Includes utility functions for tasks like saving models and evaluating the performance of different algorithms.

artifacts/: Folder where the trained model (model.pkl) is saved.

templates/: Folder containing the HTML files for rendering the web pages in the Flask app.

static/: Folder for static assets like CSS, images, and JavaScript files.

app.py: Main entry point of the Flask app where routes are defined for serving pages and making predictions.

Model Training
The machine learning model is trained using various regression algorithms. These algorithms are evaluated using different hyperparameters through GridSearchCV to ensure the best possible performance.

Algorithms Used:

Linear Regression

Random Forest Regressor

Decision Tree Regressor

XGBoost Regressor

CatBoost Regressor

AdaBoost Regressor

Gradient Boosting Regressor

Evaluation: The performance of each model is evaluated using R2 score.




