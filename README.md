# Student Performance Prediction Web Application

This project is a machine learning web application that predicts student performance based on their demographic data and test scores. The app is built using Flask for the web framework, with machine learning models trained to predict student scores using various regression techniques.


## Description

This application allows users to input student information and predict their academic performance based on historical data. The machine learning model used in this application is trained using different regression algorithms and the user’s input data is processed via the prediction pipeline.

## Features

- **Prediction of Student Performance**: The main goal of this web application is to predict the performance of students based on their demographic information and test scores.
  
- **Web Interface**: Built using Flask, the app offers a simple web interface where users can input student data and receive predictions.

- **Machine Learning Model**: The app uses machine learning models such as Linear Regression, Decision Trees, and Random Forest, which are trained on a dataset of student scores. These models are evaluated and fine-tuned to achieve the best accuracy.

- **Model Evaluation**: Multiple models are evaluated based on performance metrics like the R-squared (R2) score to ensure the most accurate prediction model is used.

## How It Works
1. **Data Ingestion**: The app first ingests the student data provided by the user through an input form on the website.
   
2. **Data Preprocessing**: The input data is processed and transformed into a format that can be used by the machine learning model.

3. **Prediction**: The processed data is fed into a pre-trained model that predicts the student’s performance. The model’s predictions are returned to the user through the web interface.

4. **Model**: The model has been trained using the **student performance dataset** and utilizes regression techniques like Random Forest Regressor, Decision Tree Regressor, and more.
  
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

### Folders and Files

- **src/**: Contains the core Python code for data ingestion, model training, and prediction pipeline.
  - **components/**: Includes scripts for training the model (`model_trainer.py`), data preprocessing (`data_ingestion.py`), and prediction (`predict_pipeline.py`).
  - **exception.py**: Handles custom exceptions.
  - **logger.py**: For logging application events.
  - **utils.py**: Includes utility functions like saving the model and evaluating model performance.
  
- **artifacts/**: Stores the trained model (`model.pkl`) after the training process.

- **templates/**: Contains HTML files for rendering the web pages (e.g., input forms and result display).

- **static/**: Stores static files like CSS, JavaScript, and images.

- **app.py**: The main entry point of the application that contains routes for serving the website and handling user interactions.




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

## Evaluation

- **Model Performance**: Each model is evaluated using the R2 score to assess the accuracy of predictions.

- **Hyperparameter Tuning**: Models are tuned using GridSearchCV to find the optimal parameters that improve performance.

## Conclusion

This project provides an easy-to-use web application to predict student performance based on various factors. The machine learning model is trained on a dataset and can be extended to include more features or use different models to improve accuracy.





