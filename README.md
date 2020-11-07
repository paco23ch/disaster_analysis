# Disaster Response Pipeline Project

This is my Udacity project for Disaster relief message analysis.  It's main objective is to take a set of pre-classified messages and be able to predict come categories for disaster response teams to take action on future messages.

## Project Structure

The project contains three major components to cover the end-to-end work:

  - **Data Ingestion & Transformation**: The first module will take care of loading the source messages and classifications, as well as do come cleanup and restructuring of the data, and then finally writing to a database for further use.  The code & data files are stored in the `data` directory.
  - **Machine Learning Training Model**: The second module will use the data ingested and transformed in the first step to train and improve a module. The code & data files are stored in the `models` directory.
  - **Web application**: The third module is a web application that will allow users to send any message and get it classified in any of the categories previously defined.  The code is stored in the `app` directory.
  
## Project Files

- `README.md`. Overview and instructions to run the application modules
- `ETL Pipeline Preparation.html` & `ETL Pipeline Preparation.ipynb`.  ETL pipepline workbook, in order to prepare the `process_data.py` file.
- `ML Pipeline Preparation.html` & `ML Pipeline Preparation.ipynb`.  ML pipeline workbook, in order to prepare the `train_classifier.py` file
- `app/`  Web application.  
  - `run.py`. Main file to run the Web Application.
  - `templates` 
    - `go.html` Template for returning results to the main page.
    - `master.html` Landing page for the app.
- `data/` Data Ingestion & Transformation.
  - `DisasterReponse.db` Data that has been cleaned up and prepared for training.
  - `disaster_categories.csv` Classification of each of the messages.
  - `disaster_messages.csv` Information about messages posted.
  - `process_data.py` This is the main process code to be invoked.
- `models/` Machine Learning Model.
  - `classifier.pkl` Best model from the last run, which can be used for the web aplication
  - `train_classifier.py` Machine Learning Model trainer.   After looking a a couple of potentical classifiers, seems like the AdaBoostClassifier was the best option.
  - `training.log` Result from the last training log.

## Parameters:

  - `run.py`. Main file to run the Web Application.
    - No parameters required
  - `process_data.py` This is the main process code to be invoked.
    - `dissaster_messages_file`. Message file.
    - `dissaster_categories_file`.  Classifications file.
    - `result_database_file`. Result database file.
  - `train_classifier.py` Machine Learning Model trainer.   After looking a a couple of potentical classifiers, seems like the AdaBoostClassifier was the best 
    - `input_database_file` Input database file.  Created in the `process_data.py` file.
    - `model_file` Output model file.
    - `training_log` Training log file, to review results after the model has been run.

## Instructions to run:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl models/training.log`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
