import sys
import pickle
import logging

from sqlalchemy import create_engine
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

parameters = {'multi__estimator__learning_rate' : [0.9, 1],
              'multi__estimator__n_estimators' : [100, 200, 300]}

def load_data(database_filepath):
    """Load the input data stored in the database.

    Args:
    database_filepath - Database file

    Returns:
    X - Fact columns for the model
    Y - Label columns for the model
    Y.columns - Label column names after splitting the input data into X & Y
    """
    x_columns = ['id','message','original','genre']
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('ClassifiedMessages',engine)
    X = df['message']
    Y = df.drop(x_columns, axis=1)
    return(X, Y, Y.columns)


def tokenize(text):
    """Convert the input into tokens for the machine learning algorithms

    Args:
    text - Text to be converted into tokens

    Returns:
    clean_tokens - List of tokens once transformed
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_tok = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_tok)
        
    return(clean_tokens)


def build_model():
    """Load the input data stored in the database.

    Args:
    none

    Returns:
    pipeline - The newly created pipeline containing all transformation steps and the output classifier.
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('multi', MultiOutputClassifier(estimator=AdaBoostClassifier()))])

    return(pipeline)

def tune_model(model, X_train, y_train, parameters):
    """Run a GridSearchCV on a given model

    Args:
    model - Previously created model to be tuned
    X_train - Input data for running the GridSearch on
    Y_train - Input labels for running the GridSearch on
    parameters - to be used in GridSearch
    

    Returns:
    best_Estimator - after nunning the GridSearch algorith, the best model it could come up with
    """
    cv = GridSearchCV(model,param_grid=parameters, verbose=25)
    cv.fit(X_train, y_train)

    return(cv.best_estimator_)

def evaluate_model(model, X_test, Y_test, category_names):
    """Print the metrics for each of the categories using the test data

    Args:
    model - to be evaluated
    X - Fact columns for the model (test set)
    Y - Label columns for the model (test set)
    category_names - Label column names after splitting the input data into X & Y

    Returns:
    None
    """
    y_pred = model.predict(X_test)

    for i,col in enumerate(category_names):
        y_pred_i = y_pred.T[i]
        y_test_i = Y_test[col]
        logging.info('*** {}. Classification report for {}'.format(i+1,col))
        logging.info(classification_report(y_test_i, y_pred_i))

    return

def save_model(model, model_filepath):
    """Save model to a file

    Args:
    model - to be saved
    model_filepath - File where the model is to be saved
    
    Returns:
    None
    """    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """Main function to be called from command line

    Args:
    database_filepath - database file where test and training data reside
    model_filepath - output model file to be created
    log_file - log file used for writing all actions
    
    Returns:
    None
    """
    if len(sys.argv) == 4:
        database_filepath, model_filepath, log_file = sys.argv[1:]

        logging.basicConfig(filename=log_file, level=logging.DEBUG, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')          
        logging.debug('Setting up log ...')
        
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        logging.info('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        logging.info('Building model...')
        model = build_model()
        
        logging.info('Training model...')
        model.fit(X_train, Y_train)
        
        logging.info('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        logging.info('Improving model ...')
        model = tune_model(model, X_train, Y_train, parameters)

        logging.info('Evaluating mode ... ')
        evaluate_model(model, X_test, Y_test, category_names)

        logging.info('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        logging.info('Trained model saved!')

    else:
        logging.info('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()