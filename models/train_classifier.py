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
    x_columns = ['id','message','original','genre']
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('ClassifiedMessages',engine)
    X = df['message']
    Y = df.drop(x_columns, axis=1)
    return(X, Y, Y.columns)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_tok = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_tok)
        
    return(clean_tokens)


def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('multi', MultiOutputClassifier(estimator=AdaBoostClassifier()))])

    return(pipeline)

def tune_model(model, X_train, y_train, parameters):
    cv = GridSearchCV(model,param_grid=parameters, verbose=25)
    cv.fit(X_train, y_train)

    return(cv.best_estimator_)

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)

    for i,col in enumerate(category_names):
        y_pred_i = y_pred.T[i]
        y_test_i = Y_test[col]
        logging.info('*** {}. Classification report for {}'.format(i+1,col))
        logging.info(classification_report(y_test_i, y_pred_i))

    return

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
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