import sys
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer
import re
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score,f1_score
from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
from sqlalchemy import create_engine, text
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer
import re
from nltk.stem.porter import PorterStemmer
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)    
    df = pd.read_sql_table('Message', engine) 
    df=df.dropna()  
    X=df[['message']]
    X=np.array(X).reshape(-1)
    Y=df.drop(['message','original','genre','id'],axis=1) 
    catagory_name=Y.columns                       
    return X,Y,catagory_name                        
                          


def tokenize(text):
    """Normalize, tokenize and stem text string
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    stemmed: list of strings. List containing normalized and stemmed word tokens
    """
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # Stem word tokens and remove stop words
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return stemmed                            
                           



def build_model():
    pipeline = Pipeline([('vectorizer',CountVectorizer(tokenizer=tokenize)),                        
                         ('tfid', TfidfTransformer()),
                         ('clf',MultiOutputClassifier(estimator=DecisionTreeClassifier()))      
                        ])  
    parameters = {
             #  'vect__stop_words': ('english', None),
                'clf__estimator__criterion':['gini','entropy'],
                'clf__estimator__max_features': ['auto','sqrt','log2'],
                'clf__estimator__min_samples_split': [2, 5]
             }
    cv = GridSearchCV(pipeline,param_grid=parameters)                     
    return cv                          
                           


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred=model.predict(X_test) 
    overall_Accuracy=  (y_pred==Y_test).mean().mean() 
    print('overall Accuracy of the model is',overall_Accuracy)                       
                           
                            


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))                           
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()