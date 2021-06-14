# Disaster Response Pipeline Project
Motovation
--This is the project from the udacity. In which we have to classify the needs of the servivor based on the different messages received from online resources.In this project user can write the message abd get the classfication result from the 36 categories.

The code will be running in python3.

#Folder Structure:-
app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py
|- classifier.pkl # saved model


#Installation
All libraries are available in Anaconda distribution of Python. The used libraries are:

|-pandas
|-re
|-sys
|-json
|-sklearn
|-nltk
|-sqlalchemy
|-pickle
|-Flask
|-plotly
|-sqlite3

#Acknowledgement
1.)Stakeover flow
2)Towards datascience
3)Documentation of python packages.



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
