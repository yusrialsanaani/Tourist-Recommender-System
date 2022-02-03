# Project Title
A Travel attractions recommendation system that suggests tourists to the attractions to visit based on their choices.

Description
The primary objective is to suggest tourist attractions to the user by considering many factors which are essential to make the recommender system a successful model. Then the model is implemented on a chatbot to enhance user experience. 
We used different techniques to create multiple models for recommender systems and choose our best model as per their performance.

Software used
* Python version 3.7
* Google Cobal for implementation of  recommender system
* Google Dialogflow for implementing the chatbot
* Jupyter notebook to integrate flask framework to integrate python code to the chatbot

Libraries used: 
* scikit-learn
* re
* nltk
* pandas
* numpy
* seaborn
* beautiful Soup
* url open
* requests
* plotly
* dash
* matplot
* surprise
* flask


Code references:

* attraction_scrapping.ipynb: This code is used to scrape data from Trip Advisor Webpage.
* comments_10location.ipynb: This code is used for scrapping user comments for different location.the data is been scrapped from trip advisor.
* EDA_TRS.ipynb: This code implements the detailed Data Exploratory analysis on the dataset that we scrapped. 
* App.py: This code includes 4 recommender systems based on similarity model to provide 4 different type of recommendations including: 
* Providing three similar recommendations to previously visited places by the user. 
* Providing the top two recommendations for the best time chosen by the user. 
* Providing the top two recommendations for the province chosen by the user. 
* Providing the top two recommendations for attraction type chosen by the user. Also, it includes a flask set up to provide fulfillment routes from/to chatbot to receive the user queries and return the corresponding recommendations. 
* Recommender_Surprise.ipynb: This code introduces both types of algorithms (Memory and Model based algorithms). We compared them by performing cross validation and evaluating them using RMSE and MAE and choose the best one. Then, create recommender system, get predictions, and explore obtained results. 
* Recommender_ClusteringKm.ipynb: This code is to build a content based recommender system using Kmeans clustering. 
* Content based recommender system.ipynb: In this code we considered user preferences. Based on those we create a user-profile which is used to recommend similar attractions to the users. We used cosine similarity to calculate distance between the features and recommend the appropriate suggestions. 
Models Implemented using this technique are:
* Based on location type
* Based on province your wants to visit
* Based on the time of year user wants to visit
* Based on mixed features(Previously visited location)
* Classification based recommender system.ipynb: This model is based on using supervised machine learning algorithms.We calculate accuracy and then choose our champion model. Based on the results, we do predictions with the testing labels and analyse the outputs.
The algorithms used are:
* Support Vector Machine
* K-nearest neighbour
* Decision Tree

Contributors:
* Yusri Al-Sanaani​ 
* Hetvi Soni​ 
* Tavleen Kour​ 
* Immanuella Iyawe
