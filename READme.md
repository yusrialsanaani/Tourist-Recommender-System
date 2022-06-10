# Project Title : Tourist Recommendation System

**A Tourist attractions recommendation system that suggests tourists to the attractions to visit based on their choices.**

**Description**

Implementing a recommendation system based on content and collaborative filtering, leveraging many algorithms to provide different prediction features such as providing recommendations based on:
- Attraction type
- Targeted province
- The best time to visit the attraction
- Multiple combined features

Resulting with multi functional recommender system that based on content, context and ratings. Then choosing the best performing model and implemented a chatbot for it. Therefore, the recommender system turned out to be very useful and convenient to use for the tourists.

## **The Recommender System (RS) Framework**

The Recommender System (RS) Framework is shown below:

![image](https://user-images.githubusercontent.com/89004966/171489035-854e7d15-908c-4ce1-998a-3b31ed21c140.png)

**Models Created:**
- Content based model based on:
  - The type of location.
  - The province of location.
  - The best time to visit.
  - Multiple combined features.
  - K-means clustering algorithm for unsupervised learning.
- Collaborative filtering model based on: 
  - Multiple classification algorithms in Scikit-learn Library for supervised learning.
  - Multiple algorithms in Surprise Library

## **Dataset Preparation:**

The dataset used in the trainging was scrapped from the trip advisor website. Then, the dataset has been processed and modified to get the desired dataset as per our requirements.

## **Content Based Filtering (CBF)**

The diagram below shows the design flow of CBF recommender system:

![image](https://user-images.githubusercontent.com/89004966/171489575-4b8a3a94-e3db-417e-a17a-696bb7e3f258.png)

### CBF Using Cosine Similarity
We created the following models based on cosine similarity: 
- Location Type
- Province
- Best time to visit
- Combined features where feature are location_type, provience, cost, best_time_to_visit, ratings

![image](https://user-images.githubusercontent.com/89004966/171489745-e4d5d405-6b40-495f-afce-3980425a2acb.png)

### **Results sample for content-based model for combined features:**

![image](https://user-images.githubusercontent.com/89004966/171489843-7649e001-e2e4-44cc-adcc-3f33effda1c2.png)



### **CBF Using Clustering**

The diagram below shows the design flow of CBF recommender system using Clustering:

![image](https://user-images.githubusercontent.com/89004966/171489982-5a053448-bd08-4f4f-a83f-24fa1fb1f691.png)


### **Results sample for CBF Using Clustering:**

![image](https://user-images.githubusercontent.com/89004966/171508794-9af35f63-0ee7-4e21-8997-25c22a82239d.png)

## Collaborative Filtering (CF)

The diagram below shows the design flow of CF recommender system:

![image](https://user-images.githubusercontent.com/89004966/171490169-9508b102-5ba4-4b32-a5fe-bf4e01a80d9f.png)

**CF Using Scikit Learn Library**

We performed classification for two cases: 
- Model 1: Using user id as labels
- Model 2: Using Location type id as labels

We choose model 2 as our model since it shows better results.

Algorithms used are: 
SVM, KNN and Decision Tree (Supervised learning)

**The Champion Model: SVM**


### **CF Using Surprise Library**

![image](https://user-images.githubusercontent.com/89004966/171490478-f1cfd09e-9180-4841-86bd-33754de41c04.png)


### **Results sample for CF:**

![image](https://user-images.githubusercontent.com/89004966/171490564-c0bbcae2-df65-4fe2-a1ce-4f28d7e8a49d.png)


## Chatbot Implementation

![image](https://user-images.githubusercontent.com/89004966/171490709-0c1c70b3-fb26-4d82-bf7a-e6bb443cf783.png)

### **Results sample using Chatbot:**

![image](https://user-images.githubusercontent.com/89004966/171490744-93733ebb-508a-4cb0-be8c-fde055a548fd.png)

![image](https://user-images.githubusercontent.com/89004966/171490761-29cf4865-6c2e-4992-bc1c-8e6232525ae8.png)



## Software used
* Python version 3.7
* Google Cobal for implementation of  recommender system
* Google Dialogflow for implementing the chatbot
* Jupyter notebook to integrate flask framework to integrate python code to the chatbot

## Libraries used: 
scikit-learn | re | nltk | pandas | numpy | seaborn | beautiful Soup | url open | requests | plotly | dash | matplot | surprise | flask


## Code references:

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

## The algorithms used are:
* Support Vector Machine
* K-nearest neighbour
* Decision Tree
