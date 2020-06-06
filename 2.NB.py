#Naive Bayes with Multiple Labels

'''
In model building part, you can use wine dataset which is a very famous multi-class classification problem. "This dataset is the result of a 
chemical analysis of wines grown in the same region in Italy but derived from three different cultivars." (UC Irvine)

Dataset comprises of 13 features (alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, 
proanthocyanins, color_intensity, hue, od280/od315_of_diluted_wines, proline) and type of wine cultivar. This data has three type of wine 
Class_0, Class_1, and Class_3. Here you can build a model to classify the type of wine.

The dataset is available in the scikit-learn library.
'''

#loading the data
#Import scikit-learn dataset library
from sklearn import datasets

#Load dataset
wine = datasets.load_wine()

#Exploring data
# print the names of the 13 features
print ("Features: ", wine.feature_names)

# print the label type of wine(class_0, class_1, class_2)
print ("Labels: ", wine.target_names)

# print data(feature)shape
print(wine.data.shape)

# print the wine data features (top 5 records)
print (wine.data[0:5])

# print the wine labels (0:Class_0, 1:class_2, 2:class_2)
print (wine.target)

#Splitting data

# Import train_test_split function
from sklearn.cross_validation import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,random_state=109) # 70% training and 30% test

#Model generation

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

#Evaluating model

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))