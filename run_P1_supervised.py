# P1 Supervised Learning

# Method 1: Decision Tree
# Method 2: Neural Network
# Method 3: Boosting
# Method 4: Support Vector Machine SVM
# Method 5: k-nearest neighbor

# https://scikit-learn.org/stable/modules/tree.html
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
# https://scikit-learn.org/stable/modules/svm.html
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

# INPUT DATA SET
# Set 1: https://www.openml.org/d/4534
# Phishing Websites
# Set 2: https://www.openml.org/d/38
# Thyroid disease records

import pandas as pd
import numpy as np
import math 
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import train_test_split, learning_curve, train_test_split
from sklearn import metrics # Calculate result accuracy
from sklearn.model_selection import ShuffleSplit, GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import TREE_LEAF
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer


if (0):
	# header: SFH,popUpWidnow,SSLfinal_State,Request_URL,URL_of_Anchor,web_traffic,URL_Length,age_of_domain,having_IP_Address,Result
	col_names  = ['SFH','popUpWidnow','SSLfinal_State','Request_URL','URL_of_Anchor','web_traffic','URL_Length','age_of_domain','having_IP_Address','Result']
	pima = pd.read_csv("PhishingData.csv", header=None, names=col_names)

	feature_cols = ['SFH','popUpWidnow','SSLfinal_State','Request_URL','URL_of_Anchor','web_traffic','URL_Length','age_of_domain','having_IP_Address']
	X = pima[feature_cols].to_numpy() # Features
	X = X[1:]
	y = (pima.Result).to_numpy() # Target variable
	y = y[1:]


	# Split dataset into training set and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
	print ('\nMethod 1: Decision Tree')
	clf = DecisionTreeClassifier()

	# For pruning, set max depth to avoid overfit
	#clf = DecisionTreeClassifier(max_depth=3)
	clf = clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

	clf = DecisionTreeClassifier(max_depth=3)
	clf = clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



	# Method 2: NN
	print ('\nMethod 2: Neural Network')
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=40)
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


	# Method 3: Boosting
	print ('\nMethod 3: Boosting')
	clf = AdaBoostClassifier(n_estimators=100, random_state=0)
	clf = clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


	# Method 4: Support Vector Machine SVM
	print ('\nMethod 4: Support Vector Machine SVM')
	clf = svm.SVC()
	clf = clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


	# MMethod 5: k-nearest neighbor
	print ('\nMethod 5: k-nearest neighbor')
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh = neigh.fit(X_train,y_train)
	y_pred = neigh.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
	
# Dataset 2	
if (0):
    a = 0
    sick = pd.read_csv("dataset_38_sick_cleanup.csv")
    sick_nonan = sick.dropna()

    X_pre = sick_nonan.loc[:,sick_nonan.columns != 'Class']
    print (X_pre.head())
    X = X_pre.to_numpy() # Features
    y = (sick_nonan.Class).to_numpy() # Target variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    print ('\nMethod 1: Decision Tree')    
    clf = DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
    
    
    # Method 2: NN
    print ('\nMethod 2: Neural Network')
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=40)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
    
    # Method 3: Boosting
    print ('\nMethod 3: Boosting')
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
    
    # Method 4: Support Vector Machine SVM
    print ('\nMethod 4: Support Vector Machine SVM')
    clf = svm.SVC()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))    
    
    print ('\nMethod 5: k-nearest neighbor')
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh = neigh.fit(X_train,y_train)
    y_pred = neigh.predict(X_test)
    print(" Accuracy:",metrics.accuracy_score(y_test, y_pred))    
    
    

def prune_index(inner_tree, index, threshold):
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are shildren, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)

 
def run_A_prep():
    col_names  = ['SFH','popUpWidnow','SSLfinal_State','Request_URL','URL_of_Anchor','web_traffic','URL_Length','age_of_domain','having_IP_Address','Result']
    pima = pd.read_csv("PhishingData.csv", header=None, names=col_names)

    feature_cols = ['SFH','popUpWidnow','SSLfinal_State','Request_URL','URL_of_Anchor','web_traffic','URL_Length','age_of_domain','having_IP_Address']
    X = pima[feature_cols].to_numpy() # Features
    X = X[1:]
    y = (pima.Result).to_numpy() # Target variable
    y = y[1:]

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
	
    return X_train, X_test, y_train, y_test
    
def run_A1(X_train, X_test, y_train, y_test):
    print ('\nMethod 1: Decision Tree')
    
    if (0): # Find best set up
        
        dtree = DecisionTreeClassifier(max_depth=44)
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        best_clf = GridSearchCV(estimator=dtree, cv=cv, param_grid=dict(max_depth=[5,10,20,40,100]), scoring='accuracy', n_jobs=-1)
        best_clf.fit(X_train, y_train)
        print(best_clf.best_estimator_)
        # RESULT: DecisionTreeClassifier(max_depth=10)

    
    if (0): # LEARNING CURVE
        clf = DecisionTreeClassifier(max_depth=3)
        clf = clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        prune_index(clf.tree_, 0, 5)
        print (len(X_train))
        train_sizes = np.linspace(1, len(X_train)*0.79, 5, dtype=int)
        print (train_sizes)
        train_sizes, train_scores, validation_scores = learning_curve(clf, X=X_train, y=y_train, train_sizes=(train_sizes), cv=5, scoring='accuracy', n_jobs=-1)
        plt.figure(1)
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
        plt.plot(train_sizes, validation_scores.mean(axis=1), label='Validating')
        plt.ylabel('Accuracy')
        plt.xlabel('Training Size')
        plt.title('Dataset A: PhishingWebsite - Decision Tree Learning Curve')
        plt.legend()
        plt.grid()
    
    
    if (0): # ACC vs MAX_DEPTH
        x = range(1, 100, 10)
        accuracy = np.zeros(len(x))
        index = 0
        for depth in x:
            print (depth)
            clf = DecisionTreeClassifier(max_depth=depth)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        plt.figure(2)
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('max_depth')
        plt.title('Dataset A: PhishingWebsite - Decision Tree Accuracy vs Max_depth')
        plt.legend()
        plt.grid()   
        
    if (0):  # ACC vs min_samples_leaf
        x = range(1, 20, 2)
        accuracy = np.zeros(len(x))
        index = 0
        for num in x:
            print (num)
            clf = DecisionTreeClassifier(min_samples_leaf =num)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        ax = plt.figure(3).gca()
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('min_samples_leaf ')
        plt.title('Dataset A: PhishingWebsite - Decision Tree Accuracy vs Min_samples_leaf ')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.grid()    

    if (0):  # ACC vs max_features
        x = range(1, 10, 1)
        accuracy = np.zeros(len(x))
        index = 0
        for num in x:
            print (num)
            clf = DecisionTreeClassifier(max_features =num)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        ax = plt.figure(5).gca()
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('max_features ')
        plt.title('Dataset A: PhishingWebsite - Decision Tree Accuracy vs Max_features ')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.grid()  

    if (0):  # Feature Importance
        
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train,y_train)
        importances = clf.feature_importances_
    #    indices = np.argsort(importances)[::-1]
        feature_cols = ['SFH','popUpWidnow','SSLfinal_State','Request_URL','URL_of_Anchor','web_traffic','URL_Length','age_of_domain','having_IP_Address']
    #    feature_names = [X_train.columns[indices[n]] for n in range(X_train.shape[1])]
        fig = plt.figure(5)
        plt.title("Dataset A: PhishingWebsite - Decision Tree - Feature Importances")
        plt.bar(range(X_train.shape[1]), importances, align="center")
        plt.xticks(range(X_train.shape[1]), feature_cols, rotation='70')
    #    plt.xlim([-1, X_train.shape[1]])      
        plt.ylabel('Feature Importance Value')        
        plt.grid()  
        fig.tight_layout()

 
    clf = DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))   
    
    
def run_A2(X_train, X_test, y_train, y_test):    
    # Method 2: NN
    print ('\nMethod 2: Neural Network')
    
    if (0): # Find best Set up
        nn = MLPClassifier(max_iter=1000)
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=5295468)
        best_clf = GridSearchCV(estimator=nn, cv=cv, param_grid=dict(activation=['logistic','relu'], solver=['adam','sgd'], hidden_layer_sizes=[(x,) for x in range(2,20,2)]), scoring='accuracy')
        best_clf.fit(X_train, y_train)
        print(best_clf.best_estimator_)
        # FAILED TO RUN
    
    if (0): # Learning curve
        clf = MLPClassifier()
        train_sizes = np.linspace(1, len(X_train)*0.79, 5, dtype=int)
        train_sizes, train_scores, validation_scores = learning_curve(clf, X=X_train, y=y_train, train_sizes=(train_sizes), cv=5, scoring='accuracy', n_jobs=-1)
        plt.figure(1)
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
        plt.plot(train_sizes, validation_scores.mean(axis=1), label='Validating')
        plt.ylabel('Accuracy')
        plt.xlabel('Training Size')
        plt.title('Dataset A: PhishingWebsite - Neural Network Learning Curve')
        plt.legend()
        plt.grid()        
        
    
    if (0): #hidden_layer_sizes default=(100,)
        x = range(1, 200, 10)
        accuracy = np.zeros(len(x))
        index = 0
        for depth in x:
            print (depth)
            clf = MLPClassifier()
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        plt.figure(2)
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('hidden_layer_sizes')
        plt.title('Dataset A: PhishingWebsite - Neural Network Accuracy vs hidden_layer_sizes')
        plt.legend()
        plt.grid()  
        
        
        
    if (0): #learning_rates {‘constant’, ‘invscaling’, ‘adaptive’}, 
        x = range(1, 200, 10)
        learning_rates = ['constant', 'invscaling', 'adaptive']
        accuracy = np.zeros(len(learning_rates))
        index = 0
        for type in learning_rates:
            print (type)
            clf = MLPClassifier(learning_rate = type)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        fig = plt.figure(2)
        plt.bar(range(len(accuracy)) ,accuracy, align="center")
        plt.xticks(range(len(accuracy)), learning_rates, rotation='vertical')
        plt.ylabel('Accuracy')
        plt.xlabel('learning_rates')
        plt.title('Dataset A: PhishingWebsite - Neural Network Accuracy vs learning learning_rates')
        plt.grid() 
        fig.tight_layout()

    if (0): # batch_size: default: min(200, n_samples)
        x = range(1, 500, 25)
        accuracy = np.zeros(len(x))
        index = 0
        for num in x:
            print (num)
            clf = MLPClassifier(batch_size = num)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        fig = plt.figure(5)
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('batch_size ')
        plt.title('Dataset A: PhishingWebsite - Neural Network Accuracy vs batch_size ')
        plt.grid()    
        fig.tight_layout()        
    
    
    if (1): # max_iter int, default=200
        x = range(1, 1000, 20)
        accuracy = np.zeros(len(x))
        index = 0
        for num in x:
            print (num)
            clf = MLPClassifier(max_iter = num)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        fig = plt.figure(4)
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('max_iter ')
        plt.title('Dataset B: SicknessDetection - Neural Network Accuracy vs max_iter ')
        plt.grid()    
        fig.tight_layout()    
    
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=40)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


def run_A3(X_train, X_test, y_train, y_test):
	# Method 3: AdaBoosting
    
    if (0): # Learning curve
        clf = AdaBoostClassifier()
        train_sizes = np.linspace(1, len(X_train)*0.79, 5, dtype=int)
        train_sizes, train_scores, validation_scores = learning_curve(clf, X=X_train, y=y_train, train_sizes=(train_sizes), cv=5, scoring='accuracy', n_jobs=-1)
        plt.figure(1)
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
        plt.plot(train_sizes, validation_scores.mean(axis=1), label='Validating')
        plt.ylabel('Accuracy')
        plt.xlabel('Training Size')
        plt.title('Dataset A: PhishingWebsite - Adaboost Learning Curve')
        plt.legend()
        plt.grid()  
        
    if (0): #learning_rate
        x =  np.arange(0.1, 3, 0.1)
        accuracy = np.zeros(len(x))
        index = 0
        for num in x:
            print (num)
            clf = AdaBoostClassifier(learning_rate = num)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        fig = plt.figure(5)
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('learning_rate ')
        plt.title('Dataset A: PhishingWebsite - Adaboost Accuracy vs learning_rate ')
        plt.grid()    
        fig.tight_layout()  

    if (0): #n_estimators
        x =  np.arange(1, 100, 5)
        accuracy = np.zeros(len(x))
        index = 0
        for num in x:
            print (num)
            clf = AdaBoostClassifier(n_estimators = num)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        fig = plt.figure(5)
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('n_estimators ')
        plt.title('Dataset A: PhishingWebsite - Adaboost Accuracy vs n_estimators ')
        plt.grid()    
        fig.tight_layout() 
        
    if (0): #Feature Importance
        clf = AdaBoostClassifier()
        clf = clf.fit(X_train,y_train)
        importances = clf.feature_importances_
        feature_cols = ['SFH','popUpWidnow','SSLfinal_State','Request_URL','URL_of_Anchor','web_traffic','URL_Length','age_of_domain','having_IP_Address']
        fig = plt.figure(5)
        plt.title("Dataset A: PhishingWebsite - Adaboost - Feature Importances")
        plt.bar(range(X_train.shape[1]), importances, align="center")
        plt.xticks(range(X_train.shape[1]), feature_cols, rotation='70') 
        plt.ylabel('Feature Importance Value')        
        plt.grid()  
        fig.tight_layout()     
    
    
    print ('\nMethod 3: Boosting')
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

def run_A4(X_train, X_test, y_train, y_test):
	
    if (0): # LEARNING CURVE
        clf = svm.SVC()

        print (len(X_train))
        train_sizes = np.linspace(1, len(X_train)*0.79, 10, dtype=int)
        print (train_sizes)
        train_sizes, train_scores, validation_scores = learning_curve(clf, X=X_train, y=y_train, train_sizes=(train_sizes), cv=5, scoring='accuracy', n_jobs=-1)
        plt.figure(1)
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
        plt.plot(train_sizes, validation_scores.mean(axis=1), label='Validating')
        plt.ylabel('Accuracy')
        plt.xlabel('Training Size')
        plt.title('Dataset A: PhishingWebsite - SVM Learning Curve')
        plt.legend()
        plt.grid()
    
    if(0): # C param default = 1 Regularization parameter.
        x = np.arange(0.1, 10, 0.1)
        accuracy = np.zeros(len(x))
        index = 0
        for val in x:
            print (val)
            clf = svm.SVC(C=val)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        plt.figure(2)
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('C_param')
        plt.title('Dataset A: PhishingWebsite - SVM Accuracy vs C_param')
        plt.legend()
        plt.grid()  
        
    if(0): # degree default=3 Degree of the polynomial kernel function 
        x = np.arange(1, 20, 1)
        accuracy = np.zeros(len(x))
        index = 0
        for val in x:
            print (val)
            clf = svm.SVC(degree=val)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        ax = plt.figure(2).gca()
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('degree')
        plt.title('Dataset A: PhishingWebsite - SVM Accuracy vs degree')
        plt.legend()
        plt.grid()  
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    if(0): # max_iter int, default=-1
        x = np.arange(1, 250, 5)
        accuracy = np.zeros(len(x))
        index = 0
        for val in x:
            print (val)
            clf = svm.SVC(max_iter=val)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        ax = plt.figure(2).gca()
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('max_iter')
        plt.title('Dataset A: PhishingWebsite - SVM Accuracy vs max_iter')
        plt.legend()
        plt.grid() 
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    # Method 4: Support Vector Machine SVM
    print ('\nMethod 4: Support Vector Machine SVM')
    clf = svm.SVC()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


def run_A5(X_train, X_test, y_train, y_test):
	
    if (0): # LEARNING CURVE
        clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
        train_sizes = np.linspace(10, len(X_train)*0.79, 5, dtype=int)
        print (train_sizes)
        train_sizes, train_scores, validation_scores = learning_curve(clf, X=X_train, y=y_train, train_sizes=(train_sizes), cv=5, scoring='accuracy', n_jobs=-1)
        plt.figure(1)
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
        plt.plot(train_sizes, validation_scores.mean(axis=1), label='Validating')
        plt.ylabel('Accuracy')
        plt.xlabel('Training Size')
        plt.title('Dataset A: PhishingWebsite - kNN Learning Curve')
        plt.legend()
        plt.grid()
    
        
    if(0): # n_neighbors   int, default=5
        x = np.arange(1, 50, 1)
        accuracy = np.zeros(len(x))
        index = 0
        for val in x:
            print (val)
            clf = KNeighborsClassifier(n_neighbors = val)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        ax = plt.figure(2).gca()
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('n_neighbors')
        plt.title('Dataset A: PhishingWebsite - kNN - Accuracy vs n_neighbors')
        plt.legend()
        plt.grid() 
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))    
    
   
    if(0): # leaf_size int, default=30
        x = np.arange(1, 100, 10)
        accuracy = np.zeros(len(x))
        index = 0
        for val in x:
            print (val)
            clf = KNeighborsClassifier(leaf_size = val)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        ax = plt.figure(2).gca()
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('leaf_size')
        plt.title('Dataset A: PhishingWebsite - kNN - Accuracy vs leaf_size')
        plt.legend()
        plt.grid() 
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))   
   
    # MMethod 5: k-nearest neighbor
    print ('\nMethod 5: k-nearest neighbor')
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh = neigh.fit(X_train,y_train)
    y_pred = neigh.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    


def run_B_prep():
    sick = pd.read_csv("dataset_38_sick_cleanup.csv")
    sick_nonan = sick.dropna()

    X_pre = sick_nonan.loc[:,sick_nonan.columns != 'Class']
    print (X_pre.head())
    X = X_pre.to_numpy() # Features
    y = (sick_nonan.Class).to_numpy() # Target variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    print (X_train.shape[1])
	
    return X_train, X_test, y_train, y_test
    
def run_B1(X_train, X_test, y_train, y_test):
    print ('\nMethod 1: Decision Tree')
    
    if (0): # LEARNING CURVE
        clf = DecisionTreeClassifier(max_depth=3)
        clf = clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        prune_index(clf.tree_, 0, 5)
        print (len(X_train))
        train_sizes = np.linspace(1, len(X_train)*0.79, 5, dtype=int)
        print (train_sizes)
        train_sizes, train_scores, validation_scores = learning_curve(clf, X=X_train, y=y_train, train_sizes=(train_sizes), cv=5, scoring='accuracy', n_jobs=-1)
        plt.figure(1)
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
        plt.plot(train_sizes, validation_scores.mean(axis=1), label='Validating')
        plt.ylabel('Accuracy')
        plt.xlabel('Training Size')
        plt.title('Dataset B: SicknessDetection - Decision Tree Learning Curve')
        plt.legend()
        plt.grid()
    
    
    if (0): # ACC vs MAX_DEPTH
        x = range(1, 100, 10)
        accuracy = np.zeros(len(x))
        index = 0
        for depth in x:
            print (depth)
            clf = DecisionTreeClassifier(max_depth=depth)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        plt.figure(2)
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('max_depth')
        plt.title('Dataset B: SicknessDetection - Decision Tree Accuracy vs Max_depth')
        plt.legend()
        plt.grid()   
        
    if (0):  # ACC vs min_samples_leaf
        x = range(1, 20, 2)
        accuracy = np.zeros(len(x))
        index = 0
        for num in x:
            print (num)
            clf = DecisionTreeClassifier(min_samples_leaf =num)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        ax = plt.figure(3).gca()
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('min_samples_leaf ')
        plt.title('Dataset B: SicknessDetection - Decision Tree Accuracy vs Min_samples_leaf ')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.grid()    

    if (0):  # ACC vs max_features
        num_feats = X_train.shape[1]
        x = range(1, num_feats+1, 1)
        accuracy = np.zeros(len(x))
        index = 0
        for num in x:
            print (num)
            clf = DecisionTreeClassifier(max_features =num)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        ax = plt.figure(5).gca()
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('max_features ')
        plt.title('Dataset A: PhishingWebsite - Decision Tree Accuracy vs Max_features ')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.grid()  

    if (0):  # Feature Importance
        sick = pd.read_csv("dataset_38_sick_cleanup.csv")
        sick_nonan = sick.dropna()
        X_pre = sick_nonan.loc[:,sick_nonan.columns != 'Class']
        print (X_pre.columns)
        print (len(X_pre.columns))

        
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train,y_train)
        importances = clf.feature_importances_
    #    indices = np.argsort(importances)[::-1]
    #    feature_cols = ['SFH','popUpWidnow','SSLfinal_State','Request_URL','URL_of_Anchor','web_traffic','URL_Length','age_of_domain','having_IP_Address']
        feature_cols = X_pre.columns
        fig = plt.figure(6)
        plt.title("Dataset B: SicknessDetection - Decision Tree - Feature Importances")
        plt.bar(range(X_train.shape[1]), importances, align="center")
        plt.xticks(range(X_train.shape[1]), feature_cols, rotation='80')
    #    plt.xlim([-1, X_train.shape[1]])      
        plt.ylabel('Feature Importance Value')        
        plt.grid()  
        fig.tight_layout()
 
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))  

    
def run_B2(X_train, X_test, y_train, y_test):    
    # Method 2: NN
    print ('\nMethod 2: Neural Network')
    
    if (0): # Find best Set up
        nn = MLPClassifier(max_iter=1000)
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=5295468)
        best_clf = GridSearchCV(estimator=nn, cv=cv, param_grid=dict(activation=['logistic','relu'], solver=['adam','sgd'], hidden_layer_sizes=[(x,) for x in range(2,20,2)]), scoring='accuracy')
        best_clf.fit(X_train, y_train)
        print(best_clf.best_estimator_)
        # FAILED TO RUN
    
    if (0): # Learning curve
        clf = MLPClassifier()
        train_sizes = np.linspace(1, len(X_train)*0.79, 5, dtype=int)
        train_sizes, train_scores, validation_scores = learning_curve(clf, X=X_train, y=y_train, train_sizes=(train_sizes), cv=5, scoring='accuracy', n_jobs=-1)
        plt.figure(1)
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
        plt.plot(train_sizes, validation_scores.mean(axis=1), label='Validating')
        plt.ylabel('Accuracy')
        plt.xlabel('Training Size')
        plt.title('Dataset B: SicknessDetection - Neural Network Learning Curve')
        plt.legend()
        plt.grid()        
        
    
    if (0): #hidden_layer_sizes default=(100,)
        x = range(1, 200, 10)
        accuracy = np.zeros(len(x))
        index = 0
        for depth in x:
            print (depth)
            clf = MLPClassifier()
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        plt.figure(2)
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('hidden_layer_sizes')
        plt.title('Dataset B: SicknessDetection - Neural Network Accuracy vs hidden_layer_sizes')
        plt.legend()
        plt.grid()  
        
        
        
    if (0): #learning_rates {‘constant’, ‘invscaling’, ‘adaptive’}, 
        x = range(1, 200, 10)
        learning_rates = ['constant', 'invscaling', 'adaptive']
        accuracy = np.zeros(len(learning_rates))
        index = 0
        for type in learning_rates:
            print (type)
            clf = MLPClassifier(learning_rate = type)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        fig = plt.figure(3)
        plt.bar(range(len(accuracy)) ,accuracy, align="center")
        plt.xticks(range(len(accuracy)), learning_rates, rotation='vertical')
        plt.ylabel('Accuracy')
        plt.xlabel('learning_rates')
        plt.title('Dataset B: SicknessDetection - Neural Network Accuracy vs learning learning_rates')
        plt.grid() 
        fig.tight_layout()

    if (0): # batch_size: default: min(200, n_samples)
        x = range(1, 500, 25)
        accuracy = np.zeros(len(x))
        index = 0
        for num in x:
            print (num)
            clf = MLPClassifier(batch_size = num)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        fig = plt.figure(4)
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('batch_size ')
        plt.title('Dataset B: SicknessDetection - Neural Network Accuracy vs batch_size ')
        plt.grid()    
        fig.tight_layout()        

    if (1): # max_iter int, default=200
        x = range(1, 1000, 20)
        accuracy = np.zeros(len(x))
        index = 0
        for num in x:
            print (num)
            clf = MLPClassifier(max_iter = num)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        fig = plt.figure(4)
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('max_iter ')
        plt.title('Dataset B: SicknessDetection - Neural Network Accuracy vs max_iter ')
        plt.grid()    
        fig.tight_layout()

    
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=40)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


def run_B3(X_train, X_test, y_train, y_test):
	# Method 3: AdaBoosting
    
    if (0): # Learning curve
        clf = AdaBoostClassifier()
        train_sizes = np.linspace(1, len(X_train)*0.79, 5, dtype=int)
        train_sizes, train_scores, validation_scores = learning_curve(clf, X=X_train, y=y_train, train_sizes=(train_sizes), cv=5, scoring='accuracy', n_jobs=-1)
        plt.figure(5)
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
        plt.plot(train_sizes, validation_scores.mean(axis=1), label='Validating')
        plt.ylabel('Accuracy')
        plt.xlabel('Training Size')
        plt.title('Dataset B: SicknessDetection - Adaboost Learning Curve')
        plt.legend()
        plt.grid()  
        
    if (0): #learning_rate
        x =  np.arange(0.1, 3, 0.1)
        accuracy = np.zeros(len(x))
        index = 0
        for num in x:
            print (num)
            clf = AdaBoostClassifier(learning_rate = num)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        fig = plt.figure(6)
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('learning_rate ')
        plt.title('Dataset B: SicknessDetection - Adaboost Accuracy vs learning_rate ')
        plt.grid()    
        fig.tight_layout()  

    if (0): #n_estimators
        x =  np.arange(1, 100, 5)
        accuracy = np.zeros(len(x))
        index = 0
        for num in x:
            print (num)
            clf = AdaBoostClassifier(n_estimators = num)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        fig = plt.figure(7)
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('n_estimators ')
        plt.title('Dataset B: SicknessDetection - Adaboost Accuracy vs n_estimators ')
        plt.grid()    
        fig.tight_layout() 
        
    if (1): #Feature Importance
        
        sick = pd.read_csv("dataset_38_sick_cleanup.csv")
        sick_nonan = sick.dropna()
        X_pre = sick_nonan.loc[:,sick_nonan.columns != 'Class']
        
        clf = AdaBoostClassifier()
        clf = clf.fit(X_train,y_train)
        importances = clf.feature_importances_
        #feature_cols = ['SFH','popUpWidnow','SSLfinal_State','Request_URL','URL_of_Anchor','web_traffic','URL_Length','age_of_domain','having_IP_Address']
        feature_cols = X_pre.columns
        fig = plt.figure(5)
        plt.title("Dataset B: SicknessDetection - Adaboost - Feature Importances")
        plt.bar(range(X_train.shape[1]), importances, align="center")
        plt.xticks(range(X_train.shape[1]), feature_cols, rotation='70') 
        plt.ylabel('Feature Importance Value')        
        plt.grid()  
        fig.tight_layout()     
    
    
    print ('\nMethod 3: Boosting')
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

def run_B4(X_train, X_test, y_train, y_test):
	
    if (1): # LEARNING CURVE
        clf = svm.SVC()

        print (len(X_train))
        train_sizes = np.linspace(1, len(X_train)*0.79, 10, dtype=int)
        print (train_sizes)
        train_sizes, train_scores, validation_scores = learning_curve(clf, X=X_train, y=y_train, train_sizes=(train_sizes), cv=5, scoring='accuracy', n_jobs=-1)
        plt.figure(8)
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
        plt.plot(train_sizes, validation_scores.mean(axis=1), label='Validating')
        plt.ylabel('Accuracy')
        plt.xlabel('Training Size')
        plt.title('Dataset B: SicknessDetection - SVM Learning Curve')
        plt.legend()
        plt.grid()
    
    if(1): # C param default = 1 Regularization parameter.
        x = np.arange(0.1, 10, 0.1)
        accuracy = np.zeros(len(x))
        index = 0
        for val in x:
            print (val)
            clf = svm.SVC(C=val)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        plt.figure(9)
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('C_param')
        plt.title('Dataset B: SicknessDetection - SVM Accuracy vs C_param')
        plt.legend()
        plt.grid()  
        
    if(1): # degree default=3 Degree of the polynomial kernel function 
        x = np.arange(1, 20, 1)
        accuracy = np.zeros(len(x))
        index = 0
        for val in x:
            print (val)
            clf = svm.SVC(degree=val)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        ax = plt.figure(10).gca()
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('degree')
        plt.title('Dataset B: SicknessDetection - SVM Accuracy vs degree')
        plt.legend()
        plt.grid()  
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    if(1): # max_iter int, default=-1
        x = np.arange(1, 250, 5)
        accuracy = np.zeros(len(x))
        index = 0
        for val in x:
            print (val)
            clf = svm.SVC(max_iter=val)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        ax = plt.figure(11).gca()
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('max_iter')
        plt.title('Dataset B: SicknessDetection - SVM Accuracy vs max_iter')
        plt.legend()
        plt.grid() 
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    # Method 4: Support Vector Machine SVM
    print ('\nMethod 4: Support Vector Machine SVM')
    clf = svm.SVC()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


def run_B5(X_train, X_test, y_train, y_test):
	
    if (1): # LEARNING CURVE
        clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
        train_sizes = np.linspace(10, len(X_train)*0.79, 5, dtype=int)
        print (train_sizes)
        train_sizes, train_scores, validation_scores = learning_curve(clf, X=X_train, y=y_train, train_sizes=(train_sizes), cv=5, scoring='accuracy', n_jobs=-1)
        plt.figure(12)
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
        plt.plot(train_sizes, validation_scores.mean(axis=1), label='Validating')
        plt.ylabel('Accuracy')
        plt.xlabel('Training Size')
        plt.title('Dataset B: SicknessDetection - kNN Learning Curve')
        plt.legend()
        plt.grid()
    
        
    if(1): # n_neighbors   int, default=5
        x = np.arange(1, 50, 1)
        accuracy = np.zeros(len(x))
        index = 0
        for val in x:
            print (val)
            clf = KNeighborsClassifier(n_neighbors = val)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        ax = plt.figure(13).gca()
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('n_neighbors')
        plt.title('Dataset B: SicknessDetection - kNN - Accuracy vs n_neighbors')
        plt.legend()
        plt.grid() 
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))    
    
   
    if(1): # leaf_size int, default=30
        x = np.arange(1, 100, 10)
        accuracy = np.zeros(len(x))
        index = 0
        for val in x:
            print (val)
            clf = KNeighborsClassifier(leaf_size = val)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            accuracy[index] = metrics.accuracy_score(y_test, y_pred)
            index = index + 1 
        
        ax = plt.figure(14).gca()
        plt.plot(x ,accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel('leaf_size')
        plt.title('Dataset B: SicknessDetection - kNN - Accuracy vs leaf_size')
        plt.legend()
        plt.grid() 
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))   
   
    # MMethod 5: k-nearest neighbor
    print ('\nMethod 5: k-nearest neighbor')
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh = neigh.fit(X_train,y_train)
    y_pred = neigh.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))    

#X_train, X_test, y_train, y_test = run_A_prep()  
#run_A1(X_train, X_test, y_train, y_test)
#run_A2(X_train, X_test, y_train, y_test)
#run_A3(X_train, X_test, y_train, y_test)
#run_A4(X_train, X_test, y_train, y_test)
#run_A5(X_train, X_test, y_train, y_test)  
    
    
#X_train, X_test, y_train, y_test = run_B_prep()  
#run_B1(X_train, X_test, y_train, y_test)
#run_B2(X_train, X_test, y_train, y_test)
#run_B3(X_train, X_test, y_train, y_test)
#run_B4(X_train, X_test, y_train, y_test)
#run_B5(X_train, X_test, y_train, y_test) 



if (0): # Runtime Dataset 1
    # Runtime analysis
    X_train, X_test, y_train, y_test = run_A_prep() 
    print ('\nMethod 1: Decision Tree') 
    t0= time.clock()
    clf = DecisionTreeClassifier(max_depth=21, min_samples_leaf=1, max_features=5)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    t1 = time.clock() - t0
    print("Time elapsed: ", t1)

    # Method 2: NN
    print ('\nMethod 2: Neural Network')
    t0= time.clock()
    clf = MLPClassifier(hidden_layer_sizes=(23,), learning_rate='adaptive', batch_size=16)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    t1 = time.clock() - t0
    print("Time elapsed: ", t1)


    # Method 3: Boosting
    print ('\nMethod 3: Boosting')
    t0= time.clock()
    clf = AdaBoostClassifier(learning_rate=1.2 , n_estimators=52)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    t1 = time.clock() - t0
    print("Time elapsed: ", t1)


    # Method 4: Support Vector Machine SVM
    print ('\nMethod 4: Support Vector Machine SVM')
    t0= time.clock()
    clf = svm.SVC(C=8.2 , max_iter=180)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    t1 = time.clock() - t0
    print("Time elapsed: ", t1)


    # MMethod 5: k-nearest neighbor
    print ('\nMethod 5: k-nearest neighbor')
    t0= time.clock()
    neigh = KNeighborsClassifier(n_neighbors= 7, leaf_size=31)
    neigh = neigh.fit(X_train,y_train)
    y_pred = neigh.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    t1 = time.clock() - t0
    print("Time elapsed: ", t1)
    
    
    

if (0): # Runtime Dataset 1
    # Runtime analysis
    X_train, X_test, y_train, y_test = run_B_prep() 
    print ('\nMethod 1: Decision Tree') 
    t0= time.clock()
    clf = DecisionTreeClassifier(max_depth=21, min_samples_leaf=3, max_features=28)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    t1 = time.clock() - t0
    print("Time elapsed: ", t1)

    # Method 2: NN
    print ('\nMethod 2: Neural Network')
    t0= time.clock()
    clf = MLPClassifier(hidden_layer_sizes=(12,), learning_rate='invscaling', batch_size=150)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    t1 = time.clock() - t0
    print("Time elapsed: ", t1)


    # Method 3: Boosting
    print ('\nMethod 3: Boosting')
    t0= time.clock()
    clf = AdaBoostClassifier(learning_rate=1.5 , n_estimators=65)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    t1 = time.clock() - t0
    print("Time elapsed: ", t1)


    # Method 4: Support Vector Machine SVM
    print ('\nMethod 4: Support Vector Machine SVM')
    t0= time.clock()
    clf = svm.SVC(C=5 , max_iter=200)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    t1 = time.clock() - t0
    print("Time elapsed: ", t1)


    # MMethod 5: k-nearest neighbor
    print ('\nMethod 5: k-nearest neighbor')
    t0= time.clock()
    neigh = KNeighborsClassifier(n_neighbors= 7, leaf_size=10)
    neigh = neigh.fit(X_train,y_train)
    y_pred = neigh.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    t1 = time.clock() - t0
    print("Time elapsed: ", t1)


    



if(1): # Compare and contrast learning curve
    X_train, X_test, y_train, y_test = run_B_prep() 
    clf = DecisionTreeClassifier(max_depth=3)
    train_sizes = np.linspace(1, len(X_train)*0.79, 5, dtype=int)
    train_sizes, train_scores, validation_scores = learning_curve(clf, X=X_train, y=y_train, train_sizes=(train_sizes), scoring='accuracy', n_jobs=-1)
    plt.figure(1)
    plt.plot(train_sizes, train_scores.mean(axis=1), 'k--', label='Training M1-Decision Tree')
    plt.plot(train_sizes, validation_scores.mean(axis=1), 'k', label='Validating M1-Decision Tree')   

    clf = MLPClassifier()
    train_sizes = np.linspace(1, len(X_train)*0.79, 5, dtype=int)
    train_sizes, train_scores, validation_scores = learning_curve(clf, X=X_train, y=y_train, train_sizes=(train_sizes), scoring='accuracy', n_jobs=-1)
    plt.plot(train_sizes, train_scores.mean(axis=1), 'b--', label='Training M2-Neural Network')
    plt.plot(train_sizes, validation_scores.mean(axis=1), 'b', label='Validating M2-Neural Network')   
    
    clf = AdaBoostClassifier()
    train_sizes = np.linspace(1, len(X_train)*0.79, 5, dtype=int)
    train_sizes, train_scores, validation_scores = learning_curve(clf, X=X_train, y=y_train, train_sizes=(train_sizes), scoring='accuracy', n_jobs=-1)
    plt.plot(train_sizes, train_scores.mean(axis=1), 'm--', label='Training M3-AdaBoost')
    plt.plot(train_sizes, validation_scores.mean(axis=1), 'm', label='Validating M3-AdaBoost')   
    
    clf = svm.SVC()
    train_sizes = np.linspace(1, len(X_train)*0.79, 5, dtype=int)
    train_sizes, train_scores, validation_scores = learning_curve(clf, X=X_train, y=y_train, train_sizes=(train_sizes), scoring='accuracy', n_jobs=-1)
    plt.plot(train_sizes, train_scores.mean(axis=1), 'g--', label='Training M4-SVM')
    plt.plot(train_sizes, validation_scores.mean(axis=1), 'g', label='Validating M4-SVM')   
    
    clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
    train_sizes = np.linspace(5, len(X_train)*0.79-4, 5, dtype=int)
    train_sizes, train_scores, validation_scores = learning_curve(clf, X=X_train, y=y_train, train_sizes=(train_sizes), scoring='accuracy', n_jobs=-1)
    plt.plot(train_sizes, train_scores.mean(axis=1), 'r--', label='Training M5-kNN')
    plt.plot(train_sizes, validation_scores.mean(axis=1), 'r', label='Validating M5-kNN')        
 
    plt.ylabel('Accuracy')
    plt.xlabel('Training Size')
    plt.title('Dataset B: SicknessDetection - All Learning Curve')
    plt.legend()
    plt.grid()

plt.show()    