README.txt

INSTRUCTION ON HOW TO RUN THE CODE


HOW TO RUN: 
    python run_P1_supervised.py
    
    
    
REQUIREMENTS:
    - pandas 1.0.5
    - numpy 1.17.0
    - sklearn 0.23.2
    - matplotlib 3.2.2
    
    
CODE DESCRIPTION:    
    The code was divided into 2 parts base on the 2 chosen data sets:
    - Part 1: Phishing Websites
    - Part 2: Sickness Detection
    
    Running both part by setting the value inside of the if statement to be '1' for both sections. To turn off any section, set the value to be '0'.
    
    For each part, the program will run the main following sections. Each main section has subsections where I explore the parameter tunning:
    0. Reprocess data and divide data into training set and testing set
    1. Performing the Supervised Learning Method 1: Decision Tree
        A. Find the best set up
        B. Plot learning curve
        C. Accuracy vs max_depth
        D. Accuracy vs min_sample_leaf
        E. Accuracy vs max_features
        F. Feature Importance

    2. Performing the Supervised Learning Method 2: Neural Network
	A. Find the best set up
        B. Plot learning curve
        C. Accuracy vs hidden_layer_sizes
        D. Accuracy vs batch_size

    3. Performing the Supervised Learning Method 3: Boosting
	A. Plot learning curve
        B. Accuracy vs learning_rate
        C. Accuracy vs n_estimators
        D. Feature Importance

    4. Performing the Supervised Learning Method 4: Support Vector Machine SVM
        A. Plot learning curve
        B. Accuracy vs C param
        C. Accuracy vs degree
        D. Accuracy vs max_iter

    5. Performing the Supervised Learning Method 5: k-nearest neighbor
        A. Plot learning curve
        B. Accuracy vs n_neighbors
        C. Accuracy vs leaf_size


LINK TO DATASETS:
    - 1 Phishing Websites: https://www.openml.org/d/4534
    - 2 Sickness Detection: https://www.openml.org/d/38

REFERENCE WEBSITES:
    - https://scikit-learn.org/stable/modules/tree.html
    - https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    - https://scikit-learn.org/stable/modules/svm.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
