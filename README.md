# diagnostic_system
Implemented a diagnostic system to classify malignant cells

## Data
The three different features for each variable, with suffixes "_0", "_1", "_2" represent the mean, standard deviation and "worst" value over a set of samples, respectively. 

## Rule_based classifier
A rule-based classifier was implemented using an if-then rule based system. For each metric, we compared the mean metric of each person to the global mean (stored in threshold variables). 

### Methodology
According to Eickhoff, we implemented these rules:
1. abnormal_size = data['radius_0'] > threshold_size
    - Cells with a higher than average mean distance (thus are significantly larger) are classified as malignant
2. abnormal_shape = data['concavity_0'] < threshold_shape
    - Benign cells show limited variance while malignant cells can develop arbitrary structures, making it unsymmetrical. 
    - Cells with a lower higher severity of concave portions on its contour suggests that there could be arbitrary structures present
    - Cells with a higher that average concavity are classified as malignant 
3. abnormal_texture = data['texture_0'] > threshold_texture
    - The texture metric measures the standard variation of gray-scale values. 
    - A larger standard deviation indicates that there is more variation in colour, which indicate the presence of darker nuclei due to more DNA present in malignant cells.
4. abnormal_concavity = data['concave points_0'] > threshold_concavity
    - A higher than average number of concave points suggest the presence of many lumps
    - Cells with a higher than average number of concave points are classified as malignant
5. abnormal_homogeneity = data['fractal dimension_0'] < threshold_homogeneity 
    - The fractal dimension metric measures the complexity and irregularity of cell structures. 
    - Cells with a lower than average fractal dimension are smoother and are classified as malignant cells 
6. abnormal_width = data[smoothness_0] > threshold_perimeter
    - The smoothness metric measures the local variation in radius lengths
    - In a healthy tissue, cell arrangement is orderly while cancer cells are spread out. 
    - A larger local variation in radius lengths suggests that neighbouring cells vary in width, suggesting that they are spread out and have a disorderly arrangement
    - Cells with a higher than average local variation are classified as malignant 

### Evaluation
We used accuracy_score(), precision_score(), recall_score() and f1_score() from the SKLearn metrics package to evaluate the model.

- accuracy_score(): proportion of correctly classified instances in the training set
- precision_score(): proportion of instances predicted as malignant that are actually malignant
- recall(): The proportion of actual malignant instances that are correctly predicted
- f1_score(): combination of precision and recall (range from 0 to 1), a lower f1_score indicates a trade off between precision and recall.

We split the data to form a test set and a training set to measure how the model generalizes to new, unseen instances. The results of the both sets differ minimally, indicating that the model does not struggle with unseen instances.

In the training and test set:
- The model had a low accuracy which suggest that it correctly classified only 42% and 39% of instances in the training and test set respectively.
- The model had a low precision and correctly classified only 39% and 38% of instances that are malignant as malignant, suggesting that the model misclassified benign cells as malignant 
- The model has a perfect recall score, suggesting that the model captures all actual positive instances 
- The model has a f1-score of 0.56 and 0.55 for the test and training set respectively, suggesting a good balance between precision and recall

The high recall and low precision suggest that the rule-based classifier tends to identify a large portion of malignant instances but also includes many false positives (benign instances incorrectly classified as malignant). 

## Random Forest Classifier
We implemented a random forest classifier using the SKLearn library to the features given in the dataset. The random forest classifier is an ensemble method, where predictions from other models are combined. Multiple decision trees are created using different random subsets of the data and features. The prediction for each decision tree is calculated and the most popular result is used as the prediction. The algorithm uses bagging (bootstrap aggregating) by training each tree on different subsets of data to reduce variance and improve generalization

### Algorithm:
1. Bootstrapping
    - Sample a random subset of training data with replacement 
2. Random Feature Selection
- At each node, a random subset of features is used for splitting. Number of features at each split = square root of total number of features (square root 30)
3. Decision Tree
- Each decision tree is built using a subset of training data and features. It recursively selects the best split at each node based on Gini impurity.
- Each tree independently predicts if the patient is benign or malignant 
4. Voting
- The trees vote to determine the predicted class (malignant / benign) 
- If 70 trees predict "malignant" and 30 trees predict "benign," the final ensemble prediction is "malignant" (assuming a majority voting threshold of 50%)

### Evaluation
**SKLearn Metrics**
The Random Forest Classifier performed well with a high accuracy, precision, recall and f1 score.

The model correctly classified 96% of instances, had a 97% accuracy in predicting positive instances, correctly identified 93.02% of the actual positive instances and a high f1 score of 0.95, suggesting that there is a very good balance between making accurate positive predictions (precision) and capturing actual positive instances (recall).

**Confusion Matrix** 
A confusion matrix was used to understand the tradeoff between false positives (1) and false negatives (3). 

**Bar Plot**
A bar plot was used to measure the importance of each feature, using the model’s internal score to find the best way to split the data within each decision tree. 

## Gaussian Naive Bayes 
Complement Naive Bayes using the SKLearn library was implemented. It uses the complement of the class frequencies when estimating the probability distribution of features, giving more weight to features that are important for the minority class (malignant).

First, value_counts() in Pandas is used to count the occurrence of each class. This suggests that the dataset is imbalanced as the benign class (class 0) dominates the malignant class (class 1). Thus, we decided to use a Complement Naive Bayes algorithm for classification.

### Algorithm
1. Probability Estimation
    - Calculate the frequency of each class in the dataset 
    - For each feature, calculate the complement of its frequency compared to the entire dataset for each class to give more importance to features that have a lower frequency in the class
2. Probability Prediction
    - Using complement class frequencies, calculate log probabilities of each feature given each class
    - Sum the log probabilities of individual features 
    - Assign the class with the highest log probability as the predicted class

### Evaluation
The model correctly classified 97% of instances in the test set. Moreover, all instances predicted as positive were positive. The model effectively identified 93% of actual positive instances and has a high f1 score of 0.96, suggesting that there is a great balance between precision and recall.

## Conclusion
Interpretability is the ability to understand and explain the decisions or predictions made by a machine learning model. Classification performance refers to how well a model can correctly categorize or predict the class labels of unseen instances.

**Rule-based classifier**
- Accuracy: 0.39
- Precision: 0.38
- Recall: 1.00
- F1-Score: 0.55
- Interpretability: Rule-based system has high interpretability as rules explicitly define decision boundaries, making it easier to understand how input features contribute to the final prediction

**Random Forest Classifier**
- Accuracy: 0.9649
- Precision: 0.9756
- Recall: 0.9302
- F1 Score: 0.9524
- Interpretability: While individual decision trees are interpretable, the combined effect of numerous trees can be challenging to interpret and the complexity increases with the number of trees in the forest.


**Complement naive bayes**
- Accuracy: 0.97
- Precision: 1.00
- Recall: 0.93
- F1 Score: 0.96
- Interpretability: Based on Bayes’ theorem, the model calculates probabilities and makes predictions based on the likelihood of observing feature values given the class labels.
Model issues that features are conditionally independent

## Conclusion
- Rule-based classifier has the highest interpretability due to its simplistic model but the lowest classification performance. 
- Random Forest classifier has the lowest interpretability due to the large number of decision trees which make it difficult to identify the most influential features and has a relatively high classification performance. 
- Complement Naive Bayes classifier is interpretable and has the highest classification performance. 
