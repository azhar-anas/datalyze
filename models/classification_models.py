import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def logr_param_selector():
    st.info('''
            - **Logistic Regression** is simple and easy to implement.
            - It provides probabilities and can be used for binary classification.
            - It performs well when the dataset is linearly separable.\n
            View more about **Logistic Regression** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
            ''')
    
    solver = st.selectbox('**Solver** -> Algorithm to use in the optimization problem', options=['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], index=0)
    penalties = ['l2', None] if solver in ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'] else ['l1', 'l2'] if solver == 'liblinear' else ['l1', 'l2', 'elasticnet', None]
    penalty = st.selectbox('**Penalty** -> Norm used in the penalization', options=penalties, index=penalties.index('l2'))
    C = st.number_input('**C (0.0001 - 100)** -> Inverse of regularization strength, must be a positive float', min_value=1e-4, max_value=1e2, value=1.0, step=1e-4, format="%.4f")
    max_iter = st.number_input('**Max Iteration (100 - 10000)** -> Maximum number of iterations taken for the solvers to converge', min_value=100, max_value=10000, step=50, value=100)
    
    if penalty == 'elasticnet':
        l1_ratio = st.slider('**L1 Ratio** -> The Elastic-Net mixing parameter', 0.0, 1.0, step=0.1, value=0.5)
        random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42, step=1)
        params = {'solver': solver, 'penalty': penalty, 'C': C, 'max_iter': max_iter, 'random_state': random_state, 'l1_ratio': l1_ratio, 'n_jobs': -1}
    else:
        random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42, step=1)
        params = {'solver': solver, 'penalty': penalty, 'C': C, 'max_iter': max_iter, 'random_state': random_state, 'n_jobs': -1}
        
    model = LogisticRegression(**params)
    return model

def dt_c_param_selector():
    st.info('''
            - **Decision Tree** is simple and easy to understand.
            - It can handle both numerical and categorical data.
            - It can be used for both classification and regression tasks.\n
            View more about **Decision Tree** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
            ''')
    
    criterion = st.selectbox('**Criterion** -> The function to measure the quality of a split', options=['gini', 'entropy', 'log_loss'], index=0)
    max_features = st.selectbox('**Max Features** -> The number of features to consider when looking for the best split', options=[None, 'sqrt', 'log2'], index=0)
    max_depth = st.number_input('**Max Depth (0 - 100)** -> The maximum depth of the tree (0 means None or Auto)', min_value=0, max_value=100, step=1, value=0)
    min_samples_split = st.number_input('**Min Samples Split (2 - 20)** -> The minimum number of samples required to split an internal node', min_value=2, max_value=20, step=1, value=2)
    min_samples_leaf = st.number_input('**Min Samples Leaf (1 - 20)** -> The minimum number of samples required to be at a leaf node', min_value=1, max_value=20, step=1, value=1)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, step=1, value=42)
    max_depth = None if max_depth == 0 else max_depth
    
    params = {'criterion': criterion, 'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'random_state': random_state}
    model = DecisionTreeClassifier(**params)
    return model

def rf_c_param_selector():
    st.info('''
            - **Random Forest** is an ensemble learning method that can be used for both classification and regression tasks.
            - It can handle both numerical and categorical data.
            - It can handle missing values and maintain accuracy when a large proportion of the data is missing.\n
            View more about **Random Forest** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
            ''')
    
    criterion = st.selectbox('**Criterion** -> The function to measure the quality of a split', options=['gini', 'entropy', 'log_loss'], index=0)
    max_features = st.selectbox('**Max Features** -> The number of features to consider when looking for the best split', options=[None, 'sqrt', 'log2'], index=1)
    n_estimators = st.number_input('**N Estimators (10 - 1000)** -> The number of trees in the forest', min_value=10, max_value=1000, step=5, value=100)
    max_depth = st.number_input('**Max Depth (0 - 100)** -> The maximum depth of the tree (0 means None or Auto)', min_value=0, max_value=100, step=1, value=0)
    min_samples_split = st.number_input('**Min Samples Split (2 - 20)** -> The minimum number of samples required to split an internal node', min_value=2, max_value=20, step=1, value=2)
    min_samples_leaf = st.number_input('**Min Samples Leaf (1 - 20)** -> The minimum number of samples required to be at a leaf node', min_value=1, max_value=20, step=1, value=1)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, step=1, value=42)
    max_depth = None if max_depth == 0 else max_depth
    
    params = { 'criterion': criterion, 'max_features': max_features, 'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'random_state': random_state, 'n_jobs': -1}
    model = RandomForestClassifier(**params)
    return model

def knn_c_param_selector():
    st.info('''
            - **K-Nearest Neighbors** is a simple and easy-to-implement algorithm.
            - It can be used for both classification and regression tasks.
            - It is a non-parametric method that does not make any assumptions about the underlying data distribution.\n
            View more about **K-Nearest Neighbors** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
            ''')
    weights = st.selectbox('**Weights** -> Weight function used in prediction', options=['uniform', 'distance'], index=0)
    algorithm = st.selectbox('**Algorithm** -> Algorithm used to compute the nearest neighbors', options=['auto', 'ball_tree', 'kd_tree', 'brute'], index=0)
    n_neighbors = st.number_input('**N Neighbors (1 - 20)** -> Number of neighbors to use', min_value=1, max_value=20, step=1, value=5)
    p = st.number_input('**P (1 - 5)** -> Power parameter for the Minkowski metric', min_value=1, max_value=5, step=1, value=2)
    
    params = {'weights': weights, 'algorithm': algorithm, 'n_neighbors': n_neighbors, 'p': p, 'n_jobs': -1}
    model = KNeighborsClassifier(**params)
    return model

def svc_param_selector():
    st.info('''
            - **Support Vector Machine** is a powerful algorithm that can be used for both classification and regression tasks.
            - It can handle both linear and non-linear data.
            - It is effective in high-dimensional spaces and is versatile due to the kernel trick.\n
            View more about **Support Vector Machine** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
            ''')
    kernel = st.selectbox('**Kernel** -> Specifies the kernel type to be used in the algorithm', options=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], index=2)
    gamma = st.selectbox('**Gamma** -> Kernel coefficient for rbf, poly and sigmoid', options=['scale', 'auto'], index=0)
    C = st.number_input('**C (0.001 - 10)** -> Regularization parameter', min_value=1e-3, max_value=10.0, value=1.0, step=1e-2, format="%.3f")
    degree = st.number_input('**Degree (1 - 10)** -> Degree of the polynomial kernel function (poly)', min_value=1, max_value=10, step=1, value=3)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, step=1, value=42)
    
    params = {'kernel': kernel, 'gamma': gamma, 'C': C,  'degree': degree, 'random_state': random_state}
    model = SVC(**params)
    return model

def nb_c_param_selector():
    st.info('''
            - **Naive Bayes** is a simple and easy-to-implement algorithm.
            - It is based on Bayes' theorem and assumes that features are independent of each other.
            - It is effective for large datasets and can be used for both binary and multi-class classification tasks.\n
            View more about **Naive Bayes** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
            ''')
    
    var_smoothing = st.number_input('**Var Smoothing (1e-9 - 1e-3)** -> Portion of the largest variance of all features added to variances for calculation stability', min_value=1e-9, max_value=1e-3, value=1e-9, step=1e-9, format="%.9f")
    
    params = {'var_smoothing': var_smoothing}
    model = GaussianNB(**params)
    return model

def ada_c_param_selector():
    st.info('''
            - **AdaBoost** is an ensemble learning method that can be used for both classification and regression tasks.
            - It combines multiple weak learners to create a strong learner.
            - It is effective in improving the accuracy of the model and can handle both numerical and categorical data.\n
            View more about **AdaBoost** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
            ''')
    
    n_estimators = st.number_input('**N Estimators (10 - 1000)** -> The maximum number of estimators at which boosting is terminated', min_value=10, max_value=1000, step=5, value=50)
    learning_rate = st.number_input('**Learning Rate (0.001 - 10.0)** -> Weight applied to each classifier at each boosting iteration', min_value=1e-3, max_value=10.0, step=1e-2, value=0.1, format="%.3f")
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, step=1, value=42)
    
    params = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'random_state': random_state}
    model = AdaBoostClassifier(**params)
    return model

def gb_c_param_selector():
    st.info('''
            - **Gradient Boosting** is an ensemble learning method that can be used for both classification and regression tasks.
            - It builds trees one at a time, where each new tree helps to correct errors made by the previously trained tree.
            - It is effective in improving the accuracy of the model and can handle both numerical and categorical data.\n
            View more about **Gradient Boosting** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
            ''')
    max_features = st.selectbox('**Max Features** -> The number of features to consider when looking for the best split', options=[None, 'sqrt', 'log2'], index=0)
    learning_rate = st.number_input('**Learning Rate (0.001 - 10.0)** -> Learning rate shrinks the contribution of each tree by learning_rate', min_value=1e-3, max_value=10.0, step=1e-2, value=0.1, format="%.3f")
    n_estimators = st.number_input('**N Estimators (10 - 1000)** -> The number of boosting stages to be run', min_value=10, max_value=1000, step=5, value=100)
    subsample = st.number_input('**Subsample (0.1 - 1.0)** -> The fraction of samples to be used for fitting the individual base learners', min_value=0.1, max_value=1.0, step=0.05, value=1.0)
    min_samples_split = st.number_input('**Min Samples Split (2 - 20)** -> The minimum number of samples required to split an internal node', min_value=2, max_value=20, step=1, value=2)
    min_samples_leaf = st.number_input('**Min Samples Leaf (1 - 20)** -> The minimum number of samples required to be at a leaf node', min_value=1, max_value=20, step=1, value=1)
    max_depth = st.number_input('**Max Depth (1 - 100)** -> The maximum depth of the tree', min_value=1, max_value=100, step=1, value=3)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, step=1, value=42)
    
    params = {'max_features': max_features, 'learning_rate': learning_rate, 'n_estimators': n_estimators, 'subsample': subsample, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'max_depth': max_depth, 'random_state': random_state}
    model = GradientBoostingClassifier(**params)
    return model

def mlp_c_param_selector():
    st.info('''
            - **Multi-Layer Perceptron** is a feedforward artificial neural network that can be used for both classification and regression tasks.
            - It can handle both numerical and categorical data.
            - It is effective in learning complex relationships in the data and can be used for deep learning tasks.\n
            View more about **Multi-Layer Perceptron** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
            ''')
    
    hidden_layer_sizes = st.text_input('**Hidden Layer Sizes** -> The ith element represents the number of neurons in the ith hidden layer', value='32, 64, 32', placeholder='Neuron_layer_1, neuron_layer_2, neuron_layer_3, etc')
    hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))
    activation = st.selectbox('**Activation** -> Activation function for the hidden layer', options=['identity', 'logistic', 'tanh', 'relu'], index=3)
    solver = st.selectbox('**Solver** -> The solver for weight optimization', options=['lbfgs', 'sgd', 'adam'], index=2)
    learning_rate = st.selectbox('**Learning Rate** -> Learning rate schedule for weight updates', options=['constant', 'invscaling', 'adaptive'], index=0)
    learning_rate_init = st.number_input('**Learning Rate Init (0.00001 - 1.0)** -> The initial learning rate used', min_value=1e-5, max_value=1.0, step=1e-5, value=1e-3, format="%.5f")
    alpha = st.number_input('**Alpha (0.0001 - 1.0)** -> L2 penalty (regularization term) parameter', min_value=1e-4, max_value=1.0, step=1e-3, value=1e-4, format="%.4f")
    max_iter = st.number_input('**Max Iteration (10 - 10000)** -> Maximum number of iterations', min_value=10, max_value=10000, step=10, value=200)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, step=1, value=42)
    
    params = {'hidden_layer_sizes': hidden_layer_sizes, 'activation': activation, 'solver': solver, 'learning_rate': learning_rate, 'learning_rate_init': learning_rate_init, 'alpha': alpha, 'max_iter': max_iter, 'random_state': random_state}
    model = MLPClassifier(**params)
    return model