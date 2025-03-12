import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

def lr_param_selector():
    st.info('''
            - **Linear Regression** is a linear approach to modeling the relationship
            between a dependent variable and one or more independent variables.
            - It is a simple and easy-to-understand algorithm that is widely used in the field of statistics and machine learning.
            - It provides interpretable results, making it easy to understand the impact of each feature on the target variable.\n
            View more about **Linear Regression** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
            ''')
    
    fit_intercept = st.selectbox('**Fit Intercept** -> Whether to calculate the intercept for this model', options=[True, False], index=0)
    copy_X = st.selectbox('**Copy X** -> If True, X will be copied; else, it may be overwritten', options=[True, False], index=0)
    
    params = {'fit_intercept': fit_intercept, 'copy_X': copy_X, 'n_jobs': -1}
    model = LinearRegression(**params)
    return model

def lasso_param_selector():
    st.info('''
            - **Lasso Regression** is a linear regression model that uses L1 regularization to prevent overfitting.
            - It adds a penalty term (L1 Regularization) to the loss function that penalizes the absolute value of the coefficients.
            - This penalty term helps to reduce the complexity of the model by shrinking the coefficients of less important features to zero.\n
            View more about **Lasso Regression** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
            ''')
    
    alpha = st.number_input('**Alpha (0.1 - 10)** -> Constant that multiplies the L1 term', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    fit_intercept = st.selectbox('**Fit Intercept** -> Whether to calculate the intercept for this model', options=[True, False], index=0)
    
    params = {'alpha': alpha, 'fit_intercept': fit_intercept}
    model = Lasso(**params)
    return model

def ridge_param_selector():
    st.info('''
            - **Ridge Regression** is a linear regression model that uses L2 regularization to prevent overfitting.
            - It adds a penalty term (L2 Regularization) to the loss function that penalizes the square of the coefficients.
            - This penalty term helps to reduce the complexity of the model by shrinking the coefficients of less important features.\n
            View more about **Ridge Regression** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
            ''')
    
    alpha = st.number_input('**Alpha (0.0 - 1.0)** -> L2 Regularization strength; must be a positive float',  min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    fit_intercept = st.selectbox('**Fit Intercept** -> Whether to calculate the intercept for this model', options=[True, False], index=0)
    
    params = {'alpha': alpha, 'fit_intercept': fit_intercept}
    model = Ridge(**params)
    return model

def dt_r_param_selector():
    st.info('''
            - **Decision Tree** is simple and easy to understand.
            - It can handle both numerical and categorical data.
            - It can be used for both classification and regression tasks.\n
            View more about **Decision Tree** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
            ''')
    
    criterion = st.selectbox('**Criterion** -> The function to measure the quality of a split', options=['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], index=0)
    max_features = st.selectbox('**Max Features** -> The number of features to consider when looking for the best split', options=[None, 'sqrt', 'log2'], index=0)
    max_depth = st.number_input('**Max Depth (0 - 100)** -> The maximum depth of the tree (0 means None or Auto)', min_value=0, max_value=100, step=1, value=0)
    min_samples_split = st.number_input('**Min Samples Split (2 - 20)** -> The minimum number of samples required to split an internal node', min_value=2, max_value=20, step=1, value=2)
    min_samples_leaf = st.number_input('**Min Samples Leaf (1 - 20)** -> The minimum number of samples required to be at a leaf node', min_value=1, max_value=20, step=1, value=1)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, step=1, value=42)
    max_depth = None if max_depth == 0 else max_depth
    
    params = {'criterion': criterion, 'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'random_state': random_state}
    model = DecisionTreeRegressor(**params)
    return model
    
def rf_r_param_selector():
    st.info('''
            - **Random Forest** is an ensemble learning method that can be used for both classification and regression tasks.
            - It can handle both numerical and categorical data.
            - It can handle missing values and maintain accuracy when a large proportion of the data is missing.\n
            View more about **Random Forest** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
            ''')
    
    criterion = st.selectbox('**Criterion** -> The function to measure the quality of a split', options=['squared_error', 'absolute_error', 'friedman_mse', 'poisson'], index=0)
    max_features = st.selectbox('**Max Features** -> The number of features to consider when looking for the best split', options=[None, 'sqrt', 'log2'], index=0)
    n_estimators = st.number_input('**N Estimators (10 - 1000)** -> The number of trees in the forest', min_value=10, max_value=1000, step=5, value=100)
    max_depth = st.number_input('**Max Depth (0 - 100)** -> The maximum depth of the tree (0 means None or Auto)', min_value=0, max_value=100, step=1, value=0)
    min_samples_split = st.number_input('**Min Samples Split (2 - 20)** -> The minimum number of samples required to split an internal node', min_value=2, max_value=20, step=1, value=2)
    min_samples_leaf = st.number_input('**Min Samples Leaf (1 - 20)** -> The minimum number of samples required to be at a leaf node', min_value=1, max_value=20, step=1, value=1)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, step=1, value=42)
    max_depth = None if max_depth == 0 else max_depth
    
    params = { 'criterion': criterion, 'max_features': max_features, 'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'random_state': random_state, 'n_jobs': -1}
    model = RandomForestRegressor(**params)
    return model
    
def knn_r_param_selector():
    st.info('''
            - **K-Nearest Neighbors** is a simple and easy-to-understand algorithm.
            - It can be used for both classification and regression tasks.
            - It is a non-parametric method that does not make any assumptions about the underlying data distribution.\n
            View more about **K-Nearest Neighbors** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
            ''')
    
    weights = st.selectbox('**Weights** -> Weight function used in prediction', options=['uniform', 'distance'], index=0)
    algorithm = st.selectbox('**Algorithm** -> Algorithm used to compute the nearest neighbors', options=['auto', 'ball_tree', 'kd_tree', 'brute'], index=0)
    n_neighbors = st.number_input('**N Neighbors (1 - 20)** -> Number of neighbors to use', min_value=1, max_value=20, step=1, value=5)
    p = st.number_input('**P (1 - 5)** -> Power parameter for the Minkowski metric', min_value=1, max_value=5, step=1, value=2)
    
    params = {'weights': weights, 'algorithm': algorithm, 'n_neighbors': n_neighbors, 'p': p, 'n_jobs': -1}
    model = KNeighborsRegressor(**params)
    return model

def svr_param_selector():
    st.info('''
            - **Support Vector Machine** is a powerful algorithm that can be used for both classification and regression tasks.
            - It can handle both linear and non-linear data.
            - It is effective in high-dimensional spaces and is versatile due to the kernel trick.\n
            View more about **Support Vector Machine** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
            ''')
    
    kernel = st.selectbox('**Kernel** -> Specifies the kernel type to be used in the algorithm', options=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], index=2)
    gamma = st.selectbox('**Gamma** -> Kernel coefficient for rbf, poly and sigmoid', options=['scale', 'auto'], index=0)
    C = st.number_input('**C (0.001 - 10)** -> Regularization parameter', min_value=1e-3, max_value=10.0, value=1.0, step=1e-2, format="%.3f")
    degree = st.number_input('**Degree (1 - 10)** -> Degree of the polynomial kernel function (poly)', min_value=1, max_value=10, step=1, value=3)
    
    params = {'kernel': kernel, 'gamma': gamma, 'C': C,  'degree': degree}
    model = SVR(**params)
    return model
    
def ada_r_param_selector():
    st.info('''
            - **AdaBoost** is an ensemble learning method that can be used for both classification and regression tasks.
            - It combines multiple weak learners to create a strong learner.
            - It is effective in improving the accuracy of the model and can handle both numerical and categorical data.\n
            View more about **AdaBoost** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
            ''')
    
    n_estimators = st.number_input('**N Estimators (10 - 1000)** -> The maximum number of estimators at which boosting is terminated', min_value=10, max_value=1000, step=5, value=50)
    learning_rate = st.number_input('**Learning Rate (0.001 - 10.0)** -> Weight applied to each regressor at each boosting iteration', min_value=1e-3, max_value=10.0, step=1e-2, value=0.1, format="%.3f")
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, step=1, value=42)
    
    params = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'random_state': random_state}
    model = AdaBoostRegressor(**params)
    return model
    
def gb_r_param_selector():
    st.info('''
            - **Gradient Boosting** is an ensemble learning method that can be used for both classification and regression tasks.
            - It builds trees one at a time, where each new tree helps to correct errors made by the previously trained tree.
            - It is effective in improving the accuracy of the model and can handle both numerical and categorical data.\n
            View more about **Gradient Boosting** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
            ''')
    
    criterion = st.selectbox('**Criterion** -> The function to measure the quality of a split', options=['friedman_mse', 'squared_error'], index=0)
    loss = st.selectbox('**Loss** -> Loss function to be optimized', options=['squared_error', 'absolute_error', 'huber', 'quantile'], index=0)
    max_features = st.selectbox('**Max Features** -> The number of features to consider when looking for the best split', options=['sqrt', 'log2', None], index=0)
    learning_rate = st.number_input('**Learning Rate (0.001 - 10.0)** -> Learning rate shrinks the contribution of each tree by learning_rate', min_value=1e-3, max_value=10.0, step=1e-2, value=0.1, format="%.3f")
    n_estimators = st.number_input('**N Estimators (10 - 1000)** -> The number of boosting stages to be run', min_value=10, max_value=1000, step=5, value=100)
    subsample = st.number_input('**Subsample (0.1 - 1.0)** -> The fraction of samples to be used for fitting the individual base learners', min_value=0.1, max_value=1.0, step=0.05, value=1.0)
    min_samples_split = st.number_input('**Min Samples Split (2 - 20)** -> The minimum number of samples required to split an internal node', min_value=2, max_value=20, step=1, value=2)
    min_samples_leaf = st.number_input('**Min Samples Leaf (1 - 20)** -> The minimum number of samples required to be at a leaf node', min_value=1, max_value=20, step=1, value=1)
    max_depth = st.number_input('**Max Depth (1 - 100)** -> The maximum depth of the tree', min_value=1, max_value=100, step=1, value=3)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, step=1, value=42)
    
    params = {'loss': loss, 'learning_rate': learning_rate, 'n_estimators': n_estimators, 'subsample': subsample, 'criterion': criterion, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'max_depth': max_depth, 'max_features': max_features, 'random_state': random_state}
    model = GradientBoostingRegressor(**params)
    return model

def mlp_r_param_selector():
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
    model = MLPRegressor(**params)
    return model

# def gb_r_param_selector():
#     n_estimators = st.number_input('**Number of Estimators (10 - 1000)** -> The number of boosting stages to be run', 10, 1000, 100)
#     learning_rate = st.number_input('**Learning Rate (0.01 - 1.0)** -> Learning rate shrinks the contribution of each tree', 0.01, 1.0, 0.1)
#     max_depth = st.number_input('**Max Depth (1 - 100)** -> Maximum depth of the individual regression estimators', 1, 100, 3)
#     random_state = st.number_input('**Random State** -> Controls the randomness of the estimator', value=42)
    
#     params = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth, 'random_state': random_state}
#     model = GradientBoostingRegressor(**params)
#     return model

# def mlp_r_param_selector():
#     hidden_layer_sizes = st.text_input('**Hidden Layer Sizes** -> The ith element represents the number of neurons in the ith hidden layer', '100')
#     activation = st.selectbox('**Activation** -> Activation function for the hidden layer', options=['identity', 'logistic', 'tanh', 'relu'], index=3)
#     solver = st.selectbox('**Solver** -> The solver for weight optimization', options=['lbfgs', 'sgd', 'adam'], index=2)
#     learning_rate = st.selectbox('**Learning Rate** -> Learning rate schedule for weight updates', options=['constant', 'invscaling', 'adaptive'], index=0)
#     random_state = st.number_input('**Random State** -> Controls the randomness of the estimator', value=42)
    
#     params = {'hidden_layer_sizes': eval(hidden_layer_sizes), 'activation': activation, 'solver': solver, 'learning_rate': learning_rate, 'random_state': random_state}
#     model = MLPRegressor(**params)
#     return model
