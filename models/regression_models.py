import streamlit as st
import optuna as opt
import gc
from sklearn.model_selection import cross_val_score

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
    
    params = {'fit_intercept': fit_intercept, 'n_jobs': -1}
    model = LinearRegression(**params)
    return model

def lasso_param_selector():
    st.info('''
            - **Lasso Regression** is a linear regression model that uses L1 regularization to prevent overfitting.
            - It adds a penalty term (L1 Regularization) to the loss function that penalizes the absolute value of the coefficients.
            - This penalty term helps to reduce the complexity of the model by shrinking the coefficients of less important features to zero.\n
            View more about **Lasso Regression** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
            ''')
    
    fit_intercept = st.selectbox('**Fit Intercept** -> Whether to calculate the intercept for this model', options=[True, False], index=0)
    alpha = st.number_input('**Alpha (0.0001 - 10)** -> Constant that multiplies the L1 term', min_value=0.0001, max_value=10.0, value=1.0, step=0.0001, key='alpha', format="%.4f")
    max_iter = st.number_input('**Max Iteration (100 - 10000)** -> The maximum number of iterations', min_value=100, max_value=10000, value=1000, step=100, key='max_iter')
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42, step=1, key='random_state')
    
    params = {'fit_intercept': fit_intercept, 'alpha': alpha, 'max_iter': max_iter, 'random_state': random_state}
    model = Lasso(**params)
    return model

def ridge_param_selector():
    st.info('''
            - **Ridge Regression** is a linear regression model that uses L2 regularization to prevent overfitting.
            - It adds a penalty term (L2 Regularization) to the loss function that penalizes the square of the coefficients.
            - This penalty term helps to reduce the complexity of the model by shrinking the coefficients of less important features.\n
            View more about **Ridge Regression** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
            ''')
    
    fit_intercept = st.selectbox('**Fit Intercept** -> Whether to calculate the intercept for this model', options=[True, False], index=0)
    alpha = st.number_input('**Alpha (0.0001 - 1.0)** -> L2 Regularization strength; must be a positive float',  min_value=0.0001, max_value=10.0, value=1.0, step=0.0001, format="%.4f")
    max_iter = st.number_input('**Max Iteration (100 - 10000)** -> The maximum number of iterations', min_value=100, max_value=10000, value=1000, step=100)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42, step=1)
    
    params = {'fit_intercept': fit_intercept, 'alpha': alpha, 'max_iter': max_iter, 'random_state': random_state}
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
    max_depth = st.number_input('**Max Depth (0 - 100)** -> The maximum depth of the tree (0 means None or Auto)', min_value=0, max_value=100, value=0, step=1)
    min_samples_split = st.number_input('**Min Samples Split (2 - 50)** -> The minimum number of samples required to split an internal node', min_value=2, max_value=50, value=2, step=1)
    min_samples_leaf = st.number_input('**Min Samples Leaf (1 - 50)** -> The minimum number of samples required to be at a leaf node', min_value=1, max_value=50, value=1, step=1)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42, step=1)
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
    n_estimators = st.number_input('**N Estimators (10 - 1000)** -> The number of trees in the forest', min_value=10, max_value=100, value=100, step=5)
    max_depth = st.number_input('**Max Depth (0 - 100)** -> The maximum depth of the tree (0 means None or Auto)', min_value=0, max_value=100, value=0, step=1)
    min_samples_split = st.number_input('**Min Samples Split (2 - 50)** -> The minimum number of samples required to split an internal node', min_value=2, max_value=50, value=2, step=1)
    min_samples_leaf = st.number_input('**Min Samples Leaf (1 - 50)** -> The minimum number of samples required to be at a leaf node', min_value=1, max_value=50, value=1, step=1)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42, step=1)
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
    n_neighbors = st.number_input('**N Neighbors (1 - 100)** -> Number of neighbors to use', min_value=1, max_value=100, value=5, step=1)
    p = st.number_input('**P (1 - 5)** -> Power parameter for the Minkowski metric', min_value=1, max_value=5, value=2, step=1)
    
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
    
    kernel = st.selectbox('**Kernel** -> Specifies the kernel type to be used in the algorithm', options=['linear', 'poly', 'rbf', 'sigmoid'], index=2)
    gamma = st.selectbox('**Gamma** -> Kernel coefficient for rbf, poly and sigmoid', options=['scale', 'auto'], index=0)
    C = st.number_input('**C (0.0001 - 10)** -> Regularization parameter', min_value=0.0001, max_value=10.0, value=1.0, step=0.0001, format="%.4f")
    degree = st.number_input('**Degree (1 - 10)** -> Degree of the polynomial kernel function (poly)', min_value=1, max_value=10, value=3, step=1)
    
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
    
    n_estimators = st.number_input('**N Estimators (10 - 1000)** -> The maximum number of estimators at which boosting is terminated', min_value=10, max_value=100, value=50, step=5)
    learning_rate = st.number_input('**Learning Rate (0.001 - 10.0)** -> Weight applied to each regressor at each boosting iteration', min_value=0.001, max_value=10.0, value=0.1, step=0.001, format="%.3f")
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
    learning_rate = st.number_input('**Learning Rate (0.001 - 10.0)** -> Learning rate shrinks the contribution of each tree by learning_rate', min_value=0.001, max_value=10.0, value=0.1, step=0.001, format="%.3f")
    n_estimators = st.number_input('**N Estimators (10 - 1000)** -> The number of boosting stages to be run', min_value=10, max_value=100, value=100, step=5)
    subsample = st.number_input('**Subsample (0.01 - 1.0)** -> The fraction of samples to be used for fitting the individual base learners', min_value=0.01, max_value=1.0, value=1.0, step=0.01, format="%.3f")
    max_depth = st.number_input('**Max Depth (1 - 100)** -> The maximum depth of the tree', min_value=1, max_value=100, value=3, step=1)
    min_samples_split = st.number_input('**Min Samples Split (2 - 50)** -> The minimum number of samples required to split an internal node', min_value=2, max_value=50, value=2, step=1)
    min_samples_leaf = st.number_input('**Min Samples Leaf (1 - 50)** -> The minimum number of samples required to be at a leaf node', min_value=1, max_value=50, value=1, step=1)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42, step=1)
    
    params = {'loss': loss, 'learning_rate': learning_rate, 'n_estimators': n_estimators, 'subsample': subsample, 'criterion': criterion, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'max_depth': max_depth, 'max_features': max_features, 'random_state': random_state}
    model = GradientBoostingRegressor(**params)
    return model

def mlp_r_param_selector():
    st.info('''
            - **Multi-Layer Perceptron** is a feedforward artificial neural network that can be used for both classification and regression tasks.
            - It can handle both numerical and categorical data.
            - It is effective in learning complex relationships in the data and can be used for deep learning tasks.\n
            View more about **Multi-Layer Perceptron** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
            ''')
    activation = st.selectbox('**Activation** -> Activation function for the hidden layer', options=['identity', 'logistic', 'tanh', 'relu'], index=3)
    solver = st.selectbox('**Solver** -> The solver for weight optimization', options=['lbfgs', 'sgd', 'adam'], index=2)
    learning_rate = st.selectbox('**Learning Rate** -> Learning rate schedule for weight updates', options=['constant', 'invscaling', 'adaptive'], index=0)
    st.write('---')
    n_layers = st.number_input('**N Layers (1 - 10)** -> The number of hidden layers in the neural network', min_value=1, max_value=10, value=1, step=1)
    layer_sizes = ()
    for i in range(n_layers):
        neurons = st.number_input(f'**Layer {i+1} Neurons (10 - 1000)** -> The number of neurons in layer {i+1}', min_value=10, max_value=1000, value=100, step=10)
        layer_sizes += (neurons,)
    st.write('---')
    learning_rate_init = st.number_input('**Learning Rate Init (0.00001 - 1.0)** -> The initial learning rate used', min_value=0.00001, max_value=1.0, step=0.00001, value=0.001, format="%.5f")
    alpha = st.number_input('**Alpha (0.0001 - 1.0)** -> L2 penalty (regularization term) parameter', min_value=0.0001, max_value=1.0, step=0.001, value=0.0001, format="%.4f")
    max_iter = st.number_input('**Max Iteration (10 - 1000)** -> Maximum number of iterations (Epoch)', min_value=10, max_value=1000, step=10, value=200)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, step=1, value=42)
    
    params = {'hidden_layer_sizes': layer_sizes, 'activation': activation, 'solver': solver, 'learning_rate': learning_rate, 'learning_rate_init': learning_rate_init, 'alpha': alpha, 'max_iter': max_iter, 'random_state': random_state}
    model = MLPRegressor(**params)
    return model









# TREE-STRUCTURED PARZEN ESTIMATOR (TPE) HYPERPARAMETER OPTIMIZATION

# Get the range of hyperparameters for the selected model
def get_r_param_range(model_name):
    if model_name == 'Linear Regression':
        st.info('''
            - **Linear Regression** is a linear approach to modeling the relationship
            between a dependent variable and one or more independent variables.
            - It is a simple and easy-to-understand algorithm that is widely used in the field of statistics and machine learning.
            - It provides interpretable results, making it easy to understand the impact of each feature on the target variable.\n
            View more about **Linear Regression** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
            ''')
        
        # Input Parameters             
        fit_intercept_range = st.multiselect('**Fit Intercept** -> Whether to calculate the intercept for this model', [True, False], default=[True])
        random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42, step=1)
        
        # Parameters Configuration Range
        params_range = {
            'fit_intercept': fit_intercept_range, 
            'random_state': random_state
        }
        del fit_intercept_range
    
    elif model_name == 'Lasso Regression':
        st.info('''
            - **Lasso Regression** is a linear regression model that uses L1 regularization to prevent overfitting.
            - It adds a penalty term (L1 Regularization) to the loss function that penalizes the absolute value of the coefficients.
            - This penalty term helps to reduce the complexity of the model by shrinking the coefficients of less important features to zero.\n
            View more about **Lasso Regression** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
            ''')
        def sync_values():
            st.session_state.lr_alpha_0 = min(st.session_state.lr_alpha_0, st.session_state.lr_alpha_1)
            st.session_state.lr_alpha_1 = max(st.session_state.lr_alpha_0, st.session_state.lr_alpha_1)
            st.session_state.lr_max_iter_0 = min(st.session_state.lr_max_iter_0, st.session_state.lr_max_iter_1)
            st.session_state.lr_max_iter_1 = max(st.session_state.lr_max_iter_0, st.session_state.lr_max_iter_1)
        
        # Input Parameters
        fit_intercept_range = st.multiselect('**Fit Intercept** -> Whether to calculate the intercept for this model', [True, False], default=[True])
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                alpha_0 = st.number_input('**Alpha (0.0001 - 10)**', min_value=0.0001, max_value=10.0, value=0.0001, step=0.0001, format='%.4f', key='lr_alpha_0', on_change=sync_values)
                max_iter_0 = st.number_input('**Max Iteration (100 - 10000)**', min_value=100, max_value=10000, value=5000, step=100, key='lr_max_iter_0')
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                alpha_1 = st.number_input('**Alpha (0.0001 - 10)**', min_value=0.0001, max_value=10.0, value=10.0, step=0.0001, format='%.4f', key='lr_alpha_1', on_change=sync_values)
                max_iter_1 = st.number_input('**Max Iteration (100 - 10000)**', min_value=100, max_value=10000, value=5000, step=100, key='lr_max_iter_1')
        random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42, step=1)
        
        # Parameters Configuration Range
        params_range = {
            'fit_intercept': fit_intercept_range,
            'alpha': [alpha_0, alpha_1],
            'max_iter': [max_iter_0, max_iter_1],
            'random_state': random_state
        }
        del fit_intercept_range, alpha_0, alpha_1, max_iter_0, max_iter_1
    
    elif model_name == 'Ridge Regression':
        st.info('''
            - **Ridge Regression** is a linear regression model that uses L2 regularization to prevent overfitting.
            - It adds a penalty term (L2 Regularization) to the loss function that penalizes the square of the coefficients.
            - This penalty term helps to reduce the complexity of the model by shrinking the coefficients of less important features.\n
            View more about **Ridge Regression** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
            ''')
        def sync_values():
            st.session_state.rr_alpha_0 = min(st.session_state.rr_alpha_0, st.session_state.rr_alpha_1)
            st.session_state.rr_alpha_1 = max(st.session_state.rr_alpha_0, st.session_state.rr_alpha_1)
            st.session_state.rr_max_iter_0 = min(st.session_state.rr_max_iter_0, st.session_state.rr_max_iter_1)
            st.session_state.rr_max_iter_1 = max(st.session_state.rr_max_iter_0, st.session_state.rr_max_iter_1)
        
        # Input Parameters
        fit_intercept_range = st.multiselect('**Fit Intercept** -> Whether to calculate the intercept for this model', [True, False], default=[True])
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                alpha_0 = st.number_input('**Alpha (0.0001 - 1.0)**', min_value=0.0001, max_value=1.0, value=0.0001, step=0.0001, format='%.4f', key='rr_alpha_0', on_change=sync_values)
                max_iter_0 = st.number_input('**Max Iteration (100 - 10000)**', min_value=100, max_value=10000, value=5000, step=100, key='rr_max_iter_0')
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                alpha_1 = st.number_input('**Alpha (0.0001 - 1.0)**', min_value=0.0001, max_value=1.0, value=1.0, step=0.0001, format='%.4f', key='rr_alpha_1', on_change=sync_values)
                max_iter_1 = st.number_input('**Max Iteration (100 - 10000)**', min_value=100, max_value=10000, value=5000, step=100, key='rr_max_iter_1')
        random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42, step=1, key='random_state')
        
        # Parameters Configuration Range
        params_range = {
            'fit_intercept': fit_intercept_range,
            'alpha': [alpha_0, alpha_1],
            'max_iter': [max_iter_0, max_iter_1],
            'random_state': random_state
        }
        del fit_intercept_range, alpha_0, alpha_1, max_iter_0, max_iter_1
    
    elif model_name == 'Decision Tree':
        st.info('''
            - **Decision Tree** is simple and easy to understand.
            - It can handle both numerical and categorical data.
            - It can be used for both classification and regression tasks.\n
            View more about **Decision Tree** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
            ''')
        def sync_values():
            st.session_state.dt_r_max_depth_0 = min(st.session_state.dt_r_max_depth_0, st.session_state.dt_r_max_depth_1)
            st.session_state.dt_r_max_depth_1 = max(st.session_state.dt_r_max_depth_0, st.session_state.dt_r_max_depth_1)
            st.session_state.dt_r_min_samples_split_0 = min(st.session_state.dt_r_min_samples_split_0, st.session_state.dt_r_min_samples_split_1)
            st.session_state.dt_r_min_samples_split_1 = max(st.session_state.dt_r_min_samples_split_0, st.session_state.dt_r_min_samples_split_1)
            st.session_state.dt_r_min_samples_leaf_0 = min(st.session_state.dt_r_min_samples_leaf_0, st.session_state.dt_r_min_samples_leaf_1)
            st.session_state.dt_r_min_samples_leaf_1 = max(st.session_state.dt_r_min_samples_leaf_0, st.session_state.dt_r_min_samples_leaf_1)
        
        # Input Parameters             
        criterion_range = st.multiselect('**Criterion** -> The function to measure the quality of a split', ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], default=['friedman_mse', 'absolute_error'])
        max_features_range = st.multiselect('**Max Features** -> The number of features to consider when looking for the best split', [None, 'sqrt', 'log2'], default=[None, 'sqrt', 'log2'])
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                max_depth_0 = st.number_input('**Max Depth (1 - 100)**', min_value=1, max_value=100, value=3, key='dt_r_max_depth_0', on_change=sync_values)
                min_samples_split_0 = st.number_input('**Min Samples Split (2 - 50)**', min_value=2, max_value=50, value=2, key='dt_r_min_samples_split_0', on_change=sync_values)
                min_samples_leaf_0 = st.number_input('**Min Samples Leaf (1 - 50)**', min_value=1, max_value=50, value=1, key='dt_r_min_samples_leaf_0', on_change=sync_values)
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                max_depth_1 = st.number_input('**Max Depth (1 - 100)**', min_value=1, max_value=100, value=10, key='dt_r_max_depth_1', on_change=sync_values)
                min_samples_split_1 = st.number_input('**Min Samples Split (2 - 50)**', min_value=2, max_value=50, value=10, key='dt_r_min_samples_split_1', on_change=sync_values)
                min_samples_leaf_1 = st.number_input('**Min Samples Leaf (1 - 50)**', min_value=1, max_value=50, value = 10, key='dt_r_min_samples_leaf_1', on_change=sync_values)
        random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42)
        
        # Parameters Configuration Range
        params_range = {
            'criterion': criterion_range,
            'max_features': max_features_range,
            'max_depth': [max_depth_0, max_depth_1],
            'min_samples_split': [min_samples_split_0, min_samples_split_1],
            'min_samples_leaf': [min_samples_leaf_0, min_samples_leaf_1],
            'random_state': random_state
        }
        del criterion_range, max_features_range, max_depth_0, max_depth_1, min_samples_split_0, min_samples_split_1, min_samples_leaf_0, min_samples_leaf_1, random_state
        
    elif model_name == 'Random Forest':
        st.info('''
            - **Random Forest** is an ensemble learning method that can be used for both classification and regression tasks.
            - It can handle both numerical and categorical data.
            - It can handle missing values and maintain accuracy when a large proportion of the data is missing.\n
            View more about **Random Forest** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
            ''')
        def sync_values():
            st.session_state.rf_r_n_estimators_0 = min(st.session_state.rf_r_n_estimators_0, st.session_state.rf_r_n_estimators_1)
            st.session_state.rf_r_n_estimators_1 = max(st.session_state.rf_r_n_estimators_0, st.session_state.rf_r_n_estimators_1)
            st.session_state.rf_r_max_depth_0 = min(st.session_state.rf_r_max_depth_0, st.session_state.rf_r_max_depth_1)
            st.session_state.rf_r_max_depth_1 = max(st.session_state.rf_r_max_depth_0, st.session_state.rf_r_max_depth_1)
            st.session_state.rf_r_min_samples_split_0 = min(st.session_state.rf_r_min_samples_split_0, st.session_state.rf_r_min_samples_split_1)
            st.session_state.rf_r_min_samples_split_1 = max(st.session_state.rf_r_min_samples_split_0, st.session_state.rf_r_min_samples_split_1)
            st.session_state.rf_r_min_samples_leaf_0 = min(st.session_state.rf_r_min_samples_leaf_0, st.session_state.rf_r_min_samples_leaf_1)
            st.session_state.rf_r_min_samples_leaf_1 = max(st.session_state.rf_r_min_samples_leaf_0, st.session_state.rf_r_min_samples_leaf_1)
        
        # Input Parameters    
        criterion_range = st.multiselect('**Criterion** -> The function to measure the quality of a split', ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], default=['friedman_mse', 'absolute_error'])
        max_features_range = st.multiselect('**Max Features** -> The number of features to consider when looking for the best split', [None, 'sqrt', 'log2'], default=[None, 'sqrt', 'log2'])
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                n_estimators_0 = st.number_input('**N Estimators (10 - 1000)**', min_value=10, max_value=1000, value=10, key='rf_r_n_estimators_0', on_change=sync_values)
                max_depth_0 = st.number_input('**Max Depth (1 - 100)**', min_value=1, max_value=100, value=3, key='rf_r_max_depth_0', on_change=sync_values)
                min_samples_split_0 = st.number_input('**Min Samples Split (2 - 50)**', min_value=2, max_value=50, value=2, key='rf_r_min_samples_split_0', on_change=sync_values)
                min_samples_leaf_0 = st.number_input('**Min Samples Leaf (1 - 50)**', min_value=1, max_value=50, value=1, key='rf_r_min_samples_leaf_0', on_change=sync_values)
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                n_estimators_1 = st.number_input('**N Estimators (10 - 1000)**', min_value=10, max_value=1000, value=100, key='rf_r_n_estimators_1', on_change=sync_values)
                max_depth_1 = st.number_input('**Max Depth (1 - 100)**', min_value=1, max_value=100, value=10, key='rf_r_max_depth_1', on_change=sync_values)
                min_samples_split_1 = st.number_input('**Min Samples Split (2 - 50)**', min_value=2, max_value=50, value=10, key='rf_r_min_samples_split_1', on_change=sync_values)
                min_samples_leaf_1 = st.number_input('**Min Samples Leaf (1 - 50)**', min_value=1, max_value=50, value=10, key='rf_r_min_samples_leaf_1', on_change=sync_values)
        random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42)
        
        # Parameters Configuration Range
        params_range = {
            'criterion': criterion_range,
            'max_features': max_features_range,
            'n_estimators': [n_estimators_0, n_estimators_1],
            'max_depth': [max_depth_0, max_depth_1],
            'min_samples_split': [min_samples_split_0, min_samples_split_1],
            'min_samples_leaf': [min_samples_leaf_0, min_samples_leaf_1],
            'random_state': random_state
        }
        del criterion_range, max_features_range, n_estimators_0, n_estimators_1, max_depth_0, max_depth_1, min_samples_split_0, min_samples_split_1, min_samples_leaf_0, min_samples_leaf_1, random_state
    
    elif model_name == 'K-Nearest Neighbors':
        st.info('''
            - **K-Nearest Neighbors
            - It is a non-parametric method that does not make any assumptions about the underlying data distribution.
            - It can be used for both classification and regression tasks.\n
            View more about **K-Nearest Neighbors** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
            ''')
        
        def sync_values():
            st.session_state.knn_r_n_neighbors_0 = min(st.session_state.knn_r_n_neighbors_0, st.session_state.knn_r_n_neighbors_1)
            st.session_state.knn_r_n_neighbors_1 = max(st.session_state.knn_r_n_neighbors_0, st.session_state.knn_r_n_neighbors_1)
            st.session_state.knn_r_p_0 = min(st.session_state.knn_r_p_0, st.session_state.knn_r_p_1)
            st.session_state.knn_r_p_1 = max(st.session_state.knn_r_p_0, st.session_state.knn_r_p_1)
        
        # Input Parameters
        weights_range = st.multiselect('**Weights** -> Weight function used in prediction', ['uniform', 'distance'], default=['uniform'])
        algorithm_range = st.multiselect('**Algorithm** -> Algorithm used to compute the nearest neighbors', ['auto', 'ball_tree', 'kd_tree', 'brute'], default=['auto', 'ball_tree', 'kd_tree', 'brute'])
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                n_neighbors_0 = st.number_input('**N Neighbors (1 - 100)**', min_value=1, max_value=100, value=3, key='knn_r_n_neighbors_0', on_change=sync_values)
                p_0 = st.number_input('**P (1 - 5)**', min_value=1, max_value=5, value=1, key='knn_r_p_0', on_change=sync_values)
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                n_neighbors_1 = st.number_input('**N Neighbors (1 - 100)**', min_value=1, max_value=100, value=50, key='knn_r_n_neighbors_1', on_change=sync_values)
                p_1 = st.number_input('**P (1 - 5)**', min_value=1, max_value=5, value=2, key='knn_r_p_1', on_change=sync_values)
        random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42)
        
        # Parameters Configuration Range
        params_range = {
            'weights': weights_range,
            'algorithm': algorithm_range,
            'n_neighbors': [n_neighbors_0, n_neighbors_1],
            'p': [p_0, p_1],
            'random_state': random_state
        }
        del weights_range, algorithm_range, n_neighbors_0, n_neighbors_1, p_0, p_1, random_state
        
    elif model_name == 'Support Vector Machine':
        st.info('''
            - **Support Vector Machine** is a powerful algorithm that can be used for both classification and regression tasks.
            - It can handle both linear and non-linear data.
            - It is effective in high-dimensional spaces and is versatile due to the kernel trick.\n
            View more about **Support Vector Machine** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
            ''')
        def sync_values():
            st.session_state.svm_r_C_0 = min(st.session_state.svm_r_C_0, st.session_state.svm_r_C_1)
            st.session_state.svm_r_C_1 = max(st.session_state.svm_r_C_0, st.session_state.svm_r_C_1)
            st.session_state.svm_r_degree_0 = min(st.session_state.svm_r_degree_0, st.session_state.svm_r_degree_1)
            st.session_state.svm_r_degree_1 = max(st.session_state.svm_r_degree_0, st.session_state.svm_r_degree_1)
            
        # Input Parameters
        kernel_range = st.multiselect('**Kernel** -> Specifies the kernel type to be used in the algorithm', ['linear', 'poly', 'rbf', 'sigmoid'], default=['poly', 'rbf', 'sigmoid'])
        gamma_range = st.multiselect('**Gamma** -> Kernel coefficient for rbf, poly and sigmoid', ['scale', 'auto'], default=['scale', 'auto'])
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                C_0 = st.number_input('**C (0.0001 - 10)**', min_value=0.0001, max_value=10.0, value=0.0001, step=0.0001, format="%.4f", key='svm_r_C_0', on_change=sync_values)
                degree_0 = st.number_input('**Degree (1 - 10)**', min_value=1, max_value=10, value=1, key='svm_r_degree_0', on_change=sync_values)
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                C_1 = st.number_input('**C (0.0001 - 10)**', min_value=0.0001, max_value=10.0, value=0.01, step=0.0001, format="%.4f", key='svm_r_C_1', on_change=sync_values)
                degree_1 = st.number_input('**Degree (1 - 10)**', min_value=1, max_value=10, value=5, key='svm_r_degree_1', on_change=sync_values)
        random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42)
        
        # Parameters Configuration Range
        params_range = {
            'kernel': kernel_range,
            'gamma': gamma_range,
            'C': [C_0, C_1],
            'degree': [degree_0, degree_1],
            'random_state': random_state
        }
        del kernel_range, gamma_range, C_0, C_1, degree_0, degree_1, random_state
        
    elif model_name == 'AdaBoost':
        st.info('''
            - **AdaBoost** is an ensemble learning method that can be used for both classification and regression tasks.
            - It combines multiple weak learners to create a strong learner.
            - It is effective in improving the accuracy of the model and can handle both numerical and categorical data.\n
            View more about **AdaBoost** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
            ''')
        def sync_values():
            st.session_state.ab_r_n_estimators_0 = min(st.session_state.ab_r_n_estimators_0, st.session_state.ab_r_n_estimators_1)
            st.session_state.ab_r_n_estimators_1 = max(st.session_state.ab_r_n_estimators_0, st.session_state.ab_r_n_estimators_1)
            st.session_state.ab_r_learning_rate_0 = min(st.session_state.ab_r_learning_rate_0, st.session_state.ab_r_learning_rate_1)
            st.session_state.ab_r_learning_rate_1 = max(st.session_state.ab_r_learning_rate_0, st.session_state.ab_r_learning_rate_1)
        
        # Input Parameters
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                n_estimators_0 = st.number_input('**N Estimators (10 - 1000)**', min_value=10, max_value=1000, value=10, key='ab_r_n_estimators_0', on_change=sync_values)
                learning_rate_0 = st.number_input('**Learning Rate (0.001 - 10.0)**', min_value=0.001, max_value=10.0, value=0.001, step=0.001, format="%.3f", key='ab_r_learning_rate_0', on_change=sync_values)
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                n_estimators_1 = st.number_input('**N Estimators (10 - 1000)**', min_value=10, max_value=1000, value=100, key='ab_r_n_estimators_1', on_change=sync_values)
                learning_rate_1 = st.number_input('**Learning Rate (0.001 - 10.0)**', min_value=0.001, max_value=10.0, value=1.0, step=0.001, format="%.3f", key='ab_r_learning_rate_1', on_change=sync_values)
        random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42)
        
        # Parameters Configuration Range
        params_range = {
            'n_estimators': [n_estimators_0, n_estimators_1],
            'learning_rate': [learning_rate_0, learning_rate_1],
            'random_state': random_state
        }
        del n_estimators_0, n_estimators_1, learning_rate_0, learning_rate_1, random_state
        
    elif model_name == 'Gradient Boosting':
        st.info('''
            - **Gradient Boosting** is an ensemble learning method that can be used for both classification and regression tasks.
            - It builds multiple decision trees to predict the residuals and adds them to the ensemble.
            - It is effective in improving the accuracy of the model and can handle both numerical and categorical data.\n
            View more about **Gradient Boosting** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
            ''')
        def sync_values():
            st.session_state.gb_r_n_estimators_0 = min(st.session_state.gb_r_n_estimators_0, st.session_state.gb_r_n_estimators_1)
            st.session_state.gb_r_n_estimators_1 = max(st.session_state.gb_r_n_estimators_0, st.session_state.gb_r_n_estimators_1)
            st.session_state.gb_r_learning_rate_0 = min(st.session_state.gb_r_learning_rate_0, st.session_state.gb_r_learning_rate_1)
            st.session_state.gb_r_learning_rate_1 = max(st.session_state.gb_r_learning_rate_0, st.session_state.gb_r_learning_rate_1)
            st.session_state.gb_r_max_depth_0 = min(st.session_state.gb_r_max_depth_0, st.session_state.gb_r_max_depth_1)
            st.session_state.gb_r_max_depth_1 = max(st.session_state.gb_r_max_depth_0, st.session_state.gb_r_max_depth_1)
            st.session_state.gb_r_min_samples_split_0 = min(st.session_state.gb_r_min_samples_split_0, st.session_state.gb_r_min_samples_split_1)
            st.session_state.gb_r_min_samples_split_1 = max(st.session_state.gb_r_min_samples_split_0, st.session_state.gb_r_min_samples_split_1)
            st.session_state.gb_r_min_samples_leaf_0 = min(st.session_state.gb_r_min_samples_leaf_0, st.session_state.gb_r_min_samples_leaf_1)
            st.session_state.gb_r_min_samples_leaf_1 = max(st.session_state.gb_r_min_samples_leaf_0, st.session_state.gb_r_min_samples_leaf_1)
        
        # Input Parameters
        criterion_range = st.multiselect('**Criterion** -> The function to measure the quality of a split', ['squared_error', 'friedman_mse'], default=['friedman_mse'])
        loss_range = st.multiselect('**Loss** -> The loss function to be optimized', ['squared_error', 'absolute_error', 'huber', 'quantile'], default=['squared_error', 'absolute_error', 'huber'])
        max_features_range = st.multiselect('**Max Features** -> The number of features to consider when looking for the best split', [None, 'sqrt', 'log2'], default=[None, 'sqrt', 'log2'])
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                n_estimators_0 = st.number_input('**N Estimators (10 - 1000)**', min_value=10, max_value=1000, value=10, key='gb_r_n_estimators_0', on_change=sync_values)
                learning_rate_0 = st.number_input('**Learning Rate (0.001 - 10.0)**', min_value=0.001, max_value=10.0, value=0.001, step=0.001, format="%.3f", key='gb_r_learning_rate_0', on_change=sync_values)
                max_depth_0 = st.number_input('**Max Depth (1 - 100)**', min_value=1, max_value=100, value=3, key='gb_r_max_depth_0', on_change=sync_values)
                min_samples_split_0 = st.number_input('**Min Samples Split (2 - 50)**', min_value=2, max_value=50, value=2, key='gb_r_min_samples_split_0', on_change=sync_values)
                min_samples_leaf_0 = st.number_input('**Min Samples Leaf (1 - 50)**', min_value=1, max_value=50, value=1, key='gb_r_min_samples_leaf_0', on_change=sync_values)
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                n_estimators_1 = st.number_input('**N Estimators (10 - 1000)**', min_value=10, max_value=1000, value=100, key='gb_r_n_estimators_1', on_change=sync_values)
                learning_rate_1 = st.number_input('**Learning Rate (0.001 - 10.0)**', min_value=0.001, max_value=10.0, value=1.0, step=0.001, format="%.3f", key='gb_r_learning_rate_1', on_change=sync_values)
                max_depth_1 = st.number_input('**Max Depth (1 - 100)**', min_value=1, max_value=100, value=10, key='gb_r_max_depth_1', on_change=sync_values)
                min_samples_split_1 = st.number_input('**Min Samples Split (2 - 50)**', min_value=2, max_value=50, value=10, key='gb_r_min_samples_split_1', on_change=sync_values)
                min_samples_leaf_1 = st.number_input('**Min Samples Leaf (1 - 50)**', min_value=1, max_value=50, value=10, key='gb_r_min_samples_leaf_1', on_change=sync_values)
        random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42)
        
        # Parameters Configuration Range
        params_range = {
            'criterion': criterion_range,
            'loss': loss_range,
            'max_features': max_features_range,
            'n_estimators': [n_estimators_0, n_estimators_1],
            'learning_rate': [learning_rate_0, learning_rate_1],
            'max_depth': [max_depth_0, max_depth_1],
            'min_samples_split': [min_samples_split_0, min_samples_split_1],
            'min_samples_leaf': [min_samples_leaf_0, min_samples_leaf_1],
            'random_state': random_state
        }
        del criterion_range, loss_range, max_features_range, n_estimators_0, n_estimators_1, learning_rate_0, learning_rate_1, max_depth_0, max_depth_1, min_samples_split_0, min_samples_split_1, min_samples_leaf_0, min_samples_leaf_1, random_state
    
    elif model_name == 'Multi-Layer Perceptron (Neural Network)':
        st.info('''
            - **Multi-Layer Perceptron** is a feedforward artificial neural network that can be used for both classification and regression tasks.
            - It can handle both numerical and categorical data.
            - It is effective in learning complex relationships in the data and can be used for deep learning tasks.\n
            View more about **Multi-Layer Perceptron** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
            ''')
        def sync_values():
            st.session_state.mlp_r_N_layers_0 = min(st.session_state.mlp_r_N_layers_0, st.session_state.mlp_r_N_layers_1)
            st.session_state.mlp_r_N_layers_1 = max(st.session_state.mlp_r_N_layers_0, st.session_state.mlp_r_N_layers_1)
            st.session_state.mlp_r_neurons_0 = min(st.session_state.mlp_r_neurons_0, st.session_state.mlp_r_neurons_1)
            st.session_state.mlp_r_neurons_1 = max(st.session_state.mlp_r_neurons_0, st.session_state.mlp_r_neurons_1)
            st.session_state.mlp_r_learning_rate_init_0 = min(st.session_state.mlp_r_learning_rate_init_0, st.session_state.mlp_r_learning_rate_init_1)
            st.session_state.mlp_r_learning_rate_init_1 = max(st.session_state.mlp_r_learning_rate_init_0, st.session_state.mlp_r_learning_rate_init_1)
            st.session_state.mlp_r_alpha_0 = min(st.session_state.mlp_r_alpha_0, st.session_state.mlp_r_alpha_1)
            st.session_state.mlp_r_alpha_1 = max(st.session_state.mlp_r_alpha_0, st.session_state.mlp_r_alpha_1)
            st.session_state.mlp_r_max_iter_0 = min(st.session_state.mlp_r_max_iter_0, st.session_state.mlp_r_max_iter_1)
            st.session_state.mlp_r_max_iter_1 = max(st.session_state.mlp_r_max_iter_0, st.session_state.mlp_r_max_iter_1)
        
        # Input Parameters
        activation_range = st.multiselect('**Activation** -> Activation function for the hidden layer', ['identity', 'logistic', 'tanh', 'relu'], default=['relu'])
        solver_range = st.multiselect('**Solver** -> The solver for weight optimization', ['lbfgs', 'sgd', 'adam'], default=['adam'])
        learning_rate = st.multiselect('**Learning Rate** -> The learning rate schedule for weight updates', ['constant', 'invscaling', 'adaptive'], default=['constant', 'adaptive'])
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                N_layers_0 = st.number_input('**N Layers (1 - 10)**', min_value=1, max_value=10, value=1, key='mlp_r_N_layers_0', on_change=sync_values)
                neurons_0 = st.number_input('**Neurons for each layer (1 - 1000)**', min_value=1, max_value=1000, value=10, key='mlp_r_neurons_0', on_change=sync_values)
                learning_rate_init_0 = st.number_input('**Learning Rate Init (0.001 - 1.0)**', min_value=0.001, max_value=1.0, value=0.001, step=0.001, format="%.3f", key='mlp_r_learning_rate_init_0', on_change=sync_values)
                alpha_0 = st.number_input('**Alpha (0.0001 - 0.1)**', min_value=0.0001, max_value=0.1, value=0.0001, step=0.0001, format="%.4f", key='mlp_r_alpha_0', on_change=sync_values)
                max_iter_0 = st.number_input('**Max Iter (100 - 1000)**', min_value=100, max_value=1000, value=200, key='mlp_r_max_iter_0', on_change=sync_values)
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                N_layers_1 = st.number_input('**N Layers (1 - 10)**', min_value=1, max_value=10, value=3, key='mlp_r_N_layers_1', on_change=sync_values)
                neurons_1 = st.number_input('**Neurons for each layer (1 - 1000)**', min_value=1, max_value=1000, value=100, key='mlp_r_neurons_1', on_change=sync_values)
                learning_rate_init_1 = st.number_input('**Learning Rate Init (0.001 - 1.0)**', min_value=0.001, max_value=1.0, value=0.5, step=0.001, format="%.3f", key='mlp_r_learning_rate_init_1', on_change=sync_values)
                alpha_1 = st.number_input('**Alpha (0.0001 - 0.1)**', min_value=0.0001, max_value=0.1, value=0.01, step=0.0001, format="%.4f", key='mlp_r_alpha_1', on_change=sync_values)
                max_iter_1 = st.number_input('**Max Iter (100 - 1000)**', min_value=100, max_value=1000, value=200, key='mlp_r_max_iter_1', on_change=sync_values)
        random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42)
        
        # Parameters Configuration Range
        params_range = {
            'activation': activation_range,
            'solver': solver_range,
            'learning_rate': learning_rate,
            'n_layers': [N_layers_0, N_layers_1],
            'neurons': [neurons_0, neurons_1],
            'learning_rate_init': [learning_rate_init_0, learning_rate_init_1],
            'alpha': [alpha_0, alpha_1],
            'max_iter': [max_iter_0, max_iter_1],
            'random_state': random_state
        }
        del activation_range, solver_range, learning_rate, N_layers_0, N_layers_1, neurons_0, neurons_1, learning_rate_init_0, learning_rate_init_1, alpha_0, alpha_1, max_iter_0, max_iter_1, random_state
    
    return params_range











# Get the best model with optimized hyperparameters
def get_best_r_model(model_choice, params_range, x_train, y_train):
    if model_choice == 'Linear Regression':
        def objective(trial):
            fit_intercept = trial.suggest_categorical('fit_intercept', params_range['fit_intercept'])
            model = LinearRegression(fit_intercept=fit_intercept, n_jobs=-1)
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        best_model = LinearRegression(**best_params, n_jobs=-1)
        
    elif model_choice == 'Lasso Regression':
        def objective(trial):
            fit_intercept = trial.suggest_categorical('fit_intercept', params_range['fit_intercept'])
            alpha = trial.suggest_float('alpha', params_range['alpha'][0], params_range['alpha'][1])
            max_iter = trial.suggest_int('max_iter', params_range['max_iter'][0], params_range['max_iter'][1])
            model = Lasso(fit_intercept=fit_intercept, alpha=alpha, max_iter=max_iter, random_state=params_range['random_state'])
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        best_model = Lasso(**best_params, random_state=params_range['random_state'])
    
    elif model_choice == 'Ridge Regression':
        def objective(trial):
            fit_intercept = trial.suggest_categorical('fit_intercept', params_range['fit_intercept'])
            alpha = trial.suggest_float('alpha', params_range['alpha'][0], params_range['alpha'][1])
            max_iter = trial.suggest_int('max_iter', params_range['max_iter'][0], params_range['max_iter'][1])
            model = Ridge(fit_intercept=fit_intercept, alpha=alpha, max_iter=max_iter, random_state=params_range['random_state'])
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        best_model = Ridge(**best_params, random_state=params_range['random_state'])
    
    elif model_choice == 'Decision Tree':
        def objective(trial):
            criterion = trial.suggest_categorical('criterion', params_range['criterion'])
            max_features = trial.suggest_categorical('max_features', params_range['max_features'])
            max_depth = trial.suggest_int('max_depth', params_range['max_depth'][0], params_range['max_depth'][1])
            min_samples_split = trial.suggest_int('min_samples_split', params_range['min_samples_split'][0], params_range['min_samples_split'][1])
            min_samples_leaf = trial.suggest_int('min_samples_leaf', params_range['min_samples_leaf'][0], params_range['min_samples_leaf'][1])
            model = DecisionTreeRegressor(criterion=criterion, max_features=max_features, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=params_range['random_state'])
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        best_model = DecisionTreeRegressor(**best_params, random_state=params_range['random_state'])
        
    elif model_choice == 'Random Forest':
        def objective(trial):
            criterion = trial.suggest_categorical('criterion', params_range['criterion'])
            max_features = trial.suggest_categorical('max_features', params_range['max_features'])
            n_estimators = trial.suggest_int('n_estimators', params_range['n_estimators'][0], params_range['n_estimators'][1])
            max_depth = trial.suggest_int('max_depth', params_range['max_depth'][0], params_range['max_depth'][1])
            min_samples_split = trial.suggest_int('min_samples_split', params_range['min_samples_split'][0], params_range['min_samples_split'][1])
            min_samples_leaf = trial.suggest_int('min_samples_leaf', params_range['min_samples_leaf'][0], params_range['min_samples_leaf'][1])
            model = RandomForestRegressor(criterion=criterion, max_features=max_features, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=params_range['random_state'], n_jobs=-1)
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        best_model = RandomForestRegressor(**best_params, random_state=params_range['random_state'], n_jobs=-1)
    
    elif model_choice == 'K-Nearest Neighbors':
        def objective(trial):
            weights = trial.suggest_categorical('weights', params_range['weights'])
            algorithm = trial.suggest_categorical('algorithm', params_range['algorithm'])
            n_neighbors = trial.suggest_int('n_neighbors', params_range['n_neighbors'][0], params_range['n_neighbors'][1])
            p = trial.suggest_int('p', params_range['p'][0], params_range['p'][1])
            model = KNeighborsRegressor(weights=weights, algorithm=algorithm, n_neighbors=n_neighbors, p=p, n_jobs=-1)
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        best_model = KNeighborsRegressor(**best_params, n_jobs=-1)
    
    elif model_choice == 'Support Vector Machine':
        def objective(trial):
            kernel = trial.suggest_categorical('kernel', params_range['kernel'])
            gamma = trial.suggest_categorical('gamma', params_range['gamma'])
            C = trial.suggest_float('C', params_range['C'][0], params_range['C'][1], log=True)
            degree = trial.suggest_int('degree', params_range['degree'][0], params_range['degree'][1])
            model = SVR(kernel=kernel, gamma=gamma, C=C, degree=degree)
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        best_model = SVR(**best_params)
    
    elif model_choice == 'AdaBoost':
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', params_range['n_estimators'][0], params_range['n_estimators'][1])
            learning_rate = trial.suggest_float('learning_rate', params_range['learning_rate'][0], params_range['learning_rate'][1], log=True)
            model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=params_range['random_state'])
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        best_model = AdaBoostRegressor(**best_params, random_state=params_range['random_state'])
        
    elif model_choice == 'Gradient Boosting':
        def objective(trial):
            criterion = trial.suggest_categorical('criterion', params_range['criterion'])
            loss = trial.suggest_categorical('loss', params_range['loss'])
            max_features = trial.suggest_categorical('max_features', params_range['max_features'])
            n_estimators = trial.suggest_int('n_estimators', params_range['n_estimators'][0], params_range['n_estimators'][1])
            learning_rate = trial.suggest_float('learning_rate', params_range['learning_rate'][0], params_range['learning_rate'][1], log=True)
            max_depth = trial.suggest_int('max_depth', params_range['max_depth'][0], params_range['max_depth'][1])
            min_samples_split = trial.suggest_int('min_samples_split', params_range['min_samples_split'][0], params_range['min_samples_split'][1])
            min_samples_leaf = trial.suggest_int('min_samples_leaf', params_range['min_samples_leaf'][0], params_range['min_samples_leaf'][1])
            model = GradientBoostingRegressor(criterion=criterion, loss=loss, max_features=max_features, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=params_range['random_state'])
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        best_model = GradientBoostingRegressor(**best_params, random_state=params_range['random_state'])
        
    elif model_choice == 'Multi-Layer Perceptron (Neural Network)':
        def objective(trial):
            activation = trial.suggest_categorical('activation', params_range['activation'])
            solver = trial.suggest_categorical('solver', params_range['solver'])
            learning_rate = trial.suggest_categorical('learning_rate', params_range['learning_rate'])
            n_layers = trial.suggest_int('n_layers', params_range['n_layers'][0], params_range['n_layers'][1])
            neurons = [trial.suggest_int(f'n_neurons_{i}', params_range['neurons'][0], params_range['neurons'][1]) for i in range(n_layers)]
            learning_rate_init = trial.suggest_float('learning_rate_init', params_range['learning_rate_init'][0], params_range['learning_rate_init'][1], log=True)
            alpha = trial.suggest_float('alpha', params_range['alpha'][0], params_range['alpha'][1], log=True)
            max_iter = trial.suggest_int('max_iter', params_range['max_iter'][0], params_range['max_iter'][1])
            model = MLPRegressor(activation=activation, solver=solver, learning_rate=learning_rate, hidden_layer_sizes=tuple(neurons), learning_rate_init=learning_rate_init, alpha=alpha, max_iter=max_iter, random_state=params_range['random_state'])
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        n_layers = best_params.pop('n_layers')
        hidden_layer_sizes = tuple(best_params.pop(f'n_neurons_{i}') for i in range(n_layers))
        best_params['hidden_layer_sizes'] = hidden_layer_sizes
        best_model = MLPRegressor(**best_params, random_state=params_range['random_state'])
    
    del study, best_params
    gc.collect()
    return best_model
