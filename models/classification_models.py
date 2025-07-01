import streamlit as st
import optuna as opt
import gc
from sklearn.model_selection import cross_val_score

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
    C = st.number_input('**C (0.0001 - 100)** -> Inverse of regularization strength, must be a positive float', min_value=0.0001, max_value=100.0, value=1.0, step=0.0001, format="%.4f")
    max_iter = st.number_input('**Max Iteration (100 - 1000)** -> Maximum number of iterations taken for the solvers to converge', min_value=100, max_value=1000,  value=100, step=50)
    
    if penalty == 'elasticnet':
        l1_ratio = st.number_input('**L1 Ratio (0 - 1)** -> The Elastic-Net mixing parameter', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
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
    max_depth = st.number_input('**Max Depth (0 - 100)** -> The maximum depth of the tree (0 means None or Auto)', min_value=0, max_value=100, value=0, step=1)
    min_samples_split = st.number_input('**Min Samples Split (2 - 50)** -> The minimum number of samples required to split an internal node', min_value=2, max_value=50, value=2, step=1)
    min_samples_leaf = st.number_input('**Min Samples Leaf (1 - 50)** -> The minimum number of samples required to be at a leaf node', min_value=1, max_value=50, value=1, step=1)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42, step=1)
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
    n_estimators = st.number_input('**N Estimators (10 - 1000)** -> The number of trees in the forest', min_value=10, max_value=1000, value=100, step=5)
    max_depth = st.number_input('**Max Depth (0 - 100)** -> The maximum depth of the tree (0 means None or Auto)', min_value=0, max_value=100, value=0, step=1)
    min_samples_split = st.number_input('**Min Samples Split (2 - 50)** -> The minimum number of samples required to split an internal node', min_value=2, max_value=50, value=2, step=1)
    min_samples_leaf = st.number_input('**Min Samples Leaf (1 - 50)** -> The minimum number of samples required to be at a leaf node', min_value=1, max_value=50, value=1, step=1)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42, step=1)
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
    n_neighbors = st.number_input('**N Neighbors (1 - 100)** -> Number of neighbors to use', min_value=1, max_value=100, value=5, step=1)
    p = st.number_input('**P (1 - 5)** -> Power parameter for the Minkowski metric', min_value=1, max_value=5, value=2, step=1)
    
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
    kernel = st.selectbox('**Kernel** -> Specifies the kernel type to be used in the algorithm', options=['linear', 'poly', 'rbf', 'sigmoid'], index=2)
    gamma = st.selectbox('**Gamma** -> Kernel coefficient for rbf, poly and sigmoid', options=['scale', 'auto'], index=0)
    C = st.number_input('**C (0.0001 - 10)** -> Regularization parameter', min_value=0.0001, max_value=10.0, value=1.0, step=0.0001, format="%.4f")
    degree = st.number_input('**Degree (1 - 10)** -> Degree of the polynomial kernel function (poly)', min_value=1, max_value=10, value=3, step=1)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42, step=1)
    
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
    
    var_smoothing = st.number_input('**Var Smoothing (1e-9 - 0.001)** -> Portion of the largest variance of all features added to variances for calculation stability', min_value=1e-9, max_value=0.001, value=1e-9, step=1e-9, format="%.9f")
    
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
    
    n_estimators = st.number_input('**N Estimators (10 - 1000)** -> The maximum number of estimators at which boosting is terminated', min_value=10, max_value=1000, value=50, step=5)
    learning_rate = st.number_input('**Learning Rate (0.001 - 10.0)** -> Weight applied to each classifier at each boosting iteration', min_value=0.001, max_value=10.0, value=0.1, step=0.001, format="%.3f")
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42, step=1)
    
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
    criterion = st.selectbox('**Criterion** -> The function to measure the quality of a split', options=['friedman_mse', 'squared_error'], index=0)
    loss = st.selectbox('**Loss** -> Loss function to be optimized', options=['log_loss', 'exponential'], index=0)
    max_features = st.selectbox('**Max Features** -> The number of features to consider when looking for the best split', options=[None, 'sqrt', 'log2'], index=0)
    learning_rate = st.number_input('**Learning Rate (0.001 - 10.0)** -> Learning rate shrinks the contribution of each tree by learning_rate', min_value=0.001, max_value=10.0, value=0.1, step=0.001, format="%.3f")
    n_estimators = st.number_input('**N Estimators (10 - 1000)** -> The number of boosting stages to be run', min_value=10, max_value=1000, step=5, value=100)
    subsample = st.number_input('**Subsample (0.01 - 1.0)** -> The fraction of samples to be used for fitting the individual base learners', min_value=0.01, max_value=1.0, value=1.0, step=0.01, format="%.2f")
    max_depth = st.number_input('**Max Depth (1 - 100)** -> The maximum depth of the tree', min_value=1, max_value=100, step=1, value=3)
    min_samples_split = st.number_input('**Min Samples Split (2 - 50)** -> The minimum number of samples required to split an internal node', min_value=2, max_value=50, value=2, step=1)
    min_samples_leaf = st.number_input('**Min Samples Leaf (1 - 50)** -> The minimum number of samples required to be at a leaf node', min_value=1, max_value=50, value=1, step=1)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42, step=1)
    
    params = {'criterion': criterion, 'loss': loss, 'max_features': max_features, 'learning_rate': learning_rate, 'n_estimators': n_estimators, 'subsample': subsample, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'random_state': random_state}
    model = GradientBoostingClassifier(**params)
    return model

def mlp_c_param_selector():
    st.info('''
            - **Multi-Layer Perceptron** is a feedforward artificial neural network that can be used for both classification and regression tasks.
            - It can handle both numerical and categorical data.
            - It is effective in learning complex relationships in the data and can be used for deep learning tasks.\n
            View more about **Multi-Layer Perceptron** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
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
    learning_rate_init = st.number_input('**Learning Rate Init (0.00001 - 1.0)** -> The initial learning rate used', min_value=0.00001, max_value=1.0, value=0.001, step=0.00001, format="%.5f")
    alpha = st.number_input('**Alpha (0.0001 - 1.0)** -> L2 penalty (regularization term) parameter', min_value=0.0001, max_value=1.0, value=0.0001, step=0.001, format="%.4f")
    max_iter = st.number_input('**Max Iteration (10 - 1000)** -> Maximum number of iterations (Epoch)', min_value=10, max_value=1000, value=200, step=10)
    random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42, step=1)
    
    params = {'hidden_layer_sizes': layer_sizes, 'activation': activation, 'solver': solver, 'learning_rate': learning_rate, 'learning_rate_init': learning_rate_init, 'alpha': alpha, 'max_iter': max_iter, 'random_state': random_state}
    model = MLPClassifier(**params)
    return model










# TREE-STRUCTURED PARZEN ESTIMATOR (TPE) HYPERPARAMETER OPTIMIZATION

# Get the range of hyperparameters for the selected model
def get_c_param_range(model_name):
    if model_name == 'Logistic Regression':
        st.info('''
            - **Logistic Regression** is simple and easy to implement.
            - It provides probabilities and can be used for binary classification.
            - It performs well when the dataset is linearly separable.\n
            View more about **Logistic Regression** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
            ''')
        def sync_values():
            st.session_state.C_0 = min(st.session_state.C_0, st.session_state.C_1)
            st.session_state.C_1 = max(st.session_state.C_0, st.session_state.C_1)
            st.session_state.max_iter_0 = min(st.session_state.max_iter_0, st.session_state.max_iter_1)
            st.session_state.max_iter_1 = max(st.session_state.max_iter_0, st.session_state.max_iter_1)
            
        # Input Parameters             
        solver_range = st.multiselect('**Solver** -> Algorithm to use in the optimization problem', ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'], default=['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'])
        penalty_range = st.multiselect('**Penalty** -> Norm used in the penalization', [None, 'l2'], default=['l2'])
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                C_0 = st.number_input('**C (0.0001 - 100)**', min_value=0.0001, max_value=100.0, value=0.0001, step=0.0001, format="%.4f", key='C_0', on_change=sync_values)
                max_iter_0 = st.number_input('**Max Iteration (100 - 1000)**',min_value=100, max_value=1000, value=100, key='max_iter_0', on_change=sync_values)
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                C_1 = st.number_input('**C (0.0001 - 100)**', min_value=0.0001, max_value=100.0, value=100.0, step=0.0001, format="%.4f", key='C_1', on_change=sync_values)
                max_iter_1 = st.number_input('**Max Iteration (100 - 1000)**', min_value=100, max_value=1000, value=100, key='max_iter_1', on_change=sync_values)
        random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42)
        
        # Parameters Configuration Range
        params_range = {
            'solver': solver_range,
            'penalty': penalty_range,
            'C': [C_0, C_1],
            'max_iter': [max_iter_0, max_iter_1],
            'random_state': random_state
        }
        del solver_range, penalty_range, C_0, C_1, max_iter_0, max_iter_1, random_state
    
    elif model_name == 'Decision Tree':
        st.info('''
            - **Decision Tree** is simple and easy to understand.
            - It can handle both numerical and categorical data.
            - It can be used for both classification and regression tasks.\n
            View more about **Decision Tree** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
            ''')
        def sync_values():
            st.session_state.dt_c_max_depth_0 = min(st.session_state.dt_c_max_depth_0, st.session_state.dt_c_max_depth_1)
            st.session_state.dt_c_max_depth_1 = max(st.session_state.dt_c_max_depth_0, st.session_state.dt_c_max_depth_1)
            st.session_state.dt_c_min_samples_split_0 = min(st.session_state.dt_c_min_samples_split_0, st.session_state.dt_c_min_samples_split_1)
            st.session_state.dt_c_min_samples_split_1 = max(st.session_state.dt_c_min_samples_split_0, st.session_state.dt_c_min_samples_split_1)
            st.session_state.dt_c_min_samples_leaf_0 = min(st.session_state.dt_c_min_samples_leaf_0, st.session_state.dt_c_min_samples_leaf_1)
            st.session_state.dt_c_min_samples_leaf_1 = max(st.session_state.dt_c_min_samples_leaf_0, st.session_state.dt_c_min_samples_leaf_1)
        
        # Input Parameters             
        criterion_range = st.multiselect('**Criterion** -> The function to measure the quality of a split', ['gini', 'entropy', 'log_loss'], default=['gini', 'entropy', 'log_loss'])
        max_features_range = st.multiselect('**Max Features** -> The number of features to consider when looking for the best split', [None, 'sqrt', 'log2'], default=[None, 'sqrt', 'log2'])
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                max_depth_0 = st.number_input('**Max Depth (1 - 100)**', min_value=1, max_value=100, value=3, key='dt_c_max_depth_0', on_change=sync_values)
                min_samples_split_0 = st.number_input('**Min Samples Split (2 - 50)**', min_value=2, max_value=50, value=2, key='dt_c_min_samples_split_0', on_change=sync_values)
                min_samples_leaf_0 = st.number_input('**Min Samples Leaf (1 - 50)**', min_value=1, max_value=50, value=1, key='dt_c_min_samples_leaf_0', on_change=sync_values)
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                max_depth_1 = st.number_input('**Max Depth (1 - 100)**', min_value=1, max_value=100, value=10, key='dt_c_max_depth_1', on_change=sync_values)
                min_samples_split_1 = st.number_input('**Min Samples Split (2 - 50)**', min_value=2, max_value=50, value=10, key='dt_c_min_samples_split_1', on_change=sync_values)
                min_samples_leaf_1 = st.number_input('**Min Samples Leaf (1 - 50)**', min_value=1, max_value=50, value = 10, key='dt_c_min_samples_leaf_1', on_change=sync_values)
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
            View more about **Random Forest** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
            ''')
        def sync_values():
            st.session_state.rf_c_n_estimators_0 = min(st.session_state.rf_c_n_estimators_0, st.session_state.rf_c_n_estimators_1)
            st.session_state.rf_c_n_estimators_1 = max(st.session_state.rf_c_n_estimators_0, st.session_state.rf_c_n_estimators_1)
            st.session_state.rf_c_max_depth_0 = min(st.session_state.rf_c_max_depth_0, st.session_state.rf_c_max_depth_1)
            st.session_state.rf_c_max_depth_1 = max(st.session_state.rf_c_max_depth_0, st.session_state.rf_c_max_depth_1)
            st.session_state.rf_c_min_samples_split_0 = min(st.session_state.rf_c_min_samples_split_0, st.session_state.rf_c_min_samples_split_1)
            st.session_state.rf_c_min_samples_split_1 = max(st.session_state.rf_c_min_samples_split_0, st.session_state.rf_c_min_samples_split_1)
            st.session_state.rf_c_min_samples_leaf_0 = min(st.session_state.rf_c_min_samples_leaf_0, st.session_state.rf_c_min_samples_leaf_1)
            st.session_state.rf_c_min_samples_leaf_1 = max(st.session_state.rf_c_min_samples_leaf_0, st.session_state.rf_c_min_samples_leaf_1)
        
        # Input Parameters    
        criterion_range = st.multiselect('**Criterion** -> The function to measure the quality of a split', ['gini', 'entropy', 'log_loss'], default=['gini', 'entropy', 'log_loss'])
        max_features_range = st.multiselect('**Max Features** -> The number of features to consider when looking for the best split', [None, 'sqrt', 'log2'], default=[None, 'sqrt', 'log2'])
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                n_estimators_0 = st.number_input('**N Estimators (10 - 1000)**', min_value=10, max_value=1000, value=10, key='rf_c_n_estimators_0', on_change=sync_values)
                max_depth_0 = st.number_input('**Max Depth (1 - 100)**', min_value=1, max_value=100, value=3, key='rf_c_max_depth_0', on_change=sync_values)
                min_samples_split_0 = st.number_input('**Min Samples Split (2 - 50)**', min_value=2, max_value=50, value=2, key='rf_c_min_samples_split_0', on_change=sync_values)
                min_samples_leaf_0 = st.number_input('**Min Samples Leaf (1 - 50)**', min_value=1, max_value=50, value=1, key='rf_c_min_samples_leaf_0', on_change=sync_values)
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                n_estimators_1 = st.number_input('**N Estimators (10 - 1000)**', min_value=10, max_value=1000, value=100, key='rf_c_n_estimators_1', on_change=sync_values)
                max_depth_1 = st.number_input('**Max Depth (1 - 100)**', min_value=1, max_value=100, value=10, key='rf_c_max_depth_1', on_change=sync_values)
                min_samples_split_1 = st.number_input('**Min Samples Split (2 - 50)**', min_value=2, max_value=50, value=10, key='rf_c_min_samples_split_1', on_change=sync_values)
                min_samples_leaf_1 = st.number_input('**Min Samples Leaf (1 - 50)**', min_value=1, max_value=50, value=10, key='rf_c_min_samples_leaf_1', on_change=sync_values)
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
            - **K-Nearest Neighbors** is a simple and easy-to-implement algorithm.
            - It can be used for both classification and regression tasks.
            - It is a non-parametric method that does not make any assumptions about the underlying data distribution.\n
            View more about **K-Nearest Neighbors** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
            ''')
        def sync_values():
            st.session_state.knn_c_n_neighbors_0 = min(st.session_state.knn_c_n_neighbors_0, st.session_state.knn_c_n_neighbors_1)
            st.session_state.knn_c_n_neighbors_1 = max(st.session_state.knn_c_n_neighbors_0, st.session_state.knn_c_n_neighbors_1)
            st.session_state.knn_c_p_0 = min(st.session_state.knn_c_p_0, st.session_state.knn_c_p_1)
            st.session_state.knn_c_p_1 = max(st.session_state.knn_c_p_0, st.session_state.knn_c_p_1)
        
        # Input Parameters
        weights_range = st.multiselect('**Weights** -> Weight function used in prediction', ['uniform', 'distance'], default=['uniform'])
        algorithm_range = st.multiselect('**Algorithm** -> Algorithm used to compute the nearest neighbors', ['auto', 'ball_tree', 'kd_tree', 'brute'], default=['auto', 'ball_tree', 'kd_tree', 'brute'])
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                n_neighbors_0 = st.number_input('**N Neighbors (1 - 100)**', min_value=1, max_value=100, value=3, key='knn_c_n_neighbors_0', on_change=sync_values)
                p_0 = st.number_input('**P (1 - 5)**', min_value=1, max_value=5, value=1, key='knn_c_p_0', on_change=sync_values)
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                n_neighbors_1 = st.number_input('**N Neighbors (1 - 100)**', min_value=1, max_value=100, value=50, key='knn_c_n_neighbors_1', on_change=sync_values)
                p_1 = st.number_input('**P (1 - 5)**', min_value=1, max_value=5, value=2, key='knn_c_p_1', on_change=sync_values)
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
            st.session_state.svm_c_C_0 = min(st.session_state.svm_c_C_0, st.session_state.svm_c_C_1)
            st.session_state.svm_c_C_1 = max(st.session_state.svm_c_C_0, st.session_state.svm_c_C_1)
            st.session_state.svm_c_degree_0 = min(st.session_state.svm_c_degree_0, st.session_state.svm_c_degree_1)
            st.session_state.svm_c_degree_1 = max(st.session_state.svm_c_degree_0, st.session_state.svm_c_degree_1)
            
        # Input Parameters
        kernel_range = st.multiselect('**Kernel** -> Specifies the kernel type to be used in the algorithm', ['linear', 'poly', 'rbf', 'sigmoid'], default=['poly', 'rbf', 'sigmoid'])
        gamma_range = st.multiselect('**Gamma** -> Kernel coefficient for rbf, poly and sigmoid', ['scale', 'auto'], default=['scale', 'auto'])
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                C_0 = st.number_input('**C (0.0001 - 10)**', min_value=0.0001, max_value=10.0, value=0.0001, step=0.0001, format="%.4f", key='svm_c_C_0', on_change=sync_values)
                degree_0 = st.number_input('**Degree (1 - 10)**', min_value=1, max_value=10, value=1, key='svm_c_degree_0', on_change=sync_values)
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                C_1 = st.number_input('**C (0.0001 - 10)**', min_value=0.0001, max_value=10.0, value=0.01, step=0.0001, format="%.4f", key='svm_c_C_1', on_change=sync_values)
                degree_1 = st.number_input('**Degree (1 - 10)**', min_value=1, max_value=10, value=5, key='svm_c_degree_1', on_change=sync_values)
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
    
    elif model_name == 'Naive Bayes':
        st.info('''
            - **Naive Bayes** is a simple and easy-to-implement algorithm.
            - It is based on Bayes' theorem and assumes that features are independent of each other.
            - It is effective for large datasets and can be used for both binary and multi-class classification tasks.\n
            View more about **Naive Bayes** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
            ''')
        def sync_values():
            st.session_state.nb_var_smoothing_0 = min(st.session_state.nb_var_smoothing_0, st.session_state.nb_var_smoothing_1)
            st.session_state.nb_var_smoothing_1 = max(st.session_state.nb_var_smoothing_0, st.session_state.nb_var_smoothing_1)
        
        # Input Parameters
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                var_smoothing_0 = st.number_input('**Var Smoothing (1e-9 - 0.001)**', min_value=1e-9, max_value=0.001, value=1e-9, step=1e-9, format="%.9f", key='nb_var_smoothing_0', on_change=sync_values)
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                var_smoothing_1 = st.number_input('**Var Smoothing (1e-9 - 0.001)**', min_value=1e-9, max_value=0.001, value=1e-7, step=1e-9, format="%.9f", key='nb_var_smoothing_1', on_change=sync_values)
        random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42)
        
        # Parameters Configuration Range
        params_range = {
            'var_smoothing': [var_smoothing_0, var_smoothing_1],
            'random_state': random_state
        }
        del var_smoothing_0, var_smoothing_1, random_state
        
    elif model_name == 'AdaBoost':
        st.info('''
            - **AdaBoost** is an ensemble learning method that can be used for both classification and regression tasks.
            - It combines multiple weak learners to create a strong learner.
            - It is effective in improving the accuracy of the model and can handle both numerical and categorical data.\n
            View more about **AdaBoost** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
            ''')
        def sync_values():
            st.session_state.ab_c_n_estimators_0 = min(st.session_state.ab_c_n_estimators_0, st.session_state.ab_c_n_estimators_1)
            st.session_state.ab_c_n_estimators_1 = max(st.session_state.ab_c_n_estimators_0, st.session_state.ab_c_n_estimators_1)
            st.session_state.ab_c_learning_rate_0 = min(st.session_state.ab_c_learning_rate_0, st.session_state.ab_c_learning_rate_1)
            st.session_state.ab_c_learning_rate_1 = max(st.session_state.ab_c_learning_rate_0, st.session_state.ab_c_learning_rate_1)
        
        # Input Parameters
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                n_estimators_0 = st.number_input('**N Estimators (10 - 1000)**', min_value=10, max_value=1000, value=10, key='ab_c_n_estimators_0', on_change=sync_values)
                learning_rate_0 = st.number_input('**Learning Rate (0.001 - 10.0)**', min_value=0.001, max_value=10.0, value=0.001, step=0.001, format="%.3f", key='ab_c_learning_rate_0', on_change=sync_values)
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                n_estimators_1 = st.number_input('**N Estimators (10 - 1000)**', min_value=10, max_value=1000, value=100, key='ab_c_n_estimators_1', on_change=sync_values)
                learning_rate_1 = st.number_input('**Learning Rate (0.001 - 10.0)**', min_value=0.001, max_value=10.0, value=1.0, step=0.001, format="%.3f", key='ab_c_learning_rate_1', on_change=sync_values)
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
            - It builds trees one at a time, where each new tree helps to correct errors made by the previously trained tree.
            - It is effective in improving the accuracy of the model and can handle both numerical and categorical data.\n
            View more about **Gradient Boosting** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
            ''')
        def sync_values():
            st.session_state.gb_c_n_estimators_0 = min(st.session_state.gb_c_n_estimators_0, st.session_state.gb_c_n_estimators_1)
            st.session_state.gb_c_n_estimators_1 = max(st.session_state.gb_c_n_estimators_0, st.session_state.gb_c_n_estimators_1)
            st.session_state.gb_c_learning_rate_0 = min(st.session_state.gb_c_learning_rate_0, st.session_state.gb_c_learning_rate_1)
            st.session_state.gb_c_learning_rate_1 = max(st.session_state.gb_c_learning_rate_0, st.session_state.gb_c_learning_rate_1)
            st.session_state.gb_c_subsample_0 = min(st.session_state.gb_c_subsample_0, st.session_state.gb_c_subsample_1)
            st.session_state.gb_c_subsample_1 = max(st.session_state.gb_c_subsample_0, st.session_state.gb_c_subsample_1)
            st.session_state.gb_c_max_depth_0 = min(st.session_state.gb_c_max_depth_0, st.session_state.gb_c_max_depth_1)
            st.session_state.gb_c_max_depth_1 = max(st.session_state.gb_c_max_depth_0, st.session_state.gb_c_max_depth_1)
            st.session_state.gb_c_min_samples_split_0 = min(st.session_state.gb_c_min_samples_split_0, st.session_state.gb_c_min_samples_split_1)
            st.session_state.gb_c_min_samples_split_1 = max(st.session_state.gb_c_min_samples_split_0, st.session_state.gb_c_min_samples_split_1)
            st.session_state.gb_c_min_samples_leaf_0 = min(st.session_state.gb_c_min_samples_leaf_0, st.session_state.gb_c_min_samples_leaf_1)
            st.session_state.gb_c_min_samples_leaf_1 = max(st.session_state.gb_c_min_samples_leaf_0, st.session_state.gb_c_min_samples_leaf_1)
        
        # Input Parameters
        criterion_range = st.multiselect('**Criterion** -> The function to measure the quality of a split', ['friedman_mse', 'squared_error'], default=['friedman_mse'])
        loss_range = st.multiselect('**Loss** -> Loss function to be optimized', ['log_loss', 'exponential'], default=['log_loss'])
        max_features_range = st.multiselect('**Max Features** -> The number of features to consider when looking for the best split', [None, 'sqrt', 'log2'], default=[None, 'sqrt', 'log2'])
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                n_estimators_0 = st.number_input('**N Estimators (10 - 1000)**', min_value=10, max_value=1000, value=10, key='gb_c_n_estimators_0', on_change=sync_values)
                learning_rate_0 = st.number_input('**Learning Rate (0.001 - 1.0)**', min_value=0.001, max_value=1.0, value=0.01, step=0.001, format="%.3f", key='gb_c_learning_rate_0', on_change=sync_values)
                subsample_0 = st.number_input('**Subsample (0.1 - 1.0)**', min_value=0.1, max_value=1.0, value=0.1, step=0.1, format="%.1f", key='gb_c_subsample_0', on_change=sync_values)
                max_depth_0 = st.number_input('**Max Depth (1 - 100)**', min_value=1, max_value=100, value=3, key='gb_c_max_depth_0', on_change=sync_values)
                min_samples_split_0 = st.number_input('**Min Samples Split (2 - 50)**', min_value=2, max_value=50, value=2, key='gb_c_min_samples_split_0', on_change=sync_values)
                min_samples_leaf_0 = st.number_input('**Min Samples Leaf (1 - 50)**', min_value=1, max_value=50, value=1, key='gb_c_min_samples_leaf_0', on_change=sync_values)
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                n_estimators_1 = st.number_input('**N Estimators (10 - 1000)**', min_value=10, max_value=1000, value=100, key='gb_c_n_estimators_1', on_change=sync_values)
                learning_rate_1 = st.number_input('**Learning Rate (0.001 - 1.0)**', min_value=0.001, max_value=1.0, value=0.5, step=0.001, format="%.3f", key='gb_c_learning_rate_1', on_change=sync_values)
                subsample_1 = st.number_input('**Subsample (0.1 - 1.0)**', min_value=0.1, max_value=1.0, value=1.0, step=0.1, format="%.1f", key='gb_c_subsample_1', on_change=sync_values)
                max_depth_1 = st.number_input('**Max Depth (1 - 100)**', min_value=1, max_value=100, value=10, key='gb_c_max_depth_1', on_change=sync_values)
                min_samples_split_1 = st.number_input('**Min Samples Split (2 - 50)**', min_value=2, max_value=50, value=10, key='gb_c_min_samples_split_1', on_change=sync_values)
                min_samples_leaf_1 = st.number_input('**Min Samples Leaf (1 - 50)**', min_value=1, max_value=50, value=10, key='gb_c_min_samples_leaf_1', on_change=sync_values)
        random_state = st.number_input('**Random State (0 - 100)** -> Controls the randomness of the estimator', min_value=0, max_value=100, value=42)
        
        # Parameters Configuration Range
        params_range = {
            'criterion': criterion_range,
            'loss': loss_range,
            'max_features': max_features_range,
            'n_estimators': [n_estimators_0, n_estimators_1],
            'learning_rate': [learning_rate_0, learning_rate_1],
            'subsample': [subsample_0, subsample_1],
            'max_depth': [max_depth_0, max_depth_1],
            'min_samples_split': [min_samples_split_0, min_samples_split_1],
            'min_samples_leaf': [min_samples_leaf_0, min_samples_leaf_1],
            'random_state': random_state
        }
        del criterion_range, loss_range, max_features_range, n_estimators_0, n_estimators_1, learning_rate_0, learning_rate_1, subsample_0, subsample_1, max_depth_0, max_depth_1, min_samples_split_0, min_samples_split_1, min_samples_leaf_0, min_samples_leaf_1, random_state
    
    elif model_name == 'Multi-Layer Perceptron (Neural Network)':
        st.info('''
            - **Multi-Layer Perceptron** is a feedforward artificial neural network that can be used for both classification and regression tasks.
            - It can handle both numerical and categorical data.
            - It is effective in learning complex relationships in the data and can be used for deep learning tasks.\n
            View more about **Multi-Layer Perceptron** [Hyperparameter Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
            ''')
        def sync_values():
            st.session_state.mlp_c_N_layers_0 = min(st.session_state.mlp_c_N_layers_0, st.session_state.mlp_c_N_layers_1)
            st.session_state.mlp_c_N_layers_1 = max(st.session_state.mlp_c_N_layers_0, st.session_state.mlp_c_N_layers_1)
            st.session_state.mlp_c_neurons_0 = min(st.session_state.mlp_c_neurons_0, st.session_state.mlp_c_neurons_1)
            st.session_state.mlp_c_neurons_1 = max(st.session_state.mlp_c_neurons_0, st.session_state.mlp_c_neurons_1)
            st.session_state.mlp_c_learning_rate_init_0 = min(st.session_state.mlp_c_learning_rate_init_0, st.session_state.mlp_c_learning_rate_init_1)
            st.session_state.mlp_c_learning_rate_init_1 = max(st.session_state.mlp_c_learning_rate_init_0, st.session_state.mlp_c_learning_rate_init_1)
            st.session_state.mlp_c_alpha_0 = min(st.session_state.mlp_c_alpha_0, st.session_state.mlp_c_alpha_1)
            st.session_state.mlp_c_alpha_1 = max(st.session_state.mlp_c_alpha_0, st.session_state.mlp_c_alpha_1)
            st.session_state.mlp_c_max_iter_0 = min(st.session_state.mlp_c_max_iter_0, st.session_state.mlp_c_max_iter_1)
            st.session_state.mlp_c_max_iter_1 = max(st.session_state.mlp_c_max_iter_0, st.session_state.mlp_c_max_iter_1)
        
        # Input Parameters
        activation_range = st.multiselect('**Activation** -> Activation function for the hidden layer', ['identity', 'logistic', 'tanh', 'relu'], default=['relu'])
        solver_range = st.multiselect('**Solver** -> The solver for weight optimization', ['lbfgs', 'sgd', 'adam'], default=['adam'])
        learning_rate = st.multiselect('**Learning Rate** -> The learning rate schedule for weight updates', ['constant', 'invscaling', 'adaptive'], default=['constant', 'adaptive'])
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([1, 15, 1, 15, 1])
            with col2:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Minimum Value</div>', unsafe_allow_html=True)
                N_layers_0 = st.number_input('**N Layers (1 - 10)**', min_value=1, max_value=10, value=1, key='mlp_c_N_layers_0', on_change=sync_values)
                neurons_0 = st.number_input('**Neurons for each layer (1 - 1000)**', min_value=1, max_value=1000, value=10, key='mlp_c_neurons_0', on_change=sync_values)
                learning_rate_init_0 = st.number_input('**Learning Rate Init (0.0001 - 1.0)**', min_value=0.0001, max_value=1.0, value=0.001, step=0.0001, format="%.4f", key='mlp_c_learning_rate_init_0', on_change=sync_values)
                alpha_0 = st.number_input('**Alpha (0.0001 - 0.1)**', min_value=0.0001, max_value=0.1, value=0.0001, step=0.0001, format="%.4f", key='mlp_c_alpha_0', on_change=sync_values)
                max_iter_0 = st.number_input('**Max Iter (100 - 1000)**', min_value=100, max_value=1000, value=200, key='mlp_c_max_iter_0', on_change=sync_values)
            with col4:
                st.markdown('<div style="text-align: center; font-weight: bold;"> Maximum Value</div>', unsafe_allow_html=True)
                N_layers_1 = st.number_input('**N Layers (1 - 10)**', min_value=1, max_value=10, value=3, key='mlp_c_N_layers_1', on_change=sync_values)
                neurons_1 = st.number_input('**Neurons for each layer (1 - 1000)**', min_value=1, max_value=1000, value=100, key='mlp_c_neurons_1', on_change=sync_values)
                learning_rate_init_1 = st.number_input('**Learning Rate Init (0.0001 - 1.0)**', min_value=0.0001, max_value=1.0, value=0.1, step=0.0001, format="%.4f", key='mlp_c_learning_rate_init_1', on_change=sync_values)
                alpha_1 = st.number_input('**Alpha (0.0001 - 0.1)**', min_value=0.0001, max_value=0.1, value=0.01, step=0.0001, format="%.4f", key='mlp_c_alpha_1', on_change=sync_values)
                max_iter_1 = st.number_input('**Max Iter (100 - 1000)**', min_value=100, max_value=1000, value=200, key='mlp_c_max_iter_1', on_change=sync_values)
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
def get_best_c_model(model_choice, params_range, x_train, y_train):
    if model_choice == 'Logistic Regression':
        def objective(trial):
            solver = trial.suggest_categorical('solver', params_range['solver'])
            penalty = trial.suggest_categorical('penalty', params_range['penalty'])
            C = trial.suggest_float('C', params_range['C'][0], params_range['C'][1], log=True)
            max_iter = trial.suggest_int('max_iter', params_range['max_iter'][0], params_range['max_iter'][1])
            model = LogisticRegression(solver=solver, penalty=penalty, C=C, max_iter=max_iter, random_state=params_range['random_state'])
            
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        model = LogisticRegression(**best_params, random_state=params_range['random_state'])
    
    elif model_choice == 'Decision Tree':
        def objective(trial):
            criterion = trial.suggest_categorical('criterion', params_range['criterion'])
            max_features = trial.suggest_categorical('max_features', params_range['max_features'])
            max_depth = trial.suggest_int('max_depth', params_range['max_depth'][0], params_range['max_depth'][1])
            min_samples_split = trial.suggest_int('min_samples_split', params_range['min_samples_split'][0], params_range['min_samples_split'][1])
            min_samples_leaf = trial.suggest_int('min_samples_leaf', params_range['min_samples_leaf'][0], params_range['min_samples_leaf'][1])
            model = DecisionTreeClassifier(criterion=criterion, max_features=max_features, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=params_range['random_state'])
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()

        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        model = DecisionTreeClassifier(**best_params, random_state=params_range['random_state'])
    
    elif model_choice == 'Random Forest':
        def objective(trial):
            criterion = trial.suggest_categorical('criterion', params_range['criterion'])
            max_features = trial.suggest_categorical('max_features', params_range['max_features'])
            n_estimators = trial.suggest_int('n_estimators', params_range['n_estimators'][0], params_range['n_estimators'][1])
            max_depth = trial.suggest_int('max_depth', params_range['max_depth'][0], params_range['max_depth'][1])
            min_samples_split = trial.suggest_int('min_samples_split', params_range['min_samples_split'][0], params_range['min_samples_split'][1])
            min_samples_leaf = trial.suggest_int('min_samples_leaf', params_range['min_samples_leaf'][0], params_range['min_samples_leaf'][1])
            model = RandomForestClassifier(criterion=criterion, max_features=max_features, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=params_range['random_state'], n_jobs=-1)
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        model = RandomForestClassifier(**best_params, random_state=params_range['random_state'], n_jobs=-1)
        
    elif model_choice == 'K-Nearest Neighbors':
        def objective(trial):
            weights = trial.suggest_categorical('weights', params_range['weights'])
            algorithm = trial.suggest_categorical('algorithm', params_range['algorithm'])
            n_neighbors = trial.suggest_int('n_neighbors', params_range['n_neighbors'][0], params_range['n_neighbors'][1])
            p = trial.suggest_int('p', params_range['p'][0], params_range['p'][1])
            model = KNeighborsClassifier(weights=weights, algorithm=algorithm, n_neighbors=n_neighbors, p=p, n_jobs=-1)
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        model = KNeighborsClassifier(**best_params, n_jobs=-1)
    
    elif model_choice == 'Support Vector Machine':
        def objective(trial):
            kernel = trial.suggest_categorical('kernel', params_range['kernel'])
            gamma = trial.suggest_categorical('gamma', params_range['gamma'])
            C = trial.suggest_float('C', params_range['C'][0], params_range['C'][1], log=True)
            degree = trial.suggest_int('degree', params_range['degree'][0], params_range['degree'][1])
            model = SVC(kernel=kernel, gamma=gamma, C=C, degree=degree, random_state=params_range['random_state'])
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        model = SVC(**best_params)
    
    elif model_choice == 'Naive Bayes':
        def objective(trial):
            var_smoothing = trial.suggest_float('var_smoothing', params_range['var_smoothing'][0], params_range['var_smoothing'][1], log=True)
            model = GaussianNB(var_smoothing=var_smoothing)
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        model = GaussianNB(**best_params)
    
    elif model_choice == 'AdaBoost':
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', params_range['n_estimators'][0], params_range['n_estimators'][1])
            learning_rate = trial.suggest_float('learning_rate', params_range['learning_rate'][0], params_range['learning_rate'][1], log=True)
            model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=params_range['random_state'])
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        model = AdaBoostClassifier(**best_params, random_state=params_range['random_state'])
        
    elif model_choice == 'Gradient Boosting':
        def objective(trial):
            criterion = trial.suggest_categorical('criterion', params_range['criterion'])
            loss = trial.suggest_categorical('loss', params_range['loss'])
            max_features = trial.suggest_categorical('max_features', params_range['max_features'])
            n_estimators = trial.suggest_int('n_estimators', params_range['n_estimators'][0], params_range['n_estimators'][1])
            learning_rate = trial.suggest_float('learning_rate', params_range['learning_rate'][0], params_range['learning_rate'][1], log=True)
            subsample = trial.suggest_float('subsample', params_range['subsample'][0], params_range['subsample'][1])
            max_depth = trial.suggest_int('max_depth', params_range['max_depth'][0], params_range['max_depth'][1])
            min_samples_split = trial.suggest_int('min_samples_split', params_range['min_samples_split'][0], params_range['min_samples_split'][1])
            min_samples_leaf = trial.suggest_int('min_samples_leaf', params_range['min_samples_leaf'][0], params_range['min_samples_leaf'][1])
            model = GradientBoostingClassifier(criterion=criterion, loss=loss, max_features=max_features, n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=params_range['random_state'])
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        model = GradientBoostingClassifier(**best_params, random_state=params_range['random_state'])
        
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
            model = MLPClassifier(activation=activation, solver=solver, learning_rate=learning_rate, hidden_layer_sizes=tuple(neurons), learning_rate_init=learning_rate_init, alpha=alpha, max_iter=max_iter, random_state=params_range['random_state'])
            return cross_val_score(model, x_train, y_train, cv=params_range['cv'], scoring=params_range['scoring'], n_jobs=-1).mean()
        
        study = opt.create_study(direction='maximize', sampler=opt.samplers.TPESampler(seed=params_range['random_state']))
        study.optimize(objective, n_trials=params_range['n_trials'], n_jobs=1)
        best_params = study.best_params
        n_layers = best_params.pop('n_layers')
        hidden_layer_sizes = tuple(best_params.pop(f'n_neurons_{i}') for i in range(n_layers))
        best_params['hidden_layer_sizes'] = hidden_layer_sizes
        model = MLPClassifier(**best_params, random_state=params_range['random_state'])
    
    del study, best_params
    gc.collect()
    return model
