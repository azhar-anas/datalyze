# Datalyze

Datalyze is an all-in-one data analytics platform that transforms raw data into meaningful insights. Designed for analysts, data scientists, and business professionals, it simplifies data exploration, cleaning, preparation, and modeling with an intuitive interface, automation, and machine learning.

Datalyze stands out for its efficiency, accuracy, and accessibility. It works seamlessly on laptops and mobile devices, catering to all skill levels without requiring coding. The platform automates analysis, supports various machine learning models from Scikit-Learn, and optimizes models using the TPE algorithm. Reports, models, and visualizations can be downloaded for further analysis.

## 🚀 Features

### 📂 Data Import & Management
- Upload datasets in CSV or Excel format
- Automatic data structure detection
- Data preview with options to delete or restore data

### 🔎 Exploratory Data Analysis (EDA)
- Automated data profiling and summary reports
- Missing value analysis and duplicate detection
- Descriptive statistics and interactive visualizations

### 🛠️ Data Cleaning & Feature Engineering
- Modify feature types and handle missing values
- Remove duplicates and detect outliers
- Feature selection and One-Hot Encoding

### 🤖 Machine Learning & Model Optimization
- Configure dataset, train-test split, and normalization
- Select regression or classification models
- Hyperparameter tuning with the TPE method
- Evaluate models with accuracy, precision, recall, and F1-score
- Export trained models in `.joblib` format

## 🛠️ Tech Stack
Datalyze is built using the following technologies:
- **Python** - Core language for backend processing
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-Learn** - Machine learning models
- **Optuna** - Hyperparameter optimization (TPE algorithm)
- **Joblib** - Model serialization and deployment
- **ydata-profiling** - Automated data profiling
- **streamlit-pandas-profiling** - Integrating pandas-profiling with Streamlit

## 📖 Getting Started

### 1️⃣ Installation
Make sure you have Python 3.11 installed. Then, clone the repository and install dependencies:
```bash
git clone https://github.com/azhar-anas/datalyze.git
cd datalyze
pip install -r requirements.txt
```

### 2️⃣ Running the App
To start the Datalyze application, use the following command:
```bash
streamlit run app.py
```

📜 License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
