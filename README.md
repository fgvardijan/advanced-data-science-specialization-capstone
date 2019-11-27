# Auto Insurance Claims Fraud Detection
This is a Capstone project for IBM Advanced Data Science Specialization. Check out [youtube video in which I explain the process.](https://www.youtube.com/watch?v=DBlM_B11KvI).


In this project I've demonstrated usage of important concepts and technologies such as:
* __Structured and repeatable process for Data Science projects__
    * Initial Data Exploration
    * Extract, Transform, Load
    * Feature Creation
    * Model Definition
    * Model Training
    * Model Evaluation
    * Model Deployment

* __Hyperparameter tuning and neural network architecture tuning using `talos`__
* __Creation of atomic assets and documenting the architectural decisions for repeatablity and reuse__

Work is structured in five notebooks, each one focusing on one phase of data science process. The reason of this __reuse__ - every task has a clear purpose and defined work product (e.g. jupyter notebook, a script or a docker container). Naming convention make it easy to collaborate and reuse.
1) __EDA - Exploratory Data Analysis__
    * Identify quality issues (e.g. missing values, wrong measurements, …) 
    * Assess feature quality and relevance (e.g. use correlation matrix)
    * Get an idea on the values distribution using statistical measures and visualizations
    
2) __ETL - Data cleansing__
    * Data Types
    * Ranges
    * Emptiness
    * Uniqueness
    * Regular Expressions

3) __Feature Creation__
    * Imputing
    * Scaling / Normalizing / Centering
    * Filtering
    * Discretizing
    * OHE, ...

4) __Model Definition, Training, Evaluation, Comparison__
    * Choosing model performance indicator
    * Dummy Classifier, Logistic Regression, Random Forest, XGBoost
    * Evaluation using chosen metrics and AUC-ROC curve
    * Feature Importances
    * Baseline Neutal network evaluation
    
5) __Hyperparameter Tuning using `Talos`__ 
     * Creating flexible Keras model
     * Defining the Parameter Space Boundary
     * Run the Hyperparameter Scan
     * Analysing the Scan results
     * Narrow down the Hyperparameter space and repeat

Example of Hyperparameter space and network architecture options for exploration with automated exploration. Total of 540,000 permutations!
``` python
p = {'lr': (0.5, 5, 10),      
     'first_neuron':[4, 8, 16, 32, 64],
     'hidden_layers':[0, 1, 2],
     'batch_size': (2, 30, 10),
     'epochs': [50, 100, 150],
     'dropout': (0, 0.5, 5),
     'kernel_initializer': ['normal'],
     'weight_regulizer':[None],
     'emb_output_dims': [None],
     'shapes': ['brick','funnel'],
     'optimizer': [Adam, Nadam, RMSprop],
     'losses': [logcosh, binary_crossentropy],
     'activation':[relu, elu],
     'last_activation': [sigmoid]}
```

## Problem Description
For this project I’ve chosen a fraud detection problem from the insurance industry. Dataset is from Kaggle which hosts the competition and provides relatively clean and structured dataset. Data is on auto insurance claims and it contains 1000 claims from Q1 2015. Each record is a claim described with 40 features. Fraud claims are labeled with 1s.

Kaggle dataset - https://www.kaggle.com/buntyshah/auto-insurance-claims-data
Talos framework -  https://github.com/autonomio/talos
