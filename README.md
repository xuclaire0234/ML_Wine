### README 

#### Project Overview
This folder contains the Machine Learning project for classifying Italian wines into their respective cultivars based on chemical profiles. It includes implementations of K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Softmax classifiers, with a random prediction model serving as the baseline.

#### Getting Started
For mac user run following command:
1. `python3 -m pip install --user virtualenv`
2. `virtualenv env`
3. `source ./env/bin/activate`  
4. Install required dependencies: `pip install -r requirements.txt`.
5. Run the main script: `python runner.py` or `python3 runner.py`.

#### Replicating Results
To replicate the results, simply follow the "Getting Started" instructions. The script `runner.py` will call each files, which train the models on the Wine Dataset and output the classification report, including accuracy, precision, recall, and F1-score.


This project aims to accurately predict the cultivar of wines based on 13 distinct chemical attributes.