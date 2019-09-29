# COMP 551 MiniProject 1

### Prerequisites

1. Set up python and pip
The python and all the libraries version that we used in this project are: 

Python version: 3.7.4 

Package           Version
----------------- -------
astroid           2.2.5  
cycler            0.10.0  
isort             4.3.21  
kiwisolver        1.1.0  
lazy-object-proxy 1.4.2   
matplotlib        3.1.1  
mccabe            0.6.1  
numpy             1.17.2  
pandas            0.25.1  
pip               19.2.3  
pylint            2.3.1  
pyparsing         2.4.2  
python-dateutil   2.8.0  
pytz              2019.2  
scipy             1.3.1   
setuptools        41.2.0  
six               1.12.0  
typed-ast         1.4.0  
wheel             0.33.6  
wrapt             1.11.2  

We also used os library to execute bash commands to install certain libraries, and download the raw datastes. 
The sudo pip install command will prompt to enter password: 

```
os.system('/bin/bash -c "sudo pip install numpy matplotlib scipy pandas"')
os.system('/bin/bash -c "curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"')
os.system('/bin/bash -c "curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"')
```
Note that those commands only works if you already have Python and pip configured, and use a MacOS machine. 


### Project structure

#### LDA files 
The LDA model was being implemented in the file `classification.py`, class LDA. The k-fold cross validation of LDA model for both datasets were being implemented in the file `main.py`. You can execute the main.py file by the follwing command after have all prerequisite libraries set up. 
```
python3 main.py
```

#### Logistic Regression files 
The logistic regression model was being implemented in the file `classification.py`, class LDA. The k-fold cross validation of LDA model for both datasets were being implemented in the file main.py. You can execute the `testLR.py` file by the follwing command after have all prerequisite libraries set up. 
```
python3 testLR.py
```

#### Others
`plotLR.py`: contains script to create the plot for logistic regression in the report 

`classification.py`: contains the logisitic regression and LDA implementation

`toolFunctions.py`: contains helper function including the `evaluate_acc()`, and `normalize()`, and helper methods for kfold validation: `partition()` and `excludeSegment()`
