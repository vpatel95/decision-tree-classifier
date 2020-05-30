<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/6wj0hh6.jpg" alt="Project logo"></a>
</p>

<h3 align="center">Decision Tree Classifier</h3>

---

<p align="center"> C++ implementation of Decision Tree Classifier and Random Boosted Forest Classifier<br> </p>

## Table of Contents
- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Built Using](#built_using)
- [Authors](#authors)

## About <a name = "about"></a>
The implementation of predicting the occupancy status of the room. The accuracy of the prediction of occupancy in an office room using data from light, temperature, humidity and CO2 sensors has been evaluated with different statistical classification models like Decision Tree Classifier, Random Forest and Boosted Ran- dom Forest Classifier.Three data sets from the UCI Machine Learning Repository were used in this work, one for training and two for testing the models. The results from the various experiments show that a proper selection of features together with an appropriate classification model can have a significant impact on the accuracy prediction of the occupancy status of the room. Typically, the best accuracy is obtained from training

## Getting Started <a name = "getting_started"></a>
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
Required tools and packages on a linux system.

```
g++ : 5.5.0 (or) 7.4.0
python : 3.5.2
Dataset : https://archive.ics.uci.edu/ml/machine-learning-databases/00357/
```

Get the repository

```
git clone https://github.com/vpatel95/decision-tree-classifier.git
```

### Installing
Following steps will get a development env running.

```
cd decision-tree-classifier
make
```
OR
```
cd decision-tree-classifier
g++ -std=c++11 -w -O3 app.cpp decision_tree.cpp -o app
```

## Usage <a name="usage"></a>
Configurations of training a model is set by a config file located  in the configs directory. The attributes in the config file are

| Attribute            | Sub values | Value(s)                                                    |
|----------------------|------------|-------------------------------------------------------------|
| classification_model | -          | random_forest (or) decision_tree>                           |
| feature_set          | -          | [ <array_of_attributes> [ array_of_attributes ] ]           |
| preprocessed_data    | test       | relative location of test file                              |
|                      | train      | relative location of train file                             |
|                      | validation | relative location of validation file                        |
| extracted_data       | attributes | relative location of attributes information file            |
|                      | test       | relative location of extracted test file                    |
|                      | train      | relative location of extracted train file                   |
|                      | validation | relative location of extracted validation file              |
| bag_size_percent     | -          | percentage of data to be used from training set for bagging |
| num_trees            | -          | number of trees for random forest                           |
| boosting             | -          | true (or) false                                             |
| verbosity            | -          | 1 (or) 2 (or) 3                                             |
| display_trees        | -          | true (or) false                                             |

## Deployment <a name = "deployment"></a>
Add additional notes about how to deploy this on a live system.

## Built Using <a name = "built_using"></a>
- C++
- Python

## ✍️ Authors <a name = "authors"></a>
- [@vpatel95](https://github.com/vpatel95)
