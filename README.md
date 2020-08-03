# Yelp User Modeling 

**Movitations**: In this Capstone Project of Summer 2020, we have gained a variety of hands-on experience in well-designed projects on modeling restaurant entities through multiple MPs. With Assma’s prompt proposal review and insightful suggestions, below are my goals and scope for this open-exploration MP7. 
* **Topic**: exploring user entities for user modeling, that provides insightful use cases leveraging on Machine Learning algorithms
* **Scope**: since the above idea is close to research ideas, I would not need to develop an UI; instead, I shall provide a GitHub link with an ReadMe on how to run the codes. 

**Who will benefit?**: After initial explorations of a week, I found two insightful use cases, and then build a reproducible workflow to efficiently bulk conduct modeling experiments. For my final product, it shall benefit 
* **Modelers**: (1) Light-weighted, only need to change config files to swift conduct experiments; (2) efficient for stepwise reproduce and model management; (3) can easily support multiple use cases by changing modelConfig
* **Business strategist**: to understand who and why tends to give high-star reviews (use case 1)
* **Restaurant owner**: can help them generate candidate pool that they would like to reach out for advertisement (use case 2)
Even though UI design is not within scope of this project, as my end product is intended to also benefit broader audiences with different levels of programming efficiency, I prepared two Jupyter Notebooks as a simple UI for each of the use cases. 

![alt text](https://github.com/hycinthgeo/YelpUserModeling/blob/master/docs/overview.png?raw=true)

##### Table of Contents  
##### Table of Contents  
[Headers](#headers)  
[Emphasis](#emphasis)  
...snip...    
<a name="headers"/>
## Headers

## Installations
The installation instruction below is based on an Ubuntu PC with pre-installed Python 3. We will use `requirements.txt` file to install dependencies.  
```sh
$ git clone https://github.com/hycinthgeo/YelpUserModeling.git
$ cd YelpUserModeling/
~/YelpUserModeling$ pip3 install -r requirements.txt
```

## Run Applications with Two Use Cases
The application supported various modes, and allows developers to either run the full application, or start from a major stage, in order to reduce the turn-around for bulk experiments. 

### Commands and descriptions 
| Commands | Descriptions |
| ------------- | ------------- |
| python main.py  | Run the full application incl. all the following modes, took approximately 80s for my Linux workstation  |
| python main.py mode="pre-processing" | Pre-process to transform the raw input of Yelp User Table ([json file](https://github.com/hycinthgeo/YelpUserModeling/blob/master/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_user.json)) to [`results/allData.csv`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/results/allData.csv), based on the modeler-defined config file [`configs/transformer-default.json`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/configs/transformer-default.json)|
|python main.py mode="split"|Split the transformed data into [`results/trainData.csv`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/results/trainData_MinMaxScaler.csv) and [`results/testData.csv`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/results/testData_MinMaxScaler.csv), based on the modeler-defined config file [`configs/data-pipeline.json`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/configs/data-pipeline.json)|
|python main.py mode="training"|Use case 1 - training the Logistic Regression model upon the train data, using model parameters defined in the modeler-defined config file [`configs/modelConfig-case1.json`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/configs/modelConfig-case1-default.json); it also supports model tuning of the key hyper-parameter `C` via K-fold cross-validation. The best model is saved in [`artefacts/model-case1-default`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/artefacts/model-case1-default). The figures showing [model tuning comparison](https://github.com/hycinthgeo/YelpUserModeling/blob/master/results/tuning-C-case1-default.png) and [final coefficients](https://github.com/hycinthgeo/YelpUserModeling/blob/master/results/coef-case1-default.png) for interpretation is saves within [`results/`](https://github.com/hycinthgeo/YelpUserModeling/tree/master/results). A simple user interace is available as [`UI-case1.ipynb`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/UI-case1.ipynb) to help further interpretation of the final model. |
|python main.py mode="prediction"| Use case 1 - prediction upon the test data, and output scores both from the console and the [`logs/info.log`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/logs/info.log)|
|python main.py mode="similar-users"| Use case 2 - filter to get the seed users that exactly match the restaurant owner's creteria, as specified in [`config/modelConfig-case2.json`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/configs/modelConfig-case2-default.json), and then expand to get a pool of more audiences using Neareast Neighbor and re-rank. A simple user interface is available as [`UI-case2.ipynb`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/UI-case2.ipynb).

### Example of my successful implementations
* [Link of info.log to show full-mode succesfful execution](https://github.com/hycinthgeo/YelpUserModeling/blob/master/logs/info.log)
* [Link of PDF file to show simple UI for Case 1](https://github.com/hycinthgeo/YelpUserModeling/blob/master/docs/UI-case1.pdf)
* [Link of PDF file to show simple UI for Case 2](https://github.com/hycinthgeo/YelpUserModeling/blob/master/docs/UI-case2.pdf)

## Use Case 1
In this use case 


## Use Case 2

## Conclusions
