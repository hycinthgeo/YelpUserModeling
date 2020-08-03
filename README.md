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


## Installations
The installation instruction below is based on an Ubuntu PC with pre-installed Python 3. We will use `requirements.txt` file to install dependencies.  
```sh
$ git clone https://github.com/hycinthgeo/YelpUserModeling.git
$ cd YelpUserModeling/
~/YelpUserModeling$ pip3 install -r requirements.txt
```

## Run Applications with Two Use Cases
The application supported various modes, and allows developers to either run the full application, or start from a major stage, in order to reduce the turn-around for bulk experiments. 

| Commands | Descriptions |
| ------------- | ------------- |
| python main.py  | Run the full application incl. all the following modes, took approximately 80s for my Linux workstation  |
| python main.py mode="pre-processing" | Pre-process to transform the raw input of Yelp User Table ([json file](https://github.com/hycinthgeo/YelpUserModeling/blob/master/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_user.json)) to [`results/allData.csv`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/results/allData.csv), based on the modeler-defined config file [`configs/transformer-default.json`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/configs/transformer-default.json)|
|python main.py mode="split"|Split the transformed data into [`results/trainData.csv`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/results/trainData_MinMaxScaler.csv) and [`results/testData.csv`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/results/testData_MinMaxScaler.csv), based on the modeler-defined config file [`configs/data-pipeline.json`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/configs/data-pipeline.json)|
|python main.py mode="training"|Use case 1 - training the Logistic Regression model upon the train data, using model parameters defined in the modeler-defined config file [`configs/modelConfig-case1.json`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/configs/modelConfig-case1-default.json); it also supports model tuning of the key hyper-parameter `C` via K-fold cross-validation. The best model is saved in [`artefacts/model-case1-default`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/artefacts/model-case1-default). The figures showing [model tuning comparison](https://github.com/hycinthgeo/YelpUserModeling/blob/master/results/tuning-C-case1-default.png) and [final coefficients](https://github.com/hycinthgeo/YelpUserModeling/blob/master/results/coef-case1-default.png) for interpretation is saves within [`results/`](https://github.com/hycinthgeo/YelpUserModeling/tree/master/results). A simple user interace is available as [`UI-case1.ipynb`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/UI-case1.ipynb) to help further interpretation of the final model. |
|python main.py mode="prediction"| Use case 1 - prediction upon the test data, and output scores both from the console and the [`logs/info.log`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/logs/info.log)|
|python main.py mode="similar-users"| Use case 2 - filter to get the seed users that exactly match the restaurant owner's creteria, as specified in [`config/modelConfig-case2.json`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/configs/modelConfig-case2-default.json), and then expand to get a pool of more audiences using Neareast Neighbor and re-rank. A simple user interface is available as [`UI-case2.ipynb`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/UI-case2.ipynb).

## Example of my successful implementations
* [Link of info.log to show full-mode succesfful execution](https://github.com/hycinthgeo/YelpUserModeling/blob/master/logs/info.log)
* [Link of PDF file to show simple UI for Case 1](https://github.com/hycinthgeo/YelpUserModeling/blob/master/docs/UI-case1.pdf)
* [Link of PDF file to show simple UI for Case 2](https://github.com/hycinthgeo/YelpUserModeling/blob/master/docs/UI-case2.pdf)

## Use Case 1
In this use case, I aimes at facilitate business strategies to understand who and why tends to give high-star reviews (average_stars >=4 for this given user). Motivated by the model performance and the interpretability of the Logistic Regression (LR) model, herein, I choose the hyper-parameter `C`=0.1, as it has both good training and validation [`roc_auc` score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score).

![alt text](https://github.com/hycinthgeo/YelpUserModeling/blob/master/results/tuning-C-case1-default.png?raw=true)

Figure blow shows the coefficients of the LR model.
![alt text](https://github.com/hycinthgeo/YelpUserModeling/blob/master/results/coef-case1-default.png?raw=true)

Here are my interpretations based on the above observations on the model coefficients. 
* "votes" plays the most dramatic role in how likely an user give high/low-star reviews; more votes on "funny" (also more votes on "useful") lead to a tendency to give low stars, that's probably because the other users feel this user's review more useful when it's negative-trending; in comparison, more votes on "cool" lead to increasing tendency to give high-star reviews
* this observations on "votes" aligns well with "compliments"
* In addition, increasing review_counts are negatively correlated with high-star reviewers, and increasing friends (or fans) are positively correlated with high-star reviewers.
* Some other features play less significant role, such as "elite" (elite_year), and "compliments-cute"

## Use Case 2
In this use case, restaurant owner can describe the ideal Yelp users that he/she would like to reach out to and how many users they would like to finally reach in this [`modelConfig-case1.json`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/configs/modelConfig-case2-default.json). In many cases, the `exact-matched` Yelp users are smaller than the desired quantity of Yelp users that a restaurant owner would like their Ads to reach; therefore, I further applied Nearest Neighbors and then re-rank based on the distance to reach the descired Yelp user target size. 

As shown in this example ([page #3 of this UI-case2 PDF](https://github.com/hycinthgeo/YelpUserModeling/blob/master/docs/UI-case2.pdf)), we 

* we disconvered 10s of additional users to reach the desired size of the candidate pool (i.e. 100 in this case for fast compute). More specifically, an user whose raw user attributes is almost the same as the desired exact-matched user, besides the year that he/she joined Yelp. 
* the full [list of recommended users with user sources (i.e. exact match or expanded)](https://github.com/hycinthgeo/YelpUserModeling/blob/master/results/recommended_user_list.csv) can now supplies to the restaurant owner, to let him/her to understand how the system has decided to further targeting for interactive feedback. 

## Engineering Considerations 
After my initial completion of this project via Jupyter Notebook, I utilized the remaining week to improve the engineering side of this project. 
For instance, now we can 
* check important statistics of the model, and runtime in the [`logs/info.log`](https://github.com/hycinthgeo/YelpUserModeling/blob/master/logs/info.log); these information can further help the modeler to decide how to improve their feature transformation or model hyperparameters. 
* leverage the reproducible scheme to convininiently select different features and apply various transformations, following the example of this [configs/tranformer-default.config](https://github.com/hycinthgeo/YelpUserModeling/blob/master/configs/transformer-default.json)
* conveniently manage model versions, by creating new model files ([configs/modelConfig-case1.json](https://github.com/hycinthgeo/YelpUserModeling/blob/master/configs/modelConfig-case1-default.json), [configs/modelConfig-case2.json](https://github.com/hycinthgeo/YelpUserModeling/blob/master/configs/modelConfig-case2-default.json)), and change the model path in this [configs/data-pipeline.json](https://github.com/hycinthgeo/YelpUserModeling/blob/master/configs/data-pipeline.json). 
* for modelers, one is welcome to extend more transformer (now supports bucketization and eval()) and model varieties (now supports Logistic Regresssion, Nearest Neighbor); while even though with minimum efforts, he or she can easily further improve model quality by feature engineering and model parameter changes.  

## Conclusion
In this final project, 
* as planned, I explored user entity to find an insightful use case 
* beyond the mileage, I discovered two insightful use cases, which ideally would benefit business strategies (use case 1) and restaurant owner (use case2) 
* for engineering contributions, the scheme I developed above could benefit modelers as well, as it can be swiftly utilized to conduct bulk experiments to improve model quality, and is extensible to support more models. 

Interesting findings
From case 1, one interesting finding is who tends to give average high-star reviews: it seems that “funny” and “useful” in both “votes” and “reviews” are quite negatively contributing to high-star reviews whereas “cool” is a positive driver. In addition, those who have a larger fans and friends circle also tend to give higher ratings. 
For case 2, it tried to generate broader candidate pool of the Yelp users that a restaurant owner would like their Ads to reach out to. Besides conditional filtering to locate exact-matched seed members, Nearest Neighbors of these seed members and then re-rank based on the distance allows us to find more similar users. By manual check the raw features of the expanded users, I found the expanded Yelp users to be of high quality. This can be further validated if integrated this candidate generation as the upstream of a recommender system. 
