# Coupon Accepting Prediction
## Project Description
This project is a MLOps project for MLOps Zoomcamp course by DataTalks.Club. The goal of this project is to enhance my understanding of building an MLOps pipeline. The main model used for predicting whether a person will accept recommended coupons while they are in their vehicles is XGBoost because I want to try a tree-based model in the classification problem. The dataset utilized for this project is the "In-Vehicle Coupon Recommendation" dataset from the UCI Machine Learning Repository. The dataset contains various features related to users, merchants, and the coupons to be recommended.

## Dataset Description
[source](https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation)   
I have selected some features that I am interrested in. The features are as follows:
- destination: The person's destination {No Urgent Place, Home, Work}
- weather: Weather type {Sunny, Rainy, Snowy}
- time: Time of the day {7AM, 10AM, 2PM, 6PM, 10PM}
- coupon: Coupon category {Restaurant(<$20), Restaurant($20-$50), Coffee House, Bar, Carry out & Take away}
- expiration: the time the coupon will expire in 2 hours or 1 day {2h, 1d}
- direction_same: The person's destination and the merchant's location are at the same direction {0(No), 1(Yes)}

## Technologies Used
* Python==3.9
  - pipenv==2023.6.26
  - scikit-learn==1.22
  - xgboost==1.7.5
  - pylint 
  - black 
  - isort 
  - pre-commit
* MLflow==2.4
* Prefect==2.11.2
* Docker
* AWS
  - EC2   
  - S3    
  - RDS   
* Terraform
#### Description
Initially, Terraform is employed to establish AWS resources like EC2 (to function as the MLflow tracking server), S3 (for storing tfstate and MLflow models), and RDS (to serve as the MLflow backend database). Next, MLflow is used to track parameters and metrics of the experiment and find the optimal model to be registered for production. This entire workflow can be orchestrated and scheduled through Prefect, which also supports notifications like emails or other methods. For model deployment, Docker can help to create a server to handle prediction result requests.    
Python modules: 
- environment: pipenv
- format and lint: pylint, black, isort, pre-commit
- model: scikit-learn, xgboost


## AWS Resources used in this project 
![AWS resource](https://github.com/ChungWasawat/dtc_mlops_project/blob/main/img/aws.png)

## Project Deployment Instructions
[create infrastruce with Terraform](https://github.com/ChungWasawat/dtc_mlops_project/blob/main/code/infrastructure/README.md)    
[implement code](https://github.com/ChungWasawat/dtc_mlops_project/blob/main/code/README.md)    

## Special Thanks:
I would like to thank DataTalks.Club for providing this course. Through this learning experience, I have gained a comprehensive understanding of how to create an MLOps pipeline. The tools introduced in the course were very helpful, and some of them were new to me, despite my previous studies in Data Science at university. I am eager to apply this knowledge to my future work. 