# Coupon Accepting Model
## Project Description
This project is a MLOps project for MLOps Zoomcamp course by DataTalks.Club. The goal of this project is to enhance my understanding of building an MLOps pipeline. The main model used for predicting whether a person will accept recommended coupons while they are in their vehicles is XGBoost. The dataset utilized for this project is the "In-Vehicle Coupon Recommendation" dataset from the UCI Machine Learning Repository. The dataset contains various features related to users, merchants, and the coupons to be recommended.

## Dataset Description
[source](https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation)   
I have selected some features that I am interrested in. The features are as follows:
- destination: The person's destination {No Urgent Place, Home, Work}
- weather: Weather type {Sunny, Rainy, Snowy}
- time: Time of the day {7AM, 10AM, 2PM, 6PM, 10PM}
- coupon: Coupon category {Restaurant(<$20), Restaurant($20-$50), Coffee House, Bar, Carry out & Take away}
- expiration: the time the coupon will expire in 2 hours or 1 day {2h, 1d}
- direction_same: The person's destination and the merchant's location are at the same direction {0(No), 1(Yes)}
