# Zero-Shot Human Mobility Forecasting via Large Language Model with Hierarchical Reflection

This repository contains resources and code for the paper:

**Zero-Shot Human Mobility Forecasting via Large Language Model with Hierarchical Reflection**

The code and related materials are being organized and will be uploaded soon. Please stay tuned for updates.


## Dataset

We conduct experiments on three widely-used public location-based social network datasets: **Foursquare-NYC**, **Foursquare-TKY**, and **Gowalla-CA**.

### Dataset Descriptions

- **Foursquare-NYC** and **Foursquare-TKY**:  
  Collected from Foursquare, covering New York City and Tokyo respectively, across an 11-month period (April 2012 to February 2013). Each record includes User ID, POI ID, Check-in Time, Latitude, Longitude, and Venue Category.

- **Gowalla-CA**:  
  Collected from Gowalla, comprising user check-ins in California and Nevada. It offers broader geographic coverage and includes User ID, POI ID, Check-in Time, Latitude, Longitude, and POI Category.

### Data Preprocessing

Following [Yan et al., 2023](https://dl.acm.org/doi/10.1145/3539618.3591770), we:
1. **Filter POIs** and users with fewer than 10 check-ins.
2. **Construct trajectory sequences** for each user based on 24-hour time intervals, and remove trajectories with only a single check-in.
3. **Partition data** chronologically into 80% training, 10% validation, and 10% testing, ensuring validation and test splits only contain users and POIs from the training set.

### Dataset Statistics

| Dataset           | #Users | #POIs | #Categories | #Check-ins | #Trajectories | Train    | Validation | Test   |
| ----------------- | ------ | ----- | ---------- | ---------- | ------------- | -------- | ---------- | ------ |
| Foursquare-NYC    | 1,048  | 4,981 | 318        | 103,941    | 14,130        | 72,206   | 1,400      | 1,347  |
| Foursquare-TKY    | 2,282  | 7,833 | 290        | 405,000    | 65,499        | 274,597  | 6,868      | 7,038  |
| Gowalla-CA        | 3,957  | 9,690 | 296        | 238,369    | 45,123        | 154,253  | 3,529      | 2,780  |

