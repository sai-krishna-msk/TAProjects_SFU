# CMPT 732 Final Project: Group 404 Vancouver Airbnb Data Analysis

To run the project locally:

Clone the repository:
git clone github.sfu.ca/sxa34/404

### Install required dependencies
pip install -r requirements.txt

## ETL
ETL is performed on raw data which is stored in raw_data directory. The .lz4 parquet files are stored in raw_data directory: calendar_parquet, listings_parquet, reviews_parquet. The ETL was processed through HDFS file storage and was executed on a cluster using Apache Spark.

## Part I Price Prediction Model using RandomForestRegressor
Price prediction model that forecasts the optimal price of listings for the coming year, using historical data and features that influence pricing, such as amenities, location, seasonality, and listing characteristics.

### Code running
    spark-submit price_pred_model.py listings_parquet calendar_parquet

### Results
Training R² = 0.9158717155922556

Training RMSE = 0.18859058190125075

Testing R² = 0.8591325842272188

Testing RMSE = 0.24354425173837743

## Part II  Review and Score analysis

### Code running

    spark-submit review_score.py

### Visualization
    
    path: 404/review_score_analysis/
    
    Sentiment Analysis: Classifying comments into positive, negative, or neutral based on sentiment scores.
    Correlation Analysis: Analyzing the relationship between review scores and sentiment scores.
    Geographical Mapping: Visualizing sentiment distribution on geographical coordinates.
    High-Frequency Word Count: Identifying common words in comments.


## Part III  Price Recommendations for Airbnb hosts Using Reinforcement Learning (RL) Model

### Code running instructions in Linux (assume files are under same path):

    # etl part
    spark-submit rl_etl.py ETL rl_data  # ETL is the folder name of Part I etl files, rl_data is the output name
    
    # rl_classification part
    spark-submit rl_classification.py rl_data rl_classification_model  # rl_classification_model is the model save folder name
    
    # rl_model part
    spark-submit rl_model.py rl_data rl_classification_model rl_output  # rl_output is the output csv file name, which save two columns: state, best_action

### RL states and Actions Settings

States: low_demand, high_demand, weekday, weekend, hot region, not hot region

Actions (price change): 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5

### Related Files:

rl_etl.py: Joined calendar_parquet and reviews_parquet ETL files in Part I, selected the features required for this part. Performed second ETL process to generate 8 Parquet files partitioned by high/low demand, weekday/weekend, and hot/not hot region.

rl_data: Results of rl_etl.py, contains 8 Parquet files.

rl_classification.py: For 8 rl_data Parquet files, trained 8 RandomForest Classification models.

    features: 'neighbour_indexed', 'property_indexed', 'roomtype_indexed', 'accommodates', 'minimum_nights', 'maximum_nights', 'price'
    
    prediction: 1 indicates under these features, the property can be rented out, 0 otherwise

rl_classification_model: Results of rl_classification.py, contains 8 classification models.

rl_model.py: Used RL model to generate best_action_for_state.

    # Q-learning hyperparameters
    alpha = 0.1  # learning rate
    gamma = 0.9  # discount rate
    epochs = 400
    epsilon = 0.7  # explore rate
    
rl_output.csv: Choosed median best_action for each state under results of nearly 100 different hyperparameter settings.

**Hint**: The method of using the median price change from nearly 100 different hyperparameter settings for the model results is necessary due to the variability in model hyperparameters, random sampling of dataframes in each epoch, and the randomness of states and actions. Here, the median of the best action for each state is used as the optimal result of this part.
