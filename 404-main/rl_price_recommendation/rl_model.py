from pyspark.sql import SparkSession, functions, types, Row
from pyspark.ml import PipelineModel
import sys
import os
import numpy as np
import pandas as pd
import random


''' Reinforcement Learning of price recommendation for Airbnb hosts'''

# define states, actions, Q-table
demand = ['lowdemand', 'highdemand']
day_type = ['weekday', 'weekend']
location = ['hotregion', 'nothotregion']
states = [f"{d}_{day}_{loc}" for d in demand for day in day_type for loc in location]
actions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
Q = np.zeros((len(states), len(actions)))

# initialize Q-learning hyperparameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount rate
epochs = 400 # training times
epsilon = 0.7  # explore rate


# use pre-trained model to predict rent out or not after action(price change)
def simulate_occupancy_rate(new_data_df, model):
    predictions = model.transform(new_data_df)
    return predictions

# define reward function
def get_reward(df_sample, state, action, model):
    demand, day_type, location = state.split('_')
  
    # calculate average of dataframe samples' original_revenue 
    df_sample = df_sample.withColumn('original_revenue', df_sample['price'] * df_sample['label'])
    original_revenue = df_sample.agg(functions.avg('original_revenue')).collect()[0][0]
    
    # calculate average of dataframe samples' new_revenue after action(price change)
    df_sample = df_sample.withColumn('new_price', df_sample['price'] * action)
    df_sample = df_sample.drop('price')
    df_sample = df_sample.withColumnRenamed('new_price', 'price')

    # call pre-trained model to predict whether rent out or not after action(price change)
    new_rentout = simulate_occupancy_rate(df_sample, model)
    new_rentout = new_rentout.withColumn('new_revenue', new_rentout['price'] * new_rentout['prediction'])

    # give different weights to different states
    if demand == 'lowdemand':
        if day_type == 'weekday':
            if location == 'nothotregion':
                new_rentout = new_rentout.withColumn('reward', functions.when((new_rentout['label']==0) & (new_rentout['prediction']==1), (new_rentout['new_revenue'] - new_rentout['original_revenue']) * 4).otherwise(-new_rentout['price'] * 4))
            else:
                new_rentout = new_rentout.withColumn('reward', functions.when((new_rentout['label']==0) & (new_rentout['prediction']==1), (new_rentout['new_revenue'] - new_rentout['original_revenue']) * 3).otherwise(-new_rentout['price'] * 3))
        else:
            if location == 'nothotregion':
                new_rentout = new_rentout.withColumn('reward', functions.when((new_rentout['label']==0) & (new_rentout['prediction']==1), (new_rentout['new_revenue'] - new_rentout['original_revenue']) * 3).otherwise(-new_rentout['price'] * 3))
            else:
                new_rentout = new_rentout.withColumn('reward', functions.when((new_rentout['label']==0) & (new_rentout['prediction']==1), (new_rentout['new_revenue'] - new_rentout['original_revenue']) * 2).otherwise(-new_rentout['price'] * 2))
    else:
        new_rentout = new_rentout.withColumn('reward', functions.when(new_rentout['prediction']==1, (new_rentout['new_revenue'] - new_rentout['original_revenue'])).otherwise(-new_rentout['price']))

    # calculate average reward
    new_revenue = new_rentout.agg(functions.avg('reward')).collect()[0][0]
    
    return new_revenue

# define get_next_state function to logically choose next state
def get_next_state(state, action):
    demand, day_type, location = state.split('_')
  
    # higher price means highdemand, lower price means lowdemand
    if action > 1.0:
        next_demand = 'highdemand'
    else:
        next_demand = 'lowdemand'
  
    # after weekday is weekend, after weekend is weekday
    if day_type == 'weekday':
        next_day_type = 'weekend'
    else:
        next_day_type = 'weekday'
  
    # hot region and not hot region may stay the same
    if location == 'hotregion':
        next_location = 'hotregion'
    else:
        next_location = 'nothotregion'
  
    return f"{next_demand}_{next_day_type}_{next_location}"

# define training Q-table function
def state_train(df_sample, state, model):
    demand, day_type, location = state.split('_')

    # explore new action or choose action from q-table
    if random.uniform(0, 1) < epsilon:
        action = random.choice(actions)  # explore
    else:
        action_index = np.argmax(Q[states.index(state)])  # use q-table
        action = actions[action_index]

    # get reward
    reward = get_reward(df_sample, state, action, model)

    # get next_state
    next_state = get_next_state(state, action)
    next_state_index = states.index(next_state)
  
    # get state and action index
    state_index = states.index(state)
    action_index = actions.index(action)

    # update Q-table
    Q[state_index, action_index] = Q[state_index, action_index] + alpha * (reward + gamma * np.max(Q[next_state_index]) - Q[state_index, action_index])


def main(etls, models, output):
    # read 8 etl files
    lowdemand_weekday_nothotregion = spark.read.parquet(os.path.join(etls, 'high_demand=0/is_weekend=0/hot_region=0'))
    lowdemand_weekday_hotregion = spark.read.parquet(os.path.join(etls, 'high_demand=0/is_weekend=0/hot_region=1'))
    lowdemand_weekend_nothotregion = spark.read.parquet(os.path.join(etls, 'high_demand=0/is_weekend=1/hot_region=0'))
    lowdemand_weekend_hotregion = spark.read.parquet(os.path.join(etls, 'high_demand=0/is_weekend=1/hot_region=1'))
    highdemand_weekday_nothotregion = spark.read.parquet(os.path.join(etls, 'high_demand=1/is_weekend=0/hot_region=0'))
    high_demand_weekday_hotregion = spark.read.parquet(os.path.join(etls, 'high_demand=1/is_weekend=0/hot_region=1'))
    high_demand_weekend_nothotregion = spark.read.parquet(os.path.join(etls, 'high_demand=1/is_weekend=1/hot_region=0'))
    high_demand_weekend_hotregion = spark.read.parquet(os.path.join(etls, 'high_demand=1/is_weekend=1/hot_region=1'))

    # load 8 models
    model_lowdemand_weekday_nothotregion = PipelineModel.load(os.path.join(models, 'model_lowdemand_weekday_nothotregion'))
    model_lowdemand_weekday_hotregion = PipelineModel.load(os.path.join(models, 'model_lowdemand_weekday_hotregion'))
    model_lowdemand_weekend_nothotregion = PipelineModel.load(os.path.join(models, 'model_lowdemand_weekend_nothotregion'))
    model_lowdemand_weekend_hotregion = PipelineModel.load(os.path.join(models, 'model_lowdemand_weekend_hotregion'))
    model_highdemand_weekday_nothotregion = PipelineModel.load(os.path.join(models, 'model_highdemand_weekday_nothotregion'))
    model_highdemand_weekday_hotregion = PipelineModel.load(os.path.join(models, 'model_highdemand_weekday_hotregion'))
    model_highdemand_weekend_nothotregion = PipelineModel.load(os.path.join(models, 'model_highdemand_weekend_nothotregion'))
    model_highdemand_weekend_hotregion = PipelineModel.load(os.path.join(models, 'model_highdemand_weekend_hotregion'))

    
    for i in range(epochs):
        state = random.choice(states)

        if state == 'lowdemand_weekday_nothotregion':
            df_sample = lowdemand_weekday_nothotregion.sample(fraction=0.01)
            state_train(df_sample, 'lowdemand_weekday_nothotregion', model_lowdemand_weekday_nothotregion)

        elif state == 'lowdemand_weekday_hotregion':
            df_sample = lowdemand_weekday_hotregion.sample(fraction=0.01)
            state_train(df_sample, 'lowdemand_weekday_hotregion', model_lowdemand_weekday_hotregion)

        elif state == 'lowdemand_weekday_nothotregion':
            df_sample = lowdemand_weekend_nothotregion.sample(fraction=0.01)
            state_train(df_sample, 'lowdemand_weekday_nothotregion', model_lowdemand_weekend_nothotregion)

        elif state == 'lowdemand_weekend_hotregion':
            df_sample = lowdemand_weekend_hotregion.sample(fraction=0.01)
            state_train(df_sample, 'lowdemand_weekend_hotregion', model_lowdemand_weekend_hotregion)

        elif state == 'highdemand_weekday_nothotregion':
            df_sample = highdemand_weekday_nothotregion.sample(fraction=0.01)
            state_train(df_sample, 'highdemand_weekday_nothotregion', model_highdemand_weekday_nothotregion)

        elif state == 'highdemand_weekday_hotregion':
            df_sample = high_demand_weekday_hotregion.sample(fraction=0.01)
            state_train(df_sample, 'highdemand_weekday_hotregion', model_highdemand_weekday_hotregion)

        elif state == 'highdemand_weekend_nothotregion':
            df_sample = high_demand_weekend_nothotregion.sample(fraction=0.01)
            state_train(df_sample, 'highdemand_weekend_nothotregion', model_highdemand_weekend_nothotregion)

        else:
            df_sample = high_demand_weekend_hotregion.sample(fraction=0.01)
            state_train(df_sample, 'highdemand_weekend_hotregion', model_highdemand_weekend_hotregion)

    best_action_for_state = {state: actions[np.argmax(Q[states.index(state)])] for state in states}
    print("best action for state：")
    print(best_action_for_state)

    result = pd.DataFrame(list(best_action_for_state.items()), columns=['state', 'best_action'])
    # save as csv file
    result.to_csv(output, index=False)


if __name__ == '__main__':
    etls = sys.argv[1]
    models = sys.argv[2]
    output = sys.argv[3]
    spark = SparkSession.builder.appName("model_rl").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    main(etls, models, output)
