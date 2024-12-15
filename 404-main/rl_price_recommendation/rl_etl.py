from pyspark.sql import SparkSession, functions, types, Row
from pyspark.sql.window import Window
from pyspark.sql.functions import dayofweek
import sys
import os


def main(inputs, output):
    # read files
    listings_path = os.path.join(inputs, "listings_parquet")
    calendar_path = os.path.join(inputs, "calendar_parquet")
    listings_df = spark.read.parquet(listings_path)
    calendar_df = spark.read.parquet(calendar_path)

    ''' Reinforcement Learning ETL Part '''

    # join listings and calendar, only keep cols need
    joined_df = listings_df.join(calendar_df, listings_df.id == calendar_df.listing_id, how='inner')
    df_sub = joined_df.select('id', 'host_id', 'neighbourhood_cleansed', 'property_type', 'room_type', 'accommodates', 'minimum_nights', 'maximum_nights', 'date', 'available', calendar_df['price'])
    
    # filter out date from 2023-10-05 to 2024-09-06 since dataset is collected at 2024-09-06
    df_sub = df_sub.filter((df_sub['date'] >= '2023-10-05') & (df_sub['date'] <= '2024-9-6'))

    # drop zero rows
    df_sub = df_sub.dropna(subset=['id', 'host_id', 'neighbourhood_cleansed', 'property_type', 'room_type', 'accommodates', 'minimum_nights', 'maximum_nights', 'date', 'available', 'price'])

    # if available is true, means did not rent out at this date, given label 0
    df_sub = df_sub.withColumn('label', functions.when(functions.col('available') == True, 0).otherwise(1))
    
    # add 'is_weekend' column to target weekday and weekend
    df_sub = df_sub.withColumn('is_weekend', functions.when((dayofweek(df_sub['date']) == 6) | (dayofweek(df_sub['date']) == 7), 1).otherwise(0))
    
    # add 'occupancy_rate' column which calculate cummulative occupancy rate
    window_spec = Window.partitionBy('id').orderBy('date')
    df_sub = df_sub.withColumn('cumulative_rentout', functions.sum('label').over(window_spec))
    df_sub = df_sub.withColumn('total_days', functions.row_number().over(window_spec))
    df_sub = df_sub.withColumn('occupancy_rate', functions.col('cumulative_rentout') / functions.col('total_days'))
    
    # if occupancy_rate > 0.5, mark 1 as high_demand
    df_sub = df_sub.withColumn('high_demand', functions.when(df_sub['occupancy_rate'] > 0.5, 1).otherwise(0))    

    # avg_occupancy_rate dataframe: neighbourhood_cleansed (distinct)   avg_occupancy_rate(float) 
    avg_occupancy_rate = df_sub.groupBy('neighbourhood_cleansed').agg(functions.avg('occupancy_rate').alias('avg_occupancy_rate')).orderBy('avg_occupancy_rate', ascending=False)

    # join avg_occupancy_rate df with df_sub
    df_final = df_sub.join(avg_occupancy_rate, df_sub['neighbourhood_cleansed'] == avg_occupancy_rate['neighbourhood_cleansed'])
    df_final = df_final.drop(avg_occupancy_rate['neighbourhood_cleansed'])

    # add 'hot_region' column, 0.542 is the median of 'avg_occupancy_rate', larger than 0.542 denote 1, otherwise 0
    df_final = df_final.withColumn('hot_region', functions.when(df_final['avg_occupancy_rate'] > 0.542, 1).otherwise(0))

    df_final.write.partitionBy('high_demand', 'is_weekend', 'hot_region').parquet(output, mode='overwrite')


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName("model_etl").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    main(inputs, output)
