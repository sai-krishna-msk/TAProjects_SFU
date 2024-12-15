import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import re

from pyspark.sql import SparkSession, types
from pyspark.sql import functions as F
from pyspark.sql.functions import regexp_replace




def main(inputs):
    
    listing_schema = types.StructType([
    types.StructField('id', types.StringType()),
    types.StructField('listing_url', types.StringType()),
    types.StructField('scrape_id', types.IntegerType()),
    types.StructField('last_scraped', types.DateType()),
    types.StructField('source', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('description', types.StringType()),
    types.StructField('neighborhood_overview', types.StringType()),
    types.StructField('picture_url', types.StringType()),
    types.StructField('host_id', types.StringType()),
    types.StructField('host_url', types.StringType()),
    types.StructField('host_name', types.StringType()),
    types.StructField('host_since', types.DateType()),
    types.StructField('host_location', types.StringType()),
    types.StructField('host_about', types.StringType()),
    types.StructField('host_response_time', types.StringType()),
    types.StructField('host_response_rate', types.StringType()),
    types.StructField('host_acceptance_rate', types.StringType()),
    types.StructField('host_is_superhost', types.StringType()),
    types.StructField('host_thumbnail_url', types.StringType()),
    types.StructField('host_picture_url', types.StringType()),
    types.StructField('host_neighbourhood', types.StringType()),
    types.StructField('host_listings_count', types.IntegerType()),
    types.StructField('host_total_listings_count', types.IntegerType()),
    types.StructField('host_verifications', types.StringType()),
    types.StructField('host_has_profile_pic', types.StringType()),
    types.StructField('host_identity_verified', types.StringType()),
    types.StructField('neighbourhood', types.StringType()),
    types.StructField('neighbourhood_cleansed', types.StringType()),
    types.StructField('neighbourhood_group_cleansed', types.StringType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('property_type', types.StringType()),
    types.StructField('room_type', types.StringType()),
    types.StructField('accommodates', types.IntegerType()),
    types.StructField('bathrooms', types.DoubleType()),
    types.StructField('bathrooms_text', types.StringType()),
    types.StructField('bedrooms', types.IntegerType()),
    types.StructField('beds', types.IntegerType()),
    types.StructField('amenities', types.StringType()),
    types.StructField('price', types.StringType()),
    types.StructField('minimum_nights', types.IntegerType()),
    types.StructField('maximum_nights', types.IntegerType()),
    types.StructField('minimum_minimum_nights', types.IntegerType()),
    types.StructField('maximum_minimum_nights', types.IntegerType()),
    types.StructField('minimum_maximum_nights', types.IntegerType()),
    types.StructField('maximum_maximum_nights', types.IntegerType()),
    types.StructField('minimum_nights_avg_ntm', types.IntegerType()),
    types.StructField('maximum_nights_avg_ntm', types.IntegerType()),
    types.StructField('calendar_updated', types.StringType()),
    types.StructField('has_availability', types.StringType()),
    types.StructField('availability_30', types.IntegerType()),
    types.StructField('availability_60', types.IntegerType()),
    types.StructField('availability_90', types.IntegerType()),
    types.StructField('availability_365', types.IntegerType()),
    types.StructField('calendar_last_scraped', types.DateType()),
    types.StructField('number_of_reviews', types.IntegerType()),
    types.StructField('number_of_reviews_ltm', types.IntegerType()),
    types.StructField('number_of_reviews_l30d', types.IntegerType()),
    types.StructField('first_review', types.DateType()),
    types.StructField('last_review', types.DateType()),
    types.StructField('review_scores_rating', types.DoubleType()),
    types.StructField('review_scores_accuracy', types.DoubleType()),
    types.StructField('review_scores_cleanliness', types.DoubleType()),
    types.StructField('review_scores_checkin', types.DoubleType()),
    types.StructField('review_scores_communication', types.DoubleType()),
    types.StructField('review_scores_location', types.DoubleType()),
    types.StructField('review_scores_value', types.DoubleType()),
    types.StructField('license', types.StringType()),
    types.StructField('instant_bookable', types.StringType()),
    types.StructField('calculated_host_listings_count', types.IntegerType()),
    types.StructField('calculated_host_listings_count_entire_homes', types.IntegerType()),
    types.StructField('calculated_host_listings_count_private_rooms', types.IntegerType()),
    types.StructField('calculated_host_listings_count_shared_rooms', types.IntegerType()),
    types.StructField('reviews_per_month', types.DoubleType()),

])


    listings = spark.read.option("multiLine",True).option("quote", "\"").option("escape", "\"").csv(inputs, header=True, schema=listing_schema).repartition(120)
    

    listings = listings.drop('scrape_id', 'last_scraped', 'source', 'host_url', 'host_since', 'host_location', 'host_about', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'host_thumbnail_url', 'host_picture_url',
                             'host_neighbourhood', 'host_listings_count', 'host_total_listings_count', 'host_verifications', 'host_has_profile_pic', 'host_identity_verified', 'neighbourhood', 'neighbourhood_group_cleansed', 'minimum_nights', 
                             'maximum_nights' , 'maximum_minimum_nights', 'minimum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calendar_updated', 'availability_30', 'availability_60', 
                             'availability_90', 'calendar_last_scraped', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'first_review', 'last_review', 'license', 'instant_bookable', 'calculated_host_listings_count', 
                             'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms'
                             )
    

    listings_filtered = listings.filter((listings['id'].isNotNull()) &
                                        (listings['host_id'].isNotNull())).cache()
                                

    listings_cleaned = listings_filtered.dropDuplicates(['id', 'host_id']) 
     

    # Remove '$' in 'price' column and cast to type float
    listings_cleaned = listings_cleaned.withColumn('price', regexp_replace('price', r'[$]', '').cast('float'))

    # Change column type from String to Boolean
    listings_cleaned = listings_cleaned.withColumn('has_availability', 
                                                    F.when(listings_cleaned['has_availability'] == 't', True)
                                                    .when(listings_cleaned['has_availability'] == 'f', False)
                                                    .otherwise(None))
    
    # Remove string '<br/>' from columns
    listings_cleaned = listings_cleaned.withColumn('name', regexp_replace('name', '<br/?\s*>', '')) \
                                    .withColumn('description', regexp_replace('description', '<br/?\s*>', '')) \
                                    .withColumn('neighborhood_overview', regexp_replace('neighborhood_overview', '<br/?\s*>', '')) \
                                    .withColumn('amenities', regexp_replace('amenities', '<br/?\s*>', ''))

    
    listings_cleaned.write.parquet('listings_parquet', compression='lz4', mode='overwrite')
    
    
if __name__ == '__main__':
    inputs = sys.argv[1]
    spark = SparkSession.builder.appName('listings ETL').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs)



    