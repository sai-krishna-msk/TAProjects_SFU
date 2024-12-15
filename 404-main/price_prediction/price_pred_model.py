import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql.functions import col, sum, split, regexp_replace, avg, when, regexp_extract, size, lit, log
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, VectorAssembler, Imputer, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import radians, col, sin, cos, atan2, sqrt


def main(input_1, input_2):

    listing_df = spark.read.parquet(input_1)

    calendar_df = spark.read.parquet(input_2)

    listing_df = listing_df.drop('listing_url', 'name', 'description', 'neighborhood_overview', 'picture_url', 'host_id', 'host_name', 'minimum_minimum_nights', 
                                 'maximum_maximum_nights', 'has_availability', 'availability_365', 'bathrooms', 'review_scores_accuracy', 'review_scores_cleanliness', 
                                 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value')

    calendar_df = calendar_df.drop('maximum_nights')

    listing_df = listing_df.withColumnRenamed('price', 'listing_price') \
                        .withColumnRenamed('neighbourhood_cleansed', 'neighbourhood') \
                        .withColumnRenamed('bedrooms', 'num_bedrooms') \
                        .withColumnRenamed('beds', 'num_beds') \
                        .withColumn("num_bathrooms", regexp_extract(listing_df["bathrooms_text"], r"(\d+\.?\d*)", 1).cast("float")) \
                        .withColumn("is_private_bath", when(listing_df["bathrooms_text"].like('%private%'), 1).otherwise(0)) \
                        .withColumn("is_shared_bath", when(listing_df["bathrooms_text"].like('%shared%'), 1).otherwise(0)) \
                            
    calendar_df = calendar_df.withColumnRenamed('price', 'calendar_price') 

    # Log transformation for target variable "calendar_price"
    calendar_df = calendar_df.withColumn('log_price', log('calendar_price'))

    # Using Haversine Formula to calculate proximity to city center
    # Earth's radius in kilometers
    R = 6371

    # Vancouver city center coordinates 
    vancouver_lat = 49.2827
    vancouver_lon = -123.1207

    # Add "distance_to_center_km" column using Haversine formula
    listing_df = listing_df.withColumn("lat_rad", radians(col("latitude"))) \
            .withColumn("lon_rad", radians(col("longitude"))) \
            .withColumn("vancouver_lat_rad", radians(lit(vancouver_lat))) \
            .withColumn("vancouver_lon_rad", radians(lit(vancouver_lon)))

    listing_df = listing_df.withColumn("dlat", col("vancouver_lat_rad") - col("lat_rad")) \
                        .withColumn("dlon", col("vancouver_lon_rad") - col("lon_rad"))

    listing_df = listing_df.withColumn("a", sin(col("dlat") / 2)**2 + cos(col("lat_rad")) * cos(col("vancouver_lat_rad")) * sin(col("dlon") / 2)**2) \
                    .withColumn("c", 2 * atan2(sqrt(col("a")), sqrt(1 - col("a")))) \
                    .withColumn("distance_to_center_km", R * col("c"))

    listing_df = listing_df.drop("lat_rad", "lon_rad", "vancouver_lat_rad", "vancouver_lon_rad", "dlat", "dlon", "a", "c", "latitude", "longitude")


    # Remove empty amenity columns with NULL
    listing_df = listing_df.withColumn("amenities", when(col("amenities") == "[]", lit(None)).otherwise(col("amenities")))

    # Clean and convert "amenities" string column to array
    # Add "amenities_count" as column
    listing_df = listing_df.withColumn("amenities_list", split(regexp_replace(col("amenities"), r"[\[\]\"]", ""),  ", "))

    listing_df = listing_df.drop("amenities")

    listing_df = listing_df.withColumn("amenities_count", size("amenities_list"))

    listing_df = listing_df.drop("amenities_list")


    # Joining listing_df and calendar_df
    joined_df = listing_df.join(calendar_df, listing_df.id == calendar_df.listing_id, "inner")


    joined_df = joined_df.drop('id', 'amenities', 'available', 'amenities_list', 'listing_price', 'bathrooms_text', 'date') 

    # Get average of "log_price" to be used as target variable
    aggregated_price_df = joined_df.groupBy("listing_id").agg(avg("log_price").alias("average_log_price"))

    joined_df = joined_df.join(aggregated_price_df, on="listing_id", how="inner")

    joined_df = joined_df.dropDuplicates(["listing_id"])
    

    joined_df = joined_df.drop('listing_id', 'calendar_price', 'date', 'id', 'amenity_freq_sum', 'amenity_freq_avg')


    joined_df = joined_df.filter(joined_df["average_log_price"].isNotNull())

    joined_df = joined_df.dropDuplicates()


    joined_df = joined_df.repartition(200)  

    # Split the data
    train_data, test_data = joined_df.randomSplit([0.80, 0.20], seed=42)
    train_data = train_data.cache()
    test_data = test_data.cache()

    #train_data.show()

    numeric_columns = ["accommodates",
            "num_bedrooms",
            "num_beds",
            "num_bathrooms",
            "number_of_reviews",
            "review_scores_rating",
            "reviews_per_month",
            "distance_to_center_km",
            "amenities_count",
            "minimum_nights"]
    
    categorical_columns = ["neighbourhood",
                   "property_type",
                   "is_private_bath",
                   "is_shared_bath",
                   "room_type"]


    # Numerical columns
    imputer = Imputer(
        inputCols=numeric_columns,
        outputCols=[num_col+"_imputed" for num_col in numeric_columns],
        strategy="mean"
    )


    numeric_assembler = VectorAssembler(
        inputCols=[num_col+"_imputed" for num_col in numeric_columns],
        outputCol="numeric_features")

    
    scaler = StandardScaler(
        inputCol="numeric_features",
        outputCol="numeric_features_scaled",
            withStd=True,
            withMean=False)


    # Categorical columns 
    indexer = StringIndexer(
        inputCols=categorical_columns,
        outputCols=[cat_col+"_index" for cat_col in categorical_columns],
        handleInvalid="keep")

    
    encoder = OneHotEncoder(inputCols=[cat_col+"_index" for cat_col in categorical_columns], 
                            outputCols=[cat_col+"_ohe" for cat_col in categorical_columns])


    assembler = VectorAssembler(inputCols=["numeric_features_scaled"] +
                                           [cat_col+"_ohe" for cat_col in categorical_columns],
                                outputCol="features")
    

    rf = RandomForestRegressor(featuresCol="features", numTrees=15, maxDepth=20, labelCol="average_log_price")


    pipeline = Pipeline(stages=[imputer, numeric_assembler, scaler, indexer, encoder, assembler, rf])


    model = pipeline.fit(train_data)

    predictions = model.transform(test_data)

    predictions.select("prediction", "average_log_price", "features").show(10)
    

    # Evaluate the predictions
    r2_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='average_log_price', metricName='r2')
    
    rmse_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='average_log_price', metricName='rmse')

    # Evaluate on Training Data
    train_predictions = model.transform(train_data)
    train_r2 = r2_evaluator.evaluate(train_predictions)
    train_rmse = rmse_evaluator.evaluate(train_predictions)

    # Evaluate on Testing Data
    test_predictions = model.transform(test_data)
    test_r2 = r2_evaluator.evaluate(test_predictions)
    test_rmse = rmse_evaluator.evaluate(test_predictions)

    # Print Results
    print('Training R² =', train_r2)
    print('Training RMSE =', train_rmse)
    print('Testing R² =', test_r2)
    print('Testing RMSE =', test_rmse)


    # Evaluate feature importances
    rf_model = model.stages[-1]  
    importances = rf_model.featureImportances

    importances_dense = importances.toArray()

    feature_names = [num_col+"_imputed" for num_col in numeric_columns] + ["neighbourhood_ohe", 
        "room_type_ohe", "property_type_ohe", "is_private_bath_ohe", "is_shared_bath_ohe"]

    # Create a DataFrame with features and their corresponding importances
    feature_importances = pd.DataFrame(
        sorted(list(zip(feature_names, importances_dense)), key=lambda x: abs(x[1]), reverse=True),
        columns=['feature', 'importance'])

    # Bar graph of feature importances
    feature_importances.plot.barh(x='feature', y='importance', legend=False)
    plt.xlabel('Importance')
    plt.title('Feature Importances (Random Forest)')
    plt.show()

    

    # Visualization of Actual vs Predicted Prices
    predictions_pd = predictions.select("average_log_price", "prediction").toPandas()

    actual = predictions_pd["average_log_price"].values  
    predicted = predictions_pd["prediction"].values

    plot_min = min(actual.min(), predicted.min())  
    plot_max = max(actual.max(), predicted.max())

    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.5, color="skyblue", label="Predicted vs. Actual")

    # Add the reference line y = x
    plt.plot([plot_min, plot_max], [plot_min, plot_max], color="red", linestyle="--", label="Reference Line (y = x)")

    plt.title("Predicted vs. Actual Prices", fontsize=14, fontweight="bold")
    plt.xlabel("Actual Average Log Price", fontsize=12)
    plt.ylabel("Predicted Average Log Price", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # Show the plot
    plt.tight_layout()
    plt.show()

 
if __name__ == '__main__':
    input_1 = sys.argv[1]
    input_2 = sys.argv[2]
    spark = SparkSession.builder.appName('Price Prediction Model').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(input_1, input_2)