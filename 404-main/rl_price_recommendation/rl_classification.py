import os
import sys
from pyspark.sql import SparkSession, functions
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator


def main(inputs, output):
    # train 8 models for 8 states
    lowdemand_weekday_nothotregion = spark.read.parquet(os.path.join(inputs, 'high_demand=0/is_weekend=0/hot_region=0'))
    lowdemand_weekday_hotregion = spark.read.parquet(os.path.join(inputs, 'high_demand=0/is_weekend=0/hot_region=1'))
    lowdemand_weekend_nothotregion = spark.read.parquet(os.path.join(inputs, 'high_demand=0/is_weekend=1/hot_region=0'))
    lowdemand_weekend_hotregion = spark.read.parquet(os.path.join(inputs, 'high_demand=0/is_weekend=1/hot_region=1'))
    highdemand_weekday_nothotregion = spark.read.parquet(os.path.join(inputs, 'high_demand=1/is_weekend=0/hot_region=0'))
    highdemand_weekday_hotregion = spark.read.parquet(os.path.join(inputs, 'high_demand=1/is_weekend=0/hot_region=1'))
    highdemand_weekend_nothotregion = spark.read.parquet(os.path.join(inputs, 'high_demand=1/is_weekend=1/hot_region=0'))
    highdemand_weekend_hotregion = spark.read.parquet(os.path.join(inputs, 'high_demand=1/is_weekend=1/hot_region=1'))

    # define RandomForestClassification pipeline
    neighbour_indexer = StringIndexer(inputCol='neighbourhood_cleansed', outputCol='neighbour_indexed')
    property_indexer = StringIndexer(inputCol='property_type', outputCol='property_indexed')
    roomtype_indexer = StringIndexer(inputCol='room_type', outputCol='roomtype_indexed')
    assembler = VectorAssembler(inputCols=['neighbour_indexed', 'property_indexed', 'roomtype_indexed', 'accommodates', 'minimum_nights', 'maximum_nights', 'price'], outputCol="features")
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", maxBins=60)
    pipeline = Pipeline(stages=[neighbour_indexer, property_indexer, roomtype_indexer, assembler, rf])

    # train 8 models
    train_df1, test_df1 = lowdemand_weekday_nothotregion.randomSplit([0.75, 0.25])
    model_lowdemand_weekday_nothotregion = pipeline.fit(train_df1)
    model_lowdemand_weekday_nothotregion.write().overwrite().save(os.path.join(output, 'model_lowdemand_weekday_nothotregion'))

    train_df2, test_df2 = lowdemand_weekday_hotregion.randomSplit([0.75, 0.25])
    model_lowdemand_weekday_hotregion = pipeline.fit(train_df2)
    model_lowdemand_weekday_hotregion.write().overwrite().save(os.path.join(output, 'model_lowdemand_weekday_hotregion'))

    train_df3, test_df3 = lowdemand_weekend_nothotregion.randomSplit([0.75, 0.25])
    model_lowdemand_weekend_nothotregion = pipeline.fit(train_df3)
    model_lowdemand_weekend_nothotregion.write().overwrite().save(os.path.join(output, 'model_lowdemand_weekend_nothotregion'))

    train_df4, test_df4 = lowdemand_weekend_hotregion.randomSplit([0.75, 0.25])
    model_lowdemand_weekend_hotregion = pipeline.fit(train_df4)
    model_lowdemand_weekend_hotregion.write().overwrite().save(os.path.join(output, 'model_lowdemand_weekend_hotregion'))

    train_df5, test_df5 = highdemand_weekday_nothotregion.randomSplit([0.75, 0.25])
    model_highdemand_weekday_nothotregion = pipeline.fit(train_df5)
    model_highdemand_weekday_nothotregion.write().overwrite().save(os.path.join(output, 'model_highdemand_weekday_nothotregion'))

    train_df6, test_df6 = highdemand_weekday_hotregion.randomSplit([0.75, 0.25])
    model_highdemand_weekday_hotregion = pipeline.fit(train_df6)
    model_highdemand_weekday_hotregion.write().overwrite().save(os.path.join(output, 'model_highdemand_weekday_hotregion'))

    train_df7, test_df7 = highdemand_weekend_nothotregion.randomSplit([0.75, 0.25])
    model_highdemand_weekend_nothotregion = pipeline.fit(train_df7)
    model_highdemand_weekend_nothotregion.write().overwrite().save(os.path.join(output, 'model_highdemand_weekend_nothotregion'))

    train_df8, test_df8 = highdemand_weekend_hotregion.randomSplit([0.75, 0.25])
    model_highdemand_weekend_hotregion = pipeline.fit(train_df8)
    model_highdemand_weekend_hotregion.write().overwrite().save(os.path.join(output, 'model_highdemand_weekend_hotregion'))

    ''' can check accuracy use below:
    predictions8 = model_highdemand_weekend_hotregion.transform(test_df8)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy8 = evaluator.evaluate(predictions8)
    print(f"Accuracy: {accuracy8}")
    '''

    ''' can check models' coefficient use below:
    rf_model = model_lowdemand_weekday_nothotregion.stages[-1]
    importances = rf_model.featureImportances
    print(importances.toArray())
    '''


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName("model_etl").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    main(inputs, output)
