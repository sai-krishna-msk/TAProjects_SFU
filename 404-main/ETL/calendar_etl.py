import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import re

from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import col, udf, regexp_replace
from pyspark.sql import functions as F  



def main(inputs):
    calendar_schema = types.StructType([
    types.StructField('listing_id', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('available', types.StringType()),
    types.StructField('price', types.StringType()),
    types.StructField('adjusted_price', types.BooleanType()),
    types.StructField('minimum_nights', types.IntegerType()),
    types.StructField('maximum_nights', types.IntegerType()),
])
    
    calendar = spark.read.option("multiLine",True).csv(inputs, header=True, schema=calendar_schema).repartition(120)

    calendar = calendar.drop('adjusted_price')

    calendar_filtered = calendar.filter((calendar['listing_id'].isNotNull()) &
                                      (calendar['date'].isNotNull()))

    
    # Change column type from String to Boolean
    calendar_cleaned = calendar_filtered.withColumn('available', 
                                                    F.when(calendar_filtered['available'] == 't', True)
                                                    .when(calendar_filtered['available'] == 'f', False)
                                                    .otherwise(None))


    # Remove '$' in 'price' column and cast to type float
    calendar_cleaned = calendar_cleaned.withColumn('price', regexp_replace('price', r'[$]', '').cast('float'))
                                                

    calendar_cleaned = calendar_cleaned.dropDuplicates(['listing_id', 'date'])

   
    calendar_cleaned.write.parquet('calendar_parquet', compression='lz4', mode='overwrite')


if __name__ == '__main__':
    inputs = sys.argv[1]
    spark = SparkSession.builder.appName('calendar ETL').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs)



    