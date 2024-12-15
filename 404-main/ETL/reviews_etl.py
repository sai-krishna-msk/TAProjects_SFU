import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import re
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import udf, regexp_replace
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import col

french_substrings = ["le", "la", "les", "un", "une", "est", "et", "par", "pour", "mais"]
german_substrings = ["der", "die", "das", "ein", "eine", "und", "weil", "aber", "von"]
spanish_substrings = ["el", "la", "los", "las", "un", "muy", "es", "y", "por", "el", "porque", "pero"]

accent_pattern = r'[áéíóúñü¿¡äöüßàâæçèéêëîïôùûüÿœ]'

# Pattern for the French, German, and Spanish substrings
pattern = r'\b(' + '|'.join(spanish_substrings + german_substrings + french_substrings) + r')\b'


# UDF to check if text contains non-ASCII characters (likely non-English)
def is_english_regex(text):
    if text is None:
        return False
    text = re.sub(r'<[^>]*>', '', text)
    if re.search(r'[^\x00-\x7F]', text):
        return False 
    return True

is_english_regex_udf = udf(is_english_regex, BooleanType())


def main(inputs):
    reviews_schema = types.StructType([
    types.StructField('listing_id', types.StringType()),
    types.StructField('id', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('reviewer_id', types.IntegerType()),
    types.StructField('reviewer_name', types.StringType()),
    types.StructField('comments', types.StringType()),
])
    
    
    reviews = spark.read.option("multiLine",True).csv(inputs, header=True, schema=reviews_schema).repartition(120)


    reviews_filtered = reviews.filter((reviews['listing_id'].isNotNull()) &
                                      (reviews['id'].isNotNull()) &
                                      (reviews['comments'].isNotNull()) &
                                      is_english_regex_udf(reviews['comments']) & 
                                      ~(col("comments").rlike(pattern)) &
                                      ~(col("comments").rlike(accent_pattern))).cache()
     
    
    reviews_cleaned = reviews_filtered.dropDuplicates(['id'])


    # Remove string '<br/>' from 'comments'column
    reviews_cleaned = reviews_cleaned.withColumn('comments', regexp_replace('comments', '<br/?\s*>', ''))

    reviews_cleaned.write.parquet('reviews_parquet', compression='lz4', mode='overwrite')


if __name__ == '__main__':
    inputs = sys.argv[1]
    spark = SparkSession.builder.appName('reviews ETL').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs)



    