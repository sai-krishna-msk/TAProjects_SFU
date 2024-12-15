from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, mean, variance, first, when, count
from pyspark.sql.types import DoubleType, StructType, StructField, StringType, FloatType
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
import seaborn as sns
import matplotlib.pyplot as plt

conf = SparkConf().setAppName('Listings and Reviews Analysis')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Base HDFS path
base_hdfs_path = "/user/gma89/project"

# Read listings data in Parquet format
listings_hdfs_path = f"{base_hdfs_path}/listings_parquet"
listings_df = spark.read.parquet(listings_hdfs_path)

# Read reviews data in Parquet format
reviews_hdfs_path = f"{base_hdfs_path}/reviews_parquet"
reviews_df = spark.read.parquet(reviews_hdfs_path)

# print("Listings Data Preview:")
# listings_df.show(5)

# print("\nReviews Data Preview:")
# reviews_df.show(5)

review_count = reviews_df.groupby('listing_id').agg(count('listing_id').alias('review_count'))

# Join listings and reviews data
comment_df = listings_df.select(
    "id", "review_scores_rating", "review_scores_communication", "review_scores_accuracy", 
    "review_scores_cleanliness", "review_scores_location", "review_scores_value", 
    "review_scores_checkin", "reviews_per_month", "latitude", "longitude", "room_type"
).join(
    reviews_df.select("listing_id", "comments", "date"),
    listings_df.id == reviews_df.listing_id,
    "inner"
)

comment_df = comment_df.join(
    review_count,
    comment_df.id == review_count.listing_id,
    "left"
).drop("listing_id")


def clean_comments(text):
    if text:
        return text.replace("br", "").strip()
    return text

# Register the UDF with PySpark
clean_comments_udf = udf(clean_comments, StringType())

comment_df = comment_df.withColumn("cleaned_comments", clean_comments_udf("comments"))

# Define a UDF to calculate sentiment score
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

get_sentiment_udf = udf(get_sentiment, DoubleType())

comment_df = comment_df.withColumn(
    "comments",
    when(col("comments").isNotNull(), col("comments")).otherwise("")
    ).withColumn(
        "sentiment_score",
        get_sentiment_udf(col("comments"))
    )

# Aggregate data to calculate average sentiment score and review metrics
score_by_id = comment_df.groupBy("id").agg(
    first("review_scores_rating").alias("review_scores_rating"),
    first("review_scores_communication").alias("review_scores_communication"),
    first("review_scores_accuracy").alias("review_scores_accuracy"),
    first("review_scores_cleanliness").alias("review_scores_cleanliness"),
    first("review_scores_location").alias("review_scores_location"),
    first("review_scores_value").alias("review_scores_value"),
    first("review_scores_checkin").alias("review_scores_checkin"),
    first("reviews_per_month").alias("reviews_per_month"),
    first("latitude").alias("latitude"),
    first("longitude").alias("longitude"),
    first("room_type").alias("room_type"),
    first("review_count").alias("review_count"),
    mean("sentiment_score").alias("sentiment_score_mean"),
    variance("sentiment_score").alias("sentiment_score_variance")
)


# Convert to Pandas DataFrame for visualization
score_by_id_pd = score_by_id.toPandas()

# Calculate correlation matrix and visualize it
correlation_columns = [
    'sentiment_score_mean',
    'review_scores_rating',
    'review_scores_communication',
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_location',
    'review_scores_value',
    'review_scores_checkin',
    'reviews_per_month',
    'review_count'
]

correlation_matrix = score_by_id_pd[correlation_columns].corr()

print("Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Sentiment Score and Review Scores")
plt.show()

# Filter out negative comments
negative_comments_df = comment_df.filter(col("sentiment_score") < -0.6)

# Extract keywords from negative comments
negative_comments_pd = negative_comments_df.select("comments").toPandas()
vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
X = vectorizer.fit_transform(negative_comments_pd['comments'])

keywords = vectorizer.get_feature_names_out()
print("Top keywords in negative comments:", keywords)

# Get the comments with the lowest sentiment scores
lowest_sentiment_comments = comment_df.orderBy(col("sentiment_score").asc()).limit(5)
lowest_sentiment_comments.show()

def categorize_sentiment(score):
    if score > 0.2:
        return "positive"
    elif score < -0.2:
        return "negative"
    else:
        return "neutral"

categorize_sentiment_udf = F.udf(categorize_sentiment, StringType())

comment_df = comment_df.withColumn(
    "sentiment_category", categorize_sentiment_udf(F.col("sentiment_score"))
)

# Group by 'sentiment_category' and compute the mean values of review scores and the comment count for each category
sentiment_analysis_df = comment_df.groupBy("sentiment_category").agg(
    F.mean("review_scores_communication").alias("avg_communication"),
    F.mean("review_scores_location").alias("avg_location"),
    F.mean("review_scores_value").alias("avg_value"),
    F.mean("review_scores_rating").alias("avg_rating"),
    F.mean("review_scores_checkin").alias("avg_checkin"),
    F.mean("review_scores_cleanliness").alias("avg_cleanliness"),
    F.count("sentiment_score").alias("comment_count")
)

sentiment_analysis_pd = sentiment_analysis_df.toPandas()

print(sentiment_analysis_pd)

# Set point size and transparency
comment_df = comment_df.withColumn(
    "size", F.when(F.col("sentiment_category") == "negative", 200).otherwise(50)
)
comment_df = comment_df.withColumn(
    "alpha", F.when(F.col("sentiment_category") == "negative", 0.8).otherwise(0.5)
)

comments_pd = comment_df.select("longitude", "latitude", "sentiment_category", "size", "alpha").toPandas()

# Plot the scatter plot
plt.figure(figsize=(12, 8))

# Use Seaborn to plot the scatter plot by sentiment category
scatter = sns.scatterplot(
    data=comments_pd, 
    x="longitude", 
    y="latitude", 
    hue="sentiment_category", 
    size="size", 
    sizes=(50, 200),  # Range of point sizes
    palette={"positive": (0.2, 0.8, 0.3, 0.2),  # Green, semi-transparent
             "neutral": (0.2, 0.2, 0.8, 0.4),   # Blue, semi-transparent
             "negative": (0.8, 0.2, 0.2, 0.6)},  # Red, clearer
    legend="full",
    edgecolor=None  # Remove edge from points
)

# Customize legend to show only sentiment categories
handles, labels = scatter.get_legend_handles_labels()
plt.legend(
    handles=handles[1:4],  # Ignore unnecessary legend items
    labels=["Positive", "Neutral", "Negative"],
    title="Sentiment Category",
    bbox_to_anchor=(1.05, 1), loc="upper left"
)

# Set the title, axis labels, and grid style
plt.title("Sentiment Distribution by GPS Coordinates", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.show()

# Function: Extract top keywords by sentiment category
def extract_keywords_by_sentiment(df, category_column, text_column, category, max_features=20):
    
    filtered_df = df.filter(F.col(category_column) == category)

    filtered_pd = filtered_df.select(text_column).toPandas()
    
    if filtered_pd.empty:
        print(f"No comments found for category: {category}")
        return

    comments = filtered_pd[text_column].fillna("") 
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X = vectorizer.fit_transform(comments)

    keywords = vectorizer.get_feature_names_out()
    tfidf_scores = X.sum(axis=0).A1 
    top_keywords = sorted(zip(keywords, tfidf_scores), key=lambda x: x[1], reverse=True)

    print(f"Top keywords in {category} comments:")
    for word, score in top_keywords:
        print(f"{word}: {score:.4f}")
    print("\n" + "-"*50 + "\n")

categories = ["positive", "neutral", "negative"]

# Loop through categories and extract keywords for each sentiment
for category in categories:
    extract_keywords_by_sentiment(
        df=comment_df,
        category_column="sentiment_category",
        text_column="cleaned_comments", 
        category=category,
        max_features=20
    )