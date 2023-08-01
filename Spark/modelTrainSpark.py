from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize a Spark session
spark = SparkSession.builder.appName("IrisRandomForest").getOrCreate()

def main():
    # Sample Iris data for demonstration (Replace this with your data)
    iris_data = [
        (5.1, 3.5, 1.4, 0.2, "setosa"),
        (4.9, 3.0, 1.4, 0.2, "setosa"),
        (7.0, 3.2, 4.7, 1.4, "versicolor"),
        (6.4, 3.2, 4.5, 1.5, "versicolor"),
        (6.3, 3.3, 6.0, 2.5, "virginica"),
        (5.8, 2.7, 5.1, 1.9, "virginica")
    ]

    # Create a DataFrame from the sample data
    iris_df = spark.createDataFrame(iris_data, ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

    # Feature engineering: Index the "species" column
    indexer = StringIndexer(inputCol="species", outputCol="indexedLabel").fit(iris_df)
    iris_df = indexer.transform(iris_df)

    # Assemble the features into a single vector column
    feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    iris_df = assembler.transform(iris_df)

    # Create and train a Random Forest classifier on the entire dataset
    rf_classifier = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=5, maxDepth=3)
    model = rf_classifier.fit(iris_df)

    # Make predictions on the entire dataset
    predictions = model.transform(iris_df)

    # Evaluate the model's accuracy
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Model Accuracy = {:.2f}%".format(accuracy * 100))

    # Save the final model to a directory (Replace "final_model" with your desired path)
    model.save("final_model")

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()

