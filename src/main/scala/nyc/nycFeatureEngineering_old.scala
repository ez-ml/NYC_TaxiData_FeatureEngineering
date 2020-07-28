package nyc


import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object nycFeatureEngineering_old {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("SparkSessionZipsExample")
      .getOrCreate()

    val news_list_rdd = spark.sparkContext.wholeTextFiles("s3a://ml-workflow-data/mini_newsgroups/*/*")
    val news_df = spark.createDataFrame(news_list_rdd).toDF("File_name", "Text")
      .withColumn("ID",split(col("file_name"),"/").getItem(5))
      .withColumn("Topics",split(col("file_name"),"/").getItem(4))
      .withColumn("Labels", substring_index(col("Topics"), ".", 1))

    //news_df.show()

    val Array(training, test)=news_df.randomSplit(Array(0.8,0.2), seed=45)


    val tokenizer=new Tokenizer().setInputCol("Text").setOutputCol("Words")
    val stopWordRemover=new StopWordsRemover().setInputCol("Words").setOutputCol("Filtered").setCaseSensitive(false)
    val hashingTf=new HashingTF().setNumFeatures(1000).setInputCol("Filtered").setOutputCol("rawFeatures")

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
    val encoder = new StringIndexer().setInputCol("Labels").setOutputCol("LabelsEncoded")

    val lr = new LogisticRegression().setRegParam(0.01).setThreshold(0.5).setLabelCol("LabelsEncoded")

    val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordRemover, hashingTf, idf, encoder,lr))


    val model = pipeline.fit(training)

    val predictions = model.transform(test)

    predictions.show()


  }

}
