package nyc

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
object nycTripFeature1 {

  def main(args: Array[String]): Unit = {


    val spark = SparkSession
      .builder()
      .appName("NYCTripFeatureEngineering")
      .enableHiveSupport()
      .getOrCreate()

    import spark.implicits._
    import spark.sql

    val df_1=sql("SELECT * from nyc_trips_final_1")
      .drop("store_and_forward","rate_code","mta_tax","tip_amt","tolls_amt","total_amt") //Drop the columns from dataframe
      .withColumn("hour",hour(col("trip_pickup_datetime"))) //Add Hour column
      .withColumn("day_of_week",date_format(col("trip_pickup_datetime"),"u")) //Add day_of_week column
      .withColumn("payment_type", upper(col("payment_type")))
      .filter(($"payment_type" =!= "NO CHARGE" && $"payment_type" =!= "DISPUTE"))
      .filter(($"passenger_count" > 0 && $"passenger_count" < 7))
      .filter(($"Start_Lat" >= 40 && $"Start_Lat" <= 41))
      .filter(($"End_Lat" >= 40 && $"End_Lat" <= 41))
      .filter(($"Start_Lon" >= -75 && $"Start_Lon" <= -72))
      .filter(($"End_Lon" >= -75 && $"End_Lon" <= -72))
      .withColumn("Lat_Diff", $"Start_Lat" - $"End_Lat")
      .withColumn("Lon_Diff", $"Start_Lon" - $"End_Lon")
      .filter($"Lat_Diff" =!= 0).filter($"Lon_Diff" =!= 0)

    val assembler = new VectorAssembler().setInputCols(Array("start_lat", "start_lon")).setOutputCol("start_features")

    val kmeans = new KMeans().setK(10).setFeaturesCol("start_features").setPredictionCol("start_cluster")

    val pipeline = new Pipeline().setStages(Array(assembler,kmeans))
    val model = pipeline.fit(df_1)
    val nyc_trip_df = model.transform(df_1)

    val columns = Seq("year","month")

    nyc_trip_df.select("vendor_name","trip_pickup_datetime","trip_dropoff_datetime","passenger_count","trip_distance","start_lon","start_lat","end_lon","end_lat","payment_type","fare_amt","surcharge","year","month","hour","day_of_week","Lat_Diff","Lon_Diff","start_cluster")
      .write.mode("append").partitionBy(columns:_*).saveAsTable("default.nyc_trips_final_102")

  }
}
