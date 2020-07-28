spark-submit \
  --class nyc.NYCFeatureEngineering\
  --master yarn\
  --deploy-mode client\
  --driver-memory 64g\
  --num-executors 30\
  --executor-cores 4\
  --executor-memory 16g\
  nyc_taxitrips_featureengineering_2.12-0.1.jar