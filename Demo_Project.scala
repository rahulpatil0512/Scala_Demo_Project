//import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator

object Demo_Project {
  def main(args:Array[String]): Unit ={
    // Create a Spark session
    val spark = SparkSession.builder.appName("CO2Prediction").getOrCreate()

    // Load your dataset
    val data = spark.read.option("header", "true").csv("C://Users//Rahul Patil//Desktop//Data Analyticas using Scala Programming//FuelConsumpionCo2.csv")

    // Select relevant columns
    val selectedColumns = Array("ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB", "FUELCONSUMPTION_COMB_MPG", "CO2EMISSIONS")

    // Drop rows with missing values
    val selectedData = data.select(selectedColumns.map(col): _*).na.drop()

    // Convert categorical variables to numerical using StringIndexer
    val makeIndexer = new StringIndexer().setInputCol("MAKE").setOutputCol("MAKE_INDEX")
    val modelIndexer = new StringIndexer().setInputCol("MODEL").setOutputCol("MODEL_INDEX")
    val vehicleClassIndexer = new StringIndexer().setInputCol("VEHICLECLASS").setOutputCol("VEHICLECLASS_INDEX")
    val fuelTypeIndexer = new StringIndexer().setInputCol("FUELTYPE").setOutputCol("FUELTYPE_INDEX")

    // Assemble feature vector
    val assembler = new VectorAssembler().setInputCols(Array("ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB", "FUELCONSUMPTION_COMB_MPG", "MAKE_INDEX", "MODEL_INDEX", "VEHICLECLASS_INDEX", "FUELTYPE_INDEX")).setOutputCol("features")

    // Define the linear regression model
    val lr = new LinearRegression().setLabelCol("CO2EMISSIONS").setFeaturesCol("features")

    // Create a pipeline
    val pipeline = new Pipeline().setStages(Array(makeIndexer, modelIndexer, vehicleClassIndexer, fuelTypeIndexer, assembler, lr))

    // Split the data into training and testing sets
    val Array(trainingData, testData) = selectedData.randomSplit(Array(0.8, 0.2), seed = 1234)

    // Fit the model on the training data
    val model = pipeline.fit(trainingData)

    // Make predictions on the testing data
    val predictions = model.transform(testData)

    // Evaluate the model
    val evaluator = new RegressionEvaluator().setLabelCol("CO2EMISSIONS").setPredictionCol("prediction").setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    // Stop the Spark session
    spark.stop()
  }

}