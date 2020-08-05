import com.hankcs.hanlp.tokenizer.StandardTokenizer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.feature.{HashingTF, IDF, StopWordsRemover, Tokenizer}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import redis.clients.jedis.Jedis

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer

object Classifier {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val spark = SparkSession.builder().appName("CategoryClassification")
//      .master("local[*]")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "512")
      .getOrCreate()

//    val c = JedisConnectionPool.getConnection
//    //    val c = new Jedis("10.205.48.52")
//    c.auth("4edeac33a259e17ae3dfeb358fea6773dbb68d75")
//    c.select(2)


    import spark.implicits._
    val file = "file:///Users/james/Desktop/silk_clean.json"
    val df = spark.sparkContext.parallelize(args.slice(1,args.length-1)).toDF("title")
    //    val data = spark.read.json(df).select("title")
    val data = df
      .map(x => {
        val title = x.getString(0)
        val token = StandardTokenizer.segment(title).map(x => {
          x.word.replaceAll(" ", "")
        }).toList
        (title, token)
      }).toDF("title", "token")



    val stopWord = spark.read.text("/user/james/stopword.txt")
      .collect().map(_.getString(0))
    val swRemove = new StopWordsRemover()
      .setInputCol("token")
      .setOutputCol("stopWordRemove")
      .setStopWords(stopWord)
      .transform(data)

    val hashTH = new HashingTF()
      .setNumFeatures(300)
      .setInputCol("stopWordRemove")
      .setOutputCol("feature")
      .transform(swRemove)

    val idf = new IDF()
      .setInputCol("feature")
      .setOutputCol("features")
      .fit(hashTH).transform(hashTH).select("title", "features")



    val mapL = Map(0.0 -> "巨乳", 1.0 -> "調教", 2.0 -> "同性", 3.0 -> "多人", 4.0 -> "口交", 5.0 -> "自拍", 6.0 -> "不倫", 7.0 -> "其他")
    val mapP = Map("巨乳" -> 0.0, "調教" -> 1.0, "同性" -> 2.0, "多人" -> 3.0, "口交" -> 4.0, "自拍" -> 5.0, "不倫" -> 6.0, "其他" -> 7.0)
    val classifier = LogisticRegressionModel.load("/user/james/classify_model_ch")
      .transform(idf)
      .select("title", "probability", "prediction")
      .map(x => {
        val probability = x.getAs[DenseVector](1)
        val title = x.getString(0)
        val predict = ListBuffer[String]()
        probability.toArray.foreach(y => {
          val index = probability.toArray.indexOf(y).toDouble
          val l = mapL.get(index).toString.replace("Some(", "").replace(")", "")
          if (index == 0.0 && y > 0.4) //巨乳
          predict.append(l)
          if (index == 1.0 && y > 0.4) //調教
          predict.append(l)
          if (index == 2.0 && y > 0.8) //同性
          predict.append(l)
          if (index == 3.0 && y > 0.35) //多人
          predict.append(l)
          if (index == 4.0 && y > 0.5) //口交
          predict.append(l)
          if (index == 5.0 && y > 0.5) //自拍
          predict.append(l)
          if (index == 6.0 && y > 0.4) //不倫
          predict.append(l)
        })
        (predict, title)
      }).toDF("prediction", "title")
      .orderBy("prediction")

    val result = classifier.toJSON.collect().mkString("[", ", ", "]")
    classifier.repartition(1).write.json(s"/user/james/classify_result/${args(args.length-1)}")
//    c.set(args(args.length-1), result)
//    c.close()
    spark.close()



  }
}
