
import com.hankcs.hanlp.tokenizer.StandardTokenizer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes, OneVsRest}
import org.apache.spark.ml.feature.{HashingTF, IDF, StopWordsRemover, Tokenizer, Word2Vec}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{Dataset, SparkSession, functions}
import org.apache.spark.sql.functions.array_contains
import scala.collection.JavaConversions._
import scala.collection.mutable.{ArrayBuffer, ListBuffer}

object CategoryClassification {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val spark = SparkSession.builder().appName("CategoryClassification")
      .master("local[*]")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "512")
      .getOrCreate()

    import spark.implicits._
    val file = "file:///Users/james/Desktop/ch_all.json"
    val data = spark.read.json(file).select("label", "title").dropDuplicates("title")
      .flatMap(x => {
        val category = x.getAs[Seq[String]](0)
        val label = ListBuffer[(Int, String, Seq[String], List[String])]()
        val title = x.getString(1)
        val token = StandardTokenizer.segment(title).map(x => {
          x.word.replaceAll(" ", "")
        }).toList
        if (category.contains("巨乳")) label.append((0, title, category, token))
        if (category.contains("調教")) label.append((1, title, category, token))
        if (category.contains("同性")) label.append((2, title, category, token))
        if (category.contains("多人")) label.append((3, title, category, token))
        if (category.contains("口交")) label.append((4, title, category, token))
        if (category.contains("自拍")) label.append((5, title, category, token))
        if (category.contains("不倫")) label.append((6, title, category, token))
        if (category.contains("其他")) label.append((7, title, category, token))
        label
      }).toDF("label", "title", "category", "token")
//      .filter(!$"label".equalTo(7))


    val stopWord = spark.read.text("file:///Users/james/IdeaProjects/categoryClassification/src/main/scala/stopword.txt")
      .collect().map(_.getString(0))
    val swRemove = new StopWordsRemover()
      .setInputCol("token")
      .setOutputCol("stopWordRemove")
      .setStopWords(stopWord)
      .transform(data)


    val hashTH = new HashingTF()
      .setNumFeatures(300)
      .setInputCol("token")
      .setOutputCol("feature")
      .transform(data)

    val idf = new IDF()
      .setInputCol("feature")
      .setOutputCol("features")
      .fit(hashTH).transform(hashTH).select("label", "title", "category", "features")

    val Array(train, test) = idf.randomSplit(Array(0.8, 0.2))

    val model = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxIter(5)
      .fit(idf)



    //        val model = new NaiveBayes()
    //          .setLabelCol("label")
    //          .setFeaturesCol("features")
    //          .fit(train)

    model.save("/user/james/classify_model_ch")

//        var correct = spark.sparkContext.doubleAccumulator
//        val count = test.count()
//        val prediction = model.transform(test).dropDuplicates("title").orderBy("title")
//        val mapL = Map(0.0 -> "巨乳", 1.0 -> "調教", 2.0 -> "同性", 3.0 -> "多人", 4.0 -> "口交", 5.0 -> "自拍", 6.0 -> "不倫", 7.0 -> "其他")
//        val output = prediction.select("label", "prediction", "probability", "title", "category")
//          .map(x => {
//            val category = x.getAs[Seq[String]](4)
//            val probability = x.getAs[DenseVector](2)
//            val title = x.getString(3)
//            val label = ListBuffer[String]()
//            if (category.contains("巨乳")) label.append("巨乳")
//            if (category.contains("調教")) label.append("調教")
//            if (category.contains("同性")) label.append("同性")
//            if (category.contains("多人")) label.append("多人")
//            if (category.contains("口交")) label.append("口交")
//            if (category.contains("自拍")) label.append("自拍")
//            if (category.contains("不倫")) label.append("不倫")
//            if (category.contains("其他")) label.append("其他")
//            val predict = ListBuffer[String]()
//            val p = mapL.get(x.getDouble(1)).toString.replace("Some(", "").replace(")", "")
//
//            if (label.contains(p)) {
//              predict.append(p)
////              predict.append(probability(x.getDouble(1).toInt).toString)
//              correct.add(1.0)
//            }
//            else {
//              predict.append("false")
//            }
//            (label, predict, title)
//          }).toDF("label", "prediction", "title").filter(!array_contains($"prediction", "false")).orderBy($"label", $"prediction")
//          .show(2000, false)
//
//        println(s"correct = ${correct.value}")
//        println(s"count = $count")
//        println(s"acc = ${correct.value / count}")

  }
}
