name := "categoryClassification"
version := "0.1"
scalaVersion := "2.11.12"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.0"
libraryDependencies += "com.hankcs" % "hanlp" % "portable-1.7.8"
libraryDependencies += "redis.clients" % "jedis" % "3.3.0"