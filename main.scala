import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating

import org.json4s._
import org.json4s.jackson.JsonMethods._

object AddonRecommender {
  def main(args: Array[String]) = {
    implicit lazy val formats = DefaultFormats

    val sc = new SparkContext("local[8]", "AddonRecommender")

    val ratings = sc.textFile("nightly").map(raw => {
      val parsedPing = parse(raw.substring(37))
      (parsedPing \ "clientID", parsedPing \ "addonDetails" \ "XPI")
    }).filter{
      // Remove sessions with missing id or add-on list
      case (JNothing, _) => false
      case (_, JNothing) => false
      case (_, JObject(List())) => false
      case _ => true
    }.map{ case (id, xpi) => {
      val addonList = xpi.children.
        map(addon => addon \ "name").
        filter(addon => addon != JNothing && addon != JString("Default"))
      (id, addonList)
    }}.filter{ case (id, addonList) => {
      // Remove sessions with empty add-on lists
      addonList != List()
    }}.flatMap{ case (id, addonList) => {
      // Create add-on ratings for each user
      addonList.map(addon => (id.extract[String], addon.extract[String], 1.0))
    }}.union(sc.parallelize(Array(("A random session", "Ghostery", 1.0))))

    def hash(x: String) = x.hashCode & 0x7FFFFF // Positive hash

    // Build list of all addon ids
    val addonIDs = ratings.map(_._2).distinct.map(addon => (hash(addon), addon)).cache

    // Train model with 40 features, 10 iterations of ALS
    val model = ALS.trainImplicit(ratings.map{case(u, a, r) => Rating(hash(u), hash(a), r)}, 40, 10)

    def recommend(userID: Int) = {
      val predictions = model.predict(addonIDs.map(addonID => (userID, addonID._1)))
      val top = predictions.top(5)(Ordering.by[Rating,Double](_.rating))
      top.map(r => (addonIDs.lookup(r.product)(0), r.rating))
    }

    println(recommend(hash("A random session")).map(println))

    sc.stop()
  }
}
