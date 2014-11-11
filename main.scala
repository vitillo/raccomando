import scala.math._

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.util.MLUtils

import org.json4s._
import org.json4s.jackson.JsonMethods._

object AddonRecommender {
  def main(args: Array[String]) = {
    implicit lazy val formats = DefaultFormats

    val sc = new SparkContext("local[8]", "AddonRecommender")

    val ratings = sc.textFile("nightlyBIG").map(raw => {
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
    }}.union(sc.parallelize(Array(("A random session", "Ghostery", 1.0)))).cache

    // Positive hash function
    def hash(x: String) = x.hashCode & 0x7FFFFF

    // Build list of all addon ids
    val addonIDs = ratings.map(_._2).distinct.map(addon => (hash(addon), addon)).cache

    // Use cross validation to find the optimal number of latent factors
    val folds = MLUtils.kFold(ratings, 10, 42)
    val features = Array(20, 30, 40)

    val factorErrors = features.flatMap(n => {
      folds.map{ case(train, test) =>
        // Train model with n features, 10 iterations of ALS
        val model = ALS.trainImplicit(train.map{ case(u, a, r) => Rating(hash(u), hash(a), r) }, n, 10)
        val usersAddons = test.map{ case (u, a, r) => (hash(u), hash(a)) }
        val predictions = model.predict(usersAddons).map{ case Rating(u, a, r) => ((u, a), r) }
        val ratesAndPreds = test.map{ case (u, a, r) => ((hash(u), hash(a)), r) }.join(predictions)
        val rmse = sqrt(ratesAndPreds.map { case ((u, a), (r1, r2)) =>
          val err = (r1 - r2)
          err * err
        }.mean)
        (model, n, rmse)
      }
    }).groupBy(_._2)
      .map{ case(k, v) => (k, v.map(_._3).reduce(_ + _) / v.length) }

    val optimalN = factorErrors.reduce((x, y) => if (x._2 < y._2) x else y)

    // Train model with optimal number of factors on all available data
    val model = ALS.trainImplicit(ratings.map{case(u, a, r) => Rating(hash(u), hash(a), r)}, optimalN._1, 10)

    def recommend(userID: Int) = {
      val predictions = model.predict(addonIDs.map(addonID => (userID, addonID._1)))
      val top = predictions.top(5)(Ordering.by[Rating,Double](_.rating))
      top.map(r => (addonIDs.lookup(r.product)(0), r.rating))
    }

    // Print top 5 add-on for fictional session
    println(recommend(hash("A random session")).map(println))

    // Print some useful information
    println("optimalN = " + optimalN._1)

    factorErrors.map{ case (n, rmse) =>
      println("Model with " + n + " latent factors has RMSE of " + rmse)
    }

    sc.stop()
  }
}
