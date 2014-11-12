import scala.math._

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.util.MLUtils

import org.json4s._
import org.json4s.jackson.JsonMethods._

object AddonRecommender extends App {
  implicit lazy val formats = DefaultFormats

  val conf = new SparkConf().setAppName("AddonRecommender")
  val sc = new SparkContext(conf)

  val ratings = sc.textFile("s3://mreid-test-src/split/").map(raw => {
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
  }}.union(sc.parallelize(Array(("Ghostery Session", "Ghostery", 1.0),
                                ("Firebug Session", "Firebug", 1.0))))

  // Positive hash function
  def hash(x: String) = x.hashCode & 0x7FFFFF

  val hashedRatings = ratings.map{ case(u, a, r) => (hash(u), hash(a), r) }.cache
  val addonIDs = ratings.map(_._2).distinct.map(addon => (hash(addon), addon)).cache

  // Use cross validation to find the optimal number of latent factors
  val folds = MLUtils.kFold(hashedRatings, 10, 42)
  val lambdas = List(0.1, 0.2, 0.3, 0.4, 0.5)
  val iterations = 10
  val factors = 100 // use as many factors as computationally possible

  val factorErrors = lambdas.flatMap(lambda => {
    folds.map{ case(train, test) =>
      val model = ALS.trainImplicit(train.map{ case(u, a, r) => Rating(u, a, r) }, factors, iterations, lambda, 1.0)
      val usersAddons = test.map{ case (u, a, r) => (u, a) }
      val predictions = model.predict(usersAddons).map{ case Rating(u, a, r) => ((u, a), r) }
      val ratesAndPreds = test.map{ case (u, a, r) => ((u, a), r) }.join(predictions)
      val rmse = sqrt(ratesAndPreds.map { case ((u, a), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
      }.mean)

      (model, lambda, rmse)
    }
  }).groupBy(_._2)
    .map{ case(k, v) => (k, v.map(_._3).reduce(_ + _) / v.length) }

  val optimalLambda = factorErrors.reduce((x, y) => if (x._2 < y._2) x else y)

  // Train model with optimal number of factors on all available data
  val model = ALS.trainImplicit(hashedRatings.map{case(u, a, r) => Rating(u, a, r)}, factors, iterations, optimalLambda._1, 1.0)

  def recommend(userID: Int) = {
    val predictions = model.predict(addonIDs.map(addonID => (userID, addonID._1)))
    val top = predictions.top(10)(Ordering.by[Rating,Double](_.rating))
    top.map(r => (addonIDs.lookup(r.product)(0), r.rating))
  }

  // Print top add-ons for fictional sessions
  val ghostery = recommend(hash("Ghostery Session"))
  val firebug = recommend(hash("Firebug Session"))

  println("Ghostery Session recommendations")
  ghostery.map(println)

  println("Firebug Session recommendations")
  firebug.map(println)

  // Print some useful information
  println("optimal lambda = " + optimalLambda._1)

  factorErrors.map{ case (lambda, rmse) =>
    println("Model with lambda " + lambda + " has RMSE of " + rmse)
  }

  sc.stop()
}
