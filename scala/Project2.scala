package project2

import scala.collection.mutable.{ArrayBuffer, LinkedHashSet => LSET}

import scalation.*
import scalation.mathstat._
import scalation.modeling._
import scalation.modeling.given

/** Project 2: Regression and Symbolic Regression (ScalaTion side).
 *
 *  For each of the three UCI datasets (Auto MPG, Concrete Compressive Strength,
 *  Airfoil Self-Noise) this runs:
 *    1. Regression                        - all 3 datasets
 *    2. Ridge & Lasso Regularized Regression - AutoMPG only
 *    3. Forward Feature Selection         - 2 of 3 datasets
 *    4. Transformed Regression (log, sqrt, Box-Cox) - 2 of 3 datasets
 *    5. Symbolic Regression               - all 3 datasets
 *
 *  Run from a ScalaTion app directory with this file in src/main/scala/project2/.
 *  The CSV folder is PROJECT2_DATA_DIR if set, otherwise the first of
 *  "../project 2/data", "../data", "data" that contains the datasets.
 */
object Project2:

    private val DATA_DIR =
        val candidates = sys.env.get ("PROJECT2_DATA_DIR").toSeq ++
                         Seq ("../project 2/data", "../data", "data")
        val d = candidates.find (c => new java.io.File (c, "auto_mpg.csv").isFile)
                          .getOrElse (candidates.head)
        if d.endsWith ("/") || d.endsWith ("\\") then d else d + "/"

    /** Bundles the loaded matrices/vectors/names needed by every section below.
     *  @param x         predictor matrix, no intercept column
     *  @param y         response vector
     *  @param fname     predictor names (matches columns of x)
     *  @param ox        predictor matrix with a leading intercept column of ones
     *  @param ox_fname  names matching columns of ox (leads with "one")
     */
    case class Dataset (x: MatrixD, y: VectorD, fname: Array [String],
                        ox: MatrixD, ox_fname: Array [String])

    /** Load a cleaned, numeric CSV (with header) and split it into predictors/response.
     *  @param fileName  csv file name (relative to DATA_DIR)
     *  @param target    name of the target/response column
     */
    def load (fileName: String, target: String): Dataset =
        val (mat, hdr) = MatrixD.loadH (DATA_DIR + fileName, fullPath = true)
        val colIdx     = hdr.zipWithIndex.toMap
        val tIdx       = colIdx (target)

        val y     = mat (?, tIdx)
        val x     = mat.not (?, tIdx)
        val fname = for (nm, j) <- hdr.zipWithIndex if j != tIdx yield nm

        val _1       = VectorD.one (x.dim)
        val ox       = _1 +^: x
        val ox_fname = Array ("one") ++ fname
        Dataset (x, y, fname, ox, ox_fname)
    end load

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Section 1: Regression on all predictors (ScalaTion `Regression`).
     */
    def sectionRegression (ds: Dataset, name: String): Unit =
        banner (s"[$name] Section 1: Regression (ScalaTion)")
        val mod = new Regression (ds.ox, ds.y, ds.ox_fname)
        mod.inSample_Test ()
        println (mod.summary ())

        banner (s"[$name] Section 1: Regression 5-fold Cross-Validation")
        FitM.showQofStatTable (mod.crossValidate ())
    end sectionRegression

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Section 2: Ridge and Lasso Regularized Regression (ScalaTion), AutoMPG only.
     *  Chooses lambda by a small grid search minimizing cross-validated SSE.
     */
    def sectionRegularized (ds: Dataset, name: String): Unit =
        val lambdas = Array (0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0)

        banner (s"[$name] Section 2: Ridge Regression lambda search (ScalaTion)")
        var bestLamR = lambdas (0)
        var bestSseR = Double.MaxValue
        for lam <- lambdas do
            RidgeRegression.hp ("lambda") = lam
            val stats = RidgeRegression.center (ds.x, ds.y, ds.fname).crossValidate ()
            val sse   = stats (QoF.sse.ordinal).mean
            println (f"lambda = $lam%8.2f  ->  cv sse = $sse%12.4f")
            if sse < bestSseR then { bestSseR = sse; bestLamR = lam }
        end for

        banner (s"[$name] Ridge Regression: best lambda = $bestLamR (ScalaTion)")
        RidgeRegression.hp ("lambda") = bestLamR
        val ridge = RidgeRegression.center (ds.x, ds.y, ds.fname)
        ridge.inSample_Test ()
        println (ridge.summary ())

        banner (s"[$name] Section 2: Lasso Regression lambda search (ScalaTion)")
        var bestLamL = lambdas (0)
        var bestSseL = Double.MaxValue
        for lam <- lambdas do
            LassoRegression.hp ("lambda") = lam
            val stats = LassoRegression.center (ds.x, ds.y, ds.fname).crossValidate ()
            val sse   = stats (QoF.sse.ordinal).mean
            println (f"lambda = $lam%8.2f  ->  cv sse = $sse%12.4f")
            if sse < bestSseL then { bestSseL = sse; bestLamL = lam }
        end for

        banner (s"[$name] Lasso Regression: best lambda = $bestLamL (ScalaTion)")
        LassoRegression.hp ("lambda") = bestLamL
        val lasso = LassoRegression.center (ds.x, ds.y, ds.fname)
        lasso.inSample_Test ()
        println (lasso.summary ())
    end sectionRegularized

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Section 3: Forward Feature Selection (ScalaTion `Regression.forwardSelAll`).
     */
    def sectionForwardSelection (ds: Dataset, name: String): Unit =
        banner (s"[$name] Section 3: Forward Feature Selection (ScalaTion)")
        val mod = new Regression (ds.ox, ds.y, ds.ox_fname)
        mod.inSample_Test ()

        val (cols, rSq) = mod.forwardSelAll ()
        reportSelection (mod, cols, rSq)
    end sectionForwardSelection

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the forward-selection path and the best subset under each criterion.
     *  k counts predictors/terms excluding the intercept.  ScalaTion chooses each
     *  added term by sMAPE-IC (sMAPE + 2(k+1)/m) on the training split; R^2 and
     *  R^2bar are also on the training split, R^2cv is the 5-fold CV mean.
     */
    private def reportSelection (mod: Predictor, cols: LSET [Int], rSq: ArrayBuffer [VectorD]): Unit =
        val fn = newFname (mod.getFname, cols)
        println (s"terms added, in order (excluding intercept) = ${fn.drop (1).mkString (", ")}")
        println ("QoF after each addition (R^2, R^2bar, sMAPE: training split; R^2cv: 5-fold CV mean; all in %):")
        for i <- rSq.indices do
            val v = rSq (i)
            println (f"  k = $i%2d  R^2 = ${v(0)}%7.3f  R^2bar = ${v(1)}%7.3f  sMAPE = ${v(2)}%7.3f  R^2cv = ${v(3)}%7.3f")

        def show (label: String, k: Int): Unit =
            println (s"best by $label: k = $k of ${rSq.size - 1} -> ${fn.slice (1, k + 1).mkString (", ")}")
        val best = mod.getBest
        show (f"sMAPE-IC = ${best.qof (QoF.smapeC.ordinal)}%.4f (ScalaTion's selection criterion)",
              best.mod_cols.size - 1)
        show ("R^2bar", rSq.indices.maxBy (rSq (_)(1)))
        show ("R^2cv", rSq.indices.maxBy (rSq (_)(3)))
    end reportSelection

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Section 4: Transformed Regression (ScalaTion `TranRegression`):
     *  log (y), sqrt (y), and Box-Cox (y).
     */
    def sectionTransformed (ds: Dataset, name: String): Unit =
        banner (s"[$name] Section 4a: Transformed Regression -- log(y) (ScalaTion)")
        val logMod = new TranRegression (ds.ox, ds.y, ds.ox_fname, yℱ = LogForm ())
        logMod.inSample_Test ()
        println (logMod.summary ())

        banner (s"[$name] Section 4b: Transformed Regression -- sqrt(y) (ScalaTion)")
        val sqrtMod = new TranRegression (ds.ox, ds.y, ds.ox_fname, yℱ = RootForm ())
        sqrtMod.inSample_Test ()
        println (sqrtMod.summary ())

        banner (s"[$name] Section 4c: Transformed Regression -- Box-Cox(y), lambda = 0.4 (ScalaTion)")
        val bcMod = new TranRegression (ds.ox, ds.y, ds.ox_fname, yℱ = BoxcoxForm ())
        bcMod.inSample_Test ()
        println (bcMod.summary ())
    end sectionTransformed

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Section 5: Symbolic Regression (ScalaTion `SymbolicRegression`).
     *  Expands predictors with several powers and cross terms, then applies
     *  forward selection to pick the best-performing symbolic terms.
     */
    def sectionSymbolic (ds: Dataset, name: String): Unit =
        banner (s"[$name] Section 5: Symbolic Regression (ScalaTion)")
        // Keep the candidate power set small (sqrt, square): combining many
        // positive AND reciprocal/negative powers of the same variable makes
        // the expanded design matrix severely multicollinear (near-singular
        // normal equations) without materially improving fit.
        val powers = LSET (0.5, 2.0)
        val mod = SymbolicRegression (ds.x, ds.y, ds.fname, powers, null,
                                      intercept = true, cross = true)
        mod.inSample_Test ()
        println (mod.summary ())

        banner (s"[$name] Section 5: Symbolic Regression Forward Selection (ScalaTion)")
        val (cols, rSq) = mod.forwardSelAll ()
        reportSelection (mod, cols, rSq)

        banner (s"[$name] Symbolic Regression 5-fold Cross-Validation (full expanded model)")
        FitM.showQofStatTable (mod.crossValidate ())
    end sectionSymbolic

end Project2

import Project2._

/** > runMain project2.project2_autoMPG
 *  Runs Sections 1-5 for AutoMPG: Regression, Ridge/Lasso, Forward Selection,
 *  Transformed Regression, and Symbolic Regression.
 */
@main def project2_autoMPG (): Unit =
    val ds = load ("auto_mpg.csv", "mpg")
    sectionRegression (ds, "AutoMPG")
    sectionRegularized (ds, "AutoMPG")
    sectionForwardSelection (ds, "AutoMPG")
    sectionTransformed (ds, "AutoMPG")
    sectionSymbolic (ds, "AutoMPG")
end project2_autoMPG

/** > runMain project2.project2_concrete
 *  Runs Sections 1, 3, 5 for Concrete: Regression, Forward Selection, Symbolic Regression.
 */
@main def project2_concrete (): Unit =
    val ds = load ("concrete.csv", "concrete_compressive_strength")
    sectionRegression (ds, "Concrete")
    sectionForwardSelection (ds, "Concrete")
    sectionSymbolic (ds, "Concrete")
end project2_concrete

/** > runMain project2.project2_airfoil
 *  Runs Sections 1, 4, 5 for Airfoil: Regression, Transformed Regression, Symbolic Regression.
 */
@main def project2_airfoil (): Unit =
    val ds = load ("airfoil.csv", "scaled_sound_pressure_level")
    sectionRegression (ds, "Airfoil")
    sectionTransformed (ds, "Airfoil")
    sectionSymbolic (ds, "Airfoil")
end project2_airfoil
