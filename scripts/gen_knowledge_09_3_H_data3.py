#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel bank: math topics (fresh for batch H)."""


def register(T):
    T["the binomial theorem"] = {
        "_cat": "math",
        "what": "a formula that expands the n-th power of a binomial (a + b) into a sum of terms involving binomial coefficients C(n,k), giving (a+b)^n as the sum from k=0 to n of C(n,k) a^(n-k) b^k",
        "how": "the coefficient C(n,k) counts the number of ways to choose k of the n factors to contribute b (with the rest contributing a). Pascal's triangle organizes the coefficients, with each entry the sum of the two above it. Algebraic induction or combinatorial counting both prove the formula. Generalizations cover negative and fractional exponents (Newton's binomial series) and noncommutative settings",
        "why": "the binomial theorem is foundational in algebra, probability (binomial distribution), and combinatorics. It underlies polynomial expansions in calculus, generating functions, and series approximations like (1+x)^n for small x in physics and finance",
        "vs": "the binomial theorem differs from the multinomial theorem, which expands powers of sums with more than two terms. It differs from polynomial multiplication in general by giving a closed-form sum rather than an algorithm. It differs from the binomial distribution, which is the probabilistic application of the same coefficients",
        "ex": "(x + y)^4 expands to x^4 + 4x^3 y + 6x^2 y^2 + 4 x y^3 + y^4, with coefficients 1, 4, 6, 4, 1 read directly from row 4 of Pascal's triangle. Each term counts the ways to assign x or y across the four factors",
        "mis": "people think binomial expansion only works for positive integer powers. Newton's generalization handles any real exponent as an infinite series for |x| < 1. Another myth is that the binomial coefficients are mysterious; they are exactly the count of subsets of a given size",
    }

    T["the prime number theorem"] = {
        "_cat": "math",
        "what": "the asymptotic statement that the number of primes less than or equal to x, denoted pi(x), grows like x divided by the natural logarithm of x as x approaches infinity",
        "how": "primes thin out among the integers, with density near n approximately 1 over log n. Adding up these densities from 2 to x gives roughly x over log x. The full theorem is proved using complex analysis of the Riemann zeta function, by showing it has no zeros on the line Re(s) = 1; later 'elementary' proofs by Selberg and Erdos avoided complex analysis but were intricate",
        "why": "the prime number theorem quantifies prime distribution, anchors analytic number theory, calibrates expectations for cryptographic prime generation, and connects to the Riemann hypothesis whose error term it would refine. It is one of the great theorems in pure mathematics",
        "vs": "the prime number theorem differs from Bertrand's postulate, which guarantees a prime between n and 2n but does not give density. It differs from twin prime estimates and the Riemann hypothesis, which refine the error term and give finer information about prime locations",
        "ex": "below one million, there are 78498 primes; the approximation x/log x predicts about 72382, a 7 percent underestimate. The closer Li(x) (the logarithmic integral) approximation gives 78627, off by less than 0.2 percent",
        "mis": "people think primes become rare quickly. They thin slowly: even at 10^18 there are still primes about every 40 integers on average. Another myth is that the theorem gives an exact count; it gives the leading-order asymptotic, with significant fluctuations on top",
    }

    T["linear regression"] = {
        "_cat": "math",
        "what": "a statistical method that models the relationship between a response variable and one or more predictors as a linear combination of the predictors plus an error term, with coefficients chosen to minimize the sum of squared residuals",
        "how": "given n observations with predictor matrix X and response vector y, ordinary least squares solves the normal equations X^T X b = X^T y to get coefficient estimates b = (X^T X)^-1 X^T y. Under standard assumptions (linearity, independence, homoscedasticity, normality of errors), b is the best linear unbiased estimator and standard errors give confidence intervals and tests",
        "why": "linear regression is the workhorse of applied statistics: economics, epidemiology, social science, machine learning. It also underlies more elaborate models (logistic regression, GLMs, mixed models) and provides interpretable coefficients that quantify effects and adjust for confounders",
        "vs": "linear regression differs from correlation in modeling a directed relationship and providing predictions, from logistic regression in handling continuous rather than binary outcomes, and from nonparametric regression (splines, kernel smoothers) in assuming a parametric linear form",
        "ex": "regressing housing price on square footage, number of bedrooms, and neighborhood gives coefficients interpretable as the marginal price of an extra square foot or bedroom, controlling for the other variables. Real estate appraisers use such models routinely",
        "mis": "people think linear regression assumes the predictors are linear in their raw form. It assumes linearity in the coefficients; predictors can be polynomial, log-transformed, or interacted. Another myth is that a high R-squared means a good model; R-squared can be high with violated assumptions or omitted confounders",
    }

    T["Euclid's algorithm for greatest common divisor"] = {
        "_cat": "math",
        "what": "an iterative procedure for computing the greatest common divisor of two integers by repeatedly replacing the larger number with its remainder when divided by the smaller, terminating when the remainder is zero",
        "how": "to find gcd(a,b) with a > b, compute a mod b, then set a = b and b = a mod b, and repeat. The remainders strictly decrease and are nonnegative integers, so the process terminates. The last nonzero remainder is the gcd. The extended algorithm tracks coefficients to express gcd(a,b) as a linear combination of a and b",
        "why": "Euclid's algorithm is fast (O(log(min(a,b))) divisions), is the foundation for modular arithmetic, RSA key generation, polynomial gcd, and the structure theorem for finitely generated abelian groups. It is one of the oldest algorithms still in active use, dating to about 300 BCE",
        "vs": "Euclid's algorithm differs from prime factorization, which is much slower but reveals more structure. It differs from the binary GCD algorithm, which avoids division and uses bit shifts, sometimes faster on hardware. It differs from the Stern-Brocot tree-based methods that organize fractions",
        "ex": "gcd(252, 105) computes as 252 = 2*105 + 42, then gcd(105, 42) = 105 = 2*42 + 21, then gcd(42, 21) = 42 = 2*21 + 0, so gcd = 21. RSA decryption uses the extended version to compute multiplicative inverses modulo a prime",
        "mis": "people think Euclid's algorithm is slow because it iterates. It is exponentially faster than checking common divisors directly; even huge integers complete in microseconds. Another myth is that it requires a > b; if not, the first step swaps them automatically",
    }

    T["the chain rule of differentiation"] = {
        "_cat": "math",
        "what": "the calculus rule that gives the derivative of a composite function f(g(x)) as f'(g(x)) times g'(x), the product of the outer function's derivative evaluated at g(x) and the inner function's derivative",
        "how": "intuitively, if y depends on u and u depends on x, then a small change dx produces a change du = g'(x) dx in u, which produces a change dy = f'(u) du in y. Multiplying gives dy/dx = f'(u) g'(x). Formally, the proof uses limits and an error term that vanishes; many treatments use the increment form of the derivative",
        "why": "the chain rule is the engine of calculus on composite functions: implicit differentiation, related rates, the multivariable chain rule, and crucially backpropagation in neural networks all rely on it. It also underlies physics calculations involving change of variables and time-dependent processes",
        "vs": "the chain rule differs from the product rule (derivative of f times g) and the quotient rule (derivative of f over g), although all three are derived from the same limit definition. It differs from substitution in integration, which is the chain rule applied in reverse",
        "ex": "differentiating sin(x^2) gives cos(x^2) times 2x by the chain rule, with f = sin and g = x^2. Backpropagation in deep networks repeatedly applies the chain rule across layers, with each layer's local derivative chained into the gradient of the loss",
        "mis": "people think the chain rule is just multiplication of derivatives. It is multiplication, but with the outer derivative evaluated at the inner function's value, a subtlety that causes most errors. Another myth is that it only applies to two-function compositions; it generalizes to any number of nested functions",
    }

    T["the Cauchy-Schwarz inequality"] = {
        "_cat": "math",
        "what": "a fundamental inequality stating that for any vectors u and v in an inner product space, the absolute value of their inner product is at most the product of their norms, with equality if and only if the vectors are linearly dependent",
        "how": "the proof considers the nonnegative quadratic in t given by ||u - tv||^2 = ||u||^2 - 2t<u,v> + t^2 ||v||^2. Choosing t = <u,v>/||v||^2 minimizes the quadratic; the minimum is nonnegative, which rearranges to <u,v>^2 <= ||u||^2 ||v||^2. Variants for sums, integrals, and probabilities follow the same template",
        "why": "Cauchy-Schwarz underpins the triangle inequality (and thus the very concept of distance), the definition of correlation coefficients in statistics, energy methods in physics, and dozens of bounds in analysis and information theory. It is one of the most reused inequalities in mathematics",
        "vs": "Cauchy-Schwarz differs from the triangle inequality (||u + v|| <= ||u|| + ||v||), which is its near-corollary, and from the AM-GM inequality, which compares arithmetic and geometric means. It differs from Holder's inequality, which generalizes Cauchy-Schwarz to L^p norms",
        "ex": "the Pearson correlation coefficient between two random variables is bounded by 1 in absolute value precisely because of Cauchy-Schwarz applied to centered random variables in the L^2 inner product. The correlation reaches plus or minus 1 when one variable is a linear function of the other",
        "mis": "people think Cauchy-Schwarz only applies to vectors in R^n. It works in any inner product space, including infinite-dimensional ones like L^2 functions, and underlies quantum mechanics's expectation value bounds. Another myth is that the equality case is rare; it is exactly linear dependence, which is sharply defined",
    }

    T["topological spaces"] = {
        "_cat": "math",
        "what": "a generalization of geometric spaces consisting of a set together with a collection of open subsets satisfying axioms (the empty set and full set are open, finite intersections of open sets are open, arbitrary unions of open sets are open) that captures abstract notions of nearness without metric",
        "how": "from the open sets, one defines closed sets (complements of open), continuity (preimages of open sets are open), connectedness (no nontrivial partition into disjoint open sets), compactness (every open cover has a finite subcover), and convergence (filters or nets approach a point). The same set can support many topologies; the discrete and indiscrete are extremes",
        "why": "topology unifies analysis, geometry, and combinatorics: it gives a setting for limits and continuity that includes function spaces, manifolds, and exotic objects like the Cantor set. Algebraic topology connects shape to algebra, with applications to data analysis (persistent homology), physics (gauge theories), and economics (fixed-point theorems)",
        "vs": "topological spaces differ from metric spaces in not requiring a numerical distance, capturing more general notions of nearness. They differ from manifolds, which add local Euclidean structure, and from measure spaces, which encode size rather than shape",
        "ex": "the topology of the real line treats open intervals as basic open sets. Removing this structure but keeping the underlying set, you can give R the discrete topology where every singleton is open, making continuous functions trivially defined and yielding very different behavior",
        "mis": "people think topology is the same as 'rubber-sheet geometry'. That is intuition for topological equivalence, but the field encompasses spaces no rubber sheet captures (function spaces, the Sorgenfrey line). Another myth is that topology requires advanced background; the basic axioms are accessible but their consequences run deep",
    }

    T["the integral test for series convergence"] = {
        "_cat": "math",
        "what": "a criterion in calculus that determines whether a series of positive terms a_n = f(n) converges by checking whether the corresponding improper integral of f from some constant N to infinity converges, requiring f to be positive, continuous, and decreasing on [N, infinity)",
        "how": "if f satisfies the conditions, the partial sums of the series can be sandwiched between integrals of f on adjacent intervals. Specifically, the integral from k to infinity bounds the tail of the sum from below by the integral from k+1 to infinity and from above by f(k) plus the integral from k to infinity. Convergence of one implies convergence of the other; divergence likewise",
        "why": "the integral test settles convergence for many classical series, including the p-series sum 1/n^p (convergent iff p > 1) and the logarithmic-corrected series. It links discrete and continuous analysis, justifies many tail bounds in probability, and is the standard tool when ratio and root tests fail",
        "vs": "the integral test differs from the ratio test (which compares consecutive terms) and the root test (which examines n-th roots). It differs from the comparison test in requiring a continuous integrable analog rather than a separate series for comparison, and from Abel's and Dirichlet's tests, which handle alternating-like series",
        "ex": "the harmonic series sum 1/n diverges because the integral of 1/x from 1 to infinity is log x, which diverges. The series sum 1/n^2 converges because the integral of 1/x^2 from 1 to infinity equals 1, finite",
        "mis": "people think the integral and the series have the same value when both converge. They have the same convergence behavior but different sums; the series typically lies between two integrals that differ by f at the endpoints. Another myth is that the test requires the function to be monotone everywhere; it only needs to be eventually monotone",
    }

    T["the Fundamental Theorem of Algebra"] = {
        "_cat": "math",
        "what": "the theorem that every non-constant polynomial with complex coefficients has at least one complex root, equivalently that a degree-n polynomial has exactly n complex roots counting multiplicity",
        "how": "many proofs exist. A topological argument considers the image of a large circle under the polynomial, which winds around zero n times, while the image of a tiny circle near zero hardly winds at all; some intermediate circle must pass through zero. Liouville's theorem gives a complex-analytic proof: 1/p(z) would be bounded entire if p had no zeros, hence constant, contradiction",
        "why": "the theorem makes the complex numbers algebraically closed, justifying their primacy in algebra, signal processing (Fourier analysis), control theory (pole placement), and quantum mechanics. It is a structural result that places the complex numbers as the natural setting for polynomial equations",
        "vs": "the fundamental theorem of algebra differs from the rational roots theorem (which lists possible rational roots), the abel-Ruffini theorem (which says no general radical formula exists for degree five and above), and Galois theory (which characterizes solvability). It differs from the fundamental theorem of arithmetic (unique prime factorization)",
        "ex": "x^2 + 1 has no real roots but has two complex roots, i and -i. x^4 - 1 factors over the complex numbers as (x-1)(x+1)(x-i)(x+i), exactly four roots as the theorem predicts",
        "mis": "people think the theorem gives a formula for the roots. It only proves existence; finding roots of high-degree polynomials requires numerical methods. Another myth is that the proof is purely algebraic; all known proofs use some analytic or topological input",
    }

    T["Bayesian updating"] = {
        "_cat": "math",
        "what": "the process of revising the probability of a hypothesis as new evidence is observed, using Bayes' theorem to combine a prior probability and a likelihood into a posterior probability",
        "how": "start with a prior P(H) reflecting belief before evidence. When data D is observed, compute the likelihood P(D|H), the probability of seeing this data if H were true. Apply Bayes' theorem: P(H|D) = P(D|H) P(H) / P(D), where P(D) is the marginal probability of the data summed over hypotheses. The posterior becomes the prior for the next observation, enabling iterative updating",
        "why": "Bayesian updating is the rational way to incorporate evidence. It powers spam filters, medical diagnosis (especially with imperfect tests), Bayesian networks, particle filters in robotics, and modern probabilistic AI. It also formalizes how belief should respond to data, with applications in epistemology and decision theory",
        "vs": "Bayesian updating differs from frequentist hypothesis testing, which yields p-values and rejects or fails to reject without producing a probability for the hypothesis. It differs from machine-learning training, although Bayesian neural networks combine the two. It differs from heuristic updating, which often violates probability axioms",
        "ex": "a 99 percent accurate test for a disease with 1 percent prevalence yields a positive result whose posterior probability of disease is only about 50 percent. The prior of 0.01 combined with sensitivity 0.99 and specificity 0.99 gives roughly equal weight to true positive and false positive scenarios",
        "mis": "people think the prior 'biases' the conclusion. The prior is necessary; without it, posteriors are undefined. Another myth is that Bayesian and frequentist methods always disagree. With weak priors and large samples they often coincide; the gap matters most for small samples or strong priors",
    }

    T["the pigeonhole principle in problem solving"] = {
        "_cat": "math",
        "what": "the elementary observation that if n+1 objects are placed into n boxes, at least one box contains two or more objects, and its generalizations that yield surprising existence results in combinatorics, geometry, and number theory",
        "how": "the proof is contrapositive: if every box held at most one object, total would be at most n, contradicting n+1. Generalizations replace n+1 with kn+1 (some box holds at least k+1) and weight the objects to handle continuous cases. The trick in applications is choosing the right pigeons and pigeonholes",
        "why": "the principle proves the existence of objects with given properties without constructing them, underlies Ramsey theory, the Chinese remainder theorem's existence claims, lossy compression bounds, and many olympiad problems. Its power lies in turning a counting argument into an existence proof",
        "vs": "the pigeonhole principle differs from the inclusion-exclusion principle (which counts sizes of unions) and from generating function arguments (which encode combinatorial structure algebraically). It differs from the probabilistic method, which uses expected counts to prove existence, although they share a non-constructive flavor",
        "ex": "in any group of 13 people, at least two share a birth month. With 367 people, at least two share a birthday. More elaborately, a chess tournament with 6 players guarantees three who all played each other or three who all did not, an instance of Ramsey's theorem provable via pigeonhole",
        "mis": "people think the principle is trivial. The principle is trivial; the cleverness is in choosing pigeons and pigeonholes for a non-obvious problem. Another myth is that it gives concrete examples; it usually only proves existence, leaving construction to other methods",
    }

    T["the law of total probability"] = {
        "_cat": "math",
        "what": "the rule that when an event A can occur through several mutually exclusive scenarios B1, B2, ..., Bn that exhaust all possibilities, the total probability of A is the sum of P(A|Bi) P(Bi) over all i",
        "how": "partition the sample space into disjoint events Bi whose union is everything. For any event A, A is the disjoint union of intersections A and Bi. Summing those gives P(A) = sum P(A and Bi) = sum P(A|Bi) P(Bi). The formula extends to continuous variables as integrals against the marginal density and underlies marginalization and Bayes' theorem",
        "why": "the law is the workhorse of probability calculations involving conditional reasoning. It justifies marginalizing nuisance variables, computing test accuracy across populations, and combining scenario analyses. Bayes' theorem follows from it directly. Whenever a problem decomposes into conditional cases, this formula assembles the pieces",
        "vs": "the law of total probability differs from the multiplication rule (which gives joint probabilities) and from independence (which constrains how joints factor). It differs from Bayes' theorem in being a forward-direction summation rather than an inversion of conditioning",
        "ex": "if 60 percent of emails are spam and 5 percent of spam contains a certain phrase, while 0.1 percent of legitimate emails contain it, the overall fraction of emails with the phrase is 0.6*0.05 + 0.4*0.001 = 0.0304, about 3 percent. Bayes' theorem then gives the spam probability given the phrase",
        "mis": "people think conditional probabilities apply to a population without weighting by base rates. The law is precisely the corrective: each conditional must be weighted by the probability of its scenario. Another myth is that it requires equal partition; the partition just needs to be disjoint and exhaustive",
    }

    T["finite state machines"] = {
        "_cat": "math",
        "what": "a mathematical model of computation consisting of a finite set of states, an input alphabet, and a transition function specifying the next state given the current state and input, used to recognize regular languages and model many practical systems",
        "how": "the machine starts in an initial state. On each input symbol it consults the transition function to choose its next state. After consuming the input, it accepts if it ended in an accepting state. Deterministic FSMs (DFAs) have one transition per (state, symbol); nondeterministic ones (NFAs) allow multiple. NFAs can be converted to equivalent DFAs by subset construction. Regular expressions, FSMs, and regular grammars all describe the same class of languages",
        "why": "finite state machines underlie regular expressions (used in every programmer's daily work), lexical analysis in compilers, network protocol implementations, hardware controllers, video game AI, and verification of safety properties in critical systems. Their tractability (linear-time recognition, decidable equivalence) makes them ubiquitous wherever bounded patterns matter",
        "vs": "finite state machines differ from pushdown automata (which add a stack and recognize context-free languages) and Turing machines (unlimited tape, full computation). They differ from Markov chains, which add probability to transitions, and from neural networks, which compute continuous functions over inputs",
        "ex": "a vending machine that accepts coins and dispenses a soda when enough money is inserted is naturally a finite state machine. States represent total money received; transitions represent coin insertions; the dispense state corresponds to reaching the price. Engineers routinely model such systems as FSMs to reason about correctness",
        "mis": "people think FSMs are ancient and obsolete. They power regular expression engines and protocol implementations every day; statecharts are the basis of UI frameworks. Another myth is that FSMs always have few states; real-world FSMs (compilers, protocols) can have thousands, but the formal framework still applies",
    }

    T["the gradient in multivariable calculus"] = {
        "_cat": "math",
        "what": "the vector of partial derivatives of a scalar function f(x1, x2, ..., xn), pointing in the direction of steepest increase of f at a given point and having magnitude equal to the rate of that increase",
        "how": "compute partial derivatives df/dxi for each variable while holding the others fixed; assemble them into the gradient vector grad f = (df/dx1, df/dx2, ..., df/dxn). The directional derivative of f in any unit direction u equals grad f dot u. The level sets of f are perpendicular to grad f, which is why gradient descent moves along the steepest descent direction (the negative gradient)",
        "why": "the gradient is central to optimization: gradient descent and its variants train almost every machine learning model, including neural networks via backpropagation. It also drives physics (gradient of potential energy gives force), economics (marginal utility vectors), and image processing (edge detection)",
        "vs": "the gradient differs from the divergence (a scalar measuring outflow of a vector field) and the curl (a vector measuring rotation). It differs from the Hessian, which is the second-derivative matrix capturing curvature. It differs from the Jacobian, which generalizes the gradient to vector-valued functions",
        "ex": "for f(x,y) = x^2 + y^2, the gradient is (2x, 2y), pointing radially outward. At (1,1) the gradient is (2,2), so f increases fastest in that direction at rate sqrt(8). Gradient descent from (1,1) with step size 0.1 moves to (0.8, 0.8), reducing f from 2 to 1.28",
        "mis": "people think the gradient is the slope. It is a vector that includes direction, while slope is its magnitude in a chosen direction. Another myth is that the gradient always points uphill in the geometric sense; on curved surfaces, intuition can mislead, and the formal definition is the only safe guide",
    }
