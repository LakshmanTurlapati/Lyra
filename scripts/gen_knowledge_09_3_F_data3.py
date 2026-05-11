#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel bank part 2 of 4: math and technology topics (fresh for batch F)."""


def register(T):
    # ===== MATH =====

    T["eigenvalues and eigenvectors"] = {
        "_cat": "math",
        "what": "for a square matrix A, an eigenvector is a nonzero vector v such that Av equals lambda times v for some scalar lambda called the eigenvalue, capturing directions in which the linear transformation acts as pure scaling",
        "how": "to find them, solve det(A minus lambda I) equals zero, the characteristic polynomial, for the eigenvalues. Then for each lambda, solve (A minus lambda I) v equals zero for the eigenvectors. Together they diagonalize A when there are enough independent eigenvectors, turning matrix powers and exponentials into elementwise operations on the eigenvalues",
        "why": "eigen-decomposition powers principal component analysis, Google's PageRank, vibration mode analysis in engineering, quantum mechanics (energy eigenstates), stability analysis of dynamical systems, and spectral graph theory. It is the workhorse of applied linear algebra",
        "vs": "eigenvalues differ from singular values, which apply to non-square matrices and are always real and nonnegative. Eigenvectors differ from generalized eigenvectors, used when a matrix is not diagonalizable. They differ from null-space basis vectors, which solve Av equals zero rather than Av equals lambda v",
        "ex": "in PageRank, web pages form a transition matrix where entry ij is the probability of moving from page j to page i. The dominant eigenvector of this matrix gives the steady-state visit probabilities, which Google originally used as a ranking score",
        "mis": "people think every matrix has a full set of independent eigenvectors. Defective matrices do not, requiring Jordan form. Another myth is that eigenvalues are always real; for non-symmetric real matrices they can be complex conjugate pairs",
    }

    T["the chain rule in calculus"] = {
        "_cat": "math",
        "what": "the rule for differentiating composed functions: if h(x) equals f(g(x)), then h prime of x equals f prime of g(x) times g prime of x, expressing how a small change in x propagates through nested functions",
        "how": "view a composition as a chain of dependencies. A perturbation in x changes g by g prime, which then changes f by f prime evaluated at g(x). Multiply these local sensitivities. For functions of several variables the rule generalizes via the Jacobian matrix, with partial derivatives multiplied along each chain",
        "why": "the chain rule underlies backpropagation in neural networks (efficient gradient computation through deep compositions), the implicit function theorem, change of variables in integration, and most physics derivations involving composed quantities like position-dependent fields",
        "vs": "the chain rule differs from the product rule (derivative of a product) and the quotient rule (derivative of a ratio), which handle different combinations. It differs from the substitution rule for integrals, which is essentially the chain rule applied in reverse",
        "ex": "training a neural network with millions of parameters relies on the multivariate chain rule applied layer by layer. Reverse-mode automatic differentiation is the chain rule made systematic, computing all partial derivatives in one backward pass at the cost of one extra forward pass",
        "mis": "people memorize 'derivative of outer times derivative of inner' without seeing the structure. The chain rule is just multiplication of local linear approximations along a dependency graph. Another myth is that it requires the inner function to be invertible; it does not",
    }

    T["the gamma function"] = {
        "_cat": "math",
        "what": "an extension of the factorial to real and complex numbers, defined for positive real x as the integral from 0 to infinity of t to the (x minus 1) times e to the minus t dt, with the property that Gamma(n) equals (n minus 1) factorial for positive integers n",
        "how": "integration by parts on the defining integral gives the recursion Gamma(x plus 1) equals x times Gamma(x), the same recursion that drives the factorial. Analytic continuation extends Gamma to all complex numbers except non-positive integers, where it has simple poles. Stirling's approximation provides a useful asymptotic for large arguments",
        "why": "the gamma function appears throughout probability (gamma, chi-squared, beta, Student t distributions), special functions, statistical mechanics (partition functions), number theory (the Riemann zeta function uses it via the functional equation), and any context where factorials need a continuous extension",
        "vs": "the gamma function differs from the factorial (defined only on non-negative integers) and from the beta function (a closely related two-variable function). It differs from the digamma function, which is its logarithmic derivative, and from the gamma distribution, which uses Gamma(x) as a normalizing constant",
        "ex": "the volume of a unit ball in n dimensions is pi to the n/2 divided by Gamma(n/2 plus 1). For n equal 3 this is 4/3 times pi, the familiar sphere volume. For very high dimensions the volume shrinks toward zero, a key fact in high-dimensional geometry and statistics",
        "mis": "people expect Gamma(n) to equal n factorial. The standard convention has Gamma(n) equal to (n minus 1) factorial; the offset is historical but well established. Another myth is that Gamma is only defined for positive numbers; analytic continuation handles negative non-integers fine",
    }

    T["the Pythagorean theorem and its proofs"] = {
        "_cat": "math",
        "what": "the foundational result of Euclidean geometry stating that for a right triangle with legs a and b and hypotenuse c, a squared plus b squared equals c squared, equivalently that the area of the square on the hypotenuse equals the sum of the areas of the squares on the legs",
        "how": "many proofs exist. A classic dissection proof rearranges four right triangles inside a square of side a plus b two ways, leaving uncovered area equal to c squared in one arrangement and a squared plus b squared in the other. An algebraic proof uses similar triangles formed by dropping the altitude to the hypotenuse, yielding two proportions whose combination gives the result",
        "why": "the theorem underlies distance computations in Euclidean space, the law of cosines (which generalizes it), the Pythagorean identity in trigonometry (sin squared plus cos squared equals one), the metric structure of inner-product spaces, and countless engineering and physics calculations",
        "vs": "the Pythagorean theorem holds in flat Euclidean geometry but fails on curved surfaces. On a sphere or hyperbolic plane it is replaced by spherical or hyperbolic versions of the law of cosines. It also differs from the converse: a triangle with a squared plus b squared equals c squared must be right-angled, which provides a useful test",
        "ex": "a 3-4-5 right triangle has 9 plus 16 equals 25. Carpenters use this exact triple to lay out square corners with just a tape measure, marking 3 feet along one wall, 4 feet along the perpendicular, and adjusting until the diagonal is exactly 5 feet",
        "mis": "people think the theorem is just trivia. It encodes the Euclidean inner product and is equivalent to the parallel postulate in a deep sense. Another myth is that Pythagoras personally proved it; the theorem was known to Babylonians and Indians earlier, and Pythagoras's school may not have given its first proof",
    }

    T["the Fibonacci sequence"] = {
        "_cat": "math",
        "what": "the sequence 0, 1, 1, 2, 3, 5, 8, 13, 21, ..., where each term after the first two is the sum of the previous two, defined by F(0) equals 0, F(1) equals 1, and F(n) equals F(n-1) plus F(n-2)",
        "how": "Fibonacci numbers grow geometrically with ratio approaching the golden ratio phi (about 1.6180), and have a closed form (Binet's formula) involving phi and 1 minus phi. Combinatorially F(n) counts the number of ways to tile a 1-by-(n-1) strip with squares and dominoes",
        "why": "Fibonacci numbers appear in the spiral patterns of pinecones, sunflowers, and pineapples, in the convergents of continued fractions for phi, in the analysis of the Euclidean algorithm's worst case, and in classical algorithms like Fibonacci heaps. They are a standard introductory example in algorithms and dynamic programming",
        "vs": "the Fibonacci sequence differs from the Lucas numbers (same recurrence, different initial values 2 and 1), the Pell numbers (recurrence with 2x weighting), and arithmetic or geometric sequences. The closely related golden ratio is the asymptotic ratio of consecutive Fibonacci terms",
        "ex": "the seed pattern in a sunflower head displays interlocking spirals whose counts are usually consecutive Fibonacci numbers like 34 and 55 or 55 and 89. The pattern arises because new seeds form at the golden angle (about 137.5 degrees), packing efficiently",
        "mis": "people think Fibonacci invented the sequence. Indian mathematicians described it centuries earlier in the context of Sanskrit poetry meters; Fibonacci popularized it in Europe via a rabbit-population example. Another myth is that the golden ratio is everywhere in nature and art; many claimed sightings (the Parthenon, the Mona Lisa) are exaggerated or post-hoc",
    }

    T["statistical hypothesis testing"] = {
        "_cat": "math",
        "what": "a framework for deciding whether observed data are consistent with a null hypothesis or provide evidence for an alternative, using a test statistic, a null sampling distribution, and a chosen significance level",
        "how": "specify a null hypothesis H0 and an alternative H1. Compute a test statistic from the data and its distribution under H0. The p-value is the probability under H0 of a result as extreme as the one observed. If p is below the pre-chosen significance level alpha (often 0.05), reject H0; otherwise fail to reject. Power analysis precomputes the chance of detecting a true effect of given size",
        "why": "hypothesis tests structure decisions in clinical trials, A/B testing, quality control, scientific publication, and policy evaluation. They are the language regulators and editors use to evaluate claims, although their misuse is one of the most discussed issues in modern statistics",
        "vs": "hypothesis testing differs from estimation, which produces a point estimate and confidence interval rather than a yes/no decision. It differs from Bayesian inference, which updates a prior to a posterior over hypotheses rather than rejecting one. It differs from descriptive statistics, which summarize without inferring",
        "ex": "in a randomized clinical trial of a new drug versus placebo, an outcome difference is summarized by a t-test or chi-squared test. A pre-registered analysis plan and an alpha of 0.05 form the basis for an FDA decision, supplemented by effect-size and confidence-interval reporting",
        "mis": "people interpret p-values as the probability the null hypothesis is true. They are not; they are conditional on H0. Another myth is that statistical significance equals practical importance; large samples can make tiny effects highly significant, while small samples can hide real effects",
    }

    T["taylor series"] = {
        "_cat": "math",
        "what": "an infinite series representation of a sufficiently smooth function as the sum of its derivatives at a single point times powers of (x minus that point) divided by factorials, exact for analytic functions within a radius of convergence",
        "how": "compute successive derivatives of f at the expansion point a. The Taylor series is the sum from n equal 0 to infinity of (f^(n)(a) / n!) times (x minus a)^n. For a equal 0 it is called a Maclaurin series. The remainder term bounds the error of truncating the series at finite order",
        "why": "Taylor series let calculators evaluate sin, cos, exp, and log; they enable numerical methods (Newton's method uses the second-order Taylor expansion), perturbation theory in physics, error analysis, and prove identities like Euler's formula relating exp(ix) to cos x and sin x",
        "vs": "Taylor series differ from Fourier series (sums of sines and cosines, suited to periodic functions) and Laurent series (which include negative powers, useful around singularities). They differ from polynomial interpolation, which fits a polynomial through specified points rather than matching derivatives at one point",
        "ex": "the Maclaurin series for e^x is 1 plus x plus x squared over 2 plus x cubed over 6 and so on. Truncating at the fourth term gives e^1 approximately 2.708, off by about 0.5 percent from the true value 2.71828. Adding more terms converges quickly because of the factorial in the denominator",
        "mis": "people think every smooth function has a convergent Taylor series. Some smooth (C-infinity) functions are not analytic; their Taylor series converge but to a different function (the standard example is e^(-1/x^2) at zero). Another myth is that the radius of convergence is always infinite; for log(1+x) it is just 1",
    }

    T["the binomial distribution"] = {
        "_cat": "math",
        "what": "the discrete probability distribution of the number of successes in n independent Bernoulli trials, each with success probability p, with probability mass function P(X equals k) equal to (n choose k) times p^k times (1 minus p)^(n minus k)",
        "how": "imagine flipping a biased coin n times. Each sequence with k heads and n minus k tails has probability p^k (1 minus p)^(n-k), and there are (n choose k) such sequences. Sum over them to get the formula. Mean is np, variance is np(1 minus p), and for large n the distribution is well approximated by a normal with the same mean and variance",
        "why": "binomial models polling, quality-control sampling, A/B test conversions, particle counts in fixed time windows when events are independent, and any process counting yes/no outcomes. It is the simplest non-trivial discrete distribution and the building block of more complex models",
        "vs": "the binomial distribution differs from the Bernoulli (a single trial), the geometric (number of trials until first success), the negative binomial (number of trials until r successes), and the Poisson (continuous limit when n is large and p is small with np fixed)",
        "ex": "in a survey of 1000 voters where 52 percent of the population supports a candidate, the number of supporters in the sample is binomial with n equal 1000 and p equal 0.52. The mean is 520 with standard deviation about 16, so a sample within plus or minus 32 of 520 is unsurprising",
        "mis": "people think the binomial requires a fair coin. Any p between 0 and 1 works as long as trials are independent and identically distributed. Another myth is that the binomial models any count; if events are not independent or p varies, it does not apply",
    }

    # ===== TECHNOLOGY =====

    T["the TCP three-way handshake"] = {
        "_cat": "technology",
        "what": "the procedure by which two TCP endpoints establish a reliable connection by exchanging three segments (SYN, SYN-ACK, ACK) to synchronize sequence numbers and confirm both directions of communication are open",
        "how": "the client sends a SYN segment with an initial sequence number x. The server replies with SYN-ACK acknowledging x plus 1 and proposing its own initial sequence number y. The client responds with ACK acknowledging y plus 1. Both sides have now confirmed receipt and can begin reliable bidirectional data transfer with appropriate sequence numbering",
        "why": "the handshake establishes the state TCP needs for reliable, ordered byte streams: sequence numbers, window sizes, and option negotiation (MSS, SACK, window scale, timestamps). It also provides the basis for connection-tracking firewalls and SYN-cookie defenses against flooding",
        "vs": "the three-way handshake differs from UDP, which is connectionless and has no setup overhead. It differs from QUIC's handshake, which integrates with TLS to combine transport and security setup in fewer round trips. TLS itself adds further round trips on top of the TCP handshake unless 0-RTT is used",
        "ex": "loading a web page over HTTPS involves a TCP handshake (typically one round trip) followed by a TLS handshake (one or two round trips). On a 100 ms round-trip path, that's 200 to 300 ms of latency before the first byte of HTML, which is why HTTP/3 over QUIC was designed to cut it",
        "mis": "people think a successful handshake guarantees the other party is who they claim to be. TCP is just a transport protocol; authentication is the job of TLS or application protocols. Another myth is that the handshake involves data; the SYN and SYN-ACK carry no application payload by default",
    }

    T["how compilers optimize code"] = {
        "_cat": "technology",
        "what": "the set of analyses and transformations a compiler applies to source or intermediate code to produce faster, smaller, or otherwise improved machine code while preserving observable behavior",
        "how": "after parsing, the compiler builds an intermediate representation and runs passes: constant folding, dead-code elimination, common subexpression elimination, loop-invariant code motion, function inlining, register allocation via graph coloring, instruction scheduling for the target pipeline, and vectorization to use SIMD instructions. Modern compilers like LLVM run dozens of passes in a tunable pipeline",
        "why": "optimization can speed real workloads by 2 to 10 times versus unoptimized output, with no source-code changes. It is what makes high-level languages competitive with hand-written assembly and underlies the performance of every released binary on modern systems",
        "vs": "compiler optimization differs from runtime profiling and JIT optimization (which can use observed behavior, like V8 in JavaScript), from manual optimization by programmers (which uses domain knowledge), and from autotuning libraries like ATLAS, which search empirical configurations rather than reason about source code",
        "ex": "modern C++ compilers will often eliminate an entire loop that computes a value never used downstream, replace constant expressions with their result at compile time, and inline small functions across files when link-time optimization is enabled. A program that 'looks' to do work may compile to almost nothing",
        "mis": "people assume compiler optimizations always help. Aggressive optimizations can expose undefined behavior, change floating-point results, or worsen instruction cache pressure. Another myth is that handwritten assembly always beats compilers; for most code, modern optimizers outperform humans except in highly tuned hot kernels",
    }

    T["RAID levels"] = {
        "_cat": "technology",
        "what": "a family of techniques for combining multiple physical disks into a logical unit to improve redundancy, performance, or both, with standardized levels (RAID 0, 1, 5, 6, 10) defining specific layouts and trade-offs",
        "how": "RAID 0 stripes data across drives for speed but offers no redundancy. RAID 1 mirrors data identically on two drives for redundancy at the cost of capacity. RAID 5 stripes data with one drive's worth of distributed parity, surviving one drive failure. RAID 6 uses two parity blocks, surviving two failures. RAID 10 mirrors then stripes, combining redundancy and speed",
        "why": "RAID lets servers tolerate drive failures without downtime and improves throughput for bandwidth-hungry workloads. It is foundational to enterprise storage, NAS appliances, and many database deployments, although cloud object storage and replication-based filesystems are increasingly displacing it",
        "vs": "RAID differs from backup (it does not protect against deletion, ransomware, or correlated failure), from erasure coding in distributed systems (which spreads parity across more nodes for higher fault tolerance), and from filesystem-level redundancy in ZFS or Btrfs, which can offer similar guarantees with more flexibility",
        "ex": "a small business file server might run RAID 6 across eight 4 TB drives, yielding about 24 TB of usable space and surviving any two simultaneous drive failures. Rebuild times for a failed drive can take days, during which a second failure would still leave data intact",
        "mis": "people treat RAID as a backup. It is not; a fire, accidental delete, or ransomware will destroy all copies equally. Another myth is that RAID 5 is always safe enough; with multi-terabyte drives and high read-error rates, the chance of an unrecoverable error during rebuild is concerning, which is why RAID 6 has become more common",
    }
