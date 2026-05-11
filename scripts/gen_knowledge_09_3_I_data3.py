#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel bank: math topics (fresh for batch I)."""


def register(T):
    T["the pigeonhole principle"] = {
        "_cat": "math",
        "what": "the elementary combinatorial principle that if n items are placed into m containers with n > m, then at least one container must hold more than one item, with generalizations to infinite and probabilistic settings",
        "how": "the proof is by contradiction: if every container held at most one item, total items would be at most m, contradicting n > m. The generalized form says some container holds at least the ceiling of n/m items. It applies to discrete and continuous quantities through measure-theoretic versions",
        "why": "despite its triviality, the principle proves striking results: any group of 367 people contains two with the same birthday, any 5 points in a unit square have two within sqrt(2)/2 of each other, and Dirichlet used it to prove fundamental results in number theory and Diophantine approximation",
        "vs": "the pigeonhole principle differs from the Ramsey theorem, which guarantees structure in any sufficiently large object, by being purely about counts in containers. It differs from the inclusion-exclusion principle, which counts unions of overlapping sets",
        "ex": "in any group of 13 people, at least two share a birth month, since 13 > 12. More striking: any 5 points placed inside a 1x1 square force two points within distance sqrt(2)/2 of each other, by partitioning the square into four smaller squares",
        "mis": "people think the principle is too obvious to be useful. Its power is that the contradiction it forces often turns hard problems into trivial ones. Another myth is that it gives a constructive method; it only proves existence, not which container is overpopulated",
    }

    T["modular arithmetic"] = {
        "_cat": "math",
        "what": "arithmetic on integers in which numbers wrap around upon reaching a fixed modulus n, with two integers considered equivalent if their difference is divisible by n, formalized via congruence classes modulo n",
        "how": "a is congruent to b mod n if n divides a minus b. Addition, subtraction, and multiplication respect this equivalence. Division is more delicate: a has a multiplicative inverse mod n iff gcd(a, n) = 1, found via the extended Euclidean algorithm. The integers mod n form a ring, and a field if n is prime",
        "why": "modular arithmetic underpins cryptography (RSA, Diffie-Hellman, elliptic curves), error-correcting codes, hashing, calendar arithmetic, and computer arithmetic with fixed-width integers. It is the bridge from elementary number theory to algebraic structures used across math",
        "vs": "modular arithmetic differs from ordinary integer arithmetic by lacking unique factorization in general (mod 12, 4 = 2*2 = 2*8 mod 12), and by allowing zero divisors when n is composite. It differs from real-number arithmetic in being finite and discrete",
        "ex": "ISBN-10 codes use a checksum mod 11 in which the last digit (or X for 10) makes the weighted sum congruent to 0. A single-digit error or transposition changes the checksum, allowing the error to be detected",
        "mis": "people think 'a mod n' is just the remainder operation. Modular arithmetic is about equivalence classes; the remainder is one canonical representative. Negative numbers cause confusion: -1 mod 12 is 11 in this framework, not negative",
    }

    T["the binomial theorem"] = {
        "_cat": "math",
        "what": "the algebraic identity that expands (a + b)^n into a sum of terms of the form C(n, k) a^(n-k) b^k for k from 0 to n, where C(n, k) are the binomial coefficients counting subsets of size k from n",
        "how": "expansion follows by repeatedly distributing (a + b) across itself n times. Each term picks either a or b from each of the n factors, and the number of ways to choose b exactly k times is C(n, k). This connects algebraic expansion directly to combinatorial counting",
        "why": "the theorem links algebra and combinatorics, gives generating functions for many counting problems, and generalizes (via Newton's binomial series) to non-integer exponents and analytic functions. It seeds probability calculations, the central limit theorem, and the binomial distribution",
        "vs": "the binomial theorem differs from the multinomial theorem, which expands sums of more than two terms. It differs from Taylor's theorem, which approximates arbitrary smooth functions, although Newton's binomial series is a special Taylor expansion",
        "ex": "(1 + x)^5 = 1 + 5x + 10x^2 + 10x^3 + 5x^4 + x^5, with coefficients 1, 5, 10, 10, 5, 1 from the fifth row of Pascal's triangle, useful in expanding polynomials and computing approximations like (1.01)^5 by setting x = 0.01",
        "mis": "people think the theorem only works for positive integer n. Newton extended it to any real exponent as an infinite series, valid when |x| < 1. Another myth is that the coefficients are mysterious; they are exactly the counts of choosing k objects from n",
    }

    T["graph isomorphism"] = {
        "_cat": "math",
        "what": "the relation between two graphs that have a one-to-one correspondence between their vertex sets preserving adjacency, so that the graphs are structurally identical regardless of vertex labels or drawing",
        "how": "to show two graphs are isomorphic, exhibit a bijection between vertices that maps edges to edges. To show they are not, find a structural invariant that differs: vertex count, edge count, degree sequence, eigenvalues of the adjacency matrix, cycle structure. Polynomial-time algorithms exist for special classes; the general problem is famously not known to be in P or NP-complete",
        "why": "graph isomorphism appears in chemistry (matching molecular structures), computer vision, network analysis, and pattern recognition. The problem's complexity status is one of the long-standing open questions; Babai's 2015 quasipolynomial algorithm was a major theoretical advance",
        "vs": "graph isomorphism differs from graph equality (same labeled vertex and edge sets) by ignoring labels. It differs from subgraph isomorphism, which asks whether one graph appears inside another and is NP-complete",
        "ex": "two molecular graphs of glucose drawn differently in 2D both represent the same molecule; chemical informatics algorithms canonicalize structures to recognize this. Cheminformatics relies on isomorphism algorithms to deduplicate compound databases",
        "mis": "people think isomorphism is the same as having the same number of edges and vertices. Counterexamples exist: two graphs with identical degree sequences can fail to be isomorphic. Only a complete invariant (one that distinguishes all non-isomorphic graphs) settles the question, and no efficient one is known in general",
    }

    T["taylor series approximations"] = {
        "_cat": "math",
        "what": "the representation of a smooth function as an infinite sum of terms calculated from the function's derivatives at a single point, providing local polynomial approximations whose accuracy improves with more terms",
        "how": "around a point a, f(x) = sum over n of f^(n)(a)(x-a)^n/n!. Truncating gives a polynomial approximation. The remainder term bounds error; for many functions (sin, exp, log) the series converges to the function on a region whose radius is set by singularities or branch points in the complex plane",
        "why": "Taylor series let calculus tools approximate analytically inconvenient functions, derive small-angle approximations, propagate uncertainty, build numerical methods, and analyze stability. Computers use truncated series and rational approximations for trig, exponential, and special functions",
        "vs": "Taylor series differs from Fourier series, which expands periodic functions in sines and cosines. It differs from Laurent series, which permits negative powers around a singularity, and from Padé approximants, which use rational rather than polynomial form",
        "ex": "for small angles, sin(x) ~ x - x^3/6 + x^5/120. At 0.1 radians, just the first two terms give 0.0998333..., agreeing with sin(0.1) to seven decimals, illustrating fast convergence near the expansion point",
        "mis": "people think Taylor series always converge to the function. There are smooth functions (e.g., exp(-1/x^2) extended by zero) whose Taylor series at 0 is identically zero but the function is not zero; convergence to the function requires analyticity, not just smoothness",
    }

    T["Cantor's diagonal argument"] = {
        "_cat": "math",
        "what": "Cantor's 1891 proof that the real numbers are uncountable, by showing any proposed enumeration of reals can be diagonalized to construct a real not in the list, establishing that infinity comes in different sizes",
        "how": "suppose the reals between 0 and 1 are listed in a sequence with decimal expansions r1, r2, r3, .... Construct a new real whose nth decimal digit differs from the nth digit of rn. The new number cannot equal rk for any k since it differs from rk in the kth place. So no enumeration is complete; the reals are uncountable",
        "why": "the argument launched modern set theory's hierarchy of infinities, gave the first proof that not all infinite sets have the same size, and the diagonal technique now appears in computability theory (the halting problem), Russell's paradox, and Goedel's incompleteness theorems",
        "vs": "diagonalization differs from finite combinatorial proofs by working with an arbitrary infinite list. It differs from Cantor's earlier proof of uncountability via nested intervals, which is more analytic. The same diagonal idea reappears in Turing's halting argument and Goedel's coding",
        "ex": "Turing's proof that the halting problem is undecidable adapts diagonalization: assume a halting oracle exists, build a program that halts iff it does not halt, and derive a contradiction. The structure mirrors Cantor's construction of a missing real",
        "mis": "people think Cantor proved that some infinity is 'bigger' in a casual sense. The precise statement is that no bijection exists between the naturals and the reals, formalizing 'bigger' as cardinality. Another myth is that diagonalization works on the integers; applied there, the constructed integer would have infinitely many digits and is not actually an integer",
    }

    T["the Pythagorean theorem"] = {
        "_cat": "math",
        "what": "the geometric identity stating that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides, written a^2 + b^2 = c^2",
        "how": "the proof can be combinatorial (rearranging tiles in a square of side a+b), algebraic (using similar triangles created by the altitude to the hypotenuse), trigonometric (cosine of zero angle equals one in the law of cosines), or vector-based (dot product of perpendicular vectors is zero)",
        "why": "the theorem underlies distance computation, the Euclidean metric, vector norms, signal processing, statistics (orthogonality), and physics (energy decomposition). It is the gateway to inner-product spaces and generalizes to higher dimensions and to non-Euclidean geometries via curvature corrections",
        "vs": "the theorem differs from the law of cosines, which generalizes to non-right triangles by adding a -2ab cos(C) term. It differs from non-Euclidean analogs: on a sphere or hyperbolic plane the relation involves trigonometric or hyperbolic functions of side lengths",
        "ex": "GPS positioning combines distances from satellites by trilateration that ultimately rests on Pythagoras: the user's coordinates are found by intersecting spheres whose squared distances are sums of squared coordinate differences",
        "mis": "people think the theorem requires angles to be exactly 90 degrees. In practice, small deviations produce small errors; the law of cosines quantifies the correction. Another myth is that Pythagoras discovered it; the relation was known to Babylonian and Indian mathematicians long earlier",
    }

    T["dynamic programming"] = {
        "_cat": "math",
        "what": "an algorithmic technique for solving problems with optimal substructure and overlapping subproblems by storing solutions to subproblems in a table, then reusing them rather than recomputing",
        "how": "express the optimal solution as a recurrence over smaller instances, identify the subproblem state space, fill in solutions in dependency order (bottom up) or memoize recursive calls (top down), and read off the final answer. Total time is the number of states times the cost per state",
        "why": "dynamic programming converts exponential-time recursive specifications into polynomial-time algorithms, powers shortest-path computation (Bellman-Ford), sequence alignment in bioinformatics, optimal control (HJB equation), and reinforcement learning (Bellman equations)",
        "vs": "dynamic programming differs from greedy algorithms by exploring all subproblem combinations rather than committing to local choices. It differs from divide-and-conquer by reusing overlapping subproblems rather than splitting into independent pieces. It differs from backtracking by not exploring infeasible branches",
        "ex": "the longest common subsequence of two strings is computed by filling an (m+1)x(n+1) table where each cell uses values to its top, left, and top-left, in O(mn) time. Used in diff utilities and bioinformatics alignment",
        "mis": "people think dynamic programming is about programming. Bellman coined the term to sound official to sponsors; it refers to mathematical optimization over time. Another myth is that any recursive algorithm benefits from memoization; only those with overlapping subproblems do",
    }

    T["Markov chain Monte Carlo"] = {
        "_cat": "math",
        "what": "a family of algorithms that sample from complex probability distributions by constructing a Markov chain whose stationary distribution is the target, then running the chain long enough that its samples approximate draws from that distribution",
        "how": "the Metropolis-Hastings algorithm proposes moves and accepts them with a probability that ensures detailed balance with respect to the target. Gibbs sampling cycles through coordinates, each updated from its conditional distribution. Hamiltonian Monte Carlo uses gradients to make long-range moves efficiently. Convergence diagnostics include trace plots and R-hat",
        "why": "MCMC made Bayesian inference practical for high-dimensional models in genetics, epidemiology, cosmology, and machine learning. It samples posteriors when normalization constants are intractable, enabling uncertainty quantification in models that would otherwise be unfittable",
        "vs": "MCMC differs from rejection sampling and importance sampling by exploring sequentially rather than independently, allowing efficiency in high dimensions. It differs from variational inference, which approximates the posterior with a tractable family but biases the result",
        "ex": "Stan and PyMC use Hamiltonian Monte Carlo to fit hierarchical Bayesian models. A clinical trial with thousands of parameters across patients, treatments, and centers becomes tractable, with full posterior uncertainty rather than a single point estimate",
        "mis": "people think MCMC samples are independent. They are correlated draws from a chain; effective sample size is much smaller than raw count. Another myth is that long runs guarantee convergence; multimodal posteriors can trap chains in local modes for arbitrarily long",
    }

    T["the central limit theorem"] = {
        "_cat": "math",
        "what": "the foundational probability result that the sum (or average) of a large number of independent random variables with finite variance is approximately normally distributed, regardless of the original distribution's shape",
        "how": "for n iid variables with mean mu and variance sigma^2, the standardized sum (sum - n*mu)/(sigma*sqrt(n)) converges in distribution to a standard normal as n grows. The proof uses characteristic functions and a Taylor expansion. Generalizations relax independence and identical distribution under regularity conditions",
        "why": "the CLT justifies treating measurement errors as Gaussian, underpins confidence intervals and hypothesis tests for means, explains why physical noise from many small contributions tends to be normal, and is the workhorse of frequentist statistics",
        "vs": "the CLT differs from the law of large numbers, which says averages converge to the expected value but says nothing about distribution shape. It differs from Lyapunov and Lindeberg generalizations, which extend the result to non-iid settings under conditions",
        "ex": "the heights of adult humans, dice-roll averages over many trials, and noise in well-designed electronic instruments are approximately Gaussian because each is a sum of many small, roughly independent contributions, an empirical CLT in action",
        "mis": "people think the CLT says everything is normally distributed. It says averages of independent finite-variance variables tend to normal; underlying populations need not be (incomes, network degrees, and earthquake magnitudes follow heavy-tailed distributions where the CLT may fail or converge slowly)",
    }

    T["the Fibonacci sequence"] = {
        "_cat": "math",
        "what": "the integer sequence defined by F1 = F2 = 1 and Fn = Fn-1 + Fn-2 for n >= 3, producing 1, 1, 2, 3, 5, 8, 13, 21, ... and famously linked to the golden ratio and patterns in nature",
        "how": "the recurrence Fn = Fn-1 + Fn-2 has closed form Fn = (phi^n - psi^n)/sqrt(5), where phi = (1+sqrt(5))/2 and psi = (1-sqrt(5))/2. The ratio Fn+1/Fn approaches phi. Matrix exponentiation gives O(log n) computation",
        "why": "Fibonacci numbers appear in counting problems (tilings, restricted compositions), the analysis of Euclid's algorithm (worst case is consecutive Fibonacci numbers), data structure analysis (Fibonacci heaps), and biological growth patterns like phyllotaxis",
        "vs": "the Fibonacci sequence differs from arithmetic and geometric sequences by mixing additive structure (linear recurrence) with exponential growth (asymptotic phi^n). It differs from the Lucas numbers, which obey the same recurrence but start 2, 1",
        "ex": "sunflower seed heads pack seeds along spirals whose counts are typically consecutive Fibonacci numbers (34 and 55, or 55 and 89). The packing optimizes seed density given a constant divergence angle of about 137.5 degrees, the golden angle",
        "mis": "people think Fibonacci numbers appear everywhere in nature in a mystical way. They appear where additive growth and packing constraints favor them; in many other systems they do not. Another myth is that Fibonacci invented the sequence; he popularized it in Europe in 1202, but it appears in earlier Indian mathematics",
    }

    T["set theory and the axiom of choice"] = {
        "_cat": "math",
        "what": "the axiom of Zermelo-Fraenkel set theory stating that for any collection of non-empty sets there exists a function selecting one element from each, an apparently obvious principle with deeply non-constructive consequences",
        "how": "the axiom posits the existence of choice functions without constructing them. Equivalents include Zorn's lemma (every partially ordered set with chain bounds has a maximal element) and the well-ordering theorem (every set has a well-ordering). Many proofs use these equivalents in algebra, topology, and analysis",
        "why": "AC is required for mainstream theorems including the existence of bases for any vector space, Tychonoff's theorem (product of compact spaces is compact), Hahn-Banach in functional analysis, and the existence of non-measurable sets. Without AC, mathematics looks much different",
        "vs": "AC differs from the other ZF axioms by being non-constructive: it asserts existence without giving a method. It differs from the weaker dependent choice (DC) and countable choice (ACω), which suffice for most analysis but not for the wildest pathologies",
        "ex": "the Banach-Tarski paradox uses AC to decompose a unit ball into finitely many pieces and reassemble them into two unit balls. The pieces are non-measurable, so the paradox does not contradict everyday geometry; it shows AC permits sets without volume",
        "mis": "people think AC is uncontroversial because the choice 'feels obvious'. For finite or countable collections of sets the principle is provable in ZF. For uncountable collections the axiom is independent of ZF and yields strange consequences. Some constructive mathematicians reject it",
    }

    T["the Euclidean algorithm"] = {
        "_cat": "math",
        "what": "the ancient procedure for computing the greatest common divisor of two integers by repeatedly replacing the larger with the remainder of dividing it by the smaller, terminating when the remainder is zero",
        "how": "given a >= b > 0, compute a mod b = r; replace (a, b) with (b, r); repeat until r = 0. The last nonzero divisor is gcd(a, b). The extended version also returns integers x, y with ax + by = gcd(a, b), useful for modular inverses",
        "why": "the algorithm is the prototype of an efficient mathematical procedure (worst-case running time tied to Fibonacci numbers) and is the basis for fraction reduction, modular inversion, and operations in cryptographic systems including RSA key generation",
        "vs": "the Euclidean algorithm differs from prime factorization-based gcd computation, which is exponentially slower for large numbers. It differs from binary gcd, which avoids division, and from polynomial gcd, which generalizes the same idea to polynomial rings",
        "ex": "gcd(252, 105): 252 mod 105 = 42; 105 mod 42 = 21; 42 mod 21 = 0. So gcd is 21. The extended algorithm finds 21 = 252*(-2) + 105*5, which can be used to compute modular inverses or solve linear Diophantine equations",
        "mis": "people think the algorithm is slow because it uses repeated division. The number of steps is at most about 5 times the number of digits of the smaller input, so it is among the fastest known number-theoretic procedures. The Fibonacci numbers achieve the worst case",
    }

    T["topology and continuous deformation"] = {
        "_cat": "math",
        "what": "the branch of mathematics concerned with properties preserved under continuous deformations such as stretching and bending but not tearing or gluing, formalized via open sets and continuous maps",
        "how": "a topology on a set is a collection of open sets closed under arbitrary unions and finite intersections. Continuous maps are those whose preimages of open sets are open. Topological invariants like connectedness, compactness, and homology classify spaces up to homeomorphism",
        "why": "topology underpins modern analysis, geometry, and physics. It distinguishes shapes by global features rather than local distance, supports algebraic invariants used in data analysis (persistent homology), classifies phases of matter (topological insulators), and shapes general relativity's spacetime",
        "vs": "topology differs from geometry by ignoring distance and angle, retaining only continuity. It differs from analysis by abstracting away epsilon-delta arguments into open-set language. Differential topology adds smoothness; algebraic topology assigns groups to spaces",
        "ex": "a coffee mug and a doughnut are topologically equivalent: each is a solid with one hole. They differ from a sphere, which has none. This 'hole-counting' is captured precisely by the Euler characteristic and homology groups",
        "mis": "people think topology is just rubber-sheet geometry. That visual picks up some intuition but misses the formal structure: open sets, continuous maps, and the resulting categorical and homological apparatus that drive serious theorems",
    }
