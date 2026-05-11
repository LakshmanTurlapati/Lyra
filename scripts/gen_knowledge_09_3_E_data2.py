#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel bank part 1 of 3: science and math topics (fresh for batch E)."""


def register(T):
    # ===== SCIENCE =====

    T["plate tectonics"] = {
        "_cat": "science",
        "what": "the geological theory that Earth's lithosphere is broken into roughly a dozen rigid plates that slowly drift over the partly molten asthenosphere, with mountain ranges, earthquakes, and volcanoes concentrated at their boundaries",
        "how": "convective currents in the mantle, driven by heat from the core and radioactive decay, push and drag plates at speeds of a few centimeters per year. New crust forms at mid-ocean ridges where plates spread apart, and old crust is consumed at subduction zones where one plate dives beneath another into the mantle",
        "why": "plate tectonics explains the locations of earthquakes and volcanoes, the formation of mountain ranges, the matching coastlines of South America and Africa, the recycling of carbon between mantle and atmosphere over geologic time, and even the long-term habitability of Earth",
        "vs": "plate tectonics differs from earlier static-Earth models and from continental drift as Wegener proposed it (which lacked a mechanism). It also differs from convection without plates, as on Venus, where the lithosphere appears to overturn episodically rather than break into stable plates",
        "ex": "the Pacific Ring of Fire is a chain of subduction zones around the Pacific plate that hosts roughly 90 percent of the world's earthquakes and most of its active volcanoes, including Mount St. Helens, Mount Fuji, and the Andes volcanic arc",
        "mis": "people picture plates floating like ice on water, but they are mechanically rigid sheets coupled to a slowly creeping mantle. Another myth is that drift is too slow to matter on human timescales; it actually drives every major earthquake felt today and shifts GPS coordinates measurably year by year",
    }

    T["dark matter"] = {
        "_cat": "science",
        "what": "a hypothetical form of matter that does not emit, absorb, or reflect electromagnetic radiation but exerts gravitational influence, inferred to make up about 27 percent of the universe's mass-energy content based on its effects on galaxy rotation, lensing, and structure formation",
        "how": "astronomers detect dark matter indirectly: galaxies rotate too fast at their edges to be held together by visible mass alone, gravitational lensing around galaxy clusters bends light more strongly than luminous matter predicts, and the cosmic microwave background's fluctuation pattern requires a non-baryonic component to seed structure",
        "why": "without dark matter, current models cannot explain galaxy formation, the size of structure in the cosmic microwave background, or the dynamics of galaxy clusters. It is foundational to the standard cosmological model (Lambda-CDM)",
        "vs": "dark matter differs from dark energy, which drives accelerated cosmic expansion and acts repulsively. It differs from ordinary baryonic matter (atoms) by lacking electromagnetic interactions, and from neutrinos, which are too light and too fast-moving to clump on galactic scales",
        "ex": "the Bullet Cluster, two colliding galaxy clusters, shows hot gas (visible in X-rays) lagging behind the bulk of the gravitational mass measured by lensing. The mass and the gas are spatially offset, which is hard to explain without an invisible non-interacting matter component",
        "mis": "people think dark matter is just dim, ordinary matter astronomers cannot see. Censuses of brown dwarfs, black holes, and gas have been done, and they fall far short. Another myth is that dark matter is established as a particle; only its gravitational footprint is confirmed, not its identity",
    }

    T["the greenhouse effect"] = {
        "_cat": "science",
        "what": "the warming of a planet's surface caused by atmospheric gases that are largely transparent to incoming visible sunlight but absorb and re-emit outgoing infrared radiation, slowing the planet's heat loss to space",
        "how": "sunlight passes through the atmosphere, warms the surface, and the surface radiates heat as infrared. Greenhouse gases (water vapor, CO2, methane, nitrous oxide, ozone) have molecular vibrations resonant with infrared wavelengths, so they absorb that radiation and re-emit it in random directions, including back downward, which raises the surface temperature until balance is restored at a higher set point",
        "why": "without it, Earth's average temperature would be about minus 18 Celsius, well below freezing. Human activity has increased CO2 from roughly 280 to over 420 parts per million since the industrial revolution, intensifying the natural effect and driving observed warming, sea-level rise, and shifting climate patterns",
        "vs": "the greenhouse effect differs from heat trapping in an actual greenhouse, where most warming comes from blocking convection rather than radiative absorption. It also differs from the ozone hole, which is about UV absorption by stratospheric ozone, a separate phenomenon",
        "ex": "Venus has a runaway greenhouse atmosphere of nearly pure CO2 at 90 atmospheres of pressure. Its surface temperature is around 460 Celsius, hotter than Mercury's even though Venus is farther from the Sun, demonstrating how powerful the effect can become",
        "mis": "people confuse the greenhouse effect with the ozone hole. They are different problems with different causes (CFCs vs CO2). Another myth is that water vapor dominates so CO2 is irrelevant; water vapor is a feedback that responds to temperature, while CO2 acts as a forcing that resets the equilibrium temperature",
    }

    T["antibiotic resistance"] = {
        "_cat": "science",
        "what": "the evolutionary process by which populations of bacteria acquire the ability to survive antibiotic treatments that previously killed them, through mutation or horizontal gene transfer of resistance genes, increasingly threatening the effectiveness of frontline medicines",
        "how": "when antibiotics are used, susceptible bacteria die while any resistant variants survive and reproduce, passing resistance to descendants. Resistance genes also spread between species via plasmids, transposons, and bacteriophages. Mechanisms include enzymatic destruction of the drug, modified targets, efflux pumps, and reduced permeability",
        "why": "common infections that were routinely curable in the 20th century, like pneumonia, urinary tract infections, and gonorrhea, are becoming harder to treat. Surgical procedures, chemotherapy, and organ transplants depend on reliable antibiotics; resistance threatens the foundations of modern medicine and may cause millions of additional deaths annually",
        "vs": "resistance differs from antibiotic tolerance, where bacteria survive temporarily without genetic change. It differs from immune evasion (the bacteria's interactions with the host's defenses) and from drug failure due to underdosing or noncompliance, though those accelerate resistance evolution",
        "ex": "MRSA (methicillin-resistant Staphylococcus aureus) carries the mecA gene encoding an altered penicillin-binding protein that beta-lactam antibiotics cannot block. It causes hospital-acquired infections that resist most first-line drugs and requires alternative treatments like vancomycin or linezolid",
        "mis": "people think antibiotics fail because individual humans become resistant. The bacteria evolve, not the patient. Another myth is that completing a course always helps; for some infections, shorter, well-targeted courses now appear to reduce selection pressure better than long ones",
    }

    T["the Doppler effect"] = {
        "_cat": "science",
        "what": "the change in observed frequency of a wave when the source and observer are moving relative to each other: frequencies rise when they approach and fall when they recede, observed for sound, light, and other waves",
        "how": "as a source moves toward an observer, successive wavefronts are emitted from progressively closer positions, compressing the wavelength and raising the frequency. Receding motion stretches the wavelength. For light, the relativistic Doppler formula adds a time-dilation factor, so the redshift or blueshift depends on the relative velocity rather than a medium",
        "why": "the effect underpins radar speed guns, weather radar, medical ultrasound for blood flow, sonar, and fundamental astronomy. Hubble's discovery of cosmic redshift, and thus the expanding universe, came directly from interpreting the spectra of distant galaxies as Doppler-like recession",
        "vs": "the Doppler effect differs from gravitational redshift, where light loses energy climbing out of a gravitational well even without relative motion. It also differs from the cosmological redshift, which arises from expanding space itself rather than motion through space, though all three look similar in spectra",
        "ex": "an ambulance siren noticeably drops in pitch as it passes you because the sound waves are compressed approaching and stretched receding. In astronomy, the spiral galaxy M31 (Andromeda) is blueshifted, indicating it is approaching the Milky Way at about 110 kilometers per second",
        "mis": "people think the source's pitch actually changes; only the observed pitch changes. The driver of the ambulance hears a constant tone. Another myth is that cosmic redshift is the same as Doppler; for distant galaxies it is properly cosmological expansion, with motion through space being a small correction",
    }

    T["CRISPR gene editing"] = {
        "_cat": "science",
        "what": "a precise genome-editing technology adapted from a bacterial immune system that uses a guide RNA to direct the Cas9 nuclease to a specific DNA sequence and cut it, allowing researchers to disable, replace, or insert genes with unprecedented ease and accuracy",
        "how": "researchers design a 20-nucleotide guide RNA complementary to the target DNA. The guide binds Cas9, which scans the genome for matches adjacent to a short protospacer-adjacent motif (PAM). On binding, Cas9 introduces a double-strand break, which the cell repairs by error-prone non-homologous end joining (often disabling the gene) or by homology-directed repair using a supplied template (enabling precise edits)",
        "why": "CRISPR has accelerated genetics research, enabled new therapies for sickle cell disease and inherited blindness, advanced agricultural breeding, and revolutionized basic biology by making targeted gene knockout fast and cheap. The Nobel Prize in Chemistry 2020 went to Charpentier and Doudna for its development",
        "vs": "CRISPR differs from earlier editing tools (zinc-finger nucleases, TALENs) in being programmed by RNA rather than protein engineering, which made it dramatically faster and cheaper to retarget. It differs from RNA interference, which silences gene expression without changing DNA",
        "ex": "Casgevy, approved by the FDA in 2023, uses CRISPR to edit a regulatory region in patients' bone marrow cells so they reactivate fetal hemoglobin, effectively treating sickle cell disease and beta-thalassemia. It is the first approved CRISPR therapy",
        "mis": "people think CRISPR is perfectly precise. Off-target edits at sites with similar sequences are a real concern that researchers actively measure. Another myth is that CRISPR creates designer babies routinely; germline editing in humans remains very rare, controversial, and largely banned",
    }

    T["the second law of thermodynamics"] = {
        "_cat": "science",
        "what": "the principle that the total entropy of an isolated system can never decrease over time, equivalently that heat does not spontaneously flow from cold to hot, and that no cyclic engine can be 100 percent efficient at converting heat into work",
        "how": "entropy quantifies the number of microscopic arrangements consistent with a macroscopic state. Statistical mechanics shows that systems overwhelmingly evolve toward higher-entropy macrostates simply because there are vastly more high-entropy microstates. Heat flow, mixing, and friction all increase total entropy until equilibrium is reached",
        "why": "the second law sets the arrow of time, places fundamental limits on engine efficiency (Carnot's theorem), explains why perpetual motion machines are impossible, governs information theory through entropy bounds, and shapes the long-term fate of the universe via heat death",
        "vs": "the second law differs from the first law (energy conservation), which permits processes the second law forbids. It also differs from kinetic limits; the second law speaks of equilibrium tendencies, not rates. A diamond is unstable to graphite at room temperature but converts so slowly the second law is invisible on human timescales",
        "ex": "a hot cup of coffee left on a desk cools to room temperature. The reverse, a room-temperature cup spontaneously heating itself by drawing energy from its surroundings, has never been observed not because it is forbidden by energy conservation but because its probability is unimaginably small",
        "mis": "people invoke the second law to argue that evolution or biological order is impossible. Earth is not isolated; it absorbs low-entropy sunlight and radiates high-entropy infrared, so local order can grow while total entropy increases. Another myth is that entropy equals disorder; better framings are number of microstates or unavailable energy",
    }

    T["photosynthesis"] = {
        "_cat": "science",
        "what": "the biochemical process by which plants, algae, and certain bacteria convert light energy into chemical energy, fixing carbon dioxide and water into glucose and releasing oxygen as a byproduct",
        "how": "in the light reactions, chlorophyll in thylakoid membranes absorbs photons, exciting electrons that drive an electron transport chain to produce ATP and NADPH while splitting water molecules and releasing O2. In the Calvin cycle, the enzyme RuBisCO uses ATP and NADPH to fix atmospheric CO2 onto a five-carbon sugar, producing three-carbon sugars that are assembled into glucose",
        "why": "photosynthesis is the primary energy input for nearly all life on Earth, sets atmospheric oxygen levels, removes carbon dioxide from the air, and underlies food chains and fossil fuel formation. Without it, Earth would have neither breathable air nor a meaningful biosphere",
        "vs": "photosynthesis differs from chemosynthesis (used by hydrothermal-vent bacteria), which derives energy from inorganic chemicals like hydrogen sulfide rather than light. It differs from cellular respiration, which runs the inverse reaction, oxidizing glucose to release stored energy",
        "ex": "in a typical maple leaf, the spongy mesophyll cells host millions of chloroplasts. On a sunny summer afternoon, a single mature tree can fix several kilograms of CO2 and produce enough oxygen to support two adult humans for a day",
        "mis": "people think plants only release oxygen; they also respire, consuming oxygen at night. Another myth is that all plants use the same pathway. C4 plants like corn and CAM plants like cacti use modified versions that reduce photorespiration losses in hot or arid conditions",
    }

    T["black holes"] = {
        "_cat": "science",
        "what": "regions of spacetime where gravity is so strong that nothing, not even light, can escape from within a boundary called the event horizon. Stellar-mass black holes form when massive stars collapse, while supermassive ones sit at galactic centers",
        "how": "general relativity predicts that any mass compressed within its Schwarzschild radius warps spacetime so severely that all future-directed paths point inward. Stellar-mass black holes form when iron cores of stars over roughly 25 solar masses run out of fusion fuel and collapse past the neutron-star limit; supermassive ones grow over cosmic time through accretion and mergers",
        "why": "black holes provide some of the strongest tests of general relativity, drive the brightest persistent objects in the universe (quasars), shape galaxy evolution through feedback, and were the first sources of detected gravitational waves, opening a new observational window onto the cosmos",
        "vs": "black holes differ from neutron stars, which are denser than any stable matter but still have a surface and emit light. They differ from naked singularities, which general relativity may permit but cosmic censorship is conjectured to forbid. They differ from white holes, time-reversed solutions that are not believed to exist physically",
        "ex": "Sagittarius A*, the supermassive black hole at the Milky Way's center, is about 4 million solar masses. The Event Horizon Telescope imaged its shadow and that of M87's central black hole (about 6 billion solar masses), confirming general-relativistic predictions for the size and shape",
        "mis": "people think black holes suck in everything nearby like cosmic vacuums. From outside the event horizon, gravity behaves like that of any equivalent mass, so the Sun replaced by a one-solar-mass black hole would not change Earth's orbit. Another myth is that nothing escapes; Hawking radiation provides a slow quantum leak",
    }

    T["the periodic table"] = {
        "_cat": "science",
        "what": "the systematic arrangement of chemical elements by atomic number into rows (periods) and columns (groups), grouping elements with similar valence-electron configurations and chemical properties together",
        "how": "elements are ordered by increasing proton count. As electrons fill quantum shells in the order set by quantum mechanics, similar outer-shell configurations recur periodically. Columns share valence-electron count, hence chemistry; rows correspond to filling a new principal shell. The block structure (s, p, d, f) reflects the angular-momentum subshell being filled",
        "why": "the table organizes all known matter, predicts properties of undiscovered elements (Mendeleev famously predicted germanium and gallium before they were found), guides the design of catalysts, semiconductors, and pharmaceuticals, and provides the periodic chemical vocabulary every scientist learns",
        "vs": "Mendeleev's original table arranged by atomic weight; the modern one uses atomic number, which fixed inversions like tellurium and iodine. It differs from earlier elemental classifications (like Dobereiner's triads or Newlands' octaves), which captured fragments of periodicity but lacked the unifying principle",
        "ex": "noble gases (group 18) all have full valence shells and share extreme chemical inertness, while alkali metals (group 1) all have a single valence electron and react vigorously with water. This single periodic pattern unified observations across hundreds of compounds",
        "mis": "people think the table is finished; new superheavy elements are still being synthesized. Another myth is that group numbers from old systems (IA, IIA, etc.) are universal; modern IUPAC numbers groups 1-18 in order, which avoids the historical A/B confusion entirely",
    }

    T["enzymes"] = {
        "_cat": "science",
        "what": "biological catalysts, almost always proteins (rarely RNA), that accelerate specific biochemical reactions by enormous factors without being consumed, enabling life to operate at moderate temperatures and physiological conditions",
        "how": "an enzyme binds its substrate at an active site shaped to fit complementary geometry and chemistry. Binding stabilizes the transition state of the reaction, lowering activation energy. After catalysis, the product dissociates and the enzyme is free to bind another substrate. Cofactors, allosteric regulators, and post-translational modifications fine-tune activity",
        "why": "enzymes make essentially every metabolic pathway possible at body temperature, give cells exquisite control over which reactions proceed, and underlie diagnostics, drug targets, and industrial biocatalysis from cheese-making to laundry detergents to high-fructose corn syrup",
        "vs": "enzymes differ from inorganic catalysts in their selectivity and from antibodies in catalyzing rather than just binding. They differ from ribozymes, which are RNA-based catalysts, in being protein-based, though both share the catalytic role and the active-site logic",
        "ex": "carbonic anhydrase converts carbon dioxide and water to bicarbonate at rates approaching the diffusion limit, about a million reactions per second per enzyme molecule, which is why blood can rapidly buffer CO2 produced by metabolism without large pH swings",
        "mis": "people think enzymes change the equilibrium of reactions; they only change the rate. The thermodynamics of products versus reactants is unchanged. Another myth is that enzymes are infinitely fast; they have turnover rates and saturate at high substrate concentrations (Michaelis-Menten kinetics)",
    }

    T["nuclear fission"] = {
        "_cat": "science",
        "what": "a nuclear reaction in which a heavy nucleus splits into two or more lighter nuclei, releasing a large amount of energy along with neutrons and gamma radiation, used both in reactors and in weapons",
        "how": "when a heavy nucleus like uranium-235 absorbs a neutron, it deforms and oscillates until the strong force can no longer hold it together against electrostatic repulsion. It splits into fragments and releases two or three neutrons, which can trigger further fissions in nearby nuclei, producing a chain reaction. About 200 MeV of energy is released per fission, mostly as kinetic energy of the fragments",
        "why": "fission powers about 10 percent of global electricity through hundreds of reactors, contributed decisively to the end of World War II via the atomic bomb, and remains central to questions of energy policy, nonproliferation, and waste storage",
        "vs": "fission differs from fusion, which combines light nuclei into heavier ones. Fusion releases more energy per kilogram and produces less long-lived waste but is much harder to sustain. Fission also differs from radioactive decay, which is spontaneous and not amplified into chain reactions",
        "ex": "a typical 1-gigawatt nuclear power plant fissions about 3 kilograms of uranium-235 per day, an amount that would supply energy equivalent to roughly 10,000 tonnes of coal. The reactor maintains a controlled chain reaction with neutron-absorbing control rods and a moderator (often water) that slows neutrons",
        "mis": "people think reactors can explode like bombs; the geometry and enrichment levels make this physically impossible. Reactor accidents involve meltdowns and pressure failures, not nuclear explosions. Another myth is that nuclear waste is uniquely hazardous forever; most fission products decay within a few hundred years, and only a small fraction (actinides) demand multi-millennium storage",
    }

    T["the carbon cycle"] = {
        "_cat": "science",
        "what": "the global biogeochemical cycle by which carbon moves among the atmosphere, oceans, land biosphere, soils, sediments, and rocks. It includes a fast component (years to centuries) driven by biology and a slow component (millions of years) driven by geology",
        "how": "photosynthesis pulls CO2 from the air into living tissue; respiration and decomposition return it. The ocean exchanges CO2 with the atmosphere via dissolution and biological pumps. Over geologic time, weathering of silicate rocks consumes CO2, while volcanic outgassing replenishes it, and burial of organic carbon in sediments locks carbon away in fossil fuels and limestone",
        "why": "the carbon cycle controls atmospheric CO2 and therefore Earth's climate. Human burning of fossil fuels has injected carbon stored over hundreds of millions of years back into the fast cycle in just two centuries, overwhelming natural sinks and driving anthropogenic climate change",
        "vs": "the carbon cycle differs from the nitrogen and phosphorus cycles, which lack a major atmospheric reservoir (phosphorus) or a different chemistry (nitrogen). It differs from the water cycle in operating on much longer timescales for its slow components",
        "ex": "a single tree both absorbs CO2 during photosynthesis and respires CO2 day and night. When it eventually dies and decays or burns, most of its stored carbon returns to the atmosphere within decades. By contrast, organic matter buried in anoxic ocean sediments may stay locked away for hundreds of millions of years",
        "mis": "people think planting trees alone can solve climate change. Trees are part of the fast cycle and reach a saturation; only permanent burial of carbon (in soils, deep aquifers, or stable mineral forms) keeps it out of the atmosphere on policy-relevant timescales. Another myth is that ocean uptake is unlimited; warming reduces solubility and acidification harms biology",
    }

    T["how vaccines work"] = {
        "_cat": "science",
        "what": "vaccines are biological preparations that train the adaptive immune system to recognize a specific pathogen by exposing it to a safe form of the pathogen's antigens, so that on later real infection the body responds faster and more effectively",
        "how": "the vaccine presents antigens (whole inactivated pathogen, attenuated live pathogen, protein subunits, or mRNA encoding antigens) to the immune system. Dendritic cells display fragments to T cells; B cells produce specific antibodies; memory cells persist for years. On future exposure, this primed response neutralizes the pathogen before it can cause disease",
        "why": "vaccines have eradicated smallpox, nearly eliminated polio, and prevent millions of deaths annually from measles, tetanus, pertussis, hepatitis B, HPV, and now COVID-19. They are arguably the highest-leverage public health intervention ever devised",
        "vs": "vaccines differ from monoclonal antibodies, which provide passive immunity that fades in weeks; vaccine immunity is active and durable. They differ from natural infection, which can also induce immunity but at the cost of disease itself. mRNA vaccines differ from traditional inactivated or subunit vaccines in delivering instructions for the cell to make the antigen",
        "ex": "the MMR vaccine uses live attenuated measles, mumps, and rubella viruses. After two doses in childhood, it confers about 97 percent protection against measles, which historically killed hundreds of thousands of children annually before vaccination programs began",
        "mis": "people believe vaccines cause the diseases they prevent or commonly cause severe harm. Modern vaccines are extensively tested for safety, and serious adverse events are very rare. Another myth is that natural immunity from infection is universally superior; for many pathogens, vaccines induce equal or better immunity at much lower risk",
    }

    # ===== MATH =====

    T["Bayes' theorem"] = {
        "_cat": "math",
        "what": "a theorem of probability that relates the conditional probability of A given B to the conditional probability of B given A, written P(A|B) = P(B|A) * P(A) / P(B). It provides a principled way to update beliefs in light of new evidence",
        "how": "to apply Bayes' theorem, you start with a prior probability P(A) reflecting belief before evidence. You then compute the likelihood P(B|A) of observing the evidence under the hypothesis. You normalize by the total probability of the evidence P(B), summed over all hypotheses, and the result is the posterior P(A|B)",
        "why": "Bayes' theorem is the foundation of Bayesian statistics, modern medical test interpretation, spam filters, machine-learning classifiers, search-and-rescue planning, and rational reasoning under uncertainty. It provides the unique consistent rule for updating probabilities given new data",
        "vs": "Bayes' theorem differs from frequentist hypothesis testing, which uses p-values and significance thresholds without a prior. It differs from naive intuition about conditional probability, which often confuses P(A|B) with P(B|A), the famous prosecutor's fallacy",
        "ex": "if a disease has 1 percent prevalence and a test is 99 percent accurate, a single positive test gives only about 50 percent posterior probability of disease, because false positives among the 99 percent uninfected outweigh true positives among the 1 percent infected. Bayes' theorem makes this counterintuitive result precise",
        "mis": "people think a positive test means high probability of disease regardless of prevalence. Base rates dominate; in rare conditions, even very accurate tests give many false positives. Another myth is that Bayes' theorem requires subjective priors; objective priors and reference priors are well-developed for many situations",
    }

    T["Markov chains"] = {
        "_cat": "math",
        "what": "stochastic processes in which the probability of the next state depends only on the current state, not on the history of how that state was reached, captured by a transition matrix among finitely or countably many states",
        "how": "a Markov chain is specified by states and transition probabilities. To evolve the system, you multiply the current probability distribution by the transition matrix. Iterating leads either to a stationary distribution (irreducible aperiodic chains converge regardless of start) or to absorbing states. Eigenvalues of the transition matrix encode mixing rates",
        "why": "Markov chains underpin Google's PageRank, hidden Markov models in speech recognition, MCMC sampling for statistics, queueing theory, and population genetics. The Markov property is restrictive but extraordinarily powerful when it applies",
        "vs": "Markov chains differ from general stochastic processes by their memorylessness. They differ from deterministic dynamical systems by being probabilistic, and from Bayesian networks by their temporal sequential structure rather than directed acyclic graph of variables",
        "ex": "a simple weather model with states sunny, cloudy, rainy, each with transition probabilities to the others, is a Markov chain. Over many days, the long-run fraction of each weather type converges to the stationary distribution determined entirely by the transition matrix, regardless of today's weather",
        "mis": "people think the Markov property requires no dependence on the past; it only requires that all relevant past information is encoded in the current state. With sufficiently rich state spaces, almost any process can be made Markov. Another myth is that all chains converge; periodic or reducible chains do not",
    }

    T["calculus of variations"] = {
        "_cat": "math",
        "what": "a branch of mathematics concerned with finding functions that minimize or maximize integral functionals, typically by solving the Euler-Lagrange equation, generalizing ordinary calculus from minimizing functions of variables to minimizing functionals of entire curves or surfaces",
        "how": "for a functional J[y] = integral of L(x, y, y') dx, the function y(x) that extremizes J satisfies the Euler-Lagrange equation: d/dx(dL/dy') - dL/dy = 0. Boundary conditions and constraints (handled with Lagrange multipliers) determine which solutions are physically relevant. Second-variation analyses distinguish minima from maxima from saddle points",
        "why": "the calculus of variations is the mathematical engine behind Lagrangian and Hamiltonian mechanics, general relativity (Einstein-Hilbert action), optimal control theory, geodesics in geometry, and economic optimization over time. Many physical laws are derivable from extremizing an action",
        "vs": "calculus of variations differs from ordinary calculus by working in infinite-dimensional function spaces rather than finite-dimensional vector spaces. It differs from finite-dimensional optimization by needing functional analysis and from optimal control by traditionally lacking explicit control inputs",
        "ex": "the brachistochrone problem (Bernoulli, 1696) asks for the curve along which a bead slides under gravity from one point to another in least time. Solving the Euler-Lagrange equation gives a cycloid, an answer impossible to find by inspection",
        "mis": "people think the calculus of variations is just optimization on curves. The infinite-dimensional setting introduces subtleties (existence of minimizers requires compactness arguments) that finite-dimensional optimization avoids. Another myth is that solutions are always classical; many problems require weak solutions in Sobolev spaces",
    }

    T["the Fourier transform"] = {
        "_cat": "math",
        "what": "a mathematical operation that decomposes a function of time (or space) into its constituent frequencies, expressing the function as a sum or integral of sinusoids weighted by complex amplitudes",
        "how": "the continuous transform integrates f(t) times exp(-2 pi i f t) over all t to produce the spectrum F(f). The inverse transform reconstructs f(t) from F(f) by integrating with exp(+2 pi i f t). Discrete variants (DFT, FFT) perform the same task on sampled data, with the FFT achieving O(N log N) complexity instead of O(N^2)",
        "why": "the Fourier transform underpins audio compression, image compression, signal processing, communication systems, partial differential equations, quantum mechanics, and crystallography. It turns convolutions into multiplications and reveals frequency structure that is hidden in the time domain",
        "vs": "the Fourier transform differs from the Laplace transform, which adds an exponentially decaying weight and handles transient signals. It differs from the wavelet transform, which provides time-frequency localization at the cost of fixed frequency resolution",
        "ex": "JPEG image compression splits an image into 8x8 blocks, applies the discrete cosine transform (a real-valued cousin of the Fourier transform), and discards small high-frequency coefficients. The reconstructed image looks nearly identical at a tenth the data, exploiting the human eye's relative insensitivity to high spatial frequencies",
        "mis": "people think Fourier transforms work only on periodic signals; the continuous transform applies to any integrable function. Another myth is that the FFT is a different transform; it is a fast algorithm for the same DFT, exploiting symmetry to avoid redundant computation",
    }

    T["modular arithmetic"] = {
        "_cat": "math",
        "what": "a system of arithmetic for integers in which numbers wrap around after reaching a fixed modulus, so that 7 mod 5 equals 2 and 12 mod 5 equals 2, with addition, subtraction, and multiplication well-defined on equivalence classes modulo n",
        "how": "two integers a and b are congruent modulo n if their difference is divisible by n, written a equiv b (mod n). Equivalence classes form the ring Z/nZ. Multiplicative inverses exist exactly for elements coprime to n, which generate the multiplicative group of order phi(n) by Euler's totient",
        "why": "modular arithmetic powers cryptography (RSA, Diffie-Hellman, elliptic curves), error-correcting codes, hash functions, the calendar, and clock arithmetic. Fermat's little theorem and Euler's theorem are foundational results that make modern public-key cryptography possible",
        "vs": "modular arithmetic differs from real arithmetic in that the order is not preserved (no a < b in general) and division is restricted. It differs from finite fields when n is composite (only prime moduli give a field). It also differs from standard integer arithmetic by being finite",
        "ex": "RSA encryption picks two large primes p and q, computes n = pq, and works modulo n. Encrypting message m as c = m^e mod n and decrypting via c^d mod n relies on Euler's theorem: m^(phi(n)) equiv 1 (mod n) when m and n are coprime. Without modular arithmetic, no RSA",
        "mis": "people think modular arithmetic is just remainders. It is a complete arithmetic system with associativity, distributivity, and structure theory. Another myth is that division is impossible; division is well-defined when the divisor is coprime to the modulus, computed via the extended Euclidean algorithm",
    }

    T["the law of large numbers"] = {
        "_cat": "math",
        "what": "a theorem of probability stating that the sample mean of independent identically distributed random variables converges to the population mean as the sample size grows, in two flavors: the weak law (convergence in probability) and the strong law (almost-sure convergence)",
        "how": "for n samples X_1 through X_n with mean mu, the sample mean is bar X_n = (X_1 + ... + X_n) / n. Markov's and Chebyshev's inequalities suffice to prove the weak law. The strong law requires more: it follows from the Kolmogorov three-series theorem or from the Borel-Cantelli lemma applied to truncated variables",
        "why": "the law of large numbers justifies why averages stabilize, why insurance and casinos are profitable in aggregate, why Monte Carlo integration works, and why polling with large samples gives reliable estimates. Without it, statistical reasoning would have no foundation",
        "vs": "the law of large numbers differs from the central limit theorem, which describes the rate (Gaussian fluctuations of order 1/sqrt(n)) rather than just the limit. It differs from the gambler's fallacy, which incorrectly expects short-run reversion to the mean to compensate past streaks",
        "ex": "flipping a fair coin 10 times might give 7 heads (70 percent), but flipping it 10,000 times gives a frequency very close to 50 percent. Insurance companies rely on this: any individual claim is unpredictable, but the average claim per policy across millions of policies is highly predictable",
        "mis": "people think the law of large numbers means deviations get corrected; it actually says deviations grow slower than the sample size, so the average converges. The total number of heads minus tails can grow without bound; it is the ratio that converges. Another myth is that small samples obey the law; they often do not",
    }

    T["the central limit theorem"] = {
        "_cat": "math",
        "what": "a foundational result in probability stating that the sum (or average) of a large number of independent identically distributed random variables with finite variance, suitably normalized, converges in distribution to a standard normal regardless of the underlying distribution",
        "how": "for X_1, ..., X_n i.i.d. with mean mu and variance sigma^2, the standardized average sqrt(n) * (bar X_n - mu) / sigma converges in distribution to N(0, 1) as n grows. The proof uses characteristic functions: the characteristic function of the standardized average converges pointwise to that of N(0,1), and Levy's continuity theorem closes the argument",
        "why": "the CLT explains why so many measured quantities (heights, measurement errors, test scores) are approximately normal: they are sums of many small independent contributions. It justifies confidence intervals, t-tests, and Z-tests in classical statistics and underlies many engineering noise models",
        "vs": "the CLT differs from the law of large numbers, which says the average converges to the mean; the CLT describes the residual fluctuation distribution. It differs from the Berry-Esseen theorem, which gives quantitative rates of convergence to normality",
        "ex": "rolling 20 fair dice and summing produces a near-Gaussian distribution centered at 70 (since each die averages 3.5) with standard deviation about 7.6. The histogram is visibly bell-shaped despite each individual die being uniform on 1-6",
        "mis": "people think the CLT applies to any sum; it requires finite variance. Sums of Cauchy variables have no normal limit. Another myth is that the result is exact for finite n; convergence is asymptotic, and tail behavior may remain non-Gaussian even at moderate n",
    }
