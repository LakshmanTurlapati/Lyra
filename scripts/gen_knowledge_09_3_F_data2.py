#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel bank part 1 of 4: science topics (fresh for batch F)."""


def register(T):
    T["the carbon-nitrogen-oxygen cycle in stars"] = {
        "_cat": "science",
        "what": "a sequence of nuclear reactions in which carbon, nitrogen, and oxygen act as catalysts to fuse four hydrogen nuclei into a helium nucleus, dominant in stars heavier than about 1.3 solar masses where core temperatures exceed roughly 17 million Kelvin",
        "how": "a carbon-12 nucleus captures a proton to form nitrogen-13, which beta-decays to carbon-13. Three more proton captures and another beta decay yield nitrogen-15, which captures a final proton and splits into a helium-4 nucleus and a regenerated carbon-12. Net result: four protons fused into helium plus two positrons, two neutrinos, and gamma rays",
        "why": "the CNO cycle explains why massive stars burn faster and hotter than the Sun, sets the brightness ceiling for high-mass main-sequence stars, and produces the cosmic abundance pattern of nitrogen-14. Its energy output scales steeply with temperature (roughly T to the 17th power), making it dominant above a sharp threshold",
        "vs": "the CNO cycle differs from the proton-proton chain, which fuses hydrogen into helium without catalysts and dominates in cooler stars like the Sun. It differs from helium burning (the triple-alpha process), which fuses helium into carbon at much higher temperatures in evolved giants",
        "ex": "Sirius A, about 2.1 solar masses with a core temperature near 22 million Kelvin, generates most of its luminosity through the CNO cycle. Borexino's 2020 detection of solar CNO neutrinos confirmed the cycle contributes about 1 percent of the Sun's energy, consistent with the standard solar model",
        "mis": "people think the cycle consumes carbon. The carbon is a catalyst and is regenerated each loop. Another myth is that all stars use it; below about 1.3 solar masses the proton-proton chain dominates because the temperature dependence of CNO cuts it off",
    }

    T["superconductivity"] = {
        "_cat": "science",
        "what": "a quantum state of matter in which certain materials, cooled below a critical temperature, conduct electricity with exactly zero resistance and expel magnetic fields from their interior (the Meissner effect)",
        "how": "in conventional superconductors, electrons pair up into Cooper pairs through a weak attraction mediated by lattice vibrations (phonons). These pairs share a single coherent quantum wavefunction and cannot scatter off impurities the way single electrons do, so current flows without dissipation. High-temperature cuprate superconductors involve a different and still-debated pairing mechanism",
        "why": "superconductors enable powerful MRI magnets, particle accelerator dipoles, magnetic levitation trains, sensitive SQUID magnetometers, and fault-current limiters. A practical room-temperature superconductor would transform the electric grid, motors, and computing",
        "vs": "superconductivity differs from ordinary conduction (low but nonzero resistance), from perfect conductors as classically imagined (which would not actively expel fields), and from superfluidity (frictionless flow of neutral atoms like helium-4, governed by similar quantum coherence but without electric current)",
        "ex": "MRI scanners use niobium-titanium superconducting wire cooled to about 4 Kelvin in liquid helium to generate steady magnetic fields of 1.5 to 3 Tesla. The persistent current in the magnet decays so slowly that the field is stable for years without external power",
        "mis": "people assume any cold metal becomes superconducting. Most do not at any temperature. Another myth is that high-temperature superconductors work at room temperature; the cuprates need cooling below about 138 Kelvin, still well below freezing",
    }

    T["the Krebs (citric acid) cycle in detail"] = {
        "_cat": "science",
        "what": "the central metabolic pathway in mitochondria that fully oxidizes acetyl-CoA derived from carbohydrates, fats, and proteins, producing carbon dioxide, NADH, FADH2, and a small amount of GTP/ATP, feeding electrons into the respiratory chain",
        "how": "acetyl-CoA condenses with oxaloacetate to form citrate. Through eight enzymatic steps, citrate is rearranged and oxidized, releasing two CO2 molecules, generating three NADH, one FADH2, and one GTP per turn, and regenerating oxaloacetate to start again. Key control points are citrate synthase, isocitrate dehydrogenase, and alpha-ketoglutarate dehydrogenase",
        "why": "the cycle supplies the bulk of reduced electron carriers that the electron transport chain uses to make ATP. Its intermediates also feed biosynthesis of amino acids, heme, and neurotransmitters, making it the metabolic crossroads between catabolism and anabolism",
        "vs": "the Krebs cycle differs from glycolysis, which occurs in the cytoplasm and breaks glucose into pyruvate without oxygen. It differs from the electron transport chain, which it feeds, and from the pentose phosphate pathway, which makes NADPH and ribose-5-phosphate for biosynthesis",
        "ex": "during sustained exercise, muscle mitochondria run the Krebs cycle continuously, oxidizing acetyl-CoA from glucose and fatty acids to keep ATP production matched to demand. A trained endurance athlete can sustain near-maximal cycle flux for hours",
        "mis": "people think the cycle creates ATP directly in large amounts. It only makes one GTP/ATP per turn; the bulk of cellular ATP comes from the electron transport chain that uses NADH and FADH2 made by the cycle",
    }

    T["prions"] = {
        "_cat": "science",
        "what": "infectious proteins that cause disease by inducing normal cellular proteins to misfold into the same pathological shape, propagating without DNA or RNA and resisting standard sterilization",
        "how": "the normal prion protein PrPC, a glycoprotein on neuron surfaces, can misfold into a beta-sheet-rich form called PrPSc. PrPSc acts as a template that converts PrPC into more PrPSc, forming aggregates and amyloid plaques that damage brain tissue. Transmission can be sporadic, inherited, or acquired through contaminated tissue",
        "why": "prions cause uniformly fatal neurodegenerative diseases like Creutzfeldt-Jakob disease, kuru, and BSE (mad cow disease). They forced reform of blood, organ, and surgical-instrument handling and reshaped how scientists think about heredity and infection, since they violate the central dogma of molecular biology",
        "vs": "prions differ from viruses, bacteria, and viroids in carrying no nucleic acid. They differ from amyloid plaques in Alzheimer's disease, which are not transmissible between individuals under natural conditions, although the templating mechanism shows striking similarities",
        "ex": "the BSE epidemic in British cattle in the 1980s and 1990s was traced to feeding cattle meat-and-bone meal that contained prion-contaminated tissue. Variant CJD in humans, linked to consuming infected beef, killed about 230 people and prompted sweeping changes in feed regulation",
        "mis": "people assume prion diseases must be rare and exotic. Sporadic CJD occurs in about one person per million annually worldwide. Another myth is that cooking destroys prions; standard cooking and even autoclaving at 121 Celsius leave them largely intact",
    }

    T["catalysts in chemistry"] = {
        "_cat": "science",
        "what": "substances that increase the rate of a chemical reaction by lowering its activation energy without being consumed, allowing reactions to proceed faster, at lower temperatures, or with greater selectivity",
        "how": "catalysts provide an alternative reaction pathway with a lower energy barrier, often by binding reactants in geometries that favor bond rearrangement, stabilizing transition states, or shuttling protons or electrons. They are regenerated at the end of each catalytic cycle, so a small amount can convert a large amount of substrate",
        "why": "catalysts make the Haber-Bosch process for ammonia, catalytic converters for cleaner exhaust, polymerization for plastics, and almost all biological metabolism (via enzymes) practical. Without catalysts, modern industry and life itself would not function",
        "vs": "catalysts differ from reactants (consumed) and products (formed). They differ from inhibitors, which slow reactions. Heterogeneous catalysts (a different phase from reactants, like solid platinum acting on gas) differ from homogeneous catalysts (same phase) and from biocatalysts (enzymes), each with distinct strengths",
        "ex": "the catalytic converter in a gasoline car uses platinum, palladium, and rhodium on a ceramic honeycomb to convert CO and unburned hydrocarbons to CO2 and water and reduce NOx to N2, all within the few seconds exhaust passes through. This single device cuts pollutant emissions by over 90 percent",
        "mis": "people think catalysts make reactions thermodynamically possible. Catalysts only speed reactions that are already favorable; they cannot make impossible reactions happen. Another myth is that catalysts are inert spectators; they actually participate intimately in each cycle and can be poisoned by impurities",
    }

    T["the human auditory system"] = {
        "_cat": "science",
        "what": "the biological apparatus that converts air pressure waves into neural signals interpreted as sound, comprising the outer ear, middle ear, inner ear (cochlea), and auditory pathways through the brainstem to the auditory cortex",
        "how": "sound waves enter the ear canal and vibrate the eardrum. Three middle-ear bones (malleus, incus, stapes) amplify and transmit vibrations to the oval window of the cochlea. Inside the fluid-filled cochlea, the basilar membrane vibrates at locations that depend on frequency. Hair cells along the membrane convert mechanical motion into electrical signals via mechanotransduction channels, sending impulses through the auditory nerve to the brain",
        "why": "hearing enables speech, music, hazard detection, and spatial awareness. Understanding the system underlies cochlear implants, hearing aids, and treatments for tinnitus and noise-induced hearing loss, which affects roughly 1.5 billion people worldwide",
        "vs": "the auditory system differs from the vestibular system (balance and head motion), which shares the inner ear's hair cells but not the cochlea. It differs from echolocation in bats and dolphins, which uses similar peripheral hardware but specialized timing circuits",
        "ex": "a cochlear implant bypasses damaged hair cells by stimulating the auditory nerve directly with an electrode array threaded into the cochlea. Patients with profound deafness often regain functional speech understanding within a year, especially when implanted young",
        "mis": "people think loud sound damages hearing only at the eardrum. The actual injury is to delicate hair cells in the cochlea, which do not regenerate in humans. Another myth is that hearing recovers with rest after loud exposure; permanent threshold shifts accumulate over a lifetime",
    }

    T["isotopes and radioactive decay"] = {
        "_cat": "science",
        "what": "isotopes are atoms of the same element with different numbers of neutrons; some are stable while others (radioisotopes) spontaneously transform into other nuclei by emitting alpha particles, beta particles, or gamma rays",
        "how": "decay happens when a nucleus is energetically unstable. Alpha decay ejects a helium-4 nucleus, reducing mass number by 4 and atomic number by 2. Beta-minus decay converts a neutron to a proton, emitting an electron and antineutrino. Gamma decay shed excess energy as photons. The probability per unit time is constant, giving each isotope a characteristic half-life",
        "why": "radioactive decay drives radiocarbon dating, medical imaging (PET, SPECT), cancer therapy, smoke detectors (americium-241), nuclear power, and geological dating that anchors Earth's 4.54-billion-year age. It also tracks contamination from nuclear accidents",
        "vs": "isotopes differ from elements (which are defined by proton count) and from ions (which differ in electron count). Radioactive decay differs from chemical reactions (which only involve electrons), and from nuclear fission (induced splitting of heavy nuclei) and fusion (combining of light nuclei)",
        "ex": "carbon-14, produced in the upper atmosphere, has a half-life of 5,730 years. Living organisms exchange carbon with the atmosphere, so the carbon-14 fraction stays constant during life. After death, exchange stops and carbon-14 decays, allowing dating of artifacts up to about 50,000 years old",
        "mis": "people think every atom of a radioisotope decays at the same time. Decay is probabilistic; over one half-life, on average half decay, but any individual atom might survive much longer or shorter. Another myth is that radiation makes objects radioactive; only neutron capture or contamination, not gamma exposure, induces radioactivity",
    }

    T["entropy and information theory"] = {
        "_cat": "science",
        "what": "Shannon entropy, a measure of uncertainty in a probability distribution, quantifying the average amount of information produced by a random source in bits per symbol and setting fundamental limits on compression and channel capacity",
        "how": "for a discrete random variable with outcomes of probability p_i, entropy H equals minus the sum of p_i times log2 p_i. Maximum entropy occurs at a uniform distribution; entropy is zero when one outcome is certain. Joint, conditional, and mutual information build from this foundation, and Shannon's source coding theorem proves H is the minimum average bits needed to encode messages",
        "why": "entropy underlies data compression (gzip, JPEG, video codecs), error-correcting codes that make modern communication reliable, cryptography (entropy of keys), and machine-learning objectives like cross-entropy loss. It connects to thermodynamic entropy through statistical mechanics",
        "vs": "Shannon entropy differs from thermodynamic entropy in units and motivation but is mathematically the same quantity for a system's microstate distribution. It differs from variance (which measures spread of numerical values rather than uncertainty) and from algorithmic (Kolmogorov) complexity, which describes individual strings rather than distributions",
        "ex": "an English letter carries roughly 1 bit of entropy on average given preceding context, even though there are 26 letters (about 4.7 bits at maximum). That is why English text compresses by about 75 percent with general-purpose compressors like gzip",
        "mis": "people equate high entropy with disorder. In information theory, high entropy means high uncertainty or unpredictability, not chaos. Another myth is that entropy can be negative; differential entropy of continuous variables can, but discrete Shannon entropy cannot",
    }

    T["the El Nino-Southern Oscillation"] = {
        "_cat": "science",
        "what": "a coupled ocean-atmosphere climate pattern in the tropical Pacific that oscillates irregularly between warm El Nino phases, cool La Nina phases, and neutral conditions roughly every two to seven years, redistributing heat and rainfall worldwide",
        "how": "in normal conditions, trade winds push warm surface water westward, allowing cold deep water to upwell off South America. During El Nino, trade winds weaken, warm water sloshes east, suppressing upwelling and shifting tropical rainfall toward the central and eastern Pacific. La Nina is an exaggeration of normal conditions with stronger trades and a steeper west-east temperature gradient. The Walker circulation in the atmosphere reinforces and is reinforced by the ocean changes",
        "why": "ENSO is the largest source of year-to-year climate variability outside the polar regions. It influences droughts in Australia and Indonesia, floods in Peru, monsoon failures in India, hurricane activity in the Atlantic, and global temperature spikes. Forecasting it months ahead saves billions in agriculture, water management, and disaster preparation",
        "vs": "ENSO differs from the seasonal cycle (annual and forced by sun angle), from the North Atlantic Oscillation (a pressure see-saw between Iceland and the Azores), and from anthropogenic climate change (a long-term trend rather than oscillation). The Pacific Decadal Oscillation is a related but slower Pacific pattern",
        "ex": "the 1997-98 El Nino was one of the strongest on record. It contributed to drought-driven wildfires in Indonesia, severe flooding in California and Peru, and a temporary global temperature record. Damage estimates reached 35 to 45 billion US dollars worldwide",
        "mis": "people think El Nino always means warm and wet everywhere. Effects are regional and depend on the phase: California gets wetter, Indonesia drier; the global average temperature rises modestly while local extremes go in opposite directions. Another myth is that ENSO causes climate change; it is internal variability layered on top of long-term warming",
    }

    T["the lymphatic system"] = {
        "_cat": "science",
        "what": "a network of vessels, lymph nodes, and lymphoid organs that drains interstitial fluid back to the bloodstream, transports dietary fats from the intestines, and houses much of the immune system's surveillance and response machinery",
        "how": "blood capillaries leak fluid into tissues; about three liters of this fluid per day enters tiny lymphatic capillaries, becomes lymph, and is pumped via skeletal-muscle contraction and one-way valves through progressively larger vessels. Lymph nodes filter the fluid, where dendritic cells present antigens to lymphocytes that mount adaptive immune responses. Lymph eventually drains into the subclavian veins via the thoracic duct and right lymphatic duct",
        "why": "without lymphatic drainage, tissues swell and immune surveillance fails. The system is central to fighting infection, absorbing fats, and propagating cancer (metastasis often follows lymphatic routes). Lymphedema, lymphomas, and lymph-node biopsies in oncology all hinge on understanding it",
        "vs": "the lymphatic system differs from the cardiovascular system in lacking a central pump and carrying lymph rather than blood. It differs from the venous system, although both return fluid to the heart, and it differs functionally from the immune system as a whole, of which it is the structural backbone",
        "ex": "in breast cancer staging, surgeons biopsy the sentinel lymph node draining the tumor. If the node is clear, distant lymphatic spread is unlikely; if not, more nodes are removed and adjuvant therapy is intensified. This procedure, developed in the 1990s, replaced routine extensive node dissection",
        "mis": "people picture lymph as a sluggish backwater. It moves at significant volume daily and is essential for fluid balance. Another myth is that lymphatic massage detoxifies the body; while it may help certain edemas, the kidneys and liver, not lymphatic flow, do the actual detoxification",
    }

    T["mass extinctions in Earth's history"] = {
        "_cat": "science",
        "what": "episodes in the geological record where a large fraction of species (often more than 50 percent of marine genera) died out in a relatively short interval, with five major events recognized in the Phanerozoic and a possible sixth driven by humans currently underway",
        "how": "different extinctions had different drivers but shared rapid environmental change outpacing species' adaptive capacity. End-Permian extinction (252 million years ago) was driven by Siberian Traps volcanism causing ocean anoxia and warming; end-Cretaceous (66 million years ago) by the Chicxulub asteroid impact and Deccan Traps volcanism; the present biodiversity crisis by habitat loss, climate change, invasive species, and overexploitation",
        "why": "mass extinctions reset evolutionary trajectories, opening niches for surviving lineages to radiate. The end-Cretaceous event ended non-avian dinosaurs and gave mammals the ecological space to diversify. Studying past extinctions calibrates how quickly biodiversity recovers (millions of years) and what makes ecosystems vulnerable",
        "vs": "mass extinctions differ from background extinction (the steady, low-rate turnover of species) and from local extirpation (loss in one region without species loss). They differ in drivers: impacts, volcanism, sea-level change, and now human activity each leave distinct signatures in the rock record",
        "ex": "the end-Permian extinction killed about 81 percent of marine species and 70 percent of terrestrial vertebrate genera. Recovery took 5 to 10 million years and reshuffled global ecosystems, with the Mesozoic 'Age of Reptiles' rising from the survivors",
        "mis": "people think dinosaurs gradually faded out. The fossil record shows a sharp boundary at the Chicxulub event 66 million years ago. Another myth is that mass extinctions only affect headline taxa; the largest losses are usually among small marine invertebrates that dominate ocean food webs",
    }
