#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel bank: science topics (fresh for batch H)."""


def register(T):
    T["how vaccines achieve herd immunity"] = {
        "_cat": "science",
        "what": "the indirect protection from infectious disease that occurs when a sufficient fraction of a population is immune (through vaccination or prior infection) so that chains of transmission cannot sustain themselves",
        "how": "each pathogen has a basic reproduction number R0, the average number of secondary infections from a single case in a fully susceptible population. If a fraction p of the population is immune, the effective reproduction number drops to R0(1-p). When R0(1-p) is below 1, outbreaks fizzle. The herd immunity threshold is therefore 1 - 1/R0",
        "why": "herd immunity protects people who cannot be vaccinated (infants, the immunocompromised, those with allergies) and is the public-health justification for childhood vaccination programs. It also explains why falling vaccination rates can suddenly trigger outbreaks of diseases that seemed eliminated",
        "vs": "herd immunity differs from individual immunity, which protects only the vaccinated person, and from sterilizing immunity, which prevents any transmission. It differs from natural population immunity acquired by uncontrolled spread, which extracts a heavy mortality cost",
        "ex": "measles, with R0 around 15, requires roughly 95 percent immunity for herd protection. When uptake drops below that, outbreaks return, as seen in the 2019 US measles resurgence in undervaccinated communities",
        "mis": "people think herd immunity always means complete eradication. It only suppresses sustained transmission; imported cases can still cause local clusters. Another myth is that herd immunity can be reached by deliberate infection at low cost, ignoring the deaths and long-term complications that path produces",
    }

    T["the human gut-brain axis"] = {
        "_cat": "science",
        "what": "the bidirectional biochemical signaling network connecting the gastrointestinal tract and the central nervous system, mediated by neural, endocrine, immune, and microbial pathways",
        "how": "the vagus nerve carries sensory information from gut to brainstem. Enteroendocrine cells secrete hormones like CCK, GLP-1, and ghrelin that influence appetite and mood. Gut microbes produce short-chain fatty acids, neurotransmitter precursors, and immunomodulators that cross or signal across the blood-brain barrier. The brain in turn modulates motility, secretion, and immune function via autonomic outflow",
        "why": "the axis underlies appetite regulation, stress responses, and the high comorbidity between gastrointestinal disorders (IBS, IBD) and psychiatric conditions (anxiety, depression). It is the target of GLP-1 weight-loss drugs, certain probiotic interventions, and emerging therapies for mood disorders",
        "vs": "the gut-brain axis differs from the older 'enteric nervous system' concept, which is just the gut's local neural network, by encompassing systemic endocrine and microbial signaling. It differs from the hypothalamic-pituitary-adrenal axis, which carries stress signals primarily through cortisol",
        "ex": "GLP-1 agonists like semaglutide were originally developed for diabetes but produced striking weight loss because GLP-1 receptors in the brain reduce hunger and slow gastric emptying. The drug works by hijacking a normal gut-brain signaling pathway",
        "mis": "people think 'gut feelings' literally come from the gut to the brain in a moment-to-moment way. The signaling is slower and more diffuse than that. Another myth is that any probiotic improves mood; evidence is strong only for specific strains in specific conditions",
    }

    T["how mitochondria produce ATP"] = {
        "_cat": "science",
        "what": "the chemiosmotic process by which mitochondria couple the oxidation of nutrients to the synthesis of adenosine triphosphate, the universal cellular energy currency, generating roughly 30 ATP per glucose molecule under aerobic conditions",
        "how": "the citric acid cycle in the matrix oxidizes acetyl-CoA, generating NADH and FADH2. These electron carriers donate electrons to the electron transport chain in the inner membrane. Complexes I, III, and IV pump protons into the intermembrane space, building an electrochemical gradient. Protons flow back through ATP synthase, whose rotation drives the phosphorylation of ADP to ATP. Oxygen accepts the final electrons, forming water",
        "why": "ATP synthesis powers nearly every cellular process: muscle contraction, neural firing, biosynthesis, ion pumping. Mitochondrial dysfunction underlies aging, neurodegeneration, metabolic disease, and many inherited disorders, and is a major target for drug discovery",
        "vs": "oxidative phosphorylation differs from substrate-level phosphorylation (in glycolysis and the citric acid cycle), which makes ATP directly without an electron chain. It differs from photosynthesis's light-driven proton gradient in using nutrient oxidation as the energy source rather than photons",
        "ex": "cyanide poisoning kills by binding cytochrome c oxidase (complex IV), halting electron flow. Cells can no longer pump protons, ATP synthesis collapses, and tissues that depend on continuous high ATP turnover (heart, brain) fail within minutes",
        "mis": "people think mitochondria 'burn' fuel like a flame. The oxidation is stepwise and electron-by-electron, which is what allows energy capture; a flame would dissipate the energy as heat. Another myth is that all ATP comes from mitochondria; glycolysis in the cytoplasm makes a small amount even without oxygen",
    }

    T["ocean acidification"] = {
        "_cat": "science",
        "what": "the ongoing decrease in the pH of Earth's oceans caused by absorption of atmospheric carbon dioxide, which reacts with seawater to form carbonic acid and lower pH by about 0.1 units since preindustrial times",
        "how": "CO2 dissolves into seawater and reacts with water to form carbonic acid, which dissociates into bicarbonate and hydrogen ions. The added hydrogen ions lower pH. They also bind carbonate ions, reducing the carbonate available for organisms that build calcium carbonate shells and skeletons. The shift propagates through ocean chemistry on timescales of decades to millennia",
        "why": "acidification threatens coral reefs, shellfish, pteropods, and other calcifying organisms at the base of marine food webs. Combined with warming and deoxygenation, it endangers fisheries that feed billions, coastal economies, and ocean carbon storage capacity. It is one of the clearest fingerprints of anthropogenic CO2 in the natural world",
        "vs": "ocean acidification differs from acid rain, which is local and driven by sulfate and nitrate emissions, and from sea level rise, which involves thermal expansion and ice melt. It differs from eutrophication, which is nutrient pollution causing dead zones",
        "ex": "Pacific oyster larvae in the US Pacific Northwest began dying in hatcheries in the late 2000s when upwelling brought corrosive water inshore. Hatcheries now monitor pH and buffer intake water to keep larvae alive, an industry adaptation forced by acidification",
        "mis": "people think the ocean is becoming literally acidic (pH below 7). It is becoming less basic, moving from about pH 8.2 toward 8.0. The biological harm comes from carbonate chemistry shifts, not the absolute pH value",
    }

    T["how RNA splicing works"] = {
        "_cat": "science",
        "what": "the process by which non-coding regions (introns) are removed from precursor messenger RNA and the remaining coding regions (exons) are joined together to produce mature mRNA ready for translation",
        "how": "the spliceosome, a large complex of small nuclear RNAs and proteins, recognizes splice sites at intron boundaries. It cuts at the 5' splice site, forms a lariat intermediate by joining the cut end to a branch-point adenosine, then cuts at the 3' splice site and ligates the flanking exons. Alternative splicing chooses which exons are included, generating multiple proteins from one gene",
        "why": "splicing is required for nearly all eukaryotic gene expression and dramatically expands proteome diversity: humans encode about 20000 genes but produce well over 100000 protein isoforms. Splicing errors cause diseases including spinal muscular atrophy, retinitis pigmentosa, and many cancers, and splice-modulating drugs like nusinersen treat them",
        "vs": "splicing differs from RNA editing, which changes individual bases, and from RNA degradation, which destroys the molecule. It differs from prokaryotic gene expression, which generally lacks introns and proceeds directly from transcription to translation",
        "ex": "the SMN2 gene in spinal muscular atrophy normally produces a truncated protein because of an alternative splicing pattern. Nusinersen is an antisense oligonucleotide that binds the pre-mRNA and shifts splicing to include the missing exon, restoring functional protein",
        "mis": "people think genes encode proteins one-to-one. Most human genes produce multiple isoforms via alternative splicing. Another myth is that introns are useless; they harbor regulatory sequences and contribute to gene evolution by enabling exon shuffling",
    }

    T["the doppler shift in astronomy"] = {
        "_cat": "science",
        "what": "the change in observed frequency of light from a moving source, used in astronomy to measure radial velocities of stars, galaxies, and exoplanet host stars by tracking spectral line shifts toward red (receding) or blue (approaching)",
        "how": "atoms emit or absorb light at characteristic wavelengths. When the source moves relative to the observer, those wavelengths shift in proportion to the radial velocity over the speed of light. Spectrographs disperse starlight into spectra, line positions are measured against laboratory references, and the wavelength shift gives velocity. For exoplanets, periodic shifts of a host star's spectrum reveal an orbiting body",
        "why": "the Doppler shift gave the first evidence of cosmic expansion (Hubble's law), measures rotation curves of galaxies (key to dark matter), discovers exoplanets (the radial velocity method), and constrains binary star orbits and stellar atmospheres",
        "vs": "Doppler shift differs from cosmological redshift in that the latter results from the expansion of space itself rather than motion through space, although both produce wavelength stretching. It differs from gravitational redshift, which arises from light climbing out of a potential well",
        "ex": "the discovery of 51 Pegasi b in 1995 used periodic Doppler shifts in a Sun-like star's spectrum to infer a Jupiter-mass planet in a four-day orbit, the first exoplanet around a normal star and the start of a new field",
        "mis": "people think Doppler shift directly measures distance to galaxies. It measures velocity; distance is inferred via the Hubble relation, with significant scatter at low redshift due to peculiar motions. Another myth is that any redshift implies cosmic recession; nearby stars and binary motion produce Doppler shifts that have nothing to do with expansion",
    }

    T["how bacteria develop antibiotic resistance"] = {
        "_cat": "science",
        "what": "the evolutionary process by which bacterial populations acquire genetic changes that allow them to survive concentrations of antibiotics that would have killed their ancestors, through mutation or horizontal transfer of resistance genes",
        "how": "antibiotics kill susceptible cells but leave rare resistant variants alive, which then reproduce. Resistance arises through point mutations that alter drug targets (penicillin-binding proteins, gyrase), expression of efflux pumps, production of inactivating enzymes (beta-lactamases), or modification of porin channels. Plasmids, transposons, and integrons spread resistance genes between species via conjugation, transformation, and transduction",
        "why": "resistance is rolling back the antibiotic era; over a million deaths per year are now directly attributable to resistant infections, and routine surgeries and chemotherapy depend on effective antibiotics. Stewardship, surveillance, and new drug development are global health priorities",
        "vs": "antibiotic resistance differs from antiviral resistance, which evolves under selection from antiviral drugs and often involves rapid RNA virus mutation, and from intrinsic insensitivity, which is the natural inability of certain species to be affected by certain drugs",
        "ex": "MRSA (methicillin-resistant Staphylococcus aureus) carries the mecA gene encoding an alternative penicillin-binding protein with low affinity for beta-lactam antibiotics. It originated in hospitals in the 1960s and now circulates in communities worldwide, complicating skin and soft-tissue infection treatment",
        "mis": "people think resistance arises because bacteria 'learn' to resist. They do not; mutation is random, and antibiotic exposure selects survivors. Another myth is that resistance only occurs with overuse; even appropriate use selects for resistance, although overuse accelerates it dramatically",
    }

    T["entanglement in quantum mechanics"] = {
        "_cat": "science",
        "what": "a uniquely quantum correlation between two or more particles whose joint state cannot be described as a product of individual states, so that measuring one instantly determines properties of the others regardless of separation",
        "how": "entangled pairs are typically generated by processes like spontaneous parametric down-conversion, where a single photon splits into two correlated photons. Their shared wavefunction encodes correlations stronger than any classical mechanism allows. When a measurement collapses one particle's state, the correlation forces the partner's measurement outcome to be consistent, even across great distances",
        "why": "entanglement is the resource behind quantum cryptography (BB84 and its successors), quantum teleportation, quantum computing speedups, and tests of foundational physics through Bell inequality experiments. It is also a key target of fundamental research into the nature of space-time",
        "vs": "entanglement differs from classical correlation in that no hidden local variables can reproduce its statistics, as Bell's theorem proved and decades of experiments confirmed. It differs from superposition, which is a single particle in multiple states, by being a multi-particle phenomenon",
        "ex": "the 2022 Nobel Prize in Physics recognized Aspect, Clauser, and Zeilinger for experiments using entangled photons to violate Bell inequalities, ruling out local hidden-variable theories. Their work also demonstrated quantum teleportation and laid groundwork for quantum networks",
        "mis": "people think entanglement allows faster-than-light communication. It does not; the no-communication theorem proves classical information cannot be sent through entangled particles alone. Another myth is that entanglement is fragile only because of distance; decoherence from any environmental interaction is the real challenge",
    }

    T["the role of chlorophyll in plants"] = {
        "_cat": "science",
        "what": "the green pigment in plant chloroplasts that absorbs sunlight and channels its energy into the photosynthetic reactions that build sugars from carbon dioxide and water",
        "how": "chlorophyll molecules cluster in light-harvesting antenna complexes embedded in thylakoid membranes. They absorb red and blue photons and transfer the excitation to reaction centers, where electrons are stripped from water and pumped through an electron transport chain. The chain builds an ATP-driving proton gradient and reduces NADP+ to NADPH; both products power the Calvin cycle's CO2 fixation",
        "why": "chlorophyll-driven photosynthesis is the basis of nearly all life on Earth, providing the food and oxygen on which animals depend. It is also the target of much agricultural and biotech research aimed at increasing crop yield, drought tolerance, and carbon capture",
        "vs": "chlorophyll a differs from chlorophyll b in absorption spectrum, broadening light capture. It differs from accessory pigments like carotenoids and phycobilins, which absorb wavelengths chlorophyll misses and protect against excess light. It differs from hemoglobin (which has a related porphyrin ring with iron) in centering on magnesium",
        "ex": "leaves turn red and yellow in autumn because chlorophyll degrades faster than other pigments as days shorten and trees prepare for dormancy, unmasking the carotenoids and anthocyanins that were always present",
        "mis": "people think plants are green because they reflect green light as 'unused' energy. The full story is that chlorophyll's absorption peaks in red and blue, leaving green relatively unabsorbed. Another myth is that chlorophyll itself does the chemistry; reaction centers do, while bulk chlorophyll just funnels energy",
    }

    T["the Milankovitch cycles and ice ages"] = {
        "_cat": "science",
        "what": "the slow, periodic variations in Earth's orbit and rotation (eccentricity, obliquity, precession) that change the seasonal and latitudinal distribution of incoming sunlight on tens-of-thousands-of-years timescales and pace glacial-interglacial cycles",
        "how": "eccentricity varies on roughly 100000-year periods, obliquity (axial tilt) on 41000 years, and precession (the wobble of the rotation axis) on about 23000 years. Combined, these change summer insolation at high northern latitudes. Reduced summer insolation lets winter snow persist year-round, building ice sheets; CO2 and albedo feedbacks amplify the orbital nudge into full glacial conditions",
        "why": "Milankovitch theory is the timekeeper of the Pleistocene ice ages and is central to paleoclimate reconstruction, climate model validation, and understanding the natural baseline against which anthropogenic warming is measured",
        "vs": "Milankovitch forcing differs from solar variability (changes in the Sun's output, much smaller) and from internal climate variability (ENSO, AMO) in operating on millennial rather than annual to decadal timescales. It differs from greenhouse gas forcing in being orbital rather than atmospheric",
        "ex": "ice cores from Antarctica preserve a record of CO2 and temperature over the past 800000 years that tracks the 100000-year eccentricity cycle, with sawtooth glacial-interglacial transitions; the last interglacial peaked about 125000 years ago",
        "mis": "people think Milankovitch cycles will quickly dominate over modern warming. The next orbital cooling is tens of thousands of years away, and current CO2 forcing far exceeds the Milankovitch signal in magnitude. Another myth is that orbital forcing alone explains ice ages; feedbacks (CO2, ice-albedo) provide most of the temperature response",
    }

    T["the structure of DNA"] = {
        "_cat": "science",
        "what": "the double-stranded helical molecule that stores genetic information, composed of two antiparallel sugar-phosphate backbones connected by complementary base pairs (A-T and G-C) that encode the instructions for life",
        "how": "each DNA strand is a polymer of nucleotides, each containing a deoxyribose sugar, a phosphate group, and one of four nitrogenous bases. The two strands run antiparallel, twisted in a right-handed double helix with about 10.5 base pairs per turn. Hydrogen bonds between A-T (two bonds) and G-C (three bonds) hold the strands together; base stacking adds further stability. The genetic code is read in triplets along one strand",
        "why": "DNA's structure made the mechanism of inheritance comprehensible, enabled molecular biology, drove the genetic revolution from PCR to sequencing to CRISPR, and underlies medicine, agriculture, and forensics. The 1953 Watson-Crick-Franklin model is one of the most consequential scientific discoveries",
        "vs": "DNA differs from RNA in having deoxyribose instead of ribose, thymine instead of uracil, and being typically double-stranded rather than single. It differs from protein in being an information-bearing polymer with a small four-letter alphabet rather than a functional polymer with twenty",
        "ex": "the Sanger sequencing method exploited DNA's structure: chain-terminating dideoxynucleotides incorporated during synthesis stop extension, producing fragments that reveal sequence by gel electrophoresis. This technique sequenced the first human genome and underlies later high-throughput methods",
        "mis": "people think DNA is the only carrier of inheritance. RNA can carry genetic information in some viruses, and epigenetic marks atop DNA also pass between generations. Another myth is that the helix is rigid; DNA is flexible and dynamically opened by replication and transcription machinery",
    }

    T["how blood clotting works"] = {
        "_cat": "science",
        "what": "the cascade of reactions that converts circulating soluble proteins into a stable insoluble fibrin mesh trapping platelets and red cells at sites of vascular injury, stopping bleeding while limiting the response to the wound site",
        "how": "injury exposes tissue factor, triggering the extrinsic pathway. Sequential activation of clotting factors (VII, X, V, II) culminates in thrombin, which cleaves fibrinogen into fibrin monomers that polymerize. Platelets, activated by exposed collagen and thrombin, aggregate and provide a phospholipid surface that accelerates the cascade. Anticoagulant pathways (protein C, antithrombin) confine clotting to the injury, and fibrinolysis later dissolves the clot",
        "why": "clotting prevents fatal bleeding from minor injuries, but pathological clotting causes heart attacks, strokes, and pulmonary embolism. Anticoagulants like warfarin, heparin, and direct oral anticoagulants are among the most prescribed and consequential drugs in medicine",
        "vs": "primary hemostasis (platelet plug) differs from secondary hemostasis (fibrin mesh) by being faster but less durable. The intrinsic pathway differs from the extrinsic pathway in initiation; both converge at factor X. Clotting differs from inflammation, although the two share signals and cellular responders",
        "ex": "patients with hemophilia A lack functional factor VIII; even small injuries can cause prolonged bleeding into joints and muscles. Recombinant factor VIII concentrates and recent gene therapies now allow many patients near-normal lives where previously the disease was severely disabling",
        "mis": "people think aspirin 'thins the blood'. It actually inhibits platelet aggregation by blocking thromboxane synthesis; it does not change blood viscosity. Another myth is that all clotting is good; deep vein thrombosis and arterial clots cause many of the leading causes of death globally",
    }

    T["how viruses replicate inside cells"] = {
        "_cat": "science",
        "what": "the process by which a virus, lacking the metabolic machinery of life, hijacks a host cell to copy its genome and produce new virions, the central mechanism of viral infection",
        "how": "a virus attaches to a specific cell-surface receptor and enters by membrane fusion or endocytosis. The capsid releases the viral genome (DNA or RNA) into the cytoplasm or nucleus. The host's machinery (ribosomes, polymerases) plus virally encoded enzymes copy the genome and translate viral proteins. New capsids assemble around copied genomes, and progeny virions exit by lysis or budding, often acquiring an envelope from the host membrane. Each step is a potential antiviral drug target",
        "why": "understanding replication underlies vaccine design (which steps to interrupt), antiviral therapy (protease inhibitors, polymerase inhibitors), and pandemic preparedness. It explains why antibiotics do not work on viruses and why some viruses evolve resistance to drugs faster than others",
        "vs": "viral replication differs from cell division (viruses have no metabolism of their own), from bacterial reproduction (bacteria divide as autonomous cells), and from prion propagation (prions copy a protein conformation rather than a genome). DNA viruses differ from RNA viruses in error rates and replication site",
        "ex": "HIV reverse transcribes its RNA genome into DNA via reverse transcriptase, integrates the DNA into the host chromosome via integrase, and later transcribes new viral RNA. Each enzyme is targeted by a specific drug class; combination therapy interrupts multiple steps and prevents resistance",
        "mis": "people think viruses are alive in the same sense as cells. They are at the boundary, requiring a host to reproduce. Another myth is that all viruses cause severe disease; many cause mild or no symptoms, and some integrated viral sequences contribute beneficial functions to host genomes",
    }

    T["the formation of stars"] = {
        "_cat": "science",
        "what": "the process by which gravitational collapse of dense regions in molecular clouds produces protostars that ignite hydrogen fusion in their cores and join the main sequence as new stars",
        "how": "a giant molecular cloud fragments under gravity into denser cores. Each core collapses, heated by compression and radiative trapping. As temperature rises, a protostar forms surrounded by an accretion disk; jets along the rotation axis carry angular momentum away. When the core reaches about 10 million Kelvin, hydrogen fusion ignites, the star contracts to its main-sequence radius, and it stabilizes by balancing fusion energy against gravity",
        "why": "star formation drives galaxy evolution, recycles heavy elements through stellar winds and supernovae, sets the conditions for planet formation, and creates the chemical complexity that life requires. Studying it links cosmology to chemistry to astrobiology",
        "vs": "star formation differs from planet formation, which occurs in the disk around a young star and produces objects too small to ignite fusion. It differs from white dwarf, neutron star, or black hole formation, which mark the deaths of stars rather than births",
        "ex": "the Eagle Nebula's 'Pillars of Creation', imaged by Hubble in 1995 and JWST in 2022, shows molecular hydrogen columns being eroded by ultraviolet light from young massive stars while new protostars form within. The image captures star birth in real time on a galactic scale",
        "mis": "people think stars form quickly. From cloud collapse to main sequence takes millions to tens of millions of years, much slower than human timescales but rapid in cosmic terms. Another myth is that all stars form alone; most form in clusters and many remain in binary or multiple-star systems",
    }
