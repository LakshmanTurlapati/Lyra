#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel bank: science topics (fresh for batch I)."""


def register(T):
    T["the carbon cycle"] = {
        "_cat": "science",
        "what": "the global biogeochemical cycle through which carbon moves between Earth's atmosphere, oceans, biosphere, soils, and rocks, with reservoirs ranging from fast (atmosphere, vegetation) to slow (sedimentary rocks)",
        "how": "photosynthesis pulls atmospheric CO2 into plants and phytoplankton; respiration and decay return it. Oceans absorb CO2 at the surface and release it elsewhere depending on temperature and circulation. Over geologic time, weathering of silicate rocks consumes CO2 and burial of organic carbon and carbonates locks it away. Volcanism returns deep carbon to the atmosphere",
        "why": "the carbon cycle regulates Earth's climate by setting atmospheric CO2 and methane levels. Human emissions of fossil carbon are perturbing the cycle faster than slow sinks can absorb, driving global warming. Understanding sinks is essential for climate policy, carbon markets, and ecosystem management",
        "vs": "the carbon cycle differs from the nitrogen cycle, which is dominated by microbial fixation and denitrification, and from the water cycle, which moves a single compound through phase changes. It differs from a closed industrial loop because natural fluxes are huge but slow",
        "ex": "the Mauna Loa CO2 record shows a clear annual sawtooth: northern hemisphere vegetation pulls CO2 down each summer, releases it each winter, with a steady upward trend on top from fossil fuel emissions",
        "mis": "people think planting trees alone can offset fossil emissions. Forest carbon is a fast pool that can be released by fire, drought, or land-use change, while fossil carbon is a slow pool. The two are not interchangeable on relevant timescales",
    }

    T["plate tectonics"] = {
        "_cat": "science",
        "what": "the unifying theory that Earth's lithosphere is broken into rigid plates that move over the ductile asthenosphere, with most geological activity concentrated at plate boundaries",
        "how": "convection in the mantle, combined with slab pull from subducting plates and ridge push at spreading centers, drives plate motion of a few centimeters per year. At divergent boundaries new crust forms; at convergent boundaries plates collide and one subducts; at transform boundaries plates slide past each other",
        "why": "plate tectonics explains earthquakes, volcanoes, mountain ranges, ocean basins, and the distribution of fossils and minerals. It is the master framework for solid-earth science and underpins hazard assessment, resource exploration, and our understanding of long-term climate via the carbon-silicate cycle",
        "vs": "plate tectonics differs from continental drift (Wegener's earlier idea) in providing a mechanism via seafloor spreading. It differs from isostasy, which describes vertical adjustment of crust under load, and from mantle plume theory, which addresses hotspots not tied to plate boundaries",
        "ex": "the San Andreas Fault is a transform boundary where the Pacific Plate slides north-northwest relative to the North American Plate at roughly 35 mm per year, producing the seismic hazard along coastal California",
        "mis": "people think continents float on liquid magma. They ride on solid but slowly flowing mantle rock; the asthenosphere is plastic, not molten. Another myth is that plates only move during earthquakes; they move continuously, with earthquakes releasing accumulated strain at locked patches",
    }

    T["enzyme kinetics"] = {
        "_cat": "science",
        "what": "the quantitative study of the rates at which enzymes catalyze biochemical reactions and the dependence of those rates on substrate concentration, enzyme concentration, temperature, pH, and inhibitors",
        "how": "the Michaelis-Menten model treats catalysis as a two-step process: enzyme and substrate form a complex, which then converts to product. The rate equation v = Vmax[S]/(Km+[S]) gives a hyperbolic curve where Vmax is the maximum rate at saturating substrate and Km is the substrate concentration at half-maximal rate. Lineweaver-Burk plots linearize this for fitting",
        "why": "kinetic parameters reveal enzyme efficiency, drug binding modes (competitive vs noncompetitive inhibition), metabolic flux control, and the design of biocatalysts. Most clinical drugs target enzymes, so understanding kinetics is essential for pharmacology and toxicology",
        "vs": "Michaelis-Menten kinetics differs from cooperative kinetics (sigmoidal, modeled by the Hill equation), seen in allosteric enzymes like hemoglobin. It differs from non-enzymatic chemical kinetics in featuring saturation behavior because of finite enzyme active sites",
        "ex": "competitive inhibitors of HMG-CoA reductase, the statins, lower cholesterol biosynthesis. Their kinetic signature is increased apparent Km with unchanged Vmax, distinguishing them from noncompetitive inhibitors that lower Vmax",
        "mis": "people think Km measures binding affinity. It approximates affinity only when substrate dissociation dominates over catalysis; otherwise it bundles binding and turnover. Another myth is that fast enzymes are always more efficient; catalytic efficiency is kcat/Km, not raw turnover number",
    }

    T["the immune system's adaptive arm"] = {
        "_cat": "science",
        "what": "the antigen-specific branch of vertebrate immunity, mediated by B and T lymphocytes that recognize particular molecular features and develop long-lived memory after exposure",
        "how": "naive lymphocytes generate diverse receptors by V(D)J recombination. When a lymphocyte encounters its cognate antigen presented on MHC molecules with appropriate co-stimulation, it proliferates and differentiates. B cells secrete antibodies; cytotoxic T cells kill infected cells; helper T cells coordinate responses. Memory cells persist for years",
        "why": "adaptive immunity allows vaccines to confer durable protection without the disease, controls many cancers, and underlies organ transplant rejection and autoimmunity. It is the basis of immunotherapy, including checkpoint inhibitors and CAR-T cell therapy that are reshaping cancer treatment",
        "vs": "adaptive immunity differs from innate immunity, which is fast, generic, and uses pattern recognition receptors. It differs from passive immunity (transferred antibodies, like maternal IgG) in being acquired by the host's own response and producing memory",
        "ex": "after a measles vaccination, B cells producing high-affinity anti-measles antibodies and memory T cells persist for decades. Reexposure triggers a rapid secondary response that prevents disease, illustrating the durability of adaptive memory",
        "mis": "people think antibodies attack pathogens directly like a poison. They mostly mark them for destruction by neutralization, opsonization, or complement activation. Another myth is that the adaptive system 'kicks in' only after the innate system fails; the two are entangled and run in parallel",
    }

    T["the standard model of particle physics"] = {
        "_cat": "science",
        "what": "the quantum field theory that describes the known fundamental particles (quarks, leptons, gauge bosons, Higgs) and their interactions via the strong, weak, and electromagnetic forces, omitting gravity",
        "how": "matter particles are spin-1/2 fermions; force carriers are spin-1 gauge bosons (photon, W and Z, gluons). Forces arise from local gauge symmetries: U(1) for electromagnetism, SU(2) for weak, SU(3) for strong. The Higgs field gives masses to W, Z, and fermions through spontaneous symmetry breaking",
        "why": "the Standard Model is the most rigorously tested theory in physics, predicting outcomes from atomic spectra to LHC collisions to extreme precision. It anchors all of accelerator physics and informs cosmology back to microseconds after the Big Bang. Its limitations point toward physics beyond, including dark matter and neutrino masses",
        "vs": "the Standard Model differs from grand unified theories that fold the three forces into one symmetry, and from string theory, which embeds particles as vibrating strings and includes gravity. It differs from classical electrodynamics by being quantum and including weak and strong forces",
        "ex": "the discovery of the Higgs boson at the LHC in 2012, with mass near 125 GeV, completed the particle content predicted by the Standard Model and confirmed the mechanism by which other particles acquire mass",
        "mis": "people think the Standard Model 'explains everything.' It omits gravity, dark matter, dark energy, neutrino masses, and the matter-antimatter asymmetry. Another myth is that all particles get mass from the Higgs; most of a proton's mass comes from QCD binding energy, not Higgs coupling",
    }

    T["meiosis and genetic recombination"] = {
        "_cat": "science",
        "what": "the specialized cell division that produces haploid gametes from diploid germ cells, halving the chromosome number and shuffling genetic material via independent assortment and crossing over",
        "how": "meiosis I separates homologous chromosomes after they pair and exchange segments at chiasmata, producing genetic recombinants. Meiosis II then separates sister chromatids like mitosis. The result is four haploid cells, each genetically unique due to recombination and independent segregation of homologs",
        "why": "meiosis is the engine of sexual reproduction's genetic diversity. Recombination produces new combinations of alleles each generation, accelerating adaptation, breaking up linkage between deleterious and beneficial mutations, and making genetic mapping possible",
        "vs": "meiosis differs from mitosis, which produces two identical diploid daughters and underlies growth and tissue maintenance. It differs from binary fission in prokaryotes, which is asexual and clonal, and from mitotic recombination, which is rare and somatic",
        "ex": "in genetic mapping, the recombination frequency between two loci on the same chromosome estimates their distance. Loci 10 percent recombinant are 10 centimorgans apart, a method that built the first genetic maps before DNA sequencing",
        "mis": "people think children inherit a 'random mix' of grandparent DNA. The mix is not uniform; recombination rates vary across the genome, and large blocks (linkage disequilibrium) tend to be inherited together. Another myth is that meiosis is error-free; nondisjunction can produce aneuploidies like Down syndrome",
    }

    T["light pollution"] = {
        "_cat": "science",
        "what": "the excessive or misdirected anthropogenic light that brightens the night sky and outdoor environments, with components including skyglow, light trespass, glare, and clutter",
        "how": "outdoor lights, particularly upward-facing or unshielded fixtures, scatter off atmospheric particles to brighten the sky. Wavelength matters: blue-rich LEDs scatter more strongly than warmer lights. Population density and weather modulate skyglow, which can extend hundreds of kilometers from cities",
        "why": "light pollution disrupts circadian rhythms in humans and wildlife, alters predator-prey interactions, impairs sea turtle hatchling navigation, suppresses pollinator activity, and erases the night sky for most of the world's population. It also wastes energy by lighting nothing useful",
        "vs": "light pollution differs from chemical air pollution in that it disappears the moment lights are switched off, making it the most reversible form of pollution. It differs from noise pollution in operating across very long distances and mostly at night",
        "ex": "the 2003 Northeast blackout left millions without power but produced a striking secondary observation: residents saw the Milky Way for the first time in years, illustrating how even normal urban lighting drowns out natural night skies",
        "mis": "people think dark-sky friendly lighting means dimmer lighting. The fix is mostly about direction and color: shielded, downward-pointing, warm-temperature fixtures can be plenty bright while sharply reducing skyglow",
    }

    T["the periodic table's structure"] = {
        "_cat": "science",
        "what": "the systematic arrangement of chemical elements by increasing atomic number into rows (periods) and columns (groups), grouping elements with similar valence-electron configurations and chemical behavior",
        "how": "elements are ordered by proton count. Quantum mechanics dictates that electrons fill orbitals in shells of increasing energy, and the table's blocks (s, p, d, f) correspond to which subshell is being filled. Group number reflects valence electrons, which dominate chemical bonding and reactivity",
        "why": "the periodic table organizes essentially all of chemistry: trends in atomic radius, ionization energy, electronegativity, and reactivity become predictable from position. It guides synthesis, materials design, and pedagogy, and was a triumph of inferring structure from observed pattern before quantum theory",
        "vs": "Mendeleev's table was ordered by atomic weight, with gaps for undiscovered elements. The modern table is ordered by atomic number, which resolved anomalies like tellurium and iodine. It differs from a flat alphabetical list by encoding chemical periodicity",
        "ex": "Mendeleev predicted properties of 'eka-silicon' (germanium) decades before its discovery in 1886, including density, atomic weight, and oxide formula, an early validation of the table's predictive power",
        "mis": "people think periodicity is exact. Trends within groups have anomalies (lithium in group 1 behaves more like magnesium in some respects, the 'diagonal relationship'). Another myth is that the table is finished; superheavy element synthesis continues to extend the seventh period",
    }

    T["the second law of thermodynamics"] = {
        "_cat": "science",
        "what": "the principle that the total entropy of an isolated system never decreases over time, and that heat flows spontaneously from hot to cold, with consequences ranging from engine efficiency limits to the arrow of time",
        "how": "any spontaneous process produces entropy in the universe. Reversible processes are an idealized limit producing zero entropy change; real processes always have friction, heat leaks, or finite gradients that produce entropy. The Carnot bound caps the efficiency of any heat engine working between two reservoirs at 1 - Tcold/Thot",
        "why": "the second law sets fundamental limits on engines, refrigerators, biological energy use, and information processing (Landauer's principle ties entropy to bit erasure). It provides the physical basis for the arrow of time and constrains what cosmological scenarios are consistent",
        "vs": "the second law differs from the first law (energy conservation), which permits processes the second law forbids. It differs from the third law (entropy approaches a constant as temperature approaches zero) and from kinetic statements like the Boltzmann H-theorem that derive irreversibility from microscopic dynamics",
        "ex": "a real car engine converts perhaps 25 percent of fuel energy to motion; the rest leaves as heat. Even a perfect Carnot engine between combustion and exhaust temperatures could not exceed roughly 60 percent, illustrating how the second law caps performance",
        "mis": "people think the second law forbids local order from arising, citing it against evolution. The law applies to isolated systems; Earth is not isolated and dumps entropy to space via thermal radiation. Local order increases at the cost of larger entropy increases elsewhere",
    }

    T["soil ecology"] = {
        "_cat": "science",
        "what": "the study of the living organisms in soil and their interactions with each other and the physical-chemical environment, encompassing microbes, fungi, protists, nematodes, arthropods, and plant roots",
        "how": "soil is structured by aggregates of mineral particles, organic matter, water, and air. Decomposers break down detritus, releasing nutrients. Mycorrhizal fungi exchange phosphorus and water for plant carbon. Nitrogen-fixing bacteria, nitrifiers, and denitrifiers cycle nitrogen. Root exudates feed and recruit specific microbial communities",
        "why": "soils underpin agriculture, store more carbon than the atmosphere and vegetation combined, filter water, and host a quarter of Earth's biodiversity. Soil degradation threatens food security; soil restoration is a key climate and ecological lever",
        "vs": "soil ecology differs from soil chemistry by foregrounding organisms; from microbiology by including macrofauna; and from plant ecology by spanning the rhizosphere as well as bulk soil. It differs from sediment ecology, which deals with submerged systems",
        "ex": "no-till agriculture preserves fungal hyphal networks and aggregate structure, which improves water retention and reduces erosion compared to conventional plowing, while keeping more carbon stored belowground",
        "mis": "people think soil is mostly dirt. By volume, healthy soil is roughly half pore space (water and air), with mineral particles and a small but biologically dominant organic and living fraction running the show",
    }

    T["redshift and cosmic expansion"] = {
        "_cat": "science",
        "what": "the observed lengthening of light wavelengths from distant galaxies, attributed to the expansion of space itself stretching the light during transit, and quantified by the redshift parameter z",
        "how": "as space expands during a photon's flight, its wavelength stretches in proportion to the scale factor. Observed wavelength divided by emitted wavelength minus 1 gives z. Hubble's law relates redshift to recession velocity at small z; at large z general-relativistic cosmology is required",
        "why": "redshift is the foundational observation behind the Big Bang model. Combined with standard candles like Type Ia supernovae, it revealed accelerating cosmic expansion in 1998, implying dark energy. The CMB's redshift to the microwave band is a key Big Bang prediction confirmed",
        "vs": "cosmological redshift differs from Doppler redshift (motion through space) and from gravitational redshift (climbing out of a potential well), although all three stretch wavelengths. Locally bound systems do not partake in cosmic expansion; redshift dominates only at large scales",
        "ex": "GN-z11, observed at redshift around 11, emitted its light when the universe was 400 million years old. JWST has now pushed the frontier past z = 14, watching galaxies form within hundreds of millions of years of the Big Bang",
        "mis": "people think galaxies move through space away from us. In the standard interpretation, space itself expands and carries galaxies with it; recession velocities at large distances can exceed the speed of light without violating special relativity, which forbids local superluminal motion",
    }

    T["the discovery of insulin"] = {
        "_cat": "science",
        "what": "the 1921 isolation of a pancreatic hormone able to lower blood sugar, by Frederick Banting, Charles Best, J.J.R. Macleod, and James Collip in Toronto, transforming type 1 diabetes from a death sentence into a manageable condition",
        "how": "Banting reasoned that ductal ligation would atrophy the digestive enzyme-producing tissue while sparing the islets. Extracts from such pancreases lowered glucose in diabetic dogs. Collip's purification produced a clinical-grade extract; the first patient, Leonard Thompson, was treated successfully in January 1922. Industrial production followed within a year",
        "why": "before insulin, type 1 diabetes killed children within months of diagnosis. Insulin made it survivable and demonstrated that hormonal replacement therapy could rescue an endocrine deficiency. It became the template for subsequent biologics and the symbol of medical translation done quickly",
        "vs": "insulin therapy differs from oral hypoglycemics like metformin, which target hepatic glucose output and insulin sensitivity rather than replacing the hormone. Modern recombinant insulin differs from the original animal-pancreas extracts by being human-sequence and far purer",
        "ex": "Elizabeth Hughes, the daughter of US Secretary of State Charles Evans Hughes, was near death from diabetes in 1922 at age 14. After treatment with insulin she lived another 59 years, an early demonstration of the drug's transformative effect",
        "mis": "people think Banting alone discovered insulin. Macleod ran the lab and provided dogs and methods; Collip's biochemistry made the extract clinically usable; Best assisted with the surgical work. The Nobel committee split the prize between Banting and Macleod; Banting and Macleod each shared their portions with Best and Collip",
    }

    T["materials science of glass"] = {
        "_cat": "science",
        "what": "the study of the structure, properties, and processing of glass, an amorphous solid that retains the disordered atomic arrangement of a liquid because it is cooled too fast for crystallization to occur",
        "how": "silica-based glasses are made by melting silicon dioxide with network modifiers (sodium oxide, calcium oxide) and cooling through the glass-transition temperature. Below Tg, viscosity is so high that atoms cannot rearrange to form crystals on practical timescales. Other formers (boron oxide, phosphorus oxide) and modifiers tune properties",
        "why": "glass enables windows, optical fibers, displays, lab and pharmaceutical containers, and many electronic substrates. It supports global telecommunications via low-loss silica fibers and is indispensable to architecture and energy-efficient buildings. Specialty glasses underpin laser systems and semiconductor lithography",
        "vs": "glass differs from a crystalline solid by lacking long-range order; the diffraction pattern is diffuse rather than sharp. It differs from a true liquid by having extreme viscosity and from a polymer in being inorganic and isotropic in mechanical response",
        "ex": "Corning's Gorilla Glass uses an ion-exchange process that swaps small sodium ions for larger potassium ions in the surface, putting the surface in compression and dramatically increasing scratch and impact resistance for smartphone screens",
        "mis": "people think old window glass is thicker at the bottom because glass slowly flows. It does not at room temperature on human timescales; the thickness variation comes from period manufacturing methods and how panes were installed. Glass viscosity is enormous below Tg",
    }

    T["the chemistry of fireworks colors"] = {
        "_cat": "science",
        "what": "the use of metal salts in pyrotechnic compositions to produce specific colors via emission lines from electronic transitions when heated to high temperatures",
        "how": "an oxidizer drives combustion of a fuel, reaching temperatures of 1500 to 3000 K. Metal atoms in the flame are excited and emit characteristic wavelengths as electrons fall back. Strontium gives red, barium green, copper blue, sodium yellow, and combinations produce purples and oranges. Binders hold pellets together; chlorine donors enhance certain emissions",
        "why": "fireworks chemistry shows quantum mechanics in everyday life: the bright distinct colors arise because electron energy levels are quantized. The same emission spectroscopy diagnoses stellar compositions, identifies elements in industrial labs, and underlies sodium-vapor lamps and fluorescent lighting",
        "vs": "pyrotechnic emission differs from fluorescent lighting (UV-excited phosphor) and from incandescence (broad-spectrum thermal radiation). It differs from chemiluminescence (cold light from chemical reaction, like glow sticks) by relying on high temperatures",
        "ex": "blue is the hardest firework color because copper emission requires precise temperatures and chloride content; too hot and the color washes out, too cool and combustion fails. Modern blue compositions use copper chloride with ammonium perchlorate as the oxidizer to control flame temperature",
        "mis": "people think the colors come from dyes in the powder. They come from atomic emission, the same physics as a flame test in chemistry class. The mineral, not a dye, defines the color",
    }
