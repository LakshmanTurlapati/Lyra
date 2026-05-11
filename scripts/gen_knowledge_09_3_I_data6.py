#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel bank: arts/humanities and everyday_life topics (fresh for batch I)."""


def register(T):
    # Arts and humanities
    T["the structure of a sonnet"] = {
        "_cat": "arts_and_humanities",
        "what": "a 14-line poem in iambic pentameter with a fixed rhyme scheme, dating to 13th-century Sicily and developed in two main forms: the Italian (Petrarchan) and English (Shakespearean) sonnet",
        "how": "the Petrarchan sonnet divides into an octave (abbaabba) presenting a problem and a sestet (cdecde or cdcdcd) offering resolution, with a turn (volta) at line 9. The Shakespearean form uses three quatrains plus a couplet (abab cdcd efef gg), with the couplet often delivering a punch line or reversal",
        "why": "the sonnet's compression and turn create concentrated argumentation in poetry, used by Petrarch, Shakespeare, Milton, Wordsworth, Hopkins, and many modern poets. Its constraints train precise expression and remain a staple of poetic education and contemporary practice",
        "vs": "the sonnet differs from the longer ode (irregular stanzas, public address) and from the looser lyric (no fixed scheme). The Petrarchan and Shakespearean forms differ in placement of the volta and in rhyme structure; the Spenserian sonnet uses interlocking quatrains (abab bcbc cdcd ee)",
        "ex": "Shakespeare's Sonnet 18 ('Shall I compare thee to a summer's day?') opens with a comparison, develops it through three quatrains, and lands the immortality claim in the closing couplet, illustrating the form's argumentative arc",
        "mis": "people think any 14-line poem is a sonnet. Without iambic pentameter and a recognizable rhyme scheme it usually isn't. Another myth is that the form is rigid; modern sonnets (Hayden, Heaney, Hill) bend rules while keeping the architectural feel",
    }

    T["realism in 19th-century literature"] = {
        "_cat": "arts_and_humanities",
        "what": "the literary movement that emerged in mid-19th-century France, England, and Russia emphasizing accurate depiction of contemporary life, ordinary characters, and social conditions, in reaction to Romantic idealization",
        "how": "realist writers used detailed observation, free-indirect-discourse narration to render consciousness, and plots focused on social relations, marriage, money, and class. They treated provincial and bourgeois subjects with the seriousness once reserved for nobility, and drew on journalism and contemporary social science",
        "why": "realism made the novel the dominant literary form, established techniques (close third-person narration, social typology) that shape fiction today, and engaged Industrial Revolution-era social transformations. It seeded later naturalism, modernism's rebellions, and continues to be the implicit baseline against which contemporary fiction is measured",
        "vs": "realism differs from Romanticism (emotional, idealizing, often medieval) and from naturalism (realism's deterministic, scientific successor in Zola and Norris). It differs from modernism, which fragmented narrative consciousness, and from genre fiction, which centered plot over psychological texture",
        "ex": "Flaubert's Madame Bovary (1856) was prosecuted for offending public morality but is now canonical for its meticulous portrayal of provincial life and its narrative innovation. Tolstoy's Anna Karenina and Eliot's Middlemarch extended the form with greater social and psychological scope",
        "mis": "people think realism aims at literal transcription. Realists shape and select; Henry James insisted that realism is an art of selection. The illusion of accuracy is achieved through technique, not surrender to facts",
    }

    T["modernist architecture"] = {
        "_cat": "arts_and_humanities",
        "what": "the early-to-mid-20th-century architectural movement that rejected ornament in favor of geometric forms, new materials (steel, concrete, glass), and the functionalist principle that form should follow function",
        "how": "architects like Le Corbusier, Mies van der Rohe, and Walter Gropius embraced industrial production, open floor plans, ribbon windows, flat roofs, and pilotis. Materials were used honestly and structure was expressed rather than concealed. The Bauhaus integrated architecture with industrial design and craft",
        "why": "modernism reshaped cities worldwide, from postwar housing blocks to corporate headquarters. It addressed urgent housing needs after two world wars but also produced sterile environments and contributed to mid-century urban decay critiques. It still anchors contemporary architectural education and influences ongoing minimalist design",
        "vs": "modernism differs from preceding Beaux-Arts and historicist styles, which used classical orders and decorative elaboration. It differs from postmodernism, which reintroduced ornament, irony, and historical reference, and from contemporary parametric architecture, which exploits computational design tools",
        "ex": "Mies van der Rohe's Seagram Building (1958) in New York exemplifies modernism's bronze-and-glass corporate idiom: rigorous proportions, expressed structure, and a plaza setback that influenced zoning codes and skyscraper design for decades",
        "mis": "people think modernism is one style. It encompasses International Style, Brutalism, Mid-Century, and others; its practitioners disagreed sharply about ornament, regionalism, and humanism. Le Corbusier and Frank Lloyd Wright shared a century but little else stylistically",
    }

    T["jazz improvisation"] = {
        "_cat": "arts_and_humanities",
        "what": "the spontaneous composition of melodic lines over a known harmonic framework in jazz, drawing on memorized vocabulary, scales, and rhythmic feel, and centering on individual voice within ensemble interplay",
        "how": "soloists internalize chord-scale relationships (e.g., Dorian over a minor 7 chord, altered scale over a dominant), patterns and licks, and rhythmic devices. They listen to the rhythm section and respond, develop motifs, vary articulation and dynamics, and reshape phrases over the form. Years of transcription and ear training build the vocabulary",
        "why": "improvisation is the soul of jazz performance, distinguishing it from notated traditions and producing a tradition where musical statement and personal expression coincide. Through bebop, modal, free, and fusion eras, improvisational language evolved while remaining the primary medium of jazz innovation",
        "vs": "jazz improvisation differs from classical cadenzas, which were once improvised but now generally composed. It differs from pop solos, which are usually composed and repeated. It differs from free-form improvisation lacking harmonic structure, which Ornette Coleman and others developed",
        "ex": "John Coltrane's solos on 'Giant Steps' (1959) navigate one of the densest harmonic progressions in jazz at high tempo, showing pre-composed patterns ('Coltrane changes') deployed improvisationally. Generations of saxophonists have studied the recording note by note",
        "mis": "people think jazz improvisation is making it up from scratch. It draws on deep memorized vocabulary; spontaneity is in selection and recombination, not invention from nothing. Another myth is that improvisation is unconstrained; harmonic and rhythmic frameworks shape every choice",
    }

    T["chiaroscuro and Baroque painting"] = {
        "_cat": "arts_and_humanities",
        "what": "the dramatic use of strong contrasts between light and dark to model form and create emotional intensity, perfected by Caravaggio in the late 16th century and central to Baroque painting across Italy, Spain, and the Low Countries",
        "how": "painters used a single directional light source against deep shadow (tenebrism) to push figures forward and emphasize gesture and expression. Brushwork built mid-tones; reserved highlights drew the eye. Compositions often placed pivotal action at the threshold between dark and light, heightening narrative impact",
        "why": "chiaroscuro revolutionized religious and historical painting, giving viewers theatrical immediacy and emotional shock. It influenced Rembrandt's psychologically penetrating portraits, Velázquez's court paintings, and the entire dramatic vocabulary of Baroque art",
        "vs": "chiaroscuro differs from earlier Renaissance balanced lighting (sfumato in Leonardo, even illumination in Raphael) by extreme contrast. It differs from Impressionism's high-key broken color, and from photography's tonal range, although photographers later borrowed Baroque lighting deliberately",
        "ex": "Caravaggio's 'Calling of Saint Matthew' (1599-1600) directs a shaft of light from upper right onto Christ's gesture and the dimly lit tax collectors, making the moment of vocation legible at a glance and immersing the viewer in the moral drama",
        "mis": "people think chiaroscuro is just dark backgrounds. It is the modeling of form by graded contrast, requiring careful tonal observation; mere blackness without considered transitions yields flatness, not the volumetric force of Caravaggio or Rembrandt",
    }

    T["the structure of Greek epic poetry"] = {
        "_cat": "arts_and_humanities",
        "what": "the conventions of long narrative poems in dactylic hexameter centered on heroic action, shaped by the Homeric Iliad and Odyssey and elaborated by later epic in Greek and Latin",
        "how": "epics open with an invocation of the muse and a statement of theme, plunge in medias res, deploy stock epithets and similes, catalog warriors or ships, feature divine intervention, and structure narrative around an aristos (best of the warriors). Oral-formulaic composition provided the poet a flexible toolkit of phrases fitting the meter",
        "why": "Homeric epic shaped Greek paideia (education) and influenced all subsequent Western literature. It established narrative conventions visible in Virgil, Dante, Milton, and Joyce. The oral-formulaic theory developed by Parry and Lord transformed our understanding of how preliterate cultures produced long, complex literature",
        "vs": "Greek epic differs from lyric (short, personal, varied meters) and from tragedy (dialogic, performed, moral focus). It differs from later Roman epic (Virgil's Aeneid) in emphasizing individual heroic excellence over national destiny, and from medieval romance, which reshapes the heroic ideal",
        "ex": "the catalog of ships in Iliad Book 2, listing contingents of the Greek expedition, is a tour de force of memorized geography and genealogy that originally helped audiences situate familiar cities and lineages within the heroic past",
        "mis": "people think Homer was a single author writing the texts as we have them. Most scholars view the Iliad and Odyssey as crystallizations of long oral traditions, with 'Homer' a name attached to the synthesizer or transcriber whose role is debated",
    }

    T["the philosophical concept of justice in Rawls"] = {
        "_cat": "arts_and_humanities",
        "what": "John Rawls's 1971 theory of justice as fairness, in which principles for a just society are those that rational people would agree to from behind a 'veil of ignorance' that hides their personal characteristics",
        "how": "behind the veil, individuals do not know their class, talents, or conception of the good. Rawls argues they would adopt two principles: equal basic liberties for all, and social and economic inequalities arranged to benefit the least advantaged (the difference principle) and attached to positions open to all under fair equality of opportunity",
        "why": "A Theory of Justice revived liberal political philosophy after midcentury dominance of utilitarianism, frames much contemporary political theory, and has influenced debates on inequality, healthcare, and constitutional design. Its method (the original position, reflective equilibrium) shaped how moral philosophy is done",
        "vs": "Rawls differs from utilitarianism, which permits sacrificing individuals for aggregate welfare; from libertarianism (Nozick), which rejects redistribution as violating property rights; and from communitarianism (Sandel, MacIntyre), which criticizes the abstraction of the original position",
        "ex": "the difference principle implies that an unequal income distribution is just only if redistributing further to the worst off would lower their absolute position. Progressive taxation and public goods financing are often justified on Rawlsian grounds",
        "mis": "people think Rawls advocates equality of outcome. He explicitly accepts inequalities that benefit the worst off, so a market economy with strong baseline protections can be Rawlsian. Another myth is that the veil is empirical; it is a thought experiment for testing principles, not a real procedure",
    }

    T["the development of Hindi cinema"] = {
        "_cat": "arts_and_humanities",
        "what": "the evolution of Hindi-language cinema from the 1913 silent feature Raja Harishchandra to today's Mumbai-based industry (Bollywood) producing hundreds of films a year and reaching global audiences across the South Asian diaspora",
        "how": "early studios established song-and-dance conventions; Partition reshaped the industry as talent moved between Bombay and Lahore. The Golden Age (1950s-60s) produced socially conscious classics from Raj Kapoor and Guru Dutt. Masala films from the 1970s blended romance, action, and music. Globalization in the 1990s and digital production now diversify formats and themes",
        "why": "Hindi cinema is one of the largest film industries by output and a major cultural force across India, Pakistan, the Gulf, and the diaspora. It shaped Indian popular music (playback singing dominates the music industry) and exported the song-and-dance feature format internationally",
        "vs": "Bollywood differs from Hollywood by integrating multiple genres in a single film, foregrounding music, and embedding stars deeply in the production economy. It differs from regional Indian cinemas (Tamil, Telugu, Bengali, Malayalam), which have distinct industries with their own conventions and stars",
        "ex": "Sholay (1975), a curry-Western blending revenge plot, comedy, and music, ran for years in single theaters and became the template for masala filmmaking. Decades later, Lagaan (2001) reached an Oscar nomination and showed the industry's international reach",
        "mis": "people think 'Bollywood' covers all Indian cinema. It refers specifically to Mumbai-based Hindi cinema; Tamil, Telugu, and other regional industries are equally large or larger by some measures. Another myth is that song sequences are decorative; they typically advance plot and character relationships",
    }

    # Everyday life
    T["how to read a wine label"] = {
        "_cat": "everyday_life",
        "what": "extracting key information from the front and back labels of a wine bottle, including producer, region, vintage, grape variety, alcohol content, and quality designation, to predict what is in the bottle",
        "how": "old-world labels (France, Italy, Spain) often emphasize region (Bordeaux, Chianti, Rioja), assuming buyers know what grapes those regions use. New-world labels (US, Australia, Chile) typically lead with grape (Cabernet, Shiraz, Malbec). Look for vintage, classification (Reserva, DOCG), alcohol percentage as a ripeness clue, and importer if buying internationally",
        "why": "label literacy turns wine shopping from intimidation to informed choice, helps you avoid overspending, and lets you find styles you actually enjoy. It also helps decode unfamiliar bottles when traveling or trying new producers",
        "vs": "old-world labeling differs from new-world by emphasizing terroir over varietal. Quality tiers differ by country: France uses AOC/AOP and Cru hierarchy; Italy uses DOC/DOCG; Germany layers ripeness levels (Kabinett to Trockenbeerenauslese); Spain uses aging tiers (Crianza, Reserva, Gran Reserva)",
        "ex": "a French label reading 'Chablis Premier Cru' tells you it's Chardonnay (the only white grape allowed in Chablis), from a specific Burgundy subregion's better vineyards, with the producer and vintage giving you a quality and style fingerprint without naming the grape at all",
        "mis": "people think higher alcohol means lower quality. It usually reflects ripeness; warmer regions and modern winemaking pushed alcohols higher, which can be a stylistic rather than quality signal. Another myth is that older is always better; many wines are made for early drinking",
    }

    T["sleep cycles and sleep quality"] = {
        "_cat": "everyday_life",
        "what": "the recurring 90-minute cycles through which sleep progresses each night, alternating non-REM stages (light, then deep slow-wave sleep) with REM (rapid eye movement) sleep where most vivid dreaming occurs",
        "how": "sleep begins with light non-REM, descends to deep slow-wave sleep that dominates the first half of the night and supports physical recovery and memory consolidation, then shifts toward longer REM periods in the second half supporting emotional and procedural learning. Adenosine builds up during waking hours and is cleared during sleep",
        "why": "consistent sleep duration and timing affect mood, memory, immune function, weight regulation, and cardiovascular health. Chronic short sleep raises risks of metabolic disease, depression, and dementia. Most adults need 7-9 hours; performance impairment from short sleep accumulates without subjective awareness",
        "vs": "natural sleep differs from drug-induced sleep, which often suppresses REM. Power naps under 30 minutes differ from longer naps that enter deep sleep and produce grogginess. Sleep differs from sedation in supporting active brain processes, including memory consolidation",
        "ex": "polyphasic schedules (multiple short sleeps) are romanticized but rarely sustainable; consolidated nightly sleep produces better cognitive outcomes for nearly everyone. Athletes who increase sleep to 9-10 hours show measurable performance gains in studies",
        "mis": "people think they can train themselves to need less sleep. Genetic short-sleepers exist but are rare; most people who 'function on five hours' are actually impaired and habituated to the impairment. Another myth is that lost sleep can be made up on weekends; partial recovery occurs but cognitive and metabolic effects of weekday deprivation persist",
    }

    T["how credit scores work"] = {
        "_cat": "everyday_life",
        "what": "a numeric summary of an individual's creditworthiness, computed from credit bureau records by scoring models (FICO, VantageScore) and used by lenders to set interest rates and approval thresholds",
        "how": "models weight factors approximately as: payment history 35 percent, amounts owed 30 percent (especially utilization on revolving accounts), length of credit history 15 percent, new credit 10 percent, credit mix 10 percent. Late payments, charge-offs, and bankruptcies hurt heavily and persist for years; on-time payments and low utilization help over time",
        "why": "credit scores affect access to mortgages, auto loans, credit cards, rental approval, insurance pricing in some states, and even some job applications. Small differences (740 vs 680) can mean tens of thousands of dollars over a 30-year mortgage. Building credit is a major early-adulthood financial task",
        "vs": "credit scores differ from credit reports, which are the underlying records; the score is a summary computation. They differ from cash flow underwriting, which uses bank statements, and from social credit systems, which evaluate broader behavior",
        "ex": "to improve a score quickly, pay down credit card balances to under 30 percent of limits (ideally under 10 percent), keep old accounts open to lengthen history, and avoid new credit applications before a major loan. Effects often appear within one or two billing cycles",
        "mis": "people think checking your own credit hurts your score. Soft inquiries (your own checks, prequalifications) do not affect the score; only hard inquiries from lenders for new credit do, and those are minor and short-lived. Another myth is that closing credit cards always helps; closing old or high-limit cards can hurt by reducing average age and increasing utilization",
    }

    T["food labeling and added sugars"] = {
        "_cat": "everyday_life",
        "what": "the disclosure of added sugars on US Nutrition Facts labels, distinguishing sugars introduced during processing from naturally occurring sugars in fruit, milk, and grains",
        "how": "since 2016 (full compliance by 2021), the FDA-required label lists 'Added Sugars' as a separate line under 'Total Sugars,' with a percent Daily Value reference based on a 50 g daily limit. Manufacturers also list added sugars under many ingredient names: cane sugar, high-fructose corn syrup, evaporated cane juice, dextrose, and many more",
        "why": "added sugars contribute to excess calorie intake, weight gain, dental decay, and metabolic disease. The labeling change made it easier for consumers to compare products: a yogurt with 12 g of total sugars could be either dominantly milk lactose or mostly added syrup, and the label now distinguishes them",
        "vs": "added sugars differ from naturally occurring sugars, which come bundled with fiber, protein, or fat that slow absorption. Sugar alcohols (erythritol, xylitol) are not counted as added sugars but as separate carbohydrates with different metabolic effects",
        "ex": "a flavored yogurt with 22 g total sugars and 14 g added sugars carries a meaningful warning; switching to plain yogurt with 8 g total sugars (all natural lactose) eliminates the added portion. The labeling change drove product reformulation toward less added sugar",
        "mis": "people think 'no added sugars' means low sugar. Fruit smoothies and dried fruit are sugar-dense without added sugar. Another myth is that natural sweeteners (honey, maple syrup, agave) are not added sugars; the FDA counts them as added when used in processed products",
    }

    T["effective interval running"] = {
        "_cat": "everyday_life",
        "what": "a training method that alternates short bouts of fast running with recovery periods, used to improve VO2 max, lactate threshold, and running economy more efficiently than steady-state runs alone",
        "how": "common workouts include 5x800m at 5K race pace with 90-second jog recoveries, or 6-8x400m at mile pace with full recoveries. Intensity should bring heart rate near maximum during work intervals; recoveries should be active enough to keep the next interval honest. Build duration and intensity gradually to avoid injury",
        "why": "interval training delivers cardiovascular and metabolic adaptations that long slow distance cannot match in equivalent time, improving racing performance from 5K through marathon. It is also time-efficient for general fitness and improves insulin sensitivity and blood pressure",
        "vs": "intervals differ from tempo runs, which sustain near-threshold pace for 20-40 minutes, and from easy long runs, which build aerobic base. They differ from sprint work used in field sports, which is shorter and not necessarily aerobic",
        "ex": "a beginner 5K runner might progress from 4x400m at goal pace with full recovery to 5x1000m at goal 10K pace with 2-minute jogs over a 12-week build, with measurable race-pace improvement reflecting both aerobic and neuromuscular adaptations",
        "mis": "people think harder is always better. Most adaptation comes from consistent moderate intervals with adequate recovery; chronic redlining produces injury and burnout. Another myth is that intervals are only for fast runners; beginners benefit from age-and-fitness-appropriate intervals as much as elites",
    }

    T["how heat pumps heat homes"] = {
        "_cat": "everyday_life",
        "what": "electrically powered heating and cooling systems that transfer thermal energy from outdoor air or ground into a building rather than generating heat by combustion or resistance, achieving efficiencies of 200-400 percent",
        "how": "a refrigerant evaporates outdoors at low pressure, absorbing heat from the cold outside air or soil. A compressor raises the refrigerant's pressure and temperature; indoors the hot refrigerant condenses, releasing heat to the home, then expands and cools again. Reversing the cycle provides cooling in summer",
        "why": "heat pumps are a key technology for decarbonizing buildings, which produce a large share of fossil emissions. With low-carbon electricity they cut heating emissions sharply. Modern cold-climate units perform well to -25C, undermining the older claim that heat pumps fail in northern climates",
        "vs": "heat pumps differ from gas furnaces (combustion, single-purpose) and from electric resistance heating (one unit of electricity produces one unit of heat). They differ from air conditioners, which are essentially heat pumps running in only one direction",
        "ex": "Maine has rapidly adopted heat pumps, with hundreds of thousands installed; cold-climate models like Mitsubishi's hyper-heat units sustain rated capacity to -15F or lower, replacing oil furnaces in older homes",
        "mis": "people think heat pumps don't work in cold climates. Modern variable-speed cold-climate units do; what fails in cold weather is older single-stage equipment. Sizing, installation quality, and supplemental heat strategy matter more than the climate per se",
    }

    T["bicycle gearing basics"] = {
        "_cat": "everyday_life",
        "what": "the ratio of front chainring teeth to rear cog teeth on a bicycle, which determines how far the bike travels per pedal revolution and how much force is needed at the pedals for a given resistance",
        "how": "a higher ratio (large front, small rear) covers more ground per pedal stroke at the cost of higher pedal force, suited to flats and downhills. A lower ratio (small front, large rear) reduces pedal force at the cost of less distance per stroke, suited to climbs. Cadence near 80-100 rpm is generally efficient",
        "why": "appropriate gearing makes climbing possible without knee strain, lets riders maintain efficient cadence across terrain, and reduces fatigue on long rides. Modern wide-range cassettes (10-50 or more teeth) cover steep climbs and fast descents on a single rear cluster, especially with 1x drivetrains",
        "vs": "bicycle gearing differs from internal hub gears (Rohloff, Shimano Nexus), which are sealed and use planetary gears, and from electric motor assistance, which adds torque rather than changing ratios. Multi-speed derailleur drivetrains offer more range than fixed gears but more maintenance",
        "ex": "a road bike with a 50/34 compact crank and 11-32 cassette can climb steep gradients at low cadence and descend at high speed; the lowest gear (34/32) approaches a 1:1 ratio, similar to walking speed at moderate cadence",
        "mis": "people think more gears are always better. What matters is gear range and step size. A well-chosen 1x12 drivetrain may cover the same range as a 2x10 with simpler shifting, while a 3x9 might offer more total ratios but with redundancy across rings",
    }
