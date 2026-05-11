#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel bank part 6 of 6: final topup to reach 84+ topics (504+ seeds)
across 7 categories."""


def register(T):
    # ===== SCIENCE =====

    T["bioluminescence in nature"] = {
        "_cat": "science",
        "what": "the production and emission of light by living organisms through chemical reactions, typically involving the oxidation of a substrate called luciferin by an enzyme called luciferase, observed widely in marine life, fungi, and some insects",
        "how": "luciferase catalyzes the oxidation of luciferin, often in the presence of oxygen and ATP, releasing energy as visible-spectrum photons (commonly blue-green in marine organisms because that wavelength penetrates water best). Different organisms use different chemistries: fireflies use a unique luciferin-luciferase pair; dinoflagellates flash from membrane-potential changes; many fish host symbiotic bioluminescent bacteria",
        "why": "bioluminescence serves predation (anglerfish lures), defense (squid ink clouds, ostracod ejecta), communication (firefly mating signals), counter-illumination camouflage (matching downwelling light), and microbial signaling. It has been engineered into laboratory tools (luciferase reporters, GFP-tagged proteins) that revolutionized cell biology",
        "vs": "bioluminescence differs from fluorescence (which re-emits absorbed light, requiring an external excitation source) and from phosphorescence (delayed re-emission). Both fluorescence and bioluminescence appear in marine biology, sometimes in the same organism (corals, jellyfish), with quite different biological roles",
        "ex": "the bay of Vieques in Puerto Rico hosts dinoflagellate concentrations dense enough that boat wakes and swimmers' movements light up the water with blue-green sparkles. The same organisms produce the famous milky seas occasionally observed from satellites in the Indian Ocean",
        "mis": "people think bioluminescence is rare. About 80 percent of deep-sea animals are bioluminescent; it is one of the most common forms of communication on Earth by biomass. Another myth is that all glowing organisms use the same chemistry; over 30 distinct luciferin systems have evolved independently",
    }

    T["the water cycle"] = {
        "_cat": "science",
        "what": "the continuous movement of water through Earth's atmosphere, surface, and subsurface via evaporation, transpiration, condensation, precipitation, infiltration, runoff, and groundwater flow, redistributing freshwater across the planet",
        "how": "solar energy evaporates water from oceans, lakes, and soil; plants release water vapor through transpiration. Water vapor rises, cools, and condenses on aerosols to form clouds, releasing latent heat. Precipitation falls as rain, snow, or hail. Surface water flows in rivers to oceans; some infiltrates into aquifers; some is taken up again by plants. Phase changes redistribute energy globally, driving weather patterns",
        "why": "the water cycle determines freshwater availability, agricultural productivity, hydropower, drought and flood patterns, and the climate's distribution of heat. Climate change is intensifying it: warmer air holds more moisture (about 7 percent more per degree Celsius), making heavy precipitation events heavier and droughts drier in different regions",
        "vs": "the water cycle differs from carbon and nitrogen cycles (which involve different reservoirs and transformations) and from local irrigation cycles. It differs from the hydrological cycle on other planets: Titan has a methane cycle, while Mars likely had a water cycle in the past now mostly preserved in ice and atmosphere",
        "ex": "atmospheric rivers, narrow corridors of concentrated water vapor flowing from tropical to temperate regions, can carry water mass equivalent to 7 to 15 times the average flow of the Mississippi. Pacific atmospheric rivers regularly drench California, sometimes producing both relief from drought and catastrophic floods within days",
        "mis": "people think groundwater is a separate, isolated reservoir. Aquifers are connected to surface waters and recharge from infiltration; over-pumping depletes them and can cause subsidence. Another myth is that the cycle is closed and total water is constant; on long timescales, water is added by volcanism and lost to space, but on human timescales the total is effectively fixed",
    }

    # ===== MATH =====

    T["limits and continuity"] = {
        "_cat": "math",
        "what": "the foundational concepts of calculus describing how function values approach a target as inputs approach a point (limits) and when a function has no jumps, holes, or asymptotes there (continuity)",
        "how": "the limit of f(x) as x approaches a equals L if for every positive epsilon there is a positive delta such that whenever the distance from x to a is less than delta and x is not equal to a, the distance from f(x) to L is less than epsilon. A function is continuous at a if its limit there equals its value there. Continuity on intervals supports theorems like the intermediate value theorem and extreme value theorem",
        "why": "limits underpin derivatives (instantaneous rate of change as a limit of average rates), integrals (Riemann sums in the limit of fine partitions), infinite series (limits of partial sums), and the formal foundations of all of calculus. Continuity is the typical assumption for theorems guaranteeing solutions, fixed points, and optimal values exist",
        "vs": "limits differ from values: a limit can exist where the function is undefined (removable discontinuities), and a function can be defined where its limit does not exist (jumps). Continuity differs from differentiability (a stronger condition: continuous functions need not have derivatives, like the absolute value at zero)",
        "ex": "the function (sin x)/x has no value at x equal 0 (division by zero), but its limit is 1. This limit underlies the derivative of sine and the small-angle approximation used in physics and engineering. The function can be extended continuously by defining f(0) equal 1",
        "mis": "people think a limit means the function takes that value. A limit only describes the approach; the function may not even be defined at the limit point. Another myth is that all functions you write down are continuous; floor, ceiling, step, and many indicator functions have intentional discontinuities",
    }

    # ===== TECHNOLOGY =====

    T["how an operating system schedules disk I/O"] = {
        "_cat": "technology",
        "what": "the layer of an operating system that orders read and write requests to storage devices to balance latency, throughput, and fairness across processes, with strategies tuned to the underlying device's characteristics",
        "how": "for spinning disks, classic schedulers (CFQ, deadline, anticipatory, BFQ) merge and sort requests to reduce seek time and respect deadlines. For solid-state and NVMe drives, simpler schedulers (mq-deadline, none, BFQ) work better since random access is cheap and parallel queues matter. Linux moved its block layer to multi-queue (blk-mq) architecture to scale to NVMe devices with millions of IOPS",
        "why": "I/O scheduling can change application latency by orders of magnitude. Database servers, build systems, and interactive applications all benefit from appropriate schedulers; misconfiguration can starve critical workloads or deliver unpredictable tail latencies",
        "vs": "OS-level disk scheduling differs from application-level batching, from filesystem-level write coalescing, and from device firmware reordering. Each layer optimizes within its visibility, sometimes at cross purposes (a database that issues fsync may have its careful ordering undone if write barriers are not respected end to end)",
        "ex": "Linux's BFQ scheduler is designed for desktop interactivity, giving small foreground reads priority over background writes. Switching from BFQ to mq-deadline on a database server can improve throughput because the database orchestrates its own ordering and does not need user-perceived latency optimization",
        "mis": "people assume schedulers do not matter on SSDs. They matter less than on spinning disks but still affect tail latency, fairness across containers, and behavior under sustained pressure. Another myth is that NVMe needs no scheduler; many distributions ship with 'none' scheduling for NVMe, but the choice depends on workload",
    }

    T["how DNS over HTTPS (DoH) protects queries"] = {
        "_cat": "technology",
        "what": "a protocol that encrypts DNS queries inside HTTPS connections to prevent on-path observers and ISPs from seeing or modifying the names being resolved, standardized in RFC 8484 and widely deployed by browsers and operating systems since 2018",
        "how": "instead of sending DNS queries in plaintext over UDP port 53, the client opens an HTTPS connection (typically HTTP/2 or HTTP/3) to a configured DoH resolver and sends queries as HTTP requests. The resolver responds with DNS-over-HTTPS-formatted answers. The TLS layer authenticates the resolver and encrypts the traffic, blending DNS with regular web traffic to resist censorship",
        "why": "plaintext DNS exposes browsing patterns to anyone on the network, enables ISP-level injection (advertising replacement, censorship), and was a reliable target for surveillance. DoH closes that channel for users who configure it. It does shift trust to the chosen resolver (Cloudflare, Google, Quad9, Mozilla's TRR partners), which can still see all queries",
        "vs": "DoH differs from DNS over TLS (DoT, RFC 7858) which uses port 853 and is easier for network operators to identify and block or allow distinctly. DoH is harder to distinguish from web traffic. Both differ from plaintext DNS in confidentiality and integrity, although neither prevents the resolver from logging queries",
        "ex": "Firefox enabled DoH via Cloudflare's 1.1.1.1 in the US in 2020, the first major browser deployment. Chrome later added similar functionality, although most operating systems have not made DoH the default at the OS level. DNS-blocking based content filters can be evaded by per-application DoH unless the network blocks the DoH endpoints",
        "mis": "people think DoH prevents anyone from knowing what sites they visit. ISPs can still see the IP addresses of the servers connected to and (in many cases) the SNI in the TLS handshake. Another myth is that DoH is fundamentally different in security from DoT; the cryptographic guarantees are the same, only the framing differs",
    }

    # ===== HISTORY =====

    T["the Mexican Revolution"] = {
        "_cat": "history",
        "what": "the prolonged civil war and political upheaval in Mexico from 1910 to 1920 that overthrew the long dictatorship of Porfirio Diaz, produced the 1917 Constitution (one of the world's first to enshrine social rights), and reshaped Mexican political and economic life for the rest of the 20th century",
        "how": "Francisco I. Madero's call for democracy in 1910 sparked uprisings led by Pancho Villa in the north and Emiliano Zapata in the south demanding land reform. Madero's government was overthrown in 1913 by Victoriano Huerta's coup; Huerta was in turn defeated by a coalition of constitutionalists. Factional fighting continued throughout the 1910s. The 1917 Constitution, written under Carranza, established land redistribution, labor rights, secular education, and state ownership of subsoil resources",
        "why": "the Revolution killed an estimated 1-2 million people, displaced hundreds of thousands more, and produced a constitutional and political order that lasted essentially intact until the 1990s. Land reform redistributed millions of hectares; the PRI political machine grew out of the revolutionary parties. Mexican muralism (Rivera, Orozco, Siqueiros) emerged from revolutionary cultural projects",
        "vs": "the Mexican Revolution differs from the Russian Revolution (more fragmented, less ideologically unified, longer-lasting compromises) and from the earlier Mexican wars of independence (which addressed colonial relations rather than internal class structure). It differs from later Latin American revolutions (Cuban, Nicaraguan) in its agrarian-populist orientation rather than explicitly socialist program",
        "ex": "the corrido 'La Cucaracha,' often associated with revolutionary soldiers, was sung in many regional variants by both Villa's and government forces, sometimes with verses mocking specific leaders. The persistence of revolutionary corridos in Mexican popular music shows how thoroughly the conflict shaped national identity",
        "mis": "people think Pancho Villa was the dominant revolutionary leader. He commanded important northern forces but was defeated in 1915 by Obregon at the Battle of Celaya; Carranza and later Obregon shaped the post-revolutionary state. Another myth is that the Revolution permanently ended dictatorship; the PRI ruled essentially uninterrupted from 1929 to 2000, with strong authoritarian features",
    }

    # ===== ARTS AND HUMANITIES =====

    T["the development of opera"] = {
        "_cat": "arts_and_humanities",
        "what": "a Western theatrical art form combining sung text, orchestral music, scenery, costume, and often dance, originating in late-16th-century Florence and evolving through Baroque, Classical, Romantic, and Modern phases into a global tradition",
        "how": "members of the Florentine Camerata around 1600 sought to revive Greek drama by setting Italian text in a flexible vocal style (recitative) over instrumental accompaniment. Monteverdi's L'Orfeo (1607) established the form. Through the 17th-19th centuries, opera split into national traditions (Italian, French, German), forms (opera seria, opera buffa, grand opera, music drama), and styles culminating in Verdi, Wagner, and Puccini. The 20th century brought Strauss, Berg, Britten, and modernist experiments",
        "why": "opera shaped the development of orchestral music, vocal training, theatrical staging, and the cultural status of public performance for four centuries. It was for much of European history the most prestigious and expensive performing art, and contemporary opera continues to commission new works that engage current themes through one of music's most demanding forms",
        "vs": "opera differs from oratorio (concert sacred works without staging), from operetta and musical theater (more spoken dialogue, lighter idiom), from cantata (smaller scale, often non-narrative), and from sung-through musicals like Les Miserables, which adapt operatic continuous-music techniques but with popular-music idioms",
        "ex": "Wagner's Ring cycle, four operas totaling about 15 hours of music, expanded the genre beyond previous limits in length, orchestral scale, and continuous music drama. The Bayreuth Festspielhaus was custom-built in 1876 for the Ring's premiere and remains the only major venue dedicated to a single composer's works",
        "mis": "people think opera is uniformly tragic and serious. Comic opera (opera buffa, opera comique, Singspiel, Gilbert and Sullivan) has been a major branch since the 18th century. Another myth is that operas always require fluency in their original language to enjoy; surtitles, recordings, and a strong dramatic-musical surface make most repertoire accessible to non-speakers",
    }

    # ===== EVERYDAY_LIFE =====

    T["how to interpret weather forecasts"] = {
        "_cat": "everyday_life",
        "what": "the skill of reading meteorological forecasts (probability of precipitation, temperature ranges, wind speeds, severe weather warnings) accurately enough to make practical decisions about activities, clothing, and travel",
        "how": "modern forecasts come from numerical weather prediction models run by national agencies (NOAA, ECMWF) and post-processed for consumer apps. Probability of precipitation is the chance that measurable precipitation falls at a point; it is not the percentage of the area that gets rain or the percentage of the time it rains. Severe weather warnings categorize hazards (watch, warning, advisory) by probability and immediacy",
        "why": "forecast literacy improves real decisions: whether to bring an umbrella, evacuate ahead of a hurricane, plant frost-sensitive crops, or schedule outdoor events. Misreading 'a 30 percent chance of rain' as 'mostly dry' or 'mostly wet' wastes plans either way; understanding model uncertainty supports better choices",
        "vs": "professional forecasts differ from app summaries (which compress hourly model output into one icon) and from raw model output (which requires expert interpretation). Probabilistic forecasts (ensembles) differ from deterministic ones in carrying explicit uncertainty. Long-range forecasts (beyond 7-10 days) differ in skill from short-range; useful skill drops sharply past about a week",
        "ex": "during Hurricane Sandy in 2012, the European model (ECMWF) consistently and correctly forecast a hard left turn into the US East Coast nearly a week ahead, while the American GFS lagged. The forecast skill difference saved many lives and prompted significant US investment in modeling capacity afterward",
        "mis": "people think a 30 percent chance of rain means it will rain 30 percent of the day. It means there is a 30 percent chance of measurable rain at any given point in the forecast area. Another myth is that the forecast is wrong if it does not match what happened; with probabilistic forecasts, the right way to evaluate skill is over many predictions, not one outcome",
    }

    # ===== CURRENT_EVENTS =====

    T["the post-pandemic supply chain shifts"] = {
        "_cat": "current_events",
        "what": "the restructuring of global manufacturing and logistics since 2020, driven by pandemic disruption, geopolitical tension between the US and China, and policy efforts to reshore or 'friend-shore' critical industries like semiconductors, batteries, and pharmaceuticals",
        "how": "COVID-era port closures, container shortages, and demand whiplash exposed the fragility of just-in-time global supply chains. Companies have added safety stock, diversified suppliers (the 'China plus one' strategy moving production to Vietnam, India, and Mexico), and accepted modest cost increases for resilience. Governments have layered subsidies (the US CHIPS Act, the EU Chips Act) and tariffs to redirect investment",
        "why": "the shifts are remaking trade patterns built up since the 1990s. Mexico has displaced China as the largest US trading partner since 2023. New US semiconductor plants in Arizona, Ohio, and Texas represent the largest such investment in decades. The reorientation will play out for a decade or more and shape inflation, employment patterns, and geopolitical alignments",
        "vs": "the post-pandemic shifts differ from earlier offshoring waves (which optimized for cost), from full reshoring (rare due to cost gaps), and from the 1970s-1980s nationalist trade tensions (which were less technology-focused). They differ from autarky in that allied diversification, not isolation, is the typical goal",
        "ex": "TSMC's Arizona fabs, planned to produce advanced semiconductors with up to 65 billion USD of investment, illustrate the new pattern: a Taiwanese firm building flagship capacity in the US under significant subsidies, with concerns about American workforce capabilities, costs, and timelines all openly debated. Production has begun, but ramping to full output is slower than originally planned",
        "mis": "people think globalization is reversing wholesale. Total trade volumes remain at historic highs; the change is in patterns and concentrations rather than in aggregate openness. Another myth is that reshoring will dramatically grow manufacturing employment; modern fabs and factories use far less labor per output than 20th-century plants, so direct employment effects are modest",
    }

    T["the housing affordability strain in major cities"] = {
        "_cat": "current_events",
        "what": "the persistent and worsening gap between housing prices or rents and median incomes in major metropolitan areas in the US, Canada, UK, Australia, and beyond, driving political debates about zoning, taxation, supply, and demand",
        "how": "constrained housing supply (from zoning, environmental review, and slow permitting) collides with rising demand from population growth, rising real incomes among the top quintile, and (in some cities) investment demand. Mortgage rate swings (low rates inflated prices in 2020-21; high rates after 2022 pushed payments up further). Rents in many cities have climbed 30-50 percent since 2019. Homelessness has risen in tandem in several US west-coast cities",
        "why": "housing affordability shapes inequality, mobility, family formation, and political alignment. Young adults priced out of home ownership delay milestones; high rents extract income from working households; concentrated wealth in housing markets benefits incumbents over newcomers. Reform of zoning and entitlement rules (Auckland, California, Minneapolis) is being tested as a supply response",
        "vs": "the current strain differs from past cycles (1970s, 2000s) in being driven more by chronic underbuilding than speculative bubble (although bubbles exist regionally). It differs across countries: zoning is the dominant constraint in the US, while the UK's planning system and Australia's state-level controls produce similar outcomes through different routes",
        "ex": "Auckland, New Zealand undertook substantial upzoning in 2016, allowing more dense and missing-middle housing across the city. Subsequent studies attribute about a 20-30 percent reduction in rents in upzoned areas relative to control areas, providing one of the cleanest empirical tests of supply-side reforms",
        "mis": "people think only foreign buyers or institutional investors drive the problem. They contribute to specific markets (Vancouver, Sydney, parts of London) but the dominant constraint in most metro areas is local zoning that prevents most residents from building duplexes, ADUs, or apartments. Another myth is that building more housing always raises rents; the empirical evidence consistently shows new supply lowers rents in nearby submarkets and citywide",
    }

    T["the rising prevalence of remote and hybrid work"] = {
        "_cat": "current_events",
        "what": "the lasting shift in how knowledge workers spend their workweek since 2020, with hybrid (some days at home, some in office) becoming the dominant arrangement for many office jobs and fully remote remaining a sizeable minority, affecting commercial real estate, cities, and labor markets",
        "how": "the pandemic forced a mass experiment in remote work that revealed productivity in many roles was at least preserved, sometimes increased. Workers strongly preferred at least partial remote arrangements, especially for caregiving, commute, and focus reasons. Companies that mandated full return saw turnover spike. The new equilibrium has settled around 2-3 days in office for hybrid roles, with full-remote roles concentrated in tech, finance, and certain professional services",
        "why": "the shift has reshaped commercial real estate (office vacancies, suburban revival), commuting and transit ridership, household relocation patterns (away from highest-cost cities), labor markets (geographic talent pools widening), and management practice (more written, asynchronous, outcome-focused work). It is one of the largest behavioral changes of the post-pandemic era",
        "vs": "current hybrid work differs from pre-pandemic remote work (rare, often part-time, stigmatized) and from full remote (more common in tech and certain freelance roles, currently a minority of jobs). It differs from telecommuting concepts of the 1990s in being supported by mature collaboration tools (video, chat, document collaboration) and broad cultural acceptance",
        "ex": "Stanford economist Nick Bloom's surveys show that as of 2024, full-time work-from-home accounts for about 25 percent of paid workdays in the US, hybrid for another 30 percent. Pre-pandemic, the fully-remote share was under 5 percent. Productivity studies show neutral-to-positive effects for hybrid arrangements on most outcome measures",
        "mis": "people think remote work is universally less productive. The evidence is mixed and depends on task type, team practices, and worker preferences; well-managed hybrid often outperforms full in-office for output and retention. Another myth is that return-to-office mandates are driven by data; many seem driven by managerial preference and commercial-real-estate concerns rather than measured productivity differences",
    }

    T["the global mental health crisis among young adults"] = {
        "_cat": "current_events",
        "what": "the documented rise in rates of anxiety, depression, self-harm, and suicide among adolescents and young adults in many high-income countries since around 2010-2012, particularly pronounced among teenage girls and concentrated in English-speaking countries",
        "how": "self-reported anxiety and depression among US adolescents roughly doubled between 2010 and 2020. ER visits for self-harm in teen girls roughly tripled. Similar but smaller rises occurred in the UK, Canada, Australia, and Nordic countries. The timing strongly correlates with the rapid spread of smartphones and social media. Other contributing factors discussed include economic uncertainty, climate anxiety, post-pandemic disruption, and changing family structures",
        "why": "the crisis is straining mental-health care systems, schools, and families. It is reshaping policies on phone use, social media age verification, school counseling, and clinical practice. The long-term effects on a generation's life trajectories, employment, and family formation are still being studied",
        "vs": "the current rise differs from earlier youth mental-health concerns (more localized, less measurable) and from the broader population: middle-aged and older adults have not seen comparable rises in most measures. It differs across countries: many non-English-speaking countries with high social-media use show smaller rises, suggesting cultural and platform factors interact",
        "ex": "Jonathan Haidt's analysis of US, UK, Canadian, and Australian data shows simultaneous, similar inflection points across countries between 2010 and 2014, particularly for girls. Some schools, regions, and entire countries (notably France) have begun restricting smartphone use during school hours in response, with early evaluations underway",
        "mis": "people think the rise is purely an artifact of better awareness and willingness to report. Multiple objective indicators (ER visits, hospitalizations, suicide rates) move together with self-report, making pure measurement-bias explanations insufficient. Another myth is that all blame can be assigned to social media; it is the most plausible single contributor by current evidence, but other factors compound it",
    }
