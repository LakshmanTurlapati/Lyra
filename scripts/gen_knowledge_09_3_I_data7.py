#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel bank: more everyday_life and current_events topics (fresh for batch I)."""


def register(T):
    # More everyday life
    T["how to choose running shoes"] = {
        "_cat": "everyday_life",
        "what": "selecting footwear suited to one's foot shape, gait, mileage, and surface, considering cushioning, drop, stability features, and fit rather than brand reputation alone",
        "how": "fit matters most: a thumb's width of room at the toe, no heel slip, no pressure points. Drop (heel-to-toe height difference) of 4-12 mm trades calf load for ankle/knee load. Cushioning depends on body weight and mileage. Stability shoes help only those with significant overpronation; for most runners, neutral shoes work fine",
        "why": "appropriate shoes reduce injury risk and make training feel sustainable. Repeated impact at typical running cadence (170-180 steps per minute, 1500-2000 steps per mile) means small fit problems compound into pain. Replacing shoes every 300-500 miles maintains cushioning and structure",
        "vs": "running shoes differ from cross-trainers, which support lateral motion at the cost of forward cushioning. Trail shoes have more aggressive tread and rock plates. Maximalist shoes (Hoka) emphasize cushioning; minimalist shoes (Vibram) emphasize ground feel; carbon-plated racers add propulsion",
        "ex": "a runner with mild overpronation, average weight, training for marathon at 40 miles per week typically does well in a neutral or mild-stability daily trainer (Brooks Ghost, Saucony Ride, Nike Pegasus) with a separate long-run shoe and tempo shoe to vary stresses",
        "mis": "people think they need their gait analyzed in a store. Casual gait analysis often misclassifies; for most runners, comfort prediction outperforms motion-control prescription. Another myth is that more cushioning prevents injury; evidence is mixed, and overcushioning can mask form issues",
    }

    T["coffee brewing methods"] = {
        "_cat": "everyday_life",
        "what": "the techniques for extracting flavor from ground coffee, including drip, espresso, French press, pour-over, AeroPress, cold brew, and moka pot, each with distinct ratios, temperatures, grind sizes, and contact times",
        "how": "espresso forces hot water through fine grounds at 9 bar in 25-30 seconds. Pour-over slowly drips 200F water through medium grounds for 3-4 minutes. French press steeps coarse grounds for 4 minutes then plunges. Cold brew steeps coarse grounds in cold water for 12-24 hours. Each combination of grind size, temperature, ratio, and time produces a different cup",
        "why": "method choice shapes the cup's body, acidity, sweetness, and bitterness more than bean origin alone. Mastering one method well typically produces better coffee than chasing equipment; the same beans can taste delicious or muddy depending on technique",
        "vs": "espresso differs from drip in pressure, contact time, and grind, producing concentrated body. Cold brew differs from iced pour-over in extracting different acid compounds, producing smoother, less acidic coffee. Capsule machines trade control for convenience",
        "ex": "for a French press, use a 1:15 ratio (about 30 g coffee to 450 g water), coarse grind, 4 minutes steep, then plunge slowly. The same beans in a pour-over with a finer grind and faster contact time will taste cleaner and brighter",
        "mis": "people think dark roast has more caffeine. Light roast actually has slightly more by volume because dark roasts lose mass during longer roasting. Another myth is that boiling water makes the best coffee; near 200F (just below boiling) is optimal for most methods",
    }

    T["sourdough bread fermentation"] = {
        "_cat": "everyday_life",
        "what": "the leavening of bread by wild yeast and lactic acid bacteria living in a sourdough starter, producing the bread's characteristic tang, open crumb, and longer shelf life compared to commercial yeast bread",
        "how": "a starter is a stable culture of yeasts and bacteria fed regularly with flour and water. To bake, the baker mixes starter into flour and water, lets it autolyse, adds salt, and folds the dough at intervals to develop gluten. Bulk fermentation builds flavor; shaping and cold proof in the fridge develop crust and crumb. The dough is baked at high heat in a covered Dutch oven or with steam",
        "why": "sourdough produces complex flavor without commercial yeast, requires only flour, water, and salt, and the fermentation slightly improves digestibility and lowers glycemic response. The process became a cultural touchstone during the 2020 pandemic when many home bakers embraced it",
        "vs": "sourdough differs from commercial yeast breads, which use selected Saccharomyces cerevisiae strains and ferment faster with less acid. It differs from quick breads, which are leavened by baking soda or powder, and from levain, which is similar but in French tradition",
        "ex": "a typical schedule: feed starter the night before, mix dough next morning, bulk ferment 4-6 hours with folds every 30-45 minutes, shape and cold-proof overnight, bake from cold in a 500F Dutch oven covered for 20 minutes then uncovered for 25 more. Total elapsed time around 24 hours, but active work is perhaps 30 minutes",
        "mis": "people think sourdough is gluten-free or safe for celiacs. It still contains gluten, although fermentation may slightly reduce certain peptide exposure. Another myth is that starters are alive in some special way; they are simply microbial cultures that can be paused (refrigerated) and revived",
    }

    T["budgeting with the 50-30-20 rule"] = {
        "_cat": "everyday_life",
        "what": "a personal-budgeting framework popularized by Senator Elizabeth Warren that allocates after-tax income roughly 50 percent to needs, 30 percent to wants, and 20 percent to savings and debt repayment beyond minimums",
        "how": "categorize expenses: needs include housing, utilities, groceries, insurance, transportation to work, minimum debt payments. Wants include dining out, subscriptions, hobbies, travel. Savings includes emergency fund, retirement contributions, extra debt principal. Adjust the splits to fit your situation but maintain explicit categories",
        "why": "the rule provides a simple sanity check: if needs exceed 50 percent of income, the household has structural pressure (rent too high relative to income, transportation costs heavy). Saving 20 percent consistently is a strong long-term wealth-building rate. Most people who track find wants larger than they assumed",
        "vs": "the 50-30-20 rule differs from envelope budgeting (cash divided into category envelopes), zero-based budgeting (every dollar assigned), and YNAB-style methods (assign jobs to dollars). It is simpler than alternatives but less precise; it works as a starting heuristic",
        "ex": "on a 6000 dollar monthly take-home, the rule allocates 3000 to needs, 1800 to wants, and 1200 to savings. If rent and food are 3500, the rule signals that housing is high relative to income; either income needs to grow or housing or other needs must shrink for the framework to work",
        "mis": "people think the 50 percent for needs includes minimum debt payments only. Including all rent, utilities, insurance, transit, basic groceries, and minimum debt service often pushes housing-burdened households well over 50 percent. Another myth is that the rule prescribes exact percentages; it is a heuristic to be adapted",
    }

    T["composting at home"] = {
        "_cat": "everyday_life",
        "what": "the controlled decomposition of organic kitchen and yard waste into a stable, soil-improving amendment, accomplished in piles, bins, tumblers, or worm composters in a backyard or even an apartment",
        "how": "balance carbon-rich 'browns' (dry leaves, paper, straw) with nitrogen-rich 'greens' (vegetable scraps, coffee grounds, fresh grass) at roughly 3-to-1 by volume. Maintain moisture like a wrung-out sponge, turn for oxygen, and microbes do the work. Hot piles reach 130-160F and finish in weeks; cold piles take months but require less attention",
        "why": "composting diverts food and yard waste from landfills, where it produces methane, and turns it into a soil amendment that improves structure, water retention, and microbial life. Roughly 30 percent of household waste by weight in many areas is compostable, and home composting closes a small but meaningful local loop",
        "vs": "home composting differs from municipal composting, which can handle meat and dairy at higher temperatures. Vermicomposting (worm bins) handles smaller volumes year-round indoors. Bokashi fermentation pre-treats food waste anaerobically and is well-suited to small spaces",
        "ex": "a basic backyard system: a wire bin or pallet structure, alternating layers of leaves and kitchen scraps, occasional turning. Within 6-12 months, you have dark crumbly compost suitable for amending garden beds and topdressing lawns",
        "mis": "people think compost smells. A balanced pile smells earthy; foul smells signal too much nitrogen, too little oxygen, or excess moisture. Another myth is that you need a yard; vermicomposting in a small bin handles a typical apartment's vegetable scraps without odor",
    }

    T["everyday cybersecurity hygiene"] = {
        "_cat": "everyday_life",
        "what": "basic personal security practices including using a password manager, enabling multi-factor authentication, keeping software updated, recognizing phishing, and limiting data shared with apps and sites",
        "how": "install a reputable password manager (1Password, Bitwarden) and use unique strong passwords for every account. Enable MFA, preferring authenticator apps or hardware keys over SMS. Update operating systems and browsers promptly. Hover over links before clicking, verify sender identities, and avoid sharing credentials in response to messages",
        "why": "most personal account takeovers result from credential reuse and phishing, not zero-days. Basic hygiene defeats the bulk of attacks. The financial and privacy consequences of account compromise (banking, email, social media) are large, and email account takeover often cascades into others via password resets",
        "vs": "good hygiene differs from the now-discouraged advice to memorize complex passwords and rotate them regularly, which leads to weak password reuse. It differs from heavy-duty operational security used by journalists or activists, which involves more rigorous threat modeling and tools",
        "ex": "the 2019 Capital One breach exposed 100 million records via a misconfigured firewall and SSRF, illustrating the institutional side. On the personal side, a stolen password to one site, reused across many, is the most common path to compromise; a password manager eliminates that vector",
        "mis": "people think they're not interesting targets. Most attacks are opportunistic and automated, hitting any account with weak credentials. Another myth is that incognito mode is private; it hides browsing locally but does nothing against your ISP, websites, or network attackers",
    }

    T["how to read nutrition labels for protein"] = {
        "_cat": "everyday_life",
        "what": "interpreting protein content on Nutrition Facts panels in light of serving size, daily targets, and quality (amino acid profile and digestibility), to make informed choices for muscle maintenance, satiety, and recovery",
        "how": "check protein grams per serving and the realistic serving size you'll eat. Total daily targets for healthy adults are typically 1.2-2.0 g per kg body weight, distributed across meals. Animal proteins are generally complete and highly digestible; plant proteins benefit from variety to cover essential amino acids",
        "why": "adequate protein supports muscle preservation in aging, recovery from exercise, and satiety in weight management. The Daily Value listed (50 g) is a minimum, not optimal, and most active or older adults benefit from substantially more, distributed in 25-40 g portions per meal",
        "vs": "protein differs in quality (DIAA score, biological value): whey > eggs > beef > soy > most other plants, although well-mixed plant diets achieve completeness. Carbohydrates and fats are rougher counts; protein is among the most useful labeling lines for active people",
        "ex": "a 6-ounce chicken breast has about 50 g protein; a cup of Greek yogurt about 17-22 g; a tablespoon of peanut butter about 4 g. Three meals at 30 g protein plus one snack at 20 g hits a 110 g daily total that suits many adults",
        "mis": "people think more protein damages kidneys. Healthy kidneys handle high protein intake well; the warning applies to people with existing kidney disease. Another myth is that protein timing requires post-workout windows; total daily intake matters more, with reasonable distribution across meals",
    }

    # Current events / contemporary
    T["large language models in education"] = {
        "_cat": "current_events",
        "what": "the integration of LLMs like GPT-4, Claude, and Gemini into education for tutoring, content generation, summarization, and feedback, raising questions about academic integrity, equity, and pedagogy",
        "how": "students use LLMs to draft essays, debug code, explain concepts, and study; teachers use them to generate quizzes, lesson plans, and personalized feedback. Schools and universities are rewriting honor codes, redesigning assessments toward in-class and oral formats, and integrating tools like Khanmigo for one-on-one tutoring at scale",
        "why": "LLMs are reshaping how students access expertise. Personalized tutoring previously cost hundreds of dollars an hour; now it is available through subscriptions or free tiers. Equitable access could close achievement gaps but unequal access risks widening them. The pedagogy must adapt or risk obsolescence",
        "vs": "LLM tutoring differs from earlier ed-tech tools (Khan Academy videos, adaptive quizzes) by being conversational and open-ended. It differs from human tutoring by lacking sustained relationships and embodied attention but offering availability and patience. It differs from search engines by synthesizing rather than linking",
        "ex": "Khan Academy's Khanmigo, built on GPT, tutors students through Socratic questioning rather than direct answers. Pilot programs at universities have integrated AI feedback on writing drafts, with mixed results: useful for surface revision, weaker for deeper argument",
        "mis": "people think LLMs replace teachers. They augment instruction but lack the relational, motivational, and curriculum-design work teachers provide. Another myth is that detection tools reliably catch AI-written work; current detectors have high false-positive rates and unreliable accuracy",
    }

    T["the energy transition and grid storage"] = {
        "_cat": "current_events",
        "what": "the global shift from fossil fuel-based electricity to wind, solar, and other low-carbon sources, with grid-scale storage emerging as a critical enabler to handle the variability of renewables",
        "how": "lithium-ion battery costs fell 90 percent over a decade, making 4-hour grid storage economically competitive in many markets. Pumped hydro remains the largest stored energy by volume. Emerging technologies include iron-air, sodium-ion, flow batteries for longer durations, and green hydrogen for very long storage. Demand response, transmission, and inverter-based controls round out the toolkit",
        "why": "decarbonizing the power sector is the largest near-term lever for cutting emissions, and storage handles the residual variability problem that critics use to dismiss renewables. As wind and solar penetrations rise past 30-40 percent, storage and flexibility become increasingly economic, not just technical, requirements",
        "vs": "lithium-ion grid storage differs from electric vehicle batteries (similar chemistry, different duty cycle), from pumped hydro (longer duration, geography-dependent), and from green hydrogen (very long duration, lower round-trip efficiency). No single technology covers all timescales",
        "ex": "California's Moss Landing battery facility, expanded to 750 MW, regularly shifts midday solar to evening peak, and has prevented blackouts during heat waves. Australia's Hornsdale Power Reserve famously paid back its cost in two years through grid services",
        "mis": "people think renewables are too intermittent for the grid. Variability is real but manageable up to high penetrations with current technology; the bigger constraints are transmission, permitting, and supply chain rather than physical feasibility. Another myth is that we need new storage breakthroughs first; deployment of existing technology is the dominant near-term need",
    }

    T["GLP-1 weight loss drugs"] = {
        "_cat": "current_events",
        "what": "a class of medications including semaglutide (Ozempic, Wegovy) and tirzepatide (Mounjaro, Zepbound) that mimic gut hormones to reduce appetite and slow gastric emptying, producing 15-22 percent average weight loss in trials",
        "how": "GLP-1 receptor agonists bind to receptors in the brain that regulate appetite, slowing gastric emptying and increasing satiety. Tirzepatide is dual-agonist for GLP-1 and GIP receptors with stronger effects. Patients typically inject weekly; common side effects include nausea, sometimes severe at dose escalation. Discontinuation often leads to weight regain",
        "why": "after decades of failed pharmacological obesity treatments, GLP-1s deliver weight loss approaching bariatric surgery. They are reshaping cardiology (showing CV benefit independent of weight), economics (drug spending, insurance coverage debates), and culture. Demand outstrips supply; the drugs are listed on the WHO essential medicines list",
        "vs": "GLP-1 drugs differ from older obesity medications (orlistat, phentermine) in efficacy and tolerability. They differ from bariatric surgery in being reversible but requiring continued use. They differ from lifestyle interventions, which produce smaller average losses and are harder to maintain at population scale",
        "ex": "the SELECT trial showed semaglutide reduced major adverse cardiovascular events by 20 percent in patients with obesity and CV disease, independent of diabetes status; this is reshaping cardiology guidelines and is a basis for broad insurance coverage arguments",
        "mis": "people think the drugs are simply appetite suppressants like older diet pills. They modify hormonal signaling that the body uses normally and have systemic effects beyond weight, including improved blood sugar and cardiovascular outcomes. Another myth is that they enable lifelong cure; weight typically rebounds when discontinued, suggesting chronic treatment for chronic disease",
    }

    T["mRNA vaccines beyond COVID"] = {
        "_cat": "current_events",
        "what": "vaccine platforms that deliver synthetic messenger RNA encoding a target antigen, used at scale first against SARS-CoV-2 and now in clinical trials for cancer, RSV, flu, malaria, and other infectious and oncologic targets",
        "how": "mRNA encoding the antigen is encapsulated in lipid nanoparticles that deliver it to cells, where ribosomes translate it into protein. The body's immune system recognizes the protein and mounts both antibody and T-cell responses. The platform allows rapid sequence updates without re-engineering manufacturing",
        "why": "mRNA platforms enable rapid response to new pathogens (the COVID-19 vaccines moved from sequence to clinic in under a year), personalized cancer vaccines (encoding tumor-specific neoantigens), and combinations against multiple targets in a single shot. The technology promises faster, more flexible vaccine development across many domains",
        "vs": "mRNA vaccines differ from inactivated and subunit vaccines, which deliver the antigen directly. They differ from viral vector vaccines, which use a modified virus to deliver DNA encoding the antigen. They differ from DNA vaccines by acting in the cytoplasm without entering the nucleus",
        "ex": "Moderna's individualized neoantigen therapy combined with pembrolizumab showed roughly 50 percent reduction in melanoma recurrence in a phase II trial. mRNA flu and RSV vaccines are now in late-stage clinical development; combination COVID-flu shots are approaching authorization in some markets",
        "mis": "people think mRNA changes DNA. The mRNA stays in the cytoplasm and is degraded within hours to days; it does not enter the nucleus or alter the genome. Another myth is that mRNA technology was rushed into use; the platform was developed and tested for over a decade before COVID",
    }

    T["AI alignment and safety research"] = {
        "_cat": "current_events",
        "what": "the field studying how to ensure that increasingly capable AI systems behave in ways aligned with human intent, including techniques for evaluation, interpretability, robustness, and governance",
        "how": "techniques include reinforcement learning from human feedback (RLHF) to shape model behavior, constitutional AI approaches that let models critique and revise their own outputs, evaluation suites for measuring deception and harm, mechanistic interpretability that reverse-engineers neural networks, and red-teaming to surface failure modes before deployment",
        "why": "as AI systems are deployed in higher-stakes domains, misalignment can cause real harm. The field combines machine learning research, philosophy, and policy. Major labs (Anthropic, OpenAI, DeepMind) have alignment teams; governments are establishing AI safety institutes (UK AISI, US AISI, EU AI Office)",
        "vs": "alignment differs from traditional AI ethics, which focuses on bias, fairness, and transparency; alignment also addresses long-term risks from advanced systems. It differs from AI capabilities research, which makes systems more powerful, although the two interact. It differs from cybersecurity, which focuses on adversarial defense rather than goal correctness",
        "ex": "Anthropic's constitutional AI uses a written set of principles to guide model behavior, with the model itself critiquing outputs and being trained on the resulting revisions. Mechanistic interpretability papers from 2024 located concepts like 'truthfulness' and 'helpfulness' as identifiable directions in neural network activations",
        "mis": "people think alignment is about giving AI moral instincts. It is about producing measurable, robust behavior consistent with intent, with evaluation and verification at the center. Another myth is that alignment is solved; current techniques are imperfect and degrade as capabilities increase",
    }

    T["the housing affordability crisis"] = {
        "_cat": "current_events",
        "what": "the rising gap between housing costs and incomes in many high-demand cities and entire countries, driven by restrictive zoning, slow construction, financialization of housing, demographic shifts, and interest rate effects",
        "how": "decades of underbuilding in supply-constrained markets, combined with population growth and rising real incomes for some, push prices up. Single-family zoning bans missing-middle housing in much of urban North America. Construction productivity has grown more slowly than other sectors. Interest rates affect purchase affordability and rent supply via developer financing",
        "why": "housing costs eat increasing shares of household income, force long commutes, slow geographic mobility (and thus economic mobility), and depress fertility rates. The crisis is now central to policy debates in the US, Canada, UK, Australia, and many European countries, intersecting with politics around immigration, transit, and inequality",
        "vs": "the current crisis differs from past housing booms (1980s, 2000s) by being driven more by supply constraints than speculative demand. It differs from rural or shrinking-city contexts where housing is cheap but jobs are scarce. It differs from purely homeless-population issues, which are at the extreme end of the spectrum",
        "ex": "Auckland's 2016 zoning reform allowing more housing in much of the city correlated with significantly slower rent growth than peer cities; California's recent ADU and single-family upzoning produced thousands of new units in markets long resistant. These cases inform the 'YIMBY' policy movement",
        "mis": "people think investors caused the crisis. Investors respond to underlying scarcity rather than create it; tightening investor purchases without expanding supply mostly redistributes the same scarce stock. Another myth is that we cannot build our way out; comparative international evidence shows that sustained construction at scale does flatten or reduce real housing costs",
    }
