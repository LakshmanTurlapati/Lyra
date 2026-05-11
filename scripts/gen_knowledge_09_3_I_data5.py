#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel bank: history topics (fresh for batch I)."""


def register(T):
    T["the Treaty of Westphalia"] = {
        "_cat": "history",
        "what": "the 1648 series of treaties ending the Thirty Years' War in the Holy Roman Empire and the Eighty Years' War between Spain and the Dutch Republic, often credited as the foundation of the modern state system based on territorial sovereignty",
        "how": "negotiations took years across Münster and Osnabrück, producing settlements that affirmed each prince's right to determine the religion of their territory (cuius regio, eius religio extended), recognized Dutch and Swiss independence, and redrew German borders. The Holy Roman Empire's central authority weakened in favor of constituent states",
        "why": "Westphalia is the conventional starting point for the principle of state sovereignty in international relations. The bloody religious wars taught Europe that interfaith coexistence was preferable to ongoing slaughter, and the framework of equal sovereign states became the template for diplomacy",
        "vs": "Westphalia differs from medieval feudal arrangements with overlapping authorities (church, emperor, lord), and from later imperial orders. It differs from the Congress of Vienna (1815), which restored monarchies after Napoleon, and from the UN-era system, which limits sovereignty in human-rights cases",
        "ex": "the Dutch Republic, having effectively been independent since the late 16th century, gained formal recognition. The Swiss Confederation similarly exited the Empire. Both became models of small states surviving by skilled diplomacy in the new system",
        "mis": "people think Westphalia invented the nation-state. It codified existing trends but did not bring nationalism, which is a 19th-century phenomenon. Sovereign states under Westphalia were dynastic and multilingual, not nationalist",
    }

    T["the Mongol Empire's postal system"] = {
        "_cat": "history",
        "what": "the Yam, the relay network of horse stations that crossed the Mongol Empire from Korea to Hungary, allowing messages, officials, and goods to traverse Eurasia at speeds unmatched until the 19th century",
        "how": "stations spaced roughly a day's ride apart provided fresh horses, food, and shelter to authorized travelers carrying paiza tablets. Couriers wore bells to clear the way and could ride day and night by switching mounts, covering up to 200 miles a day. Local populations supported the system as a state obligation",
        "why": "the Yam knit a continent-spanning empire together. It enabled rapid military coordination, trade, and information flow that made the Mongol Empire administratively functional despite its vast territory. It influenced postal systems from China to Persia and impressed travelers like Marco Polo",
        "vs": "the Yam differed from Roman cursus publicus and Persian Royal Road in scale and integration; it spanned more territory and connected more cultures. It differed from later European postal systems, which evolved from private merchant networks rather than state-imposed relays",
        "ex": "Marco Polo describes 10000 yam stations across Yuan China alone, with several hundred thousand horses serving messengers. The fastest news from frontiers reached the capital in days where caravan travel would take months",
        "mis": "people think the Mongols were purely destructive. They built sophisticated administrative infrastructure that persisted for centuries; the Yam outlived the unified empire and its descendants influenced Russian and Chinese postal systems",
    }

    T["the Apollo program"] = {
        "_cat": "history",
        "what": "the United States human spaceflight program that landed astronauts on the Moon between 1969 and 1972, comprising 17 missions and consuming about 4 percent of the federal budget at peak in service of a Cold War political goal",
        "how": "Saturn V rockets launched a Command Module, Service Module, and Lunar Module to lunar orbit. The LM landed two astronauts on the surface while the third orbited; ascent stages returned them to lunar orbit, after which the CM/SM crew returned to Earth. Six landings deposited 12 astronauts and returned 382 kg of lunar samples",
        "why": "Apollo demonstrated that very large engineering programs could solve novel hard problems on a deadline, drove advances in computers, materials, and systems engineering, and remains the only crewed lunar exploration to date. Lunar samples reshaped planetary science, and the program's prestige still anchors US space identity",
        "vs": "Apollo differed from the contemporaneous Soviet N1 program, which suffered repeated rocket failures and was canceled. It differs from later Space Shuttle missions in being expendable and singularly aimed at lunar landing. It differs from current Artemis plans in lacking sustained presence ambitions",
        "ex": "Apollo 13's oxygen tank explosion en route to the Moon required improvising CO2 scrubbing with mismatched hardware and using the LM as a lifeboat, a near-disaster that engineers recovered from in real time, captured by the famous problem-solving phrase 'failure is not an option'",
        "mis": "people think Apollo was a smooth march to success. Apollo 1 killed three astronauts in a launchpad fire; Apollo 13 nearly lost three more; many uncrewed test failures occurred. The program survived because political commitment and engineering discipline absorbed the setbacks",
    }

    T["the Suez Crisis"] = {
        "_cat": "history",
        "what": "the 1956 international crisis triggered when Egypt nationalized the Suez Canal, prompting a coordinated Israeli, British, and French military intervention that was halted under US and Soviet pressure, marking the end of British and French dominance in the region",
        "how": "Nasser nationalized the canal in July 1956 after the US and UK withdrew funding for the Aswan High Dam. Israel invaded Sinai in October; Britain and France issued an ultimatum and bombed Egyptian airfields, then landed troops. Eisenhower threatened US economic action; Soviets threatened intervention. Britain and France withdrew under pressure",
        "why": "Suez exposed the limits of European imperial power, accelerated decolonization, elevated the US and Soviet Union as the decisive arbiters of regional crises, and humiliated Anthony Eden's government, which fell shortly after. Nasser emerged stronger and Pan-Arab nationalism gained momentum",
        "vs": "Suez differed from earlier interventions like the British occupation of Egypt in 1882, which faced no comparable superpower veto. It differed from the simultaneous Hungarian Uprising, which drew Western condemnation but no intervention against Soviet forces",
        "ex": "the United Nations Emergency Force (UNEF), deployed after the crisis, was the first peacekeeping operation of its kind and established a model for UN intervention. The financial pressure on the British pound during the crisis effectively dictated the British withdrawal",
        "mis": "people think Suez was primarily about ownership of a canal. It was a referendum on whether mid-tier European powers could still act unilaterally outside the new bipolar order. The answer, decisively, was no",
    }

    T["the Silk Road's economic impact"] = {
        "_cat": "history",
        "what": "the network of overland and maritime trade routes connecting East Asia with Central Asia, the Middle East, North Africa, and Europe from roughly 130 BCE through the 15th century, transferring goods, technologies, religions, and diseases",
        "how": "caravans crossed deserts and steppes between oasis cities; ships plied the Indian Ocean monsoon system. Silk, spices, jade, glass, and horses moved long distances through many intermediate hands. Religions (Buddhism, Manichaeism, Islam) and technologies (paper, gunpowder, the compass) traveled with the goods",
        "why": "the Silk Road catalyzed cultural diffusion, urbanization of inland Asia, and the spread of crops, technologies, and ideas that shaped Eurasian civilization. It also vectored the Black Death from Asia to Europe in the 14th century, a reminder that integration carries pathogen risk",
        "vs": "the overland Silk Road differed from the maritime Indian Ocean network, which moved bulkier goods more cheaply once monsoon-driven shipping matured. It differed from later European Atlantic trade, which involved direct long-distance shipping rather than relay through middlemen",
        "ex": "Samarkand and Bukhara grew rich as caravanserai cities at the heart of the network, supporting libraries, observatories, and a Persianate cultural sphere. Their decline followed the rise of cheaper sea routes and Mongol disruptions",
        "mis": "people think there was a single road. The Silk Road was many routes shifting over centuries; few traders went the entire distance. Goods passed through dozens of hands, with each leg controlled by local merchants and political powers",
    }

    T["the abolition of the Atlantic slave trade"] = {
        "_cat": "history",
        "what": "the legal and political process by which European and American powers banned the transatlantic slave trade between 1807 and 1870, while emancipation of enslaved people followed at varying paces in different colonies and nations",
        "how": "Britain banned the trade in 1807 and used the Royal Navy's West Africa Squadron to interdict slavers and pressure other states. The US banned imports in 1808, although domestic slavery continued. Brazil, the largest destination, ended imports in 1850 and slavery in 1888. Treaties, blockades, and judicial pressure together collapsed the legal trade",
        "why": "abolition reshaped Atlantic economies, ended a defining institution of early modern empires, and established humanitarian intervention as a foreign-policy norm. It also displaced production toward sugar, coffee, and cotton economies that adapted with indentured and post-emancipation labor systems",
        "vs": "abolition of the trade differs from emancipation, which ended slavery itself and came later in many places. It differs from earlier reductions in serfdom in Europe, which were medieval and gradual, and from manumission in earlier slave societies, which was individual rather than systemic",
        "ex": "the Royal Navy intercepted around 1600 slave ships and freed roughly 150000 captives during the suppression campaign, while diplomatic pressure ended Portuguese, Spanish, and Brazilian participation. The cost was significant; Britain pursued it in part on humanitarian grounds and in part to deny rivals economic advantages",
        "mis": "people think abolition was driven only by moral conversion. Religious and humanitarian campaigners were essential, but economic shifts (industrial labor, sugar competition from new producers), enslaved resistance and revolt (Haiti above all), and great-power rivalry shaped the timing and method",
    }

    T["the Council of Nicaea"] = {
        "_cat": "history",
        "what": "the 325 CE ecumenical council convened by Emperor Constantine I in Nicaea (modern Iznik, Turkey) to resolve doctrinal disputes within Christianity, especially the Arian controversy about the relationship between God the Father and the Son",
        "how": "approximately 250-300 bishops attended, debating Arius's position that the Son was created and not coeternal with the Father. The council formulated the Nicene Creed asserting consubstantiality (homoousios) of Father and Son, and condemned Arianism. Decisions on Easter dating and church governance also issued",
        "why": "Nicaea set the precedent for emperor-led councils as authoritative in defining orthodoxy, established the Nicene Creed as the basis of Christian doctrine still recited by most denominations, and exposed how political and theological power had merged after Constantine's conversion",
        "vs": "Nicaea differed from earlier local synods by being empire-wide and imperially convened. It differed from later councils (Constantinople, Chalcedon) that refined the Trinitarian and Christological formulas in response to new disputes. It differed from the much later Reformation councils, which fractured rather than unified",
        "ex": "Athanasius of Alexandria became a leading defender of Nicene orthodoxy and was repeatedly exiled and restored as imperial favor shifted. His struggles illustrate how the council settled doctrine on paper but politics inside and outside the church remained turbulent for decades",
        "mis": "people think Nicaea decided which books were in the Bible. It did not; the canon emerged through gradual consensus over centuries. Another myth is that the council unanimously settled Arianism; Arianism revived after Nicaea, with several emperors favoring it before later councils confirmed Nicene doctrine",
    }

    T["the Iranian Revolution of 1979"] = {
        "_cat": "history",
        "what": "the upheaval that overthrew Iran's monarchy under Mohammad Reza Shah Pahlavi and established an Islamic Republic under Ayatollah Ruhollah Khomeini, transforming Iran's politics and shaking the regional and global order",
        "how": "discontent with the Shah's autocracy, inequality, and rapid Westernization fueled mass protests through 1978. Strikes paralyzed the oil industry, the army defected or stood aside, and the Shah fled in January 1979. Khomeini returned from exile, a referendum founded the Islamic Republic, and a new constitution established clerical rule with Khomeini as Supreme Leader",
        "why": "the revolution introduced political Islam as a major force in international relations, ended decades of US-Iran alliance, sparked the 1979-1981 hostage crisis, contributed to the Iran-Iraq War, and reshaped the Middle East. Its institutions still govern Iran and influence Shia communities across the region",
        "vs": "the Iranian Revolution differs from secular Arab nationalist revolutions of the 1950s-60s by its religious-ideological character. It differs from Eastern European revolutions of 1989, which were liberalizing, by replacing one authoritarian system with another. It differs from the Saudi system, which is monarchical with religious legitimacy rather than clerical rule",
        "ex": "the seizure of the US embassy in November 1979 and the 444-day hostage crisis crystallized the new order's anti-American posture and paralyzed the Carter administration. The crisis ended with the inauguration of Reagan in 1981, with hostages released minutes later",
        "mis": "people think the revolution was always destined for clerical rule. Early opposition included secular liberals, leftists, and Islamists; Khomeini's faction outmaneuvered the others post-revolution. The Islamic Republic's particular shape was contingent on internal politics in 1979-1981",
    }

    T["the printing press in East Asia"] = {
        "_cat": "history",
        "what": "the development of woodblock printing in Tang China (around the 7th century), movable clay type by Bi Sheng around 1040, and movable metal type in Goryeo Korea by 1234, predating Gutenberg's European press by centuries",
        "how": "woodblock printing carved a full page of text in mirror image into a wooden block, inked it, and pressed paper onto it. Bi Sheng made movable characters from baked clay; Korean printers later cast bronze characters using sand or wax molds. Chinese and Korean texts required thousands of distinct characters, complicating movable type relative to the European alphabet",
        "why": "East Asian printing produced enormous bodies of literature, government documents, and religious texts long before European movable type. The Diamond Sutra of 868 CE survives as the oldest dated printed book. Korean court editions show movable metal type used for state publications generations before Gutenberg",
        "vs": "East Asian woodblock differs from later European movable type in being suited to long print runs of stable texts. Korean metal type was technologically comparable to Gutenberg's but did not industrialize a vernacular book trade in the same way. Gutenberg's economic impact came from alphabet plus capitalist printing, not from priority of invention",
        "ex": "the Jikji, a Korean Buddhist text printed in 1377 with movable metal type, predates Gutenberg by 78 years and is recognized by UNESCO as the oldest extant book printed with movable metal type",
        "mis": "people think Gutenberg invented printing. He invented a particular combination of technologies (oil-based ink, hand mold for casting type, screw press) that enabled mass production in alphabetic scripts. Printing existed in East Asia for centuries before",
    }

    T["the partition of India"] = {
        "_cat": "history",
        "what": "the 1947 division of British India into the independent dominions of India and Pakistan along religious lines, accompanied by the largest mass migration in human history and communal violence that killed roughly one to two million people",
        "how": "the British Raj's hasty exit, accelerated by Mountbatten, drew borders by Cyril Radcliffe in five weeks based on district-level religious majorities. About 14 million people moved to align with their state of choice, with Hindus and Sikhs heading east into India and Muslims west to Pakistan. Princely states acceded to one or the other, with Kashmir's contested accession sparking a war",
        "why": "Partition created two large nations that have fought multiple wars and remain locked in nuclear-armed rivalry. It produced enduring refugee communities, shaped South Asian politics, and the unresolved status of Kashmir continues to trigger crises. The trauma is still central to Indian, Pakistani, and Bangladeshi national identities",
        "vs": "Partition differed from peaceful decolonizations in its scale of violence and forced migration. It differed from the later 1971 separation of Bangladesh from Pakistan, which arose from linguistic and political grievances rather than religious ones. It differed from the 1948 Israeli-Palestinian split in being intra-imperial and bilateral",
        "ex": "the Punjab and Bengal were the most violent regions, with massacres and mass abductions. Trains arriving at stations bore corpses; refugee columns walked for weeks. The state-level demographic shift was effectively complete by 1948 in the most affected areas",
        "mis": "people think Partition was inevitable given Hindu-Muslim differences. Many leaders and communities lived in mixed cities for centuries; rapid politicization of identity, colonial governance choices, and the timing of British withdrawal made Partition contingent rather than fated",
    }

    T["the Manhattan Project"] = {
        "_cat": "history",
        "what": "the US-led research and development program of World War II that produced the first nuclear weapons, employing over 130000 people across multiple sites at a cost of about 2 billion 1940s dollars",
        "how": "Los Alamos under Oppenheimer designed the bombs; Oak Ridge enriched uranium-235 by gaseous diffusion and electromagnetic separation; Hanford produced plutonium-239 in nuclear reactors. The Trinity test in July 1945 detonated a plutonium implosion device; uranium and plutonium bombs were dropped on Hiroshima and Nagasaki in August",
        "why": "the Manhattan Project ushered in the nuclear age, ended World War II in the Pacific, and launched the arms race that defined the Cold War. It created national laboratories that still shape American science and trained a generation of physicists, while also raising enduring ethical and security questions about scientific responsibility",
        "vs": "the Manhattan Project differed from the German uranium project, which never came close to a weapon, and from the more limited Soviet wartime program (Tube Alloys, also smaller). Its scale eclipsed all prior science-engineering programs and rivaled the contemporary Allied bomber and shipbuilding programs",
        "ex": "the Trinity test on July 16, 1945, in New Mexico produced a yield of about 21 kilotons. Oppenheimer reportedly recalled the Bhagavad Gita line, 'Now I am become Death, the destroyer of worlds.' Within weeks, Hiroshima and Nagasaki were destroyed; Japan surrendered on August 15",
        "mis": "people think the bomb was decisive in Japan's surrender. Most historians accept it was a major factor, but Soviet entry into the Pacific war on August 8 and the destruction of Japanese conventional capability also weighed heavily. The relative weights remain debated",
    }

    T["the rise of the Ottoman Empire"] = {
        "_cat": "history",
        "what": "the gradual expansion of the Ottoman state from a small Anatolian beylik in the late 13th century to a transcontinental empire that conquered Constantinople in 1453, ruled the Balkans, the Middle East, and North Africa, and persisted until 1922",
        "how": "the Ottomans combined ghazi warrior expansion, the timar land-grant system that supported cavalry, the Janissary corps recruited from Christian children via the devshirme, gunpowder weapons, and pragmatic administration that incorporated existing Byzantine and Arab institutions. Naval power followed, dominating the eastern Mediterranean by the 16th century",
        "why": "the Ottomans ended the Byzantine Empire, controlled trade between Asia and Europe (one impetus for European Atlantic voyages), shaped the religious and political map of the Middle East and Balkans for centuries, and left legal and architectural legacies still visible across the former empire",
        "vs": "the Ottoman state differed from the older Caliphates in being Turkish-ruled and more bureaucratic; from the Byzantine empire it succeeded in being Sunni Muslim and pluralistic via the millet system. It differed from contemporary European powers in scale, religious composition, and durable continental land-empire structure",
        "ex": "Mehmed II's 1453 conquest of Constantinople, using massive bombards including the Basilica cannon, illustrated the marriage of Ottoman organization with cutting-edge gunpowder technology. The fall of the city had been anticipated for decades but the conquest still shocked European Christendom",
        "mis": "people think the Ottoman Empire was monolithically Turkish and Muslim. It was a multiethnic, multiconfessional state with significant Greek, Armenian, Slavic, Arab, Jewish, and Kurdish populations, governed via the millet system that granted religious communities partial self-administration",
    }

    T["the Cuban Missile Crisis"] = {
        "_cat": "history",
        "what": "the 13-day confrontation in October 1962 when the US discovered Soviet nuclear missiles being installed in Cuba, leading to a naval quarantine, intense diplomacy, and the closest the world came to nuclear war during the Cold War",
        "how": "U-2 reconnaissance photos confirmed Soviet medium-range missiles. President Kennedy chose a quarantine over invasion or air strikes after debate in ExComm. Public ultimatum and back-channel negotiations produced a deal: the USSR removed missiles from Cuba; the US privately removed Jupiter missiles from Turkey and pledged not to invade Cuba",
        "why": "the crisis demonstrated the fragility of nuclear deterrence and sparked direct communication channels (the Moscow-Washington hotline) and arms control efforts (the Partial Test Ban Treaty in 1963). Both superpowers pulled back from the brink of confrontation that could have killed hundreds of millions",
        "vs": "the crisis differed from earlier Berlin crises in involving direct Soviet nuclear missiles in the Western Hemisphere; it differed from later proxy escalations (Vietnam, Afghanistan) in directly threatening superpower territory. It set norms for managing nuclear standoffs that endured through the rest of the Cold War",
        "ex": "Soviet submarine commander Vasily Arkhipov reportedly refused to authorize a nuclear-armed torpedo launch when his depth-charged sub lost contact with the surface, a single decision that may have prevented escalation; recently declassified records emphasize how close the world came to disaster",
        "mis": "people think Kennedy faced down Khrushchev with steely resolve. Both leaders made concessions, and the secret part of the deal (Jupiter removal from Turkey) was kept from the public for years to preserve Kennedy's image of toughness. Crisis management was less heroic and more pragmatic than mythologized",
    }

    T["the Black Death"] = {
        "_cat": "history",
        "what": "the devastating mid-14th-century pandemic, primarily of bubonic and pneumonic plague caused by Yersinia pestis, that killed roughly 30 to 60 percent of Europe's population between 1347 and 1351 and recurred for centuries",
        "how": "Y. pestis traveled along trade routes from Central Asia, vectored by fleas on rodents and amplified by ship and caravan transport. It spread through Europe via Mediterranean ports starting in Genoese ships from Caffa. Pneumonic forms transmitted directly between humans, accelerating spread in cold weather. Cities lost half or more of their population in months",
        "why": "the plague reshaped European society: labor shortages raised wages and weakened serfdom, religious authority cracked under the failure of prayers and rituals, antisemitic pogroms swept Christendom, and economic and demographic patterns shifted for generations. It also drove medical and public health innovations, including quarantine practices",
        "vs": "the Black Death differed from earlier Justinianic Plague (6th century, also Y. pestis) in striking a more densely connected late-medieval Europe. It differed from smallpox and measles in killing many adults rather than mostly children, which is why labor effects were so dramatic. It differed from cholera in being faster and frequently fatal within days",
        "ex": "Florence, Boccaccio's Decameron describes graphically, lost more than half its population. Civic governments adopted novel measures including quarantine (40 days, the origin of the word) and isolation hospitals (lazarettos), early state public-health interventions",
        "mis": "people think the Black Death's transmission was understood at the time. Contemporaries blamed miasma, divine wrath, and minorities; the bacterial cause was identified by Yersin only in 1894. Some historians have proposed alternative pathogens, but DNA from medieval graves now confirms Y. pestis as the agent",
    }
