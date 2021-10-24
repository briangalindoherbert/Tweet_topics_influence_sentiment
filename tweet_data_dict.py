# encoding=utf-8
"""
tweet_data_dict contains CONSTANTS and word lists for nlp processing.
tw2vec is a pretrained vector model trained on a tweet corpus from google

ADDITIONAL SUB-DIRECTORIES OFF GSTWEET FOR THIS PROJECT:
./project/  - articles and documents on this topic
./twitter/  - files with batches of tweets, json format, from twitter developer api
    endpoints which I access using Postman.  easier to tweek queries and check results
    than via scripting the http get in python.
/templates/ - html, javascript, json and yaml templates.  html+js as I'm looking at some
    cool d3 viz stuff I can do with this data, json for playing with parsing schemas, and
    yaml for config files if I make the twitter api calls from py scripts.
/output/ - my 'deliverables' such as serializing my data to file, the gensim models I
    generate, wordclouds, and other visualizations and saved data.
/models/ - pretrained or pre-labeled data for word2vec or nltk models, such as large
    vocabulary files with vector embeddings or tweets or phrases with sentiment labels
"""
MODELDIR = '/Users/bgh/dev/pydev/gsTweet/models/'
TWEETSDIR = '/Users/bgh/dev/pydev/gsTweet/twitter/'
GMEDIR = '/Users/bgh/dev/pydev/gsTweet/twitter/gamestop/'
ESLDIR = '/Users/bgh/dev/pydev/superleague/twitter/superleague/'
OUTDIR = '/Users/bgh/dev/pydev/gsTweet/output/'

W2VEC_PRE = '/Users/bgh/dev/pydev/gsTweet/models/freebase-vectors-skipgram1000-en.bin'
TW2VEC_PRE = '/Users/bgh/dev/pydev/gsTweet/models/word2vec_twitter_tokens.bin'

GS_ABSOLUTE = ["always", "horrible", "never", "perfect", "worthless", "useless",
             "infinitely", "absolutely", "completely", "totally", "exponentially",
               "idiotic"]
GS_EXTREME = ["insane", "evil", "psycho", "idiot", "rube", "crazy", "neurotic", "retarded",
              "stupid"]
GS_BALANCE = ["relative", "preferable", "optimal", "better", "inferior", "superior"]

# these are standard Adverb and Pronoun STOPS on many nlp projects
GS_ADVB = ["am", "are", "do", "does", "doing", "did", "is", "was", "were"]
GS_PPRON = ["we", "he", "her", "him", "me", "she", "them", "us", "they"]

# Words to Remove: standard plus stops based on project type, context, or tfidf analysis
STOPS_ESL = ["team", "club", "clubs", "UEFA", "ESL", "UEFA", "game", 'superlega',
            "english", "arsenal", "barcelona", "chelsea", "juventus", "liverpool",
            "mancity", "manutd", "MUFC", "madeid", "real madrid", "sheffield", "shiffeld",
            'spurs', 'tottenham', 'european', '\u2066kevinhunterday\u2069', 'david_ornstein',
            '\u2066kieranmaguire\u2069', '\u2066\u2066mjshrimper\u2069', 'markslearnenglish',
            "europesuperleague", 'europeansuperleague', "league", "'pundit", 'football',
            'minutes', 'TV', "news", 'sport_en', 'spacetravel', 'leemarkjudges', '08069830203',
            '2PACMUFC', 'phil__howarth', 'john_nufc', 'sheshe_tom', 'russell_vine', '1reddevil2',
            "united", 'super', 'league', 'florentino', 'superleague', "days", '[at', '2021]',
            'mikel', 'tourtalk', 'PGA', '⛳️', 'golf', 'pgatour', "week", "today", "year",
            "legacyfan"]

STOP_UTF2 = ['🌙', '🍭', '😋', '🌶', ]

GS_STOP = ["a", "about", "all", "almost", "also", "among", "am", "an", "and","already",
           "any", "are", "as", "at", "back", "because", "but", "by", "cause", "come",
           "could", "did", "does", "either", "else", "ever", "even", "for",
           "from", "going", "has", "had", "have", "his", "her", "hers", "how",
           "however", "if", "I", "in", "into", "is", "it", "its", "just", "least",
           "let", "lets", "likely", "may", "me", "might", "must", "much", "my", "need",
           "now", "of", "often", "on", "only", "or", "other", "our", "own", "rather",
           "really", "same", "seems", "shall", "show", "should", "since", "so", "some",
           "something", 'still', 'such', "than", "that", "the", "their", "them", "then",
           "there", "these", "they", "think", "this", "those", "thus", "to", "too",
           "was", "watch", "well", "were", "what", "while", "will", "would", "whom",
           "with", "yet", "your", "rt", "we", "what"]

STOP_ADD = ['_', '(', ')', '…', '[', ']', '_', '__', ':', '"', '️', ':', '"', '/', ']',
            '201920', '20212022', '235', '244', '247', '28t', '6', '651m', '7.406', '️',
            'about', 'acdvd', 'actual', 'additional', 'de', 'A', '|', 'le', '[', ']',
            'admi', 'affect', 'affects', 'again', 'ahead', 'ake', 'allowed',
            'alonehowever', 'als', 'anybody', 'anyone', 'anyway', 'apostrophe_',
            'app', 'ar', 'around', 'asterisk', 'b.', 'ba', 'bara', 'be', 'being',
            'besides', 'breaka', 'c', 'can', 'chuchilips', 'cian', 'clearly',
            'colon', 'comma', 'cos', 'do', 'da', 'definitely', 'delve', 'despite',
            'differen', 'doing', 'dr', 'e', 'each', 'ed', 'eg', 'el', 'eltoc',
            'emailed', 'erstes', 'everyone', 'ewww', 'f', 'fa', 'fairly', 'fe',
            'feel', 'flies', 'fully', 'gave', "give", 'get', 'gil', 'going', 'got',
            'gua', 'guess', 'happened', 'hashtag', 'having', 'hea', 'helicopter',
            'here', 'hey', 'hows', 'hyphen', 'i', 'id', 'ielts', 'ill', 'ings',
            'ins', 'instead', 'ipad', 'iphone', 'ipod', 'ive', 'j', 'ju', 'keeping',
            'l', 'la', 'lea', 'lev', 'literally', 'lot', 'm',
            'mark', 'marks', 'mars', 'matter', "'the",
            'maybe', 'micr', 'middleweigh', 'mobile', 'most', 'mr', 'n', 'name',
            'nasa', 'nearly', 'nevertheless', 'notes', 'o', 'oa', 'orbn', 'orry',
            'p', 'per', 'play', 'possibly', 'potentially', 'pu', 'punctuation',
            'put', 'quite', 'quotation', 'r', 'remains', 'ro', 'rotorcraft',
            'rul', 'said', 'say', 'seem', 'semicolon', 'single', 'sl', 'slash',
            'soo', 'st', 'sto', 'supe', 't', 'taken', 'talked', "talk", "look", "give",
            'th', 'tha', 'thats', 'themselves', 'theres', 'thing', 'things', "'save",
            'tho', 'thst', 'towar', 'trying', 'type', 'u', 'un', 'und', 'underscore',
            'uns', 'until', 'vary', 'view', 'w', 'way', 'well', 'went', 'whe',
            'whether', 'which', 'whoever', 'writes', 'x', 'y', 'ya', 'ye',
            'yep', 'yer', 'youd']

STOP_TWEET = ['RT', "says", "take", "know", "every", "time", "people", "want", "wants",
             'approximately', 'typifies', 'continuously', "many", "every", "happen",
             'GTTO', 'describes', 'remembering', 'reconsidering', 'developp',
             'phantasy_foods', "point", "artistrtweeters", "gnev",
             'see', 'compared', '=', '/', 'sure', '&', "''", "'d", "'ll", "'s",
             'great:', 'why', '1', '2', '01', '–', "according",
             'sta…', 'pod:', '4', 'thoughts', 'pgas', '"', 'theyre', '&amp',
             '60', '182', 'yall', 'OK', 'onto', '"this', 'him', 'call',
             '""suffer', 'become', 'ttanslated', 'الدوري_الانجليزي', 'دوري_السوبر_الاوروبي',
             'dear', 'youre', 'mot', 'others', 'both', '[thread]', '30',
             'station', '24ur', 'im', 'basically', 'soon', 'where',
             's', 'saidthe', 'though', 'thinks', 'thought', 'do:', 'hes', 'al',
             'five', 'sense', 'form', "make", 'needs', 'tv', '3pm', 'show', 'due', 'watch',
             'between', 'turned', 'different', 'simply', 'through', 'actually', 'support'
             ]

STOP_NONALPHA: list = [':', '!', '?', '…', '"', '😂', '️', '🤣', '🚨', '12', '⚽', '1', '👏', '3', '_', '[', '2', '/',
                       ']', '5', '2021', '🤔', '0', '👇', '🔴', '🗣', '6', '🏻', '😭', '✅', '\U000e0067', '💥', '10',
                       '🇪', '1futbol', '👍', '\u200d', '23', 'วันทองตอนจบ', '🎙', '$', '14', '4', '7', '🚫', '🏼',
                       '😎', '🏽', '€', '❌', '♂', '⚪', '❤', '💯', '😆', '⚒', '👎', '⬇', '🏉', '100', '👉', '2022', '&',
                       '🏆', '20', '💙', '🔥', 'ℹ️', '▪', '24', '2024', '🤷', '11', '2020', '🇹', '🏴', '\U000e0062',
                       '\U000e007f', '🙄', '💪', '🇸', '(', '•', '🇮', '\U000e0065', '\U000e006e', '17', '🎶', '👋',
                       '😌', '🇬', '22', ')', '21', '🔊', '😉', '❗', '🇺', '🔵', '|', '«', '😳', '😏', 'pics_by_tony',
                       '🇧', '🤦', '🙌', '8', '💸', 'https://t…', '\u2066', '\u2069', '48', '18', '➡', '🥳', '👊', '👀',
                       '😒', '15', '72', '🤡', '😅', '50', '40', '13', '1890', 'all_outside', ':)', '🤑', '🙏', '🎉',
                       '36', '🤬', '👌', '9', '99', '😃', '🇷', '🇩', '92', ':/', '✊', '_chaxza', 'p_myte', 'ryan_utb',
                       '⚫', '😍', 'https://t', '43', '😇', '🤯', 'celtic_now', '39', '😄', 'NUFC_HQ', '93', '19',
                       '_befoot', 'albie_dunn', '2026', '😜', 'efo_phil', 'i̇lerliyoruz', 'B_HQ', '🟡', '501', '5pm',
                       '89', '🇫', '90', '💰', '😁', '😡', '🎧', '♀', 'football_bm', '😫', 'uefacom_fr', 'aisa_arsenal',
                       'AB_18___', '7395', '1️⃣2️⃣', 'fu_invest', '🤨', 'nitin_cul', 'jm3s_', '1863 2021',
                       'john_w_henry', '🏟', '🆚', '360', '📺', '⏰', '300', '🚀', '96', '05', '64', '💵', '❓',
                       '3sportsgh', '📢', '🖥', '💀', '💬', '85', '😙', '80', '2023', '🎥', '🤞',
                       'lil_mr_dynamite', '🔁', '16', '»', '1994', '😢', 'pirlo_official', 'F1', '2035', '3rd', '🌏',
                       'rindai__gari', '44', 'lw_newcastle', '_owurakuampofo', 'lu_class', '中2英語', 'cfc__kevin',
                       'tommy_viccetti', 'yashveen_xyz', '210', '😬', '1021',
                       'แฟนบอลหลายกลุ่มได้ออกมารวมตัวกันแสดงจุดยืนไม่เห็นด้วยกับไอเดียยูโรเปียนซูเปอร์ลีก',
                       'การแข่งขันรายการใหม่ที่เพิ่งประกาศเปิด', 'p_hollas', 'spring_steen', '🤓', 'kookie_kuhle',
                       'betway_za', 'matty_west', '🤮', 'ms_sportsbiz', '6013092', '🪓', '11pics', 'n_kayy',
                       'buchi_laba', 'f24debate', '📻', '2004', '94', 'cc_eckner', '⚠', '🌍', ']:', '101', '77', '💩',
                       '😤', '123', 'ed_aarons', 'https://…', '5th', '1995', '🌎', 'mmu_law', '1st', '28', '8m', '😱',
                       '34', '✍', '1duval', '420day', '📍', '🌱', '😊', 'dw_sports', '8000', '10k', '️the', '__', '83',
                       '2⃣', '2manutd', '60m', '2019', '4sale', '🤝', '2005', '1904', '🟢', 'public_archive', '4evra',
                       '1545', 'i̇lkay', '__robh__', '1019', 'under_thecosh', '350000', 'sascha_p', 'TN03S',
                       'boss_himselff', 'x_godden', '310780', 'j_castelobranco', 'gala_news', '1905', 'justin_cash',
                       '_1', '📕', 'rbleipzig_fr', '1410', '56789', 'naushabah_khan', '🥇', '´', '🍀', '🥴', '📝',
                       '2021/22', '9ja', '️florentino', '⚡', '⏩', '21st', '█', '↓', 'adelaide_rams', 'voz_populi',
                       '2016', '49', '♦', 'interactive_key', '😲', '75', '📋', '91', '🔄', '09', '🙃', '👨', '00',
                       'redemmed_odds', '199', 'mr_ceyram', '_abdul_m_', '2021/2022', '⚰', 'der_augustus',
                       'theboy_whoiived', '1972', 'official_lyckez', 'british_gambler', '🥱', 'alex_slwk', '3sports',
                       '◾', 'y3nko', 'padmnaabh_s', '🤭', '65M', 'dorian__bdlp', '🔹', 'r1100gsbluenose', '🇦', '1878',
                       '04', '25', 'C1', 'o_machel', '804m', '030bn', '651m', '📰', 'nabil_djellit', 'WW1', 'lkn_jr',
                       '86', '👈', '💛', '！', '67', '🧡', '6pm', '3ingsilly', '02', '97', '1/3', '27', '15:00', '400',
                       '📸', '🛑', '🔗', '💷', 'th3n', 'wo4k', '🎵', '🖕', '8181', 'fcsm_eng', 'b_fernandes', '👑',
                       'danzila_jr', 'benjamin_watch', '☑', 're_dailymail', 'ellscott_mufc', 'shem_nyakeriga',
                       'ruvu_shootingfc', 'dave_otu', '😹', '📈', '📉', '19deabril', '30th', '12th', '🕒', '️⃣', '87',
                       'gj_thomas', 'sky_maxb', 'CR7', '14th', '💦', '300m', '45', '81', '29', '400m', 'letter_to_jack',
                       '9pm', '9th', '📌', 'caitlyn_jenner', '=', '180', '36th', '🟠', '_59', 'handsome_szn', '500M',
                       '✨', '📃', '50p', 'figo_anane', 'okt_ranking', 'M25', 'nkd_mzrx', '247', '🐯', ':D', '☕',
                       'eddie_l', '73', '🧗', '🤰', 'pat_que', 'byl_al', '19aprile', 'cfc_vish', '🤩', 'fausto__ii',
                       '250', '_2', 'fu4ad', '4wn', 'jesus_olimart', '021', '4trigger', '000', '🔎', '70', '200', 'KS1',
                       '🌈', '🌿', '🍃', 'front_threepod', 'my11', '30', '💶', '⬅', 'G7', '48hrs', '9876', '13officiel',
                       '35', '🎮', '1aaronpaul', '35x', '1m', '76', 'theo_shanks', '180m', 'af_spurs_osc',
                       'bee_lamsing', 'เรื่องด่วนตอนนี้', 'จะมีการประชุมคณะกรรมการบริหารของยูฟ่า',
                       'เย็นวันนี้ที่เมืองมองเทรอซ์', 'แน่นอนคือ', 'ถกปัญหา', 'และรับมือ', '2GB873', 'simon_hughes',
                       'owl_reporter', '405', '_1992', '5/6', 'prosper_tv', '_1102', '30BG', '65', '🖖', 'le__foireux',
                       '7barton', 'fums_magazin', '🏾', '90lcfc', 'sz_unitedcr', '️ESL', '1980', '🇨', '🇭', '8p', '31',
                       '7p', '😩', '😘', '2572', '405m', '🇳', '4th', '⏫', 'sv98', '88', '1500', '400k', '5⃣', '\u200b',
                       'fcbarcelona_cat', 'favp_org', 'ps6811', 'arminia_int', '😐', '💭', 's_redmist', '_3aaz_',
                       'cadiz_cfin', 'soccernews_nl', 'fcb_newsfr', '11freunde_de', 'reich_against', 'mr_sergiowalker',
                       'nicola_sellitti', '_le_20', 'valenciacf_en', '🥈', '🥉', 'CV1874', 'i_nautilus', '💐', '🌷',
                       '316', 'king_simian', 'newsome_mimi', '95dot9', '2308', 'teh_dewah', 'LE_FIVE', '😔',
                       'leew_sport', '4rsene', '4seen', 'ff_titans', 'lr2cblog', '3/4', '1992', '07', '⚖', 'ARIS__FC',
                       'barca_buzz', '304', '️joses', '️man', 'playmaker_en', '👆', '95', '8FM', '722', '💻', '18th',
                       '90s', '️euro', '️hopefully', '📣', '40yrs', '71', '🍎', '⚓', '⃣', 'ast_arsenal', '3⃣',
                       '2011/12', 'iam_presider', 'scott_rebnoise', 'mrhandsome_za', 'ryan_padraic', '42_ie', '🐍',
                       'jai_d', 'alex_dreyfus', '412', '1990', '989FM', 'tam_baw', 'K12', 'fpl_tactician',
                       'scoreboard_us', 'alison_mcgovern', '️statement', '500', '932', '😵', '_88', '90plusshow',
                       'm69098997eye', '10s', 'avance_carrental_official', 'twiggy_garcia', 'kathrinm_hansen',
                       'lfc_wahome', '3arnie', 'POF_POD', '📚', '✏', 'james_barker', '🥧', '12thmantweets_', 'org_scp',
                       '88r', 'cycling_memes', '03_ezz', '7514', 'P12', 'mphoeng_m', '1apponline', '👚', '👖', '🎩',
                       '160', '1961', '5bn', '🗯', '2008', '600m', '✌', '🅱️', '🎨', '🎤', '1/2', '37', '😰', '_08',
                       '💼', '1⃣', 'al___d', '🏐', '54', '⬆', '🏫', 'phantasy_foods', '350M', '5️⃣0️⃣', '1786',
                       '2arsenal', 'socialist_app', 'TV3', '00x', '⏲', '️late', 'broadcast_sport', 'broadcast_sports',
                       'jake_bickerton', 'vis_mac', '10hd', 'kofi_kwarteng', 'queen_uk', '325', 'socialist_party',
                       '_67', 'jas_pod', 'mcgowan_stephen', '🗳', 'mas_que_pelotas', '️seguimentfcb', '28t', '51',
                       'pisto_gol', 'stone_skynews', '4m', 'so_cal_geordie', 'carshaltona_fc', '92bible', '450',
                       'boycie_marlene', 'aha_com', '66', 'nick_pye', 'aaron_challoner', '_7', 'D5', 'dean_geary',
                       'l0uisqlf', '2045', 'uci_cycling', '📥', '82', '📲', 'R98UIQIGL6', '️english', '☠', '‼', '114',
                       '74', '8ème', '🇵', '55', '191', '2nd', '5sport', '41', '01918030', '68ASR', 'pin_klo', '🍻',
                       'kog_mark', '🤧', 'guardian_sport', '🤪', '🚩', '😀', '💴', 'neil_moxley', 'beinsports_en',
                       '1ère', '69', 'red_white', '9230861192', '🏦', 'the_crab_man', 'diego_bxl', '6th', '7th',
                       'dsj_itv', '400M', 'sop_soppi', 'psg_inside', 'ghrhull_eyork', '1892redspodcast', '⭐',
                       'shuhel_miah', 'dan_k', '🤜', '🤛', 'timothy_kls', '💡', '50000', '2k', '🎬', 'fx9',
                       'marina_sirtis', 'gr_yassine', '2794526', 'phil_wellbrook', 'kmtv_kent', 'ep_president',
                       '0ptimumvelocity', '😮', '_kalule', '600M', '151', 'kendrick_kamal', '1xbet', '5050',
                       '75_paname', 'the_ahmed_hayat', 'de_essieno', '😪', '️saynotoeurop', '_jenky88', '810', '92a',
                       '🍺', '🖤', '4sug2', 'offthepitch_com', 'talking_toffees', '1985', '2002', '202', '7i',
                       'owen_miller', 'MU_ST', 'ast_ars', 'sw6lionstv', 'deji_ooniabj', '2boxfutbol', 'shouvonik_bose',
                       '🐲', 'm_nialler', '2009', 'L1', 'acecast_nation', 'footy_prime', '🟥', '9supermac',
                       'rich_banks', '8billion', '15編楽しめるよ', '1417', 'mufc_larn', '006', 'psg_english', '125',
                       'kicker_bl_li', '🤢', '_frankiam', '3_', '7has', '️QUARTER', '42', '🆕', '𝟏𝟖𝟕𝟒', '️MCFC',
                       '9am', '8s', 'G20', 'shane_mangan', '🗞', '_9', 'a_liberty_rebel', 'longevity_dan', '_201',
                       'matt_law_dt', 'antifa_celtic', '1888', 'sams_keef', 'nathaniel_john', '🔛', '7221818', '6s',
                       'jo_napolitano', '_93', 'freelancers_usa', '10pm', 'football_prizes', 'رسميـًا', 'no14gifts',
                       'iam_wilsons', '☺', '50million', 'angry_voice', '996', 'thechai_vinist', '🦁', '️boris', '1981',
                       'c_barraud', '8B', 'beinsports_fr', '97th', '🤥', '⏪', 'kam_lfc', '🇱', 'juba_obr', '2050',
                       '️supporting', '4yo', 'mike_ananin', '26', '🔟', 'mose_louis', 'mmu_laws', '12thmantalks', '🍓',
                       '💚', '7jasar', '😧', '50plus1', '0perry', '32', 'pierik_agesport', 'تُطالب', 'المُلاك', '1nda',
                       '7newsperth', '3000000', '↔', '_54', '7newssydney', 'ayman_scents', '📦', '📊', 'english_atl',
                       '4886889', 'bruce_levell', '700m', 'lewiscox_star', '_H3NDRY', '1800', '595', '194', '18H10',
                       '1819R', 'a_claudereitz', '🖋', 'jp__be', 'political_wasp', '🦊', '✔', '️what', '️if', '10000',
                       'l_cetta', '72h', '4billion', 'alex_pd', 'biko_dz', '1_plate_biryani', 'oli_holmes', '67book',
                       'nana_elkh', '👂', 'S2', 'E71', '1890s', '\U000e0077', '\U000e006c', '\U000e0073', 'england_rl',
                       'liv_fit', '2010', 'craig_pankhurst', 'monkeys_show', '3liga', '2889', '224m', 'm_star_online',
                       'lerato_mkhondo', '1894', '200k', 'manutd_id', 'arsenalfc_fl', '112k', '13th', 'cb_ignoranza',
                       '1411nico', '22:30', '2XL', '☎', '7:30', '5livesport', '🪐', '🌑', 'you_m', '18280',
                       'kevin_maguire', '🙈', 'olivier_truchot', '749', 'inafr_officiel', '1976', 'lpha_bloke',
                       '111words', 'ochim_victor', '📷', '25th', 'lazio_uk', 'the_lutonian', 'rojos_municipal', '️VAR',
                       '️changes', '️sanctions', 'keir_starmer', 'พรีเมียร์ลีก', 'ลงดาบเด้งตัวแทนบิ๊ก',
                       'ของสโมสรอย่างแมนเชสเตอร์', 'ซิตี้', 'แมนเชสเตอร์', 'ยูไนเต็ด', 'เชลซี', 'ลิเวอร์พูล',
                       'อาร์เซนอล', 'ฮอตสเปอร์', 'พ้นตำแหน่งเซ่นพิษที่พวกเขาเป็นหนึ่งในทีมร่วมก่อตั้งซูเปอร์ลีก', '⛔',
                       'rash_podcast', '6amclub', '30pm', '889', 'i̇stanbul', '1xbet_campany', '7pm', '1893',
                       'neunzig_plus', '1ce', '100m', '500k', '1_fc_nuernberg', '10betsports', '1️⃣', '2️⃣', 'inter_en',
                       '1218', 'coolboy_lanre', '️lattuale', 'tim_ecksteen', '365scores', 'alamin_ys',
                       'اراضي_شرق_الرياض', 'مرزوقه_الحربي', 'هند_القحطاني_تعود_للحجاب', 'تانج_والا_فيمتو', 'جده_Iلان',
                       'الدمام_الان', '05552230', 'عماله_منزليه', 'الفلبين_سيرلانكا_بنقلاديشيه_كينيا_اثيوبيا_اغندا',
                       '20tencreative', 'fcstpauli_en', '2018', '33', 'kevin_blundell', '1983', 'axel_remaster', '1955',
                       '🙂', '؟', 'ريال_مدريد', 'انتر_نابولي', 'مانشستر_يونايتد', '24ca', '_graydorian', '324', '311',
                       '3bill', 'michael_i_jones', '️superleague', '️uefas', '2707_', 'sam_inkersoletm', 'andrew_vas',
                       '777', 'italian_average', '🍪', 'forb_english', '👫', '100s', '9fowler', 'ian_rush', '78', '48h',
                       'a_schillhaneck', '7280000', '243', '09082000056', '1kroenkeout', 't7219860live', 'lfcdt_gav',
                       'channels_sports', 'ctv_tayos', 'ctv_ceceo', '_8', '2004er', '🧠', 'dortmund_french', '2/3',
                       'DE_DON', '007', '_10', '🚗', 'bvb_goleador', '️kick', 'zrss_si', 'ep55', '2573331', 'king_fut',
                       '22nd', '23rd', 'uefacom_it', '_goalpoint', '1º', '04fussball', 'david_clarke', '1357913',
                       '4TAG', '2/36', 'gunner_x', '7bn', '10press', 'vintage_utd', '⤵', 'millar_colin', '2015',
                       '5live', '🥅', '1kminute', '🐦', '📱', '121200', '🤙', '613 750 1200', '📩', '1200', '🦈', '4H',
                       '20/04', '2footballuk', '1088', '️ex', '2025', 'vi_nl', 'giselle_zm', '🧐', 'scott_geelan',
                       'changeorg_india', '2🅰️', '°', '2014', '07190097', '150m', '232', '350m', '🍒', 'D:',
                       '️unitedcity', '🔋', 'C20', '230AH', '12V', '115000', '2017', ':p', '12promax', '12pro',
                       '12mini', 'asso_mediapi', '🚴', '👥', '🛹', '🐺', '033u', '16h', 'alamin_ghost', '307', '2003',
                       '₦', '950000', '️norwichs', '2020s', 'flohempel_darts', '🔞', '️for', '3million', '22aprile',
                       '1510', '1280', '528', '272', '2day', '00n', 'lawson_sv', 'kzs_si', 'nzs_si', 'u20', '🍑',
                       '6500', 'tk_cele', 'samurai_ando', 'shush_larawk', 'curtis_peprah', 'siya_phungula', '🙋', '🍫',
                       '6m', '38', '589m', '450m', '335m', '277m', '273m', '📽', '21aprile', '28th',
                       '4209505', '95nufc', '7⃣', '3⃣0⃣pm', 'me_granturco', '️is', 'liberty_media', '25anni', '1/4',
                       '😷', 'FM21', '🖊', 'fm21', '2013', '📄', '𝟏𝟏𝟎𝟎', '·', '155', 's_roche', '60', '120', '909',
                       'HALIT_KARAGOOZ', '130', '365', 'zonal_marking', '️ryan', '100celleasr', '_dajedepunta_',
                       'football__tweet', '🗑', '️WBA', '️top', 'OE3', 'oe3wecker', 'lobs_sport_biz', '386m', '202m',
                       '771m', '177bn', '757m', '752m', '247m', '406', '0526', '1965wendy', 'lola_united', '111',
                       '_16',
                       '123tupac', '371', 'rio_f', '79', '8e', ':d', '59', '52', '46', '']

# bracket special chars for RE compares. RE and python compare (if x in Y) different
JUNC_PUNC = "[*+%;',]"
XTRA_PUNC = "([.!?]+)"
END_PUNC = "[.!?]"      # keep ! and ?, they both have effect on tweet sentiment
PUNC_STR = ["*", "+", "%", ":", ";", "/", "|", ",", "'"]
GS_SENT_END: list = ["!", "?", ";", ".", "..", "..."]

# capture special Tweet text: user_mention, hashtag, urls, stuff inside paras, punctuation
GS_PAREN = "\((.+?)\)"
GS_URL = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b'\
         r'[-a-zA-Z0-9()@:%_\+.~#?&//=]*'
GS_MENT = "@(\w*)"                      # find user mentions as in '@UserName'
GS_HASH = "[#](\w*)"
GS_UCS4 = r"\\\\[x][0-9,a-f]{4}"        # find ucs-2 aka double-byte characters
GS_UCS = "\\u[0-9,a-f]{4}"
UCS_SYM = "\\[u][0,2]{2}[0-9,a-f]{2}"

# contractions expansions, includes forms with missing apostrophe
GS_CONTRACT = {
    "-":" ",
    "ain't": "aint",
    "aren't": "are not",
    "arent": "are not",
    "can't": "can not",
    "cant": "can not",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "dont": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "isnt": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it'll've": "it will have",
    "its": "it is",
    "it's": "it is",            # the contraction is often mis-spelled
    "let's": "let us",
    "ma'am": "mam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "o'clock": "oclock",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "theyll": "they will",
    "they're": "they are",
    "theyre": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "whats": "what is",
    "what've": "what have",
    "when's": "when is",
    "whens": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "whos ": "who is ",
    "who've": "who have",
    "why's": "why is",
    "won't": "will not",
    "wont": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "y'all": "yall",
    "you'd": "you would",
    "youd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}

# repr(xx).strip("'") displays char represented by \uxxxx code
GS_EMOJI: dict = {"\ud83d\udea8": "🚨",
                  "\ud83e\udd23": "🤣",
                  "\u26aa\ufe0f": "⚪",
                  "\u26a0\ufe0f": "⚠",
                  "\u26BD\uFE0F": "⚽️",
                  "\u2b07\ufe0f": "⬇",
                  "\ud83e\udd2c": "🤬",  # angry, cussing head
                  "\ud83d\udcca": "📊",
                  "\ud83d\udde3\ufe0f": "🗣",
                  "\ud83d\udeab": "🚫",
                  "\ud83c\uddea\ud83c\uddfa": "🇪🇺",
                  "\ud83c\udde9\ud83c\uddea": "🇩🇪",
                  "\ud83d\ude4c": "🙌 ",
                  "\ud83d\udd34\u26aa\ufe0f": "🔴⚪",
                  "\ud83d\udd34": "🔴 ",
                  "\ud83d\udeab\ud83d\udd35": "🚫🔵",
                  "\ud83e\udd21": "🤡",
                  "\ud83d\udc80": "💀",
                  "\ud83d\udc51": "👑"
                  }

# GS_UCS2 shows ucs-1 symbol equivalent to ucs-2 if symbol exists
GS_UCS2: dict = {"\u003b":  ";",
                 "\u003c":  "<",
                 "\u003e":  ">",
                 r"\u003f":  r"?",
                 r"\u0040":  r"@",
                 r"\u00a1":  r"!",       # '¡'
                 r"\u00a2":  "",         # '¢'
                 r"\u00a3":  "brP",      # '£'
                 r"\u00a4":  "",         # '¤'
                 r"\u00a6":  r":",        # '¦'
                 r"\u00a8":  "",             # unlaut  '¨'
                 r"\u00a9":  "cpyrt",        # '©'
                 r"\u00ae":  "reg copyrt",     # reg copyrt  '®'
                 r"\u00b6": r"<p>",          # para mark '¶'
                 r"\u00b8": r".",           # period "."
                 r"\u00bd":  "1/2",          # symbol '½'
                 r"\u00bf":  "",             # spanish inverted question  '¿'
                 r"\u00e0":  "a",            # a with accent grave  'à'
                 r"\u00e7":  "c",            # c with lower accent   "ç"
                 r"\u2012":  "-",
                 r"\u2013":  "–",
                 r"\u2014":  "–",
                 r"\u2015":  "–",
                 r"\u2016":  "",          # '‖'
                 r"\u2017":  "",          # '‗'
                 r"\u2018": r"'",
                 r"\u2019": r"'",
                 r"\u201a": r",",
                 r"\u201b": r"'",
                 r"\u201c": r"'",
                 r"\u201d": r"'",
                 r"\u201e": r"'",
                 r"\u201f": r"'",
                 }

emoji_dict: dict = {
                    ":-)"  : "basic smiley",
                    ":)"   : "midget smiley",
                    ",-)"  : "winking smiley",
                    "(-:"  : "left hand smiley",
                    "(:-)" : "big face smiley",
                    ":-("  : "sad face",
                    ":-(-" : "very sad face",
                    "8-O"  : "omg face",
                    "B-)"  : "smiley with glasses",
                    ":-)>" : "bearded smiley",
                    "'-)"  : "winking smiley",
                    ":-#"  : "my lips are scaled",
                    ":-*"  : "kiss",
                    ":-/"  : "skeptical smiley",
                    ":->"  : "sarcastic smiley",
                    ":-@"  : "screaming smiley",
                    ":-V"  : "shouting smiley",
                    ":-X"  : "a big wet kiss",
                    ":-\\" : "undecided smiley",
                    ":-]"  : "smiley blockhead",
                    ";-(-" : "crying sad face",
                    ">;->" : "lewd remark",
                    ";^)"  : "smirking smiley",
                    "%-)"  : "too many screens",
                    "):-(-": "nordic smiley",
                    ":-&"  : "tongue tied",
                    ":-O"  : "talkaktive smiley",
                    "+:-)" : "priest smiley",
                    "O:-)" : "angel smiley",
                    ":-<:" : "walrus smiley",
                    ":-E"  : "bucktoothed vampire",
                    ":-Q"  : "smoking smiley",
                    ":-}X" : "bowtie smiley",
                    ":-["  : "vampire smiley",
                    ":-{-" : "mustache smiley",
                    ":-{}" : "smiley wears lipstick",
                    ":^)"  : "smiley with personality",
                    "<:-l" : "dunce smiley",
                    ":=)"  : "orangutan smiley",
                    ">:->" : "devilish smiley",
                    ">:-l" : "klingon smiley",
                    "@:-)" : "smiley wearing turban",
                    "@:-}" : "smiley with hairdo",
                    "C=:-)": "chef smiley",
                    "X:-)" : "smiley with propeller beanie",
                    "[:-)" : "smiley with earbuds",
                    "[:]"  : "robot smiley",
                    "{:-)" : "smiley wears toupee",
                    "l^o"  : "hepcat smiley",
                    "}:^)" : "pointy nosed smiley",
                    "(:-(" : "saddest smiley",
                    ":-(=)": "bucktooth smiley",
                    "O-)"  : "message from cyclops",
                    ":-3"  : "handlebar mustache smiley",
                    ":-="  : "beaver smiley",
                    "P-("  : "pirate smiley",
                    "?-("  : "black eye",
                    "d:-)" : "baseball smiley",
                    ":8)"  : "piggy smiley",
                    ":-7"  : "smirking smiley",
                    "):-)" : "impish smiley",
                    ":/\\)": "bignose smiley",
                    ":-(*)": "vomit face",
                    ":(-"  : "turtle smiley",
                    ":,("  : "crying smiley",
                    ":-S"  : "confuzled face",
                    ":-[ " : "unsmiley blockhead",
                    ":-C"  : "real unhappy smiley",
                    ":-t"  : "pouting smiley",
                    ":-W"  : "forked tongue",
                    "X-("  : "brain dead" }

IDIOM_MODS = {'darth vader': -2.5, 'male privilege': -2.5, "good guys": 0.5}
VADER_MODS = {"amf":-2.0, "sociopathic": -2.5, "cartel": -1.0, "ideologues": -0.5,
             "blunder": -0.5, "commodotize": -0.5}

"""
	eyes = "[8:=;]"
	nose = "['`\-]?"
	smile = "\[|[)\]"
	frown = \(+|\)+#
	neutral = [\/|l*]/
	
	elongated = \b(\S*?)(.)\2{2,}\b     # repetition of last letter ex 'wayyy cool'

EMOJI_2BYTE = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|\
                            ([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')

GS_SMILE = re.compile(r"(\s?:X|:|;|=)(?:-)?(?:\)+|\(|O|D|P|S|\\|\/\s){1,}", re.IGNORECASE)

emoji_dict: dict = {
    :-) - basic smiley
    :) - midget smiley
    ,-) - winking happy smiley
    (-: - left hand smiley
    (:-) - smiley big face
    (:-( - very unhappy smiley
    ,-} - wry and winking smiley
    8-O - Omigod
    '-) - winking smiley
    :-# - my lips are scaled
    :-* - kiss
    :-/ - skeptical smiley
    :-> - sarcastic smiley
    :-@ - screaming smiley
    :-d - said with a smile
    :-V - shouting smiley
    :-X - a big wet kiss
    :-\\ - undecided smiley
    :-] - smiley blockhead
    ;-( - crying smiley
    >;-> - a very lewd remark was just made
    ;^) - smirking smiley
    %-) - smiley after staring at a screen for 15 hours straight
    ):-( - nordic smiley
    3:] - Lucy my pet dog smiley
    :-& - tongue tied
    8:-) - little girl smiley
    :-)8< - big girl smiley
    :-O - talkaktive smiley
    :-6 - smiley after eating something spicy
    +:-) - priest smiley
    O:-) - angel smiley
    :-< - walrus smiley
    :-? - smiley smokes a pipe
    :-E - bucktoothed vampire
    :-Q - smoking smiley
    :-}X - bow tie-wearing smiley
    :-[ - vampire smiley
    :-a - smiley touching her tongue to her nose
    :-{ - mustache
    :-{} - smiley wears lipstick
    :^) - smiley with a personality
    <:-l - dunce smiley
    :=) - orangutan smiley
    >:-> - devilish smiley
    >:-l - klingon smiley
    @:-) - smiley wearing a turban
    @:-} - smiley just back from the hairdresser
    C=:-) - chef smiley
    X:-) - little kid with a propeller beanie
    [:-) - smiley wearing a walkman
    [:] - robot smiley
    {:-) - smiley wears a toupee
    l^o - hepcat smiley
    }:^#) - pointy nosed smiley
    (:-( - the saddest smiley
    :-(=) - bucktooth smiley
    O-) - message from cyclops
    :-3 - handlebar mustache smiley
    : = - beaver smiley
    :-" - whistling smiley
    P-( - pirate smiley
    ?-( - black eye
    d:-) - baseball smiley
    :8) - pigish smiley
    :-7 - smirking smiley
    ):-) - impish smiley
    :/\\) - extremely bignosed smiley
    ([( - Robocop
    :-(*) - that comment made me sick
    :( - sad-turtle smiley
    :,( - crying smiley
    :-( - boo hoo
    :-S - what you say makes no sense
    :-[ - un-smiley blockhead
    :-C - real unhappy smiley
    :-r - smiley raspberry
    :-W - speak with forked tongue
    X-( - you are brain dead
    l-O - smiley is yawning
    l:-O - flattop loudmouth smiley
    $-) - yuppie smiley
    :-! - foot in mouth
    :----} - you lie like pinnochio
    O-) - smiley after smoking a banana
    =:-) - smiley is a punk
    =:-( - real punks never smile
    3:[ - pit bull smiley
    8<:-) - smiley is a wizard
    :#) - drunk smiley
    8-# - dead smiley
    B-) - smiley wears glasses
    8-) - smiley with big eyes...perhaps wearing contact lenses...
    H-) - cross-eyed smiley
    ]-I - smiley wearing sunglasses (cool...therefore no smile, only a smirk)
    +-( - smiley, shot between the eyes
}
"""
