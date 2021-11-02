# coding=utf-8
"""
methods for analyzing and visualizing Tweets which have been pre-processed.
includes tf-idf calculation:  there are variants to deriving both the tf and idf components
for a word, for determining overall tf*idf for word, and for applying it to a corpus such
as in determining threshold for unimportant words based on tf*idf

My take on variations with calculating TF:
raw frequency: simply the number of occurrences of the term
relative frequency: count of target word divided by total word count for document
distinct frequency: count of target word divided by total unique words in document
augmented frequency, count of target divided by count of top occurring word in doc.
"""

import math
from collections import OrderedDict
import pandas as pd

def calc_tf(sentlst, word_tokens: bool = False, calctyp: str = "UNIQ"):
    """
    tf = occurrences of word to other words in each tweet.
    term frequency can be: word occurrences/unique words, occurrences/total word count,
    or word occurrences versus most frequent word
    :param sentlst: list of tweet str, dict or list
    :param word_tokens: is sentlist for tweets word-tokenized?
    :param calctyp: options: target vs count UNIQ words, target count vs TOP word count,
    target count vs total COUNT of words  - in all cases a 'doc'= a tweet
    :return: dict of key=target word, val=frequency based on calctyp
    """

    def get_top_tf(tmp: list):
        """
        at end of processing, find and display the top 10 words by tf in the dataset
        :param tmp: list of dict from tf_table
        :return: None
        """
        tops: dict = {}
        for x in tmp:
            if isinstance(x, dict):
                for wrd, tf in x.items():
                    # fill up top ten dict with first 10 word frequencies
                    if len(tops) < 10:
                        tops[wrd] = tf
                    else:
                        # now, check each word to see if it should bump out one of top ten
                        newtop: dict = {}
                        for topw, toptf in tops.items():
                            if tf > toptf:
                                newtop[wrd] = tf
                            else:
                                newtop[topw] = toptf
                        # the 'working' top ten becomes the official one
                        tops = newtop
        print("    calc_tf Top Ten list of words and term frequencies:")
        for topw, topnum in zip(tops.items(), range(10)):
            print(" %d. term %s had a term frequency of %.2f" % (topnum, topw[0], topw[1]))
        print(" ")

    def do_fq(sent):
        """
        calculates dict of key:word, value:occurrences for each Tweet
        :param sent: text of tweet
        :return: dict with word : word count
        """
        freq_table = {}
        if isinstance(sent, str):
            wrds: list = sent.split()
        else:
            # input already word-tokenized...
            wrds: list = sent
        for w in wrds:
            if w.isalpha():
                w = w.lower()
            if w in freq_table:
                freq_table[w] += 1
            else:
                freq_table[w] = 1
        return freq_table

    tf_table: list = []
    tw_tot = len(sentlst)
    tf_low: int = 0
    print("calc_tf creating term frequency for words in %d Tweets:" % tw_tot)
    if calctyp == "UNIQ":
        print("    using ratio of target to Unique words for term frequency")
    elif calctyp == "TOP":
        print("    using ratio of target to most frequent (Top) word for term frequency")
    elif calctyp == "COUNT":
        print("    using ratio of target to total words in tweet for term frequency")
    else:
        print("calc_tf did not receive valid calctyp for term frequency- exiting...")
        return None

    for x in range(tw_tot):
        if word_tokens:
            fq_dct = do_fq(sentlst[x])
        else:
            if isinstance(sentlst[x], dict):
                tfdoc: str = sentlst[x]['text']
            elif isinstance(sentlst[x], list):
                tfdoc: str = " ".join([w for w in sentlst[x]])
            else:
                # assume list is 'sentence tokenized' aka a list of strings
                tfdoc: str = sentlst[x]
            fq_dct = do_fq(tfdoc)
        if len(fq_dct) < 2:
            tf_low += 1

        if fq_dct:
            if calctyp == "UNIQ":
                denom: int = len(fq_dct)
            elif calctyp == "TOP":
                denom: int = max(fq_dct.values())
            elif calctyp == "COUNT":
                denom: int = sum(fq_dct.values())
            else:
                print("calc_tf did not receive a Valid calctyp parameter!: %s" % calctyp)
                break

            tf_tmp: dict = {}
            for word, count in fq_dct.items():
                tf_tmp[word] = count / denom
        else:
            continue

        print("    %d words being added for tweet to tf_table" % len(tf_tmp))
        tf_table.append(tf_tmp)

    print("    calc_tf has %d tweets with 0 or 1 tokens" % tf_low)
    print("    calc_tf is returning %d rows in tf_table" % len(tf_table))

    return tf_table

def count_tweets_for_word(word_tf: dict):
    """
    identifies number of Tweets ('docs') in which target word occurs
    :param word_tf:
    :return: OrderedDict descending order of key=word : value=count Tweets where word appears
    """
    tweets_per_word = {}
    for sent, tf in word_tf.items():
        for word in tf:
            if word in tweets_per_word:
                tweets_per_word[word] += 1
            else:
                tweets_per_word[word] = 1

    w_descend: list = sorted(tweets_per_word, key=lambda x: tweets_per_word.get(x), reverse=True)
    w_docs: OrderedDict = OrderedDict([(k, tweets_per_word[k]) for k in w_descend])

    return w_docs

def calc_idf(word_tf: dict, docs_word: OrderedDict):
    """
    idf = natural log of (total number of docs / docs in which target word occurs)
    :param word_tf: dict returned by calc_tf
    :param docs_word: OrderedDict returned by count_tweets_for_word
    :return: dict of dict, each tweet has dict of inverse document frequency values
    """
    idf_matrix = {}
    doc_count: int = len(word_tf)
    for sent, tf in word_tf.items():
        idf_table = {}
        for word in tf.keys():
            idf_table[word] = math.log(doc_count / int(docs_word[word]))
        idf_matrix[sent] = idf_table

    return idf_matrix

def calc_tf_idf(tf_matrix, idf_matrix):
    """
    creates dict of doc dict, each doc dict has key:val pais of: word : tf*idf value
    :param tf_matrix: returned by calc_tf
    :param idf_matrix: returned by calc_idf
    :return: dict of dict with word and tf*idf score for each doc in corpus
    """
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)
        tf_idf_matrix[sent1] = tf_idf_table
    return tf_idf_matrix

def calc_single_tfidf(tfi_dct: dict, calctyp: str="SUM"):
    """
    tf*idf is calculated at doc level, this calcs a single tf*idf per word at corpus level.
    my search found no standards for this.  use tf average or sum?
    sum makes sense to value occurrences, offset by the idf penalty for commodity words.

    :param tfi_dct: the dict produced from calc_tf_idf
    :param calctyp: "SUM" or "AVG" to indicate how to aggregate values for target word
    :return: dict of words sorted by descending tfidf score
    """
    wrd_sc: dict = {}

    if calctyp == "SUM":
        for this in tfi_dct.values():
            for wrd, val in this.items():
                if wrd in wrd_sc:
                    wrd_sc[wrd] = wrd_sc[wrd] + val
                else:
                    wrd_sc[wrd] = val
    elif calctyp == "AVG":
        for this in tfi_dct.values():
            for wrd, val in this.items():
                if wrd in wrd_sc:
                    wrd_sc[wrd].append(val)
                else:
                    wrd_sc[wrd] = [val]

        tmp_val: dict = {}
        for wrd, vals in wrd_sc.items():
            tmp_val[wrd] = sum(vals) / len(vals)
        wrd_sc = tmp_val

    srtd: list = sorted(wrd_sc, key=lambda x: wrd_sc.get(x), reverse=True)
    wrd_sc: OrderedDict = {k: wrd_sc[k] for k in srtd}

    return wrd_sc

def do_tfidf_stops(tf_dct: dict):
    """
    from the tfi_final dict (word:avg tfidf score), create stop list from words with
    tfidf value below the average for all words.
    :param tf_dct:
    :return:
    """
    tfiavg = math.fsum(tf_dct.values()) / len(tf_dct)

    tfi_stops: list = []
    for k, v in tf_dct.items():
        if v < tfiavg:
            tfi_stops.append(k)
    return tfi_stops

def get_combined_toplist(qrrlst: list, favelst: list, sntlst: list, debug: bool=False):
    """
    use results of get_pctle_qrr and get_pctle_fave to create combined 'top' tweet list.
    adds key of 'type' to tweet dict, 7 values: 'q', 'f', 's', 'qf', 'qs', 'fs', 'qfs'
    indicates top score for f=fave, q=qrr count, s=sentiment score.
    for selective plotting of tweets which had high influence in topic threads during dates

    :param sntlst: list of dict of high sentiment Tweets
    :param qrrlst: list of dict of high qrr count Tweets
    :param favelst: list of dict of high fave count Tweets
    :param debug: bool whether to print debugging information
    :return: list of dict for combined top tweet list
    """
    def count_types(t_l: list):
        """
        inner function to count types of records making up combined toplist
        """
        topbins: dict = {'q': 0, 'f': 0, 's': 0, 'qf': 0, 'qs': 0, 'fs': 0, 'qfs': 0}
        for x in t_l:
            if x['type'] == 'qfs':
                topbins['qfs'] += 1
            elif x['type'] == 'qf':
                topbins['qf'] += 1
            elif x['type'] == 'qs':
                topbins['qs'] += 1
            elif x['type'] == 'fs':
                topbins['fs'] += 1
            elif x['type'] == 'q':
                topbins['q'] += 1
            elif x['type'] == 'f':
                topbins['f'] += 1
            else:
                topbins['s'] += 1

        print("    toplist Tweets by type (QRR, Favorite, and/or Sentiment:")
        print(topbins.items())
        print("")
        return

    toplen: int = len(qrrlst)
    tops_lst: list = qrrlst.copy()

    sentids: set = set(sorted(k['id'] for k in sntlst))
    faveids: set = set(sorted(k['id'] for k in favelst))
    qrrids: list = sorted(k['id'] for k in qrrlst)

    print("\nget_combined_toplist: combining high Qrr, Favorite and Sentiment")

    for x in tops_lst:
        if x['id'] in faveids:
            if x['id'] in sentids:
                x['type'] = 'qfs'
            else:
                x['type'] = 'qf'
        else:
            if x['id'] in sentids:
                x['type'] = 'qs'
            else:
                x['type'] = 'q'
        x['text'] = x['text'][:80]

    for x in favelst:
        if x['id'] not in qrrids:
            if x['id'] in sentids:
                x['type'] = 'fs'
            else:
                x['type'] = 'f'
            x['text'] = x['text'][:80]
            tops_lst.append(x)

    tops_lst = sorted(tops_lst, key=lambda x: x.get('date'), reverse=False)
    print("    toplist started with %d Tweets" %toplen)
    print("        final list has %d tweets\n" %len(tops_lst))
    count_types(tops_lst)

    return tops_lst

def final_toplist(qrrlst, favlst):
    """
    tight selection of toplist- Tweet must be in top 50% for qrr and fave counts
    :param qrrlst: list of dict with top Tweets by qrr count
    :param favlst: list of dict with top Tweets by favorite count
    :return: pd.DataFrame with highly selective toplist
    """
    qrrlen: int = len(qrrlst)
    toplst: list = []
    favlen: int = len(favlst)

    qrrlst = sorted(qrrlst, key=lambda x: x.get('qrr'), reverse=True)
    qrrids: list = [x.get('id') for x in qrrlst]
    favids: list = [x.get('id') for x in sorted(favlst, key=lambda x: x.get('fave'), reverse=True)]

    print("\nfinal_toplist: makes highly selective combined list of Tweets")
    midqrr: int = int(round((qrrlen + 1) * .5, ndigits=0))
    midfav: int = int(round((favlen + 1) * .5, ndigits=0))

    for x in range(midqrr):
        for y in range(midfav):
            if favids[y] == qrrids[x]:
                toplst.append(qrrlst[x])

    toplst = sorted(toplst, key=lambda x: x.get('date'), reverse=False)
    print("    Combined list started with %d Tweets" % qrrlen)
    toplen = len(toplst)
    print("     final list has %d tweets" % toplen)

    return toplst

def get_top_tags(twlst):
    """
    return the top user mentions and hashtags in
    :param twlst: list of dict of tweets
    :return list of hashtags, list of user mentions
    """
    if isinstance(twlst[0], dict):
        hashlst: dict = {}
        mentionlst: dict = {}
        for tw in twlst:
            if 'hashes' in tw:
                for x in tw['hashes']:
                    if str(x).lower() in hashlst:
                        hashlst[str(x).lower()] += 1
                    else:
                        hashlst[str(x).lower()] = 1
            if 'mentions' in tw:
                for x in tw['mentions']:
                    if str(x).lower() in mentionlst:
                        mentionlst[str(x).lower()] += 1
                    else:
                        mentionlst[str(x).lower()] = 1

        hashlst = {k: hashlst[k] for k in sorted(hashlst, key=lambda x: hashlst[x], reverse=True)}
        mentionlst = {k: mentionlst[k] for k in sorted(mentionlst, key=lambda x: mentionlst[x], reverse=True)}

        return hashlst, mentionlst

def scrub_cloud(twl: list):
    """
    scrub_text can perform numerous text removal or modification tasks on tweets,
    there is tweet-specific content handled here which can be optionally commented out
    if resulting corpus loses too much detail for downstream tasks like sentiment analysis

    :param tweetxt: str from text field of tweet
    :return: list of words OR str of words if rtn_list= False
    """
    from tweet_data_dict import GS_CONTRACT, GS_UCS2, JUNC_PUNC, PUNC_STR
    import re
    tmpl: list = []
    for tweetxt in twl:
        if isinstance(tweetxt, str):
            # remove newlines in tweets, they cause a mess with many tasks
            tweetxt = tweetxt.replace("\n", " ")
            # remove standalone period, no need in a tweet
            tweetxt = re.sub("\.", " ", tweetxt)
            # expand contractions using custom dict of contractions
            for k, v in GS_CONTRACT.items():
                tweetxt = re.sub(k, v, tweetxt)
            # convert ucs-2 chars appearing in english tweets
            for k, v in GS_UCS2.items():
                tweetxt = re.sub(k, v, tweetxt)
            # parsing tweet punc: don't want to lose sentiment or emotion
            tweetxt = re.sub(JUNC_PUNC, "", tweetxt)
            # remove spurious symbols
            for p in PUNC_STR:
                tweetxt = tweetxt.strip(p)
            splitstr = tweetxt.split()
            cleanstr: str = ""
            for w in splitstr:
                if len(w) < 2:
                    continue
                # lower case all alpha words
                if str(w).isalpha():
                    w: str = str(w).lower()
                    cleanstr = cleanstr + " " + w
            # remove leading or trailing whitespace:
            tweetxt = cleanstr.strip()

            # parameter "utf-8" escapes ucs-2 (\uxxxx), "ascii" removes multi-byte
            binstr = tweetxt.encode("ascii", "ignore")
            tweetxt = binstr.decode()

        tmpl.append(tweetxt)

    return tmpl

def get_attib_distribution(df: pd.DataFrame):
    """
    Fx to look at statistical distribution of various metrics like Tweet sentiment.
    most of these metrics will be passed from columns in pandas DataFrames.
        id_vars: Any = None,    columns to use as identifiers
         value_vars: Any = None,  Column(s) to unpivot - the values to plot
         var_name: Any = None,  defaults to 'variable'
         value_name: Any = 'value', name of column which contains all values
         ignore_index: Any = True
    :param df: pandas DataFrame slice
    :return:
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    tmpdf: pd.DataFrame = df.loc[df.sent.notna(), ['neg', 'pos', 'compound']]

    dfmelt = tmpdf.melt(value_vars=['neg', 'pos', 'compound'],
                     var_name='vader_type', ignore_index=True)
    fg = sns.FacetGrid(dfmelt, col='vader_type', margin_titles=True)
    figplt = fg.map(sns.distplot, 'value')
    figplt = fg.map(plt.plot, 'vader_type', 'value')

    fighist = tmpdf.hist(figsize=(16, 20), bins=100, xlabelsize=8, ylabelsize=8)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow()
    plt.show()

    return