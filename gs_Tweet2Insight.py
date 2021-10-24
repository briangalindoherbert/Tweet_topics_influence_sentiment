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

def calc_tf(sentlst, word_tokens: bool=False, calctyp: str="UNIQ", debug: bool=False):
    """
    tf = occurrences of word to other words in each tweet.
    term frequency can be: word occurrences/unique words, occurrences/total word count,
    or word occurrences versus most frequent word
    :param sentlst: list of tweet str, dict or list
    :param word_tokens: is sentlist for tweets word-tokenized?
    :param calctyp: options: target vs count UNIQ words, target count vs TOP word count,
    target count vs total COUNT of words  - in all cases a 'doc'= a tweet
    :param debug: boolean if True prints status messages
    :return: dict of key=target word, val=frequency based on calctyp
    """
    def do_fq(sent):
        """
        calculates dict of key:word, value:word count from text of tweet
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
                if not w.isupper():
                    w = w.lower()
            if w in freq_table:
                freq_table[w] += 1
            else:
                freq_table[w] = 1
        return freq_table

    fq_dct: dict = {}
    tf_table: dict = {}
    tw_tot = len(sentlst)
    if debug:
        print("calc_tf creating term frequency for %d Tweets:" %tw_tot)
    for x in range(tw_tot):
        if word_tokens:
            fq_dct = do_fq(sentlst[x])
            if debug:
                if len(fq_dct) < 2:
                    print("calc_tf %d got %d distinct word tokens" % (x, len(fq_dct)))
        else:
            if isinstance(sentlst[x], dict):
                tfdoc: str = sentlst[x]['text']
            elif isinstance(sentlst[x], list):
                tfdoc: str = " ".join([w for w in sentlst[x]])
            else:
                tfdoc: str = sentlst[x]
            fq_dct = do_fq(tfdoc)
            if debug:
                if len(fq_dct) < 2:
                    print("calc_tf %d got %d distinct words" %(x,len(fq_dct)))

        if fq_dct:
            if calctyp == "UNIQ":
                denom: int = len(fq_dct)
            elif calctyp == "TOP":
                denom: int = max(fq_dct.values())
            elif calctyp == "COUNT":
                denom: int = sum(fq_dct.values())
            else:
                print("calc_tf did not receive a Valid calctyp parameter!: %s" %calctyp)
                break

            tf_tmp: dict = {}
            for word, count in fq_dct.items():
                if calctyp == "UNIQ":
                    tf_tmp[word] = 1 / denom
                else:
                    tf_tmp[word] = count / denom
        else:
            tf_tmp: dict = {"":0}
        tf_table[x] = tf_tmp

    return tf_table

def count_tweets_for_word(word_tf: dict):
    """
    identifies number of Tweets ('docs') in which target word occurs
    :param word_tf:
    :return: OrderedDict descending order of key=word : value=count Tweets where word appears
    """
    docs_per_word = {}
    for sent, tf in word_tf.items():
        for word in tf:
            if word in docs_per_word:
                docs_per_word[word] += 1
            else:
                docs_per_word[word] = 1

    w_descend: list = sorted(docs_per_word, key=lambda x: docs_per_word.get(x), reverse=True)
    w_docs: OrderedDict = {k: docs_per_word[k] for k in w_descend}

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

def get_combined_toplist(qrrlst, favelst: list, sntlst: list, debug: bool=False):
    """
    use results of get_pctle_qrr and get_pctle_fave to create combined 'top' tweet list.
    adds the key 'type' to dict for each tweet, to indicate if this tweet had a top
    f=fave, q=qrr count, and/or s=sentiment score.
    this function allows selective plotting of tweets which had a disproportionate
    impact on a topic during a particular period.
    :param sntlst: list of dict of high sentiment Tweets
    :param qrrlst: list of dict of high qrr count Tweets
    :param favelst: list of dict of high fave count Tweets
    :param debug: bool whether to print debugging information
    :return: list of dict for combined top tweet list
    """
    toplen: int = len(qrrlst)
    tops_lst: list = []
    tops_lst.extend(qrrlst)
    sentids: list = sorted(k['id'] for k in sntlst)
    faveids: list = sorted(k['id'] for k in favelst)
    qrrids: list  = sorted(k['id'] for k in qrrlst)
    print("\nget_combined_toplist: list by qrr and favorite counts + sentiment score")

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
    print("    get_combined_toplist started with %d Tweets" %toplen)
    toplen = len(tops_lst)
    print("        final list has %d tweets\n" %toplen)

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

def cleanup_for_cloud(twl: list):
    tmplst: list = []
    for el in twl:
        if isinstance(el, list):
            eltmp: list = []
            for wrd in el:
                if len(wrd) < 3:
                    continue
                else:
                    if str(wrd).isalpha():
                        if str(wrd).isupper():
                            wrd: str = str(wrd).lower()
                            eltmp.append(wrd)
            if len(eltmp) > 1:
                tmplst.append(eltmp)
    return tmplst
