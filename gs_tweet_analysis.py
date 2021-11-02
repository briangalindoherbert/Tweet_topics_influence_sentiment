# coding=utf-8
"""
gs_tweet_analysis is part II of utils for tweets, the first being gs_nlp_util.
gs_tweet_analysis builds word counts, grammar analysis, and other higher level functions
"""

import io
from math import fabs
import datetime as dt
import pandas as pd
import nltk.data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tweet_data_dict import IDIOM_MODS, VADER_MODS
# from pandas.api.extensions import register_extension_dtype
# from pandas.api.types import is_datetime64_any_dtype as is_dt64

nltk.data.path.append('/Users/bgh/dev/NLP/nltk_data')
nltk.download('vader_lexicon', download_dir="/Users/bgh/dev/NLP/nltk_data")
Vsi = SentimentIntensityAnalyzer()
# TODO: manage these Vader constants in a function or class!
# Vader Constants derived from empirical research, but can be changed as needed:
# B_INCR = 0.293
# B_DECR = -0.293
# C_INCR = 0.733
# N_SCALAR = -0.74

for k, v in VADER_MODS.items():
    Vsi.lexicon[k] = v
for k, v in IDIOM_MODS.items():
    Vsi.constants.SPECIAL_CASE_IDIOMS[k] = v

def filter_rt_and_qt(twpre: list):
    """
    better version of limit_retweets, use this fx to search RT's and QT's in the dataset
      -identifies Tweet ID for QT's and RT's, and aggregates by originating Tweet ID
      -writes an 'ordinal id' number to each record to help match records back to raw
      dataset, which is hard to do once sorting or filtering take place
      -can adjust to limit how many QT's or RTs for a single TweetID
      to write to dataset
      -writes a joined, or aggregated record of all Quoted Tweet comments for each
       originating Tweet ID, plus the original tweet text

    :param twpre: list of dict of tweets prior to any filtering
    :return: filt_lst: tweet list after de-duping, rt_count: RT id's and number of copies
        in dataset, qt_count: QT id's and num of copies in dataset, qt_merge: merged Quoted
        Tweet comments by original Tweet ID
    """
    qt_merge: dict = {}
    qt_count: dict = {}
    filt_lst: list = []
    rt_count: dict = {}
    id_find_rt: int = 0
    id_find_qt: int = 0
    id_set: set = {x['id'] for x in twpre}
    ds_len: int = len(twpre)

    def append_ordinal(twl):
        """
        append an 'ord_id' field to to each tweet record.  this allows identification of
        raw records by something other than their tweet_id. once the raw records are either
        sorted or delete/adds take place, need ord_id to match on original
        :param twl: list of dict of raw tweets
        :return: list of dict with sequential ord_id field appended
        """

        ordn: int = 0
        tmplst: list = []
        for twx in twl:
            if isinstance(twx, dict):
                twx['ord_id']: int = int(ordn)
            tmplst.append(twx)
            ordn += 1

        return tmplst

    def match_ids(twrec: dict, countx: dict, mrgx: dict = None, typ: str = "qt"):
        """
        inner function to look up originating tweets for RTs and QTs in raw dataset
        if original is found, we check its metrics vs what we have in QT or RT record
        RT's provide NO new info except retweet metrics, remove duplicate retweets
        also- this combines quote comments for each originating tweet ID, for nlp experiments!
        :param twrec: the current tweet record passed from the iterator below
        :param countx: dict key=ID, value=count of duplicates in dataset
        :param mrgx: dict by tweet ID of all text comments derived from original
        :param typ: "rt" or "qt"
        :return: dict of id counts, and dict of merged text for QT
        """
        if tmpdct[typ + '_id'] in id_set:
            xindex: int = int(twrec['ord_id'])
            ord_lookup: dict = twpre[xindex]
            if twrec.get(typ + '_qrr') > ord_lookup['qrr']:
                twpre[xindex]['qrr'] = twrec.get(typ + '_qrr')
            if twrec.get(typ + '_fave') > ord_lookup['fave']:
                twpre[xindex]['qrr'] = twrec.get(typ + '_qrr')

        else:
            if twrec[typ + '_id'] in countx:
                countx[twrec[typ + '_id']] += 1
                if mrgx:
                    mrgx[twrec[typ + '_id']].append(twrec['text'])
            else:
                countx[twrec[typ + '_id']] = 1
                if mrgx:
                    mrgx[twrec[typ + '_id']] = [twrec['qt_text'], twrec['text']]

        if typ == "rt":
            return countx

        return countx, mrgx

    tw_ord: list = append_ordinal(twpre)
    for twx in tw_ord:
        tmpdct: dict = twx
        if tmpdct.get('qt_id'):
            # having the ordinal number lets us look up raw tweet in list by positon!
            qt_count, qt_merge = match_ids(twx, countx=qt_count, mrgx=qt_merge, typ="qt")
            if tmpdct.get('qt_id') in id_set:
                id_find_qt += 1

        if twx.get('rt_id'):
            rt_chk: int = len(rt_count)
            rt_count = match_ids(twx, countx=rt_count, typ="rt")
            if tmpdct.get('rt_id') in id_set:
                id_find_rt += 1
            # if no new recs in rt_count after Fx, RT was a dupe - don't add it
            if len(rt_count) == rt_chk:
                continue

        filt_lst.append(tmpdct)

    if len(rt_count) > 0:
        # sort dict (key=retweet ID, value=number of copies in data) from high to low value
        rt_count = {k: rt_count[k] for k in sorted(rt_count, key=lambda x: rt_count[x], reverse=True)}
        rt_tot: int = sum(rt_count.values())
        print("    %d ReTweets in raw dataset for %d original Tweets" % (rt_tot, len(rt_count)))
        for ctx, ctr in zip(rt_count.values(), range(1)):
            if ctr == 0:
                print("    most duplicate Retweets on a Tweet is %d " % ctx)
        print("      Originating Tweet found for %d RTs in dataset\n" % id_find_rt)

        if len(qt_count) > 0:
            qt_count = {k: qt_count[k] for k in sorted(qt_count, key=lambda x: qt_count[x], reverse=True)}
            qt_tot: int = sum(qt_count.values())
            print("    %d Quoted Tweets found for %d original Tweets" % (qt_tot, len(qt_count)))
            for qtx, ctr in zip(qt_count.values(), range(1)):
                if ctr == 0:
                    print("    most Quoted Comments for a Tweet is %d" % qtx)
            print("      Originating Tweet found for %d QTs in dataset\n" % id_find_qt)

        print("filter_rt_qt: %d raw records in, %d records out" % (ds_len, len(filt_lst)))
        return filt_lst, rt_count, qt_count, qt_merge
    else:
        ds_len = len(filt_lst)
        print("filter_rt_qt found No ReTweet duplicates, returning %d Tweets" % ds_len)
        return filt_lst, None, None, None

def id_missing_and_sync_metrics(tw_pre, rtcnt, twpost):
    """
    this Fx goes through a list of retweet ID's and checks them against a dataset of Tweets to
    see if we have original tweet, or other RTs or QT's for the same tweet

    it returns a list of missing tweets and reports on the importance of those missing
    tweets for the topic dataset based on metrics found.

    TODO: set 'retweet threshold'- search only if original generated X or more RTs/QTs.
    :param tw_pre: raw list of dict of tweets prior to filtering
    :param rtcnt: dict created by RT filter method, key=rt_id, val=count
    :param twpost: list of tweets- the output from RT duplicate filter
    :return:
    """
    print("\nid_missing searches for original tweets in threads and syncs influence metrics")
    missing: list = []
    not_missing: list = []
    id_set: set = {x['id'] for x in tw_pre}
    rt_len = len(rtcnt)
    rtcnt = {k: rtcnt[k] for k in sorted(rtcnt, key=lambda x: rtcnt[x], reverse=True)}

    def get_highest_count(rtrec):
        """
        within retweeted_status, Twitter provides the retweet, quote, reply, favorite metrics
        for the Originating Tweet.  this Fx finds the highest metrics we have for the tweet
        :param rtrec: the rt_id passed from outer function iterating through rt_count
        :return: dict with counts for rt_qrr and rt_fave
        """
        tmpdct: dict = {'rt_id': rtrec, 'rt_qrr': 0, 'rt_fave': 0}
        for twx in tw_pre:
            if twx['id'] == tmpdct['rt_id']:
                # if ID matches, this is the original Tweet for the retweet
                if twx.get('qrr') > tmpdct['rt_qrr']:
                    tmpdct['rt_qrr'] = twx.get('qrr')
                if twx.get('fave') > tmpdct['rt_fave']:
                    tmpdct['rt_fave'] = twx.get('fave')
            elif twx.get('rt_id'):
                if twx.get('rt_id') == tmpdct['rt_id']:
                    if twx.get('rt_qrr') > tmpdct['rt_qrr']:
                        tmpdct['rt_qrr'] = twx.get('rt_qrr')
                    if twx.get('rt_fave') > tmpdct['rt_fave']:
                        tmpdct['rt_fave'] = twx.get('rt_fave')
            elif twx.get('qt_id'):
                if twx.get('qt_id') == tmpdct['rt_id']:
                    if twx.get('qt_qrr'):
                        if twx.get('qt_qrr') > tmpdct['rt_qrr']:
                            tmpdct['rt_qrr'] = twx.get('qt_qrr')
                    if twx.get('qt_fave'):
                        if twx.get('qt_fave') > tmpdct['rt_fave']:
                            tmpdct['rt_fave'] = twx.get('qt_fave')
        return tmpdct

    def upd_rt_by_id(twid: dict, typ: str= "id"):
        """
        inner Fx which can check our retweet id against the three types of IDs in the
        raw data: id, retweet id, quoted tweet id.  if match- update core metrics
        (quote-retweet-reply-fave counts) with the highest count we have found for that ID.

        :param twid: dict ID string for a Tweet
        :param typ: typ string, either "id" or "rtid"
        :return: True if ID was found, False if not found
        """
        if typ == "rt_id":
            pref: str = "rt_"
        elif typ == "qt_id":
            pref: str = "qt_"
        else:
            # default case of typ == "id"
            pref: str = ""

        updates: bool = False
        for a_tw in twpost:
            if a_tw.get('type'):
                if twid['rt_id'] == a_tw.get('type'):
                    if twid['rt_qrr'] > a_tw[pref + 'qrr']:
                        a_tw[pref + 'qrr'] = twid['rt_qrr']
                        updates = True
                    if twid['rt_fave'] > a_tw[pref + 'fave']:
                        a_tw[pref + 'fave'] = twid['rt_fave']
                        updates = True
        return updates

    print("*   searching for matches on %d retweets" % rt_len)
    updates_made: int = 0
    for tw in rtcnt:
        rtd: dict = get_highest_count(tw)
        if rtd['rt_id'] not in id_set:
            # if ID is not in idset, this is one we should go GET
            missing.append(rtd)
            if upd_rt_by_id(rtd, "rt_id"):
                updates_made += 1
            if upd_rt_by_id(rtd, "qt_id"):
                updates_made += 1
        else:
            not_missing.append(rtd)
            if upd_rt_by_id(rtd, "id"):
                updates_made += 1
            if upd_rt_by_id(rtd, "rt_id"):
                updates_made += 1
            if upd_rt_by_id(rtd, "qt_id"):
                updates_made += 1


    print("*     found %d originating tweets" % len(not_missing))
    print("*     updated metrics on %d tweets" % updates_made)

    if len(missing) > 0:
        missing: list = sorted(missing, key=lambda x: x.get('rt_qrr')
                                                      + x.get('rt_fave'), reverse=True)
        mis_len: int = len(missing)
        # first missing record now has largest total count:
        buzz_max: int = int(missing[0].get('rt_qrr')) + int(missing[0].get('rt_fave'))
        print("*     --missing %d Tweets with influence on topic" % mis_len)
        print("      --Pass missing ID list to 'GET Tweets by ID' to add to topic dataset--")
        print("        Q-R-R-Like total for 1st missing Tweet is %d" % buzz_max)
        mid_miss: int = int(round(mis_len / 2, ndigits=0))
        mid_count: int = int(missing[mid_miss].get("rt_qrr")) + int(missing[mid_miss].get("rt_fave"))
        print("        missing Tweet %d of %d has Q-R-R/Fave of %d\n" % (mid_miss, mis_len, mid_count))

    return missing

def filter_rt_topic(twbatch: list):
    """
    better version of limit_retweets, use this fx to search RT's and QT's in the dataset
      -identifies Tweet ID for QT's and RT's, and aggregates by originating Tweet ID
      -writes an 'ordinal id' number to each record to help match records back to raw
      dataset, which is hard to do once sorting or filtering take place
      -can adjust to limit how many QT's or RTs for a single TweetID
      to write to dataset
      -writes a joined, or aggregated record of all Quoted Tweet comments for each
       originating Tweet ID, plus the original tweet text

    :param twbatch: list of dict of tweets
    :return: fitl_lst: tweet list after de-duping, rt_count: RT id's and number of copies
        in dataset, qt_count: QT id's and num of copies in dataset, qt_merge: merged Quoted
        Tweet comments by original Tweet ID
    """
    qt_merge: dict = {}
    qt_count: dict = {}
    filt_lst: list = []
    rt_count: dict = {}
    id_count: dict = {}
    ds_len: int = len(twbatch)
    stop_count: int = 0
    stopkey: list = ['classes', 'cognitive', 'course', 'curriculum', 'delineate',
                     'educational', 'educational', 'englishteacher', 'esol', 'fluent',
                     'givemesomeenglish', 'learnenglish', 'learning', 'lessons',
                     'past usage', 'peachpubs', 'poem', 'remotelearning',
                     'remotelearning', 'requestabet', 'resources', 'tesol', 'videoshorts',
                     'worksheet', 'writingskills']


    def append_ordinal(twl):
        """
        append an 'ord_id' field to to each tweet record.  allows identification of
        raw records by something other than their tweet_id, helpful to trace back to
        raw set after sorting or filtering takes place
        :param twl: list of dict of raw tweets
        :return: list of dict with sequential ord_id field appended
        """
        ordn: int = 0
        tmplst: list = []
        for twx in twl:
            if isinstance(twx, dict):
                twx['ord_id']: int = int(ordn)
            tmplst.append(twx)
            ordn += 1

        return tmplst

    print("filter_rt: processing raw dataset of %d Tweets" % ds_len)
    for twx in twbatch:
        if isinstance(twx, dict):
            tmpdct: dict = twx

        if tmpdct.get('qt_id'):
            if not tmpdct['qt_id'] in qt_count:
                # first time originating ID is found, create QT count
                qt_count[tmpdct['qt_id']] = 1
                qt_merge[tmpdct.get('qt_id')] = [tmpdct.get('qt_text'), tmpdct.get('text')]
            else:
                qt_count[tmpdct['qt_id']] += 1
                qt_merge[tmpdct.get('qt_id')].append(tmpdct.get('text'))

        if tmpdct.get('rt_id'):
            if not tmpdct['rt_id'] in rt_count:
                # first time seeing this retweet, create an rt_count rec for it
                rt_count[tmpdct['rt_id']] = 1
            else:
                # already seen this originating ID, add count, skip if more than 2
                rt_count[tmpdct['rt_id']] += 1
                if rt_count[tmpdct['rt_id']] >= 3:
                    continue

        temp_text: str = tmpdct.get('text')
        temp_text = temp_text.lower()
        for stopw in stopkey:
            if temp_text.find(stopw):
                stop_count += 1
                continue

        if not tmpdct['id'] in id_count:
            id_count[tmpdct['id']] = 1
        else:
            id_count[tmpdct['id']] += 1
            if id_count[tmpdct['id']] >= 3:
                continue

        filt_lst.append(tmpdct)

    if len(rt_count) > 0:
        # sort the list of retweet ID's by most frequent to least
        rt_count = {k: rt_count[k] for k in sorted(rt_count, key=lambda x: rt_count[x], reverse=True)}
        rt_tot: int = sum(rt_count.values())
        print("    %d ReTweets found for %d original Tweets" % (rt_tot, len(rt_count)))
        for rtdupes, ctr in zip(rt_count.values(), range(1)):
            if ctr == 0:
                maxdupe: int = rtdupes
        print("      highest number of duplicate RT's is %d " % maxdupe)
        if len(qt_count) > 0:
            qt_count = {k: qt_count[k] for k in sorted(qt_count, key=lambda x: qt_count[x], reverse=True)}
            qt_tot: int = sum(qt_count.values())
            print("    %d Quoted Tweets found for %d original Tweets" % (qt_tot, len(qt_count)))
            for rtdupes, ctr in zip(qt_count.values(), range(1)):
                if ctr == 0:
                    maxdupe: int = rtdupes
            print("      highest number of QT's for one Tweet is %d" % maxdupe)
            print("      QTmerge is a list of all QT comments per originating Tweet")
            print("    %d records were rejected as off-topic" % stop_count)
            id_ctr: int = 0
            for k, v in id_count.items():
                if v > 2:
                    id_ctr += 1
            print("    filter_rt_topic pass on %d duplicates" % id_ctr)
            print("--returning %d clean records--" % len(filt_lst))

        return filt_lst, rt_count, qt_count, qt_merge
    else:
        ds_len = len(filt_lst)
        print("filter_rt_qt found No ReTweet duplicates, returning %d Tweets" % ds_len)
        return filt_lst, None, None, None

def find_original_ids(tw_ds: list, rt_ct: dict, qt_ct: dict):
    """
    Fx goes through list of retweet ID's created by filter_rt_and_qt and looks for
    original tweet in dataset.  returns prioritized list of missing original tweets

    TODO: set 'retweet threshold'- search only if original generated X or more RTs/QTs.
    :param tw_ds: list of dict, the main Tweet dataset after filter step
    :param rt_ct: dict of RT ids and counts, created in previous step
    :param qt_ct: dict of QT ids and counts, created in previous step
    :return:
    """
    print("\nfind_original_ids: searching %d RTs and %d QTs for originating" %
          (len(rt_ct), len(qt_ct)))

    missing: list = []
    not_missing: list = []
    id_set: set = {x['id'] for x in tw_ds}
    cntr: dict = {"rthit": 0, "rtmiss": 0, "qthit": 0, "qtmiss": 0}
    rt_ct = {k: rt_ct[k] for k in sorted(rt_ct, key=lambda x: rt_ct[x], reverse=True)}

    def find_tweet(tw_id: str, typ: str="id"):
        for twx in tw_ds:
            if isinstance(twx, dict):
                if twx.get(typ) == tw_id:
                    return twx
        return None

    print("        total searches to perform =     %d" % (len(rt_ct) + len(qt_ct)))
    for findx, findtyp in zip([rt_ct, qt_ct], ["rt", "qt"]):
        for idx, cntx in findx.items():
            if idx in id_set:
                found_rec = find_tweet(idx)
                if found_rec:
                    not_missing.append(found_rec)
                    if findtyp in ["rt"]:
                        cntr["rthit"] += 1
                    elif findtyp in ["qt"]:
                        cntr["qthit"] += 1
            else:
                tmp_dct: dict = {}
                found_rec = find_tweet(idx, typ=findtyp + "_id")
                if found_rec:
                    tmp_dct["id"] = idx
                    tmp_dct["qrr"] = found_rec[findtyp + "_qrr"]
                    tmp_dct["fave"] = found_rec[findtyp + "_fave"]
                    tmp_dct['from'] = findtyp
                    missing.append(tmp_dct)
                    if findtyp in ["rt"]:
                        cntr["rtmiss"] += 1
                    elif findtyp in ["qt"]:
                        cntr["qtmiss"] += 1

    missing = sorted(missing, key=lambda x: x.get('qrr') + x.get('fave'), reverse=True)
    top_select: int = 10
    qrrfsum: int = 0
    for x, cnt in zip(missing, range(top_select)):
        qrrfsum += int(x['qrr']) + int(x['fave'])
    qrrf_avg = round(qrrfsum / top_select, ndigits=1)

    print("    Originating Tweets found for RT/QT= %d" % len(not_missing))
    print("               from RT search = %d" % cntr["rthit"])
    print("               from QT search = %d" % cntr["qthit"])
    print("    missing traceback on shared Tweets= %d" % len(missing))
    print("               have Retweet for %d" % cntr["rtmiss"])
    print("           have Quote Tweet for %d" % cntr["qtmiss"])
    print("    --importance of missing Tweets for topic dataset--")
    print(f"            top missing had Q-R-R= {missing[0]['qrr']:,}")
    print(f"                  and Likes count= {missing[0]['fave']:,}")
    print(f"    avg Q-R-R-F top {top_select:,} missing tweets= {qrrf_avg:,.1f}")
    print("    ---- use missing list with 'GET tweets by' ID Twitter API ----")

    return missing, not_missing

def get_word_freq(wrd_list: list, debug: bool=False):
    """
    create dict of distinct words (a 'vocabulary') and number of occurrences
    :param wrd_list: list of tweets, tweet can be list of words, a str, or dict
    :param debug: bool if True will print verbose status
    :return: wordfreq key:str= word, value:int= count of occurrences
    """
    wordfq: dict = {}
    wrd_total: int = 0
    for this_rec in wrd_list:
        if isinstance(this_rec, dict):
            this_seg: str = this_rec['text']
            this_seg: list = this_seg.split()
            for this_w in this_seg:
                if this_w in wordfq:
                    wordfq[this_w] += 1
                else:
                    wordfq[this_w] = 1

        elif isinstance(this_rec, str):
            this_seg: list = this_rec.split()
            for this_w in this_seg:
                wrd_total += 1
                if this_w in wordfq:
                    wordfq[this_w] += 1
                else:
                    wordfq[this_w] = 1

        elif isinstance(this_rec, list):
            for this_w in this_rec:
                wrd_total += 1
                if this_w in wordfq:
                    wordfq[this_w] += 1
                else:
                    wordfq[this_w] = 1

    if debug:
        print("%d unique words from %s total words" %(len(wordfq), "{:,}".format(wrd_total)))
    return wordfq

def count_words(wordlst: list):
    """
    count words in tweet text from list of list, dict, or str
    :param wordlst: list of tweets
    :return: word count, tweet count
    """
    wrd_count: int = 0
    tw_count: int = 0
    for tw in wordlst:
        if isinstance(tw, dict):
            tw_wrds: list = tw['text'].split()
        elif isinstance(tw, str):
            tw_wrds: list = tw.split()
        else:
            tw_wrds: list = tw
        tw_count += 1
        wrd_count += len(tw_wrds)
    return wrd_count, tw_count

def sort_freq(freqdict):
    """
    sort_freq reads word:frequency key:val pairs from dict, and returns a list sorted from
    highest to lowest frequency word
    :param freqdict:
    :return: list named aux
    """
    aux: list = []
    for k, v in freqdict.items():
        aux.append((v, k))
    aux.sort(reverse=True)

    return aux

def cloud_prep(wrd_tok_lst):
    """
    do_cloud calls this to create kind of a text blob from tweets for cloud to process
    :param wrd_tok_lst: preferably list of list of word tokens for tweets
    :return:
    """
    cloud_text = io.StringIO(newline="")
    for tok_rec in wrd_tok_lst:
        if isinstance(tok_rec, str):
            cloud_text.write(tok_rec + " ")
        else:
            for a_tw in tok_rec:
                if isinstance(a_tw, list):
                    cloud_text.write(" ".join([str(x) for x in a_tw]) + " ")
                if isinstance(a_tw, str):
                    # if simple list of text for each tweet:
                    cloud_text.write(a_tw + " ")

    return cloud_text.getvalue()

def apply_vader(sent_lst: list):
    """
    apply Vader's Valence scoring of words, symbols and phrases for social media sentiment,
    continuous negative-positive range, 4 scores: compound, neg, neutral, and pos.
    application of phrases and idioms, negation and punctuation (ex. ???).

    can add to or modify Vader 'constants' for terms and values.
    Vader is optimized to handle sentiment on short posts like Tweets.
    \n Author Credits:
    Hutto,C.J. & Gilbert,E.E. (2014). VADER: Parsimonious Rule-based Model for Sentiment
    Analysis of Social Media Text. Eighth International Conference on Weblogs and Social
    Media (ICWSM-14). Ann Arbor, MI, June 2014.

    :param sent_lst: list of dict or list of str with Tweet text
    :return: Vscores list of Vader sentiment scores, plus Tweet index info I embedded
    """

    vscores: list = []
    for snt_x in sent_lst:
        if isinstance(snt_x, list):
            tmpdct: dict = Vsi.polarity_scores(" ".join([str(x) for x in snt_x]))
        elif isinstance(snt_x, str):
            tmpdct: dict = Vsi.polarity_scores(snt_x)
        elif isinstance(snt_x, dict):
            tmpdct: dict = Vsi.polarity_scores(snt_x['text'])
            tmpdct.update(snt_x)
        else:
            print("apply_vader got incorrectly formatted Tweets as parameter")
            break
        vscores.append(tmpdct)

    cmp_tot: float = 0.0
    v_len = len(vscores)
    for vidx in range(v_len):
        v_rec = vscores[vidx]
        cmp_tot += v_rec['compound']
    cmp_avg = cmp_tot / v_len
    prnt_str: str = "Average Vader compound score = %1.2f for %d Tweets" %(cmp_avg, v_len)
    print(prnt_str)

    return vscores

def summarize_vader(vader_scores: list, top_lim: int = 10):
    """
    adds compound, negative, neutral, and positive components of sentence sentiment for a
    set of sentences or all sentences in corpus.
    :param vader_scores: list of scores built from apply_vader method
    :param top_lim: integer indicating number of top scores to summarize
    :return: None
    """
    rec_count: int = len(vader_scores)
    print("\nsummarize_vader: Top Sentiment for %d total Tweets:" %rec_count)
    print("    showing top %d compound, neutral, negative and positive sentiment" %top_lim)

    def get_top(scoretyp: str, toplimit: int):
        """
        inner Fx: get top score for score type, sort by descending absolute value
        :param toplimit: number of scores to identify, such as top 10
        :param scoretyp: str to indicate Vader compound, negative, neutral or positive
        :return:
        """
        srtd = sorted(vader_scores, key=lambda x: fabs(x.get(scoretyp)), reverse=True)
        tops: list = srtd[:toplimit]
        return tops

    def describe_score(scoretyp: str):
        """
        gives average, minimum and maximum for a type of sentiment score
        :param scoretyp: str as compound, neu, neg, or pos
        :return: n/a
        """
        typ_tot: float = sum(vader_scores[x][scoretyp] for x in range(rec_count))
        if scoretyp == "neu":
            typestr = "4. Neutral"
            typ_avg: float = typ_tot / rec_count
        elif scoretyp == "neg":
            typestr = "3. Negative"
            typ_avg: float = typ_tot / rec_count
        elif scoretyp == "pos":
            typestr = "2. Positive"
            typ_avg: float = typ_tot / rec_count
        else:
            typestr = "1. Compound (aggregate)"
            typ_avg: float = typ_tot / rec_count

        typ_min: float = min([vader_scores[x][scoretyp] for x in range(rec_count)])
        typ_max: float = max([vader_scores[x][scoretyp] for x in range(rec_count)])
        print("    %s " %typestr, end="")
        print(" Average= %1.3f, Minimum= %1.3f, Maximum= %1.3f" %(typ_avg, typ_min, typ_max))

        return

    def show_with_text(typ, tops: list):
        """
        prints applicable sentiment score along with text of Tweet
        :param typ: string with formal Vader sentiment type (neu, pos, neg, compound)
        :param tops: list of top tweets by sentiment type, number of tweets= top_lim
        :return: None
        """
        print("Printing top %d tweets by %s sentiment:" %(top_lim, typ))
        for tws in tops:
            # print("    %s sentiment= % 2.2f, '%d'" %(typ, tws[typ],
            #                                        twlst[int(tws['ord_id'])]['text']))
            print("    %s sentiment= % 1.3f on %s" % (typ, tws[typ], tws['date']))
            if typ in ['compound', 'neg']:
                print("        Tweet txt: %s" % tws['text'][:100])
        return

    for x in ["compound", "pos", "neg"]:
        describe_score(x)
        top_list = get_top(x, top_lim)
        show_with_text(x, top_list)
        print("")

    return None

def get_next_by_val(lst, field: str, val: float):
    """
    takes a list of dict, sorts by descending value of chosen field, then finds first matching
    index value (ordinal ID number) which is LESS THAN the identified target value
    :param lst: a list of dict, that is: list of Tweets where dict keys are Tweet fields
    :param field: str field name for retweet/quote count, fave count or sentiment value
    :param val: integer or float value to be found in sorted field values
    :return: ordinal index number, or -1 if error/not found
    """
    lenx: int = len(lst)
    if field in ['compound', 'neg', 'pos']:
        srtd: list = sorted(lst, key=lambda x: fabs(x.get(field)), reverse=True)
    else:
        srtd: list = sorted(lst, key=lambda x: x.get(field), reverse=True)

    for x in range(lenx):
        if field in ['compound', 'neg', 'pos']:
            if fabs(srtd[x][field]) <= fabs(val):
                return x
        else:
            if int(srtd[x][field]) <= val:
                return x
    return -1

def get_pctle_sentiment(twlst: list, ptile: int = 0, quota: int = 0):
    """
    create list of Tweets in top quartile for compound sentiment score
    :param twlst: list of Vader scores
    :param ptile: integer from 0 to 99 indicating percentile above which to include
    :param quota: alternate to percentile is to specify quota-  x records to select
    :return: list of str: Tweets in top quartile by sentiment
    """
    totlen: int = len(twlst)
    srtd: list = sorted(twlst, key=lambda x: fabs(x.get('compound')), reverse=True)
    if quota != 0:
        quota = int(quota)
        print("selecting top %d tweets by quota provided" % quota)
    else:
        top_pcnt: int = 100 - ptile
        quota = round(totlen * (top_pcnt / 100), ndigits=0)
        quota = int(quota)
    print("get_pctle_sentiment: selecting top %d Tweets out of %d" % (quota, totlen))

    tops: list = srtd[:quota]
    med_sent: float = tops[round(quota * 0.5)]['compound']
    top_sent: float = tops[0]['compound']
    sent_80: int = get_next_by_val(twlst, "compound", 0.80)
    print("    compound sentiment of 0.8 occurs at rec %d of %d" % (sent_80, totlen))
    print("    filtered: top sentiment is %1.2f, median is %1.2f" % (top_sent, med_sent))
    print("      least (abs) sentiment in filtered is: %1.3f" % tops[quota - 1]['compound'])

    return tops

def get_pctle_qrr(twlst: list, ptile: int = 0, quota: int = 0):
    """
    create list of Tweets in top quartile for qrr count
    :param twlst: list of dict of Tweets w/quoted/retweeted/reply counts
    :param ptile: integer from 0 to 99 indicating percentile above which to include
    :param quota: identify an integer number of records instead of a percentile
    :return: list of dict: Tweets in top quartile by popularity count
    """
    totlen: int = len(twlst)
    srtd: list = sorted(twlst, key=lambda x: x.get('qrr'), reverse=True)
    if quota != 0:
        quota: int = int(quota)
        print("selecting top %d tweets by quota provided" % quota)
    else:
        top_pcnt: int = 100 - ptile
        quota: int = round(totlen * (top_pcnt / 100), ndigits=0)
        quota: int = int(quota)
    print("get_pctle_qrr: getting top %d Tweets out of %d by qrr count" % (quota, totlen))

    tops: list = srtd[:quota]
    med_qrr: float = tops[round(quota * 0.5)]['qrr']
    top_qrr: float = tops[0]['qrr']
    qrr_50: int = get_next_by_val(twlst, "qrr", 50)
    print("    qrr of 50 occurs at record %d of %d" % (qrr_50, totlen))
    print("    filtered: top qrr is %d, median is %d" % (top_qrr, med_qrr))
    print("      least included qrr is: %d" % (tops[quota - 1]['qrr']))

    return tops

def get_pctle_fave(twlst: list, ptile: int = 0, quota: int = 0):
    """
    create list of Tweets in top quartile for qrr count
    :param twlst: list of dict of Tweets w/ favorite counts
    :param ptile: integer from 0 to 99 indicating percentile above which to include
    :param quota: alternate to percentile is to specify quota-  x records to select
    :return: list of dict: Tweets in top quartile by popularity count
    """
    totlen: int = len(twlst)
    srtd: list = sorted(twlst, key=lambda x: x.get('fave'), reverse=True)
    if quota != 0:
        quota = int(quota)
        print("selecting top %d tweets by quota provided" % quota)
    else:
        top_pcnt: int = 100 - ptile
        quota = round(totlen * (top_pcnt / 100), ndigits=0)
        quota = int(quota)

    print("get_pctle_fave: getting top %d Tweets of %d by fave count" % (quota, totlen))

    tops: list = srtd[:quota]
    med_fave: float = tops[round(quota * 0.5)]['fave']
    top_fave: float = tops[0]['fave']
    fave_100: int = get_next_by_val(twlst, "fave", 100)
    print("    fave count of 100 is at record %d out of %d" % (fave_100, totlen))
    print("    filtered: top fave is %d, median is %d" % (top_fave, med_fave))
    print("      least fave included is: %d" % tops[quota - 1]['fave'])

    return tops

def get_neg_sentiment(twlst: list, cutoff: float = 0.2):
    """
    create list of Tweets in top quartile for negative sentiment score
    :param twlst: list of Vader scores
    :param cutoff: minimum score to include
    :return: list of str: Tweets in top quartile by sentiment
    """
    totlen: int = len(twlst)
    sent_4: int = get_next_by_val(twlst, "neg", cutoff)
    srtd: list = sorted(twlst, key=lambda x: x.get('neg'), reverse=True)
    tops: list = srtd[:sent_4]
    print("get_negative_sentiment: selecting %d Tweets out of %d" % (sent_4, totlen))

    med_sent: float = tops[round(sent_4 * 0.5)]['neg']
    top_sent: float = tops[0]['neg']
    print("    filtered: top sentiment is %1.2f, median is %1.2f" % (top_sent, med_sent))

    return tops

def create_dataframe(twlist: list, dcol: str = "sent"):
    """
    create a pandas dataframe from a list of dicts, where each dict is one tweet
    :param twlist: list of dict for tweets after pre-processing
    :param dcol: optional name of date column to use in this table
    :return: pd.DataFrame from input list
    """

    df = pd.DataFrame.from_records(twlist)
    pd.options.display.float_format = '{:.3f}'.format
    if not dcol in ["sent"]:
        if dcol in df.columns:
            print("%s being used as date column in table" % dcol)
        else:
            print("ERROR: %s not found in table" % dcol)
            return None
    else:
        if dcol in df.columns:
            print("%s being used as date column in table" % dcol)
        else:
            print("ERROR: %s not found in table" % dcol)
            return None

    dcol_num: int = df.columns.get_loc(dcol)
    if isinstance(df.iat[0, dcol_num], str):
        if len(df.iat[0, dcol_num]) == 16:
            df[dcol] = df[dcol].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M"))
            df[dcol] = df[dcol].astype('datetime64[ns]')
        elif len(df.iat[0, dcol_num]) == 10:
            df[dcol] = df[dcol].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))
            df[dcol] = df[dcol].astype('datetime64[ns]')
        else:
            print("ERROR: could not convert %s column to date" % dcol)
            return None

    df.sort_values(dcol, inplace=True, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    print("create_dataframe created table with %d rows\n" % len(df))
    # to keep from breaking some earlier code which used the 'dt' column:
    df['dt'] = df[dcol]

    return df

def clean_hash_and_mentions(hsh: dict, h_s: list):
    """
    utility to apply stop words for hashtags and user mentions,
    for the superleague project, not interested in hashtags for superleague, ESL, TESL, etc.
    :param hsh: dictionary of hashtags in Tweet dataset and number of occurrences for each
    :param h_s: list of hashtag stopwords
    :return sorted dict (by descending value) with stops removed
    """
    h2 = sorted(hsh, key=lambda x: hsh.get(x), reverse=True)
    hcln: dict = {k: hsh[k] for k in h2 if k not in h_s}
    h2l: list = list(hcln.keys())
    hod: dict = {h: hcln[h] for ct, h in enumerate(h2l) if ct < 30}

    return hod

def bucket_lst(twlst: list):
    """
    collect records into multi-hour buckets, like 6-hour windows to collect tweets for
    plotting.
    make a tuple of three fields: id of a tweet, day of tweet, hour sent, then bounce the
    tuple up against the dt_bkt 'list of lists' to fill up the buckets

    :param twlst: list of dict of tweets
    :return dt_bkt list of buckets with id's of tweets in each bin
    """
    dt_bkt: list = [
        ['2021-04-18', 0, 0, []],
        ['2021-04-18', 6, 0, []],
        ['2021-04-18', 12, 0, []],
        ['2021-04-18', 18, 0, []],
        ['2021-04-19', 0, 0, []],
        ['2021-04-19', 6, 0, []],
        ['2021-04-19', 12, 0, []],
        ['2021-04-19', 18, 0, []],
        ['2021-04-20', 0, 0, []],
        ['2021-04-20', 6, 0, []],
        ['2021-04-20', 12, 0, []],
        ['2021-04-20', 18, 0, []],
        ['2021-04-21', 0, 0, []],
        ['2021-04-21', 6, 0, []],
        ['2021-04-21', 12, 0, []],
        ['2021-04-21', 18, 0, []],
        ['2021-04-22', 0, 0, []],
        ['2021-04-22', 6, 0, []],
        ['2021-04-22', 12, 0, []],
        ['2021-04-22', 18, 0, []],
        ['2021-04-23', 0, 0, []],
        ['2021-04-23', 6, 0, []],
        ['2021-04-23', 12, 0, []],
        ['2021-04-23', 18, 0, []],
        ['2021-04-24', 12, 0, []],
        ['2021-04-24', 18, 0, []],
    ]

    def find_day(dstr: str):
        """
        inner fx to find the passed in day in list dt_bkt
        :param dstr: date string in Y-m-d format
        :return:
        """
        for x in range(len(dt_bkt)):
            if dt_bkt[x][0].startswith(dstr):
                return int(x)
        return 'crap'

    for tw in twlst:
        ttpl: tuple = (tw['date'], int(tw['sent_time'][:2]), tw['id'])
        # get the first rec for that day
        strt: int = find_day(ttpl[0])
        if (strt not in ['crap']) & (strt <= len(dt_bkt) - 2):
            # if hour stamp is less than 2nd bkt, put in 1st bkt
            if ttpl[1] < dt_bkt[strt + 1][1]:
                dt_bkt[strt][2] += 1
                dt_bkt[strt][3].append(ttpl[2])
            elif ttpl[1] < dt_bkt[strt + 2][1]:
                dt_bkt[strt + 1][2] += 1
                dt_bkt[strt + 1][3].append(ttpl[2])
            elif ttpl[1] < dt_bkt[strt + 3][1]:
                dt_bkt[strt + 2][2] += 1
                dt_bkt[strt + 2][3].append(ttpl[2])
            else:
                dt_bkt[strt + 3][2] += 1
                dt_bkt[strt + 3][3].append(ttpl[2])

    return dt_bkt

def crop_df_to_date(twdf: pd.DataFrame, strtd: str = None, endd: str = None):
    """
    pass a start date and/or end date to crop a dataframe,
    handy to use prior to plotting a slice of a dataset.
    this Fx only processes valid dates provided- as "YYYY-mm-dd HH:MM" format string
    :param twdf: pd.DataFrame
    :param strtd: str as "YYYY-mm-dd HH:MM"
    :param endd: str as "YYYY-mm-dd HH:MM"
    :return: modified pd.DataFrame
    """
    tmpdf: pd.DataFrame = twdf.copy(deep=True)
    pre_len: int = len(tmpdf)
    print("\nCROP_DF_TO_DATE: trimming dataset to start and end dates")
    print("    starting with %d tweets" % pre_len)

    if 'sent' in tmpdf.columns:
        foundcol: int = tmpdf.columns.get_loc('sent')
        if isinstance(tmpdf.iat[0, foundcol], dt.datetime):
            # can also use below, but technically not a legit Fx call
            # if is_dt64(tmpdf['sent']):
            usethis: str = 'sent'
        elif isinstance(tmpdf.iat[0, foundcol], str):
            tmpdf['sent'] = tmpdf['sent'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M"))
            tmpdf['sent'].astype('datetime64', copy=False, errors='ignore')
            usethis: str = 'sent'
        print("    using SENT column to remove dates outside of range...")

    elif 'date' in tmpdf.columns:
        foundcol: int = tmpdf.columns.get_loc('date')
        if isinstance(tmpdf.iat[0, foundcol], dt.datetime):
            usethis: str = 'date'
        else:
            if isinstance(tmpdf.iat[0, foundcol], str):
                tmpdf['date'] = tmpdf['date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M"))
                tmpdf['date'].astype('datetime64', copy=False, errors='ignore')
                usethis: str = 'date'
        print("    using DATE column to remove dates outside of range...")

    elif 'datetime' in tmpdf.columns:
        foundcol: int = tmpdf.columns.get_loc('datetime')
        if isinstance(tmpdf.iat[0, foundcol], dt.datetime):
            usethis: str = 'datetime'
        else:
            if isinstance(tmpdf.iat[0, foundcol], str):
                tmpdf['datetime'] = tmpdf['datetime'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M"))
                tmpdf['datetime'].astype('datetime64', copy=False, errors='ignore')
                usethis: str = 'datetime'
        print("    using DATETIME column to remove dates outside of range...")

    if strtd:
        try:
            if len(strtd) == 10:
                # dt_strt: dt.date = dt.date.fromisoformat(strtd)
                dt_strt: dt.datetime = dt.datetime.strptime(strtd, "%Y-%m-%d")
            elif len(strtd) == 16:
                dt_strt: dt.datetime = dt.datetime.strptime(strtd, "%Y-%m-%d %H:%M")
            else:
                print("crop_df_to_date ERROR need 10 char date or 16 char date-time")
                return None
            tmpdf = tmpdf.loc[tmpdf[usethis] >= dt_strt,]
            print("    removing rows with dates prior to %s" % strtd)
        except ValueError:
            print("crop_df_to_date ERROR: invalid start date parameter")
            return None

        tmpdf.reset_index(drop=True, inplace=True)
    if endd:
        try:
            if len(endd) == 10:
                dt_end: dt.date = dt.datetime.strptime(endd, "%Y-%m-%d")
            elif len(endd) == 16:
                dt_end: dt.datetime = dt.datetime.strptime(endd, "%Y-%m-%d %H:%M")
            else:
                print("crop_df_to_date ERROR: need 10 char date or 16 char date-time")
                return None
            tmpdf = tmpdf.loc[tmpdf[usethis] <= dt_end,]
            print("    removing rows with dates later than %s" % endd)
        except ValueError:
            print("crop_df_to_date ERROR: invalid end date parameter")

    print("        %d records remain after removing %d rows\n"
          % (len(tmpdf), len(tmpdf) - pre_len))
    tmpdf.sort_values(usethis)
    tmpdf.reset_index(drop=True, inplace=True)

    return tmpdf

def split_toplist_bytyp(df: pd.DataFrame):
    """
    returns three df's after splitting into 'type' groups:
        1. top percentile sentiment (type 'qfs', 'qs' and 'fs)
        2. both influence metrics (type 'qf')
        3. just 'q' or just 'f'
    Q = high Quote-Retweet-Reply count, F= high favorited-liked count, and
    S= high sentiment score (in absolute sense- high pos or high neg, -1.0 to +1.0)
    :param df: tweet dataframe
    :return:
    """
    s_df: pd.DataFrame = df.loc[df['type'].isin(['qfs', 'qs', 'fs']),]
    qftyp_df: pd.DataFrame = df.loc[df['type'].isin(['qf']),]
    qorf_df: pd.DataFrame = df.loc[df['type'].isin(['q', 'f']),]

    return s_df, qftyp_df, qorf_df
