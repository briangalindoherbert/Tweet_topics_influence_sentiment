# coding=utf-8
"""
gs_tweet_analysis is part II of utils for tweets, the first being gs_nlp_util.
gs_tweet_analysis builds word counts, grammar analysis, and other higher level functions
"""

import io
from math import fabs
import datetime as dt
from collections import OrderedDict
import plotly.graph_objects as go
from wordcloud import WordCloud
import pandas as pd
from pandas.api.extensions import register_extension_dtype
from pandas.api.types import is_datetime64_any_dtype as is_dt64
import nltk.data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tweet_data_dict import IDIOM_MODS, VADER_MODS

nltk.data.path.append('/Users/bgh/dev/NLP/nltk_data')
nltk.download('vader_lexicon', download_dir="/Users/bgh/dev/NLP/nltk_data")
Vsi = SentimentIntensityAnalyzer()

# updates to Vader constants, I could roll these into a function
# the following are Vader Constants derived by empirical measurement, can be changed if desired:
# B_INCR = 0.293
# B_DECR = -0.293
# C_INCR = 0.733
# N_SCALAR = -0.74

for k, v in VADER_MODS.items():
    Vsi.lexicon[k] = v
for k, v in IDIOM_MODS.items():
    Vsi.constants.SPECIAL_CASE_IDIOMS[k] = v

def limit_retweets(twbatch: list):
    """
    filters a list of tweets so that only the first retweet (by RT ID) is kept.  also builds
    a dict of retweet ID's with a count of the number of copies encountered.

    :param twbatch: list of dict of tweets
    :return: rtlist: filtered list of tweets,
            rt_cnt: dict of retweet duplicates by ID, value is number of occurrences
            miss_id: list of retweet ID's for which we are missing the Original Tweet
    """
    rtlst: list = []
    rt_count: dict = {}
    ds_len: int = len(twbatch)
    ord_id: int = 0

    for twd in twbatch:
        tmpdct: dict = {'text': twd['text'], 'id': twd['id'], 'ord_id': ord_id,
                        'date': twd['date'], 'sent_time': twd['sent_time'],
                        'datetime': twd['datetime'],
                        'qrr': twd['qrr'], 'fave': twd['fave']}

        if twd.get('hashes'):
            tmpdct['hashes'] = twd['hashes']
        if twd.get('mentions'):
            tmpdct['mentions'] = twd['mentions']

        if twd.get('rt_id'):
            tmpdct['rt_id'] = twd.get('rt_id')
            if tmpdct['rt_id'] in rt_count:
                rt_count[tmpdct['rt_id']] += 1
                continue
            else:
                # first time we've seen this retweet ID, add it and add counts
                rt_count[tmpdct['rt_id']] = 1
                tmpdct['rt_qrr'] = twd.get('rt_qrr')
                tmpdct['rt_fave'] = twd.get('rt_fave')
        else:
            tmpdct['rt_id'] = ""
            tmpdct['rt_qrr'] = 0
            tmpdct['rt_fave'] = 0

        rtlst.append(tmpdct)
        ord_id += 1

    if len(rt_count) > 0:
        # sort the list of retweet ID's by most frequent to least
        srt: list = sorted(rt_count, key=lambda x: rt_count[x], reverse=True)
        rt_count: OrderedDict = OrderedDict([(k, rt_count[k]) for k in srt])
        print("check_duplicates started with %d tweets" % ds_len)
        ds_len = len(rtlst)
        print("    and ended with %d" % ds_len)
        return rtlst, rt_count
    else:
        print("check_duplicates dataset length: %d" % ds_len)
        return rtlst

def id_originating_tweets(twlst, rtcnt, tw_lim):
    """
    this Fx goes through a list of retweet ID's and checks them against a dataset of Tweets to
    see if we have the original tweet.  this Fx returns a list of those we are missing.
    And addition will only search original tweets with highest number of downstream activity

    TODO: set 'retweet threshold'- search only if original generated X or more RTs/QTs.
    :param twlst: our list of dict of tweets PRIOR to filtering retweets
    :param rtcnt: dict created by limit_retweets, key=rt_id, val=count
    :param tw_lim: list of dict of tweets after RT filtering by limit_retweets
    :return:
    """
    print("\nid_originating_tweets verifies counts and identifies important missing Tweets")
    missing: list = []
    not_missing: list = []
    id_set: set = {x['id'] for x in twlst}
    rt_len = len(rtcnt)

    srt: list = sorted(rtcnt, key=lambda x: rtcnt[x], reverse=True)
    rtcnt: list = [{'rtid': k, 'count': rtcnt[k]} for k in srt]

    def get_highest_count(rtrec):
        """
        within retweeted_status, Twitter provides the retweet, quote, and reply counts
        for a particular retweet, this Fx finds the most current, highest count for a
        single retweet ID.
        :return: integer counts for rt_qrr and rt_fave
        """
        rtrec['rtqrr'] = 0
        rtrec['rtfave'] = 0
        for twx in twlst:
            if twx['id'] == rtrec['rtid']:
                # if ID matches, this is the original Tweet for the retweet
                if twx.get('qrr') > rtrec['rtqrr']:
                    rtrec['rtqrr'] = twx.get('qrr')
                if twx.get('fave') > rtrec['rtfave']:
                    rtrec['rtfave'] = twx.get('fave')
            elif twx.get('rt_id') == rtrec['rtid']:
                if twx.get('rt_qrr') > rtrec['rtqrr']:
                    rtrec['rtqrr'] = twx.get('rt_qrr')
                if twx.get('rt_fave') > rtrec['rtfave']:
                    rtrec['rtfave'] = twx.get('rt_fave')
            elif twx.get('qt_id') == rtrec['rtid']:
                if twx.get('qt_qrr'):
                    if twx.get('qt_qrr') > rtrec['rtqrr']:
                        rtrec['rtqrr'] = twx.get('qt_qrr')
                if twx.get('qt_fave'):
                    if twx.get('qt_fave') > rtrec['rtfave']:
                        rtrec['rtfave'] = twx.get('qt_fave')
            else:
                rtrec['noupd'] = True

        return rtrec

    def find_by_id(twid, typ):
        """
        inner Fx which returns boolean indicating whether Tweet was found based on search
        on Tweet ID or Retweet ID (which is indicated in typ parameter)
        :param twid: ID string for a Tweet
        :param typ: typ string, either "id" or "rtid"
        :return: True if ID was found, False if not found
        """
        updates: bool = False
        for a_tw in tw_lim:
            if twid['rtid'] == a_tw[typ]:
                if twid['rtqrr'] > a_tw['qrr']:
                    a_tw['qrr'] = twid['rtqrr']
                    updates = True
                if twid['rtfave'] > a_tw['fave']:
                    a_tw['fave'] = twid['rtfave']
                    updates = True
        return updates

    print("    searching for %d originating tweets:" %rt_len)
    updates_made: int = 0
    for tw in rtcnt:
        tw = get_highest_count(tw)
        if tw['rtid'] not in id_set:
            # it ID is in idset, we should find qrr and fave values for the tweet
            missing.append(tw)
            if find_by_id(tw, "rt_id"):
                updates_made += 1
        else:
            not_missing.append(tw)
            if not tw['noupd']:
                if find_by_id(tw, "id"):
                    updates_made += 1

    print("    found original for %d retweets" %len(not_missing))
    print("    updated counts for %d retweets" %updates_made)

    if len(missing) > 0:
        missing: list = sorted(missing, key=lambda x: x.get('rtqrr')
                                                      + x.get('rtfave'), reverse=True)
        mis_len: int = len(missing)
        # first missing record now has largest total count:
        buzz_max: int = int(missing[0].get('rtqrr')) + int(missing[0].get('rtfave'))
        print("    Missing %d Tweets with social buzz..." %len(missing))
        print("        Use GET Tweets by ID for Missing Tweets with High Counts!")
        print("        %d missing tweets- Go GET em!\n" %mis_len)
        print("        max qrr/fave total for 1st missing Tweet is %d" % buzz_max)
        mid_miss: int = int(round(mis_len/2, ndigits=0))
        mid_count: int = int(missing[mid_miss].get("rtqrr")) + int(missing[mid_miss].get("rtfave"))
        print("        midpoint %d has a qrr/fave count total of %d \n" %(mid_miss, mid_count))

    return missing

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
    print("\nsummarize_vader: Top Sentiment for %d total Tweets:" % rec_count)
    print("    showing top %d compound, neutral, negative and positive sentiment" % top_lim)

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
        print("    %s " % typestr, end="")
        print(" Average= %1.3f, Minimum= %1.3f, Maximum= %1.3f" % (typ_avg, typ_min, typ_max))

        return

    def show_with_text(typ, tops: list):
        """
        prints applicable sentiment score along with text of Tweet
        :param typ: string with formal Vader sentiment type (neu, pos, neg, compound)
        :param tops: list of top tweets by sentiment type, number of tweets= top_lim
        :return: None
        """
        print("Printing top %d tweets by %s sentiment:" % (top_lim, typ))
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
    print("get_negative_sentiment: selecting %d Tweets out of %d" % (sent_4,totlen))

    med_sent: float = tops[round(sent_4 * 0.5)]['neg']
    top_sent: float = tops[0]['neg']
    print("    filtered: top sentiment is %1.2f, median is %1.2f" % (top_sent, med_sent))

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
        quota = int(quota)
        print("selecting top %d tweets by quota provided" % quota)
    else:
        top_pcnt: int = 100 - ptile
        quota = round(totlen * (top_pcnt / 100), ndigits=0)
        quota = int(quota)

    print("get_pctle_qrr: getting top %d Tweets out of %d by qrr count" % (quota, totlen))

    tops: list = srtd[:quota]
    med_qrr: float = tops[round(quota * 0.5)]['qrr']
    top_qrr: float = tops[0]['qrr']
    qrr_80: int = get_next_by_val(twlst, "qrr", 80)
    print("    qrr of 100 occurs at record %d of %d" % (qrr_80, totlen))
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
    fave_10: int = get_next_by_val(twlst, "fave", 10)
    print("    fave count of 10 is at record %d out of %d" % (fave_10, totlen))
    print("    filtered: top fave is %d, median is %d" % (top_fave, med_fave))
    print("      least fave included is: %d" % tops[quota - 1]['fave'])

    return tops

def create_dataframe(twlist: list):
    """
    create a pandas dataframe from a list of dicts, where each dict is one tweet
    :param twlist: list of dict for tweets after pre-processing
    :return: pd.DataFrame from input list
    """

    df = pd.DataFrame.from_records(twlist)
    pd.options.display.float_format = '{:.3f}'.format
    if "date" in df.columns:
        df['dt'] = df['date'].astype('datetime64[ns]')

    return df

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
    usethis: str = ""
    print("\nCROP_DF_TO_DATE is trimming Tweets to date range")
    print("    starting with %d tweets" % pre_len)
    if 'sent' in tmpdf.columns:
        foundcol: int = tmpdf.columns.get_loc('sent')
        if isinstance(tmpdf.iat[0, foundcol], str):
            tmpdf['sent'] = tmpdf['sent'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M"))
            usethis = 'sent'
        print(" using SENT column to filter dates")
    elif 'datetime' in tmpdf.columns:
        foundcol: int = tmpdf.columns.get_loc('datetime')
        if is_dt64(tmpdf['datetime']):
            usethis = 'datetime'
        else:
            if isinstance(tmpdf.iat[0, foundcol], str):
                tmpdf['datetime'] = tmpdf['datetime'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M"))
                usethis = 'datetime'
        print(" using DATETIME column to filter dates")
    else:
        print("crop_df_to_date could not find column to use for date selection")
        return None

    if strtd:
        try:
            dt_strt: dt.datetime = dt.datetime.strptime(strtd, "%Y-%m-%d %H:%M")
            tmpdf = tmpdf.loc[tmpdf[usethis] > dt_strt,]
            print("    removing rows with dates prior to %s"
                  % dt.datetime.strftime(dt_strt, "%Y-%m-%d %H:%M"))
        except ValueError:
            print("invalid start date received in crop_df_to_date")

        tmpdf.reset_index(drop=True, inplace=True)
    if endd:
        try:
            dt_end: dt.datetime = dt.datetime.strptime(endd, "%Y-%m-%d %H:%M")
            tmpdf = tmpdf.loc[tmpdf[usethis] < dt_end,]
            print("    removing rows with dates later than %s"
                  % dt.datetime.strftime(dt_end, "%Y-%m-%d %H:%M"))
        except ValueError:
            print("invalid end date received in crop_df_to_date")

    print("    %d records remain after removing %d rows\n"
          % (len(tmpdf), len(tmpdf) - pre_len))
    tmpdf.sort_values(usethis)
    tmpdf.reset_index(drop=True, inplace=True)

    return tmpdf
