# encoding=utf-8
"""
gs_nlp_UTIL is a complete set of utlity classes and methods to
facilitate processing of Tweets for nlp analysis.  includes
getting and saving sets of tweets, text wrangling and tokenization,
and analysis like word cloud and tf/idf.
"""

import json
from os import listdir
from os.path import isfile, join
import re
from collections import OrderedDict
import datetime as dt
from nltk.tokenize import TweetTokenizer, PunktSentenceTokenizer
from tweet_data_dict import TWEETSDIR, OUTDIR, GS_HASH, GS_URL, GS_MENT, GS_CONTRACT, \
    GS_PAREN, XTRA_PUNC, JUNC_PUNC, GS_UCS2, PUNC_STR, GS_STOP, STOP_ADD
from gs_tweet_analysis import get_word_freq

def get_file_list(jsondir: str=TWEETSDIR):
    """
    get_file_list returns a list of .json files which contain batches of tweets which were
    pulled from the Twitter dev api search endpoint.  I currently use Postman as a separate
    process to create batches of tweets for analysis, as it is very adaptable!  Postman
    connects using Oauth authentication, sends http GET queries, and returns json files.
    :param jsondir: str with pathname to directory with tweet files (def: gsTweet/twitter)
    :return: list of .json files
    """
    all_files = [f for f in listdir(jsondir) if isfile(join(jsondir, f))]
    prod_list: list = []
    fil_count: int = 0
    reject_count: int = 0
    for candidate in all_files:
        if candidate.endswith(".json"):
            fil_count += 1
            prod_list.append(candidate)
        else:
            reject_count += 1
    print("\n")
    box_prn("Input File List: " + str(fil_count) + " .json files, " + str(reject_count) +
            " files rejected")
    return prod_list

def get_tweets_from_json(file_names, workdir: str =TWEETSDIR):
    """
    process_filelist iterates through an input list of json files with tweets
    :param workdir:
    :param file_names: list of filenames or string with one filename
    :return: list of tweets
    """

    if isinstance(file_names, str):
        tw_file = workdir + file_names
        box_prn("gsTweet: get_tweets_from_json retrieving from %s" %file_names)
        tweets = retrieve_json(tw_file)
    elif isinstance(file_names, list):
        tweets: list = []
        box_prn("gsTweet: get_tweets_from_json retrieving from %s" %workdir)
        cum_recs: int = 0
        for x in iter(file_names):
            tw_file = workdir + x
            wip, cc = retrieve_json(tw_file)
            # aggregate tweets from multiple files into a single list
            for y in iter(wip):
                tweets.append(y)
            cum_recs += cc
            print("%d " %cum_recs, end="\r", flush=True)
        print("\n")
    else:
        print("get_tweets_from_json had Error with file_names")
        return 13

    return tweets

def retrieve_json(tw_f: str, txtonly: bool = False):
    """
    method reads in a json file of tweets collected via twitter dev api
    returns dict with 3 keys: results, next, and requestParameters
    \u2026 - ellipsis utf-8
    """
    with open(tw_f, mode='rb') as fh:
        rawtext = json.load(fh)
        x = len(rawtext['results'])
        fh.close()
    if txtonly:
        twttxt = []
        for y in range(x):
            twttxt.append(rawtext['results'][y]['text'])
        return twttxt, x
    else:
        return rawtext['results'], x

def save_tojson(tweets: list, savefil: str):
    """
    called by save_recs to save a large batch of tweets to file.
    easy to chunk any junk and serialize to json
    :param tweets: list of dict, each dict a set of fields for one tweet
    :param savefil: str with name of file
    :return: 0 if success
    """
    fh_j = open(savefil, mode='a+', encoding='utf-8', newline='')
    json.dump(tweets, fh_j, separators=(',', ':'))
    return fh_j.close()

def save_recs(recs: list, fnames: str = "tweetarchive"):
    """
    saves a batch of tweets to a json file, so that pre-processing steps don't have to
    be redone for a corpus. also facilitates aggregating many batches of tweets
    :param recs: list of dicts or list of lists for tweet corpus
    :param fnames: str for prefix of archive filename
    :return: 0 if success
    """
    txt_stmp: str = fnames
    if len(recs) >= 100:
        size: str = str(len(recs))
        batchdate: dt.date = dt.datetime.strptime(recs[0]['date'], '%Y-%m-%d')
        batchfile = txt_stmp + "_" + size + "_" + batchdate.strftime('%Y-%m-%d') + ".json"
        tw_savef = OUTDIR + batchfile
        if isinstance(recs[0], dict):
            for tw in recs:
                tw['datetime']: str = dt.datetime.strftime(tw['datetime'], "%Y-%m-%d %H:%M")
        save_tojson(recs, tw_savef)
        print("gs_nlp_util.save_recs:  %s tweets saved to %s \n" % (size, batchfile))
        return 0
    else:
        # if we don't have at least 100 tweets to save, something's wrong
        print("\n save_recs thinks recs is too small! \n")
        return 1

def get_date(twcreated, option: str = 'S'):
    """
    returns a date object from tweet 'created_at' string passed to the function
    :param twcreated: str timestamp indicating when tweet was posted
    return: dt.datetime object for option="DT" or str if option='S'
    """
    if str(twcreated[0:4]).isnumeric():
        tmpdt = twcreated[0:4] + "-" + twcreated[5:7] + "-" + twcreated[8:10] +\
                " " + twcreated[11:16]
        twdate: dt.datetime = dt.datetime.strptime(tmpdt, '%Y-%m-%d %H:%M')
    else:
        tmpdt = twcreated[-4:] + "-" + twcreated[4:7] + "-" + twcreated[8:10] +\
                " " + twcreated[11:16]
        twdate: dt.datetime = dt.datetime.strptime(tmpdt, '%Y-%b-%d %H:%M')

    if option == "S":
        return twdate.strftime('%Y-%m-%d')
    elif option == "DT":
        return twdate

def get_fields_simple(tw_obj: list, debug: bool=False):
    """
    get_tweet_fields parses the mutli-level, dict and list structure of tweets to populate
    certain key:vals (text, date, hashes, u_mention, or qt_text for quoted tweets).
    long text (>140 char) in tweet, quote, or retweet is found in 'extended'-'full_text'
    run this after gs_nlp_util.retrieve_json to create working records for a corpus
    text ending in ellipsis (ucs \x2026 char) indicates extended full_text needed.
    returns a list of dicts, key:val structure of each shown below.

    :param tw_obj: [{},{}] i.e. a list of dicts
    :param debug: True if you want verbose status
    :return: list of dict (dict for each tweet)

    TW_DCT key-value structure for each tweet:
    'text': <string containing full text of tweet
    'uname': <twitter handle for tweet author>
    'uloc': <location of user who wrote tweet>
    'replyto': <reply to user handle, IF ANY>
    'country': <country where tweet originated>
    'qrr': <sum of quote, reply and retweet counts>
        CONDITIONAL
        'hashes': <hashtags in primary tweet, if any>
        'u_mentions':  <user mentiones in primary tweet, if any>
        'urls':  <url's in primary tweet, if any>
        'qt_text' or 'rt_text': <full text for tweets longer than 140 chars>
        'uname': <user name who retweeted or quote tweeted>
        'rt_friends': <count of friends for rt user>
        'qt_txt': <full text of quoted tweet>
        'qt_name': <handle for quoted tweet sender>
        'qt_friends': <friends count for quoted tweet>
        'qt_ext': <extended text for a quoted tweet>

         \n1⃣   \n1⃣  \n1â.£
    """
    def do_mentions(tmp_lst, rectyp: str="arch"):
        """
        inner function to parse user mentions from Tweet sections
        :param tmp_lst: fragment of tweet input to parse
        :param rectyp: "arch" if parsing full archive record, "GET" for GET_Tweet endpoint
        :return:
        """
        x = len(tmp_lst)
        if isinstance(tmp_lst, list):
            if 'mentions' in tw_dct:
                # convert field to list to append new values
                if isinstance(tw_dct['mentions'], str):
                    tw_dct['mentions'] = [tw_dct['mentions']]
            else:
                tw_dct['mentions'] = []

            for y in range(x):
                if rectyp == "arch":
                    tw_dct['mentions'].append(tmp_lst[y]['screen_name'])
                elif rectyp == "GET":
                    if debug:
                        print("user_ment: %s" %tmp_lst[y]['username'], end="")
                    tw_dct['mentions'].append(tmp_lst[y]['username'])
        return

    def do_urls(tmp_lst):
        """
        inner function to parse url info from Tweet sections
        :param tmp_lst: fragment of tweet input to parse
        :return:
        """
        x = len(tmp_lst)
        if isinstance(tmp_lst, list):
            if 'urls' in tw_dct:
                # convert field to list to append new values
                if isinstance(tw_dct['urls'], str):
                    tw_dct['urls'] = [tw_dct['urls']]
            else:
                tw_dct['urls'] = []
            for y in range(x):
                tw_dct['urls'].append(tmp_lst[y]['expanded_url'])
        return

    def do_hashes(tmp_lst):
        """
        inner function to parse hashtag info from Tweet sections
        :param tmp_lst: fragment of tweet input to parse
        :return:
        """
        x = len(tmp_lst)
        if isinstance(tmp_lst, list):
            if 'hashes' in tw_dct:
                # convert field to list to append new values
                if isinstance(tw_dct['hashes'], str):
                    tw_dct['hashes'] = [tw_dct['hashes']]
            else:
                tw_dct['hashes'] = []
            for y in range(x):
                if 'tag' in tmp_lst[y]:
                    tw_dct['hashes'].append(tmp_lst[y]['tag'])
                else:
                    tw_dct['hashes'].append(tmp_lst[y]['text'])
        return

    tw_list: list = []
    tcount: int = 0
    for twmp in tw_obj:
        # for id, use either field id or id_str
        tw_dct: dict = {
            'text': twmp['text'],
            'date': get_date(twmp['created_at']),
            'sent_time': twmp['created_at'][11:16],
            'datetime': get_date(twmp['created_at'], option="DT")
        }
        if "user" in twmp:
            # record is from full archive endpoint
            tw_dct['id'] = twmp['id_str']
            tw_dct['name'] = twmp['user']['screen_name']
            if 'id_str' in twmp['user']:
                tw_dct['user_id'] = twmp['user']['id_str']
        else:
            # record is from GET tweets endpoint, it provides 'id', but as a string
            tw_dct['id'] = twmp['id']
            tw_dct['user_id'] = twmp['author_id']
            tw_dct['conv'] = twmp['conversation_id']

        if "retweet_count" in twmp:
            tw_dct['qrr'] = twmp['quote_count'] + twmp['reply_count'] + \
                            twmp['retweet_count']
            tw_dct['fave'] = twmp['favorite_count']
        elif "retweet_count" in twmp['public_metrics']:
            tw_dct['qrr'] = twmp['public_metrics']['quote_count'] + \
                            twmp['public_metrics']['reply_count'] + \
                            twmp['public_metrics']['retweet_count']
            tw_dct['fave'] = twmp['public_metrics']['like_count']
        if 'entities' in twmp:
            if 'user_mentions' in twmp['entities']:
                tmp_lst = twmp['entities']['user_mentions']
                do_mentions(tmp_lst, "arch")
            elif 'mentions' in twmp['entities']:
                tmp_lst = twmp['entities']['mentions']
                do_mentions(tmp_lst, "GET")
            if 'hashtags' in twmp['entities']:
                tmp_lst = twmp['entities']['hashtags']
                do_hashes(tmp_lst)
            if 'urls' in twmp['entities']:
                tmp_lst: list = twmp['entities']['urls']
                do_urls(tmp_lst)

        if 'retweeted_status' in twmp:
            tw_dct['rt_text'] = twmp['retweeted_status']['text']
            tw_dct['rt_id'] = twmp['retweeted_status']['id_str']
            tw_dct['rt_qrr'] = twmp['retweeted_status']['quote_count'] + \
                               twmp['retweeted_status']['reply_count'] + \
                               twmp['retweeted_status']['retweet_count']
            tw_dct['rt_fave'] = twmp['retweeted_status']['favorite_count']
            if 'entities' in twmp['retweeted_status']:
                if 'user_mentions' in twmp['retweeted_status']['entities']:
                    tmp_lst: list = twmp['retweeted_status']['entities']['user_mentions']
                    do_mentions(tmp_lst, "arch")
                if 'urls' in twmp['retweeted_status']['entities']:
                    tmp_lst: list = twmp['retweeted_status']['entities']['urls']
                    do_urls(tmp_lst)
                if 'hashtags' in twmp['retweeted_status']['entities']:
                    tmp_lst: list = twmp['retweeted_status']['entities']['hashtags']
                    do_hashes(tmp_lst)
            if 'extended_tweet' in twmp['retweeted_status']:
                tw_dct['rt_text'] = twmp['retweeted_status']['extended_tweet']['full_text']
                tw_dct['text'] = tw_dct['rt_text']
                if 'entities' in twmp['retweeted_status']['extended_tweet']:
                    if 'user_mentions' in twmp['retweeted_status']['extended_tweet']['entities']:
                        tmp_lst: list = twmp['retweeted_status']['extended_tweet']['entities']['user_mentions']
                        do_mentions(tmp_lst, "arch")
                    if 'urls' in twmp['retweeted_status']['extended_tweet']['entities']:
                        tmp_lst: list = twmp['retweeted_status']['extended_tweet']['entities']['urls']
                        do_urls(tmp_lst)
                    if 'hashtags' in twmp['retweeted_status']['extended_tweet']['entities']:
                        tmp_lst: list = twmp['retweeted_status']['extended_tweet']['entities']['hashtags']
                        do_hashes(tmp_lst)
        if 'quoted_status' in twmp:
            tw_dct['qt_text'] = twmp['quoted_status']['text']
            tw_dct['qt_id'] = twmp['quoted_status']['id_str']
            tw_dct['qt_qrr'] = twmp['quoted_status']['quote_count'] +\
                               twmp['quoted_status']['reply_count'] +\
                               twmp['quoted_status']['retweet_count']
            tw_dct['rt_fave'] = twmp['quoted_status']['favorite_count']
            if 'entities' in twmp['quoted_status']:
                if 'hashtags' in twmp['quoted_status']['entities']:
                    tmp_lst: list = twmp['quoted_status']['entities']['hashtags']
                    do_hashes(tmp_lst)
                if 'urls' in twmp['quoted_status']['entities']:
                    tmp_lst: list = twmp['quoted_status']['entities']['urls']
                    do_urls(tmp_lst)
                if 'user_mentions' in twmp['quoted_status']['entities']:
                    tmp_lst = twmp['quoted_status']['entities']['user_mentions']
                    do_mentions(tmp_lst, "arch")
            if 'extended_tweet' in twmp['quoted_status']:
                tw_dct['qt_text'] = twmp['quoted_status']['extended_tweet']['full_text']
                tw_dct['text'] = tw_dct['qt_text']
                if 'entities' in twmp['quoted_status']['extended_tweet']:
                    if 'hashtags' in twmp['quoted_status']['extended_tweet']['entities']:
                        tmp_lst: list = twmp['quoted_status']['extended_tweet']['entities']['hashtags']
                        do_hashes(tmp_lst)
                    if 'urls' in twmp['quoted_status']['extended_tweet']['entities']:
                        tmp_lst: list = twmp['quoted_status']['extended_tweet']['entities']['urls']
                        do_urls(tmp_lst)
                    if 'user_mentions' in twmp['quoted_status']['entities']:
                        tmp_lst = twmp['quoted_status']['extended_tweet']['entities']['user_mentions']
                        do_mentions(tmp_lst, "arch")
        if 'extended_tweet' in twmp:
            tw_dct['text'] = twmp['extended_tweet']['full_text']
            if 'entities' in twmp['extended_tweet']:
                if 'hashtags' in twmp['extended_tweet']['entities']:
                    tmp_lst = twmp['extended_tweet']['entities']['hashtags']
                    do_hashes(tmp_lst)
                if 'urls' in twmp['extended_tweet']['entities']:
                    tmp_lst: list = twmp['extended_tweet']['entities']['urls']
                    do_urls(tmp_lst)
                if 'user_mentions' in twmp['extended_tweet']['entities']:
                    tmp_lst: list = twmp['extended_tweet']['entities']['user_mentions']
                    do_mentions(tmp_lst, "arch")

        tw_list.append(tw_dct)
        tcount += 1

    print("%5d  Tweets parsed" %tcount, end="\n", flush=True)
    print("\n")
    return tw_list

def get_batch_from_file(batchfil: str):
    """
    reads a batch of tweets that were saved to file with save_tojson
    :param batchfil: str with name of file to read
    :return: list of dict
    """
    f_h = open(batchfil, mode='r', encoding='utf-8')
    rawtext = json.load(f_h)
    f_h.close()
    return rawtext

def scrub_text(tweetxt: str):
    """
    scrub_text can perform numerous removal or modification tasks specific to tweet nlp,
    there is tweet-specific content handled here which can be optionally commented out
    if resulting corpus loses too much detail for tfidf, sentiment, or other tasks.

    :param tweetxt: str from text field of tweet
    :return: list of words OR str of words if rtn_list= False
    """
    if isinstance(tweetxt, str):
        # remove newlines in tweets, they cause a mess with nlp!
        tweetxt: str = tweetxt.replace("\n", " ")
        splitstr = tweetxt.split()
        cleanstr: str = ""
        for w in splitstr:
            if str(w).isalpha():
                if not str(w).isupper():
                    w = str(w).lower()
            cleanstr = cleanstr + " " + w
        # remove any urls, remove @ symbol from username
        tweetxt = re.sub(GS_URL, "", cleanstr)
        tweetxt = re.sub(GS_MENT, "\g<1>", tweetxt)
        tweetxt = re.sub(GS_HASH, "\g<1>", tweetxt)
        # remove periods, no need for sentence demarcation in tweets!
        tweetxt = re.sub("\.", " ", tweetxt)
        # expand contractions using dict of common contractions
        for k, v in GS_CONTRACT.items():
            tweetxt = re.sub(k, v, tweetxt)
        # often ucs-2 chars appear in english tweets, can simply convert some
        for k, v in GS_UCS2.items():
            tweetxt = re.sub(k, v, tweetxt)
        # parsing tweet punc is tricky- can show sentiment or be part of smiley
        tweetxt = re.sub(JUNC_PUNC, "", tweetxt)
        # remove spurious symbols
        for p in PUNC_STR:
            tweetxt = tweetxt.strip(p)
        # remove any leading or trailing whitespace:
        tweetxt = tweetxt.strip()
        tweetxt = re.sub(XTRA_PUNC, " \g<1>", tweetxt)

        # encode-decode for ucs-2 chars: "utf-8" escapes (\uxxxx), "ascii" removes
        # binstr = cleanstr.encode("ascii", "ignore")
        # cleanstr = binstr.decode()
        # next 3 lines can remove words less than a given length, but removes emojis :-(
        # wrd_toks: list = cleanstr.split(" ")
        # wrd_toks = [x for x in wrd_toks if len(x) >= 3]
        # cleanstr: str = ' '.join([str(x) for x in wrd_toks])
        return tweetxt

def clean_text(tw_batch: list):
    """
    just a stub to iterate list of tweets and call do_tweet_scrub or other workers that
    do the wrangling work.
    :param tw_batch: list containing batch of tweets
    :return: dict with primary text now containing full text of tweet
    """
    fixed: list = []
    for atweet in tw_batch:
        if isinstance(atweet, dict):
            # EXPECTED course - if sending twrecs to this function!
            if atweet.get('text'):
                atweet['text'] = scrub_text(atweet['text'])
                fixed.append(atweet)
            elif atweet.get('full_text'):
                # different field name if groupby_date Tweets are passed
                atweet['full_text'] = scrub_text(atweet['full_text'])
                fixed.append(atweet)
        elif isinstance(atweet, list):
            # each atweet is a list of sentence strings for tweet
            if isinstance(atweet[0], str):
                fixed.append([scrub_text(seg) for seg in atweet])
            else:
                # may be word tokenized already, just toss it back
                fixed.append(atweet)
        elif isinstance(atweet, str):
            # if sentences have been flattened, each tweet just a string
            atweet = scrub_text(atweet)
            fixed.append(atweet)
    return fixed

def remove_replace_text(twlst: list, stop1: list=GS_STOP, stop2: list=None, debug: bool=False):
    """
    final pre-processing b/f word tokenize, input can be lists of list, str or dict
    fixes text in parens, converts to lower if word not all caps, and removes STOPs.

    :param twlst: list of tweets
    :param stop1: default is stop list from tweet_data_dict
    :param stop2: list a secondary list of stop words if needed
    :param debug: bool, some verbose printing to assist debugging
    :return: list of scrubbed tweets
    """
    def do_paras(twstr: str) -> str:
        """
        inner function to remove parentheses and move paren text to end of sentence
        :param twstr: str text of a single tweet
        :return: str modifies input to fx
        """
        parafound = re_paren.search(twstr)
        if parafound:
            paratext = parafound.group(1)
            twstr: str = re_paren.sub("", twstr).strip()
            twstr = twstr + " " + paratext[1:-1]
        return twstr

    def do_lcase_stops(twstr: str) -> str:
        """
        'inner' Fx: unless tweet words is allcaps, lowercase it and remove stoplist words
        :param twstr: str with text of one tweet
        :return: text of Tweet w  ith case corrected and stopwords removed
        """
        tw_wrds = twstr.split()
        tmplst: list = []
        for w in tw_wrds:
            # retain ALL CAPS words, otherwise lower-case them
            if not str(w).isupper():
                w = str(w).lower()
            if w in stop1:
                continue
            if stop2 and w in stop2:
                continue
            if debug:
                if not w.isalnum():
                    print("%s   not alpha-numeric" %w)
            tmplst.append(w)
        twstr = " ".join([x for x in tmplst])
        return twstr

    tw_clean: list = []
    re_paren = re.compile(GS_PAREN)
    for atweet in twlst:
        if isinstance(atweet, str):
            tw_text: str = do_paras(atweet)
            atweet = do_lcase_stops(tw_text)
        elif isinstance(atweet, dict):
            if atweet.get('text'):
                tw_text: str = do_paras(atweet['text'])
                atweet['text'] = do_lcase_stops(tw_text)
        elif isinstance(atweet, list):
            tw_text: str = "".join([str(x) for x in atweet])
            atweet = do_paras(tw_text)
            atweet = do_lcase_stops(atweet)
        else:
            print("problem with input passed to remove_replace_text function")
        tw_clean.append(atweet)

    return tw_clean

def flatten_twdict(twlist: list):
    """
    create flat list of tweet text from list of dict or list of list
    :param twlist: list of dict
    :return:
    """
    templst: list = []
    for twthis in twlist:
        if isinstance(twthis, dict):
            templst.append(twthis['text'])
        elif isinstance(twthis, list):
            templst.append(" ".join([str(x) for x in twthis]))
        else:
            templst.append(twthis)

    return templst

def do_sent_tok(tw_list: list):
    """
    do_sent_tok uses nltk tokenizer, which currently defaults to 'Punkt'
    I also move text within parentheses to end of tweet, its own little segment
    a Tweet is 1 'thought', like a sentence, why split it? sent tokenize not needed!
    """
    sents: list = []
    p_st = PunktSentenceTokenizer()
    for tw_inst in tw_list:
        if isinstance(tw_inst, dict):
            tw_text = tw_inst['text']
            # sentence tokenize the tweet and add to list
            sents.append(p_st.tokenize(tw_text))
    return sents

def do_wrd_tok(tweet_lst: list):
    """
    do_wrd_tok tokenizes words from a tokenized sentence list, or a string
    if tweettokenizer is used, the following constants are available: strip_handles: bool,
    reduce_len: bool, preserve_case: bool,
    :param tweet_lst: list of list of strings to word tokenize
    :return: list of list of word tokens
    """
    w_tok= TweetTokenizer(strip_handles=True, reduce_len=False, preserve_case=True)
    word_total: int = 0
    tw_total: int = 0
    wrd_tkn_lsts: list = []
    for this_tw in tweet_lst:
        if isinstance(this_tw, list):
            wrd_tkn_lsts.append(w_tok.tokenize(" ".join([str(x) for x in this_tw])))
        elif isinstance(this_tw, str):
            wrd_tkn_lsts.append(w_tok.tokenize(this_tw))
        elif isinstance(this_tw, dict):
            wrd_tkn_lsts.append(w_tok.tokenize(this_tw['text']))
    for tw in wrd_tkn_lsts:
        word_total += len(tw)
        tw_total += 1

    w_frequency: dict = get_word_freq(wrd_tkn_lsts)
    uniq_wrds: int = len(w_frequency)
    box_prn("word_tokenize: %d total words from %d Tweets, vocabulary of %d distinct words"
            %(word_total, tw_total, uniq_wrds))
    return wrd_tkn_lsts

def do_stops(twlst: list, stop1: list = GS_STOP, stop2: list = STOP_ADD):
    """
    do_stops is preprocessing function to remove word tokens based on a stop list
    :param twlst: list of list, list of dict, or list of str for Tweets
    :param stop1: list of stop words, defaults to GS_STOP
    :param stop2: list of stop words, defaults to STOP_ADD
    :return: list of tweets with word tokens and stop words removed
    """
    clean_list: list = []
    for twis in twlst:
        if isinstance(twis, list):                  # word-tokenized
            tmp_wrds: list = [cw for cw in twis if cw not in stop1]
            if stop2 is not None:
                clean_list.append([cw for cw in tmp_wrds if cw not in stop2])
            else:
                clean_list.append(tmp_wrds)
        else:
            if isinstance(twis, dict):
                twemp: list = str(twis['text']).split()
            else:                                # assume isinstance(twis, str)
                twemp: list = twis.split()

            tmp_wrds: list = [cw for cw in twemp if cw not in stop1]
            if stop2 is not None:
                clean_list.append(' '.join([str(cw) for cw in tmp_wrds if cw not in stop2]))
            else:
                clean_list.append(' '.join([str(cw) for cw in tmp_wrds]))

    return clean_list

def do_start_stops(twent):
    """
    removes or separates token strings which start with punctuation, emoji, etc.
    uses PUNC_STR defined in data dictionary

    :param twent: list of tweets, each tokenized by word
    :return: list of tweets with word tokens and stop words removed
    """
    clean_list: list = []
    if isinstance(twent, list):
        for twine in twent:
            if isinstance(twine, list):
                # if word-tokenized
                clean_list.append([cw for cw in twine if str(cw)[:1] not in PUNC_STR])
            elif isinstance(twine, str):
                # if sentence-tokenized
                twemp: list = twine.split(" ")
                twemp = [cw for cw in twemp if str(cw)[:1] not in PUNC_STR]
                clean_list.append(' '.join([str(x) for x in twemp]))

    return clean_list

def get_dates_in_batch(tws):
    """
    for a list of tweets, capture dates included and count for each day
    :param tws: list of tweets, assume the list created by sentence tokenization
    :return:
    """
    tw_days: dict = {}
    for a_tw in tws:
        if isinstance(a_tw, dict):
            if a_tw.get('date'):
                tw_d = dt.datetime.strptime(a_tw['date'], '%Y-%m-%d')
                if tw_d in tw_days:
                    if a_tw.get('sent_time'):
                        tmp_hr = str(a_tw['sent_time'][0:2])
                        if tmp_hr in tw_days[tw_d]:
                            tw_days[tw_d][tmp_hr] += 1
                        else:
                            tw_days[tw_d][tmp_hr] = 1
                    else:
                        if 'unk' in tw_days[tw_d]:
                            tw_days[tw_d]['unk'] += 1
                        else:
                            tw_days[tw_d]['unk'] = 1
                else:
                    tw_days[tw_d] = {}
                    if a_tw.get('sent_time'):
                        tw_days[tw_d][str(a_tw['sent_time'][0:2])] = 1

    # sorting prior to printing
    tw_days: OrderedDict = OrderedDict([(k, tw_days[k]) for k in sorted(tw_days.keys())])
    seq_d: int = 0
    print("Tweets per 1-hr block for each day (24hr clock)")
    for d, dh in tw_days.items():
        seq_d += 1
        print("Day %2d, Date: %10s" % (seq_d, dt.datetime.strftime(d, '%Y-%m-%d')))
        dh = {k: dh[k] for k in sorted(dh.keys())}
        tw_days[d] = dh
        for hh, cnt in dh.items():
            print("  " + str(hh) + "h: " + str(cnt) + " | ", sep="", end="")
        print("----")

    return tw_days

def box_prn(message):
    """
    prints any info passed within a box outline
    :param message: can be str or list of str
    :return: n/a
    """
    print("*" * 66)
    print("*" + " " * 64 + "*")
    print("*" + " " * 64 + "*")
    if isinstance(message, str):
        bxpr_l(message)
    elif isinstance(message, list):
        for msgseg in message:
            bxpr_l(msgseg)
    print("*" + " " * 64 + "*")
    print("*" + " " * 64 + "*")
    print("*" * 66)
    return

def bxpr_l(msg: str):
    """
    bxpr_1 is a small utility method to print message lines(s) for method box_prn
    :param msg:
    :return:
    """
    y = len(msg)
    while y > 60:
        outseg: str = msg[:62]
        print("* " + outseg + " *")
        msg = msg[62:]
        y = len(msg)
    else:
        y = 62 - len(msg)
        print("* " + msg + " " * y + " *")

    return

def print_missing_for_postman(msng: list):
    """
    get the missing tweet ids for convenience to send to postman
    """
    newtmp: list = []
    for miss in msng:
        if miss.get('rtid'):
            newtmp.append(miss.get('rtid'))
    return newtmp