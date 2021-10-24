"""
Stanza and NLTK nlp functions and classes
"""
import stanza

# import nltk.data
# nltk.data.path.append('/Users/bgh/dev/')
STANZADIR = "/Users/bgh/dev/NLP/corenlp"  # needs to be SET as OS ENV VARIABLE

def setup_stanza():
    """
    examples: for loop one sentence at a time is very slow.  best approach is
    concatenate documents together, each doc separated by blank line (\n\n)
    Wrap document in stanza.Document object
    add_property on stanza doc sets custom properties
    """
    stanza_cfg = {
        'dir': STANZADIR,
        'processors': 'tokenize,mwt,pos,sentiment',
        'lang': 'en',
        # Processor-specific keys as "{processor_name}_{argument_name}"
        'pos_model_path': './en_gsd_models/en_gsd_tagger.pt',
        'tokenize_pretokenized': True  # Use pretokenized text and disable tokenization
    }

    nlp_i = stanza.Pipeline(**stanza_cfg)

    documents = ["This is a test document.\n\n I wrote another document for fun.",
                 "and This is the 3rd document"]
    in_docs = [stanza.Document([], text=d) for d in documents]

    nlp_i.Document.add_property('char_count', default=0, getter=lambda self:
    len(self.text), setter=None)
    out_docs = nlp_i(in_docs)  # Call neural pipeline on list of documents
    print(out_docs[1])  # output is also list of stanza.Document objects

    return

