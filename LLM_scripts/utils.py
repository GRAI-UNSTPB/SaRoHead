import spacy

def return_needed_ner_labels():
    ro_nlp_spacy = spacy.load("ro_core_news_lg")
    total_labels = list(ro_nlp_spacy.get_pipe("ner").labels)
    skip_labels = ["NUMERIC_VALUE","DATETIME","ORDINAL","QUANTITY"]
    added_tokens=[label for label in total_labels if label not in skip_labels]
    added_tokens=list(map(lambda tok:"["+tok+"]",added_tokens))
    return added_tokens