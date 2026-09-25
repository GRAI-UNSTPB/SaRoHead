UNMASKED_ENTITIES = {
    "ponta": "PERSON",
    "ciolacu": "PERSON",
    "citu": "PERSON",
    "florin": "PERSON",
    "gabi": "PERSON",
    "crin": "PERSON",
    "geoana": "PERSON",
    "basescu": "PERSON",
    "stoica": "PERSON",
    "fcsb": "ORGANIZATION",
    "dinamo": "ORGANIZATION",
    "steaua": "ORGANIZATION",
    "manchester": "GPE",
    "facebook": "ORGANIZATION",
}


def is_unmasked_entity(word):
    return str(word).lower() in UNMASKED_ENTITIES


def contains_unmasked_entity(words):
    return any(is_unmasked_entity(w) for w in words)
