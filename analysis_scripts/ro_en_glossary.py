GLOSSARY = {
    "lucruri": "things",
    "zece": "ten",
    "asta": "this",
    "ei": "they",
    "am": "have",
    "n": "not",
    "au": "have",
    "-au": "have",
    "luat": "taken",
    "pus": "put",
    "dus": "carried",
    "vor": "will",
    "le": "them",
    "ca": "that",
    "cu": "with",
    "sa": "to",
    "se": "reflexive pronoun",
    "care": "which",
    "despre": "about",
    "si": "and",
    "dupa": "after",
    "lui": "his",
    "prima": "first",
    "femeie": "woman",
    "incendiu": "fire",
    "murit": "died",
    "vesti": "news",
    "foto": "photo",
    "guvernare": "governing",
    "sedinta": "session",
    "privind": "regarding",
    "guvern": "government",
    "prezidentiale": "presidential",
    "premier": "prime minister",
    "parlament": "parliament",
    "exclusiv": "exclusive",
    "spune": "says",
    "facut": "made",
    "alegerile": "elections",
    "vizita": "visit",
    "bani": "money",
    "echipa": "team",
    "finala": "final",
    "ani": "years",
    "inchisoare": "prison",
    "crima": "crime",
    "apa": "water",
    "iarna": "winter",
    "aparut": "appeared",
    "descoperit": "discovered",
    "proteste": "protests",
    "trafic": "traffic",
    "ziua": "day",
    "sarbatori": "holidays",
}

HEADLINE_TRANSLATIONS = {
    "12 lucruri despre valul recent de arestari": "12 things about the recent wave of arrests",
    "conflict [gpe] - [gpe] . [person] : au fost deja intocmite planurile necesare": "[GPE]-[GPE] conflict. [PERSON]: the necessary plans have already been drawn up",
    "lovitura pentru [person] la alegerile viitoare , desigur daca va mai ajunge la urne": "blow for [PERSON] at the upcoming elections, of course, if he still makes it to the polls",
}


def gloss(word):
    translation = GLOSSARY.get(word.lower())
    return f"{word} ({translation})" if translation else word


def gloss_pair(words):
    words = list(words)
    translations = [GLOSSARY.get(w.lower(), w) for w in words]
    head = ", ".join(words)
    if translations == [w.lower() for w in words]:
        return head
    return f"{head} ({', '.join(translations)})"


def translate_headline(words):
    return HEADLINE_TRANSLATIONS.get(" ".join(words).lower())
