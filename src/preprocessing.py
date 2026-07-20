import re
import emoji

CONTRACTIONS = {
    "can't": "can not",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'m": " am",
    "'d": " would",
    "'s": " is",
}

URL_PATTERN = re.compile(r"http\S+|www\S+")
MENTION_PATTERN = re.compile(r"@\w+")
PUNCT_PATTERN = re.compile(r"[^\w\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")

def expand_contractions(text: str) -> str:
    for old, new in CONTRACTIONS.items():
        text = text.replace(old, new)

    return text

def remove_urls(text: str) -> str:
    return URL_PATTERN.sub("", text)

def remove_mentions(text: str) -> str:
    return MENTION_PATTERN.sub("", text)

def remove_hashtag_symbol(text: str) -> str:
    return text.replace("#", "")

def remove_emojis(text: str) -> str:
    return emoji.replace_emoji(text, replace="")

def remove_punctuation(text: str) -> str:
    return PUNCT_PATTERN.sub(" ", text)

def normalize_whitespace(text: str) -> str:
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()

def clean_text_tfidf(text: str) -> str:
    text = text.lower()

    text = expand_contractions(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtag_symbol(text)
    text = remove_emojis(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)

    return text

def clean_text_lstm(text: str) -> str:
    text = text.lower()

    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtag_symbol(text)
    text = normalize_whitespace(text)

    return text

def clean_text_transformer(text: str) -> str:
    text = remove_urls(text)
    text = normalize_whitespace(text)

    return text

__all__ = [
    "clean_text_tfidf",
    "clean_text_lstm",
    "clean_text_transformer",
]
