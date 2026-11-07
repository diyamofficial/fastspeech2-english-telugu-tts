"""
English text frontend for the Svarah (Indian-accented English) FastSpeech 2 model.

Pipeline:   raw text
            -> normalize(): NFKC, times, dates, years, ordinals, decimals, currency,
                            percentages, cardinals, acronyms -> words (no external deps)
            -> g2p_en (CMUdict + neural G2P, POS-based homograph disambiguation)
            -> ARPAbet phones with stress (69 symbols) + "sp" pause at punctuation

The same g2p_en is used to build the MFA lexicon, so training alignments and
inference phones come from one phone set. For homographs (e.g. "conduct",
noun K AA1 N D AH0 K T vs. verb K AH0 N D AH1 K T, test sentence E2), the
lexicon lists both pronunciations and MFA picks the one that was actually spoken;
at inference g2p_en chooses by part of speech in the full sentence.
"""
import re
import sys
import os
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fs2.frontend_base import PAUSE, FrontendBase  # noqa: E402

_VOWELS = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
_CONSONANTS = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH",
               "T", "TH", "V", "W", "Y", "Z", "ZH"]
ARPABET = [v + s for v in _VOWELS for s in "012"] + _CONSONANTS

PUNCT_PAUSE = set(",.;:!?")

# ----------------------------------------------------------------------------- numbers
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")]
_ORD_IRREG = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
              "nine": "ninth", "twelve": "twelfth"}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]


def cardinal(n):
    n = int(n)
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + cardinal(rest))
    for value, name in _SCALES:
        if n >= value:
            rest = n % value
            head = cardinal(n // value) + " " + name
            if rest == 0:
                return head
            return head + (" and " if rest < 100 else " ") + cardinal(rest)
    raise AssertionError


def ordinal(n):
    words = cardinal(n).split()
    last = words[-1]
    if last in _ORD_IRREG:
        words[-1] = _ORD_IRREG[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def year(n):
    n = int(n)
    if 2000 <= n <= 2009:
        return cardinal(n)
    if 1100 <= n <= 2099:
        hi, lo = divmod(n, 100)
        if lo == 0:
            return cardinal(hi) + " hundred"
        return cardinal(hi) + " " + ("oh " + _ONES[lo] if lo < 10 else cardinal(lo))
    return cardinal(n)


def time_words(h, m, ampm=None):
    h, m = int(h), int(m)
    s = cardinal(h)
    if m == 0:
        s += "" if ampm else " o'clock"
    elif m < 10:
        s += " oh " + _ONES[m]
    else:
        s += " " + cardinal(m)
    if ampm:
        s += " " + spell_letters(ampm.replace(".", "").upper())
    return s


_LETTER = {"A": "ay", "B": "bee", "C": "see", "D": "dee", "E": "ee", "F": "ef", "G": "gee", "H": "aitch",
           "I": "eye", "J": "jay", "K": "kay", "L": "el", "M": "em", "N": "en", "O": "oh", "P": "pee",
           "Q": "cue", "R": "ar", "S": "ess", "T": "tee", "U": "you", "V": "vee", "W": "double you",
           "X": "ex", "Y": "why", "Z": "zee"}


def spell_letters(s):
    return " ".join(_LETTER[c] for c in s if c in _LETTER)


_MONTH_RE = "|".join(m.capitalize() for m in MONTHS) + "|" + "|".join(MONTHS)
_RE_TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?:\s*([AaPp]\.?[Mm]\.?)(?![A-Za-z]))?")
_RE_DAY_MONTH = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + _MONTH_RE + r")\b")
_RE_MONTH_DAY = re.compile(r"\b(" + _MONTH_RE + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b(?!:)")
_RE_ORD = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
_RE_YEAR = re.compile(r"\b(1[1-9]\d\d|20\d\d)\b")
_RE_MONEY = re.compile(r"([$₹£€])\s?(\d[\d,]*)(?:\.(\d{1,2}))?")
_RE_PCT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s?%")
_RE_DEC = re.compile(r"\b(\d+)\.(\d+)\b")
_RE_NUM = re.compile(r"\d[\d,]*")
_RE_ACRONYM = re.compile(r"\b([A-Z]{2,5})\b")
_CURRENCY = {"$": "dollars", "₹": "rupees", "£": "pounds", "€": "euros"}


def normalize_numbers(text):
    text = _RE_TIME.sub(lambda m: " " + time_words(m.group(1), m.group(2), m.group(3)) + " ", text)
    text = _RE_DAY_MONTH.sub(lambda m: f"{ordinal(m.group(1))} {m.group(2)}", text)
    text = _RE_MONTH_DAY.sub(lambda m: f"{m.group(1)} {ordinal(m.group(2))}", text)
    text = _RE_MONEY.sub(lambda m: f"{cardinal(m.group(2).replace(',', ''))} {_CURRENCY[m.group(1)]}"
                         + (f" and {cardinal(m.group(3))} cents" if m.group(3) else ""), text)
    text = _RE_PCT.sub(lambda m: f"{m.group(1)} percent", text)
    text = _RE_ORD.sub(lambda m: ordinal(m.group(1)), text)
    text = _RE_YEAR.sub(lambda m: year(m.group(1)), text)
    text = _RE_DEC.sub(lambda m: f"{cardinal(m.group(1))} point {' '.join(_ONES[int(c)] for c in m.group(2))}", text)
    text = _RE_NUM.sub(lambda m: cardinal(m.group(0).replace(",", "")), text)
    return text


# ----------------------------------------------------------------------------- frontend
class EnglishFrontend(FrontendBase):
    name = "english"
    PHONES = ARPABET

    def __init__(self):
        super().__init__()
        self._g2p = None

    # g2p_en is imported lazily so normalisation can be used/tested without it
    @property
    def g2p(self):
        if self._g2p is None:
            import nltk

            for res in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "cmudict"):
                try:
                    nltk.download(res, quiet=True, raise_on_error=False)
                except TypeError:  # older nltk without raise_on_error
                    nltk.download(res, quiet=True)
            from g2p_en import G2p

            self._g2p = G2p()
        return self._g2p

    def normalize(self, text):
        t = unicodedata.normalize("NFKC", text)
        t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        t = t.replace("&", " and ")
        t = normalize_numbers(t)
        t = _RE_ACRONYM.sub(lambda m: spell_letters(m.group(1)), t)
        t = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", t)  # hyphenated words -> separate words
        t = re.sub(r"[\"()\[\]{}<>*_/\\|#@^~`+=]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def words_for_alignment(self, text):
        t = self.normalize(text).lower()
        t = re.sub(r"[^a-z' ]", " ", t)
        words = []
        for w in t.split():
            w = w.strip("'")
            if w:
                words.append(w)
        return words

    def _clean(self, phones):
        return [p for p in phones if p in self.vocab]

    def lexicon_entries(self, word):
        prons = []
        main = self._clean(self.g2p(word))
        if main:
            prons.append(main)
        h = getattr(self.g2p, "homograph2features", {}).get(word.lower())
        if h:
            for alt in (h[0], h[1]):
                alt = self._clean(alt)
                if alt and alt not in prons:
                    prons.append(alt)
        return prons

    def text_to_phones(self, text):
        norm = self.normalize(text)
        out = []
        for tok in self.g2p(norm):
            if tok in self.vocab and tok != PAUSE:
                out.append(tok)
            elif tok.strip() and all(c in PUNCT_PAUSE for c in tok.strip()):
                out.append(PAUSE)
        return self.collapse_pauses(out)
