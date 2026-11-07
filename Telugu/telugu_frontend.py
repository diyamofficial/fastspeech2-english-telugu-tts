"""
Telugu text frontend for the IndicTTS-Telugu FastSpeech 2 model.

Telugu orthography is close to phonemic, so instead of a learned G2P we use a
deterministic grapheme-to-phoneme converter over Unicode akshara structure:

  * NFC normalisation; ZWJ/ZWNJ and other invisible format characters removed
    (they change glyph shaping, not pronunciation); Telugu digits -> ASCII.
  * Numbers -> Telugu words: cardinals up to crores, the ordinal suffix -వ
    (12వ -> పన్నెండవ), times H:MM (10:45 -> పది నలభై ఐదు), decimals.
  * Latin letters inside Telugu text are read by their Telugu letter names (AM -> ఏ ఎమ్).
  * G2P (romanised phone symbols, one ASCII token per phoneme):
      consonant + virama (్)      -> bare consonant (conjuncts / word-final consonants)
      consonant + vowel sign      -> consonant + that vowel
      consonant alone             -> consonant + inherent "a" (kept word-finally; Telugu,
                                     unlike Hindi, has no schwa deletion)
      independent vowels          -> vowel
      anusvara (ం)                -> homorganic nasal: ng before velars, ny before palatals,
                                     nn before retroflexes, n before dentals, m before labials,
                                     m elsewhere / word-finally
      visarga (ః)                 -> h ;   chandrabindu (ఁ) -> dropped (archaic)
    Aspiration (kh, gh, ...), retroflexion (tt, dd, nn, ll, ss), vowel length
    (a/aa, i/ii, u/uu, e/ee, o/oo) and geminates (ట్టు -> tt tt u) are all kept as
    distinct phones because they are contrastive in Telugu.

Word boundaries are kept exactly as written: a space inside a word (as in test
sentence T3, "అడ విలో") is not silently repaired.
"""
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fs2.frontend_base import PAUSE, FrontendBase  # noqa: E402

# ----------------------------------------------------------------------------- script tables
VIRAMA = "్"
ANUSVARA = "ం"
VISARGA = "ః"
CHANDRABINDU = "ఁ"
AVAGRAHA = "ఽ"

INDEPENDENT_VOWELS = {
    "అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"], "ఉ": ["u"], "ఊ": ["uu"],
    "ఋ": ["r", "u"], "ౠ": ["r", "uu"], "ఌ": ["l", "u"], "ౡ": ["l", "uu"],
    "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"], "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"],
}
VOWEL_SIGNS = {
    "ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"],
    "ృ": ["r", "u"], "ౄ": ["r", "uu"], "ౢ": ["l", "u"], "ౣ": ["l", "uu"],
    "ె": ["e"], "ే": ["ee"], "ై": ["ai"], "ొ": ["o"], "ో": ["oo"], "ౌ": ["au"],
    "ౖ": ["ai"],  # AI length mark
}
CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "ch", "ఛ": "chh", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tt", "ఠ": "tth", "డ": "dd", "ఢ": "ddh", "ణ": "nn",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "rr", "ల": "l", "ళ": "ll", "ఴ": "zh",
    "వ": "v", "శ": "sh", "ష": "ss", "స": "s", "హ": "h",
    "ౘ": "ts", "ౙ": "dz", "ౚ": "rr",
}
VOWEL_PHONES = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o", "oo", "au"]
CONS_PHONES = sorted(set(CONSONANTS.values()))
TELUGU_PHONES = VOWEL_PHONES + CONS_PHONES

# anusvara assimilation classes (by the phone that FOLLOWS the anusvara)
_NASAL_BEFORE = {}
for p in ("k", "kh", "g", "gh", "ng"):
    _NASAL_BEFORE[p] = "ng"
for p in ("ch", "chh", "j", "jh", "ny", "ts", "dz"):
    _NASAL_BEFORE[p] = "ny"
for p in ("tt", "tth", "dd", "ddh", "nn"):
    _NASAL_BEFORE[p] = "nn"
for p in ("t", "th", "d", "dh", "n"):
    _NASAL_BEFORE[p] = "n"
for p in ("p", "ph", "b", "bh", "m"):
    _NASAL_BEFORE[p] = "m"

TELUGU_DIGITS = {chr(0x0C66 + i): str(i) for i in range(10)}
INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)

PUNCT_PAUSE = set(",.;:!?।॥")

# ----------------------------------------------------------------------------- numbers
_UNITS = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
          "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు",
          "పద్దెనిమిది", "పందొమ్మిది"]
_TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}


def te_cardinal(n):
    """Telugu cardinal number words (formal written style)."""
    n = int(n)
    if n < 0:
        return "మైనస్ " + te_cardinal(-n)
    if n < 20:
        return _UNITS[n]
    if n < 100:
        t, u = divmod(n, 10)
        return _TENS[t] + ("" if u == 0 else " " + _UNITS[u])
    if n < 1000:
        h, r = divmod(n, 100)
        if h == 1:
            return "వంద" if r == 0 else "నూట " + te_cardinal(r)
        return te_cardinal(h) + (" వందలు" if r == 0 else " వందల " + te_cardinal(r))
    if n < 100000:
        th, r = divmod(n, 1000)
        if th == 1:
            head = "వెయ్యి"
            return head if r == 0 else head + " " + te_cardinal(r)
        return te_cardinal(th) + (" వేలు" if r == 0 else " వేల " + te_cardinal(r))
    if n < 10000000:
        l, r = divmod(n, 100000)
        if l == 1:
            return "లక్ష" if r == 0 else "లక్ష " + te_cardinal(r)
        return te_cardinal(l) + (" లక్షలు" if r == 0 else " లక్షల " + te_cardinal(r))
    c, r = divmod(n, 10000000)
    if c == 1:
        return "కోటి" if r == 0 else "కోటి " + te_cardinal(r)
    return te_cardinal(c) + (" కోట్లు" if r == 0 else " కోట్ల " + te_cardinal(r))


def te_ordinal_stem(word):
    """Stem a cardinal so the ordinal suffix -వ can attach: పన్నెండు -> పన్నెండ, ఒకటి -> ఒకట,
    ఇరవై -> ఇరవయ్య (ఇరవయ్యవ)."""
    if word.endswith("ు") or word.endswith("ి"):  # ends in -u / -i vowel sign
        return word[:-1]
    if word.endswith("ై"):  # -ai -> -ayya
        return word[:-1] + "య్య"
    return word


_RE_TIME = re.compile(r"(\d{1,2}):(\d{2})")
_RE_DEC = re.compile(r"(\d+)\.(\d+)")
_RE_NUM_SUFFIX = re.compile(r"(\d[\d,]*)([ఀ-౿]*)")


def _num_with_suffix(m):
    num, suffix = m.group(1).replace(",", ""), m.group(2)
    words = te_cardinal(num)
    if not suffix:
        return " " + words + " "
    head, _, last = words.rpartition(" ")
    if suffix.startswith("వ"):  # ordinal: 12వ -> పన్నెండవ, 1వ -> ఒకటవ
        last = te_ordinal_stem(last)
    joined = last + suffix
    return " " + (head + " " + joined if head else joined) + " "


def te_normalize_numbers(text):
    text = "".join(TELUGU_DIGITS.get(c, c) for c in text)
    text = _RE_TIME.sub(lambda m: " " + te_cardinal(m.group(1)) + " "
                        + ("" if int(m.group(2)) == 0 else te_cardinal(m.group(2))) + " ", text)
    text = _RE_DEC.sub(lambda m: " " + te_cardinal(m.group(1)) + " పాయింట్ "
                       + " ".join(_UNITS[int(c)] for c in m.group(2)) + " ", text)
    text = re.sub(r"(\d[\d,]*)\s?%", lambda m: m.group(1) + " శాతం", text)
    text = _RE_NUM_SUFFIX.sub(_num_with_suffix, text)
    return text


_LATIN_LETTER_TE = {
    "A": "ఏ", "B": "బీ", "C": "సీ", "D": "డీ", "E": "ఈ", "F": "ఎఫ్", "G": "జీ", "H": "హెచ్", "I": "ఐ",
    "J": "జే", "K": "కే", "L": "ఎల్", "M": "ఎమ్", "N": "ఎన్", "O": "ఓ", "P": "పీ", "Q": "క్యూ", "R": "ఆర్",
    "S": "ఎస్", "T": "టీ", "U": "యూ", "V": "వీ", "W": "డబ్ల్యూ", "X": "ఎక్స్", "Y": "వై", "Z": "జెడ్",
}


# ----------------------------------------------------------------------------- G2P
def g2p_word(word):
    """Telugu word (Unicode, NFC) -> list of phone symbols."""
    phones = []
    pending_anusvara = []  # indices in `phones` holding a placeholder nasal
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if ch in CONSONANTS:
            phones.append(CONSONANTS[ch])
            nxt = word[i + 1] if i + 1 < n else ""
            if nxt == VIRAMA:
                i += 2
                continue
            if nxt in VOWEL_SIGNS:
                phones.extend(VOWEL_SIGNS[nxt])
                i += 2
                continue
            phones.append("a")  # inherent vowel
            i += 1
            continue
        if ch in INDEPENDENT_VOWELS:
            phones.extend(INDEPENDENT_VOWELS[ch])
        elif ch in VOWEL_SIGNS:  # stray vowel sign (malformed input): pronounce the vowel
            phones.extend(VOWEL_SIGNS[ch])
        elif ch == ANUSVARA:
            pending_anusvara.append(len(phones))
            phones.append("m")
        elif ch == VISARGA:
            phones.append("h")
        # VIRAMA without a consonant, CHANDRABINDU, AVAGRAHA, nukta: no sound of their own
        i += 1
    # homorganic assimilation of the anusvara to the following consonant
    for idx in pending_anusvara:
        if idx + 1 < len(phones):
            phones[idx] = _NASAL_BEFORE.get(phones[idx + 1], "m")
    return phones


class TeluguFrontend(FrontendBase):
    name = "telugu"
    PHONES = TELUGU_PHONES

    def normalize(self, text):
        t = unicodedata.normalize("NFC", text).translate(INVISIBLE)
        t = te_normalize_numbers(t)
        t = re.sub(r"[A-Za-z]+", lambda m: " " + " ".join(_LATIN_LETTER_TE[c] for c in m.group(0).upper()) + " ", t)
        t = t.replace("।", ".").replace("॥", ".")  # danda
        t = re.sub(r"[\"'“”‘’()\[\]{}<>*_/\\|#@^~`+=&-]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return unicodedata.normalize("NFC", t)

    @staticmethod
    def _tokens(norm):
        """Split into Telugu words and punctuation tokens."""
        return re.findall(r"[ఀ-౿]+|[,.;:!?]", norm)

    def words_for_alignment(self, text):
        return [t for t in self._tokens(self.normalize(text)) if t not in PUNCT_PAUSE and g2p_word(t)]

    def lexicon_entries(self, word):
        p = g2p_word(word)
        return [p] if p else []

    def text_to_phones(self, text):
        out = []
        for tok in self._tokens(self.normalize(text)):
            if tok in PUNCT_PAUSE:
                out.append(PAUSE)
            else:
                out.extend(g2p_word(tok))
        return self.collapse_pauses(out)
