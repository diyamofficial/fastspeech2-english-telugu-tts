"""Interface every language text frontend implements (see English/ and Telugu/)."""
import unicodedata

PAD = "_"
PAUSE = "sp"


class FrontendBase:
    """
    A frontend owns everything language-specific about text:
      normalize(text)            -> spoken-form string (numbers, dates, symbols expanded)
      words_for_alignment(text)  -> list of normalised words written to the MFA .lab file
      lexicon_entries(word)      -> list of pronunciations (lists of phones) for the MFA lexicon
      text_to_phones(text)       -> phone sequence for inference ("sp" at punctuation)
    and a fixed phone inventory, from which the vocabulary is built.
    """

    name = "base"
    PHONES = []

    def __init__(self):
        self.phone_inventory = list(self.PHONES)
        self.vocab = {PAD: 0, PAUSE: 1}
        for p in self.phone_inventory:
            if p not in self.vocab:
                self.vocab[p] = len(self.vocab)
        self.vocab_index = self.vocab

    # ---------------------------------------------------------------- helpers
    def phones_to_ids(self, phones):
        missing = [p for p in phones if p not in self.vocab]
        if missing:
            raise KeyError(f"phones not in the {self.name} vocabulary: {sorted(set(missing))}")
        return [self.vocab[p] for p in phones]

    @staticmethod
    def collapse_pauses(phones):
        out = []
        for p in phones:
            if p == PAUSE and (not out or out[-1] == PAUSE):
                continue
            out.append(p)
        while out and out[-1] == PAUSE:
            out.pop()
        return out

    @staticmethod
    def nfc(text):
        return unicodedata.normalize("NFC", text)

    # ---------------------------------------------------------------- to implement
    def normalize(self, text):
        raise NotImplementedError

    def words_for_alignment(self, text):
        raise NotImplementedError

    def lexicon_entries(self, word):
        raise NotImplementedError

    def text_to_phones(self, text):
        raise NotImplementedError
