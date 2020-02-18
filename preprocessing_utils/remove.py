"""
Mostly from textacy: https://chartbeat-labs.github.io/textacy/api_reference/text_processing.html.
Extracted here to avoid a heavy dependency.

Remove aspects of raw text that may be unwanted for certain use cases.
"""
import re
import unicodedata

from .resources import _get_punct_translation_table, RE_INLINE_MATH


def remove_inline_math(text):
    """Remove inline latex math.
    """
    return RE_INLINE_MATH.sub(' ', text)


def remove_accents(text, *, fast=False):
    """
    Remove accents from any accented unicode characters in ``text``, either by
    replacing them with ASCII equivalents or removing them entirely.

    Args:
        text (str)
        fast (bool): If False, accents are removed from any unicode symbol
            with a direct ASCII equivalent ; if True, accented chars
            for all unicode symbols are removed, regardless.

            .. note:: ``fast=True`` can be significantly faster than ``fast=False``,
               but its transformation of ``text`` is less "safe" and more likely
               to result in changes of meaning, spelling errors, etc.

    Returns:
        str

    Raises:
        ValueError: If ``method`` is not in {"unicode", "ascii"}.

    See Also:
        For a more powerful (but slower) alternative, check out ``unidecode``:
        https://github.com/avian2/unidecode
    """
    if fast is False:
        return "".join(
            char
            for char in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(char)
        )
    else:
        return (
            unicodedata.normalize("NFKD", text)
                .encode("ascii", errors="ignore")
                .decode("ascii")
        )


def remove_punctuation(text, *, marks=None):
    """
    Remove punctuation from ``text`` by replacing all instances of ``marks``
    with whitespace.

    Args:
        text (str)
        marks (str): Remove only those punctuation marks specified here.
            For example, ",;:" removes commas, semi-colons, and colons.
            If None, *all* unicode punctuation marks are removed.

    Returns:
        str

    Note:
        When ``marks=None``, Python's built-in :meth:`str.translate()` is
        used to remove punctuation; otherwise, a regular expression is used.
        The former's performance is about 5-10x faster.
    """
    if marks:
        return re.sub("[{}]+".format(re.escape(marks)), " ", text, flags=re.UNICODE)
    else:
        return text.translate(_get_punct_translation_table())


def remove_punctuation_keeping_periods(text):
    punct_table = _get_punct_translation_table()
    try:
        punct_table.pop(ord('.'))
    except KeyError:
        # '.' already removed from the cached punct_table
        pass
    return text.translate(punct_table)


def strip_copyright(text):
    patterns = [r'(copyright)?\s+\(c\).*', r'\s+(c)\s+\d\d\d\d.*']
    clean_text = text.lower()
    for p in patterns:
        clean_text = re.sub(p, '', clean_text)
    return clean_text
