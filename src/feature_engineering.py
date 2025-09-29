import pandas as pd
import string
import re
import regex

UNICODE_SCRIPTS = [
    "Adlam","Ahom","Anatolian_Hieroglyphs","Arabic","Armenian","Avestan","Balinese","Bamum",
    "Bassa_Vah","Batak","Bengali","Bhaiksuki","Bopomofo","Brahmi","Braille","Buginese","Buhid",
    "Canadian_Aboriginal","Carian","Cham","Cherokee","Coptic","Cuneiform","Cypriot","Cyrillic",
    "Devanagari","Dogra","Duployan","Egyptian_Hieroglyphs","Elbasan","Ethiopic","Georgian",
    "Glagolitic","Gothic","Grantha","Greek","Gujarati","Gurmukhi","Hangul","Han","Hanunoo",
    "Hatran","Hebrew","Hiragana","Imperial_Aramaic","Inscriptional_Pahlavi","Inscriptional_Parthian",
    "Javanese","Kaithi","Kannada","Katakana","Kayah_Li","Kharoshthi","Khmer","Khojki","Khudawadi",
    "Lao","Latin","Lepcha","Limbu","Linear_B","Lisu","Lycian","Lydian","Malayalam","Mandaic",
    "Manichaean","Meetei_Mayek","Mende_Kikakui","Meroitic_Cursive","Meroitic_Hieroglyphs","Miao",
    "Modi","Mongolian","Mro","Multani","Myanmar","Nabataean","New_Tai_Lue","Nko","Nushu",
    "Ogham","Ol_Chiki","Old_Hungarian","Old_Italic","Old_North_Arabian","Old_Permic",
    "Old_Persian","Old_Sogdian","Old_South_Arabian","Old_Turkic","Oriya","Osage","Osmanya",
    "Pahawh_Hmong","Palmyrene","Phags_Pa","Phoenician","Psalter_Pahlavi","Rejang","Runic",
    "Samaritan","Saurashtra","Sharada","Siddham","SignWriting","Sinhala","Sora_Sompeng","Soyombo",
    "Sundanese","Syloti_Nagri","Syriac","Tagalog","Tagbanwa","Tai_Le","Tai_Tham","Tai_Viet","Takri",
    "Tamil","Telugu","Thaana","Thai","Tibetan","Tifinagh","Ugaritic","Vai","Yi"
]
SKIP_SCRIPTS = {"Common", "Inherited", "Unknown"}
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", "should",
    "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your", "yours",
    "yourself", "yourselves"
}
URL_RE = re.compile(r'(?i)\b(?:https?://|www\.)[^\s<>"\'\)\]\},;:]+')
EMAIL_RE = re.compile(r'(?i)\b(?:mailto:)?[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}\b')

def get_code_markup_count(text: str) -> int:
    s = str(text)
    count = 0
    def _blank_out(spans):
        nonlocal s
        parts = []
        last = 0
        for a, b in spans:
            parts.append(s[last:a])
            parts.append(' ' * (b - a))
            last = b
        parts.append(s[last:])
        s = ''.join(parts)
    spans = [m.span() for m in re.finditer(r'```.*?```|~~~.*?~~~', s, flags=re.DOTALL)]
    count += len(spans)
    _blank_out(sorted(spans))
    spans = [m.span() for m in re.finditer(r'`[^`\n]+`', s)]
    count += len(spans)
    _blank_out(sorted(spans))
    spans = [m.span() for m in re.finditer(r'<\s*/?\s*[a-zA-Z][^>]*?>', s)]
    count += len(spans)
    _blank_out(sorted(spans))
    spans = [m.span() for m in re.finditer(r'!\[[^\]]*\]\([^\)]*\)|\[[^\]]+\]\([^\)]+\)', s)]
    count += len(spans)
    _blank_out(sorted(spans))
    count += len(re.findall(r'(?m)^\s{0,3}#{1,6}\s+', s))
    return count

def get_social_media_count(text: str) -> int:
    s = str(text)
    def _blank_out(spans, current_text):
        if not spans: return current_text
        parts = []
        last = 0
        for a, b in sorted(spans):
            parts.append(current_text[last:a])
            parts.append(' ' * (b - a))
            last = b
        parts.append(current_text[last:])
        return ''.join(parts)
    mentions = [m.span() for m in re.finditer(r'(?<!\S)@[\w\.]{1,50}', s, flags=re.UNICODE)]
    s = _blank_out(mentions, s)
    hashtags = [m.span() for m in re.finditer(r'(?<!\S)#[\w]{1,200}', s, flags=re.UNICODE)]
    s = _blank_out(hashtags, s)
    camels = [m.span() for m in re.finditer(r'\b[a-z]+[A-Z][A-Za-z0-9]*\b', s)]
    return len(mentions) + len(hashtags) + len(camels)

def get_language_count(text: str) -> int:
    text = str(text)
    detected = {
        script for script in UNICODE_SCRIPTS 
        if regex.search(r'\p{Script=' + script + r'}', text)
    }
    return len(detected - SKIP_SCRIPTS)

def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    content = df['content'].astype(str).fillna('')
    df['char_count'] = content.str.len()
    words = content.str.split()
    df['word_count'] = words.str.len()
    df['whitespace_count'] = content.str.count(r'\s')
    df['has_non_ascii'] = content.apply(lambda x: not x.isascii())
    safe_char_count = df['char_count'] + 1e-6
    safe_word_count = df['word_count'] + 1e-6
    df['avg_word_length'] = (df['char_count'] / safe_word_count).fillna(0)
    df['whitespace_ratio'] = (df['whitespace_count'] / safe_char_count).fillna(0)
    alnum = content.str.count(r'[a-zA-Z0-9]')
    special = content.str.count(r'[^a-zA-Z0-9\s]')
    df['alnum_special_ratio'] = (alnum / (alnum + special + 1e-6)).fillna(0)
    df['digit_proportion'] = (content.str.count(r'\d') / safe_char_count).fillna(0)
    df['uppercase_proportion'] = (content.str.count(r'[A-Z]') / safe_char_count).fillna(0)
    df['punctuation_proportion'] = content.apply(lambda x: sum(1 for char in x if char in string.punctuation)) / safe_char_count
    df['max_word_length'] = words.apply(lambda x: max(len(w) for w in x) if x else 0)
    df['min_word_length'] = words.apply(lambda x: min(len(w) for w in x) if x else 0)
    df['code_markup_count'] = content.apply(get_code_markup_count)
    df['url_email_count'] = content.str.count(URL_RE) + content.str.count(EMAIL_RE)
    df['social_media_count'] = content.apply(get_social_media_count)
    df['language_count'] = content.apply(get_language_count)
    def count_stopwords(text):
        words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        if not words: return 0.0
        return sum(1 for word in words if word in ENGLISH_STOPWORDS) / len(words)
    df['stopword_proportion'] = content.apply(count_stopwords)
    return df
