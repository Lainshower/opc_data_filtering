"""
Filters for code documents
"""

from collections import Counter
from base import QSCodeBase, register_quality_signal
from document import QSCodeDocument
from utils.code.code_utils import get_text_python

from redpajama.core.data_types import SignalType
from redpajama.core.constants import PRECISION

import regex as re


@register_quality_signal('qsc_code_size_file_byte', 'codedocument')
class QSC_Code_Size_File_Byte(QSCodeBase):
    """
    The size of the file in bytes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, code: QSCodeDocument) -> SignalType:
        if code.file_size_in_byte is None:
            return [(0, len(code), None)]
        
        score = float(code.file_size_in_byte)
        # score = round(score, PRECISION)
        return [(0, len(code), score)]


@register_quality_signal('qsc_code_num_lines', 'codedocument')
class QSC_Code_Num_Lines(QSCodeBase):
    """
    The number of lines.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, code: QSCodeDocument) -> SignalType:
        score = len(code.raw_lines)
        return [(0, len(code), float(score))]
    

@register_quality_signal('qsc_code_num_chars_line_max', 'codedocument')
class QSC_Code_Num_Chars_Line_Max(QSCodeBase):
    """
    The maximum number of characters in a line.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, code: QSCodeDocument) -> SignalType:
        if len(code.raw_lines) != 0:
            score = max(len(text_slice.text) for text_slice in code.raw_lines)
        else:
            score = 0
        return [(0, len(code), float(score))]


@register_quality_signal('qsc_code_num_chars_line_mean', 'codedocument')
class QSC_Code_Num_Chars_Line_Mean(QSCodeBase):
    """
    The mean number of characters in a line.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, code: QSCodeDocument) -> SignalType:
        if len(code.raw_lines) == 0:
            return [(0, len(code), 0)]

        score = sum(len(text_slice.text) for text_slice in code.raw_lines) / len(code.raw_lines)
        score = round(score, PRECISION)
        return [(0, len(code), score)]

@register_quality_signal('qsc_code_frac_chars_alphabet','codedocument')
class QSC_Code_Frac_Chars_Alphabet(QSCodeBase):
    """
    The fraction of characters that are alphabetic.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, code: QSCodeDocument) -> SignalType:
        if len(code.visible_content) == 0:
            return [(0, len(code), 0.0)]
        
        score = sum(c.isalpha() for c in code.visible_content)
        score = score / len(code.visible_content)

        score = round(score, PRECISION)

        return [(0, len(code), score)]


@register_quality_signal('qsc_code_frac_chars_comments', 'codedocument')
class QSC_Code_Frac_Chars_Comments(QSCodeBase):
    """
    The fraction of characters that are in comments.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, code: QSCodeDocument) -> SignalType:

        score = None
        if len(code.raw_content) == 0:
            return [(0, len(code), None)]
        if code.program_lang == 'python':
            comments = get_text_python(code.raw_content)
            score = len(comments) / len(code.raw_content)
        else:
            score = len(code.comment_raw_content) / len(code.raw_content)
        if score is None:
            return [(0, len(code), None)]
        score = round(score, PRECISION)
        return [(0, len(code), score)]


@register_quality_signal('qsc_code_cate_xml_start', 'codedocument')
class QSC_Code_Cate_Xml_Start(QSCodeBase):
    """
    Whether the code starts with an XML declaration '<?xml version='.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, code: QSCodeDocument) -> SignalType:
        truncated_num = 100
        truncated_content = code.raw_content
        if len(code.raw_content) > truncated_num:
            truncated_content = code.raw_content[:truncated_num]
        score = bool('<?xml version=' in truncated_content)
        return [(0, len(code), float(score))]


@register_quality_signal('qsc_code_frac_lines_dupe_lines', 'codedocument')
class QSC_Code_Frac_Lines_Dupe_Lines(QSCodeBase):
    """
    The fraction of lines that are duplicates.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _is_valid(self, line):
        invalid_lines = ['},']
        if line in invalid_lines or len(line) == 1:
            return False
        return True

    def __call__(self, code: QSCodeDocument) -> SignalType:
        if len(code.raw_lines) == 0 :
            return [(0, len(code), None)]
        
        lines = [item.text for item in code.code_normalized_lines]
        lines = [re.sub(r'\s+', '', line) for line in lines]
        line2count = Counter(lines)
        count = sum([v for k, v in line2count.items() if v != 1 and self._is_valid(k)])
        total = sum([v for v in line2count.values()])

        if total == 0:
            return [(0, len(code), None)]

        score = count / total
        score = round(score, PRECISION)
        return [(0, len(code), score)]


@register_quality_signal('qsc_code_cate_autogen', 'codedocument')
class QSC_Code_Cate_AutoGen(QSCodeBase):
    """
    The code is auto-generated or not.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        autogen_strs = [
            "auto-generated",
            "autogenerated",
            "automatically generated",
            "generated automatically",
            "this file is generated"
        ]
        self.autogen_pattern = re.compile("|".join(autogen_strs), re.IGNORECASE)

    def __call__(self, code: QSCodeDocument) -> SignalType:
        target_str = code.raw_content[:200]
        result1 = bool(self.autogen_pattern.search(target_str))
        result2 = 'generate' in target_str.lower()
        result3 = 'autogen' in target_str.lower()
        result4 = 'do not edit' in target_str.lower()
        return [(0, len(code), float(result1|result2|result3|result4))]



@register_quality_signal('qsc_code_frac_lines_long_string', 'codedocument')
class QSC_Code_Frac_Lines_Long_String(QSCodeBase):
    """
    The fraction of lines that contain a long string (word length > 20).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, code: QSCodeDocument) -> SignalType:
        lines = code.code_normalized_lines
        total = len(lines)

        if total == 0:
            return [(0, len(code), None)]

        # extract priority: three single quotes > three double quotes > single quotes > double quotes, using non-greedy match
        string_pattern = re.compile(r'\'\'\'(.*?)\'\'\'|"""(.*?)"""|\'(.*?)\'|"(.*?)"', re.DOTALL)
        matches = string_pattern.findall(code.code_raw_content)

        string_list = []
        for match in matches:
            for group in match:
                if group:
                    string_list.append(group)


        count = 0
        for string in string_list:
            sub_list = string.split('\n')
            for sub_string in sub_list:
                word_list = code.tokenize_func(sub_string)
                if len(word_list) > 20:
                    count += 1

        score = count / total
        score = round(score, PRECISION)
        return [(0, len(code), score)]
    

@register_quality_signal('qsc_code_frac_chars_string_length', 'codedocument')
class QSC_Code_Frac_Chars_String_Length(QSCodeBase):
    """
    The fraction of characters that are in strings.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, code: QSCodeDocument) -> SignalType:
        total = len(code.code_raw_content)
        if total == 0:
            return [(0, len(code), None)]
        
        # extract priority: three single quotes > three double quotes > single quotes > double quotes, using non-greedy match
        string_pattern = re.compile(r'\'\'\'(.*?)\'\'\'|"""(.*?)"""|\'([^\n]*?)\'|"([^\n]*?)"', re.DOTALL)
        matches = string_pattern.findall(code.code_raw_content)

        string_list = []
        for match in matches:
            for group in match:
                if group:
                    string_list.append(group)
        

        length = sum([len(string) for string in string_list])

        score = length / total
        score = round(score, PRECISION)
        return [(0, len(code), score)]
    

@register_quality_signal('qsc_code_frac_chars_long_word_length', 'codedocument')
class QSC_Code_Frac_Chars_Long_Word_Length(QSCodeBase):
    """
    The fraction of characters that are in long words (word length > 20).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, code: QSCodeDocument) -> SignalType:
        total = len(code.code_raw_content)
        if total == 0:
            return [(0, len(code), None)]
        
        string_pattern = re.compile(r'\'\'\'(.*?)\'\'\'|"""(.*?)"""|\'([^\n]*?)\'|"([^\n]*?)"', re.DOTALL)
        try:
            matches = string_pattern.findall(code.code_raw_content, timeout=1)
        except TimeoutError:
            return [(0, len(code), None)]

        word_list = []
        for match in matches:
            for group in match:
                if group:
                    word_list.extend(group.split())

        url_pattern = r'.*https?://\S+.*'
        long_word_list = [word for word in word_list if len(word) > 20 and re.match(url_pattern, word) is None]
        length = sum([len(word) for word in long_word_list])


        score = length / total
        score = round(score, PRECISION)
        return [(0, len(code), score)]


@register_quality_signal('qsc_code_frac_lines_string_concat', 'codedocument')
class QSC_Code_Frac_Lines_String_Concat(QSCodeBase):
    """
    The fraction of lines that contain string concatenation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, code: QSCodeDocument) -> SignalType:
        lines = code.code_normalized_lines
        total = len(lines)
        if total == 0:
            return [(0, len(code), None)]
        
        count = 0
        for line in lines:
            try:
                match = re.search(r'\s*=\s*["\'].+?["\']\s*\+\s*["\'].+?["\'].*', line.text, timeout=1)
            except TimeoutError:
                continue
            if match:
                count += 1

        # 计算占比
        score = count / total
        score = round(score, PRECISION)
        
        return [(0, len(code), score)]
    

@register_quality_signal('qsc_code_cate_encoded_data', 'codedocument')
class QSC_Code_Cate_Encoded_Data(QSCodeBase):
    """
    Whether the code contains encoded data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, code: QSCodeDocument) -> SignalType:
        raw_content = code.raw_content

        patterns = {
            'base64': re.compile(r'[a-zA-Z0-9+/\n=]{64,}'),
            'hexadecimal': re.compile(r'(?:\b(?:0x|\\x)?[0-9a-fA-F]{2}(?:,|\b\s*)){8,}'),
            'unicode': re.compile(r'(?:\\u[0-9a-fA-F]{4}){8,}')
        }

        score = 0.0
        total_matched_chars = 0

        for pattern in patterns.values():
            matches = pattern.findall(raw_content)
            for match in matches:
                if len(match) > 1024:
                    score = 1.0
                total_matched_chars += len(match)
        
        fraction_matched = total_matched_chars / len(raw_content) if len(raw_content) > 0 else 0.0
    
        if fraction_matched > 0.5:
            score = 1.0

        return [(0, len(code), score)]


@register_quality_signal('qsc_code_frac_chars_hex_words', 'codedocument')
class QSC_Code_Frac_Chars_Hex_Words(QSCodeBase):
    """
    The fraction of characters that are in hexadecimal words.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, code: QSCodeDocument) -> SignalType:
        total = len(code.code_raw_content)
        if total == 0:
            return [(0, len(code), None)]
        
        count = sum(len(element) for element in re.findall(r'\b0[xX][0-9a-fA-F]+\b', code.code_raw_content))
        score = count / total
        score = round(score, PRECISION)

        return [(0, len(code), score)]
    

@register_quality_signal('qsc_code_frac_lines_prompt_comments', 'codedocument')
class QSC_Code_Frac_Lines_Prompt_Comments(QSCodeBase):
    """
    The fraction of lines that contain prompt comments (e.g., 'your code here', 'fixme', 'todo').
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, code: QSCodeDocument) -> SignalType:
        prompt_comments = ['your code here', 'fixme', 'todo']

        total = len(code.raw_lines)
        if total == 0:
            return [(0, len(code), None)]
        
        count = 0
        for line in code.comment_normalized_lines:
            for prompt_comment in prompt_comments:
                if prompt_comment in line.text.lower():
                    count += 1
                    break
        
        score = count / total
        score = round(score, PRECISION)
        
        return [(0, len(code), score)]
    

@register_quality_signal('qsc_code_frac_lines_assert', 'codedocument')
class QSC_Code_Frac_Lines_Assert(QSCodeBase):
    """
    The fraction of lines that contain assertions (e.g., 'assert').
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, code: QSCodeDocument) -> SignalType:
        total = len(code.code_normalized_lines)
        if total == 0:
            return [(0, len(code), None)]
        
        count = sum(1 for line in code.code_normalized_lines if ('assert' in line.text.lower()))
        score = count / total
        score = round(score, PRECISION)
        
        return [(0, len(code), score)]