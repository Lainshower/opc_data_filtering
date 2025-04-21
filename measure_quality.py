import os
from document import QSCodeDocument
from quality_signals.code_specific.python import *
from quality_signals.code import *
from quality_signals.doc import *

def measure_code_quality(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    doc = QSCodeDocument(
        content=content,
        filename=filename,
        language='en',
        extension='txt',
        file_size_in_byte=file_size,
        doc_type='text',
        program_lang='text'
    )

    # Python specific signals
    python_signals = {
        'qsc_codepython_cate_ast': QSC_CodePython_Cate_Ast(),
        'qsc_codepython_frac_lines_func_ratio': QSC_CodePython_Frac_Lines_Func_Ratio(),
        'qsc_codepython_cate_var_zero': QSC_CodePyton_Cate_Var_Zero(),
        'qsc_codepython_frac_lines_pass': QSC_CodePython_Frac_Lines_Pass(),
        'qsc_codepython_frac_lines_import': QSC_CodePython_Frac_Lines_Import(),
        'qsc_codepython_frac_lines_simplefunc': QSC_CodePython_Frac_Lines_SimpleFunc(),
        'qsc_codepython_score_lines_no_logic': QSC_CodePython_Score_Lines_No_Logic(),
    }

    # Code signals
    code_signals = {
        'qsc_code_size_file_byte': QSC_Code_Size_File_Byte(),
        'qsc_code_num_lines': QSC_Code_Num_Lines(),
        'qsc_code_num_chars_line_max': QSC_Code_Num_Chars_Line_Max(),
        'qsc_code_num_chars_line_mean': QSC_Code_Num_Chars_Line_Mean(),
        'qsc_code_frac_chars_alphabet': QSC_Code_Frac_Chars_Alphabet(),
        'qsc_code_frac_chars_comments': QSC_Code_Frac_Chars_Comments(),
        'qsc_code_cate_xml_start': QSC_Code_Cate_Xml_Start(),
        'qsc_code_frac_lines_dupe_lines': QSC_Code_Frac_Lines_Dupe_Lines(),
        'qsc_code_cate_autogen': QSC_Code_Cate_AutoGen(),
        'qsc_code_frac_lines_long_string': QSC_Code_Frac_Lines_Long_String(),
        'qsc_code_frac_chars_string_length': QSC_Code_Frac_Chars_String_Length(),
        'qsc_code_frac_chars_long_word_length': QSC_Code_Frac_Chars_Long_Word_Length(),
        'qsc_code_frac_lines_string_concat': QSC_Code_Frac_Lines_String_Concat(),
        'qsc_code_cate_encoded_data': QSC_Code_Cate_Encoded_Data(),
        'qsc_code_frac_chars_hex_words': QSC_Code_Frac_Chars_Hex_Words(),
        'qsc_code_frac_lines_prompt_comments': QSC_Code_Frac_Lines_Prompt_Comments(),
        'qsc_code_frac_lines_assert': QSC_Code_Frac_Lines_Assert(),
    }

    # Doc signals
    doc_signals = {
        'qsc_doc_frac_chars_curly_bracket': QSC_Doc_Frac_Chars_Curly_Bracket(),
        'qsc_doc_frac_words_redpajama_stop': QSC_Doc_Frac_Words_Redpajama_Stop(),
        'qsc_doc_num_sentences': QSC_Doc_Num_Sentences(),
        'qsc_doc_num_words': QSC_Doc_Num_Words(),
        'qsc_doc_num_chars': QSC_Doc_Num_Chars(),
        'qsc_doc_num_lines': QSC_Doc_Num_Lines(),
        'qsc_doc_mean_word_length': QSC_Doc_Mean_Word_Length(),
        'qsc_doc_frac_words_full_bracket': QSC_Doc_Frac_Words_Full_Bracket(),
        'qsc_doc_frac_lines_end_with_readmore': QSC_Doc_Frac_Lines_End_With_Readmore(),
        'qsc_doc_frac_lines_start_with_bullet': QSC_Doc_Frac_Lines_Start_With_Bullet(),
        'qsc_doc_frac_words_unique': QSC_Doc_Frac_Words_Unique(),
        'qsc_doc_entropy_unigram': QSC_Doc_Entropy_Unigram(),
        'qsc_doc_frac_words_all_caps': QSC_Doc_Frac_Words_All_Caps(),
        'qsc_doc_frac_lines_dupe_lines': QSC_Doc_Frac_Lines_Dupe_Lines(),
        'qsc_doc_frac_chars_dupe_lines': QSC_Doc_Frac_Chars_Dupe_Lines(),
        'qsc_doc_frac_chars_top_2grams': QSC_Doc_Frac_Chars_Top_2Grams(),
        'qsc_doc_frac_chars_top_3grams': QSC_Doc_Frac_Chars_Top_3Grams(),
        'qsc_doc_frac_chars_top_4grams': QSC_Doc_Frac_Chars_Top_4Grams(),
        'qsc_doc_frac_chars_dupe_5grams': QSC_Doc_Frac_Chars_Dupe_5Grams(),
        'qsc_doc_frac_chars_dupe_6grams': QSC_Doc_Frac_Chars_Dupe_6Grams(),
        'qsc_doc_frac_chars_dupe_7grams': QSC_Doc_Frac_Chars_Dupe_7Grams(),
        'qsc_doc_frac_chars_dupe_8grams': QSC_Doc_Frac_Chars_Dupe_8Grams(),
        'qsc_doc_frac_chars_dupe_9grams': QSC_Doc_Frac_Chars_Dupe_9Grams(),
        'qsc_doc_frac_chars_dupe_10grams': QSC_Doc_Frac_Chars_Dupe_10Grams(),
        'qsc_doc_frac_chars_replacement_symbols': QSC_Doc_Frac_Chars_Replacement_Symbol(),
        'qsc_doc_cate_code_related_file_name': QSC_Doc_Cate_Code_Related_File_Name(),
        'qsc_doc_num_chars_sentence_length_mean': QSC_Doc_Num_Chars_Sentence_Length_Mean(),
        'qsc_doc_frac_chars_hyperlink_html_tag': QSC_Doc_Frac_Chars_Url_Html_Tag(),
        'qsc_doc_frac_chars_alphabet': QSC_Doc_Frac_Chars_Alphabet(),
        'qsc_doc_frac_chars_digital': QSC_Doc_Frac_Chars_Digital(),
        'qsc_doc_frac_chars_whitespace': QSC_Doc_Frac_Chars_Whitespace(),
        'qsc_doc_frac_chars_hex_words': QSC_Doc_Frac_Chars_Hex_Words(),
    }

    print("\nPython-Specific Quality Signals:")
    print("-" * 50)
    for name, signal in python_signals.items():
        result = signal(doc)[0][2]
        if result is not None:
            print(f"{name:40}: {result:.3f}")
        else:
            print(f"{name:40}: None")

    print("\nCode Quality Signals:")
    print("-" * 50)
    for name, signal in code_signals.items():
        result = signal(doc)[0][2]
        if result is not None:
            print(f"{name:40}: {result:.3f}")
        else:
            print(f"{name:40}: None")

    print("\nDocument Quality Signals:")
    print("-" * 50)
    for name, signal in doc_signals.items():
        result = signal(doc)[0][2]
        if result is not None:
            print(f"{name:40}: {result:.3f}")
        else:
            print(f"{name:40}: None")

if __name__ == "__main__":
    measure_code_quality('/home/joonwon/LG_STACK/opc_data_filtering/test_data/Fixed-Input-Parameterization/requirements.txt')