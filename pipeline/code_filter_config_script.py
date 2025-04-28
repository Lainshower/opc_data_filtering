from collections import defaultdict

text_filter_config = defaultdict(dict)
code_filter_config = defaultdict(dict)

# text filter for Chinese text documents
text_filter_config['zh'] = {
    'qsc_doc_cate_code_related_file_name': None,
    'qsc_doc_frac_chars_replacement_symbols': 'lambda x: x > 0.01',
    'qsc_doc_entropy_unigram': 'lambda x: x < 3',
    'qsc_doc_frac_chars_curly_bracket': None,
    'qsc_doc_frac_chars_top_2grams': 'lambda x: x > 0.20',
    'qsc_doc_frac_chars_top_3grams': 'lambda x: x > 0.18',
    'qsc_doc_frac_chars_top_4grams': 'lambda x: x > 0.16',
    'qsc_doc_frac_chars_dupe_5grams': 'lambda x: x > 0.60',
    'qsc_doc_frac_chars_dupe_6grams': 'lambda x: x > 0.60',
    'qsc_doc_frac_chars_dupe_7grams': 'lambda x: x > 0.60',
    'qsc_doc_frac_chars_dupe_8grams': 'lambda x: x > 0.60',
    'qsc_doc_frac_chars_dupe_9grams': 'lambda x: x > 0.60',
    'qsc_doc_frac_chars_dupe_10grams': 'lambda x: x > 0.60',
    'qsc_doc_frac_chars_dupe_lines': 'lambda x: x > 0.50',
    'qsc_doc_frac_lines_dupe_lines': 'lambda x: x > 0.60',
    'qsc_doc_frac_lines_end_with_readmore': 'lambda x: x > 0.3',
    'qsc_doc_frac_lines_start_with_bullet': 'lambda x: x > 0.9',
    'qsc_doc_frac_words_all_caps': None,
    'qsc_doc_frac_words_full_bracket': 'lambda x: x > 0.1',
    'qsc_doc_frac_words_redpajama_stop': None,
    'qsc_doc_frac_words_unique': None,
    'qsc_doc_mean_word_length': 'lambda x: x < 1.3 or x > 10',
    'qsc_doc_num_chars': 'lambda x: x < 50',
    'qsc_doc_num_lines': 'lambda x: x < 10',
    'qsc_doc_num_sentences': 'lambda x: x == 1',
    'qsc_doc_num_words': 'lambda x: x < 50',

    'qsc_doc_frac_chars_hex_words': 'lambda x : x > 0.4',
    'qsc_doc_frac_chars_hyperlink_html_tag': 'lambda x : x > 0.5',
    'qsc_doc_frac_chars_alphabet': 'lambda x :  x < 0.5',
    'qsc_doc_frac_chars_digital': 'lambda x :  x > 0.2',
    'qsc_doc_frac_chars_whitespace': 'lambda x :  x > 0.5',
 }

# text filter for English and other text documents
text_filter_config['en'] = {
    'qsc_doc_cate_code_related_file_name': None,
    'qsc_doc_frac_chars_replacement_symbols': 'lambda x: x > 0.01',
    'qsc_doc_entropy_unigram': 'lambda x: x < 3',
    'qsc_doc_frac_chars_curly_bracket': None,
    'qsc_doc_frac_chars_top_2grams': 'lambda x: x > 0.20',
    'qsc_doc_frac_chars_top_3grams': 'lambda x: x > 0.18',
    'qsc_doc_frac_chars_top_4grams': 'lambda x: x > 0.16',
    'qsc_doc_frac_chars_dupe_5grams': 'lambda x: x > 0.60',
    'qsc_doc_frac_chars_dupe_6grams': 'lambda x: x > 0.60',
    'qsc_doc_frac_chars_dupe_7grams': 'lambda x: x > 0.60',
    'qsc_doc_frac_chars_dupe_8grams': 'lambda x: x > 0.60',
    'qsc_doc_frac_chars_dupe_9grams': 'lambda x: x > 0.60',
    'qsc_doc_frac_chars_dupe_10grams': 'lambda x: x > 0.60',
    'qsc_doc_frac_chars_dupe_lines': 'lambda x: x > 0.50',
    'qsc_doc_frac_lines_dupe_lines': 'lambda x: x > 0.60',
    'qsc_doc_frac_lines_end_with_readmore': 'lambda x: x > 0.1',
    'qsc_doc_frac_lines_start_with_bullet': 'lambda x: x > 0.9',
    'qsc_doc_frac_words_all_caps': 'lambda x: x > 0.5',
    'qsc_doc_frac_words_full_bracket': None,
    'qsc_doc_frac_words_redpajama_stop': None,
    'qsc_doc_frac_words_unique': None,
    'qsc_doc_mean_word_length': 'lambda x: x < 3 or x > 10',
    'qsc_doc_num_chars': 'lambda x: x < 50',
    'qsc_doc_num_lines': 'lambda x: x < 10',
    'qsc_doc_num_sentences': 'lambda x: x == 1',
    'qsc_doc_num_words': 'lambda x: x < 50',

    'qsc_doc_frac_chars_hex_words': 'lambda x : x > 0.4',
    'qsc_doc_frac_chars_hyperlink_html_tag': 'lambda x : x > 0.5',
    'qsc_doc_frac_chars_alphabet': 'lambda x :  x < 0.5',
    'qsc_doc_frac_chars_digital': 'lambda x :  x > 0.2',
    'qsc_doc_frac_chars_whitespace': 'lambda x :  x > 0.5',
 }


# code filter for general code documents
code_filter_config['others'] = {    
    # from text
    'qsc_code_frac_chars_replacement_symbols': 'lambda x: x > 0.01',
    'qsc_code_num_words': 'lambda x: x < 30',
    'qsc_code_num_chars': 'lambda x: x < 50',
    'qsc_code_mean_word_length': 'lambda x: x < 2 or x > 10',
    'qsc_code_frac_words_unique': None,
    'qsc_code_frac_chars_top_2grams': 'lambda x: x > 0.20',
    'qsc_code_frac_chars_top_3grams': 'lambda x: x > 0.18',
    'qsc_code_frac_chars_top_4grams': 'lambda x: x > 0.16',
    'qsc_code_frac_chars_dupe_5grams': 'lambda x: x > 0.80',
    'qsc_code_frac_chars_dupe_6grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_7grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_8grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_9grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_10grams': 'lambda x: x > 0.60',
    'qsc_code_frac_lines_dupe_lines': 'lambda x :  x > 0.7',

    # from code
    'qsc_code_size_file_byte': 'lambda x :  x > 3e6',
    'qsc_code_num_lines' : 'lambda x : x < 10 or x > 100000',
    'qsc_code_num_chars_line_max': 'lambda x : x > 1000',
    'qsc_code_num_chars_line_mean': 'lambda x :  x < 5 or x > 100',
    'qsc_code_frac_chars_alphabet': 'lambda x :  x < 0.5',
    'qsc_code_frac_chars_digital': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_whitespace': 'lambda x :  x > 0.5',
    'qsc_code_frac_chars_comments': 'lambda x :  x > 0.8',
    'qsc_code_cate_xml_start': 'lambda x : x == True',
    'qsc_code_cate_autogen': 'lambda x :  x == True',
    'qsc_code_frac_lines_long_string': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_string_length': 'lambda x :  x > 0.6',
    'qsc_code_frac_chars_long_word_length': 'lambda x :  x > 0.4',
    'qsc_code_frac_lines_string_concat': None,
    'qsc_code_cate_encoded_data': 'lambda x : x == True',
    'qsc_code_frac_chars_hex_words': 'lambda x : x > 0.4',
    'qsc_code_frac_lines_prompt_comments': 'lambda x :  x > 0.01',
    'qsc_code_frac_lines_assert': 'lambda x : x > 0.4'
    
} 

# code filter for data documents
code_filter_config['data'] = {
    
    # from text
    'qsc_code_frac_chars_replacement_symbols': 'lambda x: x > 0.01',
    'qsc_code_num_words': 'lambda x: x < 30',
    'qsc_code_num_chars': 'lambda x : x < 50 or x > 5000',
    'qsc_code_mean_word_length': 'lambda x: x < 2 or x > 10',
    'qsc_code_frac_words_unique': 'lambda x: x < 0.4',
    'qsc_code_frac_chars_top_2grams': 'lambda x: x > 0.20',
    'qsc_code_frac_chars_top_3grams': 'lambda x: x > 0.18',
    'qsc_code_frac_chars_top_4grams': 'lambda x: x > 0.16',
    'qsc_code_frac_chars_dupe_5grams': 'lambda x: x > 0.80',
    'qsc_code_frac_chars_dupe_6grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_7grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_8grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_9grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_10grams': 'lambda x: x > 0.60',
    'qsc_code_frac_lines_dupe_lines': 'lambda x :  x > 0.7',

    # from code
    'qsc_code_size_file_byte': 'lambda x :  x > 3e6',
    'qsc_code_num_lines' : 'lambda x : x < 10 or x > 100000',
    'qsc_code_num_chars_line_max': 'lambda x : x > 1000',
    'qsc_code_num_chars_line_mean': 'lambda x :  x < 5 or x > 100',
    'qsc_code_frac_chars_alphabet': 'lambda x :  x < 0.5',
    'qsc_code_frac_chars_digital': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_whitespace': 'lambda x :  x > 0.5',
    'qsc_code_frac_chars_comments': 'lambda x :  x > 0.8',
    'qsc_code_cate_xml_start': 'lambda x : x == True',
    'qsc_code_cate_autogen': 'lambda x :  x == True',
    'qsc_code_frac_lines_long_string': None,
    'qsc_code_frac_chars_string_length': None,
    'qsc_code_frac_chars_long_word_length': 'lambda x :  x > 0.4',
    'qsc_code_frac_lines_string_concat': None,
    'qsc_code_cate_encoded_data': 'lambda x : x == True',
    'qsc_code_frac_chars_hex_words': 'lambda x : x > 0.4',
    'qsc_code_frac_lines_prompt_comments': None,
    'qsc_code_frac_lines_assert': None
} 

# code filter for shell documents
code_filter_config['shell'] = {
    
    # from text
    'qsc_code_frac_chars_replacement_symbols': 'lambda x: x > 0.01',
    'qsc_code_num_words': 'lambda x: x < 30',
    'qsc_code_num_chars': 'lambda x: x < 50',
    'qsc_code_mean_word_length': 'lambda x: x < 2 or x > 10',
    'qsc_code_frac_words_unique': None,
    'qsc_code_frac_chars_top_2grams': 'lambda x: x > 0.20',
    'qsc_code_frac_chars_top_3grams': 'lambda x: x > 0.18',
    'qsc_code_frac_chars_top_4grams': 'lambda x: x > 0.16',
    'qsc_code_frac_chars_dupe_5grams': 'lambda x: x > 0.80',
    'qsc_code_frac_chars_dupe_6grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_7grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_8grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_9grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_10grams': 'lambda x: x > 0.60',
    'qsc_code_frac_lines_dupe_lines': 'lambda x :  x > 0.7',

    # from code
    'qsc_code_size_file_byte': 'lambda x :  x > 3e6',
    'qsc_code_num_lines' : 'lambda x : x < 10 or x > 100000',
    'qsc_code_num_chars_line_max': 'lambda x : x > 1000',
    'qsc_code_num_chars_line_mean': 'lambda x :  x < 5 or x > 100',
    'qsc_code_frac_chars_alphabet': 'lambda x :  x < 0.5',
    'qsc_code_frac_chars_digital': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_whitespace': 'lambda x :  x > 0.5',
    'qsc_code_frac_chars_comments': 'lambda x :  x > 0.8',
    'qsc_code_cate_xml_start': 'lambda x : x == True',
    'qsc_code_cate_autogen': 'lambda x :  x == True',
    'qsc_code_frac_lines_long_string': None,
    'qsc_code_frac_chars_string_length': None,
    'qsc_code_frac_chars_long_word_length': 'lambda x :  x > 0.4',
    'qsc_code_frac_lines_string_concat': None,
    'qsc_code_cate_encoded_data': 'lambda x : x == True',
    'qsc_code_frac_chars_hex_words': 'lambda x : x > 0.4',
    'qsc_code_frac_lines_prompt_comments': 'lambda x :  x > 0.01',
    'qsc_code_frac_lines_assert': None,
    
} 

# code filter for SQL documents
code_filter_config['sql'] = {

    # from text
    'qsc_code_frac_chars_replacement_symbols': 'lambda x: x > 0.01',
    'qsc_code_num_words': 'lambda x: x < 30',
    'qsc_code_num_chars': 'lambda x: x < 50',
    'qsc_code_mean_word_length': 'lambda x: x < 2 or x > 10',
    'qsc_code_frac_words_unique': None,
    'qsc_code_frac_chars_top_2grams': 'lambda x: x > 0.20',
    'qsc_code_frac_chars_top_3grams': 'lambda x: x > 0.18',
    'qsc_code_frac_chars_top_4grams': 'lambda x: x > 0.16',
    'qsc_code_frac_chars_dupe_5grams': 'lambda x: x > 0.80',
    'qsc_code_frac_chars_dupe_6grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_7grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_8grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_9grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_10grams': 'lambda x: x > 0.60',
    'qsc_code_frac_lines_dupe_lines': 'lambda x :  x > 0.7',

    # from code
    'qsc_code_size_file_byte': 'lambda x :  x > 3e6',
    'qsc_code_num_lines' : 'lambda x : x < 10 or x > 100000',
    'qsc_code_num_chars_line_max': 'lambda x : x > 1000',
    'qsc_code_num_chars_line_mean': 'lambda x :  x < 5 or x > 100',
    'qsc_code_frac_chars_alphabet': 'lambda x :  x < 0.5',
    'qsc_code_frac_chars_digital': 'lambda x :  x > 0.3',
    'qsc_code_frac_chars_whitespace': 'lambda x :  x > 0.5',
    'qsc_code_frac_chars_comments': 'lambda x :  x > 0.8',
    'qsc_code_cate_xml_start': 'lambda x : x == True',
    'qsc_code_cate_autogen': 'lambda x :  x == True',
    'qsc_code_frac_lines_long_string': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_string_length': 'lambda x :  x > 0.6',
    'qsc_code_frac_chars_long_word_length': 'lambda x :  x > 0.4',
    'qsc_code_frac_lines_string_concat': None,
    'qsc_code_cate_encoded_data': 'lambda x : x == True',
    'qsc_code_frac_chars_hex_words': 'lambda x : x > 0.4',
    'qsc_code_frac_lines_prompt_comments': 'lambda x :  x > 0.01',
    'qsc_code_frac_lines_assert': 'lambda x : x > 0.4',
}

# code filter for Python documents
code_filter_config['python'] = {    
    # from text
    'qsc_code_frac_chars_replacement_symbols': 'lambda x: x > 0.01',
    'qsc_code_num_words': 'lambda x: x < 30',
    'qsc_code_num_chars': 'lambda x: x < 50',
    'qsc_code_mean_word_length': 'lambda x: x < 2 or x > 10',
    'qsc_code_frac_words_unique': None,
    'qsc_code_frac_chars_top_2grams': 'lambda x: x > 0.20',
    'qsc_code_frac_chars_top_3grams': 'lambda x: x > 0.18',
    'qsc_code_frac_chars_top_4grams': 'lambda x: x > 0.16',
    'qsc_code_frac_chars_dupe_5grams': 'lambda x: x > 0.80',
    'qsc_code_frac_chars_dupe_6grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_7grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_8grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_9grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_10grams': 'lambda x: x > 0.60',
    'qsc_code_frac_lines_dupe_lines': 'lambda x :  x > 0.7',

    # from code
    'qsc_code_size_file_byte': 'lambda x :  x > 3e6',
    'qsc_code_num_lines' : 'lambda x : x < 10 or x > 100000',
    'qsc_code_num_chars_line_max': 'lambda x : x > 1000',
    'qsc_code_num_chars_line_mean': 'lambda x :  x < 5 or x > 100',
    'qsc_code_frac_chars_alphabet': 'lambda x :  x < 0.5',
    'qsc_code_frac_chars_digital': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_whitespace': 'lambda x :  x > 0.5',
    'qsc_code_frac_chars_comments': 'lambda x :  x > 0.8',
    'qsc_code_cate_xml_start': 'lambda x : x == True',
    'qsc_code_cate_autogen': 'lambda x :  x == True',
    'qsc_code_frac_lines_long_string': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_string_length': 'lambda x :  x > 0.6',
    'qsc_code_frac_chars_long_word_length': 'lambda x :  x > 0.4',
    'qsc_code_frac_lines_string_concat': None,
    'qsc_code_cate_encoded_data': 'lambda x : x == True',
    'qsc_code_frac_chars_hex_words': 'lambda x : x > 0.4',
    'qsc_code_frac_lines_prompt_comments': 'lambda x :  x > 0.01',
    'qsc_code_frac_lines_assert': 'lambda x : x > 0.4',

    # specific
    'qsc_codepython_cate_ast': 'lambda x: x == False',
    'qsc_codepython_frac_lines_func_ratio': 'lambda x: x > 0.2',
    'qsc_codepython_cate_var_zero': 'lambda x:  x == True',
    'qsc_codepython_frac_lines_pass': 'lambda x: x > 0.05',
    'qsc_codepython_frac_lines_import': 'lambda x : x > 0.3',
    'qsc_codepython_frac_lines_simplefunc': 'lambda x : x > 0.1',
    'qsc_codepython_score_lines_no_logic': 'lambda x : x > 0.5 and random.random() > 0.2',
    'qsc_codepython_frac_lines_print': 'lambda x : x > 0.4'
}

# code filter for C++ documents
code_filter_config['cpp'] = {

    # from text
    'qsc_code_frac_chars_replacement_symbols': 'lambda x: x > 0.01',
    'qsc_code_num_words': 'lambda x: x < 30',
    'qsc_code_num_chars': 'lambda x: x < 50',
    'qsc_code_mean_word_length': 'lambda x: x < 2 or x > 10',
    'qsc_code_frac_words_unique': None,
    'qsc_code_frac_chars_top_2grams': 'lambda x: x > 0.20',
    'qsc_code_frac_chars_top_3grams': 'lambda x: x > 0.18',
    'qsc_code_frac_chars_top_4grams': 'lambda x: x > 0.16',
    'qsc_code_frac_chars_dupe_5grams': 'lambda x: x > 0.80',
    'qsc_code_frac_chars_dupe_6grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_7grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_8grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_9grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_10grams': 'lambda x: x > 0.60',
    'qsc_code_frac_lines_dupe_lines': 'lambda x :  x > 0.7',

    # from code
    'qsc_code_size_file_byte': 'lambda x :  x > 3e6',
    'qsc_code_num_lines' : 'lambda x : x < 10 or x > 100000',
    'qsc_code_num_chars_line_max': 'lambda x : x > 1000',
    'qsc_code_num_chars_line_mean': 'lambda x :  x < 5 or x > 100',
    'qsc_code_frac_chars_alphabet': 'lambda x :  x < 0.5',
    'qsc_code_frac_chars_digital': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_whitespace': 'lambda x :  x > 0.5',
    'qsc_code_frac_chars_comments': 'lambda x :  x > 0.8',
    'qsc_code_cate_xml_start': 'lambda x : x == True',
    'qsc_code_cate_autogen': 'lambda x :  x == True',
    'qsc_code_frac_lines_long_string': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_string_length': 'lambda x :  x > 0.6',
    'qsc_code_frac_chars_long_word_length': 'lambda x :  x > 0.4',
    'qsc_code_frac_lines_string_concat': None,
    'qsc_code_cate_encoded_data': 'lambda x : x == True',
    'qsc_code_frac_chars_hex_words': 'lambda x : x > 0.4',
    'qsc_code_frac_lines_prompt_comments': 'lambda x :  x > 0.01',
    'qsc_code_frac_lines_assert': 'lambda x : x > 0.4',
    
    # specific
    'qsc_codecpp_frac_lines_func_ratio': 'lambda x :  x > 0.2',
    'qsc_codecpp_cate_bitsstdc': None,
    'qsc_codecpp_nums_lines_main': 'lambda x : x > 1',
    'qsc_codecpp_frac_lines_goto': None,
    'qsc_codecpp_cate_var_zero': None,
    'qsc_codecpp_score_lines_no_logic': 'lambda x : x > 0.5 and random.random() > 0.2',
    'qsc_codecpp_frac_lines_preprocessor_directives': 'lambda x : x > 0.8',
    'qsc_codecpp_frac_lines_print': 'lambda x : x > 0.4'
}

# code filter for C documents
code_filter_config['c'] = {
    
    # from text
    'qsc_code_frac_chars_replacement_symbols': 'lambda x: x > 0.01',
    'qsc_code_num_words': 'lambda x: x < 30',
    'qsc_code_num_chars': 'lambda x: x < 50',
    'qsc_code_mean_word_length': 'lambda x: x < 2 or x > 10',
    'qsc_code_frac_words_unique': None,
    'qsc_code_frac_chars_top_2grams': 'lambda x: x > 0.20',
    'qsc_code_frac_chars_top_3grams': 'lambda x: x > 0.18',
    'qsc_code_frac_chars_top_4grams': 'lambda x: x > 0.16',
    'qsc_code_frac_chars_dupe_5grams': 'lambda x: x > 0.80',
    'qsc_code_frac_chars_dupe_6grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_7grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_8grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_9grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_10grams': 'lambda x: x > 0.60',

    # from code
    'qsc_code_size_file_byte': 'lambda x :  x > 3e6',
    'qsc_code_num_lines' : 'lambda x : x < 10 or x > 100000',
    'qsc_code_num_chars_line_max': 'lambda x : x > 1000',
    'qsc_code_num_chars_line_mean': 'lambda x :  x < 5 or x > 100',
    'qsc_code_frac_chars_alphabet': 'lambda x :  x < 0.5',
    'qsc_code_frac_chars_digital': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_whitespace': 'lambda x :  x > 0.5',
    'qsc_code_frac_chars_comments': 'lambda x :  x > 0.8',
    'qsc_code_cate_xml_start': 'lambda x : x == True',
    'qsc_code_frac_lines_dupe_lines': 'lambda x :  x > 0.7',
    'qsc_code_cate_autogen': 'lambda x :  x == True',
    'qsc_code_frac_lines_long_string': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_string_length': 'lambda x :  x > 0.6',
    'qsc_code_frac_chars_long_word_length': 'lambda x :  x > 0.4',
    'qsc_code_frac_lines_string_concat': None,
    'qsc_code_cate_encoded_data': 'lambda x : x == True',
    'qsc_code_frac_chars_hex_words': 'lambda x : x > 0.4',
    'qsc_code_frac_lines_prompt_comments': 'lambda x :  x > 0.01',
    'qsc_code_frac_lines_assert': 'lambda x : x > 0.4',
    
    # specific
    'qsc_codec_frac_lines_func_ratio': 'lambda x :  x > 0.2',
    'qsc_codec_cate_bitsstdc': None,
    'qsc_codec_nums_lines_main': 'lambda x :  x > 1',
    'qsc_codec_frac_lines_goto': None,
    'qsc_codec_cate_var_zero': None,
    'qsc_codec_score_lines_no_logic': 'lambda x : x > 0.5 and random.random() > 0.2',
    'qsc_codec_frac_lines_preprocessor_directives': 'lambda x : x > 0.8',
    'qsc_codec_frac_lines_print': 'lambda x : x > 0.4'
}

# code filter for C# documents
code_filter_config['csharp'] = {
    
    # from text
    'qsc_code_frac_chars_replacement_symbols': 'lambda x: x > 0.01',
    'qsc_code_num_words': 'lambda x: x < 30',
    'qsc_code_num_chars': 'lambda x: x < 50',
    'qsc_code_mean_word_length': 'lambda x: x < 2 or x > 10',
    'qsc_code_frac_words_unique': None,
    'qsc_code_frac_chars_top_2grams': 'lambda x: x > 0.20',
    'qsc_code_frac_chars_top_3grams': 'lambda x: x > 0.18',
    'qsc_code_frac_chars_top_4grams': 'lambda x: x > 0.16',
    'qsc_code_frac_chars_dupe_5grams': 'lambda x: x > 0.80',
    'qsc_code_frac_chars_dupe_6grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_7grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_8grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_9grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_10grams': 'lambda x: x > 0.60',

    # from code
    'qsc_code_size_file_byte': 'lambda x :  x > 3e6',
    'qsc_code_num_lines' : 'lambda x : x < 10 or x > 100000',
    'qsc_code_num_chars_line_max': 'lambda x : x > 1000',
    'qsc_code_num_chars_line_mean': 'lambda x :  x < 5 or x > 100',
    'qsc_code_frac_chars_alphabet': 'lambda x :  x < 0.5',
    'qsc_code_frac_chars_digital': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_whitespace': 'lambda x :  x > 0.5',
    'qsc_code_frac_chars_comments': 'lambda x :  x > 0.8',
    'qsc_code_cate_xml_start': 'lambda x : x == True',
    'qsc_code_frac_lines_dupe_lines': 'lambda x :  x > 0.7',
    'qsc_code_cate_autogen': 'lambda x :  x == True',
    'qsc_code_frac_lines_long_string': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_string_length': 'lambda x :  x > 0.6',
    'qsc_code_frac_chars_long_word_length': 'lambda x :  x > 0.4',
    'qsc_code_frac_lines_string_concat': None,
    'qsc_code_cate_encoded_data': 'lambda x : x == True',
    'qsc_code_frac_chars_hex_words': 'lambda x : x > 0.4',
    'qsc_code_frac_lines_prompt_comments': 'lambda x :  x > 0.01',
    'qsc_code_frac_lines_assert': 'lambda x : x > 0.4',
    
    # specific
    'qsc_codecsharp_frac_lines_func_ratio': 'lambda x :  x > 0.2',
    'qsc_codecsharp_cate_var_zero': 'lambda x : x == True',
    'qsc_codecsharp_score_lines_no_logic': 'lambda x : x > 0.5 and random.random() > 0.2',
    'qsc_codecsharp_frac_lines_print': 'lambda x : x > 0.4'
}

# code filter for Golang documents
code_filter_config['go'] = {
    
    # from text
    'qsc_code_frac_chars_replacement_symbols': 'lambda x: x > 0.01',
    'qsc_code_num_words': 'lambda x: x < 30',
    'qsc_code_num_chars': 'lambda x: x < 50',
    'qsc_code_mean_word_length': 'lambda x: x < 2 or x > 10',
    'qsc_code_frac_words_unique': None,
    'qsc_code_frac_chars_top_2grams': 'lambda x: x > 0.20',
    'qsc_code_frac_chars_top_3grams': 'lambda x: x > 0.18',
    'qsc_code_frac_chars_top_4grams': 'lambda x: x > 0.16',
    'qsc_code_frac_chars_dupe_5grams': 'lambda x: x > 0.80',
    'qsc_code_frac_chars_dupe_6grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_7grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_8grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_9grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_10grams': 'lambda x: x > 0.60',

    # from code
    'qsc_code_size_file_byte': 'lambda x :  x > 3e6',
    'qsc_code_num_lines' : 'lambda x : x < 10 or x > 100000',
    'qsc_code_num_chars_line_max': 'lambda x : x > 1000',
    'qsc_code_num_chars_line_mean': 'lambda x :  x < 5 or x > 100',
    'qsc_code_frac_chars_alphabet': 'lambda x :  x < 0.5',
    'qsc_code_frac_chars_digital': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_whitespace': 'lambda x :  x > 0.5',
    'qsc_code_frac_chars_comments': 'lambda x :  x > 0.8',
    'qsc_code_cate_xml_start': 'lambda x : x == True',
    'qsc_code_frac_lines_dupe_lines': 'lambda x :  x > 0.7',
    'qsc_code_cate_autogen': 'lambda x :  x == True',
    'qsc_code_frac_lines_long_string': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_string_length': 'lambda x :  x > 0.6',
    'qsc_code_frac_chars_long_word_length': 'lambda x :  x > 0.4',
    'qsc_code_frac_lines_string_concat': None,
    'qsc_code_cate_encoded_data': 'lambda x : x == True',
    'qsc_code_frac_chars_hex_words': 'lambda x : x > 0.4',
    'qsc_code_frac_lines_prompt_comments': 'lambda x :  x > 0.01',
    'qsc_code_frac_lines_assert': None,
    
    # specific
    'qsc_codego_cate_testfile': 'lambda x : x == True and random.random() > 0.2',
    'qsc_codego_frac_lines_func_ratio': 'lambda x : x > 0.2',
    'qsc_codego_cate_var_zero': 'lambda x : x == True',
    'qsc_codego_score_lines_no_logic': 'lambda x : x > 0.5 and random.random() > 0.2',
    'qsc_codego_frac_lines_print': 'lambda x : x > 0.4'
}

# code filter for HTML documents
code_filter_config['html'] = {
    
    # from text
    'qsc_code_frac_chars_replacement_symbols': 'lambda x: x > 0.01',
    'qsc_code_num_words': 'lambda x: x < 30',
    'qsc_code_num_chars': 'lambda x: x < 50',
    'qsc_code_mean_word_length': 'lambda x: x < 2 or x > 10',
    'qsc_code_frac_words_unique': None,
    'qsc_code_frac_chars_top_2grams': 'lambda x: x > 0.20',
    'qsc_code_frac_chars_top_3grams': 'lambda x: x > 0.18',
    'qsc_code_frac_chars_top_4grams': 'lambda x: x > 0.16',
    'qsc_code_frac_chars_dupe_5grams': 'lambda x: x > 0.60',
    'qsc_code_frac_chars_dupe_6grams': 'lambda x: x > 0.60',
    'qsc_code_frac_chars_dupe_7grams': 'lambda x: x > 0.60',
    'qsc_code_frac_chars_dupe_8grams': 'lambda x: x > 0.60',
    'qsc_code_frac_chars_dupe_9grams': 'lambda x: x > 0.60',
    'qsc_code_frac_chars_dupe_10grams': 'lambda x: x > 0.60',

    # from code
    'qsc_code_size_file_byte': 'lambda x :  x > 3e6',
    'qsc_code_num_lines' : 'lambda x : x < 10 or x > 100000',
    'qsc_code_num_chars_line_max': 'lambda x : x > 1000',
    'qsc_code_num_chars_line_mean': 'lambda x :  x < 5 or x > 100',
    'qsc_code_frac_chars_alphabet': 'lambda x :  x < 0.5',
    'qsc_code_frac_chars_digital': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_whitespace': 'lambda x :  x > 0.5',
    'qsc_code_frac_chars_comments': None,
    'qsc_code_cate_xml_start': 'lambda x : x == True',
    'qsc_code_frac_lines_dupe_lines': 'lambda x :  x > 0.7',
    'qsc_code_cate_autogen': 'lambda x :  x == True',
    'qsc_code_frac_lines_long_string': None,
    'qsc_code_frac_chars_string_length': None,
    'qsc_code_frac_chars_long_word_length': None,
    'qsc_code_frac_lines_string_concat': None,
    'qsc_code_cate_encoded_data': 'lambda x : x == True',
    'qsc_code_frac_chars_hex_words': 'lambda x : x > 0.4',
    'qsc_code_frac_lines_prompt_comments': None,
    'qsc_code_frac_lines_assert': None,
    
    # specific
    'qsc_codehtml_cate_ast': 'lambda x : x == False',
    'qsc_codehtml_frac_words_text': 'lambda x : x < 0.2',
    'qsc_codehtml_num_chars_text': 'lambda x : x < 100'
}

# code filter for Java documents
code_filter_config['java'] = {
    
    # from text
    'qsc_code_frac_chars_replacement_symbols': 'lambda x: x > 0.01',
    'qsc_code_num_words': 'lambda x: x < 30',
    'qsc_code_num_chars': 'lambda x: x < 50',
    'qsc_code_mean_word_length': 'lambda x: x < 2 or x > 10',
    'qsc_code_frac_words_unique': None,
    'qsc_code_frac_chars_top_2grams': 'lambda x: x > 0.20',
    'qsc_code_frac_chars_top_3grams': 'lambda x: x > 0.18',
    'qsc_code_frac_chars_top_4grams': 'lambda x: x > 0.16',
    'qsc_code_frac_chars_dupe_5grams': 'lambda x: x > 0.80',
    'qsc_code_frac_chars_dupe_6grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_7grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_8grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_9grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_10grams': 'lambda x: x > 0.60',

    # from code
    'qsc_code_size_file_byte': 'lambda x :  x > 3e6',
    'qsc_code_num_lines' : 'lambda x : x < 10 or x > 100000',
    'qsc_code_num_chars_line_max': 'lambda x : x > 1000',
    'qsc_code_num_chars_line_mean': 'lambda x :  x < 5 or x > 100',
    'qsc_code_frac_chars_alphabet': 'lambda x :  x < 0.5',
    'qsc_code_frac_chars_digital': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_whitespace': 'lambda x :  x > 0.5',
    'qsc_code_frac_chars_comments': 'lambda x :  x > 0.8',
    'qsc_code_cate_xml_start': 'lambda x : x == True',
    'qsc_code_frac_lines_dupe_lines': 'lambda x :  x > 0.7',
    'qsc_code_cate_autogen': 'lambda x :  x == True',
    'qsc_code_frac_lines_long_string': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_string_length': 'lambda x :  x > 0.6',
    'qsc_code_frac_chars_long_word_length': 'lambda x :  x > 0.4',
    'qsc_code_frac_lines_string_concat': None,
    'qsc_code_cate_encoded_data': 'lambda x : x == True',
    'qsc_code_frac_chars_hex_words': 'lambda x : x > 0.4',
    'qsc_code_frac_lines_prompt_comments': 'lambda x :  x > 0.01',
    'qsc_code_frac_lines_assert': 'lambda x : x > 0.4',
    
    # specific
    'qsc_codejava_cate_var_zero': 'lambda x: x == True',
    'qsc_codejava_frac_lines_func_ratio': 'lambda x: x > 0.2',
    'qsc_codejava_score_lines_no_logic': 'lambda x : x > 0.5 and random.random() > 0.2',
    'qsc_codejava_frac_words_no_modifier': None,
    'qsc_codejava_frac_words_legal_var_name': None,
    'qsc_codejava_frac_words_legal_func_name': None,
    'qsc_codejava_frac_words_legal_class_name': None,
    'qsc_codejava_frac_lines_print': 'lambda x : x > 0.4'
}

# code filter for JavaScript documents
code_filter_config["javascript"] = {
    
    # from text
    'qsc_code_frac_chars_replacement_symbols': 'lambda x: x > 0.01',
    'qsc_code_num_words': 'lambda x: x < 30',
    'qsc_code_num_chars': 'lambda x: x < 50',
    'qsc_code_mean_word_length': 'lambda x: x < 2 or x > 10',
    'qsc_code_frac_words_unique': None,
    'qsc_code_frac_chars_top_2grams': 'lambda x: x > 0.20',
    'qsc_code_frac_chars_top_3grams': 'lambda x: x > 0.18',
    'qsc_code_frac_chars_top_4grams': 'lambda x: x > 0.16',
    'qsc_code_frac_chars_dupe_5grams': 'lambda x: x > 0.80',
    'qsc_code_frac_chars_dupe_6grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_7grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_8grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_9grams': 'lambda x: x > 0.70',
    'qsc_code_frac_chars_dupe_10grams': 'lambda x: x > 0.60',

    # from code
    'qsc_code_size_file_byte': 'lambda x :  x > 3e6',
    'qsc_code_num_lines' : 'lambda x : x < 10 or x > 100000',
    'qsc_code_num_chars_line_max': 'lambda x : x > 1000',
    'qsc_code_num_chars_line_mean': 'lambda x :  x < 5 or x > 100',
    'qsc_code_frac_chars_alphabet': 'lambda x :  x < 0.5',
    'qsc_code_frac_chars_digital': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_whitespace': 'lambda x :  x > 0.5',
    'qsc_code_frac_chars_comments': 'lambda x :  x > 0.8',
    'qsc_code_cate_xml_start': 'lambda x : x == True',
    'qsc_code_frac_lines_dupe_lines': 'lambda x :  x > 0.7',
    'qsc_code_cate_autogen': 'lambda x :  x == True',
    'qsc_code_frac_lines_long_string': 'lambda x :  x > 0.2',
    'qsc_code_frac_chars_string_length': 'lambda x :  x > 0.6',
    'qsc_code_frac_chars_long_word_length': 'lambda x :  x > 0.4',
    'qsc_code_frac_lines_string_concat': None,
    'qsc_code_cate_encoded_data': 'lambda x : x == True',
    'qsc_code_frac_chars_hex_words': 'lambda x : x > 0.4',
    'qsc_code_frac_lines_prompt_comments': 'lambda x :  x > 0.01',
    'qsc_code_frac_lines_assert': 'lambda x : x > 0.4',
    
    # specific
    'qsc_codejavascript_cate_ast': 'lambda x : x == False',
    'qsc_codejavascript_cate_var_zero': 'lambda x : x == True',
    'qsc_codejavascript_frac_lines_func_ratio': 'lambda x : x > 0.2',
    'qsc_codejavascript_num_statement_line': None,
    'qsc_codejavascript_score_lines_no_logic' : 'lambda x : x > 0.5 and random.random() > 0.2',
    'qsc_codejavascript_frac_words_legal_var_name': None,
    'qsc_codejavascript_frac_words_legal_func_name': None,
    'qsc_codejavascript_frac_words_legal_class_name': None,
    'qsc_codejavascript_frac_lines_print': 'lambda x : x > 0.4'
}