import json
import random

from pipeline.code_filter_config import text_filter_config, code_filter_config

import traceback


# programming languages that are supported by language-specific filters
support_program_langs = [
    'cpp',
    'csharp',
    'c',
    'go',
    'html',
    'java',
    'javascript',
    'python',
    'shell',
    'sql'
]


class CodeFilter(object):

    # do the filtering process for a document
    def do_filter(self, filters, quality_signal_json, effective, hit_map, err_msg):

        for filter_name, func_str in filters.items():
            if filter_name not in quality_signal_json:
                continue
            if func_str is None:
                continue
            try: 
                if quality_signal_json[filter_name] is None:
                    hit_map[filter_name] = 0
                    continue 

                func = eval(func_str) # convert string to function
                if func(quality_signal_json[filter_name]):
                    effective = 0
                    hit_map[filter_name] = 1
                else:
                    hit_map[filter_name] = 0
            except Exception as e:
                error_string = traceback.format_exc()
                err_msg[filter_name] = f"{filter_name}: value: {quality_signal_json[filter_name]}, error: {error_string}"
            
        return effective, hit_map, err_msg
    
    # processing entrance
    def evaluate(self, doc_type:str, lang:str, program_lang:str, quality_signal:str):
        hit_map = {} # hit_map for each filter
        effective = 1 # final result
        err_msg = {} # error messages for each filter

        # if doc_type is 'unknown', filter it out straightly
        if doc_type == 'unknown' or doc_type == 'excluded':
            final_result = {
                'effective': str(0),
                'hit_map': json.dumps(hit_map)
            }
            return json.dumps(final_result)
        
        # if there exists error in quality signal calculation process, filter it out straightly
        if not quality_signal:
            final_result = {
                'effective': str(0),
                'hit_map': json.dumps(hit_map)
            }
            return json.dumps(final_result)
        
        quality_signal_json = json.loads(quality_signal)
        # if doc_type is 'text', use text_filter_config
        if doc_type == 'text':
            text_filters = text_filter_config
            if lang == 'zh':
                effective, hit_map, err_msg = self.do_filter(text_filters['zh'], quality_signal_json, effective, hit_map, err_msg)
            else:
                effective, hit_map, err_msg = self.do_filter(text_filters['en'], quality_signal_json, effective, hit_map, err_msg)


        # if doc_type is 'code' or 'data', use code_filter_config
        elif doc_type == 'code' or doc_type == 'data':    

            if program_lang in support_program_langs:
                code_filters = code_filter_config[program_lang]
            elif doc_type == 'data':
                code_filters = code_filter_config['data']
            else:
                code_filters = code_filter_config['others']

            effective, hit_map, err_msg = self.do_filter(code_filters, quality_signal_json, effective, hit_map, err_msg)
        
        final_result = {
            'effective': str(effective),
            'hit_map': json.dumps(hit_map)
        }

        if len(err_msg) > 0:
            final_result['err_msg'] = json.dumps(err_msg)

        return json.dumps(final_result)
                
