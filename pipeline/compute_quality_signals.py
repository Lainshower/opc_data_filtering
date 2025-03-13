from base import quality_signals_registry
from document import QSCodeDocument

from collections import defaultdict
import json
import traceback
import time
import signal

# used for registration of all the quality signals
import quality_signals


prefix2program = {
    'qsc_codecpp_': 'cpp',
    'qsc_codecsharp_': 'csharp',
    'qsc_codec_': 'c',
    'qsc_codego_': 'go',
    'qsc_codehtml_': 'html',
    'qsc_codejava_': 'java',
    'qsc_codejavascript_': 'javascript',
    'qsc_codepython_': 'python'
}

def timed_out(signum, frame):
    raise TimeoutError("timeout")


class ComputeCodeQualitySignal():
    def __init__(self):
        self.code_instances = defaultdict(dict)
        self.specific_instances = defaultdict(dict)
        self.text_instances = defaultdict(dict)
        self.err_truncated_num = 10000 # used for truncating recorded error information

        for filter_name, filter in quality_signals_registry['codedocument'].items():
            # natural language filters
            if filter_name.startswith('qsc_doc_'):
                self.text_instances[filter_name] = filter()
            # general code filters
            elif filter_name.startswith('qsc_code_'):
                self.code_instances[filter_name] = filter()
            # language-specific code filters
            for prefix, program_lang in prefix2program.items():
                if filter_name.startswith(prefix):
                    self.specific_instances[program_lang][filter_name] = filter()

        
        self.default_time_out = 10
        self.filter_time = {} # {"quality signal name": timeout setting} You can set specific timeout for each filter
    
    def truncate_err_string(self, err_string:str):
        if self.err_truncated_num > 2000:
            return err_string[:self.err_truncated_num-1005] + "\n...\n" + err_string[-1000:]
        else:
            return err_string[:self.err_truncated_num]

    def get_timeout(self, name:str):
        return self.filter_time.get(name, self.default_time_out)

    def get_final_result(self, result, err_msg, time_map):
        final_result = {
            'quality_signal': json.dumps(result)
        }
        final_result['pre_hit'] = '0'
        if len(err_msg) != 0:
            final_result['err_msg'] = json.dumps(err_msg)
        if len(time_map) != 0:
            final_result['time_map'] = json.dumps(time_map)

        return final_result

    # compute a quality signal
    def compute_filters(self, document:QSCodeDocument, instances, result, time_map, err_msg):
        try:
            for qname, filter in instances.items():
                start = time.time()
                
                # enable the timeout signal for each filter
                signal.setitimer(signal.ITIMER_REAL, self.get_timeout(qname), 0)

                ret = filter(document)

                # disable the timeout signal
                signal.setitimer(signal.ITIMER_REAL, 0)
                
                score = ret[0][2]

                result[qname] = score
                filter_time = time.time() - start
                time_map[qname] = round(filter_time, 8)

        except TimeoutError:
            result[qname] = None
            err_msg[qname] = f"[WARN] {qname} time out error, time set: {self.default_time_out}"
            print(err_msg[qname], flush=True)

        except Exception as e:
            result[qname] = None
            error_string = traceback.format_exc()
            if len(error_string) > self.err_truncated_num:
                error_string = self.truncate_err_string(error_string)
            err_msg[qname] = f"[WARN] qname: {qname}, Exception: {error_string}"
            print(err_msg[qname], flush=True)
        
        return result, time_map, err_msg

    # compute quality signals for a document
    def compute_qs(self, text:str, filename:str, lang:str, ext:str, file_size_in_byte:int, program_lang:str, doc_type:str):

        signal.signal(signal.SIGALRM, timed_out)

        document = QSCodeDocument(content=text,
                                filename=filename,
                                language=lang,
                                extension=ext,
                                file_size_in_byte=file_size_in_byte,
                                program_lang=program_lang,
                                doc_type=doc_type
                                )
                    
        result = {}
        err_msg = {}
        time_map = {}
        
        # if doc_type is 'unknown' then do not process this document
        if doc_type == 'unknown': 
            pass
        
        # if doc_type is 'code' or 'data' then use general code filters and language-specific code filters
        elif doc_type == 'code' or doc_type == 'data':
            # general code filters
            result, time_map, err_msg = self.compute_filters(document, self.code_instances, result, time_map, err_msg)
            # language-specific code filters
            result, time_map, err_msg = self.compute_filters(document, self.specific_instances[program_lang], result, time_map, err_msg)

        # if doc_type is 'text' then use natural language filters
        elif doc_type == 'text':
            # natural language filters
            result, time_map, err_msg = self.compute_filters(document, self.text_instances, result, time_map, err_msg)

        final_result = self.get_final_result(result, err_msg, time_map)
        
        return json.dumps(final_result,ensure_ascii=False)

    # processing entrance
    def evaluate(self, text:str, filename:str, lang:str, ext:str, file_size_in_byte:int, 
                 program_lang:str, doc_type:str):

        try:
            result = self.compute_qs(text, filename, lang, ext, 
            file_size_in_byte, program_lang, doc_type)

            return result
        except Exception as e:
            error_string = traceback.format_exc()
            if len(error_string) > self.err_truncated_num:
                error_string = self.truncate_err_string(error_string)
            final_result = {
                'err_msg': json.dumps({
                    'total_crush': error_string
                })
            }
            return json.dumps(final_result,ensure_ascii=False)
