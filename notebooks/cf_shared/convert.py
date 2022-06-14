import datetime
from io import BytesIO, StringIO
import os
import tokenize
import dis
import sys
import re
import keyword
import traceback
from typing import Callable, Dict, List, Optional, Tuple
import pandas as pd
import ast
import json
from joblib import Parallel, delayed

sys.path.append("..")
from cf_shared.utils import get_elapsed_us

ConversionResult = Tuple[str, str, int, str, Optional[str]]
"""A tuple that contains:
- The source file path
- The token file path
- The time it took in microseconds
- The status indicated by "s" for success or "f" for fail
- An optional stacktrace for results that failed
"""

def multireplace(string: str, replacements: Dict[str, str], ignore_case=False) -> str:
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :param bool ignore_case: whether the match should be case insensitive
    :rtype: str
    """
    # If there is nothing to replace, the result should be the input string.
    if replacements == {}:
        return string
    # If case insensitive, we need to normalize the old string so that later a replacement
    # can be found. For instance with {"HEY": "lol"} we should match and find a replacement for "hey",
    # "HEY", "hEy", etc.
    if ignore_case:
        def normalize_old(s):
            return s.lower()
        re_mode = re.IGNORECASE
    else:
        def normalize_old(s):
            return s
        re_mode = 0

    replacements = {normalize_old(key): val for key, val in replacements.items()}
    
    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    rep_sorted = sorted(replacements, key=len, reverse=True)
    rep_escaped = map(re.escape, rep_sorted)
    
    # Create a big OR regex that matches any of the substrings to replace
    pattern = re.compile("|".join(rep_escaped), re_mode)
    
    # For each match, look up the new string in the replacements, being the key the normalized old string
    return pattern.sub(lambda match: replacements[normalize_old(match.group(0))], string)

def converts(text: str) -> str:
    """Converts python source files into token files

    :param text: The python source code as a string
    :return: The token representation of the source file
    """
    replacements = {}
    for node in ast.iter_child_nodes(ast.parse(text)):
        if isinstance(node, ast.ImportFrom):
            replacements.update({node.module: 'MODULE'})
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            for i, v in enumerate(node.names):
                if(node.names[i].asname):
                    replacements.update({node.names[i].name: 'LIB'})                
                    replacements.update({node.names[i].asname: 'ALIAS'})
                else:
                    replacements.update({node.names[i].name: 'LIBRARY'})


    # remove * from the dictionary (handle from module import * statement)
    replacements.pop('*', None)
    # print('List of modules and libraries to replace:\n', replacements)

    med = multireplace(text, replacements, ignore_case = True)

    tokens = tokenize.tokenize(BytesIO(med.encode('utf-8')).readline)
        
    ### extract important data from the output of tokenize package

    last_line = 0
    last_pos = 0
    tokss = []
    for token in tokens:
        
        tok_org = token.string
        tok_text = token.string    
        tok_type = tokenize.tok_name[token.type]

        # convert keywords to upper
        if keyword.iskeyword(tok_text):
            tok_type = str.upper(tok_text)
        
        #extract operations
        # if tok_type == 'OP':
        #     tok_type = tok_text


        # getting rid of comments and empty lines
        if tok_type in ['NL','NEWLINE','COMMENT']:
            continue
        
        #retrieve the position
        tok_line = token.start[0]
        
        if last_line == tok_line:
            last_pos +=  1
        else:
            last_pos = 1
        tok_pos = last_pos
        last_line = tok_line
        
        tokss.append({'type':tok_type,
                            'original':tok_org,
                            'text':tok_text,
                            'line':tok_line,
                            'pos':tok_pos})

    toks = pd.DataFrame(tokss)

    # remove encoding lines and end of file
    toks.line = toks.line.astype('int')
    toks.pos = toks.pos.astype('int')
    toks = toks.loc[~((toks.type == 'ENCODING') | (toks.type == 'ENDMARKER'))]
    toks['doc'] = (toks.text.str.contains('"""') | toks.text.str.contains("'''"))
    toks = toks.loc[~(toks.doc)].drop(['doc'],axis=1)

    indent = 0
    last_line = 0

    tokss = [] # PERF

    for row in toks.itertuples():
        if row.type == "INDENT":
            indent +=1
            continue
        if row.type == "DEDENT":
            indent -=1
            continue
        if row.line != last_line:
            last_line = row.line
            tokss.append({'type':'\n'+indent*'\t',
                                'text':'\n'+indent*'\t',
                                'line':row.line,
                                'pos':row.pos-1})
    
    toks = pd.concat([toks, pd.DataFrame(tokss)])

    toks = toks.loc[~((toks.type=='INDENT') | (toks.type=='DEDENT'))]
    toks = toks.sort_values(['line','pos']).reset_index(drop=True)

    # drop the first row (empty line)
    toks.drop(toks.index[:1], inplace=True)

    src = text
    stdout_backup = sys.stdout
    dis_result = StringIO()
    sys.stdout = dis_result
    dis.dis(src)
    sys.stdout = stdout_backup

    lines = dis_result.getvalue().split('\n')

    # find global variables
    glbls = [].copy()    
    for l in lines:
        clean = l.replace('>>',' ').strip().split()
        if len(clean):
            try:
                int(clean[1])
                line = int(clean[0])
            except:
                clean = [str(line)]+clean
            if 'LOAD_GLOBAL' in clean:
                # print('found a global!')
                glbls.append((int(clean[0]),clean[-1].replace('(','').replace(')','')))

    for l,n in glbls:
        line_eq = toks.loc[toks.line==l]
        line_eq.loc[line_eq.text==n, 'type'] = 'GLOBAL_VARIABLE'
        toks.loc[toks.line==l] = line_eq
    
    toks.loc[toks['text'].isin(['LIBRARY','LIB','ALIAS','MODULE']), 'type'] = toks['text']

    code_converted = ' '.join(list(toks.type)).replace('\n ','\n').replace(' \n','\n').replace('\t ','\t').replace(' . ','.').replace(' (','(')

    final_replacements = {'GLOBAL_VARIABLE(':'FUNCTION_CALL(',                      
    #                       'NAME.NAME':'NAME',
                          'NAME(':'FUNCTION_CALL(',
                          'NAME':'LOCAL_VARIABLE'}

    return multireplace(code_converted, final_replacements, ignore_case = False)

def convert(file_path: str, output_file: str) -> None:
    """Converts python source files into token files

    :param file_path: The location of the source file
    :param output_file: The location where the token file will be written
    :return: returns nothing
    """
    with open (file_path, "r") as f:
        text = f.read()  

    code_converted = converts(text)

    with open(output_file,'w') as f:
        f.write(code_converted)

def convert_optional(file_path: str, converted_path: str) -> ConversionResult:
    """Runs `convert`, tracks the amount of time it takes in microseconds, and catches any errors it might throw
    
    :param file_path: The location of the source file
    :param output_file: The location where the token file will be written

    :return: 
    """
    try:
        start_time = datetime.datetime.now()

        convert(file_path, converted_path)

        return (file_path, converted_path, get_elapsed_us(start_time), "s", None)
    except Exception as e:
        exc_info = sys.exc_info()
        error_str = ''.join(traceback.format_exception(*exc_info))
        return (file_path, converted_path, get_elapsed_us(start_time), "f", error_str)

def get_converted_file_path_replace(converted_path: str, source_file_path: str) -> str:
    """Used as a `path_converter` parameter to `convert_paths`

    Replaces the extension (everything after the last `.`) with `.txt` and swaps out the directory for the converted directory.
    Example: `hello.py` -> `hello.txt`
    Not recommended if you encounter files that have different extensions, as there could be name collisions.

    :param converted_path: Folder name where converted files should be placed
    :param source_file_path: Path to the source file to be converted
    """
    base_name = os.path.basename(source_file_path)
    no_ext = base_name[:base_name.rfind(".")]
    return os.path.join(converted_path, no_ext + ".txt")

def get_converted_file_path_add(converted_path: str, source_file_path: str) -> str:
    """Used as a `path_converter` parameter to `convert_paths`

    Adds `.txt` at the end of the file name and swaps out the directory for the converted directory.
    Example:  `hello.py` -> `hello.py.txt`

    :param converted_path: Folder name where converted files should be placed
    :param source_file_path: Path to the source file to be converted
    """
    return os.path.join(converted_path ,os.path.basename(source_file_path) + ".txt")

def convert_paths(paths: List[str], converted_path: str, times_json: Optional[str] = None,
    n_threads: int = -1, debug_output = sys.stdout,
    path_converter: Callable[[str, str], str] = get_converted_file_path_add
) -> List[ConversionResult]:
    """Converts a list of python source files into a list of token files

    :param paths: The locations of the source files
    :param converted_path: The folder that the token files will be written into
    :return: A list of tuples that contains: the source file path, the token file path, the time it took in microseconds, and status indicated by "s" for success or "f" for fail 
    """
    converted_paths_before = list(map(lambda path: path_converter(converted_path, path), paths))
    
    print("CONVERTING {} PYTHON FILES".format(len(paths)), file=debug_output)
    converted_paths_opt = Parallel(n_jobs=n_threads)(delayed(convert_optional)(path, conv_path) for (path, conv_path) in zip(paths, converted_paths_before))
    if times_json is not None:
        with open(times_json,'w') as fd:
        #     fd.write(json.dumps(converted_paths_opt))
            json.dump(converted_paths_opt, fd)

    n_successful_conversions = sum([1 for el in converted_paths_opt if el[2] == "s"])
    print("RESULT: {} FILES IN, {} FILES OUT".format(len(converted_paths_before), n_successful_conversions), file=debug_output)
    return converted_paths_opt

def get_successful_conversions(converted_paths_opt: List[ConversionResult]) -> Tuple[List[str], List[str]]:
    """Converts the results from `convert_paths` into a list of source file locations and a list of token file locations
    
    :param converted_paths_opt: A list of `ConversionResult`s
    :return: A tuple containing: a list of source file locations and a list of token file locations
    """
    successful_conversions = list(filter(lambda x: x[-1] == "s", converted_paths_opt))
    paths = list(map(lambda x: x[0], successful_conversions))
    converted_paths = list(map(lambda x: x[1], successful_conversions))

    return paths, converted_paths

