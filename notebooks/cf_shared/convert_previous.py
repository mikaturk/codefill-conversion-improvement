"""
WARNING: This module contains outdated code for testing and comparison purposes only, do not use in production.
Refer to `convert.py` for actual implementations.
"""

import traceback

"""
Comment out the following 2 lines to stop ignoring "FutureWarning"s produced by the original implementation
as a result of using the deprecated `DataFrame.append` method
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import datetime
import os
import shutil
import tempfile
import tokenize
import dis
import sys
import re
import keyword
import pandas as pd
import ast

# Add the parent folder to the path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cf_shared.convert import ConversionResult
from cf_shared.utils import get_elapsed_us

def multireplace(string, replacements, ignore_case=False):
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :param bool ignore_case: whether the match should be case insensitive
    :rtype: str
    """
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

def convert_original(file: str, output_file: str):
    """Warning: original version, only for reference timing
    
    Converts python source files into token files

    :param file_path: The location of the source file
    :param output_file: The location where the token file will be written
    :return: returns nothing
    """
    with open (file, "r") as f:
        text = f.read()  

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


    # reomve * from the dictionary (handle from module import * statement)
    replacements.pop('*', None)
    # print('List of modules and libraries to replace:\n', replacements)

    with open('med.py','w') as f:
        f.write(multireplace(text, replacements, ignore_case = True))

    file = 'med.py'
    with open(file,'rb') as f:
        tokens = list(tokenize.tokenize(f.readline))
        
    ### extract important data from the output of tokenize package
    toks = pd.DataFrame(columns = ['original','type','text', 'line','pos'])

    last_line = 0
    last_pos = 0

    for token in tokens:
        
        tok_org = token.string
        tok_text = token.string    
        tok_type = str(token).split('(')[2].split(')')[0]

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
        
        toks = toks.append({'type':tok_type,
                            'original':tok_org,
                            'text':tok_text,
                            'line':tok_line,
                            'pos':tok_pos},ignore_index=True)


    # remove encoding lines and end of file
    toks.line = toks.line.astype('int')
    toks.pos = toks.pos.astype('int')
    toks = toks.loc[~((toks.type == 'ENCODING') | (toks.type == 'ENDMARKER'))]
    toks['doc'] = (toks.text.str.contains('"""') | toks.text.str.contains("'''"))
    toks = toks.loc[~(toks.doc)].drop(['doc'],axis=1)

    toks.head(20)

    indent = 0
    last_line = 0

    for index,row in toks.iterrows():
        if row.type == "INDENT":
            indent +=1
            continue
        if row.type == "DEDENT":
            indent -=1
            continue
        if row.line != last_line:
            last_line = row.line
            toks = toks.append({'type':'\n'+indent*'\t',
                                'text':'\n'+indent*'\t',
                                'line':row.line,
                                'pos':row.pos-1},ignore_index=True)

    toks = toks.loc[~((toks.type=='INDENT') | (toks.type=='DEDENT'))]
    toks = toks.sort_values(['line','pos']).reset_index(drop=True)


    # drop the first row (empty line)
    toks.drop(toks.index[:1], inplace=True)

    toks.head(20)

    with open(file,'r') as f:
        src = f.read()

    stdout_backup = sys.stdout
    sys.stdout = open('dis.txt','w')
    dis.dis(src)
    sys.stdout = stdout_backup

    with open('dis.txt','r') as f:
        lines = f.readlines()

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
        toks.loc[(toks.line==l) & (toks.text==n),'type'] = 'GLOBAL_VARIABLE'

    toks .head(10) 

    text_imports = ' '.join(list(toks.text)).replace('\n ','\n').replace(' \n','\n').replace('\t ','\t').replace(' . ','.').replace(' (','(')
    text_imports = multireplace(text_imports, replacements, ignore_case = True)

    with open('normalized_textual_file.py','w') as f:
        f.write(text_imports)

    toks.type = toks.apply(lambda x: x['text'] if str(x['text']) in ['LIBRARY','LIB','ALIAS','MODULE'] else x['type'], axis = 1)
    code_converted = ' '.join(list(toks.type)).replace('\n ','\n').replace(' \n','\n').replace('\t ','\t').replace(' . ','.').replace(' (','(')

    final_replacements = {'GLOBAL_VARIABLE(':'FUNCTION_CALL(',                      
    #                       'NAME.NAME':'NAME',
                          'NAME(':'FUNCTION_CALL(',
                          'NAME':'LOCAL_VARIABLE'}

    code_converted = multireplace(code_converted, final_replacements, ignore_case = False)

    with open(output_file,'w') as f:
        f.write(code_converted)

def convert_old(file: str, output_file: str, tmp_dir: str):
    """Warning: old version (also known as the "first" improvement), only for reference timing

    Converts python source files into token files

    :param file_path: The location of the source file
    :param output_file: The location where the token file will be written
    :return: returns nothing
    """
    with open (file, "r") as f:
        text = f.read()  

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


    # reomve * from the dictionary (handle from module import * statement)
    replacements.pop('*', None)
    # print('List of modules and libraries to replace:\n', replacements)

    with open(tmp_dir + '/med.py','w') as f:
        f.write(multireplace(text, replacements, ignore_case = True))

    file = tmp_dir + '/med.py'
    with open(file,'rb') as f:
        tokens = list(tokenize.tokenize(f.readline))
        
    ### extract important data from the output of tokenize package

    last_line = 0
    last_pos = 0
    tokss = []
    for token in tokens:
        
        tok_org = token.string
        tok_text = token.string    
        tok_type = str(token).split('(')[2].split(')')[0]

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

    toks.head(20)

    indent = 0
    last_line = 0

    tokss = []

    for index,row in toks.iterrows():
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

    toks.head(20)

    with open(file,'r') as f:
        src = f.read()

    stdout_backup = sys.stdout
    sys.stdout = open(tmp_dir + '/dis.txt','w')
    dis.dis(src)
    sys.stdout = stdout_backup

    with open(tmp_dir + '/dis.txt','r') as f:
        lines = f.readlines()

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
        toks.loc[(toks.line==l) & (toks.text==n),'type'] = 'GLOBAL_VARIABLE'

    toks .head(10) 

    text_imports = ' '.join(list(toks.text)).replace('\n ','\n').replace(' \n','\n').replace('\t ','\t').replace(' . ','.').replace(' (','(')
    text_imports = multireplace(text_imports, replacements, ignore_case = True)

    with open('normalized_textual_file.py','w') as f:
        f.write(text_imports)

    toks.type = toks.apply(lambda x: x['text'] if str(x['text']) in ['LIBRARY','LIB','ALIAS','MODULE'] else x['type'], axis = 1)
    code_converted = ' '.join(list(toks.type)).replace('\n ','\n').replace(' \n','\n').replace('\t ','\t').replace(' . ','.').replace(' (','(')

    final_replacements = {'GLOBAL_VARIABLE(':'FUNCTION_CALL(',                      
    #                       'NAME.NAME':'NAME',
                          'NAME(':'FUNCTION_CALL(',
                          'NAME':'LOCAL_VARIABLE'}

    code_converted = multireplace(code_converted, final_replacements, ignore_case = False)

    with open(output_file,'w') as f:
        f.write(code_converted)

def convert_optional_original(file_path, converted_path) -> ConversionResult:
    """Runs `convert_original`, tracks the amount of time it takes in microseconds, and catches any errors it might throw
    
    :param file_path: The location of the source file
    :param output_file: The location where the token file will be written

    :return: A `ConversionResult` tuple containing information about the conversion
    """
    start_time = datetime.datetime.now()

    tmp_dir = tempfile.mkdtemp()
    cwd = os.getcwd()

    file_path_abs = os.path.abspath(file_path)
    converted_path_abs = os.path.abspath(converted_path)
    os.chdir(tmp_dir)
    error_str = None

    try:
        convert_original(file_path_abs, converted_path_abs)
    except Exception as e:
        exc_info = sys.exc_info()
        error_str = ''.join(traceback.format_exception(*exc_info))

    os.chdir(cwd)
    shutil.rmtree(tmp_dir)

    status = "s" if error_str is None else "f"

    return (file_path, converted_path, get_elapsed_us(start_time), status, error_str)

def convert_optional_old(file_path, converted_path) -> ConversionResult:
    """Runs `convert_old`, tracks the amount of time it takes in microseconds, and catches any errors it might throw
    
    :param file_path: The location of the source file
    :param output_file: The location where the token file will be written

    :return: A `ConversionResult` tuple containing information about the conversion
    """
    start_time = datetime.datetime.now()
    tmp_dir = tempfile.mkdtemp()
    error_str = None

    try:
        convert_old(file_path, converted_path, tmp_dir)
    except Exception as e:
        exc_info = sys.exc_info()
        error_str = ''.join(traceback.format_exception(*exc_info))
    
    shutil.rmtree(tmp_dir)

    status = "s" if error_str is None else "f"
    return (file_path, converted_path, get_elapsed_us(start_time), status, error_str)