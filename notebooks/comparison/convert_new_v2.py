def convert_new_v2(file, output_file):
    if debug_filenames: print("starting "+output_file, file=save_stdout)
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

    # toks.head(20)

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

    # adjust_lines = [True] * toks.shape[0]
    for l,n in glbls:
        # toks.loc[(toks.line==l) & (toks.text==n),'type'] = 'GLOBAL_VARIABLE'
        line_eq = toks.loc[toks.line==l]
        line_eq.loc[line_eq.text==n, 'type'] = 'GLOBAL_VARIABLE'
        
    toks.loc[toks['text'].isin(['LIBRARY','LIB','ALIAS','MODULE']), 'type'] = toks['text']

    code_converted = ' '.join(list(toks.type)).replace('\n ','\n').replace(' \n','\n').replace('\t ','\t').replace(' . ','.').replace(' (','(')

    final_replacements = {'GLOBAL_VARIABLE(':'FUNCTION_CALL(',                      
    #                       'NAME.NAME':'NAME',
                          'NAME(':'FUNCTION_CALL(',
                          'NAME':'LOCAL_VARIABLE'}

    code_converted = multireplace(code_converted, final_replacements, ignore_case = False)

    with open(output_file,'w') as f:
        f.write(code_converted)
    if debug_filenames: print("finished "+output_file, file=save_stdout)