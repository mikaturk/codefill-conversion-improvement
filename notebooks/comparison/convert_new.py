def convert_new(file, output_file):
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
    print('List of modules and libraries to replace:\n', replacements)

    with open('med.py','w') as f:
        f.write(multireplace(text, replacements, ignore_case = True))

    file = 'med.py'
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
                print('found a global!')
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