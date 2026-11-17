#!/usr/bin/env python3
"""Derive title_abstract_keywords.txt from manuscript.tex; never edit submission text by hand.

Uses the existing pylatexenc submission parser. After manuscript changes, default mode
refuses stale existing output; use --refresh to regenerate intentionally. --check verifies
both source provenance and exact generated text without writing any file.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from validate_submission import (parsed_nodes, document_nodes, render_nodes, environments,
                                 normalized, latexwalker, WORD)


def title_arguments(nodes):
    found=[]
    for node in nodes:
        if isinstance(node,latexwalker.LatexMacroNode) and node.macroname=='title':
            arguments=[argument for argument in node.nodeargd.argnlist
                       if isinstance(argument,latexwalker.LatexGroupNode) and argument.delimiters==('{','}')]
            if len(arguments)!=1:raise ValueError('Title requires one braced text argument')
            found.append(arguments[0].nodelist)
        elif isinstance(node,(latexwalker.LatexEnvironmentNode,latexwalker.LatexGroupNode)):
            found.extend(title_arguments(node.nodelist))
    return found


def extract_frontmatter(source):
    nodes=document_nodes(parsed_nodes(source))
    titles=title_arguments(nodes)
    abstracts=environments(nodes,'abstract')
    keyword_blocks=environments(nodes,'keyword')+environments(nodes,'keywords')
    if (len(titles),len(abstracts),len(keyword_blocks))!=(1,1,1):
        raise ValueError('Exactly one title, abstract and keyword environment is required')
    title=normalized(render_nodes(titles[0]))
    abstract=normalized(render_nodes(abstracts[0].nodelist))
    # Split on parsed sep macros, not rendered punctuation that a keyword might contain.
    groups=[[]]
    for node in keyword_blocks[0].nodelist:
        if isinstance(node,latexwalker.LatexMacroNode) and node.macroname=='sep':groups.append([])
        else:groups[-1].append(node)
    keywords=[normalized(render_nodes(group)) for group in groups]
    if not title or not abstract:raise ValueError('Title and abstract must be nonempty')
    if len(keywords)!=6 or any(not keyword for keyword in keywords) or len({keyword.casefold() for keyword in keywords})!=6:
        raise ValueError('Exactly six distinct nonempty keywords are required')
    return {'title':title,'abstract':abstract,'keywords':keywords}


def format_text(fields):
    return f"Title\n{fields['title']}\n\nAbstract\n{fields['abstract']}\n\nKeywords\n{'; '.join(fields['keywords'])}\n"


def generate(manuscript,output,refresh=False,check=False):
    manuscript,output=Path(manuscript),Path(output)
    raw=manuscript.read_bytes()
    fields=extract_frontmatter(raw.decode('utf-8'))
    expected=format_text(fields)
    sidecar=output.with_suffix(output.suffix+'.source.json')
    record={'schema':'c2s-submission-frontmatter-v1','source':str(manuscript.resolve()),
            'source_sha256':hashlib.sha256(raw).hexdigest(),
            'generated_text_sha256':hashlib.sha256(expected.encode('utf-8')).hexdigest(),
            'abstract_word_count':len(WORD.findall(fields['abstract'])),'keyword_count':len(fields['keywords']),
            'generator':'experiments/make_title_abstract_keywords.py',
            'parser':'Existing validate_submission.py pylatexenc contexts; math_mode=text'}
    stale=[]
    if output.exists() or check:
        if not output.is_file():stale.append('generated text is missing')
        elif output.read_text()!=expected:stale.append('generated text differs from the current manuscript')
        if not sidecar.is_file():stale.append('source provenance sidecar is missing')
        else:
            try:previous=json.loads(sidecar.read_text())
            except json.JSONDecodeError:previous={}
            if previous.get('source_sha256')!=record['source_sha256']:stale.append('source checksum changed')
            if previous.get('generated_text_sha256')!=record['generated_text_sha256']:stale.append('generated-text checksum changed')
        if stale and (check or not refresh):
            raise ValueError('Stale submission text: '+ '; '.join(stale)+'. Run with --refresh after final source edits.')
    if not check:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(expected)
        sidecar.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    return {'passed':True,'mode':'check' if check else 'generate','output':str(output.resolve()),**record}


def main():
    paper=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript',type=Path,default=paper/'manuscript.tex')
    parser.add_argument('--out',type=Path,default=paper/'title_abstract_keywords.txt')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    try:report=generate(args.manuscript,args.out,args.refresh,args.check)
    except (OSError,UnicodeError,ValueError,latexwalker.LatexWalkerParseError) as error:parser.exit(1,str(error)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
