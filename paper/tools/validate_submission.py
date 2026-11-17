#!/usr/bin/env python3
"""Report submission text violations without rewriting any source.

Requires the lightweight parser pylatexenc==2.10. This is a source-text check,
not a replacement for compiling and visually inspecting the final PDF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

try:
    from pylatexenc import latex2text, latexwalker
    from pylatexenc.latex2text import EnvironmentTextSpec, MacroTextSpec
    from pylatexenc.macrospec import MacroSpec
except ImportError as error:
    raise SystemExit('Install the text parser: python3 -m pip install pylatexenc==2.10') from error


BANNED_TERMS = (
    'reviewer', 'revision', 'revised', 'in response', 'as requested', 'we now',
    'new analysis', 'additional experiment', 'previous version', 'earlier analysis',
    'originally', 'updated', 'corrected', 'this version',
)
EXEMPT_ENVIRONMENTS = {'thebibliography', 'comment', 'filecontents', 'filecontents*'}
# These commands have nonrendered identifiers/paths, not prose arguments.
HIDDEN_MACROS = {
    'label': '{', 'ref': '{', 'pageref': '{', 'eqref': '{', 'autoref': '{',
    'cref': '{', 'Cref': '{', 'includegraphics': '*[{', 'input': '{',
    'include': '{', 'subfile': '{', 'bibliography': '{', 'bibliographystyle': '{',
    'addbibresource': '[{', 'printbibliography': '[', 'url': '{', 'path': '{',
    'index': '{', 'glossary': '{',
}
CITATION_MACROS = ('cite', 'citep', 'citet', 'citealp', 'citealt', 'citeauthor',
                   'citeyear', 'citeyearpar', 'parencite', 'textcite', 'autocite', 'nocite')
FILENAME = re.compile(
    r'(?<!\w)[\w./:~+@-]+\.(?:json|npz|npy|py|tex|pdf|png|svg|jpg|jpeg|csv|tsv|txt|md|'
    r'toml|yaml|yml|pt|pth|bin|h5ad|h5|hdf5|fasta|gtf)\b', re.IGNORECASE)
WORD = re.compile(r'[^\W_]+(?:[\-\u2010\u2011\u2019\'][^\W_]+)*', re.UNICODE)


def citation_notes(node, l2tobj):
    """Citation keys are hidden; optional pre/post notes are visible text."""
    return ' '.join(l2tobj.nodelist_to_text(argument.nodelist)
                    for argument in node.nodeargd.argnlist
                    if isinstance(argument, latexwalker.LatexGroupNode)
                    and argument.delimiters == ('[', ']'))


def visible_second_argument(node, l2tobj):
    return l2tobj.node_to_text(node.nodeargd.argnlist[-1])


def block_text(node, l2tobj):
    return '\n' + l2tobj.nodelist_to_text(node.nodelist) + '\n'


def contexts():
    parse = latexwalker.get_default_latex_context_db()
    parse_macros = [MacroSpec(name, argspec) for name, argspec in HIDDEN_MACROS.items()]
    parse_macros += [MacroSpec(name, '*[[{') for name in CITATION_MACROS]
    parse_macros += [MacroSpec('href', '{{'), MacroSpec('hypertarget', '{{'), MacroSpec('texorpdfstring', '{{'),
                     MacroSpec('caption', '*[{'), MacroSpec('sep', '')]
    parse.add_context_category('submission', macros=parse_macros, prepend=True)
    text = latex2text.get_default_latex_context_db()
    macros = [MacroTextSpec(name, discard=True) for name in HIDDEN_MACROS]
    macros += [MacroTextSpec(name, simplify_repl=citation_notes) for name in CITATION_MACROS]
    macros += [MacroTextSpec('href', simplify_repl=visible_second_argument),
               MacroTextSpec('hypertarget', simplify_repl=visible_second_argument),
               MacroTextSpec('texorpdfstring', simplify_repl='%(1)s'),
               MacroTextSpec('caption', simplify_repl='%(3)s'),
               MacroTextSpec('sep', simplify_repl=' | ')]
    text.add_context_category('submission', macros=macros,
                              environments=([EnvironmentTextSpec(name, discard=True)
                                             for name in EXEMPT_ENVIRONMENTS] +
                                            [EnvironmentTextSpec(name, simplify_repl=block_text)
                                             for name in ['abstract', 'keyword', 'keywords']]), prepend=True)
    return parse, text


def parsed_nodes(source):
    parse, _ = contexts()
    return latexwalker.LatexWalker(source, latex_context=parse,
                                   tolerant_parsing=False).get_latex_nodes()[0]


def document_nodes(nodes):
    """Do not scan macro definitions and configuration in the preamble."""
    for node in nodes:
        if isinstance(node, latexwalker.LatexEnvironmentNode) and node.environmentname == 'document':
            return node.nodelist
    return nodes  # Supplement fragments need not have a document environment.


def render_nodes(nodes):
    _, text = contexts()
    return latex2text.LatexNodes2Text(latex_context=text, math_mode='text').nodelist_to_text(nodes)


def render_tex(source):
    return render_nodes(document_nodes(parsed_nodes(source)))


def environments(nodes, name):
    found = []
    for node in nodes:
        if isinstance(node, latexwalker.LatexEnvironmentNode):
            if node.environmentname in EXEMPT_ENVIRONMENTS:
                continue
            if node.environmentname == name:
                found.append(node)
            found.extend(environments(node.nodelist, name))
        elif isinstance(node, latexwalker.LatexGroupNode):
            found.extend(environments(node.nodelist, name))
    return found


def normalized(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


def history_matches(rendered, filename):
    # Filenames remain exempt even when typeset visibly with \texttt.
    prose = normalized(FILENAME.sub(' ', rendered))
    matches = []
    for term in BANNED_TERMS:
        pattern = r'(?<!\w)' + re.escape(term).replace(r'\ ', r'\s+') + r'(?!\w)'
        for match in re.finditer(pattern, prose, re.IGNORECASE):
            matches.append({'file': str(filename), 'term': term,
                            'context': prose[max(0, match.start()-65):match.end()+65]})
    return matches


def highlight_lines(source):
    lines = [line.strip() for line in source.splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    bullets = [re.sub(r'^[-*+]\s+', '', line) for line in lines if re.match(r'^[-*+]\s+', line)]
    lines = bullets or lines
    rendered = []
    for line in lines:
        line = re.sub(r'\[([^]]+)\]\([^)]*\)', r'\1', line)
        # Markdown emphasis markers and code delimiters do not occupy visible space.
        rendered.append(normalized(render_tex(re.sub(r'[*_`]', '', line))))
    return rendered


def validate_text(manuscript, highlights, supplements=(), manuscript_name='manuscript.tex'):
    nodes = document_nodes(parsed_nodes(manuscript))
    abstracts = environments(nodes, 'abstract')
    keyword_blocks = environments(nodes, 'keyword') + environments(nodes, 'keywords')
    abstract_text = normalized(render_nodes(abstracts[0].nodelist)) if len(abstracts) == 1 else ''
    keywords = ([item.strip() for item in render_nodes(keyword_blocks[0].nodelist).split('|')
                 if item.strip()] if len(keyword_blocks) == 1 else [])
    highlight_items = [{'text': line, 'characters': len(line), 'passed': len(line) <= 85}
                       for line in highlight_lines(highlights)]
    violations = []

    def require(condition, rule, detail):
        if not condition:
            violations.append({'rule': rule, 'detail': detail})

    abstract_words = len(WORD.findall(abstract_text))
    require(len(abstracts) == 1, 'abstract_environment_count', f'Found {len(abstracts)}; expected 1.')
    require(0 < abstract_words <= 250, 'abstract_word_limit', f'{abstract_words} words; expected 1–250.')
    require(3 <= len(highlight_items) <= 5, 'highlight_count', f'{len(highlight_items)} highlights; expected 3–5.')
    for index, item in enumerate(highlight_items, 1):
        require(item['passed'], 'highlight_character_limit',
                f'Highlight {index}: {item["characters"]} characters; maximum 85.')
    require(len(keyword_blocks) == 1, 'keyword_environment_count', f'Found {len(keyword_blocks)}; expected 1.')
    require(1 <= len(keywords) <= 7, 'keyword_count', f'{len(keywords)} keywords; expected 1–7.')
    matches = history_matches(render_nodes(nodes), manuscript_name)
    for filename, source in supplements:
        matches.extend(history_matches(render_tex(source), filename))
    require(not matches, 'banned_history', f'{len(matches)} rendered text matches.')
    return {'passed': not violations, 'abstract': {'word_count': abstract_words, 'maximum': 250},
            'highlights': {'count': len(highlight_items), 'items': highlight_items},
            'keywords': {'count': len(keywords), 'items': keywords},
            'history_matches': matches, 'violations': violations,
            'counting': 'Unicode letter/digit words; hyphen/apostrophe compounds count as one. '
                        'Highlight characters include spaces, after basic Markdown/TeX rendering.',
            'limitations': ['Standard TeX syntax is parsed; arbitrary user macro expansion is not performed.',
                            'Text embedded inside figure images/PDFs is not scanned; inspect rendered figures.']}


def included_files(source, parent):
    """Resolve ordinary included TeX sources without treating their names as prose."""
    found = []

    def walk(nodes):
        for node in nodes:
            if isinstance(node, latexwalker.LatexMacroNode) and node.macroname in {'input', 'include', 'subfile'}:
                argument = node.nodeargd.argnlist[-1]
                name = render_nodes(argument.nodelist).strip()
                path = parent / name
                found.append(path if path.suffix else path.with_suffix('.tex'))
            elif isinstance(node, (latexwalker.LatexEnvironmentNode, latexwalker.LatexGroupNode)):
                if not isinstance(node, latexwalker.LatexEnvironmentNode) or node.environmentname not in EXEMPT_ENVIRONMENTS:
                    walk(node.nodelist)
    walk(document_nodes(parsed_nodes(source)))
    return found


def main():
    paper = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript', type=Path, default=paper/'manuscript.tex')
    parser.add_argument('--highlights', type=Path, default=paper/'highlights.md')
    parser.add_argument('--supplement', type=Path, action='append', default=[],
                        help='Additional TeX source; repeatable. supplementary/**/*.tex are also scanned.')
    parser.add_argument('--out', type=Path, help='Optional JSON report; sources are never modified.')
    args = parser.parse_args()
    sources = {}

    def read(path):
        path = path.resolve()
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return raw.decode('utf-8')

    try:
        manuscript = read(args.manuscript)
        highlights = read(args.highlights)
        pending = list(args.supplement) + list((args.manuscript.parent/'supplementary').rglob('*.tex'))
        pending += included_files(manuscript, args.manuscript.parent)
        supplements = []
        seen = {args.manuscript.resolve()}
        while pending:
            path = pending.pop(0).resolve()
            if path in seen:
                continue
            seen.add(path)
            source = read(path)
            supplements.append((str(path), source))
            pending.extend(included_files(source, path.parent))
        report = validate_text(manuscript, highlights, supplements, str(args.manuscript.resolve()))
        report.update({'source_sha256': sources, 'supplement_files': [name for name, _ in supplements]})
    except (OSError, UnicodeError, latexwalker.LatexWalkerParseError, ValueError) as error:
        parser.exit(2, f'Text validation could not complete: {error}\n')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: report[key] for key in ['passed', 'abstract', 'highlights', 'keywords',
                                                  'violations', 'history_matches']}, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
