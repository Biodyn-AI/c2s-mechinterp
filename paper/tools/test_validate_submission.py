"""Submission rules tested against rendered prose, including TeX hiding places."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from validate_submission import render_tex, validate_text


def manuscript(abstract='A valid abstract.', body='Clean scientific prose.', keywords=None):
    keywords = keywords or ['models', 'cells']
    return (r'\begin{document}\begin{abstract}' + abstract +
            r'\end{abstract}\begin{keyword}' + r'\sep '.join(keywords) +
            r'\end{keyword}' + body + '\n' + r'\end{document}')


HIGHLIGHTS = '\n'.join('- ' + line for line in ['First finding', 'Second finding', 'Third finding'])


class SubmissionTextTests(unittest.TestCase):
    def test_abstract_limit_counts_rendered_hyphenated_words(self):
        abstract = r'\textbf{single-cell} ' + ' '.join(['measurement'] * 249)
        record = validate_text(manuscript(abstract), HIGHLIGHTS)
        self.assertEqual(record['abstract']['word_count'], 250)
        self.assertTrue(record['passed'])
        record = validate_text(manuscript(abstract + ' extra'), HIGHLIGHTS)
        self.assertEqual(record['abstract']['word_count'], 251)
        self.assertIn('abstract_word_limit', [v['rule'] for v in record['violations']])

    def test_highlights_count_and_rendered_character_boundaries(self):
        lines = ['**' + 'a' * 85 + '**', 'Second finding', 'Third finding']
        good = '\n'.join('- ' + line for line in lines)
        self.assertTrue(validate_text(manuscript(), good)['passed'])
        bad = good.replace('a' * 85, 'a' * 86)
        self.assertIn('highlight_character_limit', [v['rule'] for v in validate_text(manuscript(), bad)['violations']])
        for count in [2, 6]:
            report = validate_text(manuscript(), '\n'.join('- Finding' for _ in range(count)))
            self.assertIn('highlight_count', [v['rule'] for v in report['violations']])

    def test_keywords_bounds_and_missing_abstract_are_not_silent(self):
        for count in [1, 7]:
            self.assertTrue(validate_text(manuscript(keywords=['cell'] * count), HIGHLIGHTS)['passed'])
        report = validate_text(manuscript(keywords=['cell'] * 8), HIGHLIGHTS)
        self.assertIn('keyword_count', [v['rule'] for v in report['violations']])
        report = validate_text(r'\begin{document}No abstract.\end{document}', HIGHLIGHTS)
        self.assertIn('abstract_environment_count', [v['rule'] for v in report['violations']])

    def test_bibliography_comments_labels_citation_keys_and_filenames_exempt(self):
        body = (r'\label{sec:revised} \citep{revision2026,reviewer2025}' + '\n' +
                '% revised in response to reviewer\n' +
                r'\includegraphics{figures/revised.pdf} \input{revision_appendix.tex}' +
                r'\texttt{corrected-results.json} \url{https://example.org/revision}' +
                r'\begin{thebibliography}{99}\bibitem{revised} Updated revision.\end{thebibliography}')
        self.assertEqual(validate_text(manuscript(body=body), HIGHLIGHTS)['history_matches'], [])

    def test_caption_table_and_visible_hyperlink_text_are_scanned(self):
        body = (r'\caption{A \textbf{revised} estimate.}'
                r'\begin{tabular}{ll}A & updated\\\end{tabular}'
                r'\href{https://example.org/clean}{New analysis}')
        matches = validate_text(manuscript(body=body), HIGHLIGHTS)['history_matches']
        self.assertEqual({m['term'] for m in matches}, {'revised', 'updated', 'new analysis'})

    def test_escaped_percent_and_styling_do_not_hide_terms(self):
        body = r'95\% re\textbf{vised}; as~\emph{requested}. % reviewer is hidden'
        self.assertEqual({m['term'] for m in validate_text(manuscript(body=body), HIGHLIGHTS)['history_matches']},
                         {'revised', 'as requested'})
        self.assertIn('95%', render_tex(body))

    def test_citation_visible_notes_scanned_but_ids_exempt(self):
        body = r'\citep[reviewer note][p.~4]{revised_2026}'
        self.assertEqual([m['term'] for m in validate_text(manuscript(body=body), HIGHLIGHTS)['history_matches']],
                         ['reviewer'])

    def test_supplement_history_is_reported_with_its_source(self):
        report = validate_text(manuscript(), HIGHLIGHTS,
                               [('supplement.tex', r'\caption{Additional experiment.}')])
        self.assertFalse(report['passed'])
        self.assertEqual(report['history_matches'][0]['file'], 'supplement.tex')
        self.assertEqual(report['history_matches'][0]['term'], 'additional experiment')

    def test_macro_definitions_and_preamble_are_not_rendered_prose(self):
        source = r'\newcommand{\unused}{revised reviewer}' + manuscript(body=r'\newcommand{\unused}{updated} Clean.')
        self.assertTrue(validate_text(source, HIGHLIGHTS)['passed'])

    def test_cli_checks_included_text_and_fails_for_missing_input(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder/'paper.tex'
            highlights = folder/'highlights.md'
            report_path = folder/'report.json'
            source.write_text(manuscript(body=r'\input{extra}'))
            highlights.write_text(HIGHLIGHTS)
            (folder/'extra.tex').write_text(r'\caption{A revised result.}')
            command = [sys.executable, str(Path(__file__).with_name('validate_submission.py')),
                       '--manuscript', str(source), '--highlights', str(highlights), '--out', str(report_path)]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 1, completed.stderr)
            report = json.loads(report_path.read_text())
            self.assertEqual(report['history_matches'][0]['file'], str((folder/'extra.tex').resolve()))
            self.assertIn(str((folder/'extra.tex').resolve()), report['source_sha256'])
            self.assertEqual(source.read_text(), manuscript(body=r'\input{extra}'))
            (folder/'extra.tex').unlink()
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2)
            self.assertIn('could not complete', completed.stderr)


if __name__ == '__main__':
    unittest.main()
