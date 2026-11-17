"""Source fidelity and stale-output checks for editor submission text."""
from pathlib import Path
import tempfile
import unittest

from make_title_abstract_keywords import extract_frontmatter, generate


SOURCE=r'''\documentclass{elsarticle}
\begin{document}\begin{frontmatter}
\title{A \textit{cell} model with $R^2$ and \{braces\}}
\begin{abstract}
The change is $-0.001$ with $\Delta\cos=0.04$ at 63.8\%.
% This comment must not leak into the submission text.
The ratio is $10^{-5}$ and the name is T\"ubingen.
\end{abstract}
\begin{keyword}one \sep two \sep sparse autoencoders \sep single-cell data \sep model evaluation \sep causal analysis\end{keyword}
\end{frontmatter}\end{document}'''


class SubmissionTextTests(unittest.TestCase):
    def test_parser_retains_title_math_escaped_text_and_exact_six_keywords(self):
        fields=extract_frontmatter(SOURCE)
        self.assertEqual(fields['title'],'A cell model with R^2 and {braces}')
        self.assertIn('-0.001',fields['abstract'])
        self.assertIn('Δcos=0.04',fields['abstract'])
        self.assertIn('63.8%',fields['abstract'])
        self.assertIn('10^-5',fields['abstract'])
        self.assertIn('Tübingen',fields['abstract'])
        self.assertNotIn('comment',fields['abstract'])
        self.assertEqual(fields['keywords'],['one','two','sparse autoencoders','single-cell data','model evaluation','causal analysis'])

    def test_missing_duplicate_or_wrong_count_frontmatter_is_rejected(self):
        for invalid in [SOURCE.replace(r'\title{',r'\other{'),
                        SOURCE.replace(r'\begin{abstract}',r'\begin{abstract}A\end{abstract}\begin{abstract}',1),
                        SOURCE.replace(r'one \sep two',r'one \sep one'),
                        SOURCE.replace(r'one \sep two',r'one')]:
            with self.subTest(source=invalid):
                with self.assertRaises(ValueError):extract_frontmatter(invalid)

    def test_stale_source_and_manual_output_edits_require_explicit_refresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            manuscript,out=base/'paper.tex',base/'title_abstract_keywords.txt'
            manuscript.write_text(SOURCE)
            generate(manuscript,out)
            self.assertTrue(generate(manuscript,out,check=True)['passed'])
            manuscript.write_text(SOURCE.replace('0.04','0.05'))
            with self.assertRaises(ValueError):generate(manuscript,out)
            with self.assertRaises(ValueError):generate(manuscript,out,check=True)
            generate(manuscript,out,refresh=True)
            self.assertIn('0.05',out.read_text())
            out.write_text(out.read_text()+'Manual text\n')
            with self.assertRaises(ValueError):generate(manuscript,out,check=True)
            generate(manuscript,out,refresh=True)
            self.assertNotIn('Manual text',out.read_text())


if __name__=='__main__':unittest.main()
