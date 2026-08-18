"""
Offline self-test for the subword -> gene attribution logic (no HF download needed).

Validates:
  1. build_cell_sentence records char spans that slice back to the gene symbols;
  2. attribute_tokens_to_genes assigns SentencePiece-style subword tokens (leading-space
     tokens, multi-subword genes, special (0,0) tokens, prompt scaffold) to the right gene.
  3. (optional) integration check against a real fast tokenizer if one is cached locally.

Run:  python test_span_attribution.py
"""

import numpy as np

from cell_sentences import SCAFFOLD, attribute_tokens_to_genes, build_cell_sentence


def test_spans_slice_back():
    genes = ["MALAT1", "CD8A", "FTL", "MT-CO1"]
    cs = build_cell_sentence(genes, prompt=False)
    assert cs.text == "MALAT1 CD8A FTL MT-CO1"
    for g, (s, e) in zip(genes, cs.gene_spans):
        assert cs.text[s:e] == g
    print("ok: spans slice back to gene symbols")


def test_subword_attribution_bare():
    genes = ["MALAT1", "CD8A", "FTL"]
    cs = build_cell_sentence(genes, prompt=False)  # "MALAT1 CD8A FTL"
    # simulated fast-tokenizer offsets (BOS, then SentencePiece-style ▁-subwords, EOS)
    offsets = [
        (0, 0),    # <bos>            -> scaffold
        (0, 5),    # "MALAT"          -> MALAT1 (0)
        (5, 6),    # "1"              -> MALAT1 (0)
        (6, 9),    # "▁CD" (w/ space) -> CD8A  (1)
        (9, 11),   # "8A"             -> CD8A  (1)
        (11, 15),  # "▁FTL"           -> FTL   (2)
        (0, 0),    # <eos>            -> scaffold
    ]
    got = attribute_tokens_to_genes(offsets, cs.gene_spans)
    exp = np.array([SCAFFOLD, 0, 0, 1, 1, 2, SCAFFOLD])
    assert np.array_equal(got, exp), f"{got} != {exp}"
    print("ok: bare cell-sentence subword attribution")


def test_prompt_scaffold_maps_to_scaffold():
    genes = ["FOO", "BAR"]
    cs = build_cell_sentence(genes, organism="human", prompt=True)
    g0, g1 = cs.gene_spans
    # a token sitting entirely inside the prompt prefix must be scaffold
    prompt_tok = (0, 5)
    offsets = [(0, 0), prompt_tok, (g0[0], g0[1]), (g1[0], g1[1])]
    got = attribute_tokens_to_genes(offsets, cs.gene_spans)
    assert got[0] == SCAFFOLD and got[1] == SCAFFOLD
    assert got[2] == 0 and got[3] == 1
    print("ok: prompt scaffold tokens map to SCAFFOLD, gene tokens map correctly")


def test_optional_real_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception as e:  # not cached / offline / not installed -> skip, don't fail
        print(f"skip: real-tokenizer integration ({type(e).__name__})")
        return
    if not tok.is_fast:
        print("skip: cached tokenizer is not fast")
        return
    genes = ["MALAT1", "CD8A", "FTL"]
    cs = build_cell_sentence(genes, prompt=False)
    enc = tok(cs.text, return_offsets_mapping=True)
    got = attribute_tokens_to_genes(enc["offset_mapping"], cs.gene_spans)
    # every gene must claim at least one token; no gene index out of range
    claimed = set(int(g) for g in got if g != SCAFFOLD)
    assert claimed == {0, 1, 2}, f"genes claimed: {claimed}"
    print(f"ok: real-tokenizer integration ({tok.name_or_path}); token->gene = {got.tolist()}")


if __name__ == "__main__":
    test_spans_slice_back()
    test_subword_attribution_bare()
    test_prompt_scaffold_maps_to_scaffold()
    test_optional_real_tokenizer()
    print("\nAll span-attribution tests passed.")
