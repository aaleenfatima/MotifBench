import numpy as np
import pytest

from evaluation.metrics import profile_from_sites, revcomp_str, revcomp_profile
from evaluation.streme import parse_meme, best_site, streme_to_found, write_meme
from evaluation.evaluate import evaluate, RESULT_COLUMNS

# Shaped like a real streme.txt, including the header noise STREME writes
STREME_TXT = """\
********************************************************************************
STREME - Sensitive, Thorough, Rapid, Enriched Motif Elicitation
********************************************************************************
MEME version 5.5.4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.300 C 0.200 G 0.200 T 0.300

MOTIF 1-GATTACA STREME-1
letter-probability matrix: alength= 4 w= 7 nsites= 48 S= 1.2e-010
 0.000000 0.000000 1.000000 0.000000
 1.000000 0.000000 0.000000 0.000000
 0.000000 0.000000 0.000000 1.000000
 0.000000 0.000000 0.000000 1.000000
 0.900000 0.100000 0.000000 0.000000
 0.000000 1.000000 0.000000 0.000000
 1.000000 0.000000 0.000000 0.000000

MOTIF 2-CCGG STREME-2
letter-probability matrix: alength= 4 w= 4 nsites= 12 S= 3.4e-002
 0.000000 1.000000 0.000000 0.000000
 0.000000 1.000000 0.000000 0.000000
 0.000000 0.000000 1.000000 0.000000
 0.000000 0.000000 1.000000 0.000000

"""


@pytest.fixture
def streme_file(tmp_path):
    p = tmp_path / "streme.txt"
    p.write_text(STREME_TXT)
    return p


def test_parse_meme_reads_all_motifs(streme_file):
    motifs, bg = parse_meme(streme_file)
    assert [m["id"] for m in motifs] == ["1-GATTACA", "2-CCGG"]
    assert motifs[0]["alt_id"] == "STREME-1"
    assert motifs[0]["width"] == 7 and motifs[0]["nsites"] == 48
    assert motifs[0]["score"] == pytest.approx(1.2e-10)
    assert np.allclose(bg, [0.3, 0.2, 0.2, 0.3])


def test_parse_meme_profile_is_4_by_w_in_acgt_order(streme_file):
    P = parse_meme(streme_file)[0][0]["profile"]
    assert P.shape == (4, 7)
    assert np.allclose(P.sum(axis=0), 1)
    assert np.allclose(P[:, 4], [0.9, 0.1, 0, 0])   # 5th position: A 0.9, C 0.1


def test_write_then_parse_roundtrip(tmp_path):
    P = profile_from_sites(["ACGTTA", "ACGATA"], pseudocount=0.5)
    path = tmp_path / "x" / "motifs.meme"
    write_meme(path, [P, revcomp_profile(P)])
    motifs, _ = parse_meme(path)
    assert len(motifs) == 2
    assert np.allclose(motifs[0]["profile"], P, atol=1e-6)


def test_best_site_forward_strand():
    P = profile_from_sites(["GATTACA"])
    s, strand, _ = best_site("CCCCCGATTACACCCCC", P)
    assert (s, strand) == (5, "+")


def test_best_site_reverse_strand():
    P = profile_from_sites(["GATTACA"])
    seq = "CCCCC" + revcomp_str("GATTACA") + "CCCCC"   # TGTAATC planted
    s, strand, _ = best_site(seq, P)
    assert (s, strand) == (5, "-")


def test_streme_to_found_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    seqs, pos = [], []
    for i in range(20):                       # plant GATTACA, half reversed
        bgseq = "".join(rng.choice(list("ACGT"), 100))
        p = int(rng.integers(0, 93))
        site = "GATTACA" if i % 2 else revcomp_str("GATTACA")
        seqs.append(bgseq[:p] + site + bgseq[p + 7:])
        pos.append(p)
    path = tmp_path / "streme.txt"
    path.write_text(STREME_TXT)
    found = streme_to_found(path, seqs)
    assert found["positions"] == pos
    assert found["consensus"] == "GATTACA" and found["n_motifs"] == 2


def test_streme_with_no_motifs_gives_none_and_failed_row(tmp_path):
    path = tmp_path / "streme.txt"
    path.write_text("MEME version 5.5.4\n\nALPHABET= ACGT\n")
    assert streme_to_found(path, ["ACGTACGT"]) is None

    meta = dict(name="toy", motif_id="TOY", strength_fraction=0.2, true_ic=2.0,
                background="uniform", replicate=0, width=7, positions=[0])
    row = evaluate(None, meta, algorithm="streme")
    assert list(row) == RESULT_COLUMNS
    assert row["overlap_rate"] == 0.0 and row["pearson_true"] is None
