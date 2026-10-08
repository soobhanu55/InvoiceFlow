"""Measures the injection scanner (agent/security.py::scan) on the hand-written corpus.

    python test_invoices/injection_eval.py

Round C is the only one the scanner was not tuned on; read that line as its real recall on unseen phrasings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import injection_corpus as c  # noqa: E402
from agent import security  # noqa: E402

for name, attacks, benign, tuned in (("A", c.ATTACKS_A, c.BENIGN_A, True), ("B", c.ATTACKS_B, c.BENIGN_B, True), ("C", c.ATTACKS_C, c.BENIGN_C, False)):
    hit = sum(bool(security.scan(a)) for a in attacks)
    fp = sum(bool(security.scan(b)) for b in benign)
    print(f"round {name} ({'tuned on' if tuned else 'HELD OUT'}): recall {hit}/{len(attacks)} = {hit / len(attacks):.0%}, "
          f"false positives {fp}/{len(benign)}")
