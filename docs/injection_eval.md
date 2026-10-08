# Prompt-injection defences: what was measured

Three hand-written rounds of 40 injection sentences (English and German, including obfuscated spelling) and 40 ordinary invoice
sentences each, in `test_invoices/injection_corpus.py`. Round A was written first and the regex scanner was built against it; round B
was written next and the scanner was tuned on it too; **round C was written last and never used for tuning**. The sentences are mine,
not a public benchmark. Reproduce: `python test_invoices/injection_eval.py` (regex) and `python test_invoices/injection_llm_judge.py`
(local Qwen2.5-1.5B-Instruct, zero-shot, one yes/no call per sentence).

| Round | Regex scanner: recall / false positives | Small LLM judge: recall / false positives |
|---|---|---|
| A (scanner built on it) | 40/40 / 0/40 | 32/40 (80%) / 4/40 |
| B (scanner tuned on it) | 40/40 / 0/40 | 22/40 (55%) / 4/40 |
| **C (held out)** | **8/40 (20%)** / 1/40 | **21/40 (52%)** / 2/40 |

**The regex scanner does not generalise.** Its first version caught 12 of 40 on round A, which is what an untuned list achieves;
after tuning it reached 100% on A and B and still catches only 20% of phrasings it has not seen. A pattern list is a tripwire for
the obvious attacks, not a defence. The small LLM judge generalises better (about half) at a cost of one more model call per
sentence and a few false alarms, and has not been wired into the pipeline.

## What actually protects the pipeline

Detection is the weak layer, so the controls that do not depend on recognising the attack carry the weight, and each has a test:

1. **The model has no tools and cannot approve anything.** Auto-approval needs arithmetic, PO, vendor and line-item checks to pass
   in code. A model that reports confidence 1.0 for a mismatched invoice is still routed to a human.
2. **Output grounding** (`security.ungrounded_fields`): every extracted identifier, vendor name and amount must appear in the
   document text (numbers are read in US and European formats). A model that was talked into reporting what the PO expects, or into
   "correcting" a total, produces values the document does not contain, and the invoice goes to review with `UNGROUNDED_FIELD`.
   The test pair shows it is this check that stops a consistent forgery: with it disabled, the same forged extraction is
   auto-approved, because PO matching alone cannot tell the values were invented.
3. **MCP tool allowlist, argument shape checks, result-type check, audit log** (see the README).

## Limits

* Grounding stops invented values, not a document whose printed values are themselves false (an invoice that really does say
  the wrong price is the normal PO-matching case, and that check is unchanged). It is untested against a live model, which may
  legitimately reformat a name or number and trigger a false `UNGROUNDED_FIELD`; that sends the invoice to a human, it does not lose it.
* Dates, currency, tax rate and descriptions are not grounded because models legitimately reformat them.
* The LLM-judge numbers use one 1.5B model and one prompt; a larger model would likely do better, unmeasured.
