# Regenerating the paper and the proposal

The draft is generated, not hand-written, for the same reason the architecture
diagrams are: results change, and a document that has to be edited by hand to
keep up eventually stops keeping up. Every number in it comes from an artefact in
`work_dirs/ggssvt`, so regenerating after a run picks the new values up.

## Build it

Node's `docx` package is not vendored here. Install it once into this directory:

```bash
cd paper && npm install docx
```

Then, from the repository root:

```bash
python paper/figures.py
```

```bash
node paper/paper.js
```

The first writes the two result figures that the pipeline does not already emit,
reading the caches directly so the plausibility counts in the figure cannot
disagree with the text. The second assembles the document, pulling the
architecture diagrams from `work_dirs/ggssvt/reports/architecture` and the
contact sheet from the gallery, so run `cli architecture` and `cli gallery`
first if those are stale.

Both documents land next to their generators, in this directory.

## The two research proposals

Both build on their own, with no figure dependency:

```bash
node paper/proposal.js
```

```bash
node paper/proposal_phd.js
```

They share `proposal_common.js`, which holds the title page, the three signature
blocks, the 11-point single-spaced body style and the contact block, so the two
documents cannot drift apart on formatting or on the candidate's details. Those
details come from the approved 2025 proposal rather than being invented.

The department's template covers both degrees and differs in three places, which
is the whole difference between the two scripts:

| | MEng | PhD |
|---|---|---|
| summary page | 1 A4, 11 pt | same |
| anticipated articles | one, labelled `Description` | two, labelled `Research gap that will be addressed` |
| faculty requirement | one submitted | one accepted and one submitted |
| full proposal | 2 A4 | 10 A4 |

Current lengths, against those caps:

| | summary (cap 1 A4) | body |
|---|---|---|
| MEng | 0.83 | 1.98 of 2 A4 |
| PhD | 0.95 | 5.87 of 10 A4 |

Those numbers come from `paper/check_length.py`, which estimates rendered height
from the XML by counting characters against the 6.27 inch measure and adding
paragraph spacing, because this machine has no LibreOffice to render with:

```bash
python paper/check_length.py
```

It exits non-zero when a section is over its cap, so it works in a pre-send
check. It is an estimate and a pessimistic one, since justified text fits
slightly more than it predicts, so treat anything above about 0.95 as needing a
look in Word rather than as settled.

The MEng body has almost no slack, and the PhD summary has little. Any addition
to either needs a matching cut. The PhD body has room that is deliberately
unused: four more pages of padding would read worse than six that earn their
place.

### Why there are two

The Postgraduate Committee approved the 2025 MEng proposal with a reservation on
the record, that four hypotheses and three papers "would indicate a study with
the scope and complexity of a PhD". The MEng document answers that by narrowing
to what the evidence now supports. The PhD document takes the reservation at face
value and proposes the study at the size the committee already judged it to be,
keeping all four hypotheses and restating each with the measurement that decides
it.

## What has to be updated by hand

The prose. `paper.js` holds the text inline, and the tables are literals rather
than being read from the JSON reports. That is deliberate for a draft: a table
whose numbers are wired to disk cannot carry the sentence explaining why a
particular row is not resolved, and at this stage the explanation matters more
than the automation.

The consequence is that **the numbers in `paper.js` can drift from the pipeline**.
Two are worth checking against `cli baselines` and `cli fuse` before every send:

- the operator comparison in Table 3
- the plausibility counts in Table 2

## Before submitting

**Verify every reference.** Only Feng and Amaducci were read directly during
this work, in either document. The rest were written from working notes and their years and
page ranges have not been checked against the publisher record. The document says
so in a note under the reference list; remove that note only once the checking is
actually done.

**Render it and look at it.** This machine has neither LibreOffice nor pandoc, so
the draft was verified structurally rather than visually: XML well-formed, images
resolving, outline correct. That is not the same as having seen it.

**Decide on section 4.5.** The DINO probe and the baseline tables use different
feature preprocessing and are not comparable. The text says so rather than hiding
it, but cutting the section is also a reasonable call.
