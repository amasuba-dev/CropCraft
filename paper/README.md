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

## The research proposal

`proposal.js` is separate and has no figure dependency, so it builds on its own:

```bash
node paper/proposal.js
```

It follows the department's MEng/PhD proposal template: title page with the three
signature blocks, a one-page summary, then the full proposal capped at two A4 for
a master's, then references and contact details. The current draft runs 378 words
in the summary and 1,154 in the body, which leaves a little room in both.

Three fields are deliberately left as `[student number]` and
`[university e-mail address]` placeholders, along with the three telephone lines,
because inventing them would be worse than leaving them visibly blank.

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
