/*
 * Research proposal, in the department's template.
 *
 * The template's structure is followed exactly: title page with signature
 * blocks, a one-page summary, a two-page full proposal for a master's, then
 * references and contact information. Content is a rewrite of the original
 * questions and hypotheses against what the work has since established.
 */

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  BorderStyle, PageBreak, LevelFormat, Footer, PageNumber,
} = require("docx");

const INK = "000000", MUTED = "444444";
const BODY = 22;              // 11pt, as the template stipulates

const p = (text, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120, line: o.line ?? 240 },   // single spacing
  alignment: o.align ?? AlignmentType.JUSTIFIED,
  children: [new TextRun({
    text, size: o.size ?? BODY, bold: o.bold, italics: o.italics,
    color: o.color ?? INK, font: "Times New Roman",
  })],
});

const rich = (parts, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120, line: 240 },
  alignment: o.align ?? AlignmentType.JUSTIFIED,
  children: parts.map(([t, x = {}]) => new TextRun({
    text: t, size: x.size ?? BODY, bold: x.bold, italics: x.italics,
    color: x.color ?? INK, font: "Times New Roman",
  })),
});

const h = (text, level = 1) => new Paragraph({
  heading: level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
  spacing: { before: level === 1 ? 240 : 180, after: 100 },
  children: [new TextRun({
    text, size: level === 1 ? 24 : 22, bold: true, color: INK,
    font: "Times New Roman",
  })],
});

const centre = (text, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120 },
  alignment: AlignmentType.CENTER,
  children: [new TextRun({
    text, size: o.size ?? BODY, bold: o.bold, color: o.color ?? INK,
    font: "Times New Roman",
  })],
});

const bullets = (items) => items.map((t) => new Paragraph({
  numbering: { reference: "b", level: 0 },
  spacing: { after: 90, line: 240 },
  children: [new TextRun({ text: t, size: BODY, font: "Times New Roman" })],
}));

const signature = (role) => [
  new Paragraph({ spacing: { before: 340, after: 0 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: INK, space: 2 } },
    children: [new TextRun({ text: "", size: BODY })] }),
  new Paragraph({
    spacing: { after: 60 },
    children: [new TextRun({ text: "(Signature)", size: 18, font: "Times New Roman" })],
  }),
  new Paragraph({
    spacing: { after: 180 },
    children: [
      new TextRun({ text: role, size: BODY, font: "Times New Roman" }),
      new TextRun({ text: "					Date", size: BODY, font: "Times New Roman" }),
    ],
  }),
];

/* ---------------- title page ---------------- */

const title = [
  new Paragraph({ spacing: { before: 900, after: 400 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({
      text: "AUTOMATED BIOMASS ESTIMATION USING SELF-SUPERVISED VISION TRANSFORMERS",
      size: 30, bold: true, font: "Times New Roman" })] }),
  centre("Research proposal: MEng", { after: 500, size: 24 }),
  centre("Candidate", { bold: true, after: 60 }),
  centre("A Masuba", { after: 240 }),
  centre("Student number", { bold: true, after: 60 }),
  centre("[student number]", { after: 240 }),
  centre("Department of Electrical, Electronic and Computer Engineering", { after: 60 }),
  centre("University of Pretoria", { after: 240 }),
  centre("25 August 2026", { after: 240 }),
  centre("Supervisor", { bold: true, after: 60 }),
  centre("Prof. H Myburgh", { after: 240 }),
  centre("Co-supervisors", { bold: true, after: 60 }),
  centre("Prof. A De Freitas", { after: 40 }),
  centre("Dr K Mokise", { after: 200 }),
  ...signature("Candidate"),
  ...signature("Supervisor/Promoter"),
  ...signature("Postgraduate Committee"),
  new Paragraph({ children: [new PageBreak()] }),
];

/* ---------------- summary page (one A4) ---------------- */

const summary = [
  h("Summary Page"),
  h("1. Research questions", 2),
  ...bullets([
    "RQ1. Can self-supervised vision transformers learn plant representations that capture geometric structure and physiological traits from minimal labelled data?",
    "RQ2. What transformer-based grounding and fusion strategies enable accurate 3D reconstruction of plant architecture from sparse multi-view RGB-D capture?",
    "RQ3. Can the reconstructed 3D representation predict above-ground biomass label-efficiently, and how does reconstruction quality relate to estimation accuracy?",
    "RQ4. Which reconstruction operator is appropriate for thin, self-occluding plant structure, and by what criterion should that be decided when no ground-truth geometry exists?",
  ]),
  h("2. Approach", 2),
  p("Thirty-eight potted Eucalyptus and Mango specimens were captured with two Kinect v2 units carried through six positions, giving twelve registered RGB-D views each, and destructively harvested for fresh above-ground mass; thirty-six carry a complete set of views and are usable. No calibration target was recorded, so camera poses are estimated from the depth data. Reconstruction operators are compared under one protocol with features, grid, views and masks held fixed, and every difference between methods is reported with a paired bootstrap interval. Because no ground-truth geometry exists, reconstruction validity is assessed by a physical criterion: measured mass divided by reconstructed volume must fall within the bulk density of fresh plant tissue."),
  h("3. Contribution", 2),
  p("A physical plausibility diagnostic for multi-view plant reconstruction that requires no reference geometry, and the finding it produces: space carving recovers the canopy envelope rather than the plant for these morphologies, which is a property of the operator rather than of resolution. Replacing silhouette intersection with depth fusion, all else held fixed, produces the first statistically resolved improvement in biomass accuracy in this work. The diagnostic also shows that conventional reconstruction metrics rank the worse reconstruction higher on this data, which is a methodological caution of wider relevance to phenotyping."),
  h("4. Anticipated peer-reviewed journal article", 2),
  rich([["Target journal: ", { bold: true }],
        ["Computers and Electronics in Agriculture (ISI, IF ~8.3)"]], { after: 80 }),
  rich([["Preliminary title: ", { bold: true }],
        ["Silhouette carving recovers the canopy envelope, not the plant: a physical criterion for validating multi-view reconstruction in biomass phenotyping"]], { after: 80 }),
  rich([["Description: ", { bold: true }],
        ["Reconstruct-then-regress pipelines for non-destructive biomass estimation almost universally adopt silhouette-based reconstruction without testing whether the resulting volume can represent the plant at all. The gap this addresses is the absence of any validation criterion for multi-view plant reconstruction when ground-truth geometry is unavailable, which is the normal case in destructive-harvest studies."]]),
  new Paragraph({ children: [new PageBreak()] }),
];

/* ---------------- full proposal (two A4) ---------------- */

const full = [
  h("Full research proposal"),

  h("1. Introduction", 2),
  p("Above-ground biomass is among the most requested traits in plant phenotyping and destructive harvest remains its reference method. Non-destructive estimation from imaging follows a near-universal pattern: reconstruct the plant in three dimensions, summarise the reconstruction with shape descriptors, and regress those descriptors on weighed mass [1]. The reconstruction step is usually treated as solved. Given silhouettes from a ring of viewpoints the natural operator is space carving, whose output Laurentini showed is the visual hull, the maximal object consistent with those silhouettes [2]. For a compact object this is a mild approximation. For a plant, whose above-ground structure is largely the space between leaves, whether it is an approximation at all is an open question that this work tests rather than assumes."),

  h("2. Overview of current literature", 2),
  p("Reviews of high-throughput phenotyping organise the field by trait, with plant height the most studied geometric quantity and biomass estimated from crop height and surface models, vegetation indices, or their combination, reporting coefficients of determination between 0.55 and 0.79 for field crops [1]. Acquisition frequency has been studied directly in a livestock imaging context, with the finding that the optimal frequency is trait-specific rather than universal [3], a question that has an obvious analogue in the number of viewpoints a plant capture needs. Agronomic work on plant density shows it affects stem biometry more than biomass yield [4], and is a reminder that the agronomic sense of density, plants per unit area, is distinct from the volumetric sense used as a diagnostic here."),
  p("On the reconstruction side, silhouette methods are bounded above by the visual hull [2], while volumetric range-image integration provides an alternative that accumulates signed distances from depth measurements into a truncated field whose zero crossing is the surface [5], demonstrated in real time with commodity depth sensors [6]. The distinction is evidential: a silhouette constrains the subject to lie somewhere along a ray, whereas a depth sample asserts a surface at a specific distance, and only the second can represent a concavity. Self-supervised vision transformers now supply general-purpose dense features without labels [7], promptable models supply subject masks without per-species training [8], and feed-forward pointmap models estimate cameras and geometry from images alone [9], [10]."),

  h("3. Motivation", 2),
  p("Two observations motivate the work. First, biomass phenotyping is label-poor: every training example costs a destroyed plant, so methods that reduce the labelled requirement have direct practical value, which is the case for self-supervised representations. Second, and less obviously, the field validates reconstruction by proxy. Where ground-truth geometry is unavailable, quality is judged by consistency between views, which for a hull is guaranteed by construction rather than earned. A pipeline can therefore pass every geometric check while reconstructing an object that could not physically weigh what the plant weighs, and nothing in the standard protocol would detect it."),

  h("4. Objective", 2),
  ...bullets([
    "Establish a validation criterion for multi-view plant reconstruction that requires no reference geometry, and characterise the reconstruction quality achievable from twelve-view RGB-D capture under it.",
    "Determine whether geometry-grounded self-supervised transformer representations improve biomass estimation over hand-crafted descriptors on the same reconstructions, at a sample size where differences must be reported with intervals.",
    "Determine which reconstruction operator is appropriate for thin, self-occluding structure, by comparing operators with every other factor held fixed.",
    "Quantify the sensitivity of both reconstruction and estimation to angular sampling density, and establish the minimum viewpoint count for which reconstruction remains physically valid.",
  ]),

  h("5. Proposed research", 2),
  p("Capture uses two Kinect v2 units carried through six positions thirty degrees apart, giving twelve azimuths per specimen, with colour mapped into the depth frame. Specimens are destructively harvested and the above-ground shoot weighed fresh; pot mass is weighed after shoot removal. Camera extrinsics are estimated from the depth data by fitting a floor plane per view and recovering the subject axis by cross-view agreement, since no calibration target was captured."),
  p("Reconstruction operators are compared with features, protocol, voxel grid, views and masks all held fixed, so that the only variable is the operator. Validity is assessed by implied bulk density, computed as weighed shoot mass divided by reconstructed above-ground volume and compared against the 300 to 900 kilograms per cubic metre of fresh plant tissue. Reconstruction quality is additionally characterised by re-projection into the captured views, giving silhouette intersection-over-union, depth error and depth peak signal-to-noise ratio, and by cross-operator agreement using Chamfer distance, the ninety-fifth-percentile Hausdorff distance, F-score at a fixed metric tolerance, and voxel intersection-over-union."),
  p("Biomass is estimated by leave-one-out cross-validation over all usable specimens, with root-mean-square error and the coefficient of determination as the reported metrics and a paired bootstrap over twenty thousand resamples on every difference between methods. Differences whose interval spans zero are reported as unresolved rather than as results. Representation learning is evaluated in two forms: frozen self-supervised features under a linear probe, and a geometry-grounded transformer trained with occupancy supervision derived from the reconstruction, which requires no manual annotation."),
  p("Angular sampling is studied by rebuilding reconstructions from evenly spaced subsets of three, four, six and twelve views and scoring each under the same validity criterion. Camera pose estimation, the least verified assumption in the pipeline, is checked independently by feed-forward pointmap models that estimate cameras from images alone and therefore share no failure mode with poses estimated from the same depth being reconstructed [9], [10]."),

  h("6. Contribution", 2),
  p("The primary contribution is a validation criterion for multi-view plant reconstruction that requires no reference geometry, together with the finding it yields on this data: under space carving only eight of thirty-six specimens produce a reconstruction whose implied bulk density falls inside a deliberately generous band, and all ten Mango specimens land between 26 and 77 kilograms per cubic metre, an order of magnitude below fresh plant tissue. Because the visual hull is the maximal solid consistent with the silhouettes, this is a property of the operator rather than of resolution, and it bounds what any silhouette-based method can achieve on these morphologies."),
  p("The second contribution is the controlled substitution that follows from it. Replacing silhouette intersection with truncated signed distance fusion of the same depth maps, with all other factors fixed, raises the count to twenty-five at the identical grid and moves biomass root-mean-square error from 0.544 to 0.335 kilograms, a paired-bootstrap difference of minus 0.209 with a ninety-five per cent interval of minus 0.363 to minus 0.066. An image-only control is unchanged, which establishes that the reconstruction rather than the regressor was the limiting component."),
  p("A third contribution is methodological and of wider relevance. Silhouette intersection-over-union, the conventional reconstruction quality measure, ranks the carve above the fusion on this data while physical plausibility and biomass accuracy both rank it below. A metric that measures agreement with the input rather than fidelity to the object will systematically favour whichever method is most consistent with its own evidence, and for a hull that consistency is guaranteed rather than earned. This is a caution the phenotyping literature does not currently carry."),
  new Paragraph({ children: [new PageBreak()] }),
];

/* ---------------- references and contact ---------------- */

const refs = [
  h("References"),
  ...[
    "[1] L. Feng, S. Chen, C. Zhang, Y. Zhang, and Y. He, “A comprehensive review on recent applications of unmanned aerial vehicle remote sensing with various sensors for high-throughput plant phenotyping,” Computers and Electronics in Agriculture, vol. 182, 106033, 2021.",
    "[2] A. Laurentini, “The visual hull concept for silhouette-based image understanding,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 16, no. 2, pp. 150–162, 1994.",
    "[3] T. Bresolin et al., “Assessing optimal frequency for image acquisition in computer vision systems developed to monitor dairy cattle,” J. Dairy Sci., vol. 106, pp. 664–675, 2023.",
    "[4] S. Amaducci et al., “Key cultivation techniques for hemp in Europe and China,” Industrial Crops and Products, vol. 68, pp. 2–16, 2015.",
    "[5] B. Curless and M. Levoy, “A volumetric method for building complex models from range images,” in Proc. SIGGRAPH, 1996, pp. 303–312.",
    "[6] R. A. Newcombe et al., “KinectFusion: real-time dense surface mapping and tracking,” in Proc. IEEE ISMAR, 2011, pp. 127–136.",
    "[7] M. Oquab et al., “DINOv2: learning robust visual features without supervision,” Trans. Machine Learning Research, 2024.",
    "[8] A. Kirillov et al., “Segment Anything,” in Proc. IEEE/CVF ICCV, 2023.",
    "[9] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “DUSt3R: geometric 3D vision made easy,” in Proc. IEEE/CVF CVPR, 2024.",
    "[10] V. Leroy, Y. Cabon, and J. Revaud, “Grounding image matching in 3D with MASt3R,” in Proc. ECCV, 2024.",
  ].map((t) => p(t, { align: AlignmentType.LEFT, after: 80 })),
  new Paragraph({
    spacing: { before: 200, after: 200 },
    border: { top: { style: BorderStyle.SINGLE, size: 6, color: "BBBBBB", space: 8 } },
    children: [new TextRun({
      text: "Note: bibliographic details should be verified against publisher records before submission. References [1] and [4] were read in full during this work; the remainder were compiled from working notes.",
      size: 18, italics: true, color: MUTED, font: "Times New Roman" })],
  }),

  h("Contact Information"),
  ...[
    ["Postal address", "Department of Electrical, Electronic and Computer Engineering, University of Pretoria, Private Bag X20, Hatfield, 0028, South Africa"],
    ["Research group", "Smart Sensing and Intelligent Systems Group"],
    ["E-mail", "[university e-mail address]"],
    ["Tel number", ""],
    ["Fax number", ""],
    ["Cell number", ""],
  ].map(([k, v]) => rich([[k + ": ", { bold: true }], [v]], { align: AlignmentType.LEFT, after: 80 })),
];

const doc = new Document({
  creator: "A Masuba",
  title: "Research proposal: MEng",
  numbering: {
    config: [{
      reference: "b",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 500, hanging: 260 } } },
      }],
    }],
  },
  styles: { default: { document: { run: { font: "Times New Roman", size: BODY } } } },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }),
    },
    children: [...title, ...summary, ...full, ...refs],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "Masuba_research_proposal.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, (buf.length / 1024).toFixed(0), "KB");
});
