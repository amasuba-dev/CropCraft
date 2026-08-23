"""The research dashboard: a self-contained page that walks through the work.

One HTML file, no external requests beyond Google Fonts. The voxel clouds are
embedded compressed and decompressed in the browser, which also drives the slice
view and the segment colouring -- so the same payload serves the 3D viewer, the
cross-sections and the pot/canopy split rather than shipping three copies.
"""

from __future__ import annotations

from pathlib import Path

from ..config import WORK_DIR

HEAD = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reconstructing Plant Biomass</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --ground:#eceff1; --surface:#fff; --surface-2:#f4f7f8; --sunken:#e3e8ea;
  --line:#d3dbdf; --line-strong:#b0bcc2;
  --ink:#0f151a; --ink-2:#465560; --ink-3:#75858f;
  --accent:#0f7d84; --accent-soft:#d9eced; --accent-ink:#0b5f65;
  --near:#e8a52c; --far:#17606e; --canopy:#2f8f6b; --pot:#9a7b4f;
  --ok:#2f7d55; --warn:#b0741f; --bad:#a8443c;
  --shadow:0 1px 2px rgba(15,21,26,.05),0 8px 24px rgba(15,21,26,.06);
  --radius:4px; --measure:66ch;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground:#0b1014; --surface:#141c22; --surface-2:#1a232a; --sunken:#0f171c;
    --line:#26323a; --line-strong:#3b4a54;
    --ink:#e8eef2; --ink-2:#a4b3bd; --ink-3:#6e7f89;
    --accent:#4bc4c9; --accent-soft:#123037; --accent-ink:#7fd8dc;
    --near:#f0b543; --far:#2b8fa0; --canopy:#57b581; --pot:#c0a273;
    --ok:#57b581; --warn:#d99a3e; --bad:#d4726a;
    --shadow:0 1px 2px rgba(0,0,0,.5),0 10px 30px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"] {
  --ground:#0b1014; --surface:#141c22; --surface-2:#1a232a; --sunken:#0f171c;
  --line:#26323a; --line-strong:#3b4a54;
  --ink:#e8eef2; --ink-2:#a4b3bd; --ink-3:#6e7f89;
  --accent:#4bc4c9; --accent-soft:#123037; --accent-ink:#7fd8dc;
  --near:#f0b543; --far:#2b8fa0; --canopy:#57b581; --pot:#c0a273;
  --ok:#57b581; --warn:#d99a3e; --bad:#d4726a;
  --shadow:0 1px 2px rgba(0,0,0,.5),0 10px 30px rgba(0,0,0,.35);
}

*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;font-size:16px;line-height:1.6;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:0 22px}
section{padding:56px 0;border-top:1px solid var(--line)}
section:first-of-type{border-top:0}

.eyebrow{font-family:"IBM Plex Sans Condensed",sans-serif;text-transform:uppercase;
  letter-spacing:.11em;font-size:11.5px;font-weight:700;color:var(--accent)}
h1{font-family:"IBM Plex Sans Condensed",sans-serif;font-weight:700;
  font-size:clamp(34px,5.2vw,54px);line-height:1.05;letter-spacing:-.015em;margin:.25em 0 .3em;
  text-wrap:balance;max-width:20ch}
h2{font-family:"IBM Plex Sans Condensed",sans-serif;font-weight:600;
  font-size:clamp(22px,2.7vw,30px);letter-spacing:-.01em;margin:0 0 .5em;text-wrap:balance}
h3{font-family:"IBM Plex Sans Condensed",sans-serif;font-weight:600;font-size:17px;
  margin:0 0 .4em;letter-spacing:.01em}
p{margin:0 0 1em;max-width:var(--measure)}
.lede{font-size:18.5px;color:var(--ink-2);max-width:60ch}
.muted{color:var(--ink-2)}
.small{font-size:13.5px}
strong{font-weight:600}
code{font-family:"IBM Plex Mono",monospace;font-size:.88em;background:var(--surface-2);
  padding:1px 5px;border-radius:3px;border:1px solid var(--line)}
a{color:var(--accent-ink)}

header.hero{padding:64px 0 44px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:var(--radius);
  overflow:hidden;margin-top:34px}
.stat{background:var(--surface);padding:13px 16px}
.stat b{font-family:"IBM Plex Mono",monospace;font-size:23px;font-weight:500;display:block;
  font-variant-numeric:tabular-nums;line-height:1.2}
.stat span{font-family:"IBM Plex Sans Condensed",sans-serif;text-transform:uppercase;
  letter-spacing:.08em;font-size:10.5px;font-weight:600;color:var(--ink-3)}

/* pipeline: a real sequence, so numbering encodes something */
.pipe{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;margin-top:26px}
.stage{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  padding:14px 15px;position:relative;overflow:hidden}
.stage .n{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--accent);
  font-weight:500;letter-spacing:.05em}
.stage h3{margin:.25em 0 .3em;font-size:14.5px}
.stage p{font-size:13px;color:var(--ink-2);margin:0}

/* explorer */
.explorer{display:grid;grid-template-columns:minmax(0,1fr) 268px;gap:16px;align-items:start}
@media(max-width:940px){.explorer{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  box-shadow:var(--shadow);overflow:hidden}
.panel-head{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;padding:12px 16px;
  border-bottom:1px solid var(--line);background:var(--surface-2)}
.panel-head h3{margin:0;font-family:"IBM Plex Mono",monospace;font-size:16px}
.chip{font-size:11px;padding:2px 8px;border-radius:99px;background:var(--accent-soft);
  color:var(--accent-ink);font-weight:600}
.grow{flex:1 1 auto}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px;align-items:center}
.seg{display:inline-flex;border:1px solid var(--line-strong);border-radius:var(--radius);
  overflow:hidden;background:var(--surface)}
.seg button{font:inherit;font-size:13px;padding:6px 13px;border:0;background:transparent;
  color:var(--ink-2);cursor:pointer;border-right:1px solid var(--line)}
.seg button:last-child{border-right:0}
.seg button[aria-pressed="true"]{background:var(--accent);color:#fff}
.seg button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}

.viewer{position:relative;background:var(--sunken)}
.viewer canvas{display:block;width:100%;height:440px;touch-action:none;cursor:grab}
.viewer canvas:active{cursor:grabbing}
.viewer .hint{position:absolute;left:12px;bottom:10px;font-size:11.5px;color:var(--ink-3);
  font-family:"IBM Plex Sans Condensed",sans-serif;letter-spacing:.04em;text-transform:uppercase}
.legend{position:absolute;right:12px;top:10px;display:flex;flex-direction:column;gap:4px;
  font-size:11px;color:var(--ink-2)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;
  vertical-align:-1px}

.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));
  border-top:1px solid var(--line)}
.fact{padding:10px 15px;border-right:1px solid var(--line)}
.fact:last-child{border-right:0}
.fact span{font-family:"IBM Plex Sans Condensed",sans-serif;text-transform:uppercase;
  letter-spacing:.07em;font-size:10px;font-weight:600;color:var(--ink-3);display:block}
.fact b{font-family:"IBM Plex Mono",monospace;font-size:14.5px;font-weight:500;
  font-variant-numeric:tabular-nums;white-space:nowrap}

.list{max-height:520px;overflow-y:auto}
.list h4{margin:0;padding:10px 15px;font-family:"IBM Plex Sans Condensed",sans-serif;
  font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--ink-3);
  position:sticky;top:0;background:var(--surface-2);border-bottom:1px solid var(--line);z-index:2}
.grp{padding:6px 15px 4px;font-family:"IBM Plex Sans Condensed",sans-serif;font-size:10.5px;
  text-transform:uppercase;letter-spacing:.08em;font-weight:600;color:var(--ink-3);
  background:var(--surface-2);border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.item{display:grid;grid-template-columns:40px 1fr auto;gap:10px;align-items:center;width:100%;
  text-align:left;font:inherit;padding:6px 15px;background:transparent;border:0;
  border-bottom:1px solid var(--line);color:var(--ink);cursor:pointer}
.item:hover{background:var(--surface-2)}
.item[aria-current="true"]{background:var(--accent-soft);box-shadow:inset 3px 0 0 var(--accent)}
.item:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.item canvas{display:block;width:40px;height:40px}
.item .id{font-family:"IBM Plex Mono",monospace;font-size:13px;font-weight:500}
.item .m{font-size:11px;color:var(--ink-3)}
.item .kg{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink-2);
  font-variant-numeric:tabular-nums}

/* slices */
.slices{display:grid;grid-template-columns:repeat(auto-fit,minmax(96px,1fr));gap:10px;margin-top:18px}
.slice{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  overflow:hidden}
.slice canvas{display:block;width:100%;height:auto;background:var(--sunken)}
.slice .cap{padding:5px 8px;font-family:"IBM Plex Mono",monospace;font-size:11px;
  color:var(--ink-2);border-top:1px solid var(--line);font-variant-numeric:tabular-nums}

table{border-collapse:collapse;width:100%;font-size:14px}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:var(--radius);
  background:var(--surface)}
th,td{padding:9px 13px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-family:"IBM Plex Sans Condensed",sans-serif;text-transform:uppercase;
  letter-spacing:.07em;font-size:10.5px;color:var(--ink-3);font-weight:600;
  background:var(--surface-2);position:sticky;top:0}
td.num{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:0}
tr.best td{background:var(--accent-soft)}
tr.best td:first-child{font-weight:600}
.tag{font-size:10.5px;padding:1px 7px;border-radius:99px;font-weight:600;
  font-family:"IBM Plex Sans Condensed",sans-serif;letter-spacing:.04em}
.tag.no{background:color-mix(in srgb,var(--warn) 16%,transparent);color:var(--warn)}
.tag.yes{background:color-mix(in srgb,var(--ok) 16%,transparent);color:var(--ok)}

.callout{border:1px solid var(--line);border-left:3px solid var(--warn);
  background:var(--surface);border-radius:var(--radius);padding:18px 20px;margin:22px 0}
.callout.bad{border-left-color:var(--bad)}
.callout h3{color:var(--ink)}
.callout p:last-child{margin:0}

.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:22px}
.scatter{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  padding:10px}
.scatter canvas{display:block;width:100%;height:auto}

footer{padding:40px 0 60px;color:var(--ink-3);font-size:13px;border-top:1px solid var(--line)}
@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}
</style>
"""

BODY = r"""<div class="wrap">

<header class="hero">
  <div class="eyebrow">GG-SSVT &middot; dual-Kinect single-plant capture</div>
  <h1>Reconstructing plant biomass without destroying the plant</h1>
  <p class="lede">Twelve registered RGB-D views per specimen, carved into a volumetric
  reconstruction, then read for above-ground mass &mdash; with every camera pose
  estimated from the depth data itself, because no calibration was ever captured.</p>
  <div class="stats" id="stats"></div>
</header>

<section>
  <div class="eyebrow">How it works</div>
  <h2>Six stages, in order</h2>
  <p class="muted">Each stage consumes the previous one's output. The numbering is the
  dependency order, not a ranking &mdash; a failure at stage two makes everything after
  it meaningless, which is why the explorer below exists.</p>
  <div class="pipe" id="pipe"></div>
</section>

<section>
  <div class="eyebrow">Explore</div>
  <h2>Every reconstruction</h2>
  <p class="muted">Drag to rotate, scroll to zoom. <strong>Segment</strong> colours the
  voxels by the pot/canopy split that the biomass estimate depends on; <strong>depth</strong>
  cues distance from the camera.</p>

  <div class="toolbar">
    <div class="seg" id="segmenter" role="group" aria-label="Segmenter"></div>
    <div class="seg" id="colour" role="group" aria-label="Colour by">
      <button data-mode="segment" aria-pressed="true">Segment</button>
      <button data-mode="depth" aria-pressed="false">Depth</button>
    </div>
    <span class="grow"></span>
    <span class="small muted" id="cloudinfo"></span>
  </div>

  <div class="explorer">
    <div class="panel">
      <div class="panel-head">
        <h3 id="sid">&mdash;</h3>
        <span class="chip" id="sspecies"></span>
        <span class="grow"></span>
        <span class="small muted" id="ssub"></span>
      </div>
      <div class="viewer">
        <canvas id="stage"></canvas>
        <div class="legend" id="legend"></div>
        <div class="hint">drag &middot; scroll</div>
      </div>
      <div class="facts" id="sfacts"></div>
    </div>
    <nav class="panel list" aria-label="Specimens"><h4>Specimens</h4><div id="list"></div></nav>
  </div>

  <h3 style="margin-top:30px">Horizontal sections</h3>
  <p class="muted small">The volume sliced at fixed heights, floor at left. This is where
  the pot ends and the canopy begins &mdash; and where the fixed
  <code>POT_HEIGHT_M</code> cut lands relative to the real rim.</p>
  <div class="slices" id="slices"></div>
</section>

<section>
  <div class="eyebrow">Results</div>
  <h2>Above-ground biomass</h2>
  <p class="muted">Leave-one-out cross-validation across every method, on identical
  specimens. Differences are paired bootstraps against the reconstruct-then-regress
  reference, because at this sample size two point estimates side by side say almost
  nothing.</p>
  <div class="scroll" style="margin-bottom:22px"><table id="methods"></table></div>

  <div class="cols">
    <div class="scatter">
      <canvas id="scatter" width="620" height="620"></canvas>
    </div>
    <div>
      <h3>Reading this plot</h3>
      <p class="small muted">Predicted mass against weighed mass for the best method.
      The dashed line is perfect prediction. Points are coloured by species. Hover a
      point for its specimen.</p>
      <p class="small muted">Two clusters, not a continuum. That structure is the single
      most important thing on this page, and it is explained below.</p>
      <div id="scatterinfo" class="small"></div>
    </div>
  </div>
</section>

<section>
  <div class="eyebrow">Honesty</div>
  <h2>What these numbers cannot say</h2>

  <div class="callout bad">
    <h3>Batch membership explains more than any method</h3>
    <p id="confound"></p>
    <p class="small">So the comparison partly measures how well a method separates
    <em>size classes</em>, not how well it estimates mass among comparable plants.
    V001&ndash;V008 was captured to break this &mdash; its masses span both existing
    clusters instead of forming a third &mdash; and it did, but not all the way.</p>
  </div>

  <div class="callout bad">
    <h3>Most reconstructions cannot weigh what the plant weighs</h3>
    <p id="plausibility"></p>
    <p class="small">A visual hull encloses the space <em>between</em> leaves and
    branches, so for a canopy it measures the envelope rather than the plant. That is a
    property of the method at this resolution, not a fitting problem &mdash; no
    regressor recovers mass from a volume that is an order of magnitude too large.</p>
  </div>

  <div class="callout">
    <h3>Three caveats that travel with every figure here</h3>
    <p class="small" id="notes"></p>
  </div>
</section>

<section>
  <div class="eyebrow">Next</div>
  <h2>Pose-free reconstruction as an independent check</h2>
  <p class="muted">Every pose on this page is estimated from depth, and the azimuth
  refinement saturates its search bound on almost every specimen &mdash; so the
  registration is the least verified assumption in the whole pipeline.</p>
  <p class="muted">DUSt3R, MASt3R and Fast3R estimate cameras <em>and</em> geometry from
  images alone. Running them gives a reconstruction that shares no failure mode with the
  carve, and a second opinion on the poses. MASt3R's metric variant matters most: it
  returns real scale, which is what a volume needs.</p>
  <div class="scroll"><table id="next"></table></div>
</section>

<footer class="wrap">
  <p>Generated from the GG-SSVT pipeline. Ground truth is as-collected fresh mass, not
  oven-dry above-ground biomass. Camera poses are estimated, not measured.</p>
</footer>
</div>

<script>
const D = __PAYLOAD__;
const S = D.specimens, byId = new Map(S.map(s => [s.id, s]));
let current = S[0].id, segmenter = D.summary.segmenters[0], colourMode = "segment";
let yaw = 0.62, pitch = 0.2, zoom = 0.9;

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const hex = h => { h=h.replace("#",""); if(h.length===3) h=h.split("").map(c=>c+c).join("");
  return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)]; };
const mix = (a,b,t) => Math.round(a+(b-a)*t);
const f = (v,d=2) => (v===null||v===undefined||Number.isNaN(v)) ? "—" : Number(v).toFixed(d);

async function inflate(b64){
  const bin = Uint8Array.from(atob(b64), c=>c.charCodeAt(0));
  if (typeof DecompressionStream === "undefined") return null;
  const st = new Blob([bin]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(st).arrayBuffer());
}

const cache = new Map();
async function cloud(id, seg){
  const key = id+"/"+seg, e = byId.get(id).clouds[seg];
  if (cache.has(key)) return cache.get(key);
  if (!e) { cache.set(key,null); return null; }
  const bytes = await inflate(e.data);
  let out = null;
  if (bytes){
    const n = Math.floor(bytes.length/3);
    const pts = new Float32Array(n*3), h = new Float32Array(n);
    let lo=[255,255,255], hi=[0,0,0];
    for(let i=0;i<n;i++) for(let a=0;a<3;a++){
      const v=bytes[i*3+a]; if(v<lo[a])lo[a]=v; if(v>hi[a])hi[a]=v; }
    const span = Math.max(1, hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]);
    const mid = [(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,(lo[2]+hi[2])/2];
    // z byte 0..255 maps to 0..(resolution*voxel) metres over the working volume
    const metresPerByte = (e.resolution * 0.024) / 255;
    for(let i=0;i<n;i++){
      pts[i*3]   = (bytes[i*3]  -mid[0])/span;
      pts[i*3+1] = (bytes[i*3+1]-mid[1])/span;
      pts[i*3+2] = (bytes[i*3+2]-mid[2])/span + 0.5;
      h[i] = bytes[i*3+2] * metresPerByte;
    }
    out = {pts, h, n, id};
  }
  cache.set(key,out); return out;
}

function draw(canvas, c, {dot=3, mode=colourMode, yawL=yaw, pitchL=pitch, zoomL=zoom}={}){
  const ctx = canvas.getContext("2d"), dpr = Math.min(devicePixelRatio||1,2);
  const w = canvas.clientWidth, hgt = canvas.clientHeight;
  if (canvas.width !== Math.round(w*dpr)){ canvas.width=Math.round(w*dpr); canvas.height=Math.round(hgt*dpr); }
  ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,w,hgt);
  if(!c){ ctx.fillStyle=css("--ink-3"); ctx.font="13px 'IBM Plex Sans',sans-serif";
    ctx.textAlign="center"; ctx.fillText("not reconstructed",w/2,hgt/2); return; }

  const cy=Math.cos(yawL), sy=Math.sin(yawL), cp=Math.cos(pitchL), sp=Math.sin(pitchL);
  const P = new Float32Array(c.n*3);
  for(let i=0;i<c.n;i++){
    const x=c.pts[i*3], y=c.pts[i*3+1], z=c.pts[i*3+2]-0.5;
    const rx=x*cy+y*sy, rz=-x*sy+y*cy;
    P[i*3]=rx; P[i*3+1]=z*cp-rz*sp; P[i*3+2]=z*sp+rz*cp;
  }
  const ord = Array.from({length:c.n},(_,i)=>i).sort((a,b)=>P[a*3+2]-P[b*3+2]);
  const scale = Math.min(w,hgt)*zoomL;
  let dmin=Infinity,dmax=-Infinity;
  for(let i=0;i<c.n;i++){const d=P[i*3+2]; if(d<dmin)dmin=d; if(d>dmax)dmax=d;}
  const dspan = Math.max(1e-6,dmax-dmin);
  const nc=hex(css("--near")), fc=hex(css("--far"));
  const cc=hex(css("--canopy")), pc=hex(css("--pot"));
  const pot = potHeight(c);

  for(const i of ord){
    const sx=w/2+P[i*3]*scale, sv=hgt/2-P[i*3+1]*scale;
    const t=(P[i*3+2]-dmin)/dspan;
    let col;
    if(mode==="segment"){
      const base = c.h[i] > pot ? cc : pc;
      const k = 0.55+0.45*t;
      col = `rgb(${Math.round(base[0]*k)},${Math.round(base[1]*k)},${Math.round(base[2]*k)})`;
    } else {
      col = `rgb(${mix(fc[0],nc[0],t)},${mix(fc[1],nc[1],t)},${mix(fc[2],nc[2],t)})`;
    }
    ctx.fillStyle=col; ctx.fillRect(sx,sv,dot,dot);
  }
}

/* Each specimen's own rim: pot mass spans 0.7-32 kg across the batches, so a
   shared cut height draws the pot/canopy boundary in the wrong place for most
   of them. Falls back to the global constant for specimens where no rim was
   detectable. */
function potHeight(c){
  const s = c && D.specimens ? D.specimens.find(x => x.id === c.id) : null;
  return (s && s.pot_height_m != null) ? s.pot_height_m : D.summary.pot_height_m;
}

/* ---------- slices ---------- */
function drawSlices(c){
  const host = document.getElementById("slices"); host.innerHTML="";
  if(!c) return;
  const pot = potHeight(c);
  const hmax = Math.max(...c.h);
  const bands = 6, step = hmax/bands;
  for(let b=0;b<bands;b++){
    const lo=b*step, hi=(b+1)*step;
    const cell=document.createElement("div"); cell.className="slice";
    const cv=document.createElement("canvas"); cv.width=180; cv.height=180;
    const cap=document.createElement("div"); cap.className="cap";
    cap.textContent = `${lo.toFixed(2)}–${hi.toFixed(2)} m`;
    cell.appendChild(cv); cell.appendChild(cap); host.appendChild(cell);

    const ctx=cv.getContext("2d");
    ctx.fillStyle=css("--sunken"); ctx.fillRect(0,0,180,180);
    const isCanopy = lo >= pot;
    const base = hex(isCanopy ? css("--canopy") : css("--pot"));
    ctx.fillStyle = `rgb(${base[0]},${base[1]},${base[2]})`;
    let count=0;
    for(let i=0;i<c.n;i++){
      if(c.h[i]<lo||c.h[i]>=hi) continue;
      const x=90+c.pts[i*3]*150, y=90+c.pts[i*3+1]*150;
      ctx.fillRect(x,y,2,2); count++;
    }
    if(!count){ ctx.fillStyle=css("--ink-3"); ctx.font="11px 'IBM Plex Sans',sans-serif";
      ctx.textAlign="center"; ctx.fillText("empty",90,92); }
    cap.textContent += isCanopy ? "  canopy" : "  pot";
  }
}

/* ---------- static sections ---------- */
function stats(){
  const s=D.summary;
  const rows=[["Specimens",s.n_specimens],["Views each",s.n_views],
    ["Species",s.species.length],["Segmenters",s.segmenters.length],
    ["Mass range",`${s.mass_range_kg[0]}–${s.mass_range_kg[1]} kg`],
    ["Methods compared",D.methods.length]];
  document.getElementById("stats").innerHTML =
    rows.map(([k,v])=>`<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("");
}

function pipeline(){
  const stages=[
    ["01","Acquisition","Two Kinect v2 units carried through six positions, 12 registered RGB-D frames per plant."],
    ["02","Registration","Floor plane per view gives tilt, roll and height; the subject axis fixes the origin. No calibration target was ever captured."],
    ["03","Segmentation","A cylinder about the plant axis, optionally refined by SAM and made consistent across views."],
    ["04","Carving","Silhouette and depth carving to a 128³ occupancy field. This is the self-supervision target."],
    ["05","Mesh","Marching cubes, then surface area, enclosed volume and solidity."],
    ["06","Biomass","Volume and shape read for above-ground mass, scored leave-one-out."]];
  document.getElementById("pipe").innerHTML = stages.map(([n,t,d])=>
    `<div class="stage"><div class="n">${n}</div><h3>${t}</h3><p>${d}</p></div>`).join("");
}

function methodTable(){
  const ref=D.summary.reference_method;
  const head=`<thead><tr><th>Method</th><th>RMSE kg</th><th>MAE kg</th><th>MARE %</th>
    <th>R²</th><th>vs ${ref}</th></tr></thead>`;
  const rows=D.methods.map((m,i)=>{
    const v=m.vs_reference;
    const cell = m.name===ref ? '<span class="muted small">reference</span>'
      : v ? `<span class="num">${v.difference>0?"+":""}${f(v.difference,3)}</span>
             <span class="tag ${v.resolved?"yes":"no"}">${v.resolved?"resolved":"not resolved"}</span>`
          : "—";
    return `<tr class="${i===0?"best":""}"><td>${m.name}</td>
      <td class="num">${f(m.rmse_kg,3)}</td><td class="num">${f(m.mae_kg,3)}</td>
      <td class="num">${f(m.mare_pct,1)}</td><td class="num">${f(m.r2,3)}</td>
      <td>${cell}</td></tr>`;}).join("");
  document.getElementById("methods").innerHTML = head+`<tbody>${rows}</tbody>`;
}

function nextTable(){
  const rows=[
    ["DUSt3R","naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt","open","Pairwise pose-free point maps; needs scale alignment against the Kinect depth."],
    ["MASt3R","naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric","open","Metric variant — returns real scale, so volume is directly comparable."],
    ["Fast3R","jedyang97/Fast3R_ViT_Large_512","open","All twelve views in one forward pass rather than pairwise."]];
  document.getElementById("next").innerHTML =
    `<thead><tr><th>Method</th><th>Weights</th><th>Access</th><th>Why it is worth running</th></tr></thead>`
    + `<tbody>${rows.map(([a,b,c,d])=>
      `<tr><td><strong>${a}</strong></td><td class="num small">${b}</td>
       <td><span class="tag yes">${c}</span></td><td class="small">${d}</td></tr>`).join("")}</tbody>`;
}

function honesty(){
  const s=D.summary;
  document.getElementById("confound").innerHTML =
    `Across the Eucalyptus specimens, knowing only which capture batch a plant came from
     explains <strong>R² = ${s.batch_confound_r2}</strong> of the mass variance &mdash;
     more than any method in the table above achieves. The batches average
     ${Object.entries(s.batch_means_kg).map(([k,v])=>`<strong>${v} kg</strong> (${k})`).join(", ")}.
     It was R² = 0.887 on the two original batches alone; V001&ndash;V008 overlaps both
     and brings it down.`;
  const pl = s.plausibility;
  document.getElementById("plausibility").innerHTML =
    `Dividing each measured mass by its reconstructed above-ground volume gives an implied
     bulk density. Fresh plant tissue is 300&ndash;900 kg/m³. Only
     <strong>${pl.n_plausible} of ${pl.n}</strong> specimens land inside a generous
     ${pl.band_kg_m3[0]}&ndash;${pl.band_kg_m3[1]} band; the median is
     <strong>${pl.median_density_kg_m3} kg/m³</strong>.
     ${pl.verdicts.envelope || 0} imply less &mdash; hull enclosing air &mdash; and
     ${pl.verdicts.missing || 0} imply more, meaning they were barely reconstructed.`;
  document.getElementById("notes").innerHTML =
    Object.values(D.notes).map(t=>`• ${t}`).join("<br>");
}

/* ---------- scatter ---------- */
function scatter(){
  const cv=document.getElementById("scatter"), ctx=cv.getContext("2d");
  const best=D.methods[0].name;
  const pts=S.map(s=>({x:s.target_kg, y:s.predictions[best], id:s.id, sp:s.species}));
  const lim=Math.max(...pts.flatMap(p=>[p.x,p.y]))*1.12, pad=54;
  const sx=v=>pad+(v/lim)*(cv.width-2*pad), sy=v=>cv.height-pad-(v/lim)*(cv.height-2*pad);

  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.fillStyle=css("--surface"); ctx.fillRect(0,0,cv.width,cv.height);
  ctx.strokeStyle=css("--line-strong"); ctx.setLineDash([5,5]); ctx.beginPath();
  ctx.moveTo(sx(0),sy(0)); ctx.lineTo(sx(lim),sy(lim)); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle=css("--ink-3"); ctx.beginPath();
  ctx.moveTo(pad,cv.height-pad); ctx.lineTo(cv.width-pad,cv.height-pad);
  ctx.moveTo(pad,pad); ctx.lineTo(pad,cv.height-pad); ctx.stroke();

  ctx.fillStyle=css("--ink-3"); ctx.font="12px 'IBM Plex Mono',monospace";
  for(let i=0;i<=4;i++){
    const v=lim*i/4;
    ctx.textAlign="center"; ctx.fillText(v.toFixed(1), sx(v), cv.height-pad+18);
    ctx.textAlign="right";  ctx.fillText(v.toFixed(1), pad-8, sy(v)+4);
  }
  ctx.textAlign="center"; ctx.font="13px 'IBM Plex Sans',sans-serif";
  ctx.fillText("weighed mass (kg)", cv.width/2, cv.height-14);
  ctx.save(); ctx.translate(16,cv.height/2); ctx.rotate(-Math.PI/2);
  ctx.fillText("predicted mass (kg)",0,0); ctx.restore();

  const palette={};
  D.summary.species.forEach((sp,i)=>{ palette[sp] = i===0 ? css("--accent") : css("--near"); });
  for(const p of pts){
    ctx.fillStyle=palette[p.sp]||css("--ink-2"); ctx.globalAlpha=.8;
    ctx.beginPath(); ctx.arc(sx(p.x),sy(p.y),6,0,7); ctx.fill(); ctx.globalAlpha=1;
  }
  ctx.font="12px 'IBM Plex Sans',sans-serif"; ctx.textAlign="left";
  D.summary.species.forEach((sp,i)=>{
    ctx.fillStyle=palette[sp]; ctx.beginPath(); ctx.arc(pad+12,pad+14+i*20,5,0,7); ctx.fill();
    ctx.fillStyle=css("--ink-2"); ctx.fillText(sp,pad+24,pad+18+i*20);
  });

  document.getElementById("scatterinfo").innerHTML =
    `<p class="small muted">Showing <strong>${best}</strong>, the lowest-RMSE method.</p>`;
  cv.onmousemove = e=>{
    const r=cv.getBoundingClientRect();
    const mx=(e.clientX-r.left)*cv.width/r.width, my=(e.clientY-r.top)*cv.height/r.height;
    const hit=pts.find(p=>Math.hypot(sx(p.x)-mx, sy(p.y)-my)<10);
    document.getElementById("scatterinfo").innerHTML = hit
      ? `<p class="small"><strong>${hit.id}</strong> &middot; ${hit.sp}<br>
         weighed ${f(hit.x)} kg &middot; predicted ${f(hit.y)} kg</p>`
      : `<p class="small muted">Showing <strong>${best}</strong>, the lowest-RMSE method.</p>`;
  };
}

/* ---------- explorer ---------- */
function segButtons(){
  document.getElementById("segmenter").innerHTML = D.summary.segmenters.map((s,i)=>
    `<button data-seg="${s}" aria-pressed="${i===0}">${s==="sam3d"?"SAM3D":"Geometric"}</button>`).join("");
}
function legend(){
  document.getElementById("legend").innerHTML = colourMode==="segment"
    ? `<div><i style="background:var(--canopy)"></i>above pot rim</div>
       <div><i style="background:var(--pot)"></i>pot and below</div>`
    : `<div><i style="background:var(--near)"></i>nearer</div>
       <div><i style="background:var(--far)"></i>further</div>`;
}
async function buildList(){
  const host=document.getElementById("list"); host.innerHTML=""; let last=null;
  for(const s of [...S].sort((a,b)=>a.id.localeCompare(b.id))){
    if(s.species!==last){ last=s.species;
      const g=document.createElement("div"); g.className="grp"; g.textContent=s.species; host.appendChild(g); }
    const b=document.createElement("button"); b.className="item"; b.type="button";
    b.setAttribute("aria-current", s.id===current);
    b.innerHTML=`<canvas width="80" height="80"></canvas>
      <span><span class="id">${s.id}</span><br><span class="m">${f(s.target_kg)} kg</span></span>
      <span class="kg">${f(s.quality[segmenter]?.above_ground_l,1)} L</span>`;
    b.onclick=()=>select(s.id);
    host.appendChild(b);
    cloud(s.id,segmenter).then(c=>draw(b.querySelector("canvas"),c,{dot:1,zoomL:0.8}));
  }
}
async function render(){
  const c = await cloud(current, segmenter);
  draw(document.getElementById("stage"), c);
  drawSlices(c);
  document.getElementById("cloudinfo").textContent =
    c ? `${c.n.toLocaleString()} voxels shown` : "";
  legend();
}
function facts(){
  const s=byId.get(current), q=s.quality[segmenter]||{}, m=s.mesh||{};
  const rows=[["Weighed",f(s.target_kg)+" kg"],["Hull",f(q.volume_l,1)+" L"],
    ["Above rim",f(q.above_ground_l,1)+" L"],["Height",f(q.height_m)+" m"],
    ["Canopy area",f(m.canopy_area_m2,2)+" m²"],["Solidity",f(m.solidity,2)]];
  document.getElementById("sfacts").innerHTML = rows.map(([k,v])=>
    `<div class="fact"><span>${k}</span><b>${v}</b></div>`).join("");
}
async function select(id){
  current=id; const s=byId.get(id);
  document.getElementById("sid").textContent=id;
  document.getElementById("sspecies").textContent=s.species;
  const best=D.methods[0].name;
  document.getElementById("ssub").textContent =
    `${best}: ${f(s.predictions[best])} kg predicted`;
  document.querySelectorAll(".item").forEach(b=>
    b.setAttribute("aria-current", b.querySelector(".id").textContent===id));
  facts(); await render();
}

function attach(){
  const cv=document.getElementById("stage");
  let drag=false, lx=0, ly=0;
  cv.addEventListener("pointerdown",e=>{drag=true;lx=e.clientX;ly=e.clientY;cv.setPointerCapture(e.pointerId)});
  cv.addEventListener("pointermove",e=>{ if(!drag)return;
    yaw+=(e.clientX-lx)*0.01; pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-ly)*0.01));
    lx=e.clientX; ly=e.clientY; cloud(current,segmenter).then(c=>draw(cv,c)); });
  const stop=e=>{drag=false; try{cv.releasePointerCapture(e.pointerId)}catch(_){}};
  cv.addEventListener("pointerup",stop); cv.addEventListener("pointercancel",stop);
  cv.addEventListener("wheel",e=>{e.preventDefault();
    zoom=Math.max(.3,Math.min(2.4,zoom*(e.deltaY>0?.92:1.08)));
    cloud(current,segmenter).then(c=>draw(cv,c)); },{passive:false});

  document.getElementById("segmenter").onclick=async e=>{
    const b=e.target.closest("button"); if(!b)return;
    segmenter=b.dataset.seg;
    [...e.currentTarget.children].forEach(x=>x.setAttribute("aria-pressed",x===b));
    await buildList(); facts(); await render(); };
  document.getElementById("colour").onclick=async e=>{
    const b=e.target.closest("button"); if(!b)return;
    colourMode=b.dataset.mode;
    [...e.currentTarget.children].forEach(x=>x.setAttribute("aria-pressed",x===b));
    await render(); };
  addEventListener("resize",()=>cloud(current,segmenter).then(c=>draw(document.getElementById("stage"),c)));
}

stats(); pipeline(); methodTable(); nextTable(); honesty(); scatter(); segButtons();
buildList().then(attach).then(()=>select(current));
</script>
"""


def build_dashboard(
    payload_json: str,
    out_path: Path = WORK_DIR / "reports" / "dashboard.html",
) -> Path:
    """Write the standalone dashboard page."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(HEAD + BODY.replace("__PAYLOAD__", payload_json), encoding="utf-8")
    return out_path


__all__ = ["build_dashboard"]
