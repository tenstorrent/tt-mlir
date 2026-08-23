# Render n150-ds-matmul-ab.md to a self-contained HTML page for publishing.
#
# The markdown is the source of truth; nothing here retypes a number. Delta cells
# (+1.23% / -4.56%) are coloured from their sign so the fleet result reads at a glance.
import html
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "n150-ds-matmul-ab.md")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "n150-ds-matmul-ab.html")

# signed deltas (µs or %), and penalty ratios read against the 0.98-1.02 noise floor
DELTA = re.compile(r"^([+−-])\d+(?:\.\d+)?%?$")
RATIO = re.compile(r"^(\d+(?:\.\d+)?)x$")


def slug(text):
    s = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s]+", "-", s.strip())


def inline(t):
    t = html.escape(t, quote=False)
    out = []
    for i, part in enumerate(t.split("`")):
        out.append(part if i % 2 == 0 else f"<code>{part}</code>")
    t = "".join(out)
    # external links stay links; repo-relative ones become code (dead in a hosted page)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
               r'<a href="\2" target="_blank" rel="noopener">\1</a>', t)
    t = re.sub(r"\[([^\]]+)\]\((#[^)]+)\)", r'<a href="\2">\1</a>', t)
    t = re.sub(r"\[<code>([^<]+)</code>\]\([^)]+\)", r"<code>\1</code>", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", t)
    return t


def cell(text, is_head):
    inner = inline(text)
    if is_head:
        return f"<th scope='col'>{inner}</th>"
    cls = ""
    s = text.strip()
    md = DELTA.match(s)
    mr = RATIO.match(s)
    if md:
        # for a duration or percentage delta, negative is faster
        cls = " class='num pos'" if md.group(1) in "-−" else " class='num neg'"
    elif mr:
        v = float(mr.group(1))
        cls = (" class='num pos'" if v < 0.98
               else " class='num neg'" if v > 1.02 else " class='num'")
    elif re.match(r"^[\d.,]+$|^—$|^[\d,]+(?:\.\d+)?%$", s):
        cls = " class='num'"
    return f"<td{cls}>{inner}</td>"


def render(md):
    lines = md.split("\n")
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]

        if ln.startswith("```"):
            lang = ln[3:].strip()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            i += 1
            out.append(f"<pre class='code' data-lang='{html.escape(lang)}'><code>"
                       + "\n".join(buf) + "</code></pre>")
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            lvl, txt = len(m.group(1)), m.group(2).strip()
            tag = f"h{lvl}"
            out.append(f"<{tag} id='{slug(txt)}'>{inline(txt)}</{tag}>")
            i += 1
            continue

        if ln.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            def cells(row):
                return [c.strip() for c in row.strip().strip("|").split("|")]
            head = cells(ln)
            i += 2
            body = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                body.append(cells(lines[i]))
                i += 1
            thead = "".join(cell(c, True) for c in head)
            rows = "".join("<tr>" + "".join(cell(c, False) for c in r) + "</tr>" for r in body)
            out.append("<div class='tablewrap'><table><thead><tr>" + thead
                       + "</tr></thead><tbody>" + rows + "</tbody></table></div>")
            continue

        if re.match(r"^\s*-\s+", ln):
            block = []
            while i < len(lines) and (re.match(r"^\s*-\s+", lines[i])
                                      or (lines[i].startswith("  ") and lines[i].strip() and block)):
                block.append(lines[i])
                i += 1
            frag, depth, open_li = ["<ul>"], 0, []
            for b in block:
                mm = re.match(r"^(\s*)-\s+(.*)$", b)
                if mm:
                    d = len(mm.group(1)) // 2
                    while d > depth:
                        frag.append("<ul>")
                        depth += 1
                        open_li.append(False)
                    while d < depth:
                        if open_li.pop():
                            frag.append("</li>")
                        frag.append("</ul>")
                        depth -= 1
                        if open_li and open_li[-1]:
                            frag.append("</li>")
                            open_li[-1] = False
                    if open_li and open_li[-1]:
                        frag.append("</li>")
                        open_li[-1] = False
                    if not open_li:
                        open_li.append(False)
                    frag.append(f"<li>{inline(mm.group(2))}")
                    open_li[-1] = True
                else:
                    frag.append(" " + inline(b.strip()))
            while depth >= 0:
                if open_li and open_li.pop():
                    frag.append("</li>")
                frag.append("</ul>")
                depth -= 1
            out.append("".join(frag))
            continue

        if ln.strip() == "":
            i += 1
            continue

        para = [ln]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^\s*(#|-|\||```)", lines[i]):
            para.append(lines[i])
            i += 1
        out.append("<p>" + inline(" ".join(x.strip() for x in para)) + "</p>")
    return "\n".join(out)


CSS = """
:root{
  --ground:#EFF1F4; --surface:#FFFFFF; --surface-2:#F7F8FA;
  --ink:#171B22; --ink-2:#4E5866; --ink-3:#79838F;
  --rule:#D5DAE1; --rule-2:#E5E9EE;
  --accent:#0F6E78; --accent-soft:#DCEBED;
  --pos:#2E7D5B; --neg:#B4552A;
}
:root:not([data-theme="light"]){ }
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#0E1218; --surface:#151B23; --surface-2:#1A212B;
    --ink:#E6EAEF; --ink-2:#A3AEBB; --ink-3:#78838F;
    --rule:#2A333E; --rule-2:#212934;
    --accent:#4FB3BD; --accent-soft:#163236;
    --pos:#5CB98D; --neg:#DD8A55;
  }
}
:root[data-theme="dark"]{
  --ground:#0E1218; --surface:#151B23; --surface-2:#1A212B;
  --ink:#E6EAEF; --ink-2:#A3AEBB; --ink-3:#78838F;
  --rule:#2A333E; --rule-2:#212934;
  --accent:#4FB3BD; --accent-soft:#163236;
  --pos:#5CB98D; --neg:#DD8A55;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:16px; line-height:1.65; -webkit-font-smoothing:antialiased;
}
.page{max-width:1180px; margin:0 auto; padding:clamp(28px,5vw,72px) clamp(18px,4vw,48px) 96px;
  display:flex; flex-direction:column; gap:0}
.col{max-width:70ch}

h1{
  font-family:Newsreader,Georgia,serif; font-weight:500; font-size:clamp(2rem,4.4vw,3.1rem);
  line-height:1.12; letter-spacing:-.015em; margin:0 0 .5em; text-wrap:balance; max-width:28ch;
}
h2{
  font-family:Newsreader,Georgia,serif; font-weight:500; font-size:clamp(1.45rem,2.6vw,1.95rem);
  line-height:1.2; letter-spacing:-.01em; margin:2.6em 0 .1em; text-wrap:balance; max-width:34ch;
}
h3{
  font-family:"IBM Plex Sans",sans-serif; font-weight:600; font-size:1.02rem;
  letter-spacing:.02em; margin:2.1em 0 .6em; color:var(--ink); text-wrap:balance; max-width:52ch;
}
h4{font-size:.95rem; font-weight:600; margin:1.6em 0 .4em}

/* section rule: 12 segments, one per n150 DRAM channel */
h2::after{
  content:""; display:block; margin:.75em 0 1.15em; height:3px; width:min(320px,60%);
  background:repeating-linear-gradient(to right,
    var(--accent) 0 calc(100%/12 - 3px), transparent calc(100%/12 - 3px) calc(100%/12));
  opacity:.85;
}
h1+p{font-size:1.08rem; color:var(--ink-2); max-width:66ch}

p{margin:0 0 1.05em; max-width:70ch}
ul{margin:0 0 1.15em; padding-left:1.15em; max-width:72ch}
li{margin:.3em 0}
li>ul{margin:.35em 0 .1em}
strong{font-weight:600}
a{color:var(--accent); text-decoration:none; border-bottom:1px solid color-mix(in srgb,var(--accent) 40%,transparent)}
a:hover{border-bottom-color:var(--accent)}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent); outline-offset:3px; border-radius:2px}

code{
  font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace; font-size:.86em;
  background:var(--surface-2); border:1px solid var(--rule-2); border-radius:3px;
  padding:.08em .34em; color:var(--ink);
}
pre.code{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.845rem; line-height:1.62;
  background:var(--surface); border:1px solid var(--rule); border-left:3px solid var(--accent);
  border-radius:4px; padding:16px 18px; overflow-x:auto; margin:0 0 1.4em; max-width:72ch;
}
pre.code code{background:none;border:none;padding:0;font-size:1em}

.tablewrap{
  overflow-x:auto; margin:.4em 0 1.9em; background:var(--surface);
  border:1px solid var(--rule); border-radius:5px; max-width:100%;
}
table{border-collapse:collapse; width:100%; font-size:.845rem}
thead th{
  position:sticky; top:0; z-index:1; background:var(--surface-2);
  font-family:"IBM Plex Sans",sans-serif; font-weight:600; font-size:.72rem;
  letter-spacing:.055em; text-transform:uppercase; color:var(--ink-2);
  text-align:left; white-space:nowrap; padding:10px 13px;
  border-bottom:1px solid var(--rule); box-shadow:inset 0 -1px 0 var(--rule);
}
tbody td{
  padding:8px 13px; border-bottom:1px solid var(--rule-2); vertical-align:top;
  font-variant-numeric:tabular-nums; white-space:nowrap;
}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--surface-2)}
td.num{text-align:right; font-family:"IBM Plex Mono",monospace; font-size:.9em}
td.pos{color:var(--pos); font-weight:600}
td.neg{color:var(--neg); font-weight:600}
tbody td:first-child{font-family:"IBM Plex Mono",monospace; font-size:.86em; color:var(--ink-2)}

@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
@media (max-width:640px){
  thead th{position:static}
  body{font-size:15px}
}
"""

md = SRC.read_text()
body = render(md)
DST.write_text(f"""<title>DRAM-Sharded Matmul on n150</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="page">
{body}
</div>
""")
print(f"wrote {DST} ({DST.stat().st_size//1024} KB)")
