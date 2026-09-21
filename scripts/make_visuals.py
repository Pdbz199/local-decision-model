"""Renders every image and GIF in assets/ from assets/data.json (real model outputs, see collect_visual_data.py).

Usage: uv run python scripts/make_visuals.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path("assets")
DATA = json.loads((ASSETS / "data.json").read_text())

# Dark-surface steps of a palette validated for color vision deficiency; categorical slots are used in fixed order.
SURFACE, CARD, TRACK = "#1a1a19", "#242422", "#383835"
TEXT, TEXT2, MUTED = "#ffffff", "#c3c2b7", "#8a897f"
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"

FONT_DIR = Path(font_manager.findfont("DejaVu Sans")).parent
S = 2  # supersampling factor for the GIFs
W, H = 960, 540


def font(size, bold=False, mono=False):
    name = ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if bold else "") + ".ttf"
    return ImageFont.truetype(str(FONT_DIR / name), int(size * S))


def ease(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def blend(c1, c2, a):
    h = lambda c: [int(c[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(x + (y - x) * a):02x}" for x, y in zip(h(c1), h(c2)))


class Canvas:
    def __init__(self):
        self.img = Image.new("RGB", (W * S, H * S), SURFACE)
        self.d = ImageDraw.Draw(self.img)

    def rect(self, x0, y0, x1, y1, fill, r=0):
        if x1 - x0 < 0.5:
            return
        self.d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=r * S, fill=fill)

    def text(self, x, y, s, f, fill=TEXT, anchor="la"):
        self.d.text((x * S, y * S), s, font=f, fill=fill, anchor=anchor)

    def width(self, s, f):
        return self.d.textlength(s, font=f) / S

    def done(self):
        return self.img.resize((W, H), Image.LANCZOS)


def save_gif(frames, path, fps):
    pal = frames[len(frames) // 2].quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True, disposal=1)
    print(f"{path}  {len(frames)} frames  {path.stat().st_size / 1e6:.2f} MB")


def header(c, kicker, headline):
    c.text(32, 24, kicker, font(11, bold=True), MUTED)
    c.text(32, 44, headline, font(23, bold=True), TEXT)


# ----------------------------------------------------------------------------- hero: state in, typed decisions out

def fmt_field(name, f):
    """Returns (value text, bar fraction, right-hand label) for one decision row."""
    t, v = f["type"], f["value"]
    if t == "Bool":
        return str(v), f["confidence"], f"{f['confidence']:.0%}"
    if t == "Enum":
        return f'"{v}"', f["confidence"], f"{f['confidence']:.0%}"
    if t == "MultiLabel":
        return "[" + ", ".join(f'"{x}"' for x in v) + "]", f["confidence"], f"{f['confidence']:.0%}"
    if t == "Int":
        return f"{v}   (expected {f['expected']:.1f} of 5)", (f["expected"] - 1) / 4, "scale"
    if t == "Float":
        return f"{v:.2f}", v, "scale"
    return (f'"{v}"' if v is not None else "None"), f["confidence"], f"{f['confidence']:.0%}"


def hero_frame(ticket, n_chars, reveal, hold):
    c = Canvas()
    header(c, "LOCAL DECISION MODEL", "Unstructured state in. Typed, calibrated decisions out.")
    # left card: the state
    c.rect(32, 92, 440, 470, CARD, r=10)
    c.text(48, 106, "STATE", font(11, bold=True), MUTED)
    c.text(100, 106, "any text or JSON", font(11), MUTED)
    mono, cw, lh = font(14, mono=True), 14 * 0.602, 25
    lines, spans, pos = [], [], 0
    for line in textwrap.wrap(ticket["text"], 42, drop_whitespace=False):
        lead = len(line) - len(line.lstrip())
        lines.append((pos + lead, line.lstrip()))
        pos += len(line)
    shown = ticket["text"][:n_chars]
    marks = [(ticket["fields"][k]["span"], col) for k, col in (("order_id", BLUE), ("email", AQUA)) if ticket["fields"][k]["span"]]
    for row, (start, line) in enumerate(lines):
        y = 136 + row * lh
        for (a, b), col in marks:  # highlight extracted spans, character-exact because the font is monospaced
            lo, hi = max(a, start), min(b, start + len(line))
            if lo < hi and reveal > 0:
                c.rect(48 + (lo - start) * cw - 2, y - 3, 48 + (hi - start) * cw + 2, y + 19, blend(CARD, col, 0.55 * ease(reveal)), r=3)
        c.text(48, y, shown[start:start + len(line)] if n_chars > start else "", mono, TEXT2)
    if n_chars < len(ticket["text"]):
        row = max(i for i, (s, _) in enumerate(lines) if s <= n_chars)
        c.rect(48 + (n_chars - lines[row][0]) * cw, 136 + row * lh, 48 + (n_chars - lines[row][0]) * cw + 8, 136 + row * lh + 18, TEXT)
    # right card: typed decisions
    c.rect(464, 92, 928, 470, CARD, r=10)
    c.text(480, 106, "TYPED DECISIONS", font(11, bold=True), MUTED)
    c.text(912, 106, "dm.decide(state, schema)", font(11, mono=True), MUTED, anchor="ra")
    for i, (name, f) in enumerate(ticket["fields"].items()):
        y = 134 + i * 41
        t = ease(reveal * 1.6 - i * 0.08)
        c.text(480, y, name, font(13, mono=True), TEXT2)
        c.text(480, y + 17, f["type"], font(10), MUTED)
        c.rect(580, y + 24, 840, y + 28, TRACK, r=2)
        if t > 0:
            value, frac, label = fmt_field(name, f)
            low = label != "scale" and f["confidence"] < 0.6
            c.text(580, y, value if len(value) < 40 else value[:38] + "…", font(13, bold=True), blend(CARD, TEXT, t))
            col = ORANGE if low else BLUE
            c.rect(580, y + 24, 580 + 260 * max(frac, 0.01) * t, y + 28, col, r=2)
            c.text(912, y + 17, "unsure" if low and t > 0.9 else (label if label == "scale" else f"{f['confidence'] * t:.0%}"),
                   font(11), blend(CARD, TEXT2, t), anchor="ra")
    if hold > 0:
        a = ease(hold)
        c.text(32, 488, f"{ticket['ms']:.0f} ms", font(30, bold=True), blend(SURFACE, TEXT, a))
        x = 32 + c.width(f"{ticket['ms']:.0f} ms", font(30, bold=True)) + 16
        c.text(x, 492, "8 typed fields, one batched forward pass, on one consumer GPU.", font(13), blend(SURFACE, TEXT2, a))
        c.text(x, 511, "Every value is guaranteed to match its declared type. Orange bar = the model itself is unsure.",
               font(13), blend(SURFACE, MUTED, a))
    return c.done()


def make_hero():
    frames, fps = [], 15
    for ticket in DATA["tickets"]:
        n = len(ticket["text"])
        frames += [hero_frame(ticket, k, 0, 0) for k in range(0, n, 7)]
        frames += [hero_frame(ticket, n, 0, 0)] * 3
        frames += [hero_frame(ticket, n, k / 16, (k - 8) / 8) for k in range(1, 17)]
        frames += [hero_frame(ticket, n, 1, 1)] * 38
    save_gif(frames, ASSETS / "hero.gif", fps)
    frames[len(frames) // 3 - 1].save(ASSETS / "hero.png")


# ----------------------------------------------------------------------------- routing: 77 options at once

def routing_frame(ex, options, n_chars, grow, hold):
    c = Canvas()
    header(c, "HIGH-CARDINALITY ROUTING, ZERO-SHOT", "77 possible intents, all scored in one forward pass.")
    c.rect(32, 92, 928, 150, CARD, r=10)
    c.text(48, 104, "INCOMING MESSAGE", font(11, bold=True), MUTED)
    c.text(48, 123, ex["text"][:n_chars], font(15, mono=True), TEXT)
    probs = ex["probs"]
    best = max(range(len(probs)), key=probs.__getitem__)
    ok = options[best] == ex["gold"]
    confident = probs[best] >= 0.7
    base, top, step = 410, 200, 896 / len(probs)
    c.rect(32, base + 1, 928, base + 2, TRACK)
    for i, p in enumerate(probs):
        x = 32 + i * step
        h = (base - top) * p * ease(grow * 1.4 - 0.3 * i / len(probs))
        col = (BLUE if confident else ORANGE) if i == best else blend(SURFACE, BLUE, 0.45)
        c.rect(x + 2, base - max(h, 2), x + step - 2, base, col, r=2)
    c.text(32, 420, "each bar is one of the 77 intent names passed in at call time (the model never trained on this dataset)", font(11), MUTED)
    if grow >= 1:
        a = ease(hold)
        x = min(max(32 + best * step + step / 2, 150), 810)
        c.text(x, base - (base - top) * probs[best] - 22, f"{options[best]}  {probs[best]:.0%}", font(13, bold=True), blend(SURFACE, TEXT, a), anchor="ma")
        c.rect(32, 452, 928, 516, CARD, r=10)
        c.rect(48, 477, 60, 489, BLUE if confident else ORANGE, r=6)
        verdict = "AUTOMATE" if confident else "ESCALATE TO A HUMAN"
        c.text(70, 466, verdict, font(15, bold=True), blend(CARD, TEXT, a))
        why = f"confidence {probs[best]:.0%} clears the 70% bar" if confident else f"confidence {probs[best]:.0%} is below the 70% bar"
        c.text(70, 488, why + f"   |   labeled answer: {ex['gold']} ({'match' if ok else 'a miss, and the low confidence flagged it'})",
               font(12), blend(CARD, TEXT2, a))
        c.text(912, 466, f"{ex['ms']:.0f} ms", font(15, bold=True), blend(CARD, TEXT, a), anchor="ra")
    return c.done()


def make_routing():
    opts, frames = DATA["routing"]["options"], []
    for idx in (2, 5, 1, 4):  # two confident hits, one low-confidence miss, one more hit
        ex = DATA["routing"]["examples"][idx]
        n = len(ex["text"])
        frames += [routing_frame(ex, opts, k, 0, 0) for k in range(0, n, 4)]
        frames += [routing_frame(ex, opts, n, k / 12, 0) for k in range(1, 12)]
        frames += [routing_frame(ex, opts, n, 1, k / 6) for k in range(1, 7)]
        frames += [routing_frame(ex, opts, n, 1, 1)] * 34
    save_gif(frames, ASSETS / "routing.gif", 15)
    frames[-1].save(ASSETS / "routing.png")
    frames[len(frames) // 2 + 50].save(ASSETS / "routing_escalate.png")


# ----------------------------------------------------------------------------- static charts

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(TRACK)
    ax.tick_params(colors=TEXT2, labelsize=10, length=0)
    ax.grid(color=TRACK, linewidth=1)
    ax.set_axisbelow(True)


def new_fig(w, h, title, subtitle):
    fig = plt.figure(figsize=(w, h), dpi=200, facecolor=SURFACE)
    fig.text(0.04, 1 - 0.30 / h, title, color=TEXT, fontsize=15, fontweight="bold", va="top")
    fig.text(0.04, 1 - 0.62 / h, textwrap.fill(subtitle, int(w * 12.2)) if "\n" not in subtitle else subtitle,
             color=TEXT2, fontsize=10.5, va="top", linespacing=1.35)
    return fig


def make_latency():
    rows = DATA["latency"][::-1]
    fig = new_fig(9.6, 4.6, "Fast enough to live inside an if-statement",
                  "Median wall-clock time per call on one RTX 5070 Ti, including tokenization and decoding. Log scale.")
    ax = fig.add_axes([0.34, 0.13, 0.62, 0.66])
    style_axes(ax)
    ax.axvspan(70, 500, color=TRACK, alpha=0.75, lw=0)
    ax.text(187, len(rows) - 0.25, "Jev's advertised range: 70 to 500 ms, hosted", color=TEXT2, fontsize=9, ha="center", va="center")
    ax.barh([r["name"] for r in rows], [r["ms"] for r in rows], color=BLUE, height=0.42, left=1)
    for i, r in enumerate(rows):
        ax.text(r["ms"] * 1.12 + 1, i, f"{r['ms']:.0f} ms", color=TEXT, fontsize=10, va="center", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(1, 1500)
    ax.set_ylim(-0.6, len(rows) + 0.1)
    ax.minorticks_off()
    ax.set_xticks([1, 10, 100, 1000], ["1 ms", "10 ms", "100 ms", "1 s"])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=10.5, colors=TEXT)
    fig.savefig(ASSETS / "latency.png", facecolor=SURFACE)
    plt.close(fig)


def make_calibration():
    rows = DATA["calibration"]
    n = sum(r["n"] for r in rows)
    ece = sum(r["n"] / n * abs(r["confidence"] - r["accuracy"]) for r in rows)
    fig = new_fig(6.4, 5.2, "When it says 90%, is it right 90% of the time?",
                  f"4,500 decisions on tasks the model never trained on. Close to the diagonal\noverall (calibration error {ece:.3f}), but overconfident between 80% and 95%.")
    ax = fig.add_axes([0.13, 0.12, 0.82, 0.64])
    style_axes(ax)
    ax.plot([0.5, 1], [0.5, 1], color=MUTED, lw=1)
    ax.text(0.575, 0.553, "perfect calibration", color=MUTED, fontsize=9, rotation=39, rotation_mode="anchor")
    xs, ys = [r["confidence"] for r in rows], [r["accuracy"] for r in rows]
    ax.plot(xs, ys, color=BLUE, lw=2, solid_capstyle="round")
    ax.scatter(xs, ys, s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"{ys[-1]:.0%} correct at\n{xs[-1]:.0%} confidence", (xs[-1], ys[-1]), (0.80, 0.985), color=TEXT, fontsize=9.5, ha="center", va="bottom")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.06)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("confidence the model reported", color=TEXT2, fontsize=10)
    ax.set_ylabel("how often it was actually right", color=TEXT2, fontsize=10)
    fig.savefig(ASSETS / "calibration.png", facecolor=SURFACE)
    plt.close(fig)


def make_selective():
    rows = DATA["selective"]
    fig = new_fig(6.4, 5.2, "Automate the confident calls, escalate the rest",
                  "77-way intent routing on a dataset the model never saw. Each dot is a confidence threshold.")
    ax = fig.add_axes([0.13, 0.12, 0.82, 0.64])
    style_axes(ax)
    xs, ys = [r["automated"] for r in rows], [r["accuracy"] for r in rows]
    ax.plot(xs, ys, color=BLUE, lw=2, solid_capstyle="round")
    ax.scatter(xs, ys, s=40, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for th, dx, dy, ha in ((0.0, -0.015, -0.035, "right"), (0.5, -0.02, -0.012, "right"), (0.7, -0.02, -0.012, "right"), (0.9, 0.02, 0.012, "left")):
        r = min(rows, key=lambda r: abs(r["threshold"] - th))
        label = "act on everything" if th == 0 else f"act if confidence is at least {th:.0%}"
        ax.text(r["automated"] + dx, r["accuracy"] + dy, f"{label}\n{r['automated']:.0%} automated, {r['accuracy']:.0%} correct",
                color=TEXT, fontsize=9, ha=ha, va="bottom" if dy > 0 else "top")
    ax.set_xlim(0.3, 1.04)
    ax.set_ylim(0.55, 0.95)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("share of messages handled automatically", color=TEXT2, fontsize=10)
    ax.set_ylabel("accuracy on the automated share", color=TEXT2, fontsize=10)
    fig.savefig(ASSETS / "selective_automation.png", facecolor=SURFACE)
    plt.close(fig)


def make_grounding():
    rows = DATA["grounding"][::-1]
    fig = new_fig(9.6, 5.0, "A guardrail that catches an LLM making things up",
                  "Eight candidate answers checked against a refund policy. Four were planted errors. About 10 ms per check.")
    ax = fig.add_axes([0.47, 0.17, 0.49, 0.62])
    style_axes(ax)
    ax.barh(range(len(rows)), [r["p"] for r in rows], color=[BLUE if r["grounded"] else ORANGE for r in rows], height=0.42)
    ax.axvline(0.5, color=MUTED, lw=1)
    ax.text(0.5, len(rows) - 0.35, "decision threshold", color=MUTED, fontsize=9, ha="center")
    for i, r in enumerate(rows):
        ax.text(r["p"] + 0.015, i, f"{r['p']:.2f}", color=TEXT, fontsize=9.5, va="center", fontweight="bold")
    ax.set_yticks(range(len(rows)), [textwrap.fill(r["answer"], 52) for r in rows])
    ax.tick_params(axis="y", labelsize=9, colors=TEXT)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("P(answer is grounded in the source document)", color=TEXT2, fontsize=10)
    for x, col, label in ((0.47, BLUE, "answer is supported by the policy"), (0.74, ORANGE, "planted error")):
        fig.patches.append(FancyBboxPatch((x, 0.035), 0.012, 0.022, boxstyle="round,pad=0,rounding_size=0.004", transform=fig.transFigure, color=col))
        fig.text(x + 0.018, 0.046, label, color=TEXT2, fontsize=9.5, va="center")
    fig.savefig(ASSETS / "grounding_guardrail.png", facecolor=SURFACE)
    plt.close(fig)


def make_architecture():
    fig = new_fig(9.6, 4.9, "How it works: one pass, no text generation",
                  "The schema is written into the input and every option is scored by the same forward pass,\nso the output is always one of the values you declared.")
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.78])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def box(x, y, w, h, label, fc, tc=TEXT, size=9, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec="none"))
        ax.text(x + w / 2, y + h / 2, label, color=tc, fontsize=size, ha="center", va="center", fontweight="bold" if bold else "normal")

    tokens = [("[CLS]", 6, TRACK), ("[ONE]", 6, TRACK), ("Which team?", 13, CARD), ("[OPT]", 6, BLUE), ("billing", 8, CARD),
              ("[OPT]", 6, BLUE), ("tech support", 12, CARD), ("[OPT]", 6, BLUE), ("sales", 7, CARD), ("[SEP]", 6, TRACK),
              ("My order arrived cracked...", 22, CARD)]
    x, opt_x = 1, []
    for label, w, fc in tokens:
        box(x, 3, w - 0.8, 6, label, fc, size=8.5, bold=fc == BLUE)
        if fc == BLUE:
            opt_x.append(x + (w - 0.8) / 2)
        x += w
    ax.text(1, 0.4, "control tokens", color=MUTED, fontsize=8.5)
    ax.text(15, 0.4, "question", color=MUTED, fontsize=8.5)
    ax.text(30, 0.4, "your options, as plain strings (2 to 255)", color=MUTED, fontsize=8.5)
    ax.text(77, 0.4, "state: any text or JSON", color=MUTED, fontsize=8.5)
    box(1, 16, 98, 9, "bidirectional encoder (ModernBERT-base, 150M parameters): every token reads every other token, once", CARD, size=10.5, bold=True)
    for cx in [4, 20, 34, 48, 62, 76, 90]:
        ax.annotate("", (cx, 15.6), (cx, 9.6), arrowprops=dict(arrowstyle="-|>", color=TRACK, lw=1.2))
    for cx, p, name in zip(opt_x, (0.82, 0.13, 0.05), ("billing", "tech support", "sales")):
        ax.annotate("", (cx, 30.5), (cx, 25.4), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6))
        ax.add_patch(FancyBboxPatch((cx - 5, 32), 10, 2.2, boxstyle="round,pad=0,rounding_size=0.6", fc=TRACK, ec="none"))
        ax.add_patch(FancyBboxPatch((cx - 5, 32), max(10 * p, 1.3), 2.2, boxstyle="round,pad=0,rounding_size=0.6", fc=BLUE, ec="none"))
        ax.text(cx, 36, f"{name}  {p:.0%}", color=TEXT, fontsize=9.5, ha="center", fontweight="bold")
    ax.text(opt_x[1], 42.5, "softmax over the [OPT] positions gives calibrated probabilities", color=TEXT2, fontsize=10, ha="center")
    ax.text(88, 34.5, 'value = "billing"\nconfidence = 0.82', color=TEXT, fontsize=10.5, ha="center", va="center", family="DejaVu Sans Mono")
    ax.annotate("", (79.5, 34.5), (opt_x[2] + 7, 34.5), arrowprops=dict(arrowstyle="-|>", color=TEXT2, lw=1.4))
    ax.text(88, 28.5, "illustrative numbers", color=MUTED, fontsize=8.5, ha="center")
    fig.savefig(ASSETS / "architecture.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    make_latency()
    make_calibration()
    make_selective()
    make_grounding()
    make_architecture()
    make_hero()
    make_routing()
