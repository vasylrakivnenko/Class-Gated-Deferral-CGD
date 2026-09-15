"""The deliverable: one cost-vs-accuracy chart that does not overstate its evidence.

Design decisions, and why each one is not negotiable:

**Log x-axis.** Candidates span roughly four orders of magnitude in cost. On a
linear axis every cheap option collapses onto the y-axis and the chart shows one
expensive dot and a smear -- the exact picture that makes "75x cheaper" look like
a rounding error.

**Error bars, always.** Each point carries its 95% Wilson interval. At n=250 the
half-width is about +/-4.5 points, so two models inside ~9 points of each other
are not distinguishable by this chart, and the reader can see that directly
rather than being told it in a footnote nobody reads.

**Three colors, by method family, not by model.** Scatter is an all-pairs form,
where only three categorical hues clear the colorblind-separation floor. Grouping
by family (prompted LLM / classical baseline / trained encoder) fits that budget
and matches the actual question -- "do I need an LLM at all?" -- better than
giving each model its own hue.

**The majority-class line.** A horizontal rule at the accuracy of always guessing
the most common label. Any model below it loses to a constant, which is the
single most useful thing a reader can learn in one glance.

**The accuracy bar.** The product's real question is not "which model is best" but
"which is the cheapest model that still clears my bar." The bar is drawn where the
user sets it, and points below it are drawn hollow -- they are disqualified, not
merely lower.

**Leader lines, not smaller type.** The story rows cluster: eight of PhraseBank's
land between 92% and 99%, and three genuinely-free rows share one x position. A
label is therefore placed by searching for a box that covers no mark, no Wilson
interval and no other label -- out to a few hundred pixels, into the empty band
below a cluster if that is where the room is -- and a thin leader line is drawn
as soon as it sits far enough away that adjacency no longer names its dot. The
alternative is shrinking the type until it technically fits, which trades a
chart whose labels collide for one whose labels cannot be read at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Validated 3-slot categorical palette (all-pairs, light surface).
# validate_palette.js: worst CVD dE 9.2, worst normal-vision dE 24.0 -> PASS.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8a87"
GRID = "#e4e4e0"

FAMILY_COLORS = {
    "Prompted LLM": "#2a78d6",       # slot 1 blue
    "Classical baseline": "#1baf7a",  # slot 3 aqua
    "Trained encoder": "#eb6834",     # slot 2 orange
}
# Aqua sits below 3:1 on this surface, so the relief rule applies: every point
# carries a visible direct label (there are ~10, and each one needs naming anyway).

MARK_S = 110      # scatter area in pt^2 for a candidate's mark
RING_S = 620      # ... and for the "GEPA ran but kept the prompt" ring
LABEL_FS = 9.5    # label type size
LABEL_PAD = 0.25  # label box padding, in multiples of LABEL_FS (a boxstyle unit)
LEADER = "#a8a8a3"


def _money(v: float, floor: float) -> str:
    """Format a decade tick with enough precision to stay distinct.

    A fixed 3-decimal format renders every sub-cent decade as "$0.000", so the
    axis repeats one label and the reader cannot tell two orders of magnitude
    apart -- on a chart whose entire point is orders of magnitude.
    """
    if v <= floor * 1.5:
        return "free"
    if v >= 1:
        return f"${v:,.0f}"
    decimals = max(2, int(np.ceil(-np.log10(v))) + 1)
    return f"${v:.{decimals}f}".rstrip("0").rstrip(".") or "$0"


@dataclass
class ChartRow:
    label: str
    family: str
    # Cost per 1,000 *classifications*, which is what the x axis claims and
    # therefore the only thing that may be plotted on it. Not cost per billed
    # LM call: ChatAdapter retries an item through JSONAdapter when its own
    # format fails to parse, so one classification can be two billed calls --
    # up to 1.99x on rows measured here. Callers pass EvalResult's
    # cost_per_1k_items, never cost_per_1k_calls.
    cost_per_1k: float
    accuracy: float
    ci_lo: float
    ci_hi: float
    annotate: str = ""       # optional extra note, e.g. "after GEPA"
    linked_from: tuple[float, float] | None = None  # draw a before->after arrow; (cost, accuracy) of the tail
    # What KIND of number cost_per_1k is. The chart plots three incompatible
    # conventions on one axis and used to caption all of them "Locally-run rows
    # are $0 marginal", which was false of six rows it was plotting at
    # $0.003-$0.061: local *generative* rows are priced at a hosted proxy rate,
    # while local classical/encoder rows really are $0. A reader cannot tell a
    # measured bill from a proxy estimate by looking at a dot, so the row has to
    # say which it is and the caption has to be derived from that.
    #   "measured" -- provider's own counters x that provider's published rate
    #   "proxy"    -- runs locally; priced at some OTHER host's rate card
    #   "free"     -- runs locally, $0 marginal, genuinely nothing billed
    cost_basis: str = "measured"
    always_label: bool = False  # baselines/encoder/reference rows: label regardless of density
    # Did GEPA actually rewrite this row's instruction? None = not known
    # (no optimization record), which falls back to the weaker numeric test.
    prompt_changed: bool | None = None


def cost_footnote(rows: list[ChartRow], prices_as_of: str) -> str:
    """Build the cost caption FROM the rows, rather than restating it by hand.

    This function exists because the caption was hand-copied into five call
    sites (rechart.py, add_model.py, merge_gepa.py, rerun_rows.py,
    experiment.py) and every copy claimed "Locally-run rows are $0 marginal".
    That was true of the classical and encoder rows and false of six local
    generative rows the same charts plotted at $0.003 to $0.061 -- a caption
    asserting a price the chart visibly contradicts. Deriving it removes the
    only way that can happen again.
    """
    n_proxy = sum(1 for r in rows if r.cost_basis == "proxy")
    n_free = sum(1 for r in rows if r.cost_basis == "free")
    # Kept to two wrapped lines: three lines collide with the x-axis label.
    parts = ["Cost = measured tokens x published rate, per classification; a retried item "
             "is billed for every call it took."]
    if n_free:
        parts.append(f"{n_free} local row(s) at $0 marginal.")
    if n_proxy:
        parts.append(f"{n_proxy} local row(s) plotted at a HOSTED PROXY rate -- an estimate, "
                     "not a bill; the frontier moves if that rate card does.")
    parts.append(f"Prices as of {prices_as_of}.")
    return " ".join(parts)


def plot_cost_vs_accuracy(rows: list[ChartRow], out_path: str,
                          majority_baseline: float | None = None,
                          accuracy_bar: float | None = None,
                          title: str = "Cheapest model that clears the bar",
                          subtitle: str = "",
                          footnote: str = "") -> str:
    # Validated BEFORE the figure exists. The check used to sit after
    # plt.subplots() and raise, and the raise skipped the plt.close(fig) at the
    # bottom -- so every refused call leaked a figure into pyplot's global
    # manager (verified: three refused calls, three figures still open), which
    # the scripts that chart in a loop do repeatedly.
    #
    # An unknown price must not reach the cost axis. models.UNKNOWN is -1.0 and
    # a NaN can arrive from arithmetic on it; both used to sail through the
    # `> 1e-9` test below and get drawn at `floor`, i.e. rendered as FREE -- the
    # most flattering position on the chart, for a row whose price nobody could
    # verify. Fail loudly instead: excluding the row is the caller's decision to
    # make explicitly, not something the plotter should do quietly.
    unpriced = [r.label for r in rows
                if r.cost_per_1k is None or np.isnan(r.cost_per_1k) or r.cost_per_1k < 0]
    if unpriced:
        raise ValueError(
            f"refusing to plot {len(unpriced)} row(s) with an unknown or negative "
            f"cost_per_1k: {unpriced}. An unverified price drawn at the axis floor reads "
            f"as free. Exclude these rows or supply a measured price.")

    # Label room is bought in pixels, and at a fixed 9.5pt label size every
    # pixel added to the figure is one the placement search can spend without
    # touching the type size. Wider does double duty on a log x-axis: it also
    # pulls apart the PhraseBank cluster, which spans half a decade of cost.
    fig, ax = plt.subplots(figsize=(13.4, 8.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Cost of zero cannot be plotted on a log axis. Free rows are placed at a
    # floor and labelled as such rather than silently dropped or nudged.
    #
    # The 1e-9 epsilon guards against a real bug this chart already hit once:
    # a "free" row computed from a large-but-finite throughput proxy (e.g.
    # 1e9 items/sec) can multiply out to a technically-positive float like
    # 1.86e-11 instead of exact 0.0. Treating anything below the epsilon as
    # free keeps one units artifact from setting a floor twelve more orders
    # of magnitude smaller and making every axis tick unreadable.
    # (The unpriced-row refusal that used to live here now runs before the
    # figure is created -- see the top of this function.)
    positive = [r.cost_per_1k for r in rows if r.cost_per_1k > 1e-9]
    floor = (min(positive) / 12) if positive else 1e-5

    def x_of(row: ChartRow) -> float:
        return row.cost_per_1k if row.cost_per_1k > 1e-9 else floor

    # ── reference rules (solid hairlines, never dashed) ────────────────────
    # Both captions are kept: the label pass treats them as keep-outs, so a
    # point label can no longer be laid over the one text that explains what
    # the line under it means.
    keepouts = []
    if majority_baseline is not None:
        ax.axhline(majority_baseline, color="#b9b9b3", linewidth=1.0, zorder=1)
        keepouts.append(ax.text(
            0.994, majority_baseline - 0.016,
            f"always guess the majority class — {majority_baseline:.1%}",
            transform=ax.get_yaxis_transform(), ha="right", va="top",
            fontsize=9, color=TEXT_SECONDARY, zorder=7))

    if accuracy_bar is not None:
        ax.axhline(accuracy_bar, color=TEXT_PRIMARY, linewidth=1.2, zorder=2)
        # Placed just past the right spine (axes-fraction x > 1, data-space y),
        # not inside the plotted area at all. An in-plot label at bar height
        # will eventually sit under SOME cluster of points on a chart whose
        # whole point is that candidates land at many different accuracies --
        # right-aligned-inside collided with TF-IDF here, then with Claude
        # Opus once that label's own collision offset moved it. Outside the
        # data area, it structurally cannot compete with a point label again.
        keepouts.append(ax.annotate(
            f"{accuracy_bar:.0%} bar", xy=(1.0, accuracy_bar),
            xycoords=("axes fraction", "data"), xytext=(6, 0),
            textcoords="offset points", ha="left", va="center",
            fontsize=9.5, color=TEXT_PRIMARY, zorder=7,
            annotation_clip=False))

    # ── before -> after connectors, drawn under the marks ──────────────────
    # An arrow is a claim that GEPA moved this row, so it is drawn only when
    # GEPA actually rewrote the prompt. A same-prompt re-run still moves both
    # coordinates (cache warmth, adapter retries) and got an arrow here once:
    # banking77 deepseek-v4-flash read as 76.5% -> 78.1% off a byte-identical
    # instruction. Those rows get a dashed ring -- "evaluated, unmoved" --
    # instead, matching the interactive chart's vocabulary.
    connectors = []  # (tail, head) in data coords; a label must not bury one
    for r in rows:
        if not r.linked_from:
            continue
        if not _shows_change(r):
            ax.scatter([x_of(r)], [r.accuracy], s=RING_S, zorder=3,
                       facecolors="none", edgecolors="#9aa7b4",
                       linewidths=1.4, linestyle=(0, (3, 3)))
            continue
        fx, fy = r.linked_from
        fx = fx if fx > 0 else floor
        connectors.append(((fx, fy), (x_of(r), r.accuracy)))
        ax.annotate("", xy=(x_of(r), r.accuracy), xytext=(fx, fy),
                    arrowprops=dict(arrowstyle="-|>", color="#9aa7b4",
                                    linewidth=1.4, shrinkA=6, shrinkB=8),
                    zorder=3)

    # ── points with Wilson intervals ───────────────────────────────────────
    for r in rows:
        colour = FAMILY_COLORS.get(r.family, "#666666")
        x = x_of(r)
        clears = accuracy_bar is None or r.accuracy >= accuracy_bar
        ax.errorbar(x, r.accuracy,
                    yerr=[[r.accuracy - r.ci_lo], [r.ci_hi - r.accuracy]],
                    fmt="none", ecolor=colour, elinewidth=1.4, capsize=3,
                    capthick=1.4, alpha=0.75, zorder=4)
        # Hollow marks are disqualified: below the bar the user set.
        ax.scatter([x], [r.accuracy], s=MARK_S, zorder=5,
                   facecolors=colour if clears else SURFACE,
                   edgecolors=colour if clears else colour,
                   linewidths=2.0 if clears else 1.8)
        # 2px surface ring so overlapping marks stay separable.
        ax.scatter([x], [r.accuracy], s=MARK_S, zorder=4.5,
                   facecolors="none", edgecolors=SURFACE, linewidths=2.0)

    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1,000 classifications  (USD, log scale — measured tokens × published "
                  "rate, billed adapter retries included)",
                  fontsize=10.5, color=TEXT_SECONDARY, labelpad=10)
    ax.set_ylabel("Accuracy on held-out test set  (95% Wilson interval)",
                  fontsize=10.5, color=TEXT_SECONDARY, labelpad=10)

    ax.grid(True, which="major", color=GRID, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9.5, length=0)

    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.xaxis.set_major_formatter(lambda v, _: _money(v, floor))
    # The locally-run rows genuinely cost $0, and zero has no place on a log
    # axis, so they are drawn at `floor`. Decade ticks never land there, which
    # left three free baselines sitting at what reads as a real price. Add an
    # explicit tick at the floor; the formatter already labels it "free".
    if any(r.cost_per_1k <= 1e-9 for r in rows):
        # `set_xticks` FIXES the tick list, and Axis.set_ticks widens the view
        # to contain every tick it is given. `get_xticks` on a log axis returns
        # the locator's full decade list, which runs past the data, so feeding
        # it back pushed the right edge up to the next decade above the priciest
        # row (measured: 29.2 -> 100.0 with a $20 maximum) and left an empty
        # decade of whitespace. Capture the data-driven right edge first and
        # restore it, or the log axis stops showing the 75x spread it exists
        # to show.
        right = ax.get_xlim()[1]
        ax.set_xticks(list(ax.get_xticks()) + [floor], minor=False)
        ax.set_xlim(floor / 2.2, right)

    families = [f for f in FAMILY_COLORS if any(r.family == f for r in rows)]
    handles = [Line2D([0], [0], marker="o", color="none", label=f,
                      markerfacecolor=FAMILY_COLORS[f], markersize=9,
                      markeredgecolor=SURFACE, markeredgewidth=1.5)
               for f in families]
    if accuracy_bar is not None:
        handles.append(Line2D([0], [0], marker="o", color="none",
                              label="below the bar (disqualified)",
                              markerfacecolor=SURFACE, markersize=9,
                              markeredgecolor="#8a8a87", markeredgewidth=1.6))
    # The dotted-circle glyph, not a hollow "o": a solid-edged ring is already
    # spoken for above (below-the-bar), and two rings that differ only in
    # stroke pattern are not separable at legend size.
    if any(r.linked_from and not _shows_change(r) for r in rows):
        handles.append(Line2D([0], [0], marker="$\u25cc$", color="#9aa7b4",
                              linestyle="none", markersize=12,
                              label="GEPA ran but kept the baseline prompt"))
    # Lifted off the corner rather than loc="lower left" alone: the legend has
    # grown to five entries and the bottom one lands on the majority-class rule,
    # which sits within a couple of percent of zero on a many-class task.
    legend = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, 0.045),
                       frameon=False, fontsize=10, labelcolor=TEXT_SECONDARY,
                       handletextpad=0.6)
    legend.set_zorder(6)

    fig.suptitle(title, x=0.055, y=0.975, ha="left", fontsize=17,
                 color=TEXT_PRIMARY, fontweight="semibold")
    if subtitle:
        fig.text(0.055, 0.925, subtitle, ha="left", fontsize=11, color=TEXT_SECONDARY)
    if footnote:
        # Wrapped, not drawn as one line. savefig uses bbox_inches="tight", so a
        # caption wider than the figure silently EXPANDS the saved image instead
        # of overflowing -- a longer footnote took the PNG from 2760px to 4519px
        # wide and squashed the plot. Wrapping keeps the caption a caption.
        import textwrap
        wrapped = "\n".join(textwrap.wrap(footnote, width=175,
                                          break_long_words=False, break_on_hyphens=False))
        fig.text(0.055, 0.004, wrapped, ha="left", va="bottom",
                 fontsize=8.5, color="#8a8a87", linespacing=1.5)

    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.905))

    # Direct labels (required: relief rule for the aqua slot), placed only
    # now: collision-checking uses transData.transform() to get real screen
    # pixels, which must reflect this axes' FINAL bbox within the figure.
    # tight_layout can still move that bbox, so placing labels any earlier
    # measures distances that are wrong by however much layout later shifts
    # things -- small in practice, but this makes "final" mean final.
    fig.canvas.draw()
    _place_labels(fig, ax, rows, x_of, keepouts=keepouts, connectors=connectors,
                  rules=[y for y in (majority_baseline, accuracy_bar) if y is not None])

    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return out_path


def _shows_change(r: "ChartRow") -> bool:
    """Did GEPA actually rewrite this row's prompt?

    `prompt_changed` is the real answer and comes from the optimization
    record. When a caller cannot supply it -- payloads written before that
    field existed -- fall back to asking whether the numbers moved. That is
    the weaker test (a same-prompt replicate moves them too), but it asserts
    less than pretending an unrecorded run definitely changed something.
    """
    if r.prompt_changed is not None:
        return r.prompt_changed
    if r.linked_from is None:
        return False
    fx, fy = r.linked_from
    return abs(r.accuracy - fy) > 1e-9 or abs(r.cost_per_1k - fx) > 1e-9


def _label_text(r: "ChartRow") -> str:
    """One label per story point, not one per dot.

    A GEPA pair (a row with `linked_from` set) collapses to a single
    "Model: before% -> after%" label anchored at the optimized point, rather
    than two separate labels at both ends of the arrow -- the arrow already
    shows direction, so a second label would just repeat the same story twice
    in the crowded region where it is least affordable.
    """
    base = r.label.replace(" + GEPA", "")
    if r.linked_from is not None:
        before = r.linked_from[1]
        if not _shows_change(r):
            # Same prompt scored twice. The gap between the two points is
            # measurement noise, so quoting it as "before -> after" would
            # hand GEPA credit for a difference it did not cause.
            return f"{base}: GEPA found no improvement ({r.accuracy:.1%})"
        return f"{base}: {before:.1%} → {r.accuracy:.1%}"
    return f"{base}  ({r.annotate})" if r.annotate else base


def _priced_frontier(rows: list[ChartRow]) -> set[int]:
    """Indices of rows on the Pareto frontier *among rows that cost money*.

    The free baselines are excluded from the domination test on purpose. On a
    many-class task TF-IDF sits at $0 and 92%, which dominates every priced row
    on both axes -- so a frontier computed over all rows is just the free
    baselines, and every paid model becomes an unlabelled dot. That is the
    chart's headline finding, not a labelling rule: the reader still needs to
    know *which* paid model is which, because "if you are going to pay anyway,
    what is the efficient choice" is the question the product answers.

    This exists because the original rule assumed every prompted LLM carried a
    GEPA pair and would be named through it (see experiment.py's note on
    ALWAYS_LABEL_KEYS). Once rows were added without running GEPA on them, five
    of banking77's twelve LLM rows -- including both Claudes and the best cheap
    model on the board -- were plotted with no name attached to them.
    """
    priced = [(i, r) for i, r in enumerate(rows) if r.cost_per_1k > 1e-9]
    keep = set()
    for i, r in priced:
        if not any(q.cost_per_1k <= r.cost_per_1k and q.accuracy > r.accuracy
                   for j, q in priced if j != i):
            keep.add(i)
    return keep


def _labelled_rows(rows: list[ChartRow]) -> list[ChartRow]:
    """The rows that carry a claim, with a pair's tail dropped once.

    `always_label` marks the baselines, the encoder and the frontier reference;
    a row with `linked_from` is a GEPA outcome. Those two sets can overlap on
    one row, and it matters: when the reference model is also the tail of a
    GEPA pair, its bare label ("Claude Sonnet 4.6") and the pair's collapsed
    label ("Claude Sonnet 4.6: GEPA found no improvement (95.6%)") name the
    same model, and when GEPA changed nothing the two marks are at *identical*
    coordinates -- so the second label is not merely redundant, it is two names
    hanging off one dot with nothing to say which is which. The pair label
    already quotes the reference's own accuracy, so dropping the bare one
    withholds nothing (this is the same reasoning as experiment.py's note that
    ALWAYS_LABEL_KEYS need not list the prompted LLMs).

    A third case was added once that assumption broke: rows on the frontier
    among priced models. See `_priced_frontier`.
    """
    frontier = _priced_frontier(rows)
    keep = []
    for i, r in enumerate(rows):
        if r.linked_from is not None or (r.always_label and not _named_by_pair(r, rows)):
            keep.append(r)
        elif i in frontier and not _named_by_pair(r, rows):
            # Same guard as the always_label branch, and for the same reason: a
            # frontier row is very often the BASE of a GEPA pair, whose single
            # collapsed label ("DeepSeek-V4-Flash: 95.6% -> 97.6%") already
            # names it and already quotes its accuracy. Without this, four rows
            # on Financial PhraseBank got two labels each, naming one dot twice.
            keep.append(r)
    return keep


def _named_by_pair(row: ChartRow, rows: list[ChartRow]) -> bool:
    """Is `row` the tail of a GEPA pair whose one label already names it?

    Matched on (name, cost, accuracy) rather than a key, because ChartRow
    carries no key: `linked_from` is built from the base row's own plotted
    coordinates, so the two agree exactly rather than approximately.
    """
    for p in rows:
        if p is row or p.linked_from is None:
            continue
        if p.label.replace(" + GEPA", "") != row.label:
            continue
        fx, fy = p.linked_from
        if abs(fx - row.cost_per_1k) < 1e-12 and abs(fy - row.accuracy) < 1e-12:
            return True
    return False


# Candidate directions in preference order, as angles in degrees: straight up
# first (the conventional spot for a scatter label), then the upper diagonals,
# then sideways, then below. Fourteen rather than the eight compass points
# because the angle a leader arrives at is what tells two clustered labels
# apart -- with a coarse set, neighbours in a cluster get near-parallel leaders
# and the reader cannot see which dot each one claims. Order is only a
# tiebreak: a farther candidate that covers nothing beats a near one that
# collides, which is the point of scoring every candidate rather than taking
# the first that fits.
_ANGLES = (90, 68, 112, 45, 135, 25, 155, 0, 180, -25, -155, -50, -130, -90)
_DIRS = tuple((float(np.cos(np.radians(a))), float(np.sin(np.radians(a))))
              for a in _ANGLES)

# Offsets to try, in device pixels. The long end is what makes this work: the
# nearest free space for PhraseBank's top-right cluster is the empty band a
# couple of hundred pixels *below* it, and a search that stops at nudging
# distance has nothing to offer a label there except somebody else's text.
_RADII = (16, 30, 52, 84, 124, 172, 228, 300, 380, 470)

_LEADER_MIN = 34.0  # px; past this a label is no longer visibly adjacent
_CLUMP_PX = 46.0    # px between mark centres; below this, adjacency proves nothing
_GRAZE_PX = 12.0    # px of clearance a leader must keep from a mark it is not for
_NEAR_X, _NEAR_Y = 320.0, 110.0  # window that counts as "the same cluster"
_MIN_FAN = 22.0     # degrees; two leaders out of one cluster must diverge by this


def _radius_pt(area: float) -> float:
    """Marker radius in points, from a scatter `s=` area in points^2."""
    return float(np.sqrt(area / np.pi))


def _text_size_px(fig, text: str) -> tuple[float, float]:
    """Rendered size of a label, in device pixels.

    Measured, not estimated: the previous pass hard-coded "~170-220px" for the
    longest label and thresholded on that, but the real spread is 330px for
    "GPT-5-nano: 94.8% → 98.8%" against 690px for "Claude Sonnet 4.6: GEPA
    found no improvement (95.6%)" -- and it is the long ones that collide.
    """
    probe = fig.text(0, 0, text, fontsize=LABEL_FS)
    box = probe.get_window_extent(fig.canvas.get_renderer())
    probe.remove()
    return box.width, box.height


def _label_box(x: float, y: float, w: float, h: float,
               ha: str, va: str, pad: float) -> tuple[float, float, float, float]:
    """The opaque box a label paints, given the anchor point it is aligned to."""
    x0 = x - (0.0 if ha == "left" else w if ha == "right" else w / 2)
    y0 = y - (0.0 if va == "bottom" else h if va == "top" else h / 2)
    return (x0 - pad, y0 - pad, x0 + w + pad, y0 + h + pad)


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _within(box, area) -> bool:
    return (box[0] >= area[0] and box[1] >= area[1]
            and box[2] <= area[2] and box[3] <= area[3])


def _crossings(p0, p1, boxes) -> int:
    """How many of `boxes` a straight line from p0 to p1 passes through.

    Sampled, not clipped: 24 steps along the segment is more than enough to
    catch a leader that would appear to point at the wrong dot, and exact
    segment/rectangle clipping is a page of code for a soft scoring term.
    """
    hits = 0
    for b in boxes:
        for i in range(25):
            t = i / 24.0
            x, y = p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t
            if b[0] < x < b[2] and b[1] < y < b[3]:
                hits += 1
                break
    return hits


def _gutter_slot(px, py, w, h, pad, area, blocked, placed):
    """Last resort: a column just outside the right spine, nearest slot to py.

    Never reached on either run in `runs/`. It exists so that a denser future
    run degrades into long leader lines into reserved margin -- which stays
    readable -- rather than back into overlapping text, and so the search has
    no failure mode that silently paints one label over another.
    """
    gx, step, best = area[2] + 22, h + 9, None
    y = area[3] - h / 2
    while y > area[1]:
        box = _label_box(gx, y, w, h, "left", "center", pad)
        if not any(_overlaps(box, b) for b in blocked) and \
           not any(_overlaps(box, b) for b in placed):
            if best is None or abs(y - py) < best[0]:
                best = (abs(y - py), gx - px, y - py, "left", "center", box, 0)
        y -= step
    if best is None:
        return (0.0, gx - px, 0.0, "left", "center",
                _label_box(gx, py, w, h, "left", "center", pad), 0)
    return best


def _place_labels(fig, ax, rows: list[ChartRow], x_of,
                  keepouts=(), rules=(), connectors=()) -> None:
    """Label the story rows, each in the nearest box that covers nothing.

    At double-digit point counts, one label per dot is unreadable regardless of
    collision-avoidance cleverness -- the fix the dataviz anti-patterns call out
    is to label the story, not the population. Everything else stays a plain,
    correctly-colored, correctly-positioned dot: its exact number is in the
    console table and results.json, and the chart's job is the shape of the
    tradeoff, not a lookup table rendered as text.

    That still leaves clusters. PhraseBank puts eight story rows between 92%
    and 99%, and three genuinely-free rows share one x position two accuracy
    points apart. The pass this replaces could only nudge -- a fan of fixed
    offsets -- so in a neighbourhood with no free space within nudging distance
    it stacked text on text, where the later label's opaque box erased the
    earlier one's ascenders. ("TF-IDF + logistic regression" lost its top half
    to the box behind "Fine-tuned encoder (ettin-encoder-68m)" and read as
    struck through by the 100% gridline.) Four things fix it properly:

    * search instead of nudge -- fourteen directions x ten offsets out to
      470px, scoring every candidate whose measured box covers no mark, no
      Wilson interval, no caption and no already-placed label, so a label can
      walk out to the empty band below a cluster instead of jostling inside it;
    * a leader line as soon as the label leaves its dot's neighbourhood,
      because past that distance position alone no longer says which dot a name
      belongs to -- and always, for a mark with a neighbour within _CLUMP_PX,
      where position never said it in the first place;
    * leaders that keep clear of marks they are not for and that fan out: two
      labels from one cluster may not leave it at the same angle, because two
      near-parallel lines into a knot of overlapping dots are unreadable even
      when each one is geometrically exact; and
    * order consistency inside a cluster: a label may not sit above a label
      whose dot is above its own. That is what makes a stack of four readable
      when the dots they point into are 20px apart -- the reader maps them by
      order, and no single leader tip has to carry the whole burden.

    Everything is in device pixels, measured after the layout is final (see the
    fig.canvas.draw() at the callsite). Data units are the wrong unit: on a log
    axis spanning ~5 decades, two points "1.09 log10-dollars apart" read as far
    apart in the abstract and still overlap on screen.
    """
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    pad = LABEL_PAD * LABEL_FS * px_per_pt
    to_px = ax.transData.transform

    # A label may cover gridlines -- that is what its opaque box is for -- but
    # never a mark or its interval. Buying room for a name by hiding the
    # measurement it names is not a legibility fix. One box per row covers the
    # mark, its error bar and (where drawn) the much larger unchanged-prompt
    # ring, which is 78px across and would swallow a nearby label whole.
    centres, marks, radii_pt = [], [], []
    for r in rows:
        cx, cy = to_px((x_of(r), r.accuracy))
        lo = to_px((x_of(r), r.ci_lo))[1]
        hi = to_px((x_of(r), r.ci_hi))[1]
        ringed = r.linked_from is not None and not _shows_change(r)
        rad_pt = _radius_pt(RING_S if ringed else MARK_S)
        rad = rad_pt * px_per_pt
        centres.append((cx, cy))
        radii_pt.append(rad_pt)
        marks.append((cx - rad - 2, min(lo, cy) - rad, cx + rad + 2, max(hi, cy) + rad))
    blocked = list(marks) + [
        (b.x0 - 5, b.y0 - 4, b.x1 + 5, b.y1 + 4)
        for b in (a.get_window_extent(renderer) for a in keepouts)]
    # For the leader test, only the mark's disc counts, not its whole interval:
    # what makes a leader ambiguous is passing near a *dot* it does not name.
    # A whisker is thin and is drawn over the leader anyway, so a line crossing
    # one reads as passing behind it, while inflating the full interval span
    # here walls off the only routes out of a cluster.
    graze = [(cx - rad - _GRAZE_PX, cy - rad - _GRAZE_PX,
              cx + rad + _GRAZE_PX, cy + rad + _GRAZE_PX)
             for (cx, cy), rad in zip(centres, [p * px_per_pt for p in radii_pt])]

    rule_y = [to_px((1.0, y))[1] for y in rules]
    conn_px = [(to_px(a), to_px(b)) for a, b in connectors]

    # Labels stay inside the data area. The run-off in the chart this replaces
    # was labels sailing over the top spine into the subtitle, where a reader
    # cannot tell whether they belong to the plot or to the header.
    area = (ax.bbox.x0 + 3, ax.bbox.y0 + 3, ax.bbox.x1 - 3, ax.bbox.y1 - 3)

    todo = []
    for r in _labelled_rows(rows):
        own = next(i for i, q in enumerate(rows) if q is r)
        w, h = _text_size_px(fig, _label_text(r))
        todo.append((r, centres[own], w, h, own))
    # Highest dot first, always. A cluster's labels end up stacked -- there is
    # no other way to fit five 350px names into 165px of x -- and a stack is
    # only readable if it runs in the same order as the dots it points into, so
    # the stack has to be built from the top down. Going in any other order
    # lets a lower dot take the slot the top one needed and forces an
    # order-crossing pair, which is the failure that makes a whole stack
    # unreadable rather than just one line of it.
    order = sorted(range(len(todo)), key=lambda i: (-todo[i][1][1], todo[i][1][0]))

    placed = []  # (box, anchor_y, label_centre_y, anchor_x, leader_angle)
    for i in order:
        r, (px, py), w, h, own = todo[i]
        # A mark with a neighbour this close cannot be claimed by adjacency:
        # "the label beside the dot" means nothing when three dots share one x
        # position and two accuracy points. Push those labels out past their
        # own mark by enough that a leader line has room to be seen -- measured
        # from the mark's own radius, because the unchanged-prompt ring's is
        # 39px and a 52px offset would leave the leader hidden underneath it.
        clump = [j for j, (cx, cy) in enumerate(centres)
                 if j != own and (cx - px) ** 2 + (cy - py) ** 2 < _CLUMP_PX ** 2]
        own_r = radii_pt[own] * px_per_pt
        radii = tuple(rad for rad in _RADII if not clump or rad >= own_r + 36)
        # ... and those same neighbours are left out of the leader clearance
        # test: their discs overlap this one, so no exit angle avoids them and
        # scoring one would only distort the choice between the angles that are
        # actually different. In a clump the fan and order rules carry the
        # attachment, not the leader's first few pixels.
        others = [b for j, b in enumerate(graze) if j != own and j not in clump]
        near = [p for p in placed
                if abs(p[3] - px) < _NEAR_X and abs(p[1] - py) < _NEAR_Y]

        best = None
        for rank, (ux, uy) in enumerate(_DIRS):
            # Align the text away from the point, so the box grows outward from
            # the anchor rather than back across the mark.
            ha = "left" if ux > 0.3 else "right" if ux < -0.3 else "center"
            va = "bottom" if uy > 0.3 else "top" if uy < -0.3 else "center"
            for rad in radii:
                dx, dy = ux * rad, uy * rad
                box = _label_box(px + dx, py + dy, w, h, ha, va, pad)
                if not _within(box, area):
                    continue
                if any(_overlaps(box, b) for b in blocked):
                    continue
                # Against every already-placed label, not just the ones next
                # to it in this sort order. Two GEPA outcomes once landed
                # within 1 accuracy point and 6% cost of each other with an
                # unrelated third point's accuracy between them, so a
                # predecessor-only check compared each of the close pair to
                # that stranger, correctly saw no conflict, and let both pick
                # the same offset -- stacking their text exactly.
                if any(_overlaps(box, p[0]) for p in placed):
                    continue
                # The leader is drawn from the rim, not the centre (that is
                # what shrinkB does below), so the rim is where the test has to
                # start -- from the centre, every candidate of a clumped row
                # "crosses" the neighbour whose disc already overlaps its own,
                # and the term stops discriminating exactly where it is needed.
                rim = (px + ux * own_r, py + uy * own_r)
                tip = (px + dx, py + dy)
                mid = (box[1] + box[3]) / 2
                # Soft costs, in the same units as distance so they trade off
                # against it directly: sitting on a reference rule breaks the
                # one hairline the reader is measuring against; a leader that
                # grazes another mark or crosses another label claims the wrong
                # dot; a label over a connector hides the GEPA arrow itself; and
                # the last two -- order and fan -- are what make a cluster
                # readable at all, so they outrank the rest. Fan beats graze
                # deliberately: a line passing wide of a dot is legible, two
                # near-parallel lines into a knot of dots are not.
                score = (rad + 12 * rank
                         + 150 * sum(1 for ry in rule_y if box[1] < ry < box[3])
                         + 95 * _crossings(rim, tip, others)
                         + 70 * _crossings(rim, tip, [p[0] for p in placed])
                         + 45 * sum(_crossings(a, b, [box]) for a, b in conn_px)
                         + 130 * sum(1 for p in near
                                     if (py - p[1]) * (mid - p[2]) < 0)
                         + 170 * sum(1 for p in near
                                     if abs((_ANGLES[rank] - p[4] + 180) % 360 - 180)
                                     < _MIN_FAN))
                if best is None or score < best[0]:
                    best = (score, dx, dy, ha, va, box, _ANGLES[rank])
        if best is None:
            best = _gutter_slot(px, py, w, h, pad, area, blocked,
                                [p[0] for p in placed])

        _, dx, dy, ha, va, box, angle = best
        if (dx * dx + dy * dy) ** 0.5 > _LEADER_MIN:
            # Hairline, headless, and lighter than the before->after connectors
            # (1.4pt slate with an arrowhead): a leader is typography, not
            # evidence, and must not read as a second kind of GEPA arrow. It
            # stops exactly on its own mark's rim -- the attachment is the
            # whole job -- and is drawn under the marks, so where one is
            # unavoidably crossed the line passes visibly *behind* the dot
            # instead of appearing to end on it.
            ax.annotate("", xy=(x_of(r), r.accuracy), xytext=(dx, dy),
                        textcoords="offset pixels", zorder=3.6,
                        annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=LEADER,
                                        linewidth=0.7, shrinkA=0.0,
                                        shrinkB=radii_pt[own] + 0.5))
        ax.annotate(_label_text(r), xy=(x_of(r), r.accuracy),
                    xytext=(dx, dy), textcoords="offset pixels",
                    ha=ha, va=va, fontsize=LABEL_FS, color=TEXT_PRIMARY,
                    zorder=8, annotation_clip=False,
                    bbox=dict(boxstyle=f"round,pad={LABEL_PAD}", facecolor=SURFACE,
                              edgecolor="none", alpha=1.0))
        placed.append((box, py, (box[1] + box[3]) / 2, px, angle))
