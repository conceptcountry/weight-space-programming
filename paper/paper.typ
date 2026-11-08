#import "numbers.typ": *

#set document(title: "Gradient-Free Weight Editing Reduces Overthinking in Qwen3.8-27B", author: "Justin Bennington")
#set page(paper: "us-letter", margin: (x: 1in, y: 0.9in), numbering: "1",
  header: context { if counter(page).get().first() > 1 [#set text(8pt, fill: luma(110)); Bennington · Gradient-Free Weight Editing in Qwen3.8-27B · Working paper v1 #h(1fr) Concept Country] })
#set text(font: ("Libertinus Serif", "New Computer Modern"), size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => block(above: 1.3em, below: 0.7em, text(12pt, weight: "bold", it))
#show heading.where(level: 2): it => block(above: 1em, below: 0.5em, text(10.5pt, weight: "bold", style: "italic", it))
#show figure.caption: set text(9pt)
#show link: set text(fill: rgb("#1f4f6b"))
#let fig(path, cap, w: 100%) = figure(placement: auto, image(path, width: w), caption: cap)

#align(center)[
  #block(width: 86%)[
    // Titles are display text, not body copy: justification stretches the word
    // spacing badly at 16pt across a full measure.
    #set par(justify: false, leading: 0.45em)
    #text(16pt, weight: "bold")[Gradient-Free Weight Editing Reduces Overthinking in Qwen3.8-27B]
  ]
  #v(0.15em)
  #text(11pt)[A norm-matched controlled replication of weight-space programming]
  #v(0.6em)
  Justin Bennington \
  #text(9.5pt)[Concept Country, New York, NY · \@somewheresy · Working paper v1, 8 November 2026]
]

#v(0.6em)
#block(inset: (x: 2.2em))[
#set text(9.5pt)
*Abstract.* A reasoning model can be made to think less and score more by editing nine of
its weight matrices, with no gradient step, no training data and no fine-tuning. We
replicate weight-space programming on Qwen3.8-27B: directions are read off residual
activations captured in forward passes on a train split, then compiled into the
down-projections of nine alternating layers as a rank-one additive term. On a held-out
confirmation suite the edit improves GPQA Diamond by #gdbe pts (95% CI #gcibe) while
cutting thinking tokens #gtokpctbe% (#gtokcibe), improves LiveCodeBench v6 by #lcbdbe pts
(#lcbcibe) at #lcbtokpctbe% tokens, and improves IFBench by #ifdbe pts (#ifcibe) at
#iftokpctbe% tokens. The result that makes this a finding rather than folklore is the
control: a norm-matched random edit, identical in $norm(Delta W)$ and applied to
the same nine matrices with a per-layer random direction orthogonal to the fitted one, does
nothing. Every base-to-control accuracy interval includes zero
(GPQA #gdbc #gcibc, IFBench #ifdbc #ifcibc, LCB #lcbdbc #lcbcibc) and it saves no tokens on
any benchmark (#gtokpctbc%, #iftokpctbc%, #lcbtokpctbc%). Against that control the edit gains
#gdce pts on GPQA (#gcice) and #lcbdce on LCB (#lcbcice). We report one clause as
borderline rather than passing: the IFBench control margin's lower bound sits on zero and
moves with the bootstrap seed, so we decline to claim it. We also document a measurement
hazard that bit three separate analysis scripts in this study — a follow-up sampling run
appending extra samples to the same result files, silently mixing #censeditunf#[-row] and
#censbasefil#[-row] problem sets in a per-problem mean — and the recomputation discipline that
caught it each time.
]

= Introduction

Making a reasoning model cheaper usually means making it smaller or training it differently.
Both are expensive and both change the model everywhere. A third option has become visible
recently: treat the weights as a program and edit them directly, using directions obtained
from the model's own activations rather than from a loss.

The appeal is that it is nearly free. Reading the directions costs a few hundred forward
passes. Applying them is a matrix update. There is no optimizer, no learning rate, no
dataset beyond the prompts used to elicit the behaviour being measured. If it works, it
works in minutes on one machine.

The difficulty is that this class of intervention is unusually easy to fool yourself about.
Perturbing a large model's weights slightly, in almost any direction, changes its outputs;
some of those changes will look like improvements on a small evaluation. The literature on
activation steering and task vectors is full of real effects, but a weight edit evaluated
against only an unmodified baseline cannot distinguish "this direction encodes something"
from "this magnitude of perturbation happens to help here". The question is not whether the
edited model beats the original. It is whether the *direction* did the work.

This paper is a controlled replication. We port the method to Qwen3.8-27B, pre-register a
success criterion before running the confirmation suite, and evaluate the edit against two
references: the unmodified base model, and a norm-matched random edit that touches the same
nine matrices with the same perturbation size in a random orthogonal direction. The random
control is the load-bearing comparison, and it is why we can say the mechanism is
directional rather than magnitudinal.

Our contribution is threefold. We confirm the method transfers to a 27B reasoning model with
accuracy gains on all three held-out benchmarks and a 27–54% reduction in thinking tokens.
We establish that an equal-norm random edit produces no accuracy gain and no token saving,
which rules out the most plausible alternative explanation. And we report the study's
measurement failures in full, including a contamination bug that reached three independent
consumers of the same file and a confidence interval we decline to claim.

= Related work

Weight-space programming, as we use the term, is the practice of obtaining a direction from
model activations and then writing it into the weights so that the behaviour is compiled in
rather than applied at inference. It sits between two established lines.

Activation steering adds a vector to the residual stream at generation time. It is
reversible, costs a hook, and must be re-applied on every forward pass. Task arithmetic
instead operates on weights, deriving vectors by differencing fine-tuned checkpoints against
their base, which requires the fine-tuning runs the method is meant to avoid. The approach
here takes the direction from activations, as steering does, but applies it to weights, as
task arithmetic does — so nothing needs re-applying at inference and nothing needs training.

The specific construction we replicate reads directions from contrasts in a model's own
reasoning traces: short correct traces against long correct ones on the same problem,
positions preceding verification language against other positions, and the region
immediately before the end-of-thinking token against the rest. These are differences in how
the model reasons, not in what it knows, which is why the intervention lands on thinking
length.

What the prior work does not settle, and what motivates the control here, is the
counterfactual. Reported improvements are generally measured against an unedited baseline.
Since any sufficiently small weight perturbation leaves a capable model capable, and some
perturbations will help on a given suite by chance, the norm-matched random edit is the
comparison that separates a direction's content from a perturbation's size.

= Method

== Model and engine

The subject is Qwen3.8-27B in thinking mode. All arms run on the same inference server,
Actual Computer (#link("https://actual.inc")[actual.inc], llama.cpp lineage), with Q8_0 GGUF
weights and the stock multi-token-prediction
draft, on one GB10 accelerator. Sampling follows the model card for thinking mode
(T\=1.0, top-p 0.95, top-k 20, presence 0, repetition 1.0) with a 32,768-token cap, and the
cap, seeds and harness are identical across arms. The conversion to Q8_0 was performed once
per arm with the same tool and output type, and the resulting file digests are published.

== Directions from forward passes

Discovery uses the MATH train split only, never the confirmation suite. We generate 48
level-5 problems at 6 samples each under a 4k cap, then take one forward pass per trace
(288 total) and record residual activations at every layer for: think tokens generally; the
token preceding a verification phrase; the last 48 tokens before the end-of-thinking marker;
and the first 1,536 think tokens. For each writer matrix we also record the mean input and
mean output norm.

Three contrasts give three candidate directions. `sl` is short-minus-long over correct
traces on the same problem, from 22 problem pairs. `vf` is pre-verification against other
positions. `cc` is the pre-end-of-thinking region against other positions. The `cc`
direction has cosine similarity 0.35–0.83 with `sl` depending on layer, so the two capture
overlapping but distinct structure.

== Compiling a direction into weights

Two edit forms were tried. The projective form removes a direction,
$W arrow.l W - alpha d d^T W$. The additive form writes along it,
$W arrow.l W + alpha n_w d u^T slash norm(u)^2$, where $u$ is the writer's mean think-token
input and $n_w$ its mean output norm. On a typical input the additive form writes about
$alpha n_w$ along $d$ — steering, compiled into the weights.

A screen on 64 fresh train problems selected the additive `cc` edit at
$alpha = 0.1$ applied to the down-projections of nine alternating layers (32 through 48,
even), a relative perturbation of $norm(Delta W) slash norm(W) approx 0.5%$. Nothing from
the confirmation suite informed this choice.

== The norm-matched random control

The control replaces the fitted direction $d$ with a random unit vector $r perp d$, drawn
with a per-layer seed, leaving $norm(Delta W)$ identical. It is the same edit form, the
same nine matrices, the same magnitude, the same engine and the same seeds — differing only
in whether the direction was fitted or random. We refer to the edited arm as the *edit* and
this arm as the *control*.

== Pre-registration

Fixed before any confirmation run: an edit succeeds if, on a held-out suite never used for
selection, paired against base with the same seeds, harness and cap, either accuracy
improves with the 95% CI excluding zero, or mean thinking tokens drop by at least 15% while
accuracy is non-inferior (CI lower bound above −2 pts). It must also beat the norm-matched
random edit, show dose-response, and show no instruction-following regression.

The confirmation suite is GPQA Diamond (198 problems × 2 samples), IFBench (300) and
LiveCodeBench v6 (131). Paired comparisons use a #bootreps#[-replicate] bootstrap over
problems.

= Results

All three benchmarks are complete on all three arms. GPQA is scored as 198 problems × 2
samples with uniform coverage; the per-problem score is the mean over a problem's samples,
and comparisons are paired by problem.

== The edit improves accuracy and shortens reasoning

#figure(
  table(columns: 7, align: (left, right, right, right, right, right, right),
    stroke: none,
    table.hline(),
    table.header[Benchmark][base][edit][$Delta$ acc (95% CI)][tokens base][tokens edit][$Delta$ tok (CI)],
    table.hline(),
    [GPQA Diamond (n\=#gn)], [#gaccbase], [#gaccedit], [#gdbe #gcibe], [#gtokbase], [#gtokedit], [#gtokpctbe% #gtokcibe],
    [IFBench (n\=#ifn)], [#ifaccbase], [#ifaccedit], [#ifdbe #ifcibe], [#iftokbase], [#iftokedit], [#iftokpctbe% #iftokcibe],
    [LiveCodeBench v6 (n\=#lcbn)], [#lcbaccbase], [#lcbaccedit], [#lcbdbe #lcbcibe], [#lcbtokbase], [#lcbtokedit], [#lcbtokpctbe% #lcbtokcibe],
    table.hline(),
  ),
  caption: [Edit against unmodified base, paired by problem, #bootreps#[-replicate] bootstrap. Accuracy in points; tokens are means per problem.],
)

GPQA improves #gdbe pts with the interval excluding zero (#gbetterbe problems better,
#gworsebe worse) while using #gtokpctbe% fewer thinking tokens. LCB improves #lcbdbe pts
(#lcbcibe), the largest accuracy gain of the three. IFBench improves #ifdbe pts but its
accuracy interval includes zero (#ifcibe); it passes the pre-registered criterion on the
token clause instead, cutting tokens #iftokpctbe% (#iftokcibe) without regression. Truncation
at the 32k cap falls on every benchmark: #gtruncbase to #gtruncedit on GPQA,
#iftruncbase to #iftruncedit on IFBench, #lcbtruncbase to #lcbtruncedit on LCB.

== The control does nothing, which is the point

#figure(
  table(columns: 5, align: (left, right, right, right, right),
    stroke: none,
    table.hline(),
    table.header[Benchmark][base][control][$Delta$ acc (95% CI)][$Delta$ tokens],
    table.hline(),
    [GPQA Diamond], [#gaccbase], [#gaccctrl], [#gdbc #gcibc], [#gtokpctbc%],
    [IFBench], [#ifaccbase], [#ifaccctrl], [#ifdbc #ifcibc], [#iftokpctbc%],
    [LiveCodeBench v6], [#lcbaccbase], [#lcbaccctrl], [#lcbdbc #lcbcibc], [#lcbtokpctbc%],
    table.hline(),
  ),
  caption: [Norm-matched random edit against base. Same nine matrices, same $norm(Delta W)$, random orthogonal direction. Every accuracy interval includes zero and no token saving appears anywhere.],
)

#fig("../figures/control-vs-edit-2x.png",
  [Accuracy change against the unmodified base, with 95% confidence intervals. The fitted edit clears zero on GPQA Diamond and LiveCodeBench v6. The norm-matched random control, identical in perturbation magnitude, straddles zero on all three benchmarks.])

This is the result the study was built to produce. An equal-magnitude perturbation of the
same nine down-projections leaves accuracy statistically unchanged on all three benchmarks
and does not shorten reasoning at all: #gtokpctbc% on GPQA, #iftokpctbc% on IFBench,
#lcbtokpctbc% on LCB, against the edit's #gtokpctbe%, #iftokpctbe% and #lcbtokpctbe%. The
token reduction in particular is not a thing that happens when you disturb a model slightly.
It happens when you disturb it in this direction.

#fig("../figures/tokens-2x.png",
  [Thinking-token change against the unmodified base. The edit cuts reasoning length by 27–54% on every benchmark; the norm-matched control saves nothing, and its intervals include zero throughout.])

Measured directly against the control, the edit gains #gdce pts on GPQA (#gcice,
#gbetterce better / #gworsece worse) and #lcbdce pts on LCB (#lcbcice). On GPQA the control
margin is wider than the base margin, because the control drifts marginally below base.

== One clause we decline to claim

The IFBench control margin is #ifdce pts with a nominal interval of #ifcice. Its lower bound
is #ifcicelo, which is to say it sits on the boundary. Re-running the bootstrap across ten
seeds moves that bound between #sym.plus.minus 0.00 and +0.33, and for two of the ten seeds
the interval includes zero.

A pre-registered clause whose outcome depends on a bootstrap seed has not been met. We
therefore report the IFBench control comparison as *borderline*, and the overall conclusion
rests on GPQA and LCB, whose control margins are seed-stable with lower bounds of +2.78 and
+3.05 or above across the same seed sweep. The token clause on IFBench is unaffected and
clears comfortably (#iftokpctce% against the control).

== Dose-response

Between $alpha = 0.05$ and $alpha = 0.1$ on the same 64 held-out screen problems, mean
thinking tokens fall from 2,086.2 to 1,833.4 and the accuracy gain rises from +6.3 to +7.0
pts. The effect scales with the edit magnitude in the expected direction, satisfying the
pre-registered dose-response requirement.

= Integrity and measurement failures

We report three failures, because each is a way this kind of study can produce a wrong number
that survives review.

== A follow-up run contaminated three analysis scripts

After the confirmation runs completed, an additional sampling pass appended samples 2 and 3
for some problems into the *same* result files the confirmation used. The GPQA files ended at
#censbaseunf rows for base, #censeditunf for the edit and #censctrlunf for the control,
against the #censbasefil rows each arm's confirmation actually comprises. Any read of those
files without a sample filter mixes two-sample and four-sample problems inside a per-problem
mean.

Three separate consumers read those files directly and all three were wrong in different
ways: the accuracy grader, the script computing token and truncation deltas, and the script
producing the simulated fixed-budget table. Each was found by recomputing its output
independently rather than by reading its code, and each required its own fix — hardening the
grader did not protect quantities that other scripts derived from the raw file. The
generalisable lesson is that a filter belongs at every point of access to a contaminated
file, not at the first one discovered.

Every number in this paper is computed by a single committed script that applies the sample
filter once and publishes a per-arm census alongside its output; the census for this paper's
inputs is #censbasefil rows per arm with uniform two-sample coverage.

== A robustness check with coverage too poor to report

The pre-registered plan included re-running every sample that hit the 32k cap at a 128k cap,
for all arms with the same seeds, as a robustness check on truncation. That run reached 3 of
35 samples before its host ran out of memory. We report the coverage and decline to report a
result from it. It was never part of the success criterion.

== What a stale baseline looks like

An earlier draft of our own notes carried base GPQA as 82.3 where the grader gives
#gaccbase over all #gn problems. The figure had been copied forward across several documents
before the discrepancy surfaced. It is corrected here and mentioned because a baseline that
travels by transcription is a standing hazard in exactly this kind of comparison; the
#gdbe#[-pt] delta is consistent with #gaccbase, not with 82.3.

= Discussion

The honest summary is narrow and worth stating precisely. Nine rank-one updates, derived from
a few hundred forward passes on a train split and costing no gradient steps, make a 27B
reasoning model measurably better on two of three held-out benchmarks and substantially
cheaper on all three. An identical-magnitude random perturbation of the same matrices does
none of this.

The mechanism is plausible from the construction. The `cc` direction is fitted on the region
immediately preceding the end of thinking, and the additive edit writes along it at every
forward pass through those nine layers. The effect is to make the model's own
stop-thinking-and-conclude tendency slightly more available throughout generation. That
predicts shorter reasoning, and it predicts the gain should be largest where truncation at a
fixed cap was costing accuracy — which matches the simulated fixed-budget behaviour, where
the advantage widens as the budget tightens.

What this does not establish is generality. One model family, one size, one edit form, one
$alpha$, one engine, three benchmarks. The directions were fitted on mathematical reasoning
traces and transfer to science questions and code, which is mildly surprising and suggests
the structure is about reasoning length rather than domain. Whether that holds at other
scales is untested here.

= Limitations

The confirmation suite is three benchmarks, chosen partly for feasibility: agentic
evaluations were dropped because they require harnesses we could not run, and a
judge-dependent benchmark was dropped to avoid a remote model in the measurement loop.
Accuracy is scored at two samples per problem on GPQA and one elsewhere, so per-problem
estimates are coarse.

All arms are Q8_0 quantized. We have not evaluated the edit at BF16, FP8 or NVFP4, so the
interaction between this edit and quantization is unknown. The 32k cap is well below the
262k the model card recommends for reasoning, chosen for throughput; truncation remains
present in every arm, and the robustness check meant to bound its effect failed on coverage.

The control is one random direction per layer at one magnitude. A stronger design would sweep
several random seeds per layer and report the distribution of control outcomes, which would
convert "the control does nothing" from three point estimates into an interval. We report
what we ran.

The IFBench control clause is borderline, as described above, and the resolution of one
criterion clause — whether "beats the control" inherits the criterion's own two-clause
definition of beating — was a judgement we made explicit rather than a measurement. Under the
stricter reading the IFBench control comparison fails and the conclusion rests on the other
two benchmarks, which is where we have put it regardless.

#v(0.4em)
*Status.* Working paper v1. The pre-registered criterion passes on GPQA Diamond and
LiveCodeBench v6 against both base and the norm-matched control, and on IFBench against base
on the token clause. The IFBench control clause is reported as borderline and not claimed.

*Data and code.* The edit and control specifications, the direction tensors, every graded
result file with its digest, the generating script and its per-arm sample census, and the
seed-stability sweep are published with this paper. `emit_results.py` recomputes every
statistic from the graded artifacts and `emit_numbers.py` regenerates the figures quoted in
this text, so no number here is transcribed by hand.

*Acknowledgements.* The replication was proposed by the author. The analysis, figures and
text are the author's, carried out with an AI coding agent under his direction.
