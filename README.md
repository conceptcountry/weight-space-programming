# Weight-Space Programming

Analysis code and evaluation artifacts for *Gradient-Free Weight Editing Reduces
Overthinking in Qwen3.8-27B*, a norm-matched controlled replication of
weight-space programming.

Paper: https://concept.country/research/weight-space-programming

## What this is

A nine-layer, gradient-free edit to Qwen3.8-27B that cuts thinking tokens by up
to 54% while accuracy goes up rather than down. The contribution is not the edit
— it is the control. An equal-norm random edit, applied to the same layers with
the same magnitude, does nothing. That is what separates "this direction means
something" from "perturbation helps."

Every statistic in the paper is generated from the artifacts in `results/`. No
number is transcribed by hand. If you change an artifact and re-run the
generators, the paper changes with it.

## Layout

```
emit_results.py     graded + raw artifacts  ->  results/results.json
emit_numbers.py     results.json            ->  paper/numbers.typ  (111 constants)
make_figures.py     results.json            ->  figures/*.png
seed_check.py       bootstrap seed stability of the control intervals
paper/paper.typ     the paper source (typst)
results/            the evidence the numbers come from
```

## Reproducing the paper

```sh
python emit_results.py          # rebuild results.json from artifacts
python emit_numbers.py          # rebuild the paper's constants
python make_figures.py          # rebuild the figures
typst compile --root . paper/paper.typ paper/paper.pdf
```

`make_figures.py` needs matplotlib and pillow; everything else is standard
library plus the artifact files.

## A note on one result

The IFBench control interval's lower bound lands exactly on zero for 2 of 10
bootstrap seeds. `seed_check.py` is what found that. The paper reports it as
borderline rather than as a win, because it is. The GPQA and LCB control margins
are seed-stable, and the verdict rests on those.

This is the kind of thing a replication is for.

## What is not here

No model weights. The subject model is Qwen3.8-27B (Apache-2.0, Alibaba Cloud);
the edit is described in the paper in enough detail to reproduce.

## License

Apache-2.0. See LICENSE and NOTICE.
