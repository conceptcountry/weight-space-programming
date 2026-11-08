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

## Weights

Published as `conceptcountry/qwen-concept-27b`, on both
[models.concept.country](https://models.concept.country) and
[HuggingFace](https://huggingface.co/conceptcountry).

| format | size | what it is for |
|---|---|---|
| bf16 | ~52 GB | full precision; the weights every result was computed from |
| FP8 | ~30 GB | FP8_BLOCK, matching Qwen's own FP8 layout |
| NVFP4A16 | ~20 GB | NVFP4 weights, BF16 activations, for Blackwell-class hardware |
| Q8_0 GGUF | ~27 GB | llama.cpp; the exact file the benchmarks ran on |
| Q4_K_M GGUF | ~16 GB | llama.cpp, one hop from bf16 |

The registry speaks the HuggingFace Hub API, so either source works with
standard tooling:

```sh
HF_ENDPOINT=https://models.concept.country hf download conceptcountry/qwen-concept-27b
```

The quantizations are deployment conveniences, not separately evaluated
artifacts. Every number in the paper comes from bf16 weights served at Q8_0.
Nothing here claims a 4-bit conversion preserves the effect — that would need
its own evaluation, and has not had one.

Experimental, under evaluation.

## What is not here

No model weights in this repository. The weights live in the registry and on HuggingFace, linked above. The base
model is Qwen3.8-27B (Apache-2.0, Alibaba Cloud).

## License

Apache-2.0. See LICENSE and NOTICE.
