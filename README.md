# dimwit-evidence-custom

Thirteen real tensor-shape defects, taken from the issue trackers and pull requests of
well-known repositories, each reconstructed four times — plain NumPy/PyTorch, plain JAX,
JAX + jaxtyping, and DimWit — so that the claim *"names as types catch this at compile
time"* can be checked rather than asserted.

Built for the evaluation section of the DimWit paper, in response to the review note:

> 10–15 reale Shape-Bugs aus GitHub-Issues bekannter Repos nachbauen; Tabelle: welche fängt
> DimWit zur Compile-Zeit vs. Haliax / jaxtyping / plain JAX.

Haliax is out of scope here, as requested.

The set started at fifteen. Two were dropped after a closer read of their upstream sources:
the Whisper case (a usage question rather than a reported defect) and the SFData case (a
corpus entry, not a bug report). Their numbers were not reused — the Gymnasium case, filed
as 14 while they were still in, is now case 13. See *Cases that were dropped* below.

## Running it

```sh
./run_all.sh            # pytest over all reconstructions, then the DimWit compile checks
uv run pytest -q        # just the Python side
./scripts/check_scala.sh -v   # just the Scala side, with compiler output
./scripts/check_tensor_annotations.sh -v   # the tensor_annotations column (mypy + pyright)
```

Requirements: [uv](https://github.com/astral-sh/uv) and
[scala-cli](https://scala-cli.virtuslab.org/). Nothing else has to be installed by hand:
the Scala sources declare `ch.contrafactus::dimwit-core:0.1.0`, the published release on
Maven Central, so scala-cli fetches DimWit itself. No local `sbt publishLocal` is involved
and the results below are reproducible against a fixed, public artifact.

One test is skipped by default: `cases/02/plain.py::test_torch_only_warns` needs PyTorch,
which is not in the lock file. It was verified separately against torch 2.5.1 — see
`RESULTS.md`.

## What is in each case folder

| file | contents |
|---|---|
| `README.md` | the defect, a link to the upstream issue and fix, and why the case is here |
| `plain.py` | buggy and fixed reconstruction in NumPy (or PyTorch where the semantics matter), with tests that assert the *erroneous* behaviour as well as the correct one |
| `plain_jax.py` | the same in JAX |
| `jaxtyping_case.py` | the same annotated with jaxtyping + beartype |
| `tensor_annotations_case.py` | the same annotated with tensor_annotations, with tests that run mypy and pyright on the file and assert their exact diagnostics |
| `Buggy.scala` | the DimWit transliteration of the mistake — normally rejected by the compiler |
| `Fixed.scala` | the DimWit program that is correct, with a `@main` check that runs it |
| `RankOnly.scala` | the defect in DimWit with positional axis labels (`A0`, `A1`, ...) instead of semantic names — the rank-only ablation |
| `PLAIN_SOLUTION.md`, `PLAIN_JAX_SOLUTION.md`, `JAXTYPING_SOLUTION.md`, `TENSOR_ANNOTATIONS_SOLUTION.md`, `RANK_ONLY_SOLUTION.md`, `DIMWIT_SOLUTION.md` | one verdict each, with the reasoning and the honest limits |

## Categories

Every verdict is one of three:

* **`compile`** — the compiler rejects the program. Nothing runs.
* **`run-time`** — the program starts and fails with an error on the path that contains the
  defect.
* **`missed`** — the program runs to completion and produces a result of the right shape and
  the wrong values. Only inspecting the numbers reveals it.

## Results

Verified by `run_all.sh`: **184 Python tests pass**, all 13 `Fixed.scala` compile, 12 of 13
`Buggy.scala` are rejected at compile time, and the 1 that is not is recorded as `missed`.
The tensor_annotations column is verified separately by `./scripts/check_tensor_annotations.sh`
(**55 tests pass**), which runs in a throwaway environment so the lock file is untouched.
The rank-only column is not covered by either script: each `RankOnly.scala` was checked by
hand with `scala-cli compile`.

| # | case | source | Plain | jaxtyping | tensor_annotations¹⁰ | DimWit rank-only¹⁵ | **DimWit** |
|---|---|---|---|---|---|---|---|
| 01 | attention mask on the query axis | [transformers#23974](https://github.com/huggingface/transformers/issues/23974) | missed | missed¹ | missed | missed | **compile** |
| 02 | MSE over `[n,1]` vs `[n]` | [pytorch#16045](https://github.com/pytorch/pytorch/issues/16045) | missed (warns)² | run-time³ | compile¹¹ | compile | **compile** |
| 03 | `cross` on the first length-3 axis | [keras#23219](https://github.com/keras-team/keras/pull/23219) | missed | missed | compile¹² | compile¹⁶ | **compile** |
| 04 | displacement normalised by the wrong side | [torchvision#9299](https://github.com/pytorch/vision/issues/9299) | missed | missed | missed | missed | **missed**⁴ |
| 05 | NCHW height/width swapped | [ultralytics#23126](https://github.com/ultralytics/ultralytics/issues/23126) | missed | missed⁵ | missed | missed | **compile** |
| 06 | softmax reduction dropped its axis | [triton#11406](https://github.com/triton-lang/triton/issues/11406) | missed⁶ | run-time⁷ | missed | missed¹⁷ | **compile** |
| 07 | stacked dim name collided | [arviz#1647](https://github.com/arviz-devs/arviz/pull/1647) | missed | missed | missed | missed | **compile** |
| 08 | reshape with pre-transpose sizes | [pymc#7178](https://github.com/pymc-devs/pymc/issues/7178) | missed | run-time⁸ | missed | missed¹⁸ | **compile** |
| 09 | single G-vector broadcast | [pyscf#2961](https://github.com/pyscf/pyscf/issues/2961) | missed | run-time | compile¹³ | compile | **compile** |
| 10 | model called without a batch axis | [lit-llama#166](https://github.com/Lightning-AI/lit-llama/pull/166) | missed | run-time | compile | compile | **compile** |
| 11 | NCHW frames into an NHWC pipeline | [vllm#51076](https://github.com/vllm-project/vllm/pull/51076) | missed | missed⁹ | compile¹⁴ | missed | **compile** |
| 12 | batched contraction over the wrong axis | [mlx#4125](https://github.com/ml-explore/mlx/pull/4125) | missed⁹ | missed⁹ | missed | missed¹⁸ | **compile** |
| 13 | `(w,h)` extents into an `(h,w)` API | [Gymnasium#1312](https://github.com/Farama-Foundation/Gymnasium/pull/1312) | missed | missed | missed | missed¹⁹ | **compile** |

| | compile | run-time | missed |
|---|---|---|---|
| Plain (NumPy/PyTorch) | 0 | 0 | **13** |
| jaxtyping | 0 | 5 | 8 |
| tensor_annotations | 5 | 0 | 8 |
| DimWit rank-only | 4 | 0 | 9 |
| **DimWit** | **12** | 0 | 1 |

**Plain JAX has no column of its own.** It is still reconstructed in every case
(`plain_jax.py`) and still machine-checked, but with the Whisper case dropped there is no
longer a single cell in which it differs from plain NumPy/PyTorch: it misses all thirteen.
A separate column would add no information.

¹ caught for cross-attention, where the query and key lengths differ; Pix2Struct is self-attention.
² PyTorch emits a broadcasting `UserWarning`, then computes and returns a loss — a diagnostic, not a detection.
³ caught if the loss is annotated `"batch"` on both arguments — but missed with the signature the buggy model actually had, since `nn.Linear(2, 1)` really does return `[batch, 1]`. Both are in `jaxtyping_case.py`.
⁴ the defect is a scalar divisor, and `size(Axis[Width]): Int` is a plain number. Not detected — but `size[0]` cannot be written, so the mistake reads as `alpha._1 / size(Axis[Height])`. Legibility, not detection.
⁵ jaxtyping binds axis names only from *array* annotations, so `height: int` constrains nothing. See case 05.
⁶ for the reported square tile, at the scope upstream used. A non-square tile raises — from `jnp` inside the body, not from any annotation at the boundary. Triton is a GPU kernel DSL, and only its *tile algebra* is compared here, not its kernel-orchestration layer, which has no DimWit counterpart — see that case's README.
⁷ at minimal scope: annotating softmax as `Float[Array, "n"] -> Float[Array, "n"]` removes the defect and rejects a tile at run time. At the upstream scope it is missed. The upgrade comes from the scope, not from jaxtyping — but only a type system can enforce a scope, which is why the plain column stays `missed` with the same vector function.
⁸ caught because pymc's chains and draws differed; it would be silent had the two coincided.
⁹ silent on the square data that shipped, and caught on a rectangular input — jaxtyping fires exactly when the two candidate axes have different lengths. Cases 01, 05, 11 and 12 are one phenomenon, not four. mlx#4125 itself raised a `ValueError` from an inconsistent size-1 batch broadcast; case 12 reconstructs the *underlying* hazard — axes identified by position, interchangeable at equal length — which is silent.
¹⁰ [tensor_annotations](https://github.com/google-deepmind/tensor_annotations) 2.0.3 with its JAX stubs, checked by mypy *and* pyright (`tensor_annotations_case.py` asserts the exact diagnostics of both). `compile` here means rejected by the type checker. An operation the stubs leave untyped returns `Any` and counts as `missed` — indexing, `reshape`, `tensordot`, `einsum`, `cross`, `max`, `vmap` among them; no workaround around an `Any` is credited. Not part of `run_all.sh`: run `./scripts/check_tensor_annotations.sh`.
¹¹ also with the signature as written upstream: `[Batch, Out] - [Batch]` aligns the trailing axes by name, and `Out` is not `Batch`. Contrast ³.
¹² at the minimal interface, `cross` of two vectors on one axis as `Fixed.scala` defines it, where a `Spatial` × `Rgb` cross is rejected although every axis has length 3. Missed at the upstream scope, where `jnp.cross` takes an axis argument and is `Any`. The batching lift (`jax.vmap`) is `Any` too, so mapping over the wrong axis is accepted.
¹³ for a caller whose argument is typed: the rank error sits in the interface (`Array2` vs `Array1`). Missed when the vector reaches the call as `Any`, e.g. from indexing or `jnp.array`.
¹⁴ axes are compared by identity, not length, so this holds also for the square frames that shipped — where jaxtyping is silent (⁹).

¹⁵ the DimWit program with the semantic names removed: same library, same compiler, but an axis is labelled by its position (`A0`, `A1`, ...) and the result of every operation is relabelled back to those positions, so a type states the rank of a tensor and nothing else. `RankOnly.scala` and `RANK_ONLY_SOLUTION.md` in each case folder. Not part of `run_all.sh`; each file is checked with `scala-cli compile`. See *The rank-only ablation* below.
¹⁶ at the minimal scope of `cross`, rank 1 x rank 1 => rank 1, as for the DimWit column: the upstream call hands it two matrices, which is a rank error. `cross` is defined in the case, not in DimWit. The `Spatial` x `Rgb` mistake that `Buggy.scala` constructs compiles under positional labels.
¹⁷ at the scope upstream used: `softmaxMatrix` of `Fixed.scala` without names is right over axis 1 and wrong over axis 0, because a reduced rank-1 result broadcasts onto axis 0 whichever axis it came from. The `keep_dims` flag itself is rejected under positional labels as well, and at the minimal scope (`softmaxVector` lifted over axis 0) the result is correct, so this cell is `compile` if scored like 03.
¹⁸ with the same restricted API as the DimWit column, `flatten` in 08 and a contraction by one axis in 12. Without names the restriction rejects nothing: a position changes its meaning under a transpose (08), and every rank-2 tensor has an axis 0 (12).
¹⁹ with the extents as plain `Int`s: without names an extent has no axis to be named after. Typed by position instead, `AxisExtent[A0]` and `AxisExtent[A1]`, the swap is rejected, but that keeps part of DimWit's naming machinery and is not rank-only.

Where a verdict turned on a choice, the table reports the upstream data with the most
favourable annotation, which credits jaxtyping in 02 and 06, and tensor_annotations in 03
and 09.

## The rank-only ablation

The rank-only column answers one question: of the twelve defects DimWit rejects, how many
does it reject because of the *names*, and how many would static rank typing with the same
API reject as well? It keeps DimWit's library, compiler and API and removes only the names.

* **Rejected without names: 4** (02, 03, 09, 10). 02, 09 and 10 are rank errors, which
  jaxtyping and tensor_annotations catch too. 03 is a rank error at the minimal scope of
  `cross`.
* **Rejected only with names: 8** (01, 05, 06, 07, 08, 11, 12, 13). In each, the two things
  that are confused have the same rank, so a rank-only type cannot tell them apart. 06 is
  the least clear of them (¹⁷).
* **04 is missed either way.**

Rank-only DimWit catches the same four cases as tensor_annotations, which additionally
catches 11 by axis identity, something positions cannot do. On these thirteen cases it is
therefore below tensor_annotations, although it never loses the rank of a tensor, where
tensor_annotations falls back to `Any` at every operation its stubs leave untyped.

Cases 08 and 12 bear on whether the detections come from the names or from a more
restrictive API. The API is the same in both columns there, `flatten` in place of `reshape`
and a contraction that names its axis, and without names it rejects nothing (¹⁸).

## The DimWit miss is the point of the exercise

`04` is included deliberately and its `Buggy.scala` compiles.

* **04 — the divisor is a number, not a tensor.** A shape entry read out as an `Int` has
  left the type system. This marks the boundary of the thesis: names as types govern
  tensors. DimWit still removes positional shape access, so the mistake is spelled
  `size(Axis[Height])` rather than `size[0]` — legible, not checked.

Case 06 states a second, narrower limit that no `Buggy.scala` can exhibit: the *deliberate*
choice of the wrong axis stays well typed, because reducing over either axis of a tile has
the same type. Case 13 is the instructive contrast — an axis argument that must agree with
the tensor it is applied to *is* checked, because there are then two occurrences of the label
to disagree. The limit is one occurrence, not axis choice as such.

## Cases that were dropped

Two of the original fifteen were removed rather than rewritten, because neither reported a
defect upstream:

* **whisper#839** (`13-whisper-batch-spectrogram`, "spectrogram with no batch dimension") is
  a usage question about calling the encoder, not a bug report against Whisper. It was also
  the only case in which plain JAX differed from plain NumPy, which is why plain JAX no
  longer earns a column.
* **SFData `s55237206`** (`15-sfdata-missing-batch-feed`, "example fed without its batch
  axis") is an entry in a fault corpus, not an issue anyone filed. Its subject matter
  survives as case 10, which is the same mechanism against a real pull request.

## Corrections the experiment produced

Building the cases changed four verdicts from the initial estimate, and the changes are
worth carrying into the paper:

1. **The Gymnasium case (now 13) was reconstructed as the wrong defect twice before
   landing.** Gymnasium#1312 is not "a single `screen_size` forces square observations" —
   `screen_size` was already a tuple. It is a `(width, height)` pair handed to `cv2.resize`
   while the observation space was declared `(width, height)` and the array produced is
   `(height, width)`. Reconstructed faithfully it is `compile`, not the `missed` first
   recorded — but only because the extents are typed with the frame's own labels. The
   intermediate version, which gave the output fresh axis variables, compiled the swap and
   is written up in `13/DIMWIT_SOLUTION.md` as the signature that gives the check away.
2. **Case 05 under jaxtyping was estimated "run-time detection"; it is `missed` as written.**
   jaxtyping does not bind names from scalar `int` parameters, so the return annotation
   constrains nothing. It only fires if the layout is carried by an array *and* the model
   input is rectangular.
3. **Case 04 was estimated `compile`; it is `missed`.** The straight transliteration of
   `dx * alpha[0] / size[0]` divides by a number, and a number carries no axis. DimWit
   could catch it via `AxisExtent[Width]`, but only by changing the interface, which is a
   different claim from catching the code that was written.
4. **Case 06 was reconstructed as the wrong defect, and is `compile`, not `missed`.**
   triton#11406 is not "the author picked the wrong reduction axis" — it is `keep_dims=False`
   dropping the reduced axis so that the broadcast back re-attached it positionally. That is
   the same mechanism as case 01, and strict axis types remove it. The uncatchable
   *choice* of reduction axis survives as a stated limit in `06/DIMWIT_SOLUTION.md`; it is
   not what the issue reports.

## Suggested next step

Case 10 is the representative of the largest known class of shape faults: in the SFData
corpus, feature-input and label-output mismatches are 65.8% of its 146 crashing faults.
Classifying all 146 patches by the DimWit mechanism that would have prevented them converts
this hand-picked table into a dataset-scale result over peer-reviewed data. Dataset:
[github.com/tensfa/tensfa](https://github.com/tensfa/tensfa),
paper: [arXiv:2106.02887](https://arxiv.org/abs/2106.02887).
