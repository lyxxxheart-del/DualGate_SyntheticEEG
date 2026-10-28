# Final Text for Proofreading (plain text, no LaTeX)

> Source: `main.tex`. This is the extracted prose for spelling/grammar checking.
> LaTeX math has been converted to readable inline notation; citations are shown
> in author–year form instead of bracketed numbers. Tables are reproduced in
> Markdown below their mention. The full reference list is at the end.

---

## Title

When Physiological Plausibility Misleads: Covariance-Aligned Quality Gates for Synthetic Motor-Imagery EEG

## Abstract

Generative models are increasingly used to augment scarce motor-imagery (MI)
electroencephalography (EEG) datasets, and their synthetic samples are commonly
filtered by *physiological* quality gates such as the event-related
desynchronization (ERD) that a motor-imagery trial is expected to produce. We
show that this premise is systematically flawed: an ERD gate measures a
*temporal* energy contrast, whereas the downstream common-spatial-pattern (CSP)
classifier consumes the *spatial* covariance of each trial. Because the two
criteria occupy orthogonal representational spaces, an ERD gate can be both
blind to harmful corruption and itself damaging. We propose a dual gate that
operates in the exact covariance feature space of the downstream decoder: a
plausibility gate based on the cosine similarity of a trial's covariance to its
class template, and a discriminative gate based on the Fisher linear-discriminant
projection gap. On a real state-of-the-art generative model — DDF-Net, a
denoising-diffusion model whose per-channel synthesis collapses cross-channel
covariance (synthetic cosine similarity 0.368 versus real 0.996) — unfiltered
samples degrade CSP accuracy from 0.833 to 0.667, whereas our dual gate admits
0/144 samples (0% leakage) and losslessly restores the 0.833 baseline. These
results argue that generative EEG models must jointly model the channel
distribution, and that a quality gate is only effective when it shares the
decoder's representation.

## Keywords

EEG, motor imagery, synthetic data, generative models, diffusion, common spatial
patterns, data quality.

---

## I. Introduction

Decoding motor imagery from EEG underpins non-invasive brain–computer interfaces
(BCIs), but the scarcity and high inter-subject variability of labeled trials
limit the performance of data-hungry decoders (Blankertz et al., 2008;
Schirrmeister et al., 2017; Lawhern et al., 2018; Fahimi et al., 2019). To
relieve this bottleneck, researchers increasingly synthesize artificial MI-EEG
with generative models — generative adversarial networks (GANs), variational
autoencoders (VAEs), and, most recently, denoising diffusion probabilistic
models (DDPMs) — and use the samples to augment training (Goodfellow et al.,
2014; Kingma & Welling, 2014; Ho et al., 2020; Dhariwal & Nichol, 2021; Song et
al., 2021; Hartmann et al., 2018; Lashgari et al., 2020; Vetter et al., 2024;
Dere et al., 2025; Li et al., 2026).

Because generated samples can be poor, a natural practice is to gate them by a
*physiological* criterion before mixing them into the training set. In MI the
dominant such criterion is the event-related desynchronization (ERD): a real
motor-imagery trial should exhibit a band-power decrease over the contralateral
sensorimotor cortex during the task (Pfurtscheller & Lopes da Silva, 1999). This
paper interrogates the implicit assumption behind that practice — that a
*physiologically plausible* sample is necessarily a *useful* sample for the
downstream decoder — and shows that the assumption does not hold.

The root cause is a representational mismatch. ERD is a *temporal* quantity: a
ratio of band power between a baseline and a task window. The dominant MI
decoder, CSP followed by a linear discriminant (Ramoser et al., 2000; Blankertz
et al., 2008; Lotte et al., 2018), consumes a *spatial* quantity: the per-trial
channel covariance matrix X·Xᵀ. A gate built on the former is therefore, in
principle, blind to exactly the structure the latter exploits, and vice versa. We
demonstrate three concrete consequences on controlled corruption: an ERD gate
cannot detect white-noise corruption that destroys the covariance off-diagonal,
it spuriously deletes harmless time-shifted trials, and in a full-data regime it
can itself reduce accuracy. A covariance-aligned gate, in contrast, rejects the
corruption while preserving the harmless perturbation.

Our contributions are threefold:

1. We identify and quantify the negative result that a physiological ERD gate is
   misaligned with the covariance representation of the CSP decoder, and can be
   both ineffective and harmful (Section II, Section IV).
2. We propose a *dual covariance-aligned gate* operating on the exact feature
   space of the downstream decoder: a plausibility gate using cosine similarity
   to the class covariance template, and a discriminative gate using the Fisher
   linear-discriminant projection gap (Section II).
3. We validate the gate against a real state-of-the-art generative model —
   DDF-Net, a diffusion-based MI generator — and show that its per-channel
   synthesis collapses spatial covariance, that unfiltered samples are actively
   harmful, and that our gate achieves 0% leakage while losslessly restoring
   baseline accuracy (Section IV, Section V).

## II. Method

### II-A. Problem setting and downstream decoder

We consider labeled MI-EEG trials (X_i, y_i), where X_i ∈ R^(C×T) is a trial
with C = 22 EEG channels and T time samples, and y_i ∈ {−1, +1} denotes left-
versus right-hand imagery. The downstream decoder is a two-stage CSP +
linear-discriminant-analysis (LDA) pipeline (Ramoser et al., 2000; Blankertz et
al., 2008). CSP computes spatial filters from the per-class mean *spatial
covariance* matrices; that is, the decoder's only contact with a trial is
through

    C_i = X_i · X_iᵀ ∈ R^(C×C),

the uncentered energy covariance. Any quality gate that is to remain faithful to
the decoder must therefore judge trials in the same covariance space.

We reduce C_i to a scale-invariant feature vector by trace normalization
followed by vectorization of the upper triangle (including the diagonal),

    φ_i = triu( C_i / tr(C_i) ) ∈ R^d,   with   d = C(C+1)/2 = 253,

which matches the affine-invariance step of Riemannian covariance alignment and
isolates cross-channel structure from overall trial power.

### II-B. Gate 1: covariance plausibility

The first gate tests whether a trial *looks like* plausible EEG of its class.
From the real calibration set we form a per-class covariance template
μ_c = E[φ | y = c] and score each trial by the cosine similarity of its
covariance features to its own class template,

    sim_cos(φ, μ_c) = (φ · μ_c) / ( ‖φ‖ · ‖μ_c‖ ).

An acceptance window is calibrated as the 5th–95th percentile of the within-class
similarity distribution of real trials,
A_c = [ Q5({s_i}_{y_i=c}), Q95({s_i}_{y_i=c}) ], and a trial is retained if and
only if sim_cos(φ_i, μ_{y_i}) ∈ A_{y_i}. Because white-noise corruption inflates
the diagonal (self-power) relative to the off-diagonal (cross-channel) entries,
it rotates φ away from μ_c and drives the similarity out of the window.
Crucially, however, a class-ambiguous blend of a left and a right trial is still
a *valid* covariance and passes Gate 1; this motivates a second, discriminative
gate.

### II-C. Gate 2: discriminative projection gap

The second gate tests whether a trial carries the *correct class* information. We
fit a Fisher LDA on the covariance features of real trials, obtaining a
discriminative direction w ∈ R^d (with a small ridge regularization) that best
separates the two classes (Fisher, 1936). A trial is projected to a scalar score

    z_i = φ_i · w,

and per-class acceptance windows
D_c = [ Q5({z_j}_{y_j=c}), Q95({z_j}_{y_j=c}) ] are calibrated on real trials. A
trial is retained if and only if z_i ∈ D_{y_i}. For real MI data the two class
score distributions are well separated with an intermediate *rejection gap*: a
class-mixed trial, being an average of the two class covariances, projects near
the midpoint between the class modes and lands in the gap, where it is rejected
by both windows. The dual gate admits a trial only if it passes Gate 1 *and*
Gate 2.

## III. Experiments

### III-A. Dataset and protocol

We use the BCI Competition IV dataset 2a, which records 22 EEG channels at 250 Hz
from 9 subjects (A01–A09) performing four-class MI (Tangermann et al., 2012;
Brunner et al., 2008). We restrict to the left- and right-hand classes. Gates are
calibrated on subject A01's training session (A01T) with a fixed 108/36 trial
split, and downstream accuracy is evaluated over 10 seeds; synthetic batches
contain 144 trials (72 per class, balanced).

### III-B. Controlled corruption modes

To probe the gates under known, interpretable defects, we generate three
synthetic-batch families from real A01T trials: **Mode 1** (class mixing),
averaging one left and one right trial, producing a valid but class-ambiguous
covariance; **Mode 2** (white noise), adding 0 dB Gaussian noise, which
attenuates cross-channel structure while inflating the diagonal; and **Mode 3**
(time shift), a circular temporal shift, which preserves the covariance and is
harmless to CSP.

### III-C. Real generative model: DDF-Net

As a terminal test we use DDF-Net, a diffusion-based MI-EEG model that augments a
dual-domain classification network with DDPM-generated samples (Li et al., 2026).
We train its Stage-1 DDPM (a UNet-style architecture with T = 100 diffusion
steps and 38.93 M parameters) on A01T and sample 144 synthetic trials. Because
the generator models each of the channels *independently* and reassembles them
into a trial afterwards, the joint cross-channel distribution is never learned —
an ideal stress test for a covariance gate.

## IV. Results

### IV-A. The gate is selective on controlled corruption

The dual gate behaves exactly as designed. On the A01T synthetic batch, Gate 1
rejects 0/144 white-noise trials (Mode 2) but retains 123/144 class-mixed trials
(Mode 1), which are still valid covariances; Gate 2 then rejects the
class-ambiguous blends, reducing Mode-1 retention to 50/144. For time-shifted
trials (Mode 3), whose covariance is unchanged, the dual gate correctly retains
114/144, i.e., it does not spuriously kill a harmless perturbation. By contrast,
an ERD gate retains 106/144 of the white-noise trials while deleting roughly 20%
of the harmless time-shifted trials.

Table 1 reports aggregate downstream accuracy across the 9 subjects at the most
severe (100%) contamination level. On white noise (Mode 2), unfiltered
augmentation drops accuracy from 0.762 to 0.690 (paired t-test p < 0.001), and
the covariance-aligned gates restore it to 0.762, matching baseline. On class
mixing (Mode 1) the dual gate recovers a modest but significant fraction of the
loss at lower contamination (p = 0.016 at 25%, p = 0.014 at 50%); at 100%
contamination the improvement is not significant (p = 0.524), consistent with a
ceiling effect when abundant real data remain. Time shifts (Mode 3) induce no
systematic degradation.

**Table 1.** Aggregate downstream accuracy (mean ± std over 9 subjects) at 100%
contamination. Mode 1: class mixing; Mode 2: white noise; Mode 3: time shift.
"Cov" and "Dual" denote the covariance-only gate (Gate 1) and the dual gate
(Gates 1+2).

| Mode | Base | Unfiltered | Cov | Dual |
|------|------|-----------|-----|------|
| 1 (mix)   | 0.762 ± 0.125 | 0.732 ± 0.117 | 0.737 ± 0.112 | 0.739 ± 0.128 |
| 2 (noise) | 0.762 ± 0.125 | 0.690 ± 0.156 | 0.762 ± 0.125 | 0.762 ± 0.125 |
| 3 (shift) | 0.762 ± 0.125 | 0.740 ± 0.126 | 0.743 ± 0.128 | 0.744 ± 0.130 |

**Figure 1** (projection gap). Fisher LDA discriminative scores of real left- and
right-hand trials (blue/red) and class-mixed synthetic trials (green). The two
classes' 5%–95% acceptance windows are separated by a shaded *rejection gap*;
class-ambiguous blends project near the midpoint and fall into this gap, where
Gate 2 rejects them.

### IV-B. Real-model validation: DDF-Net collapses covariance

Against the real DDF-Net generator the gate is decisive. Figure 2 visualizes the
root cause: the trace-normalized mean covariance of DDF-Net synthetic right-hand
trials is nearly diagonal, whereas real trials carry substantial off-diagonal
(cross-channel) structure; the difference map isolates exactly the structure that
per-channel generation discards. Figure 3 translates this into the gate's
decision variable: real trials concentrate at cosine similarity 0.996 to their
class template, while synthetic trials cluster at 0.368 — well below the
[0.990, 0.999] acceptance window, so 144/144 synthetic trials fall outside and
are rejected (0% leakage).

Table 2 reports the terminal result. Feeding all 144 unfiltered synthetic trials
into training degrades CSP accuracy from 0.8333 to 0.6667, a 0.17 drop that
confirms the synthetic data are not merely uninformative but actively harmful.
Applying the covariance gate (or the full dual gate) keeps 0/144 samples and
restores accuracy to the 0.8333 baseline with no loss.

**Table 2.** Downstream CSP accuracy on DDF-Net synthetic augmentation. "Kept" is
the number of synthetic trials admitted into training.

| Condition | Accuracy | Kept |
|-----------|----------|------|
| Baseline (no augmentation) | 0.8333 | — |
| Unfiltered                 | 0.6667 | 144/144 |
| Covariance gate (Gate 1)   | 0.8333 | 0/144 |
| Dual gate (Gates 1+2)      | 0.8333 | 0/144 |

**Figure 2** (covariance collapse). Trace-normalized mean spatial covariance
(22×22) of (a) real right-hand trials, (b) DDF-Net synthetic right-hand trials,
and (c) their element-wise difference. Per-channel independent generation
produces a near-diagonal synthetic covariance, discarding the off-diagonal
(cross-channel) structure on which CSP spatial filters depend.

**Figure 3** (cosine-similarity rejection). Distributions of per-trial covariance
cosine similarity to the class template for real A01T trials (blue) and the 144
DDF-Net synthetic trials (red), with the 5%–95% acceptance window shaded.
Synthetic similarity (mean 0.368) collapses far below the window [0.990, 0.999];
all 144 trials fall outside, so Gate 1 rejects 100%.

## V. Discussion

### V-A. Why per-channel generation is harmful

CSP is a spatial method: its filters are the eigenvectors of a generalized
eigenvalue problem formed from per-class *covariance* matrices (Ramoser et al.,
2000; Blankertz et al., 2008). A generator that synthesizes each channel
independently and stitches channels together afterwards can reproduce each
channel's marginal power spectrum, but it cannot reproduce the cross-channel
correlations that constitute the spatial structure CSP exploits. The result is
the diagonalization visible in Figure 2: the synthetic trials occupy a region of
covariance space so far from the real class templates (cosine similarity 0.368
versus 0.996) that even a coarse plausibility gate rejects them all. This is a
stronger statement than "the samples are noisy" — it shows that a *per-channel*
generative model is architecturally incapable of producing CSP-useful data,
regardless of how realistic its single-channel signals appear. We therefore argue
that generative EEG models intended for spatial decoders must model the *joint*
channel distribution.

### V-B. Why the representation must match the decoder

The negative result on the ERD gate follows the same logic in reverse. ERD is a
temporal band-power contrast; it is orthogonal to the spatial covariance the
decoder reads. Consequently an ERD gate is blind to covariance collapse and can
simultaneously reject harmless temporal perturbations. Our dual gate sidesteps
this by construction: both of its decision variables — cosine similarity and the
LDA projection — are computed on the identical trace-normalized covariance
features that the decoder consumes. The alignment, not the specific threshold, is
what makes the gate effective, and we expect the principle to transfer to any
decoder whose representation can be made explicit.

### V-C. Limitations

Three limitations bound our conclusions. First, a ceiling effect: when real data
are abundant (e.g., Mode 1 at 100% contamination), the marginal value of the gate
disappears, so significance is expected to vanish. Second, boundary leakage:
class-mixed discriminative scores follow a distribution centered near the class
boundary, so a fraction of Mode-1 blends still land inside a class window (50/144
retained); this is the inherent cost of a percentile window, and tightening it
would misclassify the harmless Mode-3 trials. Third, our gate is calibrated on a
single subject and session (A01T); cross-subject and cross-session generalization
remains to be established.

## VI. Conclusion

We showed that physiological plausibility is an unreliable proxy for downstream
utility in synthetic MI-EEG: an ERD gate operates in a temporal space orthogonal
to the spatial covariance that CSP decodes, and can be both blind to harmful
corruption and itself damaging. A dual gate aligned to the decoder's covariance
representation — cosine-similarity plausibility plus a discriminative projection
gap — recovers downstream accuracy under controlled corruption and, against the
real DDF-Net diffusion generator, achieves 0% leakage (0/144) while losslessly
restoring the 0.833 baseline. The finding that per-channel synthesis collapses
spatial covariance carries a direct design imperative: generative EEG models for
spatial decoders must model the joint channel distribution, and quality gates
must share the decoder's representation to serve as the last line of defense
before deployment.

---

## References (author–year; bracketed numbers match the compiled PDF)

1. Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., & Müller, K.-R. (2008). Optimizing spatial filters for robust EEG single-trial analysis. *IEEE Signal Processing Magazine*, 25(1), 41–56.
2. Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W., & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391–5420.
3. Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.
4. Fahimi, F., Zhang, Z., Goh, W. B., Lee, T.-S., Ang, K. K., & Guan, C. (2019). Inter-subject transfer learning with an end-to-end deep convolutional neural network for EEG-based BCI. *Journal of Neural Engineering*, 16(2), 026007.
5. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial nets. *Advances in Neural Information Processing Systems*, 27, 2672–2680.
6. Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *International Conference on Learning Representations*.
7. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*, 33, 6840–6851.
8. Dhariwal, P., & Nichol, A. (2021). Diffusion models beat GANs on image synthesis. *Advances in Neural Information Processing Systems*, 34, 8780–8794.
9. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *International Conference on Learning Representations*.
10. Hartmann, K. G., Schirrmeister, R. T., & Ball, T. (2018). EEG-GAN: Generative adversarial networks for electroencephalographic (EEG) brain signals. *arXiv preprint arXiv:1806.01875*.
11. Lashgari, E., Liang, D., & Maoz, U. (2020). Data augmentation for deep-learning-based electroencephalography. *Journal of Neuroscience Methods*, 346, 108885.
12. Vetter, J., Macke, J. H., & Gao, R. (2024). Generating realistic neurophysiological time series with denoising diffusion probabilistic models. *Patterns*, 5(9), 101047.
13. Dere, M. D., Jo, J.-H., & Lee, B. (2025). Motor-intent decoding from synthetic EEG data using denoising diffusion probabilistic models. *Expert Systems with Applications*, 299, 130134.
14. Li, X., Wang, J.-H., An, J., Ren, L.-L., Bian, L.-F., & Yang, C. (2026). Diffusion enhancement-based dual-domain fusion attention motor imagery electroencephalography classification network. *Engineering Applications of Artificial Intelligence*, 167, 113954.
15. Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology*, 110(11), 1842–1857.
16. Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. *IEEE Transactions on Rehabilitation Engineering*, 8(4), 441–446.
17. Lotte, F., Bougrain, L., Cichocki, A., Clerc, M., Congedo, M., Rakotomamonjy, A., & Yger, F. (2018). A review of classification algorithms for EEG-based brain–computer interfaces: a 10 year update. *Journal of Neural Engineering*, 15(3), 031005.
18. Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179–188.
19. Tangermann, M., Müller, K.-R., Aertsen, A., Birbaumer, N., Braun, C., Brunner, C., Leeb, R., Mehring, C., Miller, K. J., Müller-Putz, G. R., et al. (2012). Review of the BCI competition IV. *Frontiers in Neuroscience*, 6, 55.
20. Brunner, C., Leeb, R., Müller-Putz, G., Schlögl, A., & Pfurtscheller, G. (2008). BCI Competition 2008–Graz data set A. *Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces), Graz University of Technology*.
