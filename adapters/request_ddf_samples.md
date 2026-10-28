# 邮件模板：向 DDF-Net 作者请求预生成合成样本

> 用途：当本地无法完成 DDF-Net 推理时，用本模板联系论文作者，请求一批
> BCI IV-2a 的预生成合成样本。请在 `[ ]` 处填写真实信息后发送。

---

**Subject:** Request for pre-generated synthetic MI-EEG samples from DDF-Net

Dear Prof./Dr. [Author name],

I hope this message finds you well. I am [Your name], a [PhD student / research
engineer] at [Your institution]. I read your paper *"Diffusion enhancement-based
dual-domain fusion attention motor imagery electroencephalography classification
network"* (Engineering Applications of Artificial Intelligence, 2026) with great
interest, and I would like to ask a small favor regarding the released DDF-Net
code and pretrained weights.

Our group works on **quality filtering of generative MI-EEG data** — specifically,
we study which physiological/statistical criteria a diffusion model's synthetic
samples should satisfy in order to *improve* a downstream BCI classifier rather
than degrade it. We would like to validate our filter on real diffusion-generated
samples rather than only on hand-crafted artifacts.

We would be very grateful if you could share one batch of **pre-generated synthetic
samples** produced by DDF-Net for the **BCI Competition IV-2a** dataset
(22 channels, 250 Hz), for example:

- ~144–288 epochs, any or all of the 9 subjects, in `.npy` format
  (shape `[n_epochs, 22, n_times]` or a flattened equivalent), with per-epoch
  class labels (left-hand / right-hand) if available.

If sharing the samples is not convenient, pointing us to the exact inference entry
point and pretrained-weight location in the repository would already help us
reproduce generation locally.

We will of course acknowledge your work and the dataset in any resulting
publication. Thank you very much for your time and for making your code and weights
publicly available.

Best regards,

[Your name]
[Your title / affiliation]
[Your email]
