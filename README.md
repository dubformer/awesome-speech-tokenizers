# Speech Tokenizers

This guide with interactive demo is best viewed on out [blog page.](https://dubformer.github.io/awesome-speech-tokenizers/)

**TL;DR**

This is a highly opinionated guide on speech tokenizers based on our experience at [dubformer.ai](http://dubformer.ai) building voice cloning and TTS models. If you're just starting out, we recommend the linguistic tokenizer S3 or the semantic tokenizer MaskGCT.

# Introduction

This guide provides a short overview of modern (and a few classic) speech tokenizers. We focus on tokenizers for downstream TTS models—not general audio or music tokenization. The guide divides modern tokenizers into three categories: acoustic (low-level waveform information), semantic (pseudo-phonetic), and linguistic (high-level text information used as a training signal). Start by reviewing the diagram and table below, then explore the detailed description of each tokenizer group.

## Overview

The following table presents architectural and key information for each tokenizer across these columns:

- **Frame Rate** — How many acoustic tokens per second the tokenizer outputs. Lower is generally better, but the current standard is 25 Hz. This rate is easier for downstream speech models to handle: higher frequencies hurt intonation, while lower frequencies lose information and reduce sound quality.
- **Encoder/Decoder** — Architecture choice. "T" stands for Transformer.
- **Rep.** — The representation format the tokenizer accepts as input. "T" means time domain (raw waveform); "T-F" means time-frequency (mel spectrogram).
- **Training Objectives** — The main losses used during training:
  - **Rec** — Reconstruction (L2 loss on waveform or spectrogram)
  - **VQ** — Vector-Quantization loss. Rarely used nowadays since FSQ and similar methods don't require it. This means the model includes **explicit losses to train the quantizer/codebook**, in addition to reconstruction/feature/GAN losses. These are necessary because straight-through quantization bypasses gradients through the codebook. Primarily the commitment loss.
  - **Feat** — Feature matching loss (GAN-specific). Compares the discriminator's intermediate activations on real vs. reconstructed audio to stabilize GAN training and improve perceptual similarity.
  - **Diff** — Diffusion or flow matching objective (instead of standard L2 mel or waveform reconstruction)
- **Aux.** — Auxiliary losses, primarily for the semantic and linguistic groups:
  - **SD** — **Semantic Distillation**: Uses pretrained SSL/linguistic features to guide (typically early) codebooks or latents
  - **SST (linguistic for short)** — **Supervised Semantic Tokenization**: Supervised objectives (e.g., ASR/phonetic classification) applied to the tokens or first codebook to explicitly encode phonetic detail; sometimes combined with generative decoders (e.g., OT-CFM/diffusion)
  - **Dis** — **Disentanglement**: Additional losses/modules that separate factors (e.g., speaker vs. content, speech vs. background), often via multi-branch encoders or codebooks, to reduce redundancy and enable independent control

| #   | Tokenizer (paper)                                                            | Frame Rate | Encoder | Decoder | Rep. | Quant     | Training Objective(s) | Aux.    |
| --- | ---------------------------------------------------------------------------- | ---------- | ------- | ------- | ---- | --------- | --------------------- | ------- |
| 1   | [EnCodec (Défossez et al., 2023)](https://arxiv.org/abs/2210.13438)          | 75, 150    | CNN+RNN | CNN     | T    | RVQ       | GAN, Feat, Rec, VQ    | –       |
| 2   | [DAC (Kumar et al., 2023)](https://arxiv.org/abs/2306.06546)                 | 75         | CNN     | CNN     | T    | RVQ       | GAN, Feat, Rec, VQ    | –       |
| 3   | [WavTokenizer (Ji et al., 2024)](https://arxiv.org/abs/2408.16532)           | 40, 75     | CNN+RNN | CNN+T   | T    | SVQ       | GAN, Feat, Rec, VQ    | –       |
| 4   | [SpeechTokenizer (Zhang et al., 2024)](https://arxiv.org/abs/2308.16692)     | 50         | CNN+RNN | CNN     | T    | RVQ       | GAN, Rec, Feat, VQ    | SD      |
| 5   | [Mimi (Défossez et al., 2024)](https://arxiv.org/abs/2410.00037)             | 12.5       | CNN+T   | CNN+T   | T    | RVQ       | GAN, Feat, Rec, VQ    | SD      |
| 6   | [X-Codec (Ye et al., 2025)](https://arxiv.org/abs/2408.17175)                | 50         | CNN     | CNN     | T    | RVQ       | GAN, Rec, VQ          | SD      |
| 7   | [X-codec2 (Zhen et al., 2025)](https://arxiv.org/pdf/2502.04128)             | 50         | CNN+T   | T       | T-F  | FSQ       | GAN, Rec              | SD      |
| 8   | [FACodec (Ju et al., 2024)](https://arxiv.org/abs/2403.03100)                | 80         | CNN+RNN | CNN+RNN | T    | GRVQ      | GAN, Feat, Rec, VQ    | Dis     |
| 9   | [LSCodec (Guo et al., 2025)](https://arxiv.org/abs/2410.15764)               | 25, 50     | CNN     | CNN     | T    | SVQ       | GAN, Feat, Rec        | Dis, SD |
| 10  | [S3 (Du et al., 2024)](https://fun-audio-llm.github.io/pdf/CosyVoice_v1.pdf) | –          | T       | T       | T    | FSQ       | SST                   | \*Diff  |
| 11  | [TaDiCodec (Wang et al., 2025)](https://arxiv.org/pdf/2508.16790)            | 6.25       | T       | T       | T-F  | FSQ (BSQ) | Diff                  | \*SST   |
| 12  | [MaskGCT (Wang1 et al., 2024)](https://arxiv.org/pdf/2409.00750)             | 49         | CNN     | CNN     | T-F  | SVQ       | Rec, VQ               | SD      |
| 13  | [MiniMax-Speech (MiniMax, 2025)](https://arxiv.org/pdf/2505.07916)           | 25         | -       | -       | -    | -         | Rec                   | \*SST   |

## 1–3. Acoustic Tokenizers: EnCodec, DAC, and WavTokenizer

EnCodec, DAC, and WavTokenizer are all characterized as **acoustic tokenizers**. This classification means they are "typically learned through encoder-decoder architectures optimized for waveform reconstruction"

**Commonalities**

EnCodec, DAC, and WavTokenizer share several fundamental design principles, as detailed in the survey's taxonomy:

1. **Acoustic Focus:** All three are optimized primarily for high-fidelity waveform reconstruction.
2. **Target Domain:** They are multi-domain tokenizers, trained to handle Speech, Music, and General Audio.
3. **Training Objectives:** All three models are optimized using the same core combination of losses: Adversarial (GAN), Feature Matching (Feats), Reconstruction (Rec), and Vector Quantization (VQ).
4. **Representation:** They operate in the Time (T) domain, processing raw waveforms as input and reconstructing them directly (Page 10).

**Differences**

The primary differences between the three models lie in their quantization strategies, architectural designs, streamability, and computational complexity.

Quantization Method is a key differentiator in how they convert continuous features into discrete tokens.

- **EnCodec and DAC:** Both employ **Residual Vector Quantization (RVQ)**. RVQ uses multiple codebooks and refines the quantization iteratively. It maps a feature to a codebook entry, computes the residual error, and then sequentially applies further quantizers to that residual (Page 7). This results in a multi-stream token output.
- **WavTokenizer:** Employs **Single Vector Quantization (SVQ)**. SVQ uses only a single codebook without iterative residual refinement (Page 8), resulting in a single-stream token output.

## 4–7. Semantic Tokenizers: SpeechTokenizer, Mimi, X-Codec, X-Codec2

These four tokenizers are representative of "hybrid" models designed to bridge the gap between purely acoustic tokenizers (optimized for reconstruction fidelity) and semantic tokenizers (optimized for phonetic content). They aim to achieve high-quality audio reconstruction while ensuring the resulting tokens are rich in semantic information, making them suitable for use in Large Language Models (LLMs).

**What they have in common?**

1. **Hybrid Goal:** All four are hybrid tokenizers aiming to produce discrete tokens that preserve both acoustic fidelity and semantic/phonetic content.
2. **SSL Guidance/Semantic Distillation:** They all leverage knowledge from powerful pre-trained SSL models (such as HuBERT, WavLM, or Wav2Vec2-BERT) to guide the learning process and enrich the tokens with phonetic information.

**What are their differences?**

| Feature          | SpeechTokenizer    | Mimi               | X-Codec            | X-Codec2            |
| ---------------- | ------------------ | ------------------ | ------------------ | ------------------- |
| **Quantization** | RVQ (Hierarchical) | RVQ (Hierarchical) | RVQ (Hierarchical) | SVQ/FSQ (Flat)      |
| **Domain**       | Speech             | Speech             | Multi-domain       | Multilingual Speech |
| **Frame Rate**   | 50 Hz              | **12.5 Hz**        | 50 Hz              | 50 Hz               |
| **Architecture** | CNN+RNN/CNN        | CNN+T/CNN+T        | CNN/CNN (Dual)     | Transformer/Vocos   |
| **Streamable**   | Yes                | **No**             | Yes                | Yes                 |

The key differences lie in the quantization strategy (RVQ vs. SVQ), the significantly lower frame rate of Mimi, and the domain scope (speech-specific vs. universal/multilingual).

## 8–9. Latent Disentanglement: FACodec and LSCodec

LSCodec is designed to address the limitations of traditional discrete speech tokens, which often feature high bitrates and encode redundant timbre information across time steps. Its primary goals are to achieve extremely low bitrates and to explicitly decouple speaker identity (timbre) from the speech representation.

LSCodec is notable for being entirely **unsupervised** and utilizing only a **single codebook** with a small vocabulary size. This results in highly compact representations; for instance, it achieves bitrates of 0.45kbps (at 50Hz) and 0.25kbps (at 25Hz).

It employs a **three-stage training framework**:

1. **Speech VAE with Speaker Perturbation:** A Variational Autoencoder (VAE) is trained in a continuous space. Speaker perturbation is applied to the input to alter the timbre, while clean timbre information (extracted via WavLM from a reference prompt) is provided to the decoder. This establishes an initial information bottleneck.
2. **Speech VQ-VAE:** A Vector Quantization (VQ) layer is inserted into the trained VAE. This discretizes the space and further restricts timbre information from being encoded into the tokens.
3. **Vocoder Training:** A specialized vocoder (CTX-vec2wav) is trained on the discrete tokens to synthesize high-fidelity waveforms.

FACodec, introduced as part of the NaturalSpeech 3 system, is designed to decompose speech waveforms into distinct, disentangled subspaces. This factorization aims to simplify the complex task of speech generation using a "divide-and-conquer" approach, improving quality and controllability in zero-shot Text-to-Speech (TTS).

FACodec explicitly factorizes speech into four attributes: **Content, Prosody, Timbre, and Acoustic Details**. It achieves this using **Factorized Vector Quantization (FVQ)**, employing separate quantizers for content, prosody, and acoustic details, while timbre is captured by a dedicated timbre extractor as a global vector.

FACodec relies on a combination of techniques to enforce this disentanglement, including information bottlenecks, **explicit supervision** (using phoneme labels, F0, and speaker IDs), and **adversarial training**. It operates at higher bitrates than LSCodec (e.g., 4.8 kbps) and uses multiple quantizers across the different factors.

**What they have in common?**

1. **Goal of Disentanglement:** Both codecs explicitly aim to disentangle speech attributes, with a strong focus on isolating speaker timbre from the time-variant discrete tokens.
2. **Vector Quantization (VQ):** Both utilize VQ to convert continuous latent representations into discrete tokens suitable for language modeling.
3. **Information Bottlenecks:** Both architectures leverage information bottlenecks (LSCodec via the VAE/VQ structure; FACodec via projecting to low-dimensional spaces before quantization) to help discard redundant information and enforce disentanglement.
4. **Separate Timbre Conditioning:** Both models extract timbre information separately and provide it to the decoder (LSCodec via cross-attention with a prompt; FACodec via conditional layer normalization using the timbre extractor output), ensuring the main tokens do not need to encode speaker identity.
5. **Voice Conversion:** Due to their ability to disentangle timbre, both codecs can be utilized for zero-shot voice conversion tasks.

What is their difference?

| Feature                 | LSCodec                                                         | FACodec                                                       |
| ----------------------- | --------------------------------------------------------------- | ------------------------------------------------------------- |
| **Supervision**         | Entirely unsupervised.                                          | Supervised (requires phonemes, F0, speaker IDs).              |
| **Primary Goal**        | Ultra-low bitrate and speaker decoupling.                       | Comprehensive attribute factorization and high fidelity.      |
| **Bitrate**             | Very low (e.g., 0.25–0.45 kbps).                                | Moderate (e.g., 4.8 kbps).                                    |
| **Quantization**        | Single codebook.                                                | Factorized VQ (FVQ) with multiple codebooks (e.g., 6 total).  |
| **Factorization Scope** | Decouples Speaker from Content/Prosody.                         | Decouples Content, Prosody, Timbre, and Acoustic Details.     |
| **Training Method**     | Multi-stage (VAE -> VQ-VAE -> Vocoder) with input perturbation. | Combined losses, supervision, and adversarial training (GRL). |

Export to Sheets

What were their novelties?

- **LSCodec:** The authors claim it is the first effort to explicitly achieve speaker decoupling in a single-codebook, low-bitrate speech codec using a purely unsupervised approach. Its novelty lies in the multi-stage training framework combined with a specific time-stretching speaker perturbation technique.
- **FACodec:** The introduction of Factorized Vector Quantization (FVQ) to successfully decompose speech into four distinct subspaces (content, prosody, timbre, acoustic details) while maintaining high-quality reconstruction. It also introduced a novel combination of techniques (supervision, Gradient Reversal Layers, Detail Dropout) to enforce this factorization.

**How is different their ways to disentangle the latent representations?**

The methodologies for achieving disentanglement are fundamentally different, primarily driven by LSCodec's unsupervised nature versus FACodec's reliance on supervision.

**LSCodec: Disentanglement via Perturbation and Bottleneck**

LSCodec relies on data manipulation and architectural constraints to force disentanglement without labels:

1. **Speaker Perturbation:** LSCodec actively modifies the input audio using a time-stretching approach (WSOLA algorithm). This alters the global pitch and timbre while retaining the content and pitch variations.
2. **Forced Information Routing:** Because the encoder input is timbre-perturbed, but the decoder must reconstruct the _original_ audio, the model is forced to obtain the correct timbre information from the separate reference prompt provided to the decoder.
3. **VAE/VQ Bottleneck:** The inherent bottleneck in the VAE and VQ layers encourages the model to discard information provided elsewhere. Since timbre is supplied via the prompt, the tokens prioritize encoding only the time-variant information (content and prosody).

**FACodec: Disentanglement via Supervision and Adversarial Constraints**

FACodec uses explicit guidance and constraints to define the contents of each subspace:

1. **Supervision (Pulling information in):** FACodec uses auxiliary losses to guide specific FVQs toward specific information. The Content FVQ is supervised by predicting phoneme labels; the Prosody FVQ is supervised by predicting normalized F0 (pitch); the Timbre extractor is supervised via speaker classification.
2. **Gradient Reversal Layers (GRL) (Pushing information out):** GRLs are used for adversarial training to actively eliminate information leakage. For instance, F0-GRL is applied to the content subspace to remove prosody, and Phoneme-GRL is applied to the prosody subspace to remove content. A Speaker-GRL is applied to the sum of these latents to eliminate timbre.
3. **Detail Dropout:** The acoustic detail representation is randomly masked during training. This forces the model to achieve reconstruction using only prosody, content, and timbre, ensuring these components are fully utilized and properly decoupled from the details.

## 11. TaDiCodec

TaDiCodec (Text-aware Diffusion Transformer Speech Codec) designed to produce highly compressed, discrete tokens optimized for speech language modeling (SLM) and text-to-speech (TTS). It utilizes a fully Transformer-based architecture within a diffusion autoencoder framework, operating on mel-spectrogram representations. The system is trained end-to-end in a single stage to jointly optimize quantization and reconstruction.

**What makes TaDiCodec unique and different:**

1. **Text-Aware Diffusion Decoder:** The primary innovation is the integration of text guidance directly into the diffusion decoder. By conditioning the reconstruction process on the corresponding text transcription, TaDiCodec leverages strong semantic priors. This allows the model to maintain high fidelity and intelligibility even at extreme compression levels.
2. **Extreme Compression with a Single Codebook:** TaDiCodec achieves an exceptionally low frame rate of **6.25 Hz** and a bitrate of **0.0875 kbps** for 24 kHz speech. Crucially, it achieves this using a single-layer codebook, avoiding the complexity of the multi-layer Residual Vector Quantization (RVQ) structures common in other high-fidelity codecs.
3. **End-to-End Diffusion Training (Flow Matching):** Unlike many contemporary codecs that rely on Generative Adversarial Networks (GANs) and complex loss combinations, TaDiCodec is trained in a single stage using only a diffusion objective (specifically, Flow Matching). This aims for stable optimization without adversarial training.

## 12. X-codec2

X-codec2 is a speech tokenizer introduced within the Llasa framework. The codec aims to capture the entirety of the speech signal - content, prosody, and timbre - within this unified token stream, requiring no additional information during the decoding process.

**Vocabulary (Codebook) Size:** The vocabulary size is **65,536**.

**What Makes it Unique and Different:** X-codec2 builds upon its predecessor (X-codec) but introduces critical modifications that differentiate it from many other neural codecs and enhance its compatibility with LLMs:

1. **Fusion of Semantic and Acoustic Features:** The architecture employs a dual-encoder approach to fuse information before quantization:
   - A **Semantic Encoder** (a pre-trained Wav2Vec2-BERT) captures high-level multilingual content and emotional cues.
   - An **Acoustic Encoder** (residual convolutional blocks) captures low-level acoustic details and timbre.
2. **Semantic Supervision:** To ensure the unified codebook retains sufficient linguistic information, an auxiliary semantic decoder is used during training (though not during inference). This provides a supervisory signal by reconstructing semantic features from the quantized representation.
3. **Transformer-based Decoder:** It utilizes a Transformer-based decoder (following the Vocos design) that predicts the Short-Time Fourier Transform (STFT) magnitude and phase, which are then converted back to the waveform via an iSTFT head.

## 13. MaskGCT

MaskGCT (Masked Generative Codec Transformer) is a fully non-autoregressive (NAR) Text-to-Speech (TTS) system introduced in the paper "MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer."

The system employs a two-stage architecture: Text-to-Semantic (T2S) and Semantic-to-Acoustic (S2A). Both stages utilize a mask-and-predict training paradigm, enabling parallel generation during inference and allowing control over the duration of the synthesized speech.

Semantic Codec is responsible for discretizing semantic information (content and partial prosody) extracted from the 17th layer of the W2v-BERT 2.0 self-supervised learning (SSL) model.

- **Architecture:** It is a Vector Quantized Variational Autoencoder (VQ-VAE) architecture. Both the encoder and decoder are composed of ConvNext blocks (CNN).
- **Tokens per second:** 50 (operating at 50 Hz, derived from 16kHz audio with a hop size of 320).
- **Vocabulary Size:** 8,192 (single codebook).

## 14. MiniMax-Speech Audio Tokenizer Description

MiniMax-Speech is an autoregressive (AR) Transformer-based Text-to-Speech (TTS) system. The audio tokenizer in MiniMax-Speech employs an **Encoder-VQ-Decoder architecture** (a type of VQ-VAE).

- **Tokens per second:** The tokenizer operates at a rate of **25 Hz** (25 tokens per second).
- **Vocab size:** The vocabulary size (codebook size) and the number of quantizers used (e.g., whether RVQ is employed) are **not specified** in the paper.

**What Makes it Unique and Different:**

The specific implementation of the tokenizer and the overall architecture surrounding it present several unique aspects:

1. **CTC Supervision:** A key feature of the tokenizer training is the use of **Connectionist Temporal Classification (CTC) supervision**. According to the authors, this allows the tokenizer to achieve a high compression rate while effectively preserving necessary acoustic details and semantic information.

## Metrics to Compare Speech Tokenizers

Here’s a compact map of the metrics this paper uses to compare (speech) tokenizers, grouped by what they measure.

**1) Reconstruction & perceptual quality (Codec-SUPERB / VERSA)**

- **SDR** — signal-to-distortion ratio; higher means the resynthesized waveform is closer to the reference. 2506.10274v3
- **SI-SNR** — scale-invariant SNR; like SNR but insensitive to overall gain. 2506.10274v3
- **PESQ** — ITU-T P.862 perceptual speech quality (≈MOS-like) in [1, 5]. 2506.10274v3
- **UTMOS** — MOS predictor from VoiceMOS 2022, [1, 5]. 2506.10274v3
- **DNSMOS P.808 / P.835** — MOS predictors targeted at noise-suppression quality, [1, 5]. 2506.10274v3
- **PLCMOS** — MOS focused on packet-loss concealment, [1, 5]. 2506.10274v3
- **STOI** — short-time objective intelligibility in [0, 1]. 2506.10274v3
- **VISQOL** — “virtual listener” objective MOS for speech/audio, [1, 5]. 2506.10274v3
- **SingMOS** — MOS estimator for singing voice, [1, 5]. 2506.10274v3

**Application-level on resynthesis**

- **WER** — word error rate of ASR run on reconstructions (lower is better). 2506.10274v3
- **Speaker similarity** — similarity between speaker embeddings of reference vs. reconstruction (range −1…1). 2506.10274v3

**2) Downstream task metrics (DASB)**

- **ASR / Low-resource ASR:** **WER**. 2506.10274v3
- **Speaker ID / Verification:** **Accuracy** (ID) and **EER** (verification). 2506.10274v3 2506.10274v3
- **Emotion recognition, keyword spotting, intent classification:** **Accuracy**. 2506.10274v3 2506.10274v3
- **Speech enhancement:** **DNSMOS** and **dWER** (WER after enhancement). 2506.10274v3
- **Speech separation:** **DNSMOS**, **dWER**, **Speaker similarity**. 2506.10274v3

**3) Acoustic language-modeling metrics (Zero-Resource / SALMon)**

- **sBLIMP** — grammatical acceptability via perplexity/accuracy on minimal pair sentences. 2506.10274v3
- **sWUGGY** — lexical discrimination (real word vs. phonological non-word). 2506.10274v3
- **Spoken/Topic Story-Cloze (sSC/tSC)** — choose coherent continuation; probes semantic/common-sense and topical coherence. 2506.10274v3
- **SALMon suite:** tests **acoustic consistency** (sensitivity to speaker/gender/sentiment changes) and **sentiment–acoustic alignment** (does acoustic sentiment match content).

## General Trends

The comparison of tokenizers from 2023 to 2025 reveals several significant shifts in methodology, architecture, and objectives.

1. The Shift from Acoustic Fidelity to Semantic Richness

Early tokenizers (e.g., EnCodec, DAC) were primarily acoustic, focusing on high-fidelity waveform reconstruction.1 The dominant trend is the move towards hybrid and linguistic tokenizers, which ensure the tokens capture phonetic content essential for language modeling. This is achieved through:

- **Semantic Distillation (SD):** Utilizing pre-trained Self-Supervised Learning (SSL) models (like HuBERT or WavLM) to guide the tokenizer training (e.g., SpeechTokenizer, Mimi, X-codec series).2
- **Supervised Semantic Tokenization (SST):** Directly incorporating linguistic information, such as ASR objectives (MiniMax-Speech) or text-aware decoding (TaDiCodec), during training.3

2. The Drive for Extreme Compression

To make speech modeling computationally tractable for LLMs, sequence lengths must be reduced. There is a continuous push for lower frame rates (tokens per second).4 While the guide notes 25 Hz as a current standard, the trend is aggressively downward. EnCodec operates up to 150 Hz, whereas recent models like Mimi achieve 12.5 Hz, and TaDiCodec reaches an exceptionally low 6.25 Hz.

3. Simplification of Quantization for LLM Compatibility

To achieve high fidelity, Residual Vector Quantization (RVQ) has been common (e.g., EnCodec, SpeechTokenizer).5 RVQ uses multiple codebooks iteratively, resulting in a complex, multi-stream output.6 However, LLMs prefer a "flat," single stream of tokens. The trend is shifting towards Single Vector Quantization (SVQ) (e.g., WavTokenizer, LSCodec) or Finite Scalar Quantization (FSQ) (e.g., X-codec2, TaDiCodec).

4. Evolving Training Objectives: The Decline of GANs and Rise of Diffusion

Many high-fidelity codecs rely on a complex combination of losses, including Generative Adversarial Networks (GANs), Feature Matching, and Reconstruction losses. GANs can be unstable to train.7 A significant emerging trend is the adoption of Diffusion or Flow Matching objectives (as seen in S3 and TaDiCodec).8 TaDiCodec, notably, is trained end-to-end using only Flow Matching, offering stability without adversarial training.

5. Disentanglement and Factorization for Controllability

A major area of innovation is the explicit disentanglement of speech attributes: Content, Prosody, Timbre (Speaker Identity), and Acoustic Details. This factorization reduces redundancy and enables advanced control, such as zero-shot voice conversion.

- **FACodec** achieves comprehensive factorization using explicit supervision (phoneme labels, F0) and adversarial training.
- **LSCodec** focuses on decoupling the speaker from the content using an entirely unsupervised approach involving speaker perturbation and information bottlenecks.9

6. Architectural Shifts: The Adoption of Transformers

The underlying architectures are moving away from purely Convolutional (CNN) or Recurrent (RNN) designs. Transformers (T) are increasingly integrated into encoders and decoders (e.g., Mimi, X-codec2). Some cutting-edge models, like TaDiCodec, utilize a fully Transformer-based architecture (T/T).10

## Future Predictions: What to Expect in 2026

Extrapolating from these trends, we can anticipate the following developments in speech tokenization by 2026:

1. Sub-10 Hz as the New Standard

The current standard of 25 Hz will likely be considered inefficient by 2026. Building on the success of TaDiCodec (6.25 Hz), we expect state-of-the-art models to operate consistently below 10 Hz. This extreme compression will be crucial for modeling long-form audio efficiently, significantly reducing the computational cost of training speech LLMs.

2. Dominance of Diffusion and Flow Matching over GANs

The shift away from GANs will accelerate. Due to the stability, simplicity (fewer auxiliary losses), and end-to-end trainability demonstrated by models like TaDiCodec, Diffusion and Flow Matching objectives will likely replace GANs as the primary method for training high-fidelity speech tokenizers.

3. Advanced Unsupervised Disentanglement

The next major breakthrough will be combining the strengths of recent models: achieving the comprehensive factorization of FACodec (separating Content, Prosody, Timbre, and Acoustic details) using the unsupervised methodology of LSCodec. By 2026, we predict the development of highly controllable tokenizers that do not rely on explicit labels (like F0 or phonemes), making them more scalable.

4. "Linguistic-First" Tokenizers via Text-Aware Training

The integration of linguistic information (SST) will become standard. Following TaDiCodec’s text-aware diffusion decoder approach, future tokenizers will integrate text or semantic priors directly into the optimization process. This ensures that even at very low bitrates, the tokens are optimized for intelligibility and downstream language modeling tasks.

5. The Obsolescence of RVQ for Speech Modeling

By 2026, the need for seamless integration with LLMs will mandate the use of flat quantization schemes (SVQ/FSQ). Hierarchical RVQ will likely become obsolete for tasks involving language modeling. The challenge of achieving high fidelity with a single codebook will be solved by powerful diffusion decoders and Transformer architectures.

6. Fully Transformer-Native Architectures

The transition from CNN/RNN and hybrid models will likely complete. State-of-the-art tokenizers in 2026 will predominantly adopt fully Transformer-based architectures for both encoding and decoding, capitalizing on their scalability and effectiveness in modeling long-range dependencies.

## Contributions

If you want to contribute to this guide, feel free to open a pull request with changes to the README.md file.
