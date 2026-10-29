# ============================================================
# MULTIMODAL DEPRESSION DETECTION (TEXT + AUDIO) — V24
# V19 + LoRA ADAPTERS on the frozen HuBERT, trained per CV fold on the depression task (text-distilled)
# Top-K acoustic segment selection BEFORE the audio encoder (edge / bandwidth-aware)
# + honest CV (inner early stopping) + CV hyperparameter search + late fusion + paired significance tests
# ============================================================
# >>> WHERE TO MAKE SMALL CHANGES: search for "<-- CHANGE" (all knobs are in the CONFIG section) <<<
#
# Kaggle: run this in the FIRST cell (it is a comment here so the file stays valid Python)
#   !pip install -q transformers librosa pandas scikit-learn tqdm seaborn matplotlib scipy onnx onnxruntime opensmile
#
# What is new vs V23 (search for "[W1]" ... "[W4]")
#   [W1] Stable decision threshold: the Youden-optimal threshold is computed on THR_BOOT class-stratified bootstrap
#        resamples of the dev out-of-fold predictions and the MEDIAN is used (V21-V23: one argmax on a flat
#        plateau -> specificity jumped 0.66 / 0.74 / 0.54 between runs with the same AUC). Vectorised, so fast.
#        The final report prints the threshold's bootstrap interval and the dev operating point.
#   [W2] The late-fusion audio partner is fixed in advance (LATE_AUDIO_FIXED = "audio_textsel_distill": text-guided
#        Top-K, i.e. the deployed edge pipeline). All candidates are still printed for information.
#   [W3] Labels: the speaker branch is shown as +speaker[timing] (or [eGeMAPS+timing]); HuBERT is labelled
#        "LoRA ... (k/5 folds adapted)" or "LoRA rejected (= frozen)"; duplicate paired tests removed; latency /
#        uplink columns renamed to speaker features.
#   [W4] "AUDIO SUMMARY" block: audio-only AUCs with bootstrap CIs and the net effect of audio in each fusion type.
#
# What was new in V23 (search for "[V1]" ... "[V3]")
#   [V1] The speaker branch (timing [+ eGeMAPS]) is FORCED into every audio / fusion model (AUDIO_GM = True). In V22
#        stage 1 scored a tie and switched it off, so timing never reached a neural model. "fusion_nospk" is the
#        ablation without it.
#   [V2] Timing features are normalised WITHIN the interview condition (Wizard-of-Oz: id < 600, autonomous AI
#        agent: id >= 600): robust centre (median) and scale (1.4826 * MAD) per condition from the DEV speakers only,
#        shrunk towards the pooled statistics (TIMING_SHRINK) because only ~30 dev speakers are AI-agent sessions.
#        No labels are used. The test set (all AI-agent) and the device use the AI statistics. Ablation:
#        "LR-timing-raw" (no normalisation) vs "LR-timing".
#   [V3] Integrated Gradients made complete: (a) Expected-Gradients-style baselines = real chunks of IG_BASELINES
#        random dev speakers (a mean input is constant after standardisation and sits exactly in the
#        scale-invariant point of the LayerNorms, where the path integral cannot see the jump); (b) midpoint rule;
#        (c) steps doubled per baseline until |sum(attr) - (f(x) - f(b))| <= IG_TOL * |f(x) - f(b)| (up to
#        IG_STEPS_MAX); (d) completeness is now reported relative to f(x) - f(baseline), not to f(x).
#
# What was new in V22 (search for "[C1]" ... "[C4]" and "[T1]" ... "[T3]")
#   [C1] Chunk-level (multiple-instance) Stage A. V21 trained on ~140 speaker-level examples per fold; V22 trains the
#        LoRA adapters and the head on single 15-s CHUNKS that inherit their speaker's label (~5,800 per fold).
#        A speaker's score is the 10%-trimmed mean of its chunk log-odds (MIL_AGG). Chunk labels are noisy (not
#        every chunk of a depressed speaker sounds depressed) -> label smoothing MIL_SMOOTH.
#   [C2] Class- and speaker-balanced chunk sampling (class 50/50, then a speaker, then a chunk), so talkative
#        speakers do not dominate; LoRA batches mix chunks of many speakers (better gradients than V21).
#   [C3] Stricter adapter guard: adapters are kept only if their early-stop AUC is also >= the probe's (in V21,
#        fold 0 kept adapters with a LOWER AUC because only the loss was compared). The deployed encoder is the
#        best fold that actually KEPT adapters; if none did, the log says so.
#   [C4] Integrated Gradients baseline = mean training input instead of all-zeros (the zero baseline passes through
#        the scale-invariant LayerNorms, which made V21's completeness error 0.52 instead of < 0.05).
#   [T1] Timing / turn-taking / pause features per speaker from the transcript timestamps and the participant
#        signal: utterance length, words per utterance, speaking rate, gaps between answers, response latency (when
#        interviewer turns are in the transcript), filler rate, pause ratio / rate / length, loudness variability.
#        Cached in timing_features_*.csv; missing values -> dev medians.
#   [T2] They join the speaker-level branch ("speaker" player = eGeMAPS if installed + timing), so every audio /
#        fusion model can use them (stage 1 decides gm=True/False by CV; ablation "fusion_nospk").
#   [T3] "timing_lr" (logistic regression on the timing features, same folds), "late_timing" = text + timing and
#        "late3" = text + audio + timing late fusion; the timing features also run on the device (ONNX demo).
#
# What was new in V21 (search for "[P1]" ... "[P3]") — fixes the V20 Stage A, which did not learn (e2e CV AUC 0.50)
#   [P1] Per-dimension standardisation of the HuBERT layer means inside the Stage-A head (statistics of the fold's
#        fit speakers). V20 only LayerNorm-ed each vector, which leaves HuBERT's large constant dimensions in place;
#        every head that worked before (LR probe 0.615, audio heads ~0.6) standardises per dimension.
#   [P2] LP-FT (linear probing, then fine-tuning; Kumar et al., ICLR 2022): phase 1 trains the head on the cached
#        FROZEN features (all chunks, seconds per fold); phase 2 starts LoRA from that head with a small head LR, so
#        the random-head gradients of V20 no longer scramble the adapters. 8 chunks per step, all chunks for early
#        stopping (V20: 4 and 12 -> very noisy).
#   [P3] Never worse than frozen: epoch 0 of phase 2 (adapters = 0, i.e. the probe) is a candidate for early
#        stopping. If the adapters never beat it on the early-stopping speakers, that fold keeps the frozen features.
#        New entry "audio_probe_e2e" (same head, frozen encoder) isolates the effect of the adapters exactly.
#   Also: LoRA models now ignore a default FRAME_ATTN instead of crashing (frame attention applies to frozen models).
#
# What was new in V20 (search for "[R1]" ... "[R4]")
#   [R1] LoRA adapters. V15-V19 used a FROZEN HuBERT; its depression signal was weak (audio CV AUC ~0.56-0.62).
#        V20 adds low-rank adapters  W' = W + (alpha/r) B A  to the attention projections (q, k, v, out) of the
#        HuBERT transformer layers; the 95M base weights stay frozen, only ~0.6M LoRA weights are trained.
#        Stage A (per CV fold, BEFORE any head is trained): HuBERT+LoRA -> per-chunk layer means -> learnable layer
#        mix -> chunk attention pooling -> classifier, trained on the fold's FIT speakers with class-weighted CE +
#        text-teacher distillation (+ PHQ-8 regression when the labels are found), random chunks + gain + HuBERT's
#        own time masking, early stopping on the fold's inner ES speakers. The adapters are then MERGED into the
#        weights (zero extra inference cost, same ONNX graph) and that fold's adapted encoder re-extracts the
#        features of every speaker. Raw chunk waveforms are cached once as int16 on disk for this.
#   [R2] Honesty: fold f's encoder never sees fold f's validation speakers, its inner ES speakers or any test
#        speaker. Every downstream head of fold f uses fold f's features and the SAME inner ES split as Stage A, so
#        early stopping is always measured on speakers the encoder was not adapted on. Test predictions of the
#        5-fold ensemble use each member's own fold encoder. Ablations "audio_all_frozen" / "fusion_frozen".
#   [R3] "audio_lora_e2e": the Stage-A model itself as an audio classifier (OOF on dev, mean of the 5 fold models
#        on test), and "late_e2e" = late fusion of text with it.
#   [R4] Deployment: ONE adapted encoder (the fold with the best Stage-A early-stopping AUC, merged) plus that
#        fold's heads; its test AUC is reported separately ("deployed single-encoder system").
#   Defaults changed from the V19 run: FRAME_ATTN = "off" (frame attention stayed uniform), the proposed neural
#   fusion has modality dropout OFF (it hurt significantly on dev CV), CHUNK_SEC = 15 and KEEP_FRAC = 0.6 to
#   match your cache and run.
#
# What was new in V19 (search for "[F1]" ... "[F4]")
#   [F1] Frame attention. Up to V18 each 5-s chunk embedding was the plain TIME-MEAN of the encoder frames, so every
#        frame got the same weight regardless of its relevance. Now every chunk keeps a sequence of frames
#        H_1..H_F (FRAME_POOL x 20 ms each) and the model learns which regions of the chunk matter:
#            e_t = v^T tanh(W H_t + b),   alpha_t = exp(e_t) / sum_j exp(e_j),   z = sum_t alpha_t H_t
#        Pipeline per chunk: per-layer standardisation -> layer attention PER FRAME [L1] -> frame attention -> the
#        existing chunk-level encoder + chunk attention pooling. v is zero-initialised, so training starts from the
#        frame mean; frame dropout (FRAME_DROP) masks random frames during training.
#        The V15-V18 cache only stores time-means, so a FRAME CACHE (n_chunks, F, L_f, 768) fp16 is built ONCE with
#        the frozen encoder (raw wav + transcripts must be attached) and read lazily from a .npy memmap on disk.
#        Size ~ n_chunks * F * L_f * 1.5 KB; FRAME_POOL and FRAME_LAYERS keep it manageable (defaults: a few GB).
#   [F2] Ablations "audio_all_framemean" / "fusion_framemean" use the identical frame pipeline with uniform weights
#        (alpha_t = 1/F), so the paired tests isolate the effect of frame attention. "*_layermean" = V17 baseline.
#   [F3] Interpretability: effective number of attended frames per chunk (depressed vs non-depressed), position
#        profile of the weights inside the chunk, Spearman(attention, frame energy) (does it only pick loud frames?),
#        agreement of frame attention with frame-level Integrated Gradients, and an example chunk.
#   [F4] ONNX: audio_encoder.onnx outputs (n_chunks, F, L_f, 768); the heads take (1, K, F, L_f, 768).
#
# What was new in V18 (search for "[L1]" ... "[L4]")
#   [L1] The audio branch no longer averages a fixed window of encoder layers picked by a linear probe. Every chunk
#        carries ALL hidden states (L, 768) — these are already in the V15-V17 cache, so NO re-extraction — and a
#        LayerAttention module learns a softmax weighting over the layers, end-to-end with the classifier:
#          "scalar"  : one learnable logit per layer shared by all chunks (SUPERB / ELMo weighted sum) + a gain
#          "dynamic" : per-chunk weights = softmax(global logits + MLP(layer embedding)); the MLP is
#                      zero-initialised, so training starts exactly from the scalar mix
#        Each layer is LayerNorm-ed before mixing. Layer dropout (LAYER_DROP) masks random layers during training.
#        The layer logits + gain get a higher LR (LAYER_LR_MULT) and no weight decay, so they actually move.
#   [L2] Ablations "audio_all_layermean" and "fusion_layermean" keep the V17 fixed probe-window average; paired
#        tests compare them with the layer-attention models (dev OOF = main evidence, test = information).
#   [L3] Interpretability: learned layer weights per model (mean ± sd over ensemble members, effective number of
#        layers) plotted against the probe AUC, plus Integrated-Gradients mass per layer for the fusion model.
#   [L4] ONNX: audio_encoder.onnx now outputs all layers (n_chunks, L, 768); the heads take (1, K, L, 768) and do
#        the mixing themselves. If the head runs OFF the device, the embedding uplink grows by a factor L.
#   The layer probe still runs: it picks the window for the fixed-mean ablation, the LR baseline and the
#   learned_wavlm scorer input, and can initialise the layer logits (LAYER_INIT = "probe").
#
# Kept from V17: late fusion [N1], modality dropout [N2], text-guided selector [N3], PHQ-8 multi-task [N4],
# diagnosis-shortcut ablation [N5], CV hyperparameter search [N6], non-inferiority tests [N7], final-system choice by
# dev CV [N8], audio-mode K-sweep [N9], inner early stopping, IG / Shapley / deletion interpretability, ONNX export.
#
# HONESTY NOTE: the hyperparameter search and the late-fusion choice are made on dev CV, so the CV AUC of the
# chosen settings is slightly optimistic. The TEST numbers stay unbiased because test is never used to choose.
# LAYER_INIT = "probe" uses a probe fitted on all dev labels -> keep "uniform" for the cleanest CV numbers.
# ============================================================
import os, re, json, copy, random, warnings, zlib, time, shutil
from itertools import combinations
from math import factorial
import numpy as np
import pandas as pd
import torch
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from tqdm import tqdm
from scipy import stats
from scipy.special import logit
from sklearn.metrics import (roc_auc_score, f1_score, matthews_corrcoef, accuracy_score,
                             balanced_accuracy_score, precision_score, recall_score,
                             confusion_matrix, roc_curve, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor

warnings.filterwarnings("ignore", category=UserWarning)
try:
    import opensmile
    HAS_SMILE = True
except ImportError:
    HAS_SMILE = False
    print("opensmile not installed -> eGeMAPS branch disabled, scorer uses log-mel features only")

# ============================================================
# CONFIG
# ============================================================
TEXT_PATH  = "/kaggle/input/datasets/majjivikram/tencon-vikram/E_DAIC"     # <-- CHANGE if your dataset path differs
AUDIO_PATH = "/kaggle/input/datasets/majjivikram/audio-data/Audio"         # <-- CHANGE if your dataset path differs
OUT_DIR    = "/kaggle/working/outputs_v24"
CACHE_ROOT = "/kaggle/working/outputs_v15/cache"   # same folder as V15-V17 -> an existing cache is reused
# Pre-computed feature cache (Train.pt, Val.pt, Test.pt) uploaded as a Kaggle dataset -> skips the ~3 h extraction.
#   "auto" = search /kaggle/input | a path = use that folder | None = extract and cache in CACHE_ROOT
PRECOMPUTED_CACHE_DIR = None    # <-- CHANGE
# [N4] CSV(s) with Participant_ID + PHQ score (+ Gender). "auto" searches the dataset folders and /kaggle/input
# for E-DAIC metadata such as train_split.csv / dev_split.csv / Detailed_PHQ8_Labels.csv. None = off.
PHQ_CSV = "auto"                # <-- CHANGE to a file path if "auto" finds nothing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoders (changing these invalidates the cache -> full re-extraction)
TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"
TEXT_POOLING = "mean_norm" if TEXT_MODEL.startswith("sentence-transformers") else "cls_mean"
WAVLM_NAME = "facebook/hubert-base-ls960"
# Fixed probe window: used by the layer-mean ablation, the LR baseline and the learned_wavlm scorer
AUTO_LAYERS = True
LAYER_WIN = 2
DEFAULT_LAYERS = (6, 9)
USE_EGEMAPS = HAS_SMILE

# ---- [L1] Learnable layer-wise attention (replaces fixed layer averaging in the audio branch) ----
LAYER_ATTN = True                 # <-- CHANGE: True = learnable weighted sum over layers | False = V17 fixed mean
LAYER_ATTN_KIND = "scalar"        # <-- CHANGE: "scalar" (one weight per layer) | "dynamic" (per-chunk weights)
LAYER_SET = None                  # <-- CHANGE: None = all hidden states (0 = CNN features .. last) | (lo, hi) subset
LAYER_INIT = "uniform"            # <-- CHANGE: "uniform" | "probe" (logits from the probe AUC; mild optimism)
LAYER_DROP = 0.1                  # P(a layer is masked out of the softmax) per training step
LAYER_LR_MULT = 10.0              # layer logits + gain: lr * LAYER_LR_MULT, no weight decay
LAYER_SCORE_HIDDEN = 32           # MLP width for the "dynamic" kind

# ---- [F1] Frame-level attention INSIDE each 5-s chunk (replaces the time-mean over encoder frames) ----
FRAME_ATTN = "off"                # <-- CHANGE: "attn" = learned frame weights | "mean" = uniform | "off" = V18
FRAME_POOL = 10                   # <-- CHANGE: 20-ms encoder frames averaged per attention step (10 -> 200 ms)
FRAME_LAYERS = "even"             # <-- CHANGE: "even" (0,2,..,12) | "all" | tuple of hidden-state indices
FRAME_ATTN_HIDDEN = 64            # size of W in e_t = v^T tanh(W H_t + b)
FRAME_DROP = 0.1                  # P(a frame is masked out of the softmax) per training step
FRAME_CACHE_DIR = None            # <-- CHANGE: folder holding frames.npy + frames_meta.pt (e.g. a Kaggle dataset)
FRAME_MAX_GB = 18.0               # refuse to build a frame cache larger than this (Kaggle /kaggle/working ~ 20 GB)
HUBERT_HOP = 320                  # samples per encoder frame (20 ms at 16 kHz)

# ---- [R1] LoRA adapters on the frozen HuBERT (trained per CV fold on the depression task) ----
LORA = True                       # <-- CHANGE: True = audio models use fold-wise LoRA-adapted HuBERT features
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.1     # <-- CHANGE: rank / scaling / dropout of the adapters
LORA_TARGETS = ["attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.out_proj"]
#   add "feed_forward.intermediate_dense", "feed_forward.output_dense" to also adapt the feed-forward blocks
LORA_LAYERS = "all"               # "all" transformer layers | int k = only the top k layers
LORA_LR, LORA_HEAD_LR, LORA_WD = 2e-4, 1e-3, 1e-2
LORA_EPOCHS, LORA_MIN_EPOCHS, LORA_PATIENCE = 15, 3, 5
LP_EPOCHS, LP_PATIENCE = 40, 8    # [P2] phase 1: head on frozen features
LORA_HEAD_LR_FT = 1e-4            # [P2] phase 2: head LR while the adapters train
LORA_CHUNKS_PER_SPK = 8           # random full chunks per speaker per step
LORA_ACCUM = 2                    # micro-batches per optimizer step (V22: LORA_MICRO chunks each)
LORA_EVAL_CHUNKS = 0              # evenly spaced chunks per early-stopping speaker (0 = all chunks)
LORA_GAIN_DB = 6.0                # random gain augmentation (+- dB)
LORA_TEXT_DISTILL = 0.5           # text teacher -> Stage-A audio model (training speakers only)
LORA_HEAD_HIDDEN, LORA_HEAD_DROP = 64, 0.3
LORA_SEED = 7
# ---- [C1] Chunk-level (multiple-instance) Stage A ----
MIL_AGG = "trim"                  # speaker score = "mean" or 10%-"trim"med mean of the chunk log-odds
MIL_SMOOTH = 0.1                  # label smoothing on chunk labels (chunk labels are noisy copies of the speaker's)
LP_CHUNKS_PER_EPOCH, LP_BATCH = 4096, 64      # phase 1 (cached frozen features)
LORA_MICRO = 8                    # chunks per forward / backward through HuBERT
LORA_CHUNKS_PER_EPOCH = 1600      # chunks per LoRA epoch, then an early-stopping check
# ---- [T1] Timing / turn-taking / pause features ----
USE_TIMING = True                 # <-- CHANGE
TIMING_BY_CONDITION = True        # <-- CHANGE: [V2] normalise timing within interview condition (WoZ / AI)
TIMING_SHRINK = 10.0              # [V2] pseudo-count pulling a condition's statistics towards the pooled ones
PAUSE_DB = 30.0                   # frame is a pause if its energy < 95th-percentile energy - PAUSE_DB
MIN_PAUSE_S = 0.25                # shortest pause that counts
WAVE_CACHE_DIR = None             # <-- CHANGE: folder with waves.npy + waves_meta.pt from an earlier run
LORA_CACHE_DIR = None             # <-- CHANGE: folder with lora_fold{0..4}.pt from an earlier run (skips Stage A)
WAVE_MAX_GB = 14.0                # refuse to build a waveform cache larger than this
AMP = torch.cuda.is_available()   # mixed precision for Stage A

# Preprocessing (must match the cache)
SR = 16000
CHUNK_SEC = 15                   # must equal the value used when the cache was built (your V19 run: 15)
CHUNK_SIZE = SR * CHUNK_SEC
MAX_AUDIO_CHUNKS = 120
USE_PARTICIPANT_SEGMENTS = True
MIN_WORDS = 1
MAX_SENTS = 400
EMB_BATCH, AUDIO_BATCH = 32, 16

# ---- Top-K acoustic segment selection ----
SELECTOR = "text"                 # <-- CHANGE: text | learned | learned_wavlm | energy | random | first | all
KEEP_FRAC = 0.6                   # <-- CHANGE: K = ceil(KEEP_FRAC * T); 1.0 = no selection
K_MIN = 2
LEARNED = ("learned", "learned_wavlm")
CHEAP_EGEMAPS = HAS_SMILE
N_MELS, MEL_NFFT, MEL_HOP = 64, 400, 160
SCORER_HIDDEN = 32
GUMBEL_SCALE = 1.0
TAU_START, TAU_END = 1.0, 0.1
SEL_DISTILL = 0.1

# ---- Honest CV + audio-branch ablation ----
INNER_ES = True
INNER_ES_FRAC = 0.2
ES_METRIC = "loss"                # "loss" | "auc" | "fixed"
ES_START = 5
FIXED_EPOCHS = 50
AUDIO_GM = True                   # <-- CHANGE: [V1] True = speaker branch (timing [+eGeMAPS]) always on
AUDIO_ENC = "linear"              # <-- CHANGE: None = pick by CV in stage 1 | "lstm" / "linear" / "stats" = force
AUDIO_STD = True
RUN_ABLATION = True
ABL_SEEDS = [0]

# ---- [N6] Hyperparameters + CV search ----
HP_DEFAULT = dict(hidden=64, in_drop=0.2, head_drop=0.3, lr=2e-4, wd=1e-2, act="gelu", smooth=0.0)
RUN_HPO = True                    # <-- CHANGE: False = use HP_DEFAULT everywhere (much faster)
HPO_TRIALS = 10                   # <-- CHANGE: trials per model type (trial 0 = HP_DEFAULT); 5 folds x seeds each
HPO_SEEDS = [0]
HPO_SEED = 123
HPO_SPACE = dict(hidden=[32, 64, 128],
                 in_drop=[0.1, 0.2, 0.3, 0.4],
                 head_drop=[0.2, 0.3, 0.5],
                 lr=[1e-4, 2e-4, 5e-4, 1e-3],
                 wd=[1e-3, 1e-2, 5e-2],
                 act=["gelu", "relu", "silu", "mish"],
                 smooth=[0.0, 0.05, 0.1])
HP = {"text": dict(HP_DEFAULT), "audio": dict(HP_DEFAULT), "fusion": dict(HP_DEFAULT)}   # filled by the search

# ---- [N2] Modality dropout (neural fusion, training only) ----
MOD_DROP_AUDIO = 0.4              # <-- CHANGE: P(audio + eGeMAPS features zeroed for a training sample)
MOD_DROP_TEXT = 0.1               # <-- CHANGE: P(text features zeroed for a training sample)

# ---- [N4] PHQ-8 multi-task ----
PHQ_AUX = 1.0                     # <-- CHANGE: weight of MSE(sigmoid(phq head), PHQ/24); 0 = off
PHQ_MAX = 24.0

# ---- Text -> audio distillation (training only) ----
TEXT_DISTILL = 0.5
TEXT_DISTILL_T = 2.0
AUDIO_AUX = 0.3
TEACHER_C = 0.05

# ---- [N1] Late fusion ----
LATE_C = 1.0
LATE_AUDIO_CANDIDATES = ["audio_textsel_distill", "audio_textsel", "audio_all"]
LATE_AUDIO_FIXED = "audio_textsel_distill"   # <-- CHANGE: [W2] fixed audio partner (None = choose by dev CV)
NONINF_MARGIN = 0.05              # [N7] non-inferiority margin in AUC

# ---- [N5] Diagnosis-shortcut words ----
DIAG_WORDS = {"therapy", "therapist", "therapists", "ptsd", "diagnosed", "diagnosis", "medication",
              "medications", "counseling", "counselor", "psychiatrist", "psychologist", "depression",
              "meds", "treatment", "antidepressant", "antidepressants", "prozac", "zoloft"}


def cfg(name, mode, selector="all", frac=1.0, gm=None, enc=None, distill=False, moddrop=None,
        text_key="text", hp=None, la=None, fa=None, lo=None):
    """gm / enc: None = stage-1 choice. moddrop: None = on for fusion. text_key: "text" | "text_nd" (shortcut-free).
    hp: None = the searched hyperparameters for this mode. la: None = LAYER_ATTN | True / False = force [L1].
    fa: None = FRAME_ATTN | "attn" / "mean" / "off" = force [F1]. lo: None = LORA | True / False = force [R1]."""
    return dict(name=name, mode=mode, selector=selector, frac=frac, gm=gm, enc=enc, distill=distill,
                moddrop=moddrop, text_key=text_key, hp=hp, la=la, fa=fa, lo=lo)


# Main experiments. "text" must come first: its fold models drive the text-guided selector.
MAIN_CONFIGS = [cfg("text", "text"),
                cfg("text_nodiag", "text", text_key="text_nd"),                          # [N5] shortcut ablation
                cfg("audio_all", "audio", "all", 1.0),                                   # LoRA-HuBERT features
                cfg("audio_all_frozen", "audio", "all", 1.0, lo=False),                  # [R2] frozen HuBERT
                cfg("audio_learned", "audio", "learned", KEEP_FRAC),
                cfg("audio_textsel", "audio", "text", KEEP_FRAC),
                cfg("audio_textsel_distill", "audio", "text", KEEP_FRAC, distill=True),
                cfg("fusion", "fusion", SELECTOR, KEEP_FRAC, distill=True, moddrop=False),     # proposed neural fusion
                cfg("fusion_frozen", "fusion", SELECTOR, KEEP_FRAC, distill=True, moddrop=False, lo=False),  # [R2]
                cfg("fusion_moddrop", "fusion", SELECTOR, KEEP_FRAC, distill=True, moddrop=True),   # [N2] ablation
                cfg("fusion_all", "fusion", "all", 1.0, distill=True, moddrop=False),               # no selection
                cfg("fusion_nospk", "fusion", SELECTOR, KEEP_FRAC, distill=True, moddrop=False, gm=False)]  # [T2]
# Optional older ablations (no gain in the V19 run):
# cfg("audio_all_layermean", "audio", "all", 1.0, la=False, fa="off", lo=False),
# cfg("fusion_framemean", "fusion", SELECTOR, KEEP_FRAC, distill=True, fa="mean", lo=False),

# K-sweep: accuracy vs fraction of audio kept, per selector
RUN_SWEEP = True                   # <-- CHANGE: False for a faster run
SWEEP_MODE = "audio"               # "audio" shows selector effects; "fusion" hides them behind text
SWEEP_FRACS = [0.25, 0.5, 0.75]
SWEEP_SEEDS = [0, 1]
SWEEP_VARIANTS = [("text-guided", "text"), ("learned", "learned"), ("learned_wavlm", "learned_wavlm"),
                  ("energy", "energy"), ("random", "random"), ("first", "first")]
SWEEP_REF = "text-guided"


def _fa(v, default="off"):
    """Frame-attention setting -> "attn" | "mean" | "off" (True = "attn", False = "off", None = default)."""
    v = default if v is None else v
    v = "attn" if v is True else "off" if v is False else str(v)
    if v not in ("attn", "mean", "off"):
        raise ValueError(f"Unknown frame-attention setting {v}")
    return v


FRAME_ATTN = _fa(FRAME_ATTN)
# Which feature stores are needed: frame cache [F1] and / or per-layer chunk means on the GPU [L1].
# "_default" stands for the stage-1 ablation, HPO and K-sweep configs, which use the default settings.
_audio_cfgs = [c for c in MAIN_CONFIGS if c["mode"] != "text"] + [cfg("_default", "audio")]
_eff_la = lambda c: LAYER_ATTN if c.get("la") is None else bool(c["la"])
_eff_lo = lambda c: LORA if c.get("lo") is None else bool(c["lo"])
NEED_FRAMES = any(_fa(c.get("fa"), FRAME_ATTN) != "off" and not (_eff_lo(c) and c.get("fa") is None)
                  for c in _audio_cfgs)
NEED_LORA = any(_eff_lo(c) for c in _audio_cfgs)                                              # [R1]
NEED_LAYERS = any(_eff_la(c) and _fa(c.get("fa"), FRAME_ATTN) == "off" and not _eff_lo(c) for c in _audio_cfgs)

# Model
SEQ_LEN, STRIDE = 50, 30
GM_DIM = 32

# Training / evaluation
EPOCHS, MIN_EPOCHS, PATIENCE = 50, 12, 10
ACCUM = 1
N_FOLDS, CV_SEED = 5, 42
SEEDS = [0, 1, 2]                 # <-- CHANGE: [0] for a quick run
THR_METRIC = "youden"
THR_STABLE = True                 # <-- CHANGE: [W1] median of bootstrap-optimal thresholds (False = single argmax)
THR_BOOT, THR_SEED = 300, 2024
AUGMENT, AUG_FRAC = True, (0.6, 1.0)
N_VIEWS, TTA_FRAC = 5, 0.8
DETERMINISTIC = True

# Interpretability
IG_STEPS = 32
IG_STEPS_MAX = 256                # [V3] steps are doubled per baseline until complete (or this cap)
IG_BASELINES = 4                  # [V3] real dev speakers used as baselines (Expected-Gradients style)
IG_TOL = 0.05                     # [V3] completeness tolerance relative to |f(x) - f(baseline)|
TOPK_SENTS = 12
DEL_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_RANDOM_DEL = 5
N_BOOT = 1000
OPSET = 16

_tag = lambda s: re.sub(r"[^A-Za-z0-9]+", "-", s)
CACHE_DIR = os.path.join(CACHE_ROOT,
                         f"{_tag(TEXT_MODEL)}_{_tag(WAVLM_NAME)}_mw{MIN_WORDS}_ps{int(USE_PARTICIPANT_SEGMENTS)}"
                         f"_gm{int(USE_EGEMAPS)}_c{CHUNK_SEC}_mx{MAX_AUDIO_CHUNKS}_mel{N_MELS}_ce{int(CHEAP_EGEMAPS)}")
FIG_DIR, ONNX_DIR = os.path.join(OUT_DIR, "figures"), os.path.join(OUT_DIR, "onnx")
for d in (OUT_DIR, CACHE_DIR, FIG_DIR, ONNX_DIR):
    os.makedirs(d, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = DETERMINISTIC
    torch.backends.cudnn.benchmark = False


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches="tight")
    plt.show(); plt.close()


def make_act(name):
    return {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "mish": nn.Mish, "elu": nn.ELU}[name]()


sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x, float)))
tokenize = lambda s: re.findall(r"[a-z']+", s.lower())

# ============================================================
# FROZEN ENCODERS
# ============================================================
class TextEncoder(nn.Module):
    def __init__(self, enc, pooling):
        super().__init__(); self.enc, self.pooling = enc, pooling

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        if self.pooling == "mean_norm":
            return mean / mean.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return (h[:, 0] + mean) / 2


class AudioEncoder(nn.Module):
    """forward(): (B, 768) fixed layer mean | (B, L, 768) per-layer time-means when all_layers=True [L1]
    | (B, N_FRAMES, L_f, 768) pooled frames of the layers frame_ids when frame_ids is set [F4]."""
    def __init__(self, wavlm, normalize, l0, l1):
        super().__init__()
        self.wavlm, self.normalize, self.l0, self.l1 = wavlm, normalize, l0, l1
        self.all_layers, self.la0, self.la1 = False, 0, None
        self.frame_ids, self.n_frames = None, None                  # [F4] frame-level export

    def hidden_states(self, x):
        if self.normalize:
            x = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-7)
        return torch.stack(self.wavlm(x, output_hidden_states=True).hidden_states)

    def forward(self, x):
        h = self.hidden_states(x)                                   # (L, B, frames, 768)
        if self.frame_ids is not None:                              # [F4] (B, N_FRAMES, L_f, 768)
            return pool_frames(h, self.frame_ids, self.n_frames)
        if self.all_layers:
            return h[self.la0:self.la1].mean(2).permute(1, 0, 2)    # (B, L, 768), same as the cache
        return h[self.l0:self.l1].mean(0).mean(1)


def _load(cls, name):
    try:
        return cls.from_pretrained(name, attn_implementation="eager")
    except (TypeError, ValueError):
        return cls.from_pretrained(name)


print("Loading encoders")
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
feature_extractor = AutoFeatureExtractor.from_pretrained(WAVLM_NAME)
WAVLM_NORMALIZE = bool(getattr(feature_extractor, "do_normalize", False))
text_encoder = TextEncoder(_load(AutoModel, TEXT_MODEL), TEXT_POOLING).to(device).eval()
audio_encoder = AudioEncoder(_load(AutoModel, WAVLM_NAME), WAVLM_NORMALIZE, *DEFAULT_LAYERS).to(device).eval()
for p in list(text_encoder.parameters()) + list(audio_encoder.parameters()):
    p.requires_grad = False
D_TEXT = text_encoder.enc.config.hidden_size

# [L1] layers that enter the learnable layer attention
N_LAY_ALL = audio_encoder.wavlm.config.num_hidden_layers + 1
LA0, LA1 = (0, N_LAY_ALL) if LAYER_SET is None else (int(LAYER_SET[0]), int(LAYER_SET[1]))
assert 0 <= LA0 < LA1 <= N_LAY_ALL, f"LAYER_SET must lie inside [0, {N_LAY_ALL}]"
N_LAYERS = LA1 - LA0
audio_encoder.la0, audio_encoder.la1 = LA0, LA1
print(f"[L1] layer attention: {LAYER_ATTN} ({LAYER_ATTN_KIND}) over hidden states {LA0}..{LA1 - 1} "
      f"({N_LAYERS} layers)")

# [F1] frame-level settings
FRAME_IDS = (list(range(0, N_LAY_ALL, 2)) if FRAME_LAYERS == "even" else
             list(range(N_LAY_ALL)) if FRAME_LAYERS == "all" else sorted(int(i) for i in FRAME_LAYERS))
assert FRAME_IDS and all(0 <= i < N_LAY_ALL for i in FRAME_IDS), f"FRAME_LAYERS must lie inside [0, {N_LAY_ALL})"
N_FLAYERS = len(FRAME_IDS)
with torch.no_grad():
    N_RAW_FRAMES = audio_encoder.hidden_states(torch.zeros(1, CHUNK_SIZE, device=device)).shape[2]
N_FRAMES = N_RAW_FRAMES // FRAME_POOL
assert N_FRAMES >= 2, "FRAME_POOL too large: fewer than 2 frames per chunk"
audio_encoder.n_frames = N_FRAMES
print(f"[F1] frame attention: {FRAME_ATTN} | {N_RAW_FRAMES} encoder frames per {CHUNK_SEC}-s chunk -> "
      f"{N_FRAMES} frames of {FRAME_POOL * 20} ms | layers {FRAME_IDS}")


def pool_frames(h, ids, n_frames):
    """(L, B, frames, 768) hidden states -> (B, n_frames, len(ids), 768): FRAME_POOL consecutive frames averaged
    (the trailing frames that do not fill a whole pooled frame are dropped)."""
    h = torch.stack([h[i] for i in ids])[:, :, :n_frames * FRAME_POOL]
    h = h.reshape(h.shape[0], h.shape[1], n_frames, FRAME_POOL, h.shape[-1]).mean(3)
    return h.permute(1, 2, 0, 3)

SMILE = (opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                         feature_level=opensmile.FeatureLevel.Functionals) if HAS_SMILE else None)
GM_NAMES = list(SMILE.feature_names) if HAS_SMILE else []
# [T1] speaker-level timing / pause features (order = feature order everywhere, also on the device)
TRANSCRIPT_TIMING = ["n_utts_log", "speech_time_log", "speech_ratio", "utt_dur_mean", "utt_dur_median",
                     "utt_dur_std", "words_per_utt_mean", "words_per_utt_median", "short_utt_frac", "one_word_frac",
                     "rate_wps_mean", "rate_wps_median", "rate_wps_std", "gap_median_log", "gap_mean_log",
                     "gap_p90_log", "gap_std_log", "long_gap_frac", "latency_median_log", "latency_mean_log",
                     "latency_p90_log", "has_latency", "filler_rate"]
PAUSE_TIMING = ["pause_ratio", "pauses_per_min", "pause_mean_s", "pause_median_s", "voiced_run_mean_s",
                "voiced_db_std", "voiced_db_range"]
TIMING_NAMES = TRANSCRIPT_TIMING + PAUSE_TIMING
# speaker-level vector of the "speaker" branch = eGeMAPS (if openSMILE is installed) + timing features [T2]
SPK_NAMES = (GM_NAMES if USE_EGEMAPS else []) + (TIMING_NAMES if USE_TIMING else [])
SPK_LABEL = "+speaker[" + "+".join((["eGeMAPS"] if USE_EGEMAPS else []) + (["timing"] if USE_TIMING else [])) + "]"
N_GM = len(SPK_NAMES)
CHEAP_NAMES = ([f"logmel{i}_mean" for i in range(N_MELS)] + [f"logmel{i}_std" for i in range(N_MELS)]
               + (GM_NAMES if CHEAP_EGEMAPS else []))
GM_SLICE = slice(2 * N_MELS, 2 * N_MELS + len(GM_NAMES))

# ============================================================
# PREPROCESSING (pure numpy / librosa / openSMILE: identical on the device)
# ============================================================
def load_transcript(csv_path):
    df = pd.read_csv(csv_path)
    lc = {c.lower().strip(): c for c in df.columns}
    if "speaker" in lc:
        df = df[~df[lc["speaker"]].astype(str).str.lower().str.contains("ellie")]
    tcol = lc.get("text", lc.get("value"))
    if tcol is None:
        raise ValueError(f"No text column in {csv_path}")
    texts = df[tcol].fillna("").astype(str).str.strip().str.lower().tolist()
    segs = []
    if "start_time" in lc and "end_time" in lc:
        segs = list(zip(df[lc["start_time"]].astype(float), df[lc["end_time"]].astype(float)))
    sents = [t for t in texts if t not in ("", "nan") and len(t.split()) >= MIN_WORDS][:MAX_SENTS]
    return sents, segs


def _concat_to_orig(s, offsets):
    for pos, a, length in offsets:
        if pos <= s < pos + length:
            return a + (s - pos)
    return s


def load_participant_audio(wav_path, segs):
    """Returns chunks (T, CHUNK_SIZE), chunk start times (s, original recording), participant signal."""
    y, _ = librosa.load(wav_path, sr=SR)
    offsets, y_use = [], y
    if USE_PARTICIPANT_SEGMENTS and segs:
        pieces, pos = [], 0
        for s, e in segs:
            a, b = int(s * SR), min(int(e * SR), len(y))
            if b - a < int(0.3 * SR):
                continue
            pieces.append(y[a:b]); offsets.append((pos, a, b - a)); pos += b - a
        if pieces and pos >= CHUNK_SIZE:
            y_use = np.concatenate(pieces)
        else:
            offsets = []
    if not offsets:
        y_use, offsets = y, [(0, 0, len(y))]
    y_use = librosa.util.normalize(y_use).astype(np.float32)
    n = len(y_use) // CHUNK_SIZE
    if n == 0:
        return np.zeros((0, CHUNK_SIZE), np.float32), np.zeros(0), y_use
    starts = np.arange(n) * CHUNK_SIZE
    if n > MAX_AUDIO_CHUNKS:
        starts = starts[np.unique(np.linspace(0, n - 1, MAX_AUDIO_CHUNKS).round().astype(int))]
    chunks = np.stack([y_use[s:s + CHUNK_SIZE] for s in starts]).astype(np.float32)
    times = np.array([_concat_to_orig(s, offsets) for s in starts]) / SR
    return chunks, times, y_use


def speaker_egemaps(y_use):
    return np.nan_to_num(SMILE.process_signal(y_use, SR).iloc[0].to_numpy(np.float32))


def chunk_energy_db(c):
    return float(10.0 * np.log10(np.mean(c.astype(np.float64) ** 2) + 1e-10))


def chunk_features(chunks):
    """Cheap per-chunk features, computed on the device BEFORE any neural encoder."""
    feats, energy = [], []
    for c in chunks:
        m = librosa.power_to_db(librosa.feature.melspectrogram(y=c, sr=SR, n_fft=MEL_NFFT, hop_length=MEL_HOP,
                                                               n_mels=N_MELS), ref=1.0, top_db=None)
        f = [m.mean(1), m.std(1)]
        if CHEAP_EGEMAPS:
            f.append(np.nan_to_num(SMILE.process_signal(c, SR).iloc[0].to_numpy(np.float32)))
        feats.append(np.concatenate(f).astype(np.float32))
        energy.append(chunk_energy_db(c))
    return np.stack(feats), np.array(energy, np.float32)


def window_index(n, seq_len=SEQ_LEN, stride=STRIDE):
    if n <= seq_len:
        return np.arange(n)[None, :]
    starts = list(range(0, n - seq_len + 1, stride))
    if starts[-1] + seq_len < n:
        starts.append(n - seq_len)
    return np.stack([np.arange(s, s + seq_len) for s in starts])


def eval_views(sid, n_s, n_c, n_views=N_VIEWS, frac=TTA_FRAC):
    """Deterministic test-time views. n_c = number of SELECTED chunks. View 0 = everything kept."""
    rng = np.random.RandomState(zlib.crc32(str(sid).encode()) % (2 ** 31))
    views = [(np.arange(n_s), np.arange(n_c))]
    for _ in range(n_views - 1):
        views.append((np.sort(rng.choice(n_s, max(1, int(round(frac * n_s))), replace=False)),
                      np.sort(rng.choice(n_c, max(1, int(round(frac * n_c))), replace=False))))
    return views


@torch.no_grad()
def embed_sentences(sents):
    out = []
    for i in range(0, len(sents), EMB_BATCH):
        tok = tokenizer(sents[i:i + EMB_BATCH], return_tensors="pt", truncation=True,
                        padding=True, max_length=128).to(device)
        out.append(text_encoder(tok["input_ids"], tok["attention_mask"]).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def embed_audio_layers(chunks):
    out = []
    for i in range(0, len(chunks), AUDIO_BATCH):
        x = torch.from_numpy(chunks[i:i + AUDIO_BATCH]).to(device)
        out.append(audio_encoder.hidden_states(x).mean(2).permute(1, 0, 2).half().cpu())
    L = torch.cat(out)
    return L, L.float().mean(0)


# ============================================================
# CACHE
# ============================================================
def find_transcript(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith("_Transcript.csv"):
                return os.path.join(root, f)
    return None


def _find_precomputed(setting, names=("Train.pt", "Val.pt", "Test.pt")):
    if setting is None:
        return None
    if setting != "auto":
        if not all(os.path.exists(os.path.join(setting, n)) for n in names):
            raise FileNotFoundError(f"PRECOMPUTED_CACHE_DIR={setting} does not contain {names}")
        return setting
    for root, _, files in os.walk("/kaggle/input"):
        if all(n in files for n in names):
            return root
    return None


PRE_DIR = _find_precomputed(PRECOMPUTED_CACHE_DIR)
print("Pre-computed cache folder:", PRE_DIR if PRE_DIR else "none found -> features will be extracted")


def build_split(split):
    for d in ([PRE_DIR] if PRE_DIR else []) + [CACHE_DIR]:
        path = os.path.join(d, f"{split}.pt")
        if os.path.exists(path):
            print(f"Loading cached {split} features from {path}")
            return torch.load(path, weights_only=False)
    path = os.path.join(CACHE_DIR, f"{split}.pt")
    cache = {}
    tdir, adir = os.path.join(TEXT_PATH, split), os.path.join(AUDIO_PATH, split)
    for cat in sorted(os.listdir(tdir)):
        cp = os.path.join(tdir, cat)
        if not os.path.isdir(cp):
            continue
        label = int(cat[0])
        ap = os.path.join(adir, cat)
        wavs = [f for f in os.listdir(ap) if f.endswith(".wav")] if os.path.isdir(ap) else []
        for subj in tqdm(sorted(os.listdir(cp)), desc=f"{split}/{cat}"):
            sid = subj.split("_")[0]
            tcsv = find_transcript(os.path.join(cp, subj))
            if tcsv is None:
                continue
            sents, segs = load_transcript(tcsv)
            if not sents:
                continue
            wav = next((os.path.join(ap, f) for f in wavs if f.split("_")[0] == sid), None)
            if wav is None:
                continue
            chunks, times, y_use = load_participant_audio(wav, segs)
            if len(chunks) == 0:
                continue
            if sid in cache:
                raise RuntimeError(f"Speaker {sid} appears twice in split {split}")
            layers, layer_means = embed_audio_layers(chunks)
            cheap, energy = chunk_features(chunks)
            cache[sid] = dict(
                sid=sid, label=label, split=split, sents=sents, text=embed_sentences(sents),
                audio_layers=layers, layer_means=layer_means, times=times, tcsv=tcsv, wav=wav,
                gm=speaker_egemaps(y_use) if USE_EGEMAPS else None,
                cheap=torch.from_numpy(cheap), energy=energy)
    torch.save(cache, path)
    return cache


print("Building / loading cache:", PRE_DIR or CACHE_DIR)
caches = {s: build_split(s) for s in ["Train", "Val", "Test"]}


def check_cache(caches):
    e = next(iter(caches["Train"].values()))
    exp_cheap = 2 * N_MELS + (len(GM_NAMES) if CHEAP_EGEMAPS else 0)
    got_cheap = e["cheap"].shape[1]
    assert got_cheap == exp_cheap, (
        f"Cached cheap features have {got_cheap} dims, this run expects {exp_cheap} "
        f"(N_MELS={N_MELS}, CHEAP_EGEMAPS={CHEAP_EGEMAPS}). Is opensmile installed? Same N_MELS as when cached?")
    assert (e["gm"] is not None) == bool(USE_EGEMAPS), "Cache eGeMAPS setting differs from USE_EGEMAPS"
    n_lay = N_LAY_ALL
    assert e["audio_layers"].shape[1] == n_lay, f"Cache has {e['audio_layers'].shape[1]} layers, model has {n_lay}"
    n_chunks = max(len(x["times"]) for c in caches.values() for x in c.values())
    assert n_chunks <= MAX_AUDIO_CHUNKS, f"Cache has up to {n_chunks} chunks/speaker > MAX_AUDIO_CHUNKS"
    missing = [x["wav"] for c in caches.values() for x in c.values() if not os.path.exists(x["wav"])]
    if missing:
        print(f"WARNING: {len(missing)} cached wav paths do not exist (e.g. {missing[0]}). "
              "Attach the original audio + transcript datasets at the same paths for the ONNX demo.")
    print(f"Cache check OK: cheap dims {got_cheap}, layers {n_lay}, max chunks/speaker {n_chunks} "
          f"(make sure CHUNK_SEC={CHUNK_SEC} matches the value used when the cache was built)")


check_cache(caches)


# ============================================================
# SENTENCE <-> CHUNK MAPPING (transcript timestamps; no re-extraction)
# ============================================================
def sentence_times(csv_path):
    df = pd.read_csv(csv_path)
    lc = {c.lower().strip(): c for c in df.columns}
    if "speaker" in lc:
        df = df[~df[lc["speaker"]].astype(str).str.lower().str.contains("ellie")]
    tcol = lc.get("text", lc.get("value"))
    if tcol is None or "start_time" not in lc or "end_time" not in lc:
        return None
    texts = df[tcol].fillna("").astype(str).str.strip().str.lower().tolist()
    st, en = df[lc["start_time"]].astype(float).tolist(), df[lc["end_time"]].astype(float).tolist()
    return [(a, b) for t, a, b in zip(texts, st, en)
            if t not in ("", "nan") and len(t.split()) >= MIN_WORDS][:MAX_SENTS]


def chunk_intervals(segs, times):
    offsets, pos = [], 0
    for s, e in segs:
        a, b = int(s * SR), int(e * SR)
        if b - a < int(0.3 * SR):
            continue
        offsets.append((pos, a, b - a)); pos += b - a
    T, starts = len(times), None
    if USE_PARTICIPANT_SEGMENTS and offsets and pos >= CHUNK_SIZE:
        n = pos // CHUNK_SIZE
        starts = np.arange(n) * CHUNK_SIZE
        if n > MAX_AUDIO_CHUNKS:
            starts = starts[np.unique(np.linspace(0, n - 1, MAX_AUDIO_CHUNKS).round().astype(int))]
        if len(starts) != T or not np.allclose([_concat_to_orig(s, offsets) / SR for s in starts], times, atol=0.05):
            starts = None
    out = []
    for i in range(T):
        if starts is None:
            out.append([(float(times[i]), float(times[i]) + CHUNK_SEC)])
            continue
        s0, s1, iv = starts[i], starts[i] + CHUNK_SIZE, []
        for p, a, L in offsets:
            lo, hi = max(s0, p), min(s1, p + L)
            if lo < hi:
                iv.append(((a + lo - p) / SR, (a + hi - p) / SR))
        out.append(iv)
    return out


def overlap_matrix(chunk_iv, sent_iv):
    S = np.asarray(sent_iv, float).reshape(-1, 2)
    dur = np.maximum(S[:, 1] - S[:, 0], 1e-3)
    ov = np.zeros((len(chunk_iv), len(S)), np.float32)
    for i, ivs in enumerate(chunk_iv):
        for a, b in ivs:
            ov[i] += np.clip(np.minimum(b, S[:, 1]) - np.maximum(a, S[:, 0]), 0, None)
    return ov / dur[None]


n_ov, cov = 0, []
for c_ in caches.values():
    for e in c_.values():
        e["ov"] = None
        try:
            st_ = sentence_times(e["tcsv"])
            if st_ is not None and len(st_) == len(e["sents"]):
                _, segs_ = load_transcript(e["tcsv"])
                e["ov"] = overlap_matrix(chunk_intervals(segs_, e["times"]), st_)
                n_ov += 1; cov.append(float((e["ov"].sum(1) > 0).mean()))
        except Exception as ex:
            print("  sentence/chunk mapping failed for", e["sid"], repr(ex))
n_all = sum(len(c_) for c_ in caches.values())
print(f"Sentence-chunk mapping: {n_ov}/{n_all} speakers | chunks overlapping >= 1 sentence: "
      f"{np.mean(cov) if cov else 0:.1%}")



# ============================================================
# [F1] FRAME-LEVEL CACHE for frame attention inside each chunk
#   The V15-V18 cache only holds the TIME-MEAN of each chunk, so frame attention needs one more pass of the frozen
#   encoder over exactly the same chunks: (n_chunks, N_FRAMES, N_FLAYERS, 768) fp16 in a .npy memmap on disk (read
#   lazily: only the selected chunks of one speaker are loaded per training step). Needs the raw wav + transcript
#   files at the cached paths. Build once, then upload the folder as a Kaggle dataset and set FRAME_CACHE_DIR.
#   Also stored: per-speaker feature sums (fast per-fold standardisation) and the energy under every frame.
# ============================================================
FRAME_MM, FRAME_ENERGY, FRAME_OFF, FRAME_SPK = None, None, {}, {}


def _frame_energy_db(chunk):
    """Energy (dB) of the audio under each pooled frame."""
    span = FRAME_POOL * HUBERT_HOP
    return np.array([chunk_energy_db(chunk[j * span:(j + 1) * span + 80]) for j in range(N_FRAMES)], np.float32)


@torch.no_grad()
def build_frame_cache():
    order = [(sp, sid) for sp in ("Train", "Val", "Test") for sid in sorted(caches[sp])]
    T = [len(caches[sp][sid]["times"]) for sp, sid in order]
    offs = np.concatenate([[0], np.cumsum(T)[:-1]]).astype(np.int64)
    n_tot = int(sum(T))
    tag = f"frames_{_tag(WAVLM_NAME)}_p{FRAME_POOL}_n{N_FRAMES}_l{'-'.join(map(str, FRAME_IDS))}"
    want = dict(tag=tag, n_tot=n_tot, sids=[sid for _, sid in order], T=T)
    for d in ([FRAME_CACHE_DIR] if FRAME_CACHE_DIR else []) + [os.path.join(CACHE_DIR, tag)]:
        mp = os.path.join(d, "frames_meta.pt")
        if os.path.exists(mp):
            meta = torch.load(mp, weights_only=False)
            if all(meta.get(k) == v for k, v in want.items()):
                print(f"[F1] loading frame cache from {d}")
                return d, meta
            print(f"[F1] frame cache in {d} does not match this run (settings / speakers / chunks) -> ignored")
    out = os.path.join(CACHE_DIR, tag); os.makedirs(out, exist_ok=True)
    gb = n_tot * N_FRAMES * N_FLAYERS * 768 * 2 / 1e9
    free = shutil.disk_usage(out).free / 1e9
    print(f"[F1] building frame cache: {n_tot} chunks x {N_FRAMES} frames x {N_FLAYERS} layers x 768 "
          f"= {gb:.1f} GB (free disk {free:.1f} GB)")
    if gb > FRAME_MAX_GB or gb > 0.95 * free:
        raise RuntimeError(f"The frame cache needs {gb:.1f} GB. Raise FRAME_POOL, use fewer FRAME_LAYERS, "
                           f"or raise FRAME_MAX_GB / free disk space.")
    mm = np.lib.format.open_memmap(os.path.join(out, "frames.npy"), mode="w+", dtype=np.float16,
                                   shape=(n_tot, N_FRAMES, N_FLAYERS, 768))
    energy, spk = np.zeros((n_tot, N_FRAMES), np.float32), {}
    for (sp, sid), off in zip(tqdm(order, desc="frame cache"), offs):
        e = caches[sp][sid]
        if not (os.path.exists(e["wav"]) and os.path.exists(e["tcsv"])):
            raise FileNotFoundError(f"Raw files of speaker {sid} missing ({e['wav']}). Attach the audio + transcript "
                                    "datasets at the cached paths, or set FRAME_CACHE_DIR to a built frame cache.")
        _, segs = load_transcript(e["tcsv"])
        chunks, times, _ = load_participant_audio(e["wav"], segs)
        if len(chunks) != len(e["times"]) or not np.allclose(times, e["times"], atol=0.05):
            raise RuntimeError(f"Re-chunking speaker {sid} does not reproduce the cached chunks "
                               "(were CHUNK_SEC / MAX_AUDIO_CHUNKS / USE_PARTICIPANT_SEGMENTS changed?)")
        s1, s2 = np.zeros((N_FLAYERS, 768)), np.zeros((N_FLAYERS, 768))
        for i in range(0, len(chunks), AUDIO_BATCH):
            x = torch.from_numpy(chunks[i:i + AUDIO_BATCH]).to(device)
            h = pool_frames(audio_encoder.hidden_states(x), FRAME_IDS, N_FRAMES).half()
            mm[off + i: off + i + len(x)] = h.cpu().numpy()
            hf = h.float()
            s1 += hf.sum((0, 1)).double().cpu().numpy(); s2 += (hf * hf).sum((0, 1)).double().cpu().numpy()
        energy[off: off + len(chunks)] = np.stack([_frame_energy_db(c) for c in chunks])
        spk[sid] = (s1, s2, len(chunks) * N_FRAMES)
    mm.flush(); del mm
    np.save(os.path.join(out, "frame_energy.npy"), energy)
    meta = dict(want, offsets=dict(zip(want["sids"], offs.tolist())), spk=spk, layers=FRAME_IDS,
                pool=FRAME_POOL, n_frames=N_FRAMES)
    torch.save(meta, os.path.join(out, "frames_meta.pt"))       # written last: a partial build is never reused
    return out, meta


if NEED_FRAMES:
    _fdir, _fmeta = build_frame_cache()
    FRAME_MM = np.load(os.path.join(_fdir, "frames.npy"), mmap_mode="r")
    FRAME_ENERGY = np.load(os.path.join(_fdir, "frame_energy.npy"))
    FRAME_OFF, FRAME_SPK = _fmeta["offsets"], _fmeta["spk"]
    print(f"[F1] frame cache ready: {tuple(FRAME_MM.shape)} fp16 at {_fdir}")



# ============================================================
# [T1] TIMING / TURN-TAKING / PAUSE FEATURES (per speaker; identical code runs on the device)
# ============================================================
FILLERS = {"um", "uh", "uhm", "umm", "hmm", "mm", "mhm", "er", "erm", "ah"}
_lg = lambda x: float(np.log1p(max(float(x), 0.0)))


def transcript_timing(csv_path):
    """Timing of the participant's utterances (and response latency if interviewer turns are present)."""
    out = dict.fromkeys(TRANSCRIPT_TIMING, np.nan)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return out
    lc = {c.lower().strip(): c for c in df.columns}
    tcol, scol, ecol = lc.get("text", lc.get("value")), lc.get("start_time"), lc.get("end_time", lc.get("stop_time"))
    if tcol is None or scol is None or ecol is None:
        return out
    st = pd.to_numeric(df[scol], errors="coerce").to_numpy(float)
    en = pd.to_numeric(df[ecol], errors="coerce").to_numpy(float)
    text = df[tcol].fillna("").astype(str).str.lower().tolist()
    ellie = (df[lc["speaker"]].astype(str).str.lower().str.contains("ellie").to_numpy()
             if "speaker" in lc else np.zeros(len(df), bool))
    ok = np.isfinite(st) & np.isfinite(en) & (en > st)
    order = np.argsort(np.where(ok, st, np.inf), kind="stable")
    toks = [tokenize(t) for t in text]
    nw = np.array([len(t) for t in toks])
    part = [i for i in order if ok[i] and not ellie[i] and nw[i] > 0]
    if not part:
        return out
    s_, e_ = st[part], en[part]
    dur, w = e_ - s_, nw[part].astype(float)
    out.update(n_utts_log=_lg(len(part)), speech_time_log=_lg(dur.sum()),
               speech_ratio=float(dur.sum() / max(e_.max() - s_.min(), 1e-3)),
               utt_dur_mean=float(dur.mean()), utt_dur_median=float(np.median(dur)), utt_dur_std=float(dur.std()),
               words_per_utt_mean=float(w.mean()), words_per_utt_median=float(np.median(w)),
               short_utt_frac=float((w <= 3).mean()), one_word_frac=float((w == 1).mean()))
    m = dur >= 0.5
    if m.any():
        r = np.clip(w[m] / dur[m], 0, 8)
        out.update(rate_wps_mean=float(r.mean()), rate_wps_median=float(np.median(r)), rate_wps_std=float(r.std()))
    if len(part) >= 2:
        g = np.clip(s_[1:] - e_[:-1], 0, 60)
        out.update(gap_median_log=_lg(np.median(g)), gap_mean_log=_lg(g.mean()), gap_p90_log=_lg(np.percentile(g, 90)),
                   gap_std_log=float(np.log1p(g).std()), long_gap_frac=float((g > 5).mean()))
    lat, prev = [], None
    for i in order:                                    # participant turn right after an interviewer turn
        if not ok[i]:
            continue
        if prev is not None and ellie[prev] and not ellie[i] and nw[i] > 0:
            lat.append(np.clip(st[i] - en[prev], 0, 20))
        prev = i
    out["has_latency"] = float(len(lat) > 0)
    if lat:
        lat = np.array(lat)
        out.update(latency_median_log=_lg(np.median(lat)), latency_mean_log=_lg(lat.mean()),
                   latency_p90_log=_lg(np.percentile(lat, 90)))
    all_t = [t for i in part for t in toks[i]]
    out["filler_rate"] = float(sum(t in FILLERS for t in all_t) / max(len(all_t), 1))
    return out


def _runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def pause_timing(y):
    """Pauses inside the participant's speech: 20-ms frames below (95th-percentile energy - PAUSE_DB)."""
    out = dict.fromkeys(PAUSE_TIMING, np.nan)
    if y is None or len(y) < SR:
        return out
    n = len(y) // HUBERT_HOP
    fr = np.asarray(y[:n * HUBERT_HOP], np.float64).reshape(n, HUBERT_HOP)
    db = 10.0 * np.log10((fr ** 2).mean(1) + 1e-10)
    sil = db < np.percentile(db, 95) - PAUSE_DB
    p = _runs(sil) * HUBERT_HOP / SR
    p = p[p >= MIN_PAUSE_S]
    v = _runs(~sil) * HUBERT_HOP / SR
    vd = db[~sil]
    out.update(pause_ratio=float(p.sum() / (n * HUBERT_HOP / SR)), pauses_per_min=float(len(p) / (n * HUBERT_HOP / SR / 60)),
               pause_mean_s=float(p.mean()) if len(p) else 0.0, pause_median_s=float(np.median(p)) if len(p) else 0.0,
               voiced_run_mean_s=float(v.mean()) if len(v) else 0.0,
               voiced_db_std=float(vd.std()) if len(vd) > 1 else np.nan,
               voiced_db_range=float(np.percentile(vd, 95) - np.percentile(vd, 5)) if len(vd) > 1 else np.nan)
    return out


def timing_features(tcsv, y_use):
    d = {**transcript_timing(tcsv), **pause_timing(y_use)}
    return np.array([d[k] for k in TIMING_NAMES], np.float64)


TIMING_FILL = np.zeros(len(TIMING_NAMES))
if USE_TIMING:
    _tpath = os.path.join(CACHE_DIR, f"timing_features_v1_db{PAUSE_DB:g}_mp{MIN_PAUSE_S:g}.csv")
    if os.path.exists(_tpath):
        _tdf = pd.read_csv(_tpath, dtype={"sid": str})
        print("[T1] timing features loaded from", _tpath)
    else:
        _rows = []
        for _sp, _c in caches.items():
            for _sid, _e in tqdm(sorted(_c.items()), desc=f"timing {_sp}"):
                _y = None
                if os.path.exists(_e["wav"]):
                    _, _segs = load_transcript(_e["tcsv"])
                    _, _, _y = load_participant_audio(_e["wav"], _segs)
                _rows.append(dict(sid=str(_sid), **dict(zip(TIMING_NAMES, timing_features(_e["tcsv"], _y)))))
        _tdf = pd.DataFrame(_rows)
        _tdf.to_csv(_tpath, index=False)
    _tmap = {str(r_["sid"]): r_[TIMING_NAMES].to_numpy(float) for _, r_ in _tdf.iterrows()}
    _dev_t = np.stack([_tmap[str(sid)] for sp in ("Train", "Val") for sid in caches[sp]])
    TIMING_FILL = np.nan_to_num(np.nanmedian(_dev_t, 0))       # dev medians only (never test)
    for _c in caches.values():
        for _sid, _e in _c.items():
            _t = _tmap.get(str(_sid), np.full(len(TIMING_NAMES), np.nan))
            _e["timing"] = np.where(np.isnan(_t), TIMING_FILL, _t).astype(np.float32)
    print(f"[T1] {len(TIMING_NAMES)} timing features | pause features for {int(np.isfinite(_dev_t[:, -1]).sum())}"
          f"/{len(_dev_t)} dev speakers | response latency available for "
          f"{int(np.nansum(_dev_t[:, TIMING_NAMES.index('has_latency')]))}/{len(_dev_t)} dev speakers")


def interview_condition(sid):
    """[V2] E-DAIC: ids >= 600 were interviewed by the autonomous AI agent, earlier ids by Wizard-of-Oz."""
    try:
        return "AI" if int(float(sid)) >= 600 else "WoZ"
    except (TypeError, ValueError):
        return "AI"


def _robust_stats(X):
    med = np.median(X, 0)
    s = 1.4826 * np.median(np.abs(X - med), 0)
    s = np.where(s > 1e-6, s, X.std(0))
    return med, np.where(s > 1e-6, s, 1.0)


TIMING_NORM = {}
if USE_TIMING and TIMING_BY_CONDITION:
    _dev_items = [(sid, e) for sp in ("Train", "Val") for sid, e in caches[sp].items()]
    _Xr = np.stack([e["timing"] for _, e in _dev_items]).astype(np.float64)
    _cond = np.array([interview_condition(sid) for sid, _ in _dev_items])
    _m_all, _s_all = _robust_stats(_Xr)
    for _cn in ("WoZ", "AI"):
        _Xc = _Xr[_cond == _cn]; _n = len(_Xc)
        _m_c, _s_c = _robust_stats(_Xc) if _n >= 3 else (_m_all, _s_all)
        _lam = _n / (_n + TIMING_SHRINK) if _n >= 3 else 0.0
        TIMING_NORM[_cn] = dict(center=_lam * _m_c + (1 - _lam) * _m_all, scale=_lam * _s_c + (1 - _lam) * _s_all,
                                n=int(_n), weight=float(_lam))
    for _cc in caches.values():
        for _sid, _e in _cc.items():
            _nm = TIMING_NORM[interview_condition(_sid)]
            _e["timing_raw"] = _e["timing"]
            _e["timing"] = np.clip((_e["timing_raw"] - _nm["center"]) / _nm["scale"], -6, 6).astype(np.float32)
    _diff = []
    for _j, _nm_f in enumerate(TIMING_NAMES):              # how different are the two conditions (dev, no labels)?
        _a, _b = _Xr[_cond == "WoZ", _j], _Xr[_cond == "AI", _j]
        if len(_a) > 2 and len(_b) > 2 and (np.ptp(_a) > 0 or np.ptp(_b) > 0):
            _diff.append((_nm_f, stats.mannwhitneyu(_a, _b).pvalue, np.median(_a), np.median(_b)))
    _sig = sorted([d for d in _diff if d[1] < 0.01], key=lambda d: d[1])
    print(f"[V2] timing normalised within condition (dev WoZ n={TIMING_NORM['WoZ']['n']}, AI n={TIMING_NORM['AI']['n']}, "
          f"AI weight {TIMING_NORM['AI']['weight']:.2f}) | {len(_sig)}/{len(_diff)} features differ between "
          "conditions (Mann-Whitney p < 0.01)")
    for _nm_f, _p, _ma, _mb in _sig[:8]:
        print(f"     {_nm_f:22s} WoZ median {_ma:8.3f} | AI median {_mb:8.3f} | p={_p:.1e}")


def speaker_features(tcsv, y_use, sid=None):
    """[T2] Speaker-level vector of the 'speaker' branch (eGeMAPS if available + timing), as on the device."""
    parts = []
    if USE_EGEMAPS:
        parts.append(speaker_egemaps(y_use))
    if USE_TIMING:
        t = timing_features(tcsv, y_use)
        t = np.where(np.isnan(t), TIMING_FILL, t)
        if TIMING_BY_CONDITION:                                # [V2] the device knows its interview condition
            nm = TIMING_NORM[interview_condition(sid)]
            t = np.clip((t - nm["center"]) / nm["scale"], -6, 6)
        parts.append(t)
    return np.concatenate(parts).astype(np.float32)


def speaker_vec(e):
    parts = ([np.asarray(e["gm"], np.float32)] if USE_EGEMAPS else []) + ([e["timing"]] if USE_TIMING else [])
    return torch.from_numpy(np.concatenate(parts)).float().unsqueeze(0).to(device) if parts else None


text_encoder.cpu(); audio_encoder.cpu()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

ids = {k: set(v) for k, v in caches.items()}
for a, b in [("Train", "Val"), ("Train", "Test"), ("Val", "Test")]:
    overlap = ids[a] & ids[b]
    assert not overlap, f"Speaker leakage between {a} and {b}: {sorted(overlap)}"
print("Speaker-disjoint splits verified:", {k: len(v) for k, v in ids.items()})
pd.DataFrame([(sid, s, caches[s][sid]["label"]) for s in caches for sid in caches[s]],
             columns=["speaker_id", "split", "label"]).to_csv(os.path.join(OUT_DIR, "splits.csv"), index=False)

dev_entries = [caches[s][k] for s in ("Train", "Val") for k in sorted(caches[s])]
test_entries = [caches["Test"][k] for k in sorted(caches["Test"])]
ydev = np.array([e["label"] for e in dev_entries]); yte = np.array([e["label"] for e in test_entries])
assert len({e["sid"] for e in dev_entries}) == len(dev_entries)
folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=CV_SEED).split(np.zeros(len(ydev)), ydev))
D_CHEAP = dev_entries[0]["cheap"].shape[1]
print(f"Dev (Train+Val): n={len(ydev)} dep={ydev.sum()} | Test: n={len(yte)} dep={yte.sum()} | {N_FOLDS}-fold CV")


# ============================================================
# [N4] PHQ-8 SCORES + GENDER (metadata CSVs; scores used for TRAINING speakers only)
# ============================================================
def _norm_id(x):
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return str(x).strip()


def load_meta(setting):
    phq, gender = {}, {}
    if setting is None:
        return phq, gender
    if setting != "auto":
        files = [setting]
    else:
        files = []
        for root in dict.fromkeys([TEXT_PATH, AUDIO_PATH, "/kaggle/input"]):
            if not os.path.isdir(root):
                continue
            for r, _, fs in os.walk(root):
                files += [os.path.join(r, f) for f in fs
                          if f.lower().endswith(".csv") and not f[:1].isdigit() and "transcript" not in f.lower()]
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        lc = {c.lower().strip(): c for c in df.columns}
        pid = next((lc[c] for c in lc if "participant" in c), None)
        if pid is None:
            continue
        sc = next((lc[c] for c in lc if re.search(r"phq.*(score|total)", c) and "binary" not in c), None)
        gc = next((lc[c] for c in lc if "gender" in c), None)
        for _, row in df.iterrows():
            k = _norm_id(row[pid])
            if sc is not None and pd.notna(row[sc]):
                phq[k] = float(row[sc])
            if gc is not None and pd.notna(row[gc]):
                gender[k] = row[gc]
    return phq, gender


PHQ_SCORES, GENDER = load_meta(PHQ_CSV)
n_phq = sum(e["sid"] in PHQ_SCORES for e in dev_entries)
print(f"PHQ scores found for {n_phq}/{len(dev_entries)} dev speakers | gender for "
      f"{sum(e['sid'] in GENDER for e in test_entries)}/{len(test_entries)} test speakers")
if n_phq:
    agree = np.mean([(PHQ_SCORES[e["sid"]] >= 10) == e["label"] for e in dev_entries if e["sid"] in PHQ_SCORES])
    print(f"  sanity: PHQ>=10 agrees with folder label for {agree:.1%} of dev speakers (should be ~100%)")
    if agree < 0.9:
        print("  WARNING: PHQ scores do not match the labels -> PHQ multi-task disabled")
        PHQ_SCORES = {}
USE_PHQ = PHQ_AUX > 0 and len(PHQ_SCORES) > 0


# ============================================================
# LAYER PROBE (CV linear probe on the dev pool only)
#   -> fixed window for the layer-mean ablation / LR baseline / learned_wavlm scorer
#   -> optional initialisation of the learnable layer logits [L1]
# ============================================================
def cv_probe_auc(X, y):
    oof = np.zeros(len(y))
    for tr, va in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); oof[va] = clf.predict_proba(X[va])[:, 1]
    return roc_auc_score(y, oof)


Xlay = np.stack([e["layer_means"].numpy() for e in dev_entries])
layer_auc = np.array([cv_probe_auc(Xlay[:, l], ydev) for l in range(Xlay.shape[1])])
if AUTO_LAYERS:
    cands = range(1, Xlay.shape[1] - LAYER_WIN + 1)
    l0 = max(cands, key=lambda s: layer_auc[s:s + LAYER_WIN].mean())
    WAVLM_LAYERS = (l0, l0 + LAYER_WIN)
else:
    WAVLM_LAYERS = DEFAULT_LAYERS
audio_encoder.l0, audio_encoder.l1 = WAVLM_LAYERS
print("Layer probe CV AUC:", np.round(layer_auc, 3), "-> fixed window (layer-mean models)", WAVLM_LAYERS)


def init_layer_logits(ids):
    """[L1] Initial layer logits for the hidden states `ids`: zeros ("uniform") or z-scored probe AUC ("probe")."""
    if LAYER_INIT != "probe":
        return None
    a_ = layer_auc[np.asarray(ids)]
    return torch.tensor(2.0 * (a_ - a_.mean()) / (a_.std() + 1e-6), dtype=torch.float)


if LAYER_INIT == "probe":
    print("[L1] layer logits initialised from the probe:", np.round(init_layer_logits(range(LA0, LA1)).numpy(), 2))
plt.figure(figsize=(5, 3))
plt.plot(range(len(layer_auc)), layer_auc, marker="o")
plt.axvspan(WAVLM_LAYERS[0] - 0.5, WAVLM_LAYERS[1] - 0.5, alpha=0.15, color="tab:orange", label="fixed window")
plt.xlabel("Hidden state (0 = CNN features)"); plt.ylabel("CV AUC (linear probe)")
plt.legend(); plt.title("Layer-wise depression information"); savefig("wavlm_layer_probe.png")


def diag_mask(sents):
    m = np.array([not (set(tokenize(s)) & DIAG_WORDS) for s in sents])
    if not m.any():
        m[0] = True
    return m


for e in dev_entries + test_entries:
    e["audio"] = e["audio_layers"][:, WAVLM_LAYERS[0]:WAVLM_LAYERS[1]].float().mean(1)
    e["nd_mask"] = diag_mask(e["sents"])
    e["text_nd"] = e["text"][torch.from_numpy(e["nd_mask"])]
removed = np.array([1 - e["nd_mask"].mean() for e in dev_entries])
print(f"[N5] diagnosis-related sentences: mean {removed.mean():.1%} per speaker | depressed "
      f"{removed[ydev == 1].mean():.1%} vs non-depressed {removed[ydev == 0].mean():.1%}")


def to_sample(e):
    return dict(sid=e["sid"], y=e["label"], y_t=torch.tensor([e["label"]], device=device),
                text=e["text"].to(device), text_nd=e["text_nd"].to(device), audio=e["audio"].to(device),
                # [L1] all hidden states per chunk (T, N_LAYERS, 768), kept in fp16 to save GPU memory
                audio_L=e["audio_layers"][:, LA0:LA1].contiguous().to(device) if NEED_LAYERS else None,
                f_off=FRAME_OFF.get(e["sid"]) if NEED_FRAMES else None,           # [F1] row in the frame cache
                cheap=e["cheap"].float().to(device), energy=np.asarray(e["energy"]), times=e["times"],
                gm=speaker_vec(e),                                                # [T2] eGeMAPS + timing
                # PHQ scores only for dev speakers (never for test speakers)
                phq=PHQ_SCORES.get(e["sid"]) if (USE_PHQ and e["split"] != "Test") else None)


dev_s, test_s = [to_sample(e) for e in dev_entries], [to_sample(e) for e in test_entries]
assert all(s["phq"] is None for s in test_s)
if NEED_LAYERS:
    mb = sum(s["audio_L"].numel() * 2 for s in dev_s + test_s) / 2 ** 20
    print(f"[L1] per-layer chunk embeddings on {device}: {N_LAYERS} layers, {mb:.0f} MB (fp16)")
GM_BASE = torch.cat([s["gm"] for s in dev_s]).mean(0, keepdim=True) if N_GM > 0 else None
for s, e in zip(dev_s + test_s, dev_entries + test_entries):
    s["ov"], s["tsc"] = e["ov"], None
XTEXT_DEV = np.stack([e["text"].mean(0).numpy() for e in dev_entries])


# ============================================================
# MODEL
# ============================================================
class AttnPool(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, x):
        w = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return (x * w.unsqueeze(-1)).sum(1), w


class LayerAttention(nn.Module):
    """[L1] Learnable weighted sum over encoder layers: (..., L, d) -> (..., d), weights (..., L).
    Works per chunk (B, K, L, d) and per frame (B, K, F, L, d) [F1].
    scalar : w = softmax(logits), one weight per layer shared by all chunks (SUPERB / ELMo scalar mix)
    dynamic: w = softmax(logits + MLP(layer_l)) per chunk; MLP zero-initialised -> starts as the scalar mix.
    Every layer is LayerNorm-ed (no affine) first so layers with large norms cannot dominate the sum."""
    def __init__(self, n_layers, d=768, kind="scalar", drop=0.0, init=None, hidden=LAYER_SCORE_HIDDEN):
        super().__init__()
        if kind not in ("scalar", "dynamic"):
            raise ValueError(f"Unknown LAYER_ATTN_KIND {kind}")
        self.kind, self.drop, self.n = kind, float(drop), int(n_layers)
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        init = torch.zeros(n_layers) if init is None else torch.as_tensor(init, dtype=torch.float)
        self.logits = nn.Parameter(init.clone())
        self.gamma = nn.Parameter(torch.ones(1))
        self.score = None
        if kind == "dynamic":
            self.score = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.score[-1].weight); nn.init.zeros_(self.score[-1].bias)
        self.last_w = None

    def forward(self, x):
        x = self.norm(x)
        s = self.logits.expand(x.shape[:-1])
        if self.score is not None:
            s = s + self.score(x).squeeze(-1)
        if self.training and self.drop > 0:                        # layer dropout (training only)
            keep = torch.rand(self.n, device=x.device) >= self.drop
            if not bool(keep.any()):
                keep[random.randrange(self.n)] = True
            s = s.masked_fill(~keep, float("-inf"))
        w = torch.softmax(s, dim=-1)
        self.last_w = w.detach()
        return self.gamma * (w.unsqueeze(-1) * x).sum(-2), w


class FrameAttention(nn.Module):
    """[F1] Attention pooling over the frames of each chunk (Bahdanau-style additive attention):
        e_t = v^T tanh(W H_t + b),   alpha_t = exp(e_t) / sum_j exp(e_j),   z = sum_t alpha_t H_t
    (..., F, d) -> (..., d), weights (..., F).
    kind="mean" is the uniform baseline (alpha_t = 1/F) through exactly the same code path.
    v is zero-initialised, so training starts from the frame mean and moves away only if that helps.
    Frame dropout (training only) masks random frames of every chunk out of the softmax."""
    def __init__(self, d=768, hidden=FRAME_ATTN_HIDDEN, kind="attn", drop=0.0):
        super().__init__()
        if kind not in ("attn", "mean"):
            raise ValueError(f"Unknown frame-attention kind {kind}")
        self.kind, self.drop = kind, float(drop)
        self.W = self.v = None
        if kind == "attn":
            self.W = nn.Linear(d, hidden)                          # W, b
            self.v = nn.Linear(hidden, 1, bias=False)              # v
            nn.init.zeros_(self.v.weight)
        self.last_a = None

    def forward(self, H):
        e = self.v(torch.tanh(self.W(H))).squeeze(-1) if self.kind == "attn" else torch.zeros_like(H[..., 0])
        if self.training and self.drop > 0:                        # frame dropout (training only)
            keep = torch.rand(e.shape, device=e.device) >= self.drop
            keep[..., 0] |= ~keep.any(-1)                           # keep at least one frame per chunk
            e = e.masked_fill(~keep, float("-inf"))
        a = torch.softmax(e, dim=-1)
        self.last_a = a.detach()
        return (a.unsqueeze(-1) * H).sum(-2), a


class ChunkScorer(nn.Module):
    """Tiny MLP giving one relevance score per chunk.
    'learned' : cheap per-chunk features (runs BEFORE WavLM) | 'learned_wavlm' : WavLM embeddings (no saving)."""
    def __init__(self, d_in, mean, std, act="gelu"):
        super().__init__()
        self.register_buffer("mean", mean.view(1, -1).clone())
        self.register_buffer("std", std.view(1, -1).clone())
        self.body = nn.Sequential(nn.Linear(d_in, SCORER_HIDDEN), make_act(act), nn.Dropout(0.1))
        self.out = nn.Linear(SCORER_HIDDEN, 1)

    def forward(self, x):
        return self.out(self.body(((x - self.mean) / self.std).clamp(-5, 5))).squeeze(-1)


class DepressionNet(nn.Module):
    """Players: 'text', 'wavlm', 'speaker' (eGeMAPS + timing [T2]). The audio input of forward() is the SELECTED chunks.
    text (W, L, D_TEXT) windows · audio (1, K, 768) | layer attention (1, K, N_LAYERS, 768)
    | frame attention (1, K, N_FRAMES, N_FLAYERS, 768)
    · gm (1, N_GM) speaker eGeMAPS."""
    def __init__(self, mode, hp, gm_mean=None, gm_std=None, selector="all", keep_frac=1.0, sc_mean=None,
                 sc_std=None, use_gm=True, enc="lstm", au_mean=None, au_std=None, audio_aux=False,
                 use_phq=False, moddrop=False, text_key="text", layer_attn=False, frame_mode="off"):
        super().__init__()
        self.hp, self.text_key = dict(hp), text_key
        self.ts_fold = None
        self.lora_fold = None                                      # [R1] fold whose adapted encoder feeds this model
        self._last_feats = self._last_hidden = None
        # au_mean / au_std: (768,) for layer-mean input, (N_LAYERS, 768) for layer attention
        self.register_buffer("au_mean", None if au_mean is None else
                             au_mean.reshape((1, 1) + tuple(au_mean.shape)).clone())
        self.register_buffer("au_std", None if au_std is None else
                             au_std.reshape((1, 1) + tuple(au_std.shape)).clone())
        self.mode, self.players = mode, []
        self.enc = enc if mode != "text" else None
        self.use_gm = bool(use_gm) and N_GM > 0 and mode in ("audio", "fusion")
        self.selector = selector if mode != "text" else "all"
        self.keep_frac = float(keep_frac) if mode != "text" else 1.0
        self.moddrop = bool(moddrop) and mode == "fusion"
        self.layered = bool(layer_attn) and mode in ("audio", "fusion")          # [L1]
        self.frame_mode = frame_mode if mode in ("audio", "fusion") else "off"    # [F1]
        self.layer_ids = ((FRAME_IDS if self.frame_mode != "off" else list(range(LA0, LA1)))
                          if self.layered else None)
        self.scorer = self.layer_mix = self.frame_pool = None
        Hd = int(hp["hidden"]); H = 2 * Hd; dims = {}
        act = lambda: make_act(hp["act"])
        if mode in ("text", "fusion"):
            self.players.append("text"); dims["text"] = H
            self.text_norm, self.text_drop = nn.LayerNorm(D_TEXT), nn.Dropout(hp["in_drop"])
            self.text_lstm = nn.LSTM(D_TEXT, Hd, batch_first=True, bidirectional=True)
            self.sent_attn, self.win_attn = AttnPool(H), AttnPool(H)
        if mode in ("audio", "fusion"):
            self.players.append("wavlm"); dims["wavlm"] = H
            if self.layered:
                self.layer_mix = LayerAttention(len(self.layer_ids), 768, LAYER_ATTN_KIND, LAYER_DROP,
                                                init_layer_logits(self.layer_ids))
            if self.frame_mode != "off":
                self.frame_pool = FrameAttention(768, FRAME_ATTN_HIDDEN, self.frame_mode, FRAME_DROP)
            self.audio_norm, self.audio_drop = nn.LayerNorm(768), nn.Dropout(hp["in_drop"])
            if enc == "lstm":
                self.audio_seq = nn.LSTM(768, Hd, batch_first=True, bidirectional=True)
            elif enc == "linear":
                self.audio_seq = nn.Sequential(nn.Linear(768, H), act(), nn.Dropout(hp["in_drop"]))
            elif enc == "stats":
                self.audio_seq = nn.Sequential(nn.Dropout(hp["in_drop"]), nn.Linear(2 * 768, H), act())
            else:
                raise ValueError(f"Unknown audio encoder {enc}")
            self.audio_attn = AttnPool(H)
            if self.selector in LEARNED:
                self.scorer = ChunkScorer(sc_mean.numel(), sc_mean, sc_std, act=hp["act"])
            if self.use_gm:
                self.players.append("speaker"); dims["speaker"] = GM_DIM
                self.register_buffer("gm_mean", gm_mean.view(1, -1).clone())
                self.register_buffer("gm_std", gm_std.view(1, -1).clone())
                self.gm_mlp = nn.Sequential(nn.Dropout(0.2), nn.Linear(N_GM, GM_DIM), act())
        d_f = sum(dims[p] for p in self.players)
        self.audio_aux = nn.Linear(H, 2) if (audio_aux and mode == "fusion") else None
        self.head_body = nn.Sequential(nn.LayerNorm(d_f), nn.Dropout(hp["head_drop"]), nn.Linear(d_f, 64),
                                       act(), nn.Dropout(hp["head_drop"]))
        self.head_out = nn.Linear(64, 2)
        self.phq_out = nn.Linear(64, 1) if use_phq else None     # [N4] training-only PHQ regression head

    def chunk_scores(self, s, cand):
        x = s["cheap"] if self.selector == "learned" else s["audio"]
        return self.scorer(x[torch.as_tensor(cand, device=x.device)])

    def encode_text(self, text):
        h, _ = self.text_lstm(self.text_drop(self.text_norm(text)))
        win, w_sent = self.sent_attn(h)
        t, w_win = self.win_attn(win.unsqueeze(0))
        return t, w_sent, w_win.squeeze(0)

    def encode_audio(self, audio, st=None):
        if self.au_mean is not None:
            audio = ((audio - self.au_mean) / self.au_std).clamp(-5, 5)
        if self.frame_mode != "off":                               # [F1] (1, K, F, L_f, 768) -> (1, K, 768)
            audio = self.layer_mix(audio)[0] if self.layered else audio.mean(-2)   # mix the layers of every frame
            audio, _ = self.frame_pool(audio)                      # attention over the frames of each chunk
        elif self.layered:                                         # [L1] (1, K, L, 768) -> (1, K, 768)
            audio, _ = self.layer_mix(audio)
        if self.enc == "stats":
            x = self.audio_norm(audio)
            w = st.view(1, -1, 1) if st is not None else torch.ones_like(x[..., :1])
            w = w / w.sum(1, keepdim=True)
            mu = (x * w).sum(1)
            sd = (((x - mu.unsqueeze(1)) ** 2) * w).sum(1).clamp(min=1e-6).sqrt()
            a = self.audio_seq(torch.cat([mu, sd], -1))
            return a, torch.softmax(torch.zeros_like(x[0, :, 0]), -1)
        x = self.audio_drop(self.audio_norm(audio))
        if st is not None:
            x = x * st.view(1, -1, 1)
        h = self.audio_seq(x)[0] if self.enc == "lstm" else self.audio_seq(x)
        a, w = self.audio_attn(h)
        return a, w.squeeze(0)

    def encode_gm(self, gm):
        return self.gm_mlp(((gm - self.gm_mean) / self.gm_std).clamp(-5, 5))

    def player_feats(self, text, audio, gm=None, st=None):
        f, att = {}, dict(w_sent=None, w_win=None, w_audio=None)
        if "text" in self.players:
            f["text"], att["w_sent"], att["w_win"] = self.encode_text(text)
        if "wavlm" in self.players:
            f["wavlm"], att["w_audio"] = self.encode_audio(audio, st)
        if "speaker" in self.players:
            f["speaker"] = self.encode_gm(gm)
        return f, att

    def classify(self, f):
        h = self.head_body(torch.cat([f[p] for p in self.players], -1))
        self._last_hidden = h if self.training else None
        return self.head_out(h)

    def forward(self, text, audio, gm=None, return_attn=False, st=None):
        f, att = self.player_feats(text, audio, gm, st)
        self._last_feats = dict(f) if self.training else None
        if self.training and self.moddrop:                         # [N2] modality dropout (training only)
            r = random.random()
            drop = (("wavlm", "speaker") if r < MOD_DROP_AUDIO else
                    ("text",) if r < MOD_DROP_AUDIO + MOD_DROP_TEXT else ())
            f = {p: (torch.zeros_like(v) if p in drop else v) for p, v in f.items()}
        logits = self.classify(f)
        return (logits, att) if return_attn else logits


def zscore(x, dim=-1):
    mu = x.mean(dim, keepdim=True)
    return (x - mu) / ((x - mu).pow(2).mean(dim, keepdim=True).sqrt() + 1e-6)


class Ensemble(nn.Module):
    """Averages member logits and attention maps. Selection is SHARED across members."""
    def __init__(self, members):
        super().__init__(); self.members = nn.ModuleList(members)
        self.selector, self.keep_frac = members[0].selector, members[0].keep_frac
        self.text_key = members[0].text_key
        self.layered = getattr(members[0], "layered", False)
        self.frame_mode = getattr(members[0], "frame_mode", "off")
        self.ts_fold = None
        # [R1] members of different folds read different (fold-adapted) audio features -> evaluate them one by one
        lf = {getattr(m, "lora_fold", None) for m in members}
        self.lora_fold = next(iter(lf)) if len(lf) == 1 else None
        self.per_member = len(lf) > 1 and any(f is not None for f in lf)

    def chunk_scores(self, s, cand):
        return zscore(torch.stack([m.chunk_scores(s, cand) for m in self.members])).mean(0)

    def forward(self, text, audio, gm=None, return_attn=False, st=None):
        outs = [m(text, audio, gm, return_attn=True) for m in self.members]
        logits = torch.stack([o[0] for o in outs]).mean(0)
        if not return_attn:
            return logits
        att = {k: (torch.stack([o[1][k] for o in outs]).mean(0) if outs[0][1][k] is not None else None)
               for k in outs[0][1]}
        return logits, att


def logodds(logits):
    return logits[..., 1] - logits[..., 0]


def audio_feats(model, s, idx):
    """Audio-branch input for chunks idx: (K, N_FRAMES, N_FLAYERS, 768) with frame attention [F1] (read from the
    on-disk frame cache), (K, N_LAYERS, 768) with layer attention [L1], else (K, 768)."""
    f = getattr(model, "lora_fold", None)
    if f is not None:                                              # [R1] features of fold f's adapted encoder
        x = LORA_FEATS[f][s["sid"]][torch.as_tensor(np.asarray(idx, dtype=np.int64))]
        x = x[:, LA0:LA1] if getattr(model, "layered", False) else x[:, WAVLM_LAYERS[0]:WAVLM_LAYERS[1]].float().mean(1)
        return x.to(device).float()
    if getattr(model, "frame_mode", "off") != "off":
        rows = s["f_off"] + np.asarray(idx, dtype=np.int64)
        return torch.from_numpy(np.ascontiguousarray(FRAME_MM[rows])).to(device).float()
    i = torch.as_tensor(np.asarray(idx), device=device)
    if getattr(model, "layered", False):
        return s["audio_L"][i].float()
    return s["audio"][i]


# ============================================================
# TOP-K SELECTION
# ============================================================
def n_keep(n, frac):
    return int(n) if frac >= 1.0 else int(min(n, max(K_MIN, np.ceil(frac * n - 1e-9))))


def _sid_rng(sid):
    return np.random.RandomState((zlib.crc32(str(sid).encode()) + 1) % (2 ** 31))


def text_chunk_scores(model, s, cand):
    """Text-guided chunk scores. Dev: the model's own fold's text models; test / ensembles: all text models."""
    t = s["tsc"]
    if t is None:
        raise RuntimeError("Text-guided scores missing: run the 'text' config first (build_text_scores).")
    f = getattr(model, "ts_fold", None)
    row = t[f] if (f is not None and t.shape[0] > 1) else t.mean(0)
    e = np.asarray(s["energy"], float)
    return row[cand] + 1e-9 * (e[cand] - e.min())


@torch.no_grad()
def pick(model, s, cand=None):
    """Hard Top-K used at inference (and on the device). Returns chunk indices in temporal order."""
    cand = np.arange(len(s["audio"])) if cand is None else np.sort(np.asarray(cand))
    k = n_keep(len(cand), model.keep_frac)
    if model.selector == "all" or k >= len(cand):
        return cand
    if model.selector == "first":
        return cand[:k]
    if model.selector == "random":
        return np.sort(_sid_rng(s["sid"]).choice(cand, k, replace=False))
    if model.selector == "energy":
        sc = s["energy"][cand]
    elif model.selector == "text":
        sc = text_chunk_scores(model, s, cand)
    else:
        sc = model.chunk_scores(s, cand).float().cpu().numpy()
    return np.sort(cand[np.argsort(-sc, kind="stable")[:k]])


def relaxed_topk(keys, k, tau):
    khot, r = torch.zeros_like(keys), keys
    for _ in range(k):
        a = torch.softmax(r / tau, dim=-1)
        khot = khot + a
        r = r + torch.log((1.0 - a).clamp(min=1e-10))
    return khot


def train_select(model, s, cand, tau, rng):
    """Training-time selection -> (selected indices, straight-through weights or None, auxiliary loss)."""
    k = n_keep(len(cand), model.keep_frac)
    if model.selector in LEARNED:
        scores = model.chunk_scores(s, cand)
        if k >= len(cand):
            return cand, None, 0.0
        u = torch.rand_like(scores).clamp(1e-6, 1 - 1e-6)
        keys = scores + GUMBEL_SCALE * (-torch.log(-torch.log(u)))
        soft = relaxed_topk(keys, k, tau)
        top = torch.topk(keys, k).indices.sort().values
        w = soft[top]
        st = 1.0 + w - w.detach()
        aux = 0.0
        if SEL_DISTILL > 0:
            with torch.no_grad():
                _, w_full = model.encode_audio(audio_feats(model, s, cand).unsqueeze(0))
            aux = SEL_DISTILL * (w_full * (w_full.clamp(min=1e-8).log() - torch.log_softmax(scores, -1))).sum()
        return cand[top.cpu().numpy()], st, aux
    if model.selector == "all" or k >= len(cand):
        return cand, None, 0.0
    if model.selector == "first":
        return cand[:k], None, 0.0
    if model.selector == "random":
        return np.sort(rng.choice(cand, k, replace=False)), None, 0.0
    if model.selector == "energy":
        return np.sort(cand[np.argsort(-s["energy"][cand], kind="stable")[:k]]), None, 0.0
    if model.selector == "text":
        return np.sort(cand[np.argsort(-text_chunk_scores(model, s, cand), kind="stable")[:k]]), None, 0.0
    raise ValueError(f"Unknown selector {model.selector}")


def head_forward(model, s, sel, keep_s=None, keep_c=None, return_attn=False, st=None):
    """Runs the classifier on the selected chunks `sel`; keep_c indexes INTO sel (TTA / deletion)."""
    if getattr(model, "per_member", False):                    # [R1] each member with its own fold's features
        outs = [head_forward(m, s, sel, keep_s, keep_c, return_attn=True) for m in model.members]
        logits = torch.stack([o[0] for o in outs]).mean(0)
        if not return_attn:
            return logits
        att = {k: (torch.stack([o[1][k] for o in outs]).mean(0) if outs[0][1][k] is not None else None)
               for k in outs[0][1]}
        return logits, att
    tk = getattr(model, "text_key", "text")
    text = s[tk] if keep_s is None else s[tk][torch.as_tensor(np.sort(keep_s), device=device)]
    c = np.asarray(sel) if keep_c is None else np.asarray(sel)[np.sort(keep_c)]
    audio = audio_feats(model, s, c)
    idx = torch.from_numpy(window_index(text.shape[0])).to(device)
    return model(text[idx], audio.unsqueeze(0), s["gm"], return_attn=return_attn, st=st)


def full_inputs(s, sel, model):
    idx = torch.from_numpy(window_index(s["text"].shape[0])).to(device)
    return s["text"][idx], audio_feats(model, s, sel).unsqueeze(0), s["gm"]


@torch.no_grad()
def predict_logodds(model, samples, tta=True):
    model.eval(); out = []
    tk = getattr(model, "text_key", "text")
    for s in samples:
        sel = pick(model, s)
        views = eval_views(s["sid"], len(s[tk]), len(sel)) if tta else [(None, None)]
        out.append(np.mean([logodds(head_forward(model, s, sel, ks, kc))[0].item() for ks, kc in views]))
    return np.array(out)


# ============================================================
# TRAIN / EVAL UTILITIES
# ============================================================
def _thr_scores(y, p, grid, metric):
    """Vectorised Youden / F1 / macro-F1 for every threshold in grid (predict 1 if p >= t)."""
    pred = p[None, :] >= grid[:, None]
    pos = y == 1
    P, N = int(pos.sum()), int((~pos).sum())
    tp = (pred & pos).sum(1); fp = (pred & ~pos).sum(1)
    fn, tn = P - tp, N - fp
    if metric == "youden":
        return tp / max(P, 1) + tn / max(N, 1) - 1
    f1p = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    if metric != "f1_macro":
        return f1p
    return (f1p + 2 * tn / np.maximum(2 * tn + fn + fp, 1)) / 2


def _best_threshold_once(y, p, metric):
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    s = _thr_scores(y, p, grid, metric)
    i = int(np.argmax(s))
    return float(grid[i]), float(s[i])


def bootstrap_thresholds(y, p, metric=THR_METRIC, n=THR_BOOT, seed=THR_SEED):
    """[W1] Optimal threshold on n class-stratified bootstrap resamples of (y, p)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    rng = np.random.RandomState(seed)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    out = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))])
        out.append(_best_threshold_once(y[idx], p[idx], metric)[0])
    return np.array(out)


def best_threshold(y, p, metric=THR_METRIC):
    """Decision threshold from dev out-of-fold predictions. [W1] THR_STABLE: median of the bootstrap-optimal
    thresholds (robust to the flat top of the Youden curve); otherwise a single argmax."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    if not THR_STABLE or len(np.unique(y)) < 2:
        return _best_threshold_once(y, p, metric)
    thr = float(np.median(bootstrap_thresholds(y, p, metric)))
    return thr, float(_thr_scores(y, p, np.array([thr]), metric)[0])


def compute_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    return dict(AUC=roc_auc_score(y, p),
                F1_macro=f1_score(y, yhat, average="macro", zero_division=0),
                F1_pos=f1_score(y, yhat, zero_division=0),
                MCC=matthews_corrcoef(y, yhat), Acc=accuracy_score(y, yhat),
                BalAcc=balanced_accuracy_score(y, yhat),
                Precision=precision_score(y, yhat, zero_division=0),
                Recall=recall_score(y, yhat, zero_division=0),
                Specificity=recall_score(y, yhat, pos_label=0, zero_division=0))


METRICS = ["AUC", "F1_macro", "F1_pos", "MCC", "Acc", "BalAcc", "Precision", "Recall", "Specificity"]


def bootstrap_ci(y, p, thr, n=N_BOOT, seed=0):
    rng, rows = np.random.RandomState(seed), []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        rows.append(compute_metrics(y[i], p[i], thr))
    return pd.DataFrame(rows).quantile([0.025, 0.975])


def random_subset(n, rng):
    k = max(1, int(round(n * rng.uniform(*AUG_FRAC))))
    return np.sort(rng.choice(n, k, replace=False))


def text_teacher(tr_idx, seed):
    """Cross-fitted logistic-regression text teacher on the training speakers only."""
    tr_idx, out = np.asarray(tr_idx), np.zeros(len(tr_idx))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr_idx, ydev[tr_idx]):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=TEACHER_C, max_iter=5000, class_weight="balanced"))
        clf.fit(XTEXT_DEV[tr_idx[a]], ydev[tr_idx[a]])
        out[b] = clf.decision_function(XTEXT_DEV[tr_idx[b]])
    return {dev_s[i]["sid"]: float(v) for i, v in zip(tr_idx, out)}


@torch.no_grad()
def feature_stats(samples, key):
    """Mean / std over all chunks of the training speakers, streamed (the per-layer tensors are large)."""
    if key == "frames":                                            # [F1] per-speaker sums stored with the cache
        n = sum(FRAME_SPK[s["sid"]][2] for s in samples)
        s1 = sum(FRAME_SPK[s["sid"]][0] for s in samples)
        s2 = sum(FRAME_SPK[s["sid"]][1] for s in samples)
        mu = s1 / n
        sd = np.sqrt(np.clip((s2 - n * mu * mu) / max(n - 1, 1), 0, None))
        return (torch.tensor(mu, dtype=torch.float, device=device),
                torch.tensor(sd, dtype=torch.float, device=device).clamp(min=1e-6))
    n, s1, s2 = 0, None, None
    for s in samples:
        x = s[key].float()
        s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
        s2 = (x * x).sum(0) if s2 is None else s2 + (x * x).sum(0)
        n += x.shape[0]
    mu = s1 / n
    sd = ((s2 - n * mu * mu) / max(n - 1, 1)).clamp(min=0).sqrt().clamp(min=1e-6)
    return mu, sd


def make_optimizer(model, lr, wd):
    """[L1] Layer logits + gain: higher LR, no weight decay. Everything else: AdamW(lr, wd)."""
    fast = {"layer_mix.logits", "layer_mix.gamma"}
    p_fast = [p for n, p in model.named_parameters() if n in fast and p.requires_grad]
    p_rest = [p for n, p in model.named_parameters() if n not in fast and p.requires_grad]
    groups = [dict(params=p_rest, lr=lr, weight_decay=wd)]
    if p_fast:
        groups.append(dict(params=p_fast, lr=lr * LAYER_LR_MULT, weight_decay=0.0))
    return torch.optim.AdamW(groups)


def train_member(r, seed, fold, tr_idx, va_idx, verbose=True):
    """r = resolved config. tr_idx = speakers for gradient steps, va_idx = speakers ONLY for early stopping."""
    set_seed(seed * 100 + fold)
    mode, selector, frac, hp = r["mode"], r["selector"], r["frac"], r["hp"]
    tr, va = [dev_s[i] for i in tr_idx], [dev_s[i] for i in va_idx]
    y_tr, y_va = ydev[tr_idx], ydev[va_idx]
    gm_mean = gm_std = None
    if N_GM > 0 and r["gm"] and mode != "text":
        G = torch.cat([s["gm"] for s in tr])
        gm_mean, gm_std = G.mean(0), G.std(0).clamp(min=1e-6)
    sc_mean = sc_std = None
    if mode != "text" and selector in LEARNED:
        X = torch.cat([s["cheap" if selector == "learned" else "audio"] for s in tr])
        sc_mean, sc_std = X.mean(0), X.std(0).clamp(min=1e-6)
    au_mean = au_std = None
    if mode != "text" and AUDIO_STD:
        if r["lora"]:                                              # [R1] statistics of fold-adapted features
            au_mean, au_std = lora_feature_stats(tr, fold, r["la"])
        else:
            key = "frames" if r["fa"] != "off" else "audio_L" if r["la"] else "audio"   # [F1] / [L1] per layer
            au_mean, au_std = feature_stats(tr, key)
    use_phq = USE_PHQ and any(s["phq"] is not None for s in tr)
    model = DepressionNet(mode, hp, gm_mean, gm_std, selector, frac, sc_mean, sc_std,
                          use_gm=r["gm"] and gm_mean is not None, enc=r["enc"], au_mean=au_mean, au_std=au_std,
                          audio_aux=r["distill"], use_phq=use_phq, moddrop=r["moddrop"],
                          text_key=r["text_key"], layer_attn=r["la"], frame_mode=r["fa"]).to(device)
    model.ts_fold = fold
    model.lora_fold = fold if r["lora"] else None
    tteach = text_teacher(tr_idx, seed * 100 + fold) if (r["distill"] and mode != "text") else None
    cw_np = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    cw = torch.tensor(cw_np, dtype=torch.float, device=device)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=float(hp["smooth"]))
    n_ep = FIXED_EPOCHS if ES_METRIC == "fixed" else EPOCHS
    opt = make_optimizer(model, float(hp["lr"]), float(hp["wd"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    rng = np.random.RandomState(seed * 100 + fold)
    tk = model.text_key
    best, bad, hist = dict(score=-np.inf, auc=np.nan, epoch=0), 0, dict(train_loss=[], val_auc=[], val_loss=[])
    for ep in range(n_ep):
        tau = TAU_START * (TAU_END / TAU_START) ** (ep / max(1, n_ep - 1))
        model.train(); opt.zero_grad()
        tot = torch.zeros((), device=device)
        order = rng.permutation(len(tr))
        for step, i in enumerate(order, 1):
            s = tr[i]
            ks = random_subset(len(s[tk]), rng) if AUGMENT else None
            cand = random_subset(len(s["audio"]), rng) if AUGMENT else np.arange(len(s["audio"]))
            sel, st, aux = train_select(model, s, cand, tau, rng)
            logits = head_forward(model, s, sel, ks, st=st)
            loss = crit(logits, s["y_t"]) + aux
            if model.phq_out is not None and s["phq"] is not None:          # [N4] PHQ-8 multi-task
                pred = torch.sigmoid(model.phq_out(model._last_hidden).view(-1))
                tgt = torch.tensor([min(s["phq"], PHQ_MAX) / PHQ_MAX], device=device)
                loss = loss + PHQ_AUX * nn.functional.mse_loss(pred, tgt)
            if tteach is not None:                                           # text -> audio distillation
                q = torch.sigmoid(torch.tensor([tteach[s["sid"]] / TEXT_DISTILL_T], device=device))
                if model.audio_aux is not None:
                    a_logits = model.audio_aux(model._last_feats["wavlm"])
                    loss = loss + AUDIO_AUX * crit(a_logits, s["y_t"])
                    student = logodds(a_logits)
                else:
                    student = logodds(logits)
                loss = loss + TEXT_DISTILL * nn.functional.binary_cross_entropy_with_logits(
                    student / TEXT_DISTILL_T, q)
            (loss / ACCUM).backward(); tot += loss.detach()
            model._last_feats = model._last_hidden = None
            if step % ACCUM == 0 or step == len(order):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        sched.step()
        lo = predict_logodds(model, va, tta=False)
        v_auc = roc_auc_score(y_va, lo)
        nll = np.where(y_va == 1, np.logaddexp(0, -lo), np.logaddexp(0, lo))
        wv = cw_np[y_va]; v_loss = float((wv * nll).sum() / wv.sum())
        hist["train_loss"].append(tot.item() / len(tr)); hist["val_auc"].append(v_auc); hist["val_loss"].append(v_loss)
        if ep + 1 < ES_START and ES_METRIC != "fixed":
            continue
        score = {"auc": v_auc, "loss": -v_loss, "fixed": float(ep)}[ES_METRIC]
        if score > best["score"]:
            best = dict(score=score, auc=v_auc, loss=v_loss, epoch=ep + 1,
                        state={k: v.detach().clone() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
        if ES_METRIC != "fixed" and ep + 1 >= MIN_EPOCHS and bad >= PATIENCE:
            break
    model.load_state_dict(best["state"]); model.eval()
    if verbose:
        lw = ""
        if model.layer_mix is not None:
            w_ = torch.softmax(model.layer_mix.logits.detach(), 0).cpu().numpy()
            lw = f" top layer {model.layer_ids[int(w_.argmax())]} (w={w_.max():.2f})"
        print(f"  [{mode:6s} {model.selector:13s} keep {model.keep_frac:.0%} gm {int(model.use_gm)} "
              f"enc {model.enc} la {int(model.layered)} fa {model.frame_mode} lora {int(r['lora'])} "
              f"drop {int(model.moddrop)} "
              f"phq {int(use_phq)} "
              f"seed {seed} fold {fold}] best epoch {best['epoch']:02d}  early-stop AUC {best['auc']:.3f}  "
              f"final train loss {hist['train_loss'][-1]:.3f}{lw}")
    return model, dict(val_auc=best["auc"], epoch=best["epoch"], hist=hist, tr_idx=tr_idx, fold=fold)



# ============================================================
# [R1] STAGE A: LoRA ADAPTATION OF THE FROZEN HuBERT, PER CV FOLD
# ============================================================
class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B A dropout(x).  W, b frozen; A (r x in), B (out x r) trained; B = 0 at start,
    so training starts exactly from the frozen encoder."""
    def __init__(self, base, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT):
        super().__init__()
        self.base, self.r, self.scale = base, int(r), float(alpha) / float(r)
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.lora_drop = nn.Dropout(dropout)

    def forward(self, x):
        lora = nn.functional.linear(nn.functional.linear(self.lora_drop(x), self.lora_A), self.lora_B)
        return self.base(x) + self.scale * lora


def _get_mod(mod, path):
    for p in path.split("."):
        mod = getattr(mod, p)
    return mod


def _set_mod(mod, path, new):
    *parents, last = path.split(".")
    for p in parents:
        mod = getattr(mod, p)
    setattr(mod, last, new)


def lora_layer_ids(model):
    n = len(model.encoder.layers)
    return list(range(n)) if LORA_LAYERS == "all" else list(range(n - int(LORA_LAYERS), n))


def inject_lora(model):
    """Freezes everything, wraps the target Linear layers of the chosen transformer layers with LoRALinear."""
    for p in model.parameters():
        p.requires_grad = False
    for i in lora_layer_ids(model):
        layer = model.encoder.layers[i]
        for t in LORA_TARGETS:
            base = _get_mod(layer, t)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"LoRA target encoder.layers.{i}.{t} is {type(base).__name__}, not nn.Linear")
            _set_mod(layer, t, LoRALinear(base))
    return sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)


def lora_deltas(model):
    """{ 'i|path': dict(A, B, scale) } on CPU — small (~0.6M numbers), enough to rebuild the merged encoder."""
    out = {}
    for i in lora_layer_ids(model):
        for t in LORA_TARGETS:
            m = _get_mod(model.encoder.layers[i], t)
            out[f"{i}|{t}"] = dict(A=m.lora_A.detach().float().cpu().clone(), B=m.lora_B.detach().float().cpu().clone(),
                                   scale=m.scale)
    return out


@torch.no_grad()
def apply_lora_deltas(model, deltas):
    """Merges LoRA into a plain (un-injected) HuBERT copy: W <- W + scale * B A. No extra inference cost."""
    for key, d in deltas.items():
        i, t = key.split("|")
        lin = _get_mod(model.encoder.layers[int(i)], t)
        lin.weight += (d["scale"] * (d["B"] @ d["A"])).to(lin.weight.device, lin.weight.dtype)
    for p in model.parameters():
        p.requires_grad = False
    return model


class ChunkHead(nn.Module):
    """[C1] Chunk-level classifier: layer means of single chunks (N, L, 768) -> per-dimension standardisation [P1]
    -> layer mix -> MLP -> chunk logits (N, 2). A speaker's score aggregates its chunk log-odds (MIL_AGG)."""
    def __init__(self, n_layers, mu, sd, hidden=LORA_HEAD_HIDDEN, drop=LORA_HEAD_DROP, use_phq=False):
        super().__init__()
        self.register_buffer("mu", mu.reshape(1, n_layers, 768).clone().float())
        self.register_buffer("sd", sd.reshape(1, n_layers, 768).clone().float())
        self.mix = LayerAttention(n_layers, 768, "scalar", LAYER_DROP, None)
        self.norm, self.drop = nn.LayerNorm(768), nn.Dropout(drop)
        self.body = nn.Sequential(nn.Linear(768, 2 * hidden), nn.GELU(), nn.Dropout(drop))
        self.out = nn.Linear(2 * hidden, 2)
        self.phq = nn.Linear(2 * hidden, 1) if use_phq else None

    def forward(self, x):
        z, _ = self.mix(((x - self.mu) / self.sd).clamp(-5, 5))
        h = self.body(self.drop(self.norm(z)))
        return self.out(h), h


def aggregate(lo):
    """[C1] Speaker log-odds from chunk log-odds: mean, or 10%-trimmed mean."""
    lo = np.sort(np.asarray(lo, float))
    if MIL_AGG == "trim" and len(lo) >= 5:
        k = int(0.1 * len(lo))
        lo = lo[k:len(lo) - k] if k > 0 else lo
    return float(lo.mean())


# ---- waveform cache: the exact participant chunks of the feature cache, int16, read lazily from disk ----
WAVE_MM, WAVE_OFF = None, {}


def build_wave_cache():
    order = [(sp, sid) for sp in ("Train", "Val", "Test") for sid in sorted(caches[sp])]
    T = [len(caches[sp][sid]["times"]) for sp, sid in order]
    offs = np.concatenate([[0], np.cumsum(T)[:-1]]).astype(np.int64)
    n_tot = int(sum(T))
    tag = f"waves_c{CHUNK_SEC}_mx{MAX_AUDIO_CHUNKS}_ps{int(USE_PARTICIPANT_SEGMENTS)}"
    want = dict(tag=tag, n_tot=n_tot, sids=[sid for _, sid in order], T=T)
    for d in ([WAVE_CACHE_DIR] if WAVE_CACHE_DIR else []) + [os.path.join(CACHE_DIR, tag)]:
        mp = os.path.join(d, "waves_meta.pt")
        if os.path.exists(mp):
            meta = torch.load(mp, weights_only=False)
            if all(meta.get(k) == v for k, v in want.items()):
                print(f"[R1] loading waveform cache from {d}")
                return d, meta
            print(f"[R1] waveform cache in {d} does not match this run -> ignored")
    out = os.path.join(CACHE_DIR, tag); os.makedirs(out, exist_ok=True)
    gb = n_tot * CHUNK_SIZE * 2 / 1e9
    free = shutil.disk_usage(out).free / 1e9
    print(f"[R1] building waveform cache: {n_tot} chunks x {CHUNK_SEC} s = {gb:.1f} GB (free disk {free:.1f} GB)")
    if gb > WAVE_MAX_GB or gb > 0.95 * free:
        raise RuntimeError(f"The waveform cache needs {gb:.1f} GB: free disk space or raise WAVE_MAX_GB.")
    mm = np.lib.format.open_memmap(os.path.join(out, "waves.npy"), mode="w+", dtype=np.int16,
                                   shape=(n_tot, CHUNK_SIZE))
    for (sp, sid), off in zip(tqdm(order, desc="waveform cache"), offs):
        e = caches[sp][sid]
        if not (os.path.exists(e["wav"]) and os.path.exists(e["tcsv"])):
            raise FileNotFoundError(f"Raw files of speaker {sid} missing ({e['wav']}). Attach the audio + transcript "
                                    "datasets at the cached paths, or set WAVE_CACHE_DIR / LORA_CACHE_DIR.")
        _, segs = load_transcript(e["tcsv"])
        chunks, times, _ = load_participant_audio(e["wav"], segs)
        if len(chunks) != len(e["times"]) or not np.allclose(times, e["times"], atol=0.05):
            raise RuntimeError(f"Re-chunking speaker {sid} does not reproduce the cached chunks "
                               "(were CHUNK_SEC / MAX_AUDIO_CHUNKS / USE_PARTICIPANT_SEGMENTS changed?)")
        mm[off: off + len(chunks)] = (np.clip(chunks, -1, 1) * 32767).astype(np.int16)
    mm.flush(); del mm
    meta = dict(want, offsets=dict(zip(want["sids"], offs.tolist())))
    torch.save(meta, os.path.join(out, "waves_meta.pt"))
    return out, meta


def wave_tensor(sid, idx, rng=None):
    x = WAVE_MM[WAVE_OFF[sid] + np.asarray(idx, dtype=np.int64)].astype(np.float32) / 32767.0
    if rng is not None and LORA_GAIN_DB > 0:
        x = x * (10.0 ** (rng.uniform(-LORA_GAIN_DB, LORA_GAIN_DB, size=(len(x), 1)) / 20.0)).astype(np.float32)
    return torch.from_numpy(x).to(device)


def _autocast():
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP)


def chunk_layer_means(enc, x):
    """(K, samples) -> (1, K, L, 768): time-mean of every hidden state, as in the feature cache."""
    with _autocast():
        h = enc.hidden_states(x).mean(2).permute(1, 0, 2)
    return h.float().unsqueeze(0)


@torch.no_grad()
def lora_eval(enc, head, idx):
    """Speaker log-odds of early-stopping speakers from raw waveforms: aggregate of all chunk log-odds."""
    enc.eval(); head.eval(); out = []
    for i in idx:
        s = dev_s[i]; T = len(s["audio"])
        ci = (np.arange(T) if LORA_EVAL_CHUNKS <= 0 else
              np.unique(np.linspace(0, T - 1, min(T, LORA_EVAL_CHUNKS)).round().astype(int)))
        lo = [logodds(head(chunk_layer_means(enc, wave_tensor(s["sid"], ci[j:j + AUDIO_BATCH]))[0])[0]).cpu().numpy()
              for j in range(0, len(ci), AUDIO_BATCH)]
        out.append(aggregate(np.concatenate(lo)))
    return np.array(out)


@torch.no_grad()
def lora_extract(enc):
    """Per-chunk layer means of EVERY speaker (dev + test) with the merged, adapted encoder -> fp16 on CPU."""
    enc.eval(); feats = {}
    for e in tqdm(dev_entries + test_entries, desc="LoRA features", leave=False):
        T, parts = len(e["times"]), []
        for i in range(0, T, AUDIO_BATCH):
            parts.append(chunk_layer_means(enc, wave_tensor(e["sid"], np.arange(i, min(i + AUDIO_BATCH, T))))[0]
                         .half().cpu())
        feats[e["sid"]] = torch.cat(parts)
    return feats


def lora_head_lo(head, feats, sid):
    return head_lo(head, feats[sid])


def wave_items(items, rng=None):
    """[C2] Waveforms of (dev index, chunk index) items from many speakers, with random gain."""
    rows = np.array([WAVE_OFF[dev_s[i]["sid"]] + t for i, t in items], dtype=np.int64)
    x = WAVE_MM[rows].astype(np.float32) / 32767.0
    if rng is not None and LORA_GAIN_DB > 0:
        x = x * (10.0 ** (rng.uniform(-LORA_GAIN_DB, LORA_GAIN_DB, size=(len(x), 1)) / 20.0)).astype(np.float32)
    return torch.from_numpy(x).to(device)


@torch.no_grad()
def frozen_stats(idx):
    """Per layer and dimension mean / std of the cached FROZEN layer means over all chunks of dev speakers idx."""
    n, s1, s2 = 0, None, None
    for i in idx:
        x = dev_entries[i]["audio_layers"].float()
        s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
        s2 = (x * x).sum(0) if s2 is None else s2 + (x * x).sum(0)
        n += len(x)
    mu = s1 / n
    return mu, ((s2 - n * mu * mu) / max(n - 1, 1)).clamp(min=0).sqrt().clamp(min=1e-6)


@torch.no_grad()
def head_lo(head, x):
    """Speaker log-odds from all chunk features x (T, L, 768): aggregate of the chunk log-odds."""
    head.eval()
    lo = [logodds(head(x[j:j + 256].to(device).float())[0]).cpu().numpy() for j in range(0, len(x), 256)]
    return aggregate(np.concatenate(lo))


def es_score(lo, y, cw_np):
    auc = roc_auc_score(y, lo)
    nll = np.where(y == 1, np.logaddexp(0, -lo), np.logaddexp(0, lo))
    wv = cw_np[y]; v_loss = float((wv * nll).sum() / wv.sum())
    return {"auc": auc, "loss": -v_loss}.get(ES_METRIC, -v_loss), auc, v_loss


class ChunkSampler:
    """[C2] Class-balanced (50/50), then speaker-uniform, then chunk-uniform sampling of (dev index, chunk index)."""
    def __init__(self, idx, rng):
        self.by_cls = {c: [int(i) for i in idx if ydev[i] == c] for c in (0, 1)}
        self.rng = rng

    def draw(self, n):
        out = []
        for _ in range(n):
            grp = self.by_cls[int(self.rng.rand() < 0.5)]
            i = grp[self.rng.randint(len(grp))]
            out.append((i, int(self.rng.randint(len(dev_s[i]["audio"])))))
        return out


def chunk_loss(head, logits, h, items, teacher):
    """[C1] Chunk-level loss: smoothed CE with the speaker label (+ text-teacher distillation, + PHQ-8 regression)."""
    y = torch.tensor([int(ydev[i]) for i, _ in items], device=device)
    loss = nn.functional.cross_entropy(logits, y, label_smoothing=MIL_SMOOTH)
    if teacher is not None:
        q = torch.sigmoid(torch.tensor([teacher[dev_s[i]["sid"]] for i, _ in items], dtype=torch.float,
                                       device=device) / TEXT_DISTILL_T)
        loss = loss + LORA_TEXT_DISTILL * nn.functional.binary_cross_entropy_with_logits(
            logodds(logits) / TEXT_DISTILL_T, q)
    if head.phq is not None:
        m = [k for k, (i, _) in enumerate(items) if dev_s[i]["phq"] is not None]
        if m:
            tgt = torch.tensor([min(dev_s[items[k][0]]["phq"], PHQ_MAX) / PHQ_MAX for k in m], device=device)
            pred = torch.sigmoid(head.phq(h[torch.as_tensor(m, device=device)]).view(-1))
            loss = loss + PHQ_AUX * nn.functional.mse_loss(pred, tgt)
    return loss


def lp_train(f, fit_idx, es_idx, mu, sd, teacher, use_phq, cw_np):
    """[P2] Phase 1: the chunk head alone on the cached frozen features (balanced chunk batches)."""
    set_seed(LORA_SEED + 100 + f)
    head = ChunkHead(N_LAY_ALL, mu, sd, use_phq=use_phq).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LORA_HEAD_LR, weight_decay=1e-2)
    sampler = ChunkSampler(fit_idx, np.random.RandomState(LORA_SEED + 100 + f))
    best, bad = dict(score=-np.inf, auc=np.nan, loss=np.nan, epoch=0), 0
    for ep in range(LP_EPOCHS):
        head.train()
        for _ in range(max(1, LP_CHUNKS_PER_EPOCH // LP_BATCH)):
            items = sampler.draw(LP_BATCH)
            x = torch.stack([dev_entries[i]["audio_layers"][t] for i, t in items]).to(device).float()
            logits, h = head(x)
            loss = chunk_loss(head, logits, h, items, teacher)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        lo = np.array([head_lo(head, dev_entries[i]["audio_layers"]) for i in es_idx])
        score, auc, v_loss = es_score(lo, ydev[es_idx], cw_np)
        if score > best["score"]:
            best = dict(score=score, auc=auc, loss=v_loss, epoch=ep + 1,
                        head={k: v.detach().clone() for k, v in head.state_dict().items()})
            bad = 0
        else:
            bad += 1
        if ep + 1 >= 10 and bad >= LP_PATIENCE:
            break
    head.load_state_dict(best["head"]); head.eval()
    print(f"  [probe fold {f}] best epoch {best['epoch']:02d} early-stop AUC {best['auc']:.3f} loss {best['loss']:.3f}")
    return head, best


def lora_train_fold(f, tr_idx, va_idx):
    set_seed(LORA_SEED + f)
    fit_idx, es_idx = train_test_split(tr_idx, test_size=INNER_ES_FRAC, stratify=ydev[tr_idx], random_state=f)
    use_phq = USE_PHQ and any(dev_s[i]["phq"] is not None for i in fit_idx)
    teacher = text_teacher(fit_idx, LORA_SEED + f) if LORA_TEXT_DISTILL > 0 else None
    cw_np = compute_class_weight("balanced", classes=np.array([0, 1]), y=ydev[fit_idx])
    mu, sd = frozen_stats(fit_idx)                                  # [P1]
    n_chunks = sum(len(dev_s[i]["audio"]) for i in fit_idx)
    print(f"\n[R1] fold {f}: fit {len(fit_idx)} speakers / {n_chunks} chunks, early-stop {len(es_idx)}, "
          f"validation {len(va_idx)} speakers")

    # ---- phase 1: chunk head on frozen features
    head, lp = lp_train(f, fit_idx, es_idx, mu, sd, teacher, use_phq, cw_np)
    lp_oof = np.array([head_lo(head, dev_entries[i]["audio_layers"]) for i in va_idx])
    lp_test = np.array([head_lo(head, e["audio_layers"]) for e in test_entries])

    # ---- phase 2: LoRA + chunk head on balanced chunk batches, starting from the probe head
    set_seed(LORA_SEED + f)
    hub = copy.deepcopy(audio_encoder.wavlm).to(device)
    n_lora = inject_lora(hub)
    enc = AudioEncoder(hub, WAVLM_NORMALIZE, 0, N_LAY_ALL).to(device)
    lora_params = [p for n, p in hub.named_parameters() if "lora_" in n]
    opt = torch.optim.AdamW([dict(params=lora_params, lr=LORA_LR, weight_decay=LORA_WD),
                             dict(params=head.parameters(), lr=LORA_HEAD_LR_FT, weight_decay=1e-2)])
    n_micro = max(1, LORA_CHUNKS_PER_EPOCH // LORA_MICRO)
    steps_ep = int(np.ceil(n_micro / LORA_ACCUM)); total = steps_ep * LORA_EPOCHS; warm = steps_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, t / max(1, total)))))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    rng = np.random.RandomState(LORA_SEED + f)
    sampler = ChunkSampler(fit_idx, rng)
    snap = lambda: dict(lora={n: p.detach().cpu().clone() for n, p in hub.named_parameters() if "lora_" in n},
                        head={k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
    score0, auc0, loss0 = es_score(lora_eval(enc, head, es_idx), ydev[es_idx], cw_np)
    print(f"  [lora fold {f} epoch 00] adapters = 0 ({n_lora / 1e6:.2f}M LoRA params) | early-stop AUC {auc0:.3f} "
          f"loss {loss0:.3f}")
    snap0 = dict(score=score0, auc=auc0, loss=loss0, epoch=0, **snap())
    best, bad = dict(snap0), 0
    for ep in range(LORA_EPOCHS):
        t0 = time.time(); hub.train(); head.train(); tot = 0.0; opt.zero_grad()
        for mb in range(1, n_micro + 1):
            items = sampler.draw(LORA_MICRO)
            logits, h = head(chunk_layer_means(enc, wave_items(items, rng))[0])
            loss = chunk_loss(head, logits, h, items, teacher)
            scaler.scale(loss / LORA_ACCUM).backward(); tot += loss.item()
            if mb % LORA_ACCUM == 0 or mb == n_micro:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        score, auc, v_loss = es_score(lora_eval(enc, head, es_idx), ydev[es_idx], cw_np)
        print(f"  [lora fold {f} epoch {ep + 1:02d}] chunk loss {tot / n_micro:.3f} | early-stop AUC {auc:.3f} "
              f"loss {v_loss:.3f} | {time.time() - t0:.0f}s")
        if score > best["score"]:
            best = dict(score=score, auc=auc, loss=v_loss, epoch=ep + 1, **snap()); bad = 0
        else:
            bad += 1
        if ep + 1 >= LORA_MIN_EPOCHS and bad >= LORA_PATIENCE:
            break
    if best["epoch"] > 0 and best["auc"] < auc0:                   # [C3] must not lose AUC against the probe
        print(f"  [lora fold {f}] best adapters (epoch {best['epoch']}) have a lower early-stop AUC than the probe "
              f"({best['auc']:.3f} < {auc0:.3f})")
        best = snap0
    with torch.no_grad():
        for n, p in hub.named_parameters():
            if n in best["lora"]:
                p.copy_(best["lora"][n].to(device))
    head.load_state_dict(best["head"]); head.eval(); hub.eval()
    used = best["epoch"] > 0
    deltas = lora_deltas(hub)
    w_ = torch.softmax(head.mix.logits.detach(), 0).cpu().numpy()
    if used:
        merged = apply_lora_deltas(copy.deepcopy(audio_encoder.wavlm).to(device), deltas)
        enc_m = AudioEncoder(merged, WAVLM_NORMALIZE, 0, N_LAY_ALL).to(device).eval()
        with torch.no_grad():                                      # merged weights == adapters (sanity check)
            x0 = wave_tensor(dev_s[int(es_idx[0])]["sid"], [0])
            err = float((enc.hidden_states(x0) - enc_m.hidden_states(x0)).abs().max())
        print(f"  [lora fold {f}] KEPT adapters of epoch {best['epoch']} (early-stop AUC {best['auc']:.3f} vs probe "
              f"{auc0:.3f}) | merge check max diff {err:.1e} | top layer {int(w_.argmax())} (w={w_.max():.2f})")
        del enc, hub
        feats = lora_extract(enc_m)
        del enc_m, merged
    else:
        print(f"  [lora fold {f}] adapters discarded -> this fold keeps the frozen features")
        del enc, hub
        feats = {e["sid"]: e["audio_layers"] for e in dev_entries + test_entries}
    info = dict(es_auc=float(best["auc"]), epoch=best["epoch"], lora_used=used, deltas=deltas, head=best["head"],
                use_phq=use_phq, layer_w=w_.tolist(), lp_es_auc=float(lp["auc"]), lp_epoch=lp["epoch"],
                probe_es_auc_wave=float(auc0), lp_oof=lp_oof, lp_test=lp_test,
                oof=np.array([lora_head_lo(head, feats, dev_s[i]["sid"]) for i in va_idx]),
                test=np.array([lora_head_lo(head, feats, s["sid"]) for s in test_s]))
    del head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feats, info


def lora_feature_stats(samples, fold, layered):
    """Mean / std of fold `fold`'s adapted features over the training speakers' chunks."""
    n, s1, s2 = 0, None, None
    for s in samples:
        x = LORA_FEATS[fold][s["sid"]].float()
        x = x[:, LA0:LA1] if layered else x[:, WAVLM_LAYERS[0]:WAVLM_LAYERS[1]].mean(1)
        s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
        s2 = (x * x).sum(0) if s2 is None else s2 + (x * x).sum(0)
        n += x.shape[0]
    mu = s1 / n
    sd = ((s2 - n * mu * mu) / max(n - 1, 1)).clamp(min=0).sqrt().clamp(min=1e-6)
    return mu.to(device), sd.to(device)


LORA_CFG = dict(model=WAVLM_NAME, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT, targets=LORA_TARGETS,
                layers=LORA_LAYERS, lr=LORA_LR, head_lr=LORA_HEAD_LR, wd=LORA_WD, epochs=LORA_EPOCHS,
                min_epochs=LORA_MIN_EPOCHS, patience=LORA_PATIENCE, chunks_per_spk=LORA_CHUNKS_PER_SPK,
                accum=LORA_ACCUM, eval_chunks=LORA_EVAL_CHUNKS, gain_db=LORA_GAIN_DB, distill=LORA_TEXT_DISTILL,
                phq=bool(USE_PHQ), seed=LORA_SEED, cv_seed=CV_SEED, n_folds=N_FOLDS, es_frac=INNER_ES_FRAC,
                chunk_sec=CHUNK_SEC, es_metric=ES_METRIC, scheme="mil-chunk-lpft", lp_epochs=LP_EPOCHS,
                mil_agg=MIL_AGG, mil_smooth=MIL_SMOOTH, lp_chunks=LP_CHUNKS_PER_EPOCH, lp_batch=LP_BATCH,
                micro=LORA_MICRO, chunks_per_epoch=LORA_CHUNKS_PER_EPOCH, guard="auc+score",
                lp_patience=LP_PATIENCE, head_lr_ft=LORA_HEAD_LR_FT)
LORA_FEATS, LORA_INFO, LORA_E2E = {}, {}, {}
if NEED_LORA:
    _wdir, _wmeta = build_wave_cache()
    WAVE_MM = np.load(os.path.join(_wdir, "waves.npy"), mmap_mode="r")
    WAVE_OFF = _wmeta["offsets"]
    _key = dict(cfg=LORA_CFG, sids=[e["sid"] for e in dev_entries + test_entries],
                T=[len(e["times"]) for e in dev_entries + test_entries])
    _ldir = os.path.join(CACHE_DIR, f"lora_{zlib.crc32(json.dumps(LORA_CFG, sort_keys=True).encode()):08x}")
    os.makedirs(_ldir, exist_ok=True)
    audio_encoder.to(device)
    oof_lo, lp_oof_lo = np.zeros(len(dev_s)), np.zeros(len(dev_s)); test_los, lp_test_los = [], []
    for f, (tr_idx, va_idx) in enumerate(folds):
        path, got = None, None
        for d in ([LORA_CACHE_DIR] if LORA_CACHE_DIR else []) + [_ldir]:
            p_ = os.path.join(d, f"lora_fold{f}.pt")
            if os.path.exists(p_):
                got = torch.load(p_, weights_only=False)
                if got.get("key") == _key:
                    path = p_; break
                got = None
        if got is not None:
            print(f"[R1] fold {f}: loaded adapted features from {path}")
            LORA_FEATS[f], LORA_INFO[f] = got["feats"], got["info"]
        else:
            LORA_FEATS[f], LORA_INFO[f] = lora_train_fold(f, tr_idx, va_idx)
            torch.save(dict(key=_key, feats=LORA_FEATS[f], info=LORA_INFO[f]), os.path.join(_ldir, f"lora_fold{f}.pt"))
        oof_lo[va_idx] = LORA_INFO[f]["oof"]; test_los.append(LORA_INFO[f]["test"])
        lp_oof_lo[va_idx] = LORA_INFO[f]["lp_oof"]; lp_test_los.append(LORA_INFO[f]["lp_test"])
    audio_encoder.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def _e2e(name, oof, tests, lora):
        oof_p, test_p = sigmoid(oof), sigmoid(np.mean(tests, 0))
        thr, _ = best_threshold(ydev, oof_p)
        return dict(name=name, mode="e2e", desc=(("LoRA-HuBERT" if any(LORA_INFO[f_]["lora_used"] for f_ in LORA_INFO)
                          else "LoRA-rejected (= frozen) HuBERT") if lora else "frozen-HuBERT") + " chunk-level MIL head",
                    selector="all", frac=1.0, gm=False, enc="lora-e2e" if lora else "probe-e2e",
                    la=True, fa="off", lora=lora, distill=LORA_TEXT_DISTILL > 0, moddrop=False, text_key="text",
                    hp={}, infos=[], thr=thr, oof_p=oof_p, test_p=test_p, cv=compute_metrics(ydev, oof_p, thr),
                    test=compute_metrics(yte, test_p, thr), sel={s["sid"]: np.arange(len(s["audio"])) for s in test_s})

    LORA_E2E = {"audio_lora_e2e": _e2e("audio_lora_e2e", oof_lo, test_los, True),
                "audio_probe_e2e": _e2e("audio_probe_e2e", lp_oof_lo, lp_test_los, False)}
    print(f"\n[R1] Stage A done ({_ldir}).")
    print("  early-stop AUC  probe:", [round(LORA_INFO[f]["lp_es_auc"], 3) for f in range(N_FOLDS)],
          "| LoRA:", [round(LORA_INFO[f]["es_auc"], 3) for f in range(N_FOLDS)],
          "| adapters kept:", [bool(LORA_INFO[f]["lora_used"]) for f in range(N_FOLDS)])
    for k, v in LORA_E2E.items():
        print(f"[R3] {k}: CV AUC {v['cv']['AUC']:.3f} | TEST AUC {v['test']['AUC']:.3f}")
    print(f"     (frozen-HuBERT LR probe CV AUC {cv_probe_auc(Xlay[:, WAVLM_LAYERS[0]:WAVLM_LAYERS[1]].mean(1), ydev):.3f})")
    pd.DataFrame([dict(fold=f, es_auc=LORA_INFO[f]["es_auc"], best_epoch=LORA_INFO[f]["epoch"],
                       probe_es_auc=LORA_INFO[f]["lp_es_auc"], adapters_kept=LORA_INFO[f]["lora_used"],
                       **{f"w_layer{l}": w for l, w in enumerate(LORA_INFO[f]["layer_w"])}) for f in range(N_FOLDS)]
                 ).to_csv(os.path.join(OUT_DIR, "lora_stage_a.csv"), index=False)


SEL_GM = True if AUDIO_GM is None else bool(AUDIO_GM)
SEL_ENC = "lstm" if AUDIO_ENC is None else AUDIO_ENC


def resolve(c):
    r = dict(c)
    mode = c["mode"]
    r["gm"] = (SEL_GM if c["gm"] is None else bool(c["gm"])) and N_GM > 0 and mode != "text"
    r["enc"] = (SEL_ENC if c["enc"] is None else c["enc"]) if mode != "text" else None
    r["distill"] = bool(c.get("distill", False)) and mode != "text" and TEXT_DISTILL > 0
    r["moddrop"] = (mode == "fusion") if c.get("moddrop") is None else (bool(c["moddrop"]) and mode == "fusion")
    r["text_key"] = c.get("text_key", "text")
    r["hp"] = dict(c["hp"]) if c.get("hp") else dict(HP[mode])
    r["la"] = (LAYER_ATTN if c.get("la") is None else bool(c["la"])) and mode != "text"      # [L1]
    r["fa"] = _fa(c.get("fa"), FRAME_ATTN) if mode != "text" else "off"                       # [F1]
    r["lora"] = _eff_lo(c) and mode != "text"                                                    # [R1]
    if r["lora"] and r["fa"] != "off":
        if c.get("fa") is not None:
            raise RuntimeError("LoRA features are chunk-level: frame attention cannot be combined with LoRA.")
        r["fa"] = "off"                                    # the default FRAME_ATTN applies to frozen models only
    if r["fa"] != "off" and not NEED_FRAMES:
        raise RuntimeError("A config uses frame attention but the frame cache was not loaded (NEED_FRAMES).")
    if r["lora"] and not NEED_LORA:
        raise RuntimeError("A config uses LoRA but Stage A was not run (NEED_LORA).")
    if r["la"] and r["fa"] == "off" and not r["lora"] and not NEED_LAYERS:
        raise RuntimeError("A config uses layer attention but per-layer features were not loaded (NEED_LAYERS).")
    return r


def run_config(c, seeds, keep_members=True, verbose=True):
    """Trains len(seeds) x N_FOLDS members. Test predictions use the ensemble; thresholds come from OOF dev."""
    r = resolve(c); name, mode = r["name"], r["mode"]
    print(f"\n=== {name}: mode={mode} selector={r['selector']} keep={r['frac']:.0%} gm={r['gm']} enc={r['enc']} "
          f"layer_attn={r['la']} frames={r['fa']} lora={r['lora']} distill={r['distill']} moddrop={r['moddrop']} text={r['text_key']} "
          f"hp={r['hp']} ===")
    oof_lo = np.zeros((len(seeds), len(dev_s)))
    members, infos, rows = [], [], []
    for si, seed in enumerate(seeds):
        seed_members = []
        for fi, (tr_idx, va_idx) in enumerate(folds):
            if INNER_ES:
                # [R2] LoRA models reuse Stage A's inner split, so early stopping is on speakers the encoder never saw
                fit_idx, es_idx = train_test_split(tr_idx, test_size=INNER_ES_FRAC, stratify=ydev[tr_idx],
                                                   random_state=fi if r["lora"] else seed * 100 + fi)
            else:
                fit_idx, es_idx = tr_idx, va_idx
            model, info = train_member(r, seed, fi, fit_idx, es_idx, verbose)
            oof_lo[si, va_idx] = predict_logodds(model, [dev_s[i] for i in va_idx])
            seed_members.append(model); infos.append(info)
        thr_s, _ = best_threshold(ydev, sigmoid(oof_lo[si]))
        p_s = sigmoid(predict_logodds(Ensemble(seed_members).eval(), test_s))
        rows.append(dict(name=name, mode=mode, selector=r["selector"], keep_frac=r["frac"], gm=r["gm"],
                         enc=r["enc"], la=r["la"], fa=r["fa"], lora=r["lora"], distill=r["distill"],
                         moddrop=r["moddrop"], text_key=r["text_key"], hp=json.dumps(r["hp"]), seed=seed, thr=thr_s,
                         cv_auc=roc_auc_score(ydev, oof_lo[si]), **compute_metrics(yte, p_s, thr_s)))
        members += seed_members
    E = Ensemble(members).eval()
    oof_p, test_p = sigmoid(oof_lo.mean(0)), sigmoid(predict_logodds(E, test_s))
    thr, _ = best_threshold(ydev, oof_p)
    out = dict(name=name, mode=mode, selector=E.selector, frac=E.keep_frac, gm=bool(members[0].use_gm),
               enc=r["enc"], la=r["la"], fa=r["fa"], lora=r["lora"], distill=r["distill"], moddrop=r["moddrop"],
               text_key=r["text_key"],
               hp=r["hp"], infos=infos, thr=thr, oof_p=oof_p, test_p=test_p, cv=compute_metrics(ydev, oof_p, thr),
               test=compute_metrics(yte, test_p, thr), sel={s["sid"]: pick(E, s) for s in test_s})
    print(f"{name} ensemble: CV AUC {out['cv']['AUC']:.3f} | TEST AUC {out['test']['AUC']:.3f} "
          f"F1-macro {out['test']['F1_macro']:.3f} Acc {out['test']['Acc']:.3f} (thr {thr:.2f})")
    if keep_members:
        out["members"] = members
    else:
        del members, E, seed_members
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out, rows


# ============================================================
# STAGE 1: audio-branch ablation (chosen by honest CV AUC; test AUC printed for information only)
# ============================================================
if RUN_ABLATION and (AUDIO_GM is None or AUDIO_ENC is None):
    gm_opts = [AUDIO_GM] if AUDIO_GM is not None else ([True, False] if N_GM > 0 else [False])
    enc_opts = [AUDIO_ENC] if AUDIO_ENC is not None else ["lstm", "linear", "stats"]
    abl = []
    for g in gm_opts:
        for en in enc_opts:
            o, _ = run_config(cfg(f"abl_gm{int(g)}_{en}", "audio", "all", 1.0, gm=g, enc=en), ABL_SEEDS,
                              keep_members=False, verbose=False)
            abl.append(dict(gm=g, enc=en, cv_auc=o["cv"]["AUC"], cv_F1_macro=o["cv"]["F1_macro"],
                            mean_best_epoch=np.mean([i["epoch"] for i in o["infos"]]),
                            test_auc_info_only=o["test"]["AUC"]))
    abl = pd.DataFrame(abl).sort_values("cv_auc", ascending=False)
    abl.to_csv(os.path.join(OUT_DIR, "stage1_audio_ablation.csv"), index=False)
    print("\nStage 1 audio ablation (chosen by CV AUC):\n", abl.round(3).to_string(index=False))
    SEL_GM, SEL_ENC = bool(abl.iloc[0].gm), str(abl.iloc[0].enc)
    if abl.iloc[0].cv_auc < 0.55:
        print("WARNING: every audio variant is near chance (best CV AUC < 0.55).")
print(f"\n>>> Audio branch used from now on: speaker branch {SPK_LABEL} = {SEL_GM}, encoder = {SEL_ENC}, "
      f"layer attention = {LAYER_ATTN} ({LAYER_ATTN_KIND}), frame attention = {FRAME_ATTN}")


# ============================================================
# [N6] HYPERPARAMETER SEARCH (dev CV only; the test set is never used to choose)
# ============================================================
def sample_hps(n, seed):
    rng, out = np.random.RandomState(seed), [dict(HP_DEFAULT)]
    seen = {json.dumps(HP_DEFAULT, sort_keys=True)}
    for _ in range(2000):
        if len(out) >= n:
            break
        hp = {k: v[rng.randint(len(v))] for k, v in HPO_SPACE.items()}
        key = json.dumps(hp, sort_keys=True)
        if key not in seen:
            seen.add(key); out.append(hp)
    return out


def run_hpo(mode, base, seed):
    trials = []
    for t, hp in enumerate(sample_hps(HPO_TRIALS, seed)):
        o, _ = run_config(dict(base, name=f"hpo_{mode}_{t}", hp=hp), HPO_SEEDS, keep_members=False, verbose=False)
        trials.append(dict(trial=t, **hp, cv_auc=o["cv"]["AUC"], cv_F1_macro=o["cv"]["F1_macro"],
                           mean_best_epoch=np.mean([i["epoch"] for i in o["infos"]])))
    df = pd.DataFrame(trials).sort_values("cv_auc", ascending=False)
    df.to_csv(os.path.join(OUT_DIR, f"hpo_{mode}.csv"), index=False)
    print(f"\nHyperparameter search ({mode}, dev CV AUC):\n", df.round(4).to_string(index=False))
    best = df.iloc[0]
    return {k: type(HP_DEFAULT[k])(best[k]) for k in HP_DEFAULT}


if RUN_HPO:
    HP["text"] = run_hpo("text", cfg("hpo_text", "text"), HPO_SEED)
    HP["audio"] = run_hpo("audio", cfg("hpo_audio", "audio", "all", 1.0), HPO_SEED + 1)
HP["fusion"] = dict(HP["text"])           # the fusion model is text-dominated -> text hyperparameters
print("\n>>> Hyperparameters used:", json.dumps(HP, indent=1))
LAYER_CFG = dict(lora=LORA_CFG if NEED_LORA else None, layer_attn=LAYER_ATTN, kind=LAYER_ATTN_KIND, layers=[LA0, LA1], init=LAYER_INIT,
                 layer_drop=LAYER_DROP, lr_mult=LAYER_LR_MULT, fixed_window=list(WAVLM_LAYERS),
                 frame_attention=dict(mode=FRAME_ATTN, pool=FRAME_POOL, frame_ms=FRAME_POOL * 20,
                                      n_frames=N_FRAMES, layers=FRAME_IDS, hidden=FRAME_ATTN_HIDDEN,
                                      drop=FRAME_DROP))
with open(os.path.join(OUT_DIR, "chosen_hyperparameters.json"), "w") as f:
    json.dump(dict(HP=HP, audio_gm=SEL_GM, audio_enc=SEL_ENC, layer_attention=LAYER_CFG), f, indent=2)


# ============================================================
# TEXT-GUIDED CHUNK SCORES (from the trained text models)
# ============================================================
class TextSel(nn.Module):
    """Mean sentence / window attention of several text-only models (exported as text_selector.onnx)."""
    def __init__(self, members):
        super().__init__(); self.members = nn.ModuleList(members)

    def forward(self, text):
        outs = [m.encode_text(text) for m in self.members]
        return torch.stack([o[1] for o in outs]).mean(0), torch.stack([o[2] for o in outs]).mean(0)


def sentence_importance(ws, ww, n_s):
    imp = np.zeros(n_s); np.add.at(imp, window_index(n_s), np.asarray(ww)[:, None] * np.asarray(ws))
    return imp


@torch.no_grad()
def build_text_scores(text_members):
    sels = [TextSel([m for k, m in enumerate(text_members) if k % N_FOLDS == f]).eval() for f in range(N_FOLDS)]
    sel_all = TextSel(text_members).eval()

    def row(sel, s):
        if s["ov"] is None:
            return np.zeros(len(s["audio"]))
        idx = torch.from_numpy(window_index(len(s["text"]))).to(device)
        ws, ww = sel(s["text"][idx])
        return s["ov"] @ sentence_importance(ws.cpu().numpy(), ww.cpu().numpy(), len(s["text"]))
    for s in dev_s:
        s["tsc"] = np.stack([row(sl, s) for sl in sels])
    for s in test_s:
        s["tsc"] = row(sel_all, s)[None]
    print("Text-guided chunk scores ready (dev: per fold, test: all text models).")


# ============================================================
# MAIN EXPERIMENTS
# ============================================================
rows, ens = [], {}
for c in MAIN_CONFIGS:
    if c["selector"] == "text" and dev_s[0]["tsc"] is None:
        raise RuntimeError("Put the 'text' config before any config with selector='text' in MAIN_CONFIGS.")
    ens[c["name"]], r_ = run_config(c, SEEDS)
    rows += r_
    if c["name"] == "text":
        build_text_scores(ens["text"]["members"])
for _k, _v in LORA_E2E.items():                                # [R3] Stage-A models as audio classifiers
    ens[_k] = _v


# ============================================================
# LOGISTIC-REGRESSION BASELINES (all chunks, same folds, same threshold protocol)
# ============================================================
def feat_matrix(entries, kind):
    if kind == "text":
        return np.stack([e["text"].mean(0).numpy() for e in entries])
    if kind == "text_nd":
        return np.stack([e["text_nd"].mean(0).numpy() for e in entries])
    if kind == "wavlm":
        return np.stack([e["audio"].mean(0).numpy() for e in entries])
    if kind == "timing":
        return np.stack([e["timing"] for e in entries])
    if kind == "timing_raw":
        return np.stack([e["timing_raw"] for e in entries])
    return np.stack([e["gm"] for e in entries])


def lr_baseline(kinds):
    Xd = np.hstack([feat_matrix(dev_entries, k) for k in kinds])
    Xt = np.hstack([feat_matrix(test_entries, k) for k in kinds])
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))
    oof = np.zeros(len(ydev))
    for tr, va in folds:
        oof[va] = mk().fit(Xd[tr], ydev[tr]).predict_proba(Xd[va])[:, 1]
    thr, _ = best_threshold(ydev, oof)
    p = mk().fit(Xd, ydev).predict_proba(Xt)[:, 1]
    return dict(cv_auc=roc_auc_score(ydev, oof), thr=thr, p=p, oof=oof, **compute_metrics(yte, p, thr))


baseline_sets = {"LR-text": ["text"], "LR-text-nodiag": ["text_nd"], "LR-wavlm": ["wavlm"],
                 "LR-fusion": ["text", "wavlm"]}
if USE_EGEMAPS:
    baseline_sets.update({"LR-eGeMAPS": ["gm"], "LR-fusion+eGeMAPS": ["text", "wavlm", "gm"]})
if USE_TIMING:                                                  # [T3]
    baseline_sets.update({"LR-timing": ["timing"], "LR-text+timing": ["text", "timing"]})
    if TIMING_BY_CONDITION:
        baseline_sets["LR-timing-raw"] = ["timing_raw"]                       # [V2] ablation
baselines = {name: lr_baseline(k) for name, k in baseline_sets.items()}
if USE_TIMING:
    _b = baselines["LR-timing"]
    ens["timing_lr"] = dict(name="timing_lr", mode="e2e", desc="LR on timing / turn-taking / pause features",
                            selector="all", frac=1.0, gm=False, enc="lr", la=False, fa="off", lora=False,
                            distill=False, moddrop=False, text_key="text", hp={}, infos=[], thr=_b["thr"],
                            oof_p=_b["oof"], test_p=_b["p"], cv=compute_metrics(ydev, _b["oof"], _b["thr"]),
                            test=compute_metrics(yte, _b["p"], _b["thr"]), sel={})
    _clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))
    _clf.fit(feat_matrix(dev_entries, "timing"), ydev)
    _coef = pd.Series(_clf[-1].coef_[0], index=TIMING_NAMES).sort_values()
    print(f"\n[T3] timing_lr: CV AUC {_b['cv_auc']:.3f} | TEST AUC {_b['AUC']:.3f}")
    print("     strongest standardised coefficients (+ = more likely depressed):\n",
          pd.concat([_coef.head(5), _coef.tail(5)]).round(3).to_string())
    _coef.to_csv(os.path.join(OUT_DIR, "timing_lr_coefficients.csv"))


# ============================================================
# [N1] LATE FUSION: stacking on out-of-fold log-odds
# ============================================================
LOGIT_CLIP = 1e-4
_lo = lambda p: logit(np.clip(p, LOGIT_CLIP, 1 - LOGIT_CLIP))


def late_fusion(name, parts, C=LATE_C):
    Xd = np.column_stack([_lo(ens[n]["oof_p"]) for n in parts])
    Xt = np.column_stack([_lo(ens[n]["test_p"]) for n in parts])
    mk = lambda: LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
    oof = np.zeros(len(ydev))
    for tr, va in folds:                                  # the stacker gets its own honest CV
        oof[va] = mk().fit(Xd[tr], ydev[tr]).decision_function(Xd[va])
    oof_p = sigmoid(oof); thr, _ = best_threshold(ydev, oof_p)
    clf = mk().fit(Xd, ydev); test_p = sigmoid(clf.decision_function(Xt))
    a = ens[parts[-1]]
    out = dict(name=name, mode="late", parts=list(parts), coef=clf.coef_[0].tolist(),
               intercept=float(clf.intercept_[0]), selector=a["selector"], frac=a["frac"], gm=a["gm"],
               enc=a["enc"], la=a.get("la", False), fa=a.get("fa", "off"), lora=a.get("lora", False),
               distill=a["distill"], moddrop=False, thr=thr, oof_p=oof_p,
               test_p=test_p, cv=compute_metrics(ydev, oof_p, thr), test=compute_metrics(yte, test_p, thr))
    print(f"  late {parts}: weights {np.round(out['coef'], 3)} | CV AUC {out['cv']['AUC']:.3f} | "
          f"TEST AUC {out['test']['AUC']:.3f}")
    return out


print(f"\nLate fusion candidates (for information; audio partner "
      f"{'fixed in advance: ' + LATE_AUDIO_FIXED if LATE_AUDIO_FIXED else 'chosen by dev CV AUC'}):")
late_c = {a: late_fusion(f"late[{a}]", ["text", a]) for a in LATE_AUDIO_CANDIDATES if a in ens}
LATE_AUDIO = (LATE_AUDIO_FIXED if LATE_AUDIO_FIXED in late_c                       # [W2]
              else max(late_c, key=lambda a: late_c[a]["cv"]["AUC"]))
ens["late"] = dict(late_c[LATE_AUDIO], name="late")
if "text_nodiag" in ens:
    ens["late_nodiag"] = late_fusion("late_nodiag", ["text_nodiag", LATE_AUDIO])
if "audio_lora_e2e" in ens:                                    # [R3] reported, not deployed
    ens["late_e2e"] = late_fusion("late_e2e", ["text", "audio_lora_e2e"])
if "timing_lr" in ens:                                         # [T3]
    ens["late_timing"] = late_fusion("late_timing", ["text", "timing_lr"])
    ens["late3"] = late_fusion("late3", ["text", LATE_AUDIO, "timing_lr"])
print(f">>> Late fusion uses audio model '{LATE_AUDIO}', weights (text, audio) = {np.round(ens['late']['coef'], 3)}")


# ============================================================
# RESULTS TABLES
# ============================================================
res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT_DIR, "results_per_seed.csv"), index=False)
fmt = lambda m, s: f"{m:.3f} ± {s:.3f}"


def hubert_label(E_):
    """[W3] Honest encoder label: LoRA only if some fold actually kept its adapters."""
    if not E_.get("lora"):
        return "frozen HuBERT"
    kept = sum(bool(LORA_INFO[f].get("lora_used", True)) for f in LORA_INFO)
    return f"LoRA HuBERT ({kept}/{len(LORA_INFO)} folds adapted)" if kept else "LoRA rejected (= frozen HuBERT)"


table = []
for name, E_ in ens.items():
    if E_["mode"] == "e2e":
        table.append(dict(model=f"{name} [{E_.get('desc', 'Stage-A head')}]",
                          CV_AUC=f"{E_['cv']['AUC']:.3f}", **{m: f"{E_['test'][m]:.3f}" for m in METRICS}))
        continue
    if E_["mode"] == "late":
        table.append(dict(model=f"{name} [stack {E_['parts']}, w={np.round(E_['coef'], 2).tolist()}]",
                          CV_AUC=f"{E_['cv']['AUC']:.3f}", **{m: f"{E_['test'][m]:.3f}" for m in METRICS}))
        continue
    r = res[res["name"] == name]
    extra = "" if E_["mode"] == "text" else (f", {E_['enc']}{', ' + SPK_LABEL if E_['gm'] else ''}"
                                             f", {'layer-attn ' + LAYER_ATTN_KIND if E_.get('la') else 'layer-mean'}"
                                             f", frames-{E_.get('fa', 'off')}"
                                             f", {hubert_label(E_)}")
    tag = (f"{name} [{E_['selector']}, keep {E_['frac']:.0%}{', distilled' if E_.get('distill') else ''}"
           f"{', moddrop' if E_.get('moddrop') else ''}{', no-diag text' if E_['text_key'] == 'text_nd' else ''}"
           f"{extra}]")
    table.append(dict(model=f"{tag} per-seed", CV_AUC=fmt(r.cv_auc.mean(), r.cv_auc.std()),
                      **{m: fmt(r[m].mean(), r[m].std()) for m in METRICS}))
    table.append(dict(model=f"{tag} ensemble {len(SEEDS)}x{N_FOLDS}", CV_AUC=f"{E_['cv']['AUC']:.3f}",
                      **{m: f"{E_['test'][m]:.3f}" for m in METRICS}))
for name, b in baselines.items():
    table.append(dict(model=name, CV_AUC=f"{b['cv_auc']:.3f}", **{m: f"{b[m]:.3f}" for m in METRICS}))
table = pd.DataFrame(table).set_index("model")
print("\nTEST RESULTS (threshold from dev CV)\n", table.to_string())
table.to_csv(os.path.join(OUT_DIR, "results_summary.csv"))


def _sid_num(sid):
    try:
        return int(sid)
    except ValueError:
        return -1


dev_ai = np.array([_sid_num(s["sid"]) >= 600 for s in dev_s])
diag = []
for name, E_ in ens.items():
    d = dict(config=name, cv_auc=E_["cv"]["AUC"], test_auc=E_["test"]["AUC"])
    for lab, m in [("cv_auc_AI", dev_ai), ("cv_auc_WoZ", ~dev_ai)]:
        d[lab] = roc_auc_score(ydev[m], E_["oof_p"][m]) if m.sum() > 5 and len(np.unique(ydev[m])) == 2 else np.nan
    ci_ = bootstrap_ci(yte, E_["test_p"], E_["thr"])["AUC"]
    d["test_auc_ci"] = f"[{ci_.iloc[0]:.3f}, {ci_.iloc[1]:.3f}]"
    diag.append(d)
diag = pd.DataFrame(diag)
print(f"\nDiagnostics (dev AI-agent n={dev_ai.sum()}, WoZ n={(~dev_ai).sum()}):\n", diag.round(3).to_string(index=False))
diag.to_csv(os.path.join(OUT_DIR, "diagnostics_condition.csv"), index=False)


# ---- Paired bootstrap ΔAUC (+ [N7] non-inferiority) ----
def _paired(y, pa, pb, n=2 * N_BOOT, seed=0):
    rng, d = np.random.RandomState(seed), []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    d = np.array(d)
    lo_, hi_ = np.percentile(d, 2.5), np.percentile(d, 97.5)
    return dict(delta=roc_auc_score(y, pa) - roc_auc_score(y, pb), ci_low=lo_, ci_high=hi_,
                p_one_sided=float((d <= 0).mean()), noninferior=bool(lo_ > -NONINF_MARGIN))


paired_auc_diff = lambda pa, pb: _paired(yte, pa, pb)
paired_oof_diff = lambda pa, pb: _paired(ydev, pa, pb)

probs = {k: v["test_p"] for k, v in ens.items()}
probs.update({k: v["p"] for k, v in baselines.items()})
PAIRS = [("late", "text"), ("fusion", "text"), ("late", "fusion"),
         ("audio_all", "audio_all_layermean"), ("fusion", "fusion_layermean"),          # [L2] layer attention
         ("audio_all", "audio_all_framemean"), ("fusion", "fusion_framemean"),          # [F2] frame attention
         ("audio_all", "audio_all_frozen"), ("fusion", "fusion_frozen"),                # [R2] LoRA vs frozen
         ("audio_lora_e2e", "audio_probe_e2e"), ("audio_lora_e2e", "audio_all_frozen"),     # [P3] adapters
         ("audio_lora_e2e", "LR-wavlm"), ("late_e2e", "text"),
         ("late_e2e", "late"), ("fusion", "fusion_moddrop"),
         ("timing_lr", "LR-wavlm"), ("late_timing", "text"), ("late3", "late"), ("late3", "text"),   # [T3]
         ("fusion", "fusion_nospk"), ("LR-text+timing", "LR-text"), ("LR-timing", "LR-timing-raw"),
         ("audio_all", "audio_all_frozen"),
         ("fusion", "fusion_nodrop"), ("fusion", "fusion_all"), ("text", "text_nodiag"),
         ("audio_textsel", "audio_all"), ("audio_textsel", "audio_learned"),
         ("audio_textsel_distill", "audio_textsel"), ("late", "LR-fusion"), ("fusion", "LR-fusion")]
PAIRS = list(dict.fromkeys(PAIRS + [("LR-fusion", "LR-text")]))                  # [W3] no duplicates
paired = pd.DataFrame([dict(A=a, B=b, **paired_auc_diff(probs[a], probs[b])) for a, b in PAIRS
                       if a in probs and b in probs])
print(f"\nPaired bootstrap ΔAUC (A − B) on TEST ({len(yte)} speakers, low power; "
      f"noninferior = lower CI > -{NONINF_MARGIN}):\n", paired.round(3).to_string(index=False))
paired.to_csv(os.path.join(OUT_DIR, "paired_auc_tests.csv"), index=False)
paired_cv = pd.DataFrame([dict(A=a, B=b, **paired_oof_diff(ens[a]["oof_p"], ens[b]["oof_p"])) for a, b in PAIRS
                          if a in ens and b in ens])
print("\nPaired bootstrap ΔCV-AUC (A − B) on DEV out-of-fold predictions (main evidence):\n",
      paired_cv.round(3).to_string(index=False))
paired_cv.to_csv(os.path.join(OUT_DIR, "paired_cv_tests.csv"), index=False)


# ============================================================
# [W4] AUDIO SUMMARY: how good is audio alone, and what does it add to each kind of fusion?
# ============================================================
def _auc_ci(y, p, n=N_BOOT, seed=0):
    rng, a = np.random.RandomState(seed), []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            a.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), np.percentile(a, 2.5), np.percentile(a, 97.5)


_arows = []
for _n in ["audio_all", "audio_all_frozen", "audio_textsel", "audio_textsel_distill", "audio_learned",
           "audio_lora_e2e", "timing_lr"]:
    if _n in ens:
        d_, dl, dh = _auc_ci(ydev, ens[_n]["oof_p"]); t_, tl, th = _auc_ci(yte, ens[_n]["test_p"])
        _arows.append(dict(model=_n, dev_auc=d_, dev_ci=f"[{dl:.3f}, {dh:.3f}]", test_auc=t_, test_ci=f"[{tl:.3f}, {th:.3f}]",
                           dev_above_chance=bool(dl > 0.5)))
_net = []
for _a, _b, _what in [("late", "text", f"late fusion: text + {LATE_AUDIO} vs text"),
                      ("fusion", "text", "neural fusion vs text"),
                      ("fusion_nospk", "text", "neural fusion without speaker branch vs text"),
                      ("late3", "late", "late fusion + timing vs late fusion")]:
    if _a in ens and _b in ens:
        dv, tv = paired_oof_diff(ens[_a]["oof_p"], ens[_b]["oof_p"]), paired_auc_diff(ens[_a]["test_p"], ens[_b]["test_p"])
        _net.append(dict(comparison=_what, dev_delta=dv["delta"], dev_ci=f"[{dv['ci_low']:.3f}, {dv['ci_high']:.3f}]",
                         test_delta=tv["delta"], test_ci=f"[{tv['ci_low']:.3f}, {tv['ci_high']:.3f}]"))
if "LR-fusion" in baselines:
    tv = paired_auc_diff(baselines["LR-fusion"]["p"], baselines["LR-text"]["p"])
    dv = paired_oof_diff(baselines["LR-fusion"]["oof"], baselines["LR-text"]["oof"])
    _net.append(dict(comparison="logistic regression: text + HuBERT vs text", dev_delta=dv["delta"],
                     dev_ci=f"[{dv['ci_low']:.3f}, {dv['ci_high']:.3f}]", test_delta=tv["delta"],
                     test_ci=f"[{tv['ci_low']:.3f}, {tv['ci_high']:.3f}]"))
_L = ens["late"]
_sd = [np.std(_lo(ens[p_]["oof_p"])) for p_ in _L["parts"]]
_std_w = np.abs(np.array(_L["coef"]) * np.array(_sd))
print("\n================ AUDIO SUMMARY [W4] ================")
print("Audio-only models (AUC with 95% bootstrap CI):\n", pd.DataFrame(_arows).round(3).to_string(index=False))
print("\nNet effect of audio (ΔAUC, paired bootstrap):\n", pd.DataFrame(_net).round(3).to_string(index=False))
print(f"\nLate fusion: standardised weights {dict(zip(_L['parts'], np.round(_std_w, 3)))} -> audio share "
      f"{_std_w[1:].sum() / _std_w.sum():.1%} of the evidence")
pd.DataFrame(_arows).to_csv(os.path.join(OUT_DIR, "audio_only_summary.csv"), index=False)
pd.DataFrame(_net).to_csv(os.path.join(OUT_DIR, "audio_net_effect.csv"), index=False)

# ============================================================
# [N8] FINAL SYSTEM = neural fusion or late fusion, chosen by dev CV AUC
# ============================================================
FINAL = max([k for k in ("fusion", "late") if k in ens], key=lambda k: ens[k]["cv"]["AUC"])
FIN = ens[FINAL]; THR_FIN, p_fin = FIN["thr"], FIN["test_p"]
yhat_fin = (p_fin >= THR_FIN).astype(int)
print(f"\n>>> FINAL SYSTEM = '{FINAL}' (dev CV AUC: fusion {ens['fusion']['cv']['AUC']:.3f}, "
      f"late {ens['late']['cv']['AUC']:.3f}) | threshold {THR_FIN:.3f}")
if THR_STABLE:                                                             # [W1] how stable is the threshold?
    _bt = bootstrap_thresholds(ydev, FIN["oof_p"])
    _dv = compute_metrics(ydev, FIN["oof_p"], THR_FIN)
    print(f"    threshold = median of {THR_BOOT} bootstrap optima (95% range {np.percentile(_bt, 2.5):.3f} - "
          f"{np.percentile(_bt, 97.5):.3f}) | dev operating point: sensitivity {_dv['Recall']:.3f}, "
          f"specificity {_dv['Specificity']:.3f}")
print(classification_report(yte, yhat_fin, digits=4))
ci = bootstrap_ci(yte, p_fin, THR_FIN)
ci_table = pd.DataFrame({m: [FIN["test"][m], ci.loc[0.025, m], ci.loc[0.975, m]] for m in METRICS},
                        index=["point", "ci_low", "ci_high"]).T
print(ci_table.round(4)); ci_table.to_csv(os.path.join(OUT_DIR, "final_bootstrap_ci.csv"))

if GENDER:
    groups = np.array([str(GENDER.get(s["sid"], "unknown")) for s in test_s])
    sub = [dict(group=g, n=int((groups == g).sum()),
                **compute_metrics(yte[groups == g], p_fin[groups == g], THR_FIN))
           for g in np.unique(groups) if len(np.unique(yte[groups == g])) == 2]
    if sub:
        print("\nSubgroup performance (final system)\n", pd.DataFrame(sub).round(4).to_string(index=False))
        pd.DataFrame(sub).to_csv(os.path.join(OUT_DIR, "subgroup_gender.csv"), index=False)

# ---- Plots ----
F = ens["fusion"]                       # neural fusion: used for training curves + interpretability
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for info in F["infos"]:
    ax[0].plot(info["hist"]["train_loss"], color="tab:blue", alpha=0.3)
    ax[1].plot(info["hist"]["val_auc"], color="tab:orange", alpha=0.3)
ax[0].set_title("Train loss (fusion members)"); ax[1].set_title("Early-stopping split AUC (fusion members)")
for a in ax: a.set_xlabel("Epoch")
savefig("training_curves.png")

plt.figure(figsize=(4, 3.5))
sns.heatmap(confusion_matrix(yte, yhat_fin), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-dep", "Dep"], yticklabels=["Non-dep", "Dep"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Final system: {FINAL} (test)")
savefig("confusion_matrix.png")

ROC_SHOW = ["text", "audio_all", "audio_all_frozen", "audio_lora_e2e", "audio_probe_e2e", "audio_textsel",
            "fusion", "fusion_frozen", "fusion_all", "timing_lr", "late", "late3"]
plt.figure(figsize=(4.8, 4.2))
for name in [n for n in ROC_SHOW if n in ens]:
    fpr, tpr, _ = roc_curve(yte, ens[name]["test_p"])
    plt.plot(fpr, tpr, lw=2.0 if name == FINAL else 1.0, label=f"{name} (AUC={ens[name]['test']['AUC']:.3f})")
bl = "LR-fusion+eGeMAPS" if USE_EGEMAPS else "LR-fusion"
fpr, tpr, _ = roc_curve(yte, baselines[bl]["p"])
plt.plot(fpr, tpr, "--", label=f"{bl} (AUC={baselines[bl]['AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k:", lw=0.8)
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(fontsize=7); plt.title("ROC (test)")
savefig("roc_modalities.png")


# ============================================================
# [L3] LEARNED LAYER WEIGHTS (mean over chunks and test speakers, one row per ensemble member)
# ============================================================
@torch.no_grad()
def layer_weight_profile(members, samples):
    out = []
    for m in members:
        m.eval(); acc = []
        for s in samples:
            m.encode_audio(audio_feats(m, s, pick(m, s)).unsqueeze(0))
            acc.append(m.layer_mix.last_w.reshape(-1, m.layer_mix.n).mean(0).cpu().numpy())
        out.append(np.mean(acc, 0))
    return np.array(out)


LAYER_W, LAYER_IDS, LAYER_SUMMARY = {}, {}, {}
for name, E_ in ens.items():
    if E_.get("la") and E_.get("members"):
        LAYER_W[name] = layer_weight_profile(E_["members"], test_s)
        LAYER_IDS[name] = np.asarray(E_["members"][0].layer_ids)
if LAYER_W:
    lw_rows = []
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for name, W in LAYER_W.items():
        ids_, mu, sd = LAYER_IDS[name], W.mean(0), W.std(0)
        eff = float(np.exp(-(mu * np.log(mu + 1e-12)).sum()))          # effective number of layers
        rho = stats.spearmanr(mu, layer_auc[ids_]).correlation if len(ids_) > 2 else np.nan
        LAYER_SUMMARY[name] = dict(layers=ids_.tolist(), weights=np.round(mu, 4).tolist(),
                                   sd=np.round(sd, 4).tolist(), top_layer=int(ids_[mu.argmax()]),
                                   effective_layers=round(eff, 2), spearman_probe=float(rho))
        lw_rows += [dict(config=name, layer=int(l), weight_mean=m_, weight_sd=s_, probe_auc=layer_auc[l])
                    for l, m_, s_ in zip(ids_, mu, sd)]
        print(f"[L3] {name}: top layer {ids_[mu.argmax()]} (w={mu.max():.3f}), effective layers {eff:.2f}/{len(ids_)}, "
              f"Spearman(weight, probe AUC) = {rho:.2f}")
        ax.errorbar(ids_, mu, yerr=sd, marker="o", ms=3, capsize=2, label=f"{name} (uniform {1 / len(ids_):.2f})")
    pd.DataFrame(lw_rows).to_csv(os.path.join(OUT_DIR, "learned_layer_weights.csv"), index=False)
    ax.axvspan(WAVLM_LAYERS[0] - 0.5, WAVLM_LAYERS[1] - 0.5, alpha=0.1, color="tab:orange")
    ax.set_xlabel("Hidden state (0 = CNN features)"); ax.set_ylabel("Learned layer weight")
    ax2 = ax.twinx(); ax2.plot(np.arange(N_LAY_ALL), layer_auc, "k--", alpha=0.5); ax2.set_ylabel("Probe CV AUC (dashed)")
    ax.legend(fontsize=7, loc="upper left"); ax.set_title("Learned layer attention vs linear probe")
    savefig("learned_layer_weights.png")


# ============================================================
# [F3] FRAME ATTENTION INSIDE THE 5-s CHUNKS: which regions does each model emphasise?
# ============================================================
@torch.no_grad()
def frame_attention_profile(E_):
    """Per test speaker: frame weights (K, N_FRAMES) on the ensemble-selected chunks, averaged over members."""
    out = []
    for s in test_s:
        sel, A = E_["sel"][s["sid"]], []
        for m in E_["members"]:
            m.eval(); m.encode_audio(audio_feats(m, s, sel).unsqueeze(0))
            A.append(m.frame_pool.last_a.reshape(-1, N_FRAMES).cpu().numpy())
        out.append(np.mean(A, 0))
    return out


FRAME_SUMMARY = {}
for name in [n for n, E_ in ens.items() if E_.get("fa") == "attn" and E_.get("members")]:
    E_ = ens[name]
    A_list = frame_attention_profile(E_)
    eff, rho, pos = [], [], []
    for s, A in zip(test_s, A_list):
        en = FRAME_ENERGY[s["f_off"] + np.asarray(E_["sel"][s["sid"]])]
        eff.append(np.exp(-(A * np.log(A + 1e-12)).sum(1)).mean())          # effective number of frames
        pos.append(A.mean(0))
        r_ = [stats.spearmanr(a_, e_).correlation for a_, e_ in zip(A, en) if np.ptp(e_) > 0 and np.ptp(a_) > 1e-6]
        rho.append(np.nanmedian(r_) if r_ else np.nan)
    eff, rho, pos = np.array(eff), np.array(rho), np.array(pos)
    ok = ~np.isnan(rho)
    p_eff = stats.mannwhitneyu(eff[yte == 1], eff[yte == 0]).pvalue
    p_rho = stats.wilcoxon(rho[ok]).pvalue if ok.sum() > 5 else np.nan
    FRAME_SUMMARY[name] = dict(n_frames=N_FRAMES, frame_ms=FRAME_POOL * 20,
                               eff_frames_dep=float(eff[yte == 1].mean()), eff_frames_nondep=float(eff[yte == 0].mean()),
                               p_eff_dep_vs_nondep=float(p_eff), spearman_attn_energy_median=float(np.nanmedian(rho)),
                               p_spearman_vs_0=float(p_rho), position_profile=np.round(pos.mean(0), 4).tolist())
    print(f"[F3] {name}: effective frames per chunk {eff.mean():.1f}/{N_FRAMES} (dep {eff[yte == 1].mean():.1f} vs "
          f"non-dep {eff[yte == 0].mean():.1f}, p={p_eff:.3g}) | Spearman(attention, frame energy) median "
          f"{np.nanmedian(rho):.2f} (p={p_rho:.3g})")
    if eff.mean() > 0.97 * N_FRAMES:
        print(f"  NOTE: {name} frame attention stayed close to uniform (the data gave no reason to move away "
              f"from the mean); compare with '{name.replace('audio_all', 'audio_all_framemean')}' / framemean.")
    pd.DataFrame(dict(sid=[s["sid"] for s in test_s], y=yte, eff_frames=eff, spearman_attn_energy=rho)).to_csv(
        os.path.join(OUT_DIR, f"frame_attention_{name}.csv"), index=False)

    t_ms = (np.arange(N_FRAMES) + 0.5) * FRAME_POOL * 20
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    for y_, lab in [(1, "Dep"), (0, "Non-dep")]:
        ax[0].plot(t_ms, pos[yte == y_].mean(0), marker="o", ms=3, label=lab)
    ax[0].axhline(1 / N_FRAMES, c="k", ls=":", lw=0.8, label="uniform")
    ax[0].set_xlabel("Position inside chunk (ms)"); ax[0].set_ylabel("Mean frame weight"); ax[0].legend(fontsize=7)
    ax[0].set_title("Where inside the chunk", fontsize=9)
    ax[1].hist(rho[ok], bins=15, color="tab:gray"); ax[1].axvline(0, c="k", lw=0.8)
    ax[1].set_xlabel("Spearman(frame weight, frame energy)"); ax[1].set_title("Attention vs loudness", fontsize=9)
    dep_idx = [i for i in range(len(test_s)) if yte[i] == 1]
    if dep_idx:
        i = max(dep_idx, key=lambda j: E_["test_p"][j]); A = A_list[i]
        c = int(np.argmin(np.exp(-(A * np.log(A + 1e-12)).sum(1))))          # most peaked chunk
        en = FRAME_ENERGY[test_s[i]["f_off"] + int(E_["sel"][test_s[i]["sid"]][c])]
        ax[2].bar(t_ms, A[c], width=FRAME_POOL * 20 * 0.9, color="tab:red", alpha=0.7, label="frame weight")
        ax2 = ax[2].twinx(); ax2.plot(t_ms, en, "k-", lw=1); ax2.set_ylabel("Energy (dB)")
        ax[2].set_xlabel("Time inside chunk (ms)"); ax[2].set_ylabel("alpha_t")
        ax[2].set_title(f"Example: speaker {test_s[i]['sid']}, chunk {c}", fontsize=9)
    savefig(f"frame_attention_{name}.png")


# ============================================================
# UPLINK / COMPUTE TABLE
# ============================================================
def uplink_table(fracs):
    T = np.array([len(s["audio"]) for s in test_s]); n_s = np.array([len(s["text"]) for s in test_s])
    out = []
    for f in fracs:
        K = np.array([n_keep(t, f) for t in T])
        out.append(dict(keep_frac=f, chunks_total=T.mean(), chunks_kept=K.mean(),
                        raw_audio_all_KB=(T * CHUNK_SIZE * 2).mean() / 1024,
                        raw_audio_topk_KB=(K * CHUNK_SIZE * 2).mean() / 1024,
                        wavlm_emb_all_KB=(T * 768 * 2).mean() / 1024,
                        wavlm_emb_topk_KB=(K * 768 * 2).mean() / 1024,
                        # [L1] uplink if the head runs off-device: all layers of the K chunks
                        wavlm_layers_topk_KB=(K * N_LAYERS * 768 * 2).mean() / 1024,
                        # [F4] uplink if the head runs off-device with frame attention: all frames of the K chunks
                        wavlm_frames_topk_KB=(K * N_FRAMES * N_FLAYERS * 768 * 2).mean() / 1024,
                        text_emb_KB=(n_s * D_TEXT * 2).mean() / 1024,
                        speaker_feats_KB=N_GM * 4 / 1024,
                        wavlm_calls_saved_pct=100 * (1 - K.sum() / T.sum())))
    return pd.DataFrame(out)


up = uplink_table(sorted(set(SWEEP_FRACS + [KEEP_FRAC, 1.0])))
up["raw_reduction_x"] = up.raw_audio_all_KB / up.raw_audio_topk_KB
up["topk_emb_vs_all_raw_x"] = up.raw_audio_all_KB / up.wavlm_emb_topk_KB
up["topk_layers_vs_all_raw_x"] = up.raw_audio_all_KB / up.wavlm_layers_topk_KB
print("\nPer-session uplink (test speakers, participant speech only). wavlm_emb_* = one mixed 768-d vector per chunk "
      "(head on device); wavlm_layers_* = all layers (head off device, layer attention):\n",
      up.round(2).to_string(index=False))
up.to_csv(os.path.join(OUT_DIR, "uplink_per_session.csv"), index=False)

# ============================================================
# K-SWEEP: accuracy vs fraction of audio kept, per selector
# ============================================================
if RUN_SWEEP:
    sweep, sweep_oof = [], {}
    configs = [(lab, sk, f) for f in SWEEP_FRACS for lab, sk in SWEEP_VARIANTS] + [("all", "all", 1.0)]
    for lab, sk, f in configs:
        E_, _ = run_config(cfg(f"sweep_{SWEEP_MODE}_{lab}_{f}", SWEEP_MODE, sk, f), SWEEP_SEEDS,
                           keep_members=False, verbose=False)
        sweep_oof[(lab, f)] = E_["oof_p"]
        sweep.append(dict(variant=lab, selector=sk, keep_frac=f, cv_auc=E_["cv"]["AUC"],
                          cv_F1_macro=E_["cv"]["F1_macro"], **{f"test_{m}": E_["test"][m] for m in METRICS}))
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(os.path.join(OUT_DIR, f"topk_sweep_{SWEEP_MODE}.csv"), index=False)
    print(f"\nK-sweep ({SWEEP_MODE}, {len(SWEEP_SEEDS)} seeds):\n", sweep.round(3).to_string(index=False))

    cmp_pairs = [(SWEEP_REF, lab) for lab, _ in SWEEP_VARIANTS if lab != SWEEP_REF] + [("learned", "random")]
    srows = []
    for f in SWEEP_FRACS:
        for a, b in cmp_pairs:
            if (a, f) in sweep_oof and (b, f) in sweep_oof:
                srows.append(dict(keep_frac=f, A=a, B=b, **paired_oof_diff(sweep_oof[(a, f)], sweep_oof[(b, f)])))
        if (SWEEP_REF, f) in sweep_oof:
            srows.append(dict(keep_frac=f, A=SWEEP_REF, B="all chunks",
                              **paired_oof_diff(sweep_oof[(SWEEP_REF, f)], sweep_oof[("all", 1.0)])))
    if srows:
        sel_tests = pd.DataFrame(srows)
        print("\nSelector comparisons at equal K (paired bootstrap ΔCV-AUC, A − B):\n",
              sel_tests.round(3).to_string(index=False))
        sel_tests.to_csv(os.path.join(OUT_DIR, f"selector_tests_{SWEEP_MODE}.csv"), index=False)

    ref = sweep[sweep.variant == "all"].iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.5))
    for lab, g in sweep[sweep.variant != "all"].groupby("variant"):
        g = g.sort_values("keep_frac")
        ls = "-" if lab == SWEEP_REF else "--"
        ax[0].plot(g.keep_frac * 100, g.cv_auc, ls, marker="o", label=lab)
        ax[1].plot(g.keep_frac * 100, g.test_AUC, ls, marker="o", label=lab)
    ax[0].axhline(ref.cv_auc, c="k", lw=0.8, ls=":", label="all chunks")
    ax[1].axhline(ref.test_AUC, c="k", lw=0.8, ls=":", label="all chunks")
    ax[0].set_ylabel(f"Dev CV AUC ({SWEEP_MODE})"); ax[1].set_ylabel(f"Test AUC ({SWEEP_MODE})")
    for a in ax:
        a.set_xlabel("% of audio chunks kept (K / T)")
    ax[0].legend(fontsize=7); savefig(f"topk_sweep_{SWEEP_MODE}.png")

# ============================================================
# INTERPRETABILITY (neural fusion Top-K ensemble, on the selected chunks)
# ============================================================
M = Ensemble(F["members"]).eval()
THR, p_test, SEL_TEST = F["thr"], F["test_p"], F["sel"]
F_GM = F["gm"]
F_LAYERED = bool(F.get("la"))
F_FA = F.get("fa", "off")
PLAYERS = F["members"][0].players
interp = dict(final_system=FINAL, wavlm_fixed_window=list(WAVLM_LAYERS),
              layer_probe_cv_auc=[round(float(a), 4) for a in layer_auc], hyperparameters=HP,
              layer_attention=dict(config=LAYER_CFG, learned=LAYER_SUMMARY), frame_attention=FRAME_SUMMARY,
              selector=F["selector"], keep_frac=F["frac"], egemaps_branch=F_GM, audio_encoder=F["enc"],
              modality_dropout=dict(audio=MOD_DROP_AUDIO, text=MOD_DROP_TEXT), phq_multitask=USE_PHQ,
              late_fusion=dict(audio_model=LATE_AUDIO, coef=ens["late"]["coef"], intercept=ens["late"]["intercept"]),
              inner_early_stopping=INNER_ES)


def exact_shapley(v, players):
    n, phi = len(players), {}
    for p in players:
        others, val = [q for q in players if q != p], 0.0
        for r in range(n):
            for S in combinations(others, r):
                w = factorial(r) * factorial(n - r - 1) / factorial(n)
                val += w * (v(frozenset(S) | {p}) - v(frozenset(S)))
        phi[p] = val
    return phi


@torch.no_grad()
def member_bases(member, tr_idx):
    feats = [member.player_feats(*full_inputs(dev_s[i], pick(member, dev_s[i]), member))[0] for i in tr_idx]
    return {p: torch.cat([f[p] for f in feats]).mean(0, keepdim=True) for p in member.players}


bases = [member_bases(m, info["tr_idx"]) for m, info in zip(F["members"], F["infos"])]


@torch.no_grad()
def speaker_shapley(s):
    acc = {p: 0.0 for p in PLAYERS}
    for m, base in zip(F["members"], bases):
        f = m.player_feats(*full_inputs(s, SEL_TEST[s["sid"]], m))[0]
        v = lambda S: logodds(m.classify({p: (f[p] if p in S else base[p]) for p in m.players}))[0].item()
        for p, val in exact_shapley(v, m.players).items():
            acc[p] += val / len(F["members"])
    return acc


shap = pd.DataFrame([dict(sid=s["sid"], y=s["y"], p=p, **{f"phi_{k}": v for k, v in speaker_shapley(s).items()})
                     for s, p in zip(test_s, p_test)])
absphi = shap[[f"phi_{p}" for p in PLAYERS]].abs()
share = absphi.div(absphi.sum(1) + 1e-9, axis=0)
share.columns = PLAYERS
shap.join(share.add_prefix("share_")).to_csv(os.path.join(OUT_DIR, "modality_shapley.csv"), index=False)
interp["shapley_mean_share"] = share.mean().round(4).to_dict()
print("\nMean share of evidence per branch (neural fusion):", interp["shapley_mean_share"])

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw=dict(width_ratios=[2, 1]))
long = shap.melt(id_vars="y", value_vars=[f"phi_{p}" for p in PLAYERS], var_name="branch", value_name="phi")
long["branch"] = long.branch.str.replace("phi_", ""); long["class"] = long.y.map({0: "Non-dep", 1: "Dep"})
sns.boxplot(data=long, x="branch", y="phi", hue="class", ax=ax[0])
ax[0].axhline(0, c="k", lw=0.6); ax[0].set_ylabel("Shapley value (log-odds of depression)")
share.mean().plot.bar(ax=ax[1], color="tab:gray"); ax[1].set_ylabel("Mean share of |evidence|")
savefig("modality_shapley.png")


TEXT_BASE = torch.tensor(XTEXT_DEV.mean(0), dtype=torch.float, device=device)


def ig_baselines(model, s, win, audio, gm, n):
    """[V3] Expected-Gradients-style baselines: the mean text vector + REAL chunks (and speaker vector) of n random
    dev speakers. Real inputs are never constant after standardisation, so the path never enters the
    scale-invariant point of the LayerNorms. Same baselines for every member (seeded by the speaker id)."""
    rng = np.random.RandomState(zlib.crc32(f"ig-{s['sid']}".encode()) % (2 ** 31))
    out = []
    for _ in range(n):
        sb = dev_s[rng.randint(len(dev_s))]
        idx = np.sort(rng.choice(len(sb["audio"]), audio.shape[1], replace=True))
        b = [TEXT_BASE.expand_as(win).clone(), audio_feats(model, sb, idx).unsqueeze(0).reshape(audio.shape)]
        if gm is not None:
            b.append((sb["gm"] if sb["gm"] is not None else GM_BASE).clone())
        out.append(b)
    return out


def _ig_path(model, xs, b, steps, gm):
    """Midpoint-rule path integral from baseline b to input xs."""
    grads = [torch.zeros_like(x) for x in xs]
    for k in range(steps):
        a = (k + 0.5) / steps
        cur = [(bb + a * (x - bb)).requires_grad_(True) for x, bb in zip(xs, b)]
        out = model(cur[0], cur[1], cur[2] if len(cur) > 2 else gm)
        gs = torch.autograd.grad(logodds(out)[0], cur, allow_unused=True)
        for g_acc, g in zip(grads, gs):
            if g is not None:
                g_acc += g
    with torch.no_grad():
        f_0 = logodds(model(*b[:2], b[2] if len(b) > 2 else gm))[0].item()
    return [((x - bb) * g / steps).detach() for x, bb, g in zip(xs, b, grads)], f_0


def integrated_gradients(model, s, sel, steps=IG_STEPS):
    if getattr(model, "per_member", False):            # [R1] members read different features: average member IG
        rs = [integrated_gradients(m, s, sel, steps) for m in model.members]
        return {k: (sum(r_[k] for r_ in rs) / len(rs) if rs[0][k] is not None else None) for k in rs[0]}
    model.eval()
    win, audio, gm = full_inputs(s, sel, model)
    use_gm = gm is not None and F_GM
    xs = [win, audio] + ([gm] if use_gm else [])
    acc, f0s, n_used = None, [], []
    with torch.backends.cudnn.flags(enabled=False):
        with torch.no_grad():
            f_x = logodds(model(*xs[:2], xs[2] if len(xs) > 2 else gm))[0].item()
        for b in ig_baselines(model, s, win, audio, gm if use_gm else None, IG_BASELINES):
            n = steps
            while True:                                 # [V3] refine until the path integral is complete
                attrs, f_0 = _ig_path(model, xs, b, n, gm)
                err = abs(sum(a.sum().item() for a in attrs) - (f_x - f_0))
                if err <= IG_TOL * max(abs(f_x - f_0), 1e-3) or n >= IG_STEPS_MAX:
                    break
                n *= 2
            acc = attrs if acc is None else [a0 + a1 for a0, a1 in zip(acc, attrs)]
            f0s.append(f_0); n_used.append(n)
    attrs = [a / len(f0s) for a in acc]
    f_0 = float(np.mean(f0s))
    total = sum(a.sum().item() for a in attrs)
    a_aud = attrs[1]          # (1, K, 768) | layer attention (1, K, L, 768) | frame attention (1, K, F, L_f, 768)
    a_ld = a_aud[0].sum(-1)   # (K,) | (K, L) | (K, F, L_f)
    return dict(win=attrs[0].sum(-1).cpu(), chunk=a_aud.flatten(2).sum(-1)[0].cpu().numpy(),
                layer=a_ld.reshape(-1, a_ld.shape[-1]).sum(0).cpu().numpy() if a_aud.dim() >= 4 else None,  # [L3]
                frame=a_ld.sum(-1).cpu().numpy() if a_aud.dim() == 5 else None,                           # [F3]
                gm=attrs[2][0].cpu().numpy() if len(attrs) > 2 else None,
                f_x=f_x, f_0=f_0, delta=total - (f_x - f_0), steps=float(np.mean(n_used)))


def per_sentence(values, n_s):
    idx = torch.from_numpy(window_index(n_s))
    return torch.zeros(n_s).index_add_(0, idx.flatten(), values.flatten().float().cpu()).numpy()


print("\nIntegrated Gradients on the neural fusion ensemble (selected chunks)")
ig = []
for s in tqdm(test_s):
    sel = SEL_TEST[s["sid"]]
    r = integrated_gradients(M, s, sel)
    with torch.no_grad():
        _, att = head_forward(M, s, sel, return_attn=True)
    n_s = len(s["text"])
    ig.append(dict(sid=s["sid"], y=s["y"], f_x=r["f_x"], f_0=r["f_0"], delta=r["delta"], steps=r["steps"],
                   sent_attr=per_sentence(r["win"], n_s), chunk_attr=r["chunk"], layer_attr=r["layer"],
                   frame_attr=r["frame"],
                   gm_attr=r["gm"], att_sent=per_sentence(att["w_win"].unsqueeze(1) * att["w_sent"], n_s),
                   att_audio=att["w_audio"].cpu().numpy()))

# [V3] completeness relative to what the attributions must explain: f(x) - f(baseline)
rel_delta = np.median([abs(r["delta"]) / max(abs(r["f_x"] - r["f_0"]), 1e-3) for r in ig])
print(f"IG: median steps per baseline {np.median([r['steps'] for r in ig]):.0f} | median |delta| "
      f"{np.median([abs(r['delta']) for r in ig]):.4f} (log-odds)")
rho_t = [stats.spearmanr(r["sent_attr"], r["att_sent"]).correlation for r in ig if len(r["sent_attr"]) > 3]
rho_a = [stats.spearmanr(np.abs(r["chunk_attr"]), r["att_audio"]).correlation for r in ig if len(r["chunk_attr"]) > 3]
print(f"IG completeness: median |delta| / |f(x) - f(baseline)| = {rel_delta:.4f} "
      f"({'OK' if rel_delta <= IG_TOL else 'raise IG_STEPS_MAX'})")
print(f"Attention vs IG (Spearman): text {np.nanmedian(rho_t):.3f}, audio {np.nanmedian(rho_a):.3f}")
interp["ig"] = dict(completeness_rel_delta=float(rel_delta), attn_ig_spearman_text=float(np.nanmedian(rho_t)),
                    attn_ig_spearman_audio=float(np.nanmedian(rho_a)))

if F_LAYERED:                                           # [L3] which encoder layers carry the audio evidence
    LA_ = np.stack([r["layer_attr"] for r in ig])
    x = np.asarray(F["members"][0].layer_ids)
    lay_abs, lay_dep = np.abs(LA_).mean(0), LA_[yte == 1].mean(0)
    plt.figure(figsize=(6, 3.2))
    plt.bar(x, lay_abs, color=np.where(lay_dep > 0, "tab:red", "tab:green"))
    plt.xlabel("Hidden state (0 = CNN features)"); plt.ylabel("Mean |IG| (log-odds)")
    plt.title("Audio evidence per encoder layer (fusion; red = raises P(dep) in depressed)")
    savefig("layer_ig.png")
    interp["ig"]["layer_mean_abs"] = {int(l): float(v) for l, v in zip(x, lay_abs)}
    interp["ig"]["layer_mean_signed_dep"] = {int(l): float(v) for l, v in zip(x, lay_dep)}
    print("IG per layer (mean |attr|):", dict(zip(x.tolist(), np.round(lay_abs, 4).tolist())))

if F_FA == "attn":                                      # [F3] do the frame weights agree with frame-level IG?
    rho_f = []
    for r, A in zip(ig, frame_attention_profile(F)):
        rho_f += [stats.spearmanr(a_, np.abs(g_)).correlation for a_, g_ in zip(A, r["frame_attr"])
                  if np.ptp(a_) > 1e-6 and np.ptp(g_) > 0]
    interp["ig"]["frame_attn_ig_spearman"] = float(np.nanmedian(rho_f)) if rho_f else float("nan")
    print(f"Frame attention vs |frame IG| (Spearman, median over chunks): {interp['ig']['frame_attn_ig_spearman']:.3f}")

if F_GM:
    A = np.stack([r["gm_attr"] for r in ig]); names = np.array(SPK_NAMES)
    imp = np.abs(A).mean(0); top = np.argsort(-imp)[:15][::-1]
    signed = A[yte == 1].mean(0)
    plt.figure(figsize=(6, 4.5))
    plt.barh(names[top], imp[top], color=np.where(signed[top] > 0, "tab:red", "tab:green"))
    plt.xlabel("Mean |IG| (log-odds)"); plt.title("Top speaker-level features (red = raises P(dep) in depressed)")
    plt.yticks(fontsize=6); savefig("egemaps_ig_importance.png")
    interp["egemaps_top_features"] = [dict(feature=str(names[i]), mean_abs_ig=float(imp[i]),
                                           mean_signed_ig_dep=float(signed[i])) for i in top[::-1]]


@torch.no_grad()
def logodds_subset(model, s, keep_s, keep_c):
    return logodds(head_forward(model, s, SEL_TEST[s["sid"]], keep_s, keep_c))[0].item()


def deletion_curve(model, modality, seed=0):
    rng, targeted, random_ = np.random.RandomState(seed), [], []
    for s, r in zip(test_s, ig):
        ns, nc = len(s["text"]), len(SEL_TEST[s["sid"]])
        all_s, all_c = np.arange(ns), np.arange(nc)
        attr, n = (r["sent_attr"], ns) if modality == "text" else (r["chunk_attr"], nc)
        base = logodds_subset(model, s, all_s, all_c)
        order = np.argsort(-attr)
        run = (lambda keep: logodds_subset(model, s, keep, all_c)) if modality == "text" else \
              (lambda keep: logodds_subset(model, s, all_s, keep))
        tg, rd = [], []
        for f in DEL_FRACS:
            k = min(int(round(f * n)), n - 1)
            tg.append(base - run(order[k:]))
            rd.append(np.mean([base - run(rng.permutation(n)[k:]) for _ in range(N_RANDOM_DEL)]))
        targeted.append(tg); random_.append(rd)
    return np.array(targeted), np.array(random_)


fig, axes = plt.subplots(1, 2, figsize=(9, 3.5)); interp["deletion"] = {}
for ax, modality in zip(axes, ["text", "audio"]):
    tg, rd = deletion_curve(M, modality)
    x = np.array(DEL_FRACS) * 100
    for arr, lab in [(tg, "IG-ranked"), (rd, "Random")]:
        mu, se = arr.mean(0), arr.std(0) / np.sqrt(len(arr))
        ax.plot(x, mu, marker="o", label=lab); ax.fill_between(x, mu - se, mu + se, alpha=0.2)
    aopc_t, aopc_r = tg[:, 1:].mean(1), rd[:, 1:].mean(1)
    try:
        p = stats.wilcoxon(aopc_t, aopc_r).pvalue
    except ValueError:
        p = np.nan
    ax.set_title(f"{modality.capitalize()} deletion (AOPC {aopc_t.mean():.2f} vs {aopc_r.mean():.2f}, p={p:.1e})",
                 fontsize=9)
    ax.set_xlabel(f"% {'sentences' if modality == 'text' else 'selected audio chunks'} removed")
    ax.set_ylabel("Drop in log-odds (depressed)"); ax.legend()
    interp["deletion"][modality] = dict(aopc_ig=float(aopc_t.mean()), aopc_random=float(aopc_r.mean()),
                                        wilcoxon_p=float(p))
savefig("deletion_curves.png")

LEX = {
    "first_person_sg": {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll"},
    "absolutist": {"always", "never", "nothing", "everything", "completely", "totally", "entire",
                   "constantly", "definitely", "all", "whole", "every", "anything", "nobody", "everyone"},
    "negative_emotion": {"sad", "tired", "depressed", "lonely", "hate", "worry", "worried", "anxious",
                         "hurt", "bad", "cry", "crying", "hopeless", "stress", "stressed", "afraid",
                         "angry", "upset", "scared", "miserable", "empty", "awful", "terrible"},
}
PHQ_DOMAINS = {
    "anhedonia": {"enjoy", "fun", "interest", "interested", "hobby", "hobbies", "pleasure", "bored", "motivation"},
    "mood": {"sad", "depressed", "down", "hopeless", "unhappy", "cry", "crying", "lonely", "upset", "empty"},
    "sleep": {"sleep", "sleeping", "insomnia", "awake", "nightmare", "nightmares", "bed", "nap"},
    "energy": {"tired", "energy", "exhausted", "fatigue", "lazy", "drained", "worn"},
    "appetite": {"eat", "eating", "appetite", "food", "weight", "hungry"},
    "self_worth": {"failure", "fail", "failed", "guilt", "guilty", "worthless", "blame", "ashamed", "regret",
                   "disappointed"},
    "concentration": {"focus", "concentrate", "concentration", "distracted", "forget", "forgetful", "memory"},
    "psychomotor": {"slow", "restless", "fidget", "agitated", "nervous"},
    "diagnosis_treatment": DIAG_WORDS,
}
lex_rate = lambda sents, lex: (lambda toks: sum(t in lex for t in toks) / max(len(toks), 1))(
    [t for s in sents for t in tokenize(s)])

lrows, drows, trows = [], [], []
for e, r in zip(test_entries, ig):
    order = np.argsort(-r["sent_attr"])
    k = min(TOPK_SENTS, max(1, len(order) // 3))
    top, rest = [e["sents"][i] for i in order[:k]], [e["sents"][i] for i in order[k:]]
    lrows.append(dict(y=e["label"], **{f"{c}_top": lex_rate(top, L) for c, L in LEX.items()},
                      **{f"{c}_rest": lex_rate(rest, L) for c, L in LEX.items()}))
    drows.append(dict(y=e["label"], **{d: np.mean([bool(set(tokenize(s)) & kw) for s in top])
                                       for d, kw in PHQ_DOMAINS.items()}))
    trows += [dict(sid=e["sid"], y=e["label"], rank=j + 1, ig=float(r["sent_attr"][i]), sentence=e["sents"][i])
              for j, i in enumerate(order[:5])]
ling = pd.DataFrame(lrows); interp["linguistic"] = {}
print("\nLinguistic markers: top-IG sentences vs rest (Wilcoxon)")
for y_, name in [(1, "depressed"), (0, "non-depressed")]:
    sub = ling[ling.y == y_]
    for c in LEX:
        try:
            p = stats.wilcoxon(sub[f"{c}_top"], sub[f"{c}_rest"]).pvalue
        except ValueError:
            p = np.nan
        print(f"  {name:13s} {c:17s} top {sub[f'{c}_top'].mean():.4f} vs rest {sub[f'{c}_rest'].mean():.4f} p={p:.4f}")
        interp["linguistic"][f"{name}_{c}"] = dict(top=float(sub[f"{c}_top"].mean()),
                                                  rest=float(sub[f"{c}_rest"].mean()), p=float(p))
pd.DataFrame(trows).to_csv(os.path.join(OUT_DIR, "PRIVATE_top_sentences.csv"), index=False)
dom = pd.DataFrame(drows).groupby("y").mean().T; dom.columns = ["Non-dep", "Dep"]
dom.plot.barh(figsize=(5, 4)); plt.xlabel("Fraction of top-IG sentences mentioning domain")
plt.title("PHQ-8 domains in most important sentences"); savefig("phq_domains_top_sentences.png")
interp["phq_domains"] = dom.round(4).to_dict()


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1)
    return out


def chunk_egemaps_df(e, idx):
    return pd.DataFrame(e["cheap"].numpy()[:, GM_SLICE][np.asarray(idx)], columns=GM_NAMES).reset_index(drop=True)


ACOUSTIC_FEATS = ["F0semitoneFrom27.5Hz_sma3nz_amean", "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
                  "loudness_sma3_amean", "jitterLocal_sma3nz_amean", "shimmerLocaldB_sma3nz_amean",
                  "HNRdBACF_sma3nz_amean", "VoicedSegmentsPerSec", "MeanUnvoicedSegmentLength", "loudnessPeaksPerSec"]
if CHEAP_EGEMAPS:
    diffs, hi_means = [], []
    for e, r in zip(test_entries, ig):
        sel = SEL_TEST[e["sid"]]
        if len(sel) < 4:
            continue
        g = chunk_egemaps_df(e, sel)
        feats = [f for f in ACOUSTIC_FEATS if f in g.columns]
        a = r["chunk_attr"]; hi, lo = a >= np.quantile(a, 0.75), a <= np.quantile(a, 0.25)
        diffs.append(dict(y=e["label"], **(g.loc[hi, feats].mean() - g.loc[lo, feats].mean()).to_dict()))
        hi_means.append(dict(y=e["label"], **g.loc[hi, feats].mean().to_dict()))
    if diffs:
        diffs, hi_means = pd.DataFrame(diffs), pd.DataFrame(hi_means)
        feats = [f for f in ACOUSTIC_FEATS if f in diffs.columns]
        arows = []
        for f in feats:
            dd = diffs.loc[diffs.y == 1, f].dropna()
            pw = stats.wilcoxon(dd).pvalue if len(dd) > 5 and (dd != 0).any() else np.nan
            h1, h0 = hi_means.loc[hi_means.y == 1, f].dropna(), hi_means.loc[hi_means.y == 0, f].dropna()
            pb = stats.mannwhitneyu(h1, h0).pvalue if len(h1) > 2 and len(h0) > 2 else np.nan
            arows.append(dict(feature=f, dep_high_minus_low=dd.mean(), p_within=pw,
                              dep_high=h1.mean(), nondep_high=h0.mean(), p_between=pb))
        ac = pd.DataFrame(arows)
        ac["q_within"], ac["q_between"] = bh_fdr(ac.p_within.fillna(1)), bh_fdr(ac.p_between.fillna(1))
        print("\nAcoustic profile of high- vs low-attribution selected chunks (BH-FDR):\n",
              ac.round(4).to_string(index=False))
        ac.to_csv(os.path.join(OUT_DIR, "acoustic_grounding.csv"), index=False)
        interp["acoustic"] = ac.round(5).to_dict(orient="records")

correct = [(i, p) for i, p in enumerate(p_test) if (p >= THR) == yte[i]]
dep_i = max([c for c in correct if yte[c[0]] == 1], key=lambda c: c[1], default=(None,))[0]
ctl_i = min([c for c in correct if yte[c[0]] == 0], key=lambda c: c[1], default=(None,))[0]
cases = [(i, lab) for i, lab in [(dep_i, "Depressed"), (ctl_i, "Non-depressed")] if i is not None]
if cases:
    fig, axes = plt.subplots(len(cases), 2, figsize=(11, 2.6 * len(cases)), squeeze=False)
    for row, (i, lab) in enumerate(cases):
        r, e = ig[i], test_entries[i]
        t_all, t_sel = e["times"] / 60, e["times"][SEL_TEST[e["sid"]]] / 60
        axes[row, 0].bar(range(len(r["sent_attr"])), r["sent_attr"],
                         color=np.where(r["sent_attr"] > 0, "tab:red", "tab:green"))
        axes[row, 0].set_title(f"{lab} speaker, P(dep)={p_test[i]:.2f}: sentence IG", fontsize=9)
        axes[row, 0].set_xlabel("Sentence index")
        axes[row, 1].scatter(t_all, np.zeros_like(t_all), marker="|", c="lightgray", label="not sent")
        axes[row, 1].bar(t_sel, r["chunk_attr"], width=CHUNK_SEC / 60,
                         color=np.where(r["chunk_attr"] > 0, "tab:red", "tab:green"))
        axes[row, 1].set_title(f"IG of the {len(t_sel)}/{len(t_all)} selected chunks", fontsize=9)
        axes[row, 1].set_xlabel("Interview time (min)")
    savefig("explanation_timelines.png")

if F["selector"] != "all" and F["frac"] < 1.0:
    pos = np.concatenate([SEL_TEST[s["sid"]] / max(1, len(s["audio"]) - 1) for s in test_s])
    ks_p = stats.kstest(pos, "uniform").pvalue
    jac = []
    for s in test_s:
        a = set(SEL_TEST[s["sid"]].tolist()); k = len(a); T_ = len(s["audio"])
        if k >= T_:
            continue
        b = set(np.argsort(-s["energy"], kind="stable")[:k].tolist())
        c = set(np.random.RandomState(0).choice(T_, k, replace=False).tolist())
        jac.append(dict(vs_energy=len(a & b) / len(a | b), vs_random=len(a & c) / len(a | c)))
    jac = pd.DataFrame(jac)
    print(f"\nSelection position vs uniform: KS p={ks_p:.3g} | Jaccard with loudest-K {jac.vs_energy.mean():.3f}, "
          f"with random-K {jac.vs_random.mean():.3f}")
    interp["selection"] = dict(position_ks_p=float(ks_p), jaccard_vs_energy=float(jac.vs_energy.mean()),
                               jaccard_vs_random=float(jac.vs_random.mean()))
    plt.figure(figsize=(4.5, 3))
    plt.hist(pos, bins=10, range=(0, 1), density=True, color="tab:blue", alpha=0.8)
    plt.axhline(1.0, c="k", ls=":", lw=0.8)
    plt.xlabel("Relative position in interview"); plt.ylabel("Density of kept chunks")
    plt.title("Where the selector keeps audio"); savefig("selection_position.png")
    if CHEAP_EGEMAPS:
        srows = []
        for e in test_entries:
            sel = SEL_TEST[e["sid"]]; T_ = len(e["times"])
            if len(sel) >= T_ or len(sel) < 2:
                continue
            g = chunk_egemaps_df(e, np.arange(T_)); mask = np.zeros(T_, bool); mask[sel] = True
            feats = [f for f in ACOUSTIC_FEATS if f in g.columns]
            srows.append(dict(y=e["label"], **(g.loc[mask, feats].mean() - g.loc[~mask, feats].mean()).to_dict()))
        if srows:
            srows = pd.DataFrame(srows); feats = [f for f in ACOUSTIC_FEATS if f in srows.columns]
            prof = []
            for f in feats:
                d = srows[f].dropna()
                p = stats.wilcoxon(d).pvalue if len(d) > 5 and (d != 0).any() else np.nan
                prof.append(dict(feature=f, kept_minus_dropped=d.mean(), p=p))
            prof = pd.DataFrame(prof); prof["q"] = bh_fdr(prof.p.fillna(1))
            print("Acoustic profile: kept vs dropped chunks (BH-FDR):\n", prof.round(4).to_string(index=False))
            prof.to_csv(os.path.join(OUT_DIR, "selection_acoustic_profile.csv"), index=False)
            interp["selection"]["acoustic_profile"] = prof.round(5).to_dict(orient="records")

with open(os.path.join(OUT_DIR, "interpretability_summary.json"), "w") as f:
    json.dump(interp, f, indent=2, default=float)
print("\nInterpretability results saved to", OUT_DIR)

# ============================================================
# SAVE + ONNX EXPORT (on-device pipeline for the FINAL system)
# ============================================================
#   final = "fusion": fusion_head.onnx (text windows, K audio chunks[, egemaps]) -> prob, logodds, attention
#   final = "late"  : text_head.onnx (text windows) -> logodds ; audio_head.onnx (K chunks[, egemaps]) -> logodds
#                     P = sigmoid(w_t * clip(lo_text) + w_a * clip(lo_audio) + b)   (weights in deploy_config.json)
#   always          : text_encoder.onnx, audio_encoder.onnx (only K chunks), and the selector
#                     (text_selector.onnx for "text", chunk_scorer.onnx for "learned")
#   [L4] with layer attention the audio encoder outputs (n_chunks, N_LAYERS, 768) and the heads take
#        audio of shape (1, K, N_LAYERS, 768); the learned layer mixing runs inside the head.
import onnxruntime as ort

if FINAL == "late":
    A_ENS = ens[LATE_AUDIO]
    DEP = dict(selector=A_ENS["selector"], frac=A_ENS["frac"], members=A_ENS["members"], gm=A_ENS["gm"],
               la=bool(A_ENS.get("la")), fa=A_ENS.get("fa", "off"))
else:
    DEP = dict(selector=F["selector"], frac=F["frac"], members=F["members"], gm=F_GM, la=F_LAYERED, fa=F_FA)
DEP_GM, DEP_LAYERED, DEP_FA = bool(DEP["gm"]), bool(DEP["la"]), DEP["fa"]

# [R4] With LoRA every fold has its own adapted encoder. The device runs ONE: the fold with the best Stage-A
# early-stopping AUC (merged weights), together with the heads of that fold. Its test AUC is reported separately.
DEP_LORA_FOLD = None
if any(getattr(m, "lora_fold", None) is not None for m in DEP["members"]):
    _kept = [f for f in LORA_INFO if LORA_INFO[f].get("lora_used", True)]             # [C3]
    DEP_LORA_FOLD = max(_kept or list(LORA_INFO), key=lambda f: LORA_INFO[f]["es_auc"])
    if not _kept:
        print("[C3] NOTE: no fold kept its adapters -> the deployed audio encoder is the frozen HuBERT")
    DEP["members"] = [m for m in DEP["members"] if m.lora_fold == DEP_LORA_FOLD]
DEP_ENS = Ensemble(DEP["members"]).eval()
_lo_dep = predict_logodds(DEP_ENS, test_s)
if FINAL == "late":
    _clip = float(logit(1 - LOGIT_CLIP))
    _x = np.column_stack([np.clip(_lo(ens["text"]["test_p"]), -_clip, _clip), np.clip(_lo_dep, -_clip, _clip)])
    p_dep = sigmoid(_x @ np.asarray(ens["late"]["coef"]) + ens["late"]["intercept"])
else:
    p_dep = sigmoid(_lo_dep)
print(f"\n[R4] Deployed system ({FINAL}, {len(DEP['members'])} audio heads"
      f"{f', single LoRA encoder of fold {DEP_LORA_FOLD}' if DEP_LORA_FOLD is not None else ''}): "
      f"test AUC {roc_auc_score(yte, p_dep):.3f} (full {N_FOLDS}-fold ensemble: {roc_auc_score(yte, p_fin):.3f})")

bundle = dict(final=FINAL, threshold=THR_FIN, wavlm_layers=WAVLM_LAYERS, text_model=TEXT_MODEL, gm_names=SPK_NAMES,
              selector=DEP["selector"], keep_frac=DEP["frac"], k_min=K_MIN, cheap_names=CHEAP_NAMES,
              chunk_sec=CHUNK_SEC, hyperparameters=HP, layer_attention=dict(LAYER_CFG, used=DEP_LAYERED, frames_used=DEP_FA),
              fusion_states=[m.state_dict() for m in F["members"]],
              lora=(dict(fold=DEP_LORA_FOLD, deltas=LORA_INFO[DEP_LORA_FOLD]["deltas"], config=LORA_CFG)
                    if DEP_LORA_FOLD is not None else None))
if FINAL == "late":
    bundle.update(text_states=[m.state_dict() for m in ens["text"]["members"]],
                  audio_states=[m.state_dict() for m in DEP["members"]],
                  late=dict(audio_model=LATE_AUDIO, coef=ens["late"]["coef"], intercept=ens["late"]["intercept"],
                            logit_clip=LOGIT_CLIP))
torch.save(bundle, os.path.join(OUT_DIR, "final_system.pt"))


class DeployHead(nn.Module):
    def __init__(self, ens_model):
        super().__init__(); self.ens = ens_model

    def forward(self, text, audio, egemaps=None):
        logits, att = self.ens(text, audio, egemaps, return_attn=True)
        lo = logits[:, 1] - logits[:, 0]
        return torch.sigmoid(lo), lo, att["w_sent"], att["w_win"], att["w_audio"]


class TextHead(nn.Module):
    def __init__(self, ens_model):
        super().__init__(); self.ens = ens_model

    def forward(self, text):
        logits, att = self.ens(text, None, None, return_attn=True)
        return logodds(logits), att["w_sent"], att["w_win"]


class AudioHead(nn.Module):
    def __init__(self, ens_model):
        super().__init__(); self.ens = ens_model

    def forward(self, audio, egemaps=None):
        logits, att = self.ens(None, audio, egemaps, return_attn=True)
        return logodds(logits), att["w_audio"]


class EnsembleScorer(nn.Module):
    def __init__(self, scorers):
        super().__init__(); self.scorers = nn.ModuleList(scorers)

    def forward(self, x):
        return zscore(torch.stack([sc(x) for sc in self.scorers])).mean(0)


def export_onnx(module, args, path, input_names, output_names, dynamic_axes):
    kw = dict(input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
              opset_version=OPSET, do_constant_folding=True)
    try:
        torch.onnx.export(module, args, path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(module, args, path, **kw)
    print("Exported", path)


def verify_onnx(path, module, args, names, atol=1e-3):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = module(*args)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    out = sess.run(None, {n: a.numpy() for n, a in zip(names, args)})
    err = max(float(np.abs(o - r.numpy()).max()) for o, r in zip(out, ref))
    print(f"  parity {os.path.basename(path)}: max |onnx - torch| = {err:.2e} {'OK' if err < atol else 'CHECK'}")
    return sess


rand_text = lambda w, l: torch.randn(w, l, D_TEXT) * 0.05


def rand_audio(t, layered, fa="off"):                 # [L4] / [F4] example audio input of the right shape
    if fa != "off":
        return torch.randn(1, t, N_FRAMES, N_FLAYERS, 768)
    return torch.randn(*((1, t, N_LAYERS, 768) if layered else (1, t, 768)))


sess_head = sess_thead = sess_ahead = None
if FINAL == "fusion":
    head = DeployHead(copy.deepcopy(DEP_ENS).cpu().eval()).eval()
    head_inputs = ["text", "audio"] + (["egemaps"] if F_GM else [])
    mk_args = lambda w, l, t: tuple([rand_text(w, l), rand_audio(t, F_LAYERED, F_FA)] +
                                    ([GM_BASE.cpu().clone()] if F_GM else []))
    p_head = os.path.join(ONNX_DIR, "fusion_head.onnx")
    export_onnx(head, mk_args(3, SEQ_LEN, 40), p_head, head_inputs,
                ["prob_depressed", "logodds", "w_sent", "w_win", "w_audio"],
                {"text": {0: "n_windows", 1: "win_len"}, "audio": {1: "n_chunks"},
                 "w_sent": {0: "n_windows", 1: "win_len"}, "w_win": {0: "n_windows"}, "w_audio": {0: "n_chunks"}})
    sess_head = verify_onnx(p_head, head, mk_args(2, 37, 13), head_inputs)
else:
    thead = TextHead(copy.deepcopy(Ensemble(ens["text"]["members"])).cpu().eval()).eval()
    p_th = os.path.join(ONNX_DIR, "text_head.onnx")
    export_onnx(thead, (rand_text(3, SEQ_LEN),), p_th, ["text"], ["logodds", "w_sent", "w_win"],
                {"text": {0: "n_windows", 1: "win_len"}, "w_sent": {0: "n_windows", 1: "win_len"},
                 "w_win": {0: "n_windows"}})
    sess_thead = verify_onnx(p_th, thead, (rand_text(2, 37),), ["text"])
    ahead = AudioHead(copy.deepcopy(DEP_ENS).cpu().eval()).eval()
    a_inputs = ["audio"] + (["egemaps"] if DEP_GM else [])
    a_args = lambda t: tuple([rand_audio(t, DEP_LAYERED, DEP_FA)] + ([GM_BASE.cpu().clone()] if DEP_GM else []))
    p_ah = os.path.join(ONNX_DIR, "audio_head.onnx")
    export_onnx(ahead, a_args(40), p_ah, a_inputs, ["logodds", "w_audio"],
                {"audio": {1: "n_chunks"}, "w_audio": {0: "n_chunks"}})
    sess_ahead = verify_onnx(p_ah, ahead, a_args(13), a_inputs)

sess_scorer = None
if DEP["selector"] in LEARNED:
    scorer_cpu = EnsembleScorer([copy.deepcopy(m.scorer).cpu().eval() for m in DEP["members"]]).eval()
    d_in = D_CHEAP if DEP["selector"] == "learned" else 768
    p_scorer = os.path.join(ONNX_DIR, "chunk_scorer.onnx")
    export_onnx(scorer_cpu, (torch.randn(30, d_in),), p_scorer, ["chunk_feats"], ["chunk_scores"],
                {"chunk_feats": {0: "n_chunks"}, "chunk_scores": {0: "n_chunks"}})
    ref_x = dev_s[0]["cheap" if DEP["selector"] == "learned" else "audio"].float().cpu()
    sess_scorer = verify_onnx(p_scorer, scorer_cpu, (ref_x,), ["chunk_feats"])

sess_tsel = None
if DEP["selector"] == "text":
    tsel_cpu = TextSel([copy.deepcopy(m).cpu().eval() for m in ens["text"]["members"]]).eval()
    p_tsel = os.path.join(ONNX_DIR, "text_selector.onnx")
    export_onnx(tsel_cpu, (rand_text(3, SEQ_LEN),), p_tsel, ["text"], ["w_sent", "w_win"],
                {"text": {0: "n_windows", 1: "win_len"}, "w_sent": {0: "n_windows", 1: "win_len"},
                 "w_win": {0: "n_windows"}})
    sess_tsel = verify_onnx(p_tsel, tsel_cpu, (rand_text(2, 37),), ["text"])

te_cpu = copy.deepcopy(text_encoder).cpu().eval()
p_text = os.path.join(ONNX_DIR, "text_encoder.onnx")
tok = tokenizer(["i have been feeling tired lately", "no"], return_tensors="pt", padding=True)
export_onnx(te_cpu, (tok["input_ids"], tok["attention_mask"]), p_text, ["input_ids", "attention_mask"], ["sent_emb"],
            {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "sent_emb": {0: "batch"}})
tok2 = tokenizer(["a longer sentence to check that dynamic axes work properly", "ok", "not really"],
                 return_tensors="pt", padding=True)
sess_text = verify_onnx(p_text, te_cpu, (tok2["input_ids"], tok2["attention_mask"]), ["input_ids", "attention_mask"])

if DEP_LORA_FOLD is not None:                         # [R4] HuBERT with the LoRA adapters merged into its weights
    _hub = apply_lora_deltas(copy.deepcopy(audio_encoder.wavlm).cpu(), LORA_INFO[DEP_LORA_FOLD]["deltas"])
    ae_cpu = AudioEncoder(_hub, WAVLM_NORMALIZE, *WAVLM_LAYERS).eval()
    ae_cpu.la0, ae_cpu.la1 = LA0, LA1
else:
    ae_cpu = copy.deepcopy(audio_encoder).cpu().eval()
ae_cpu.all_layers = DEP_LAYERED                       # [L4] (batch, N_LAYERS, 768) when the head mixes layers
if DEP_FA != "off":                                   # [F4] (batch, N_FRAMES, N_FLAYERS, 768) for frame attention
    ae_cpu.frame_ids, ae_cpu.n_frames = FRAME_IDS, N_FRAMES
p_audio = os.path.join(ONNX_DIR, "audio_encoder.onnx")
sess_audio = None
try:
    export_onnx(ae_cpu, (torch.randn(2, CHUNK_SIZE) * 0.1,), p_audio, ["input_values"], ["chunk_emb"],
                {"input_values": {0: "batch"}, "chunk_emb": {0: "batch"}})
    sess_audio = verify_onnx(p_audio, ae_cpu, (torch.randn(3, CHUNK_SIZE) * 0.1,), ["input_values"])
except Exception as ex:
    print("Audio encoder export failed (try another OPSET / transformers version):", repr(ex))

tokenizer.save_pretrained(os.path.join(ONNX_DIR, "tokenizer"))
gm_shape = f",egemaps:1x{N_GM}" if DEP_GM else ""
a_dim = (f"x{N_FRAMES}x{N_FLAYERS}x768" if DEP_FA != "off" else f"x{N_LAYERS}x768" if DEP_LAYERED else "x768")
deploy_cfg = dict(
    final_system=FINAL, threshold=THR_FIN, sample_rate=SR,
    lora=(dict(fold=DEP_LORA_FOLD, merged_into_audio_encoder=True, r=LORA_R, alpha=LORA_ALPHA,
               targets=LORA_TARGETS, layers=LORA_LAYERS) if DEP_LORA_FOLD is not None else None),
    chunk_samples=CHUNK_SIZE,
    max_audio_chunks=MAX_AUDIO_CHUNKS, use_participant_segments=USE_PARTICIPANT_SEGMENTS,
    min_words=MIN_WORDS, max_sents=MAX_SENTS, max_token_len=128, seq_len=SEQ_LEN, stride=STRIDE,
    n_views=N_VIEWS, tta_frac=TTA_FRAC, view_seed="crc32(speaker_id)",
    tta_scope=("text head: views over (n_sents, all chunks); audio head: views over (n_sents, K selected chunks)"
               if FINAL == "late" else "views over (n_sents, K selected chunks)"),
    text_model=TEXT_MODEL, text_pooling=TEXT_POOLING, wavlm=WAVLM_NAME,
    audio_features=(dict(type="frame-level features; layer attention per frame + frame attention per chunk "
                              "inside the head", hidden_states=FRAME_IDS, frame_pool=FRAME_POOL,
                         frame_ms=FRAME_POOL * 20, n_frames=N_FRAMES, frame_mode=DEP_FA, layer_attention=DEP_LAYERED,
                         encoder_output="(n_chunks, %d, %d, 768): mean of %d consecutive 20-ms frames"
                                        % (N_FRAMES, N_FLAYERS, FRAME_POOL))
                    if DEP_FA != "off" else
                    dict(type="all layers, learnable layer attention inside the head",
                         hidden_states=[LA0, LA1], kind=LAYER_ATTN_KIND,
                         encoder_output="(n_chunks, %d, 768): time-mean of each hidden state" % N_LAYERS)
                    if DEP_LAYERED else
                    dict(type="fixed layer mean", wavlm_layers=list(WAVLM_LAYERS),
                         encoder_output="(n_chunks, 768)")),
    late_fusion=(dict(coef_text_audio=ens["late"]["coef"], intercept=ens["late"]["intercept"],
                      input="mean TTA log-odds of each head, clipped to +-logit(1 - %g)" % LOGIT_CLIP)
                 if FINAL == "late" else None),
    egemaps=dict(enabled=DEP_GM, feature_set=("eGeMAPSv02 Functionals + " if USE_EGEMAPS else "") + "timing [T1]",
                 computed_on="participant signal + transcript timestamps", feature_order=SPK_NAMES,
                 timing_fill_values=TIMING_FILL.tolist(), pause_db=PAUSE_DB, min_pause_s=MIN_PAUSE_S,
                 timing_condition_norm=({c: dict(center=v["center"].tolist(), scale=v["scale"].tolist())
                                         for c, v in TIMING_NORM.items()} if TIMING_NORM else None),
                 condition_rule="AI agent if participant id >= 600 else Wizard-of-Oz; deployment = AI"),
    selection=dict(
        selector=DEP["selector"], keep_frac=DEP["frac"], k_min=K_MIN,
        k_rule="K = T if keep_frac >= 1 else min(T, max(k_min, ceil(keep_frac * T)))",
        order="descending score (stable sort); kept chunk indices re-sorted in time",
        scorer_onnx=("chunk_scorer.onnx" if sess_scorer is not None else
                     "text_selector.onnx" if sess_tsel is not None else None),
        learned_wavlm_scorer_input=f"mean of hidden states {WAVLM_LAYERS[0]}..{WAVLM_LAYERS[1] - 1}",
        text_guided=("chunk score = sum_j overlap_sec(chunk, sentence j) / dur_j * importance_j, "
                     "importance_j = sum over windows of w_win * w_sent from text_selector.onnx; "
                     "ties broken by chunk energy") if DEP["selector"] == "text" else None,
        cheap_features=dict(log_mel=dict(n_mels=N_MELS, n_fft=MEL_NFFT, hop=MEL_HOP,
                                         power_to_db="ref=1.0, top_db=None", stats=["mean", "std"]),
                            egemaps_per_chunk=bool(CHEAP_EGEMAPS), order=CHEAP_NAMES),
        energy_db="10*log10(mean(x^2) + 1e-10) per chunk", random_seed="(crc32(speaker_id) + 1) % 2**31"),
    opset=OPSET,
    tensorrt=dict(
        text_encoder=("trtexec --onnx=text_encoder.onnx --saveEngine=text_encoder_fp16.engine --fp16 "
                      "--minShapes=input_ids:1x4,attention_mask:1x4 --optShapes=input_ids:32x32,attention_mask:32x32 "
                      "--maxShapes=input_ids:32x128,attention_mask:32x128"),
        audio_encoder=(f"trtexec --onnx=audio_encoder.onnx --saveEngine=audio_encoder_fp16.engine --fp16 "
                       f"--minShapes=input_values:1x{CHUNK_SIZE} --optShapes=input_values:8x{CHUNK_SIZE} "
                       f"--maxShapes=input_values:{AUDIO_BATCH}x{CHUNK_SIZE}"),
        audio_shapes=(f"audio:1x1{a_dim}{gm_shape} .. 1x{n_keep(60, DEP['frac'])}{a_dim} .. "
                      f"1x{MAX_AUDIO_CHUNKS}{a_dim}")))
with open(os.path.join(ONNX_DIR, "deploy_config.json"), "w") as f:
    json.dump(deploy_cfg, f, indent=2)


def onnx_wavlm(ch):
    return np.concatenate([sess_audio.run(None, {"input_values": np.ascontiguousarray(ch[i:i + AUDIO_BATCH])})[0]
                           for i in range(0, len(ch), AUDIO_BATCH)])


def probe_window(emb):
    """learned_wavlm scorer input: fixed probe-window mean, also when the encoder returns all layers or frames.
    (From frames it is the mean over the pooled frames, a close approximation of the cached time-mean.)"""
    if emb.ndim == 2:
        return emb
    if emb.ndim == 4:                                  # [F4] (K, F, L_f, 768)
        emb, ids = emb.mean(1), FRAME_IDS
    else:
        ids = list(range(LA0, LA1))
    cols = [i for i, l in enumerate(ids) if WAVLM_LAYERS[0] <= l < WAVLM_LAYERS[1]]
    return emb[:, cols].mean(1) if cols else emb.mean(1)


def onnx_select(sid, cheap, energy, chunks, select=True, text=None, tcsv=None, segs=None, times=None):
    """Device-side selection, identical to pick(). Returns (indices, embeddings of all chunks or None)."""
    n = len(cheap); k = n_keep(n, DEP["frac"]); selr = DEP["selector"]
    if not select or selr == "all" or k >= n:
        return np.arange(n), None
    if selr == "first":
        return np.arange(k), None
    if selr == "random":
        return np.sort(_sid_rng(sid).choice(np.arange(n), k, replace=False)), None
    if selr == "energy":
        return np.sort(np.argsort(-energy, kind="stable")[:k]), None
    if selr == "text":
        st_ = sentence_times(tcsv)
        if st_ is None or len(st_) != len(text):
            sc = np.zeros(n)
        else:
            idx = window_index(len(text))
            ws, ww = sess_tsel.run(None, {"text": text[idx].astype(np.float32)})
            sc = overlap_matrix(chunk_intervals(segs, times), st_) @ sentence_importance(ws, ww, len(text))
        e = np.asarray(energy, float)
        return np.sort(np.argsort(-(sc + 1e-9 * (e - e.min())), kind="stable")[:k]), None
    emb_all = None
    if selr == "learned":
        sc = sess_scorer.run(None, {"chunk_feats": cheap.astype(np.float32)})[0]
    else:
        emb_all = onnx_wavlm(chunks)
        sc = sess_scorer.run(None, {"chunk_feats": probe_window(emb_all).astype(np.float32)})[0]
    return np.sort(np.argsort(-sc, kind="stable")[:k]), emb_all


def onnx_predict(transcript_csv, wav_path, sid, select=True):
    """Full on-device pipeline for the FINAL system, with per-stage timing."""
    tm, clock = {}, time.perf_counter
    t0 = clock(); sents, segs = load_transcript(transcript_csv)
    chunks, times, y_use = load_participant_audio(wav_path, segs); tm["load_audio"] = clock() - t0
    t0 = clock(); cheap, energy = chunk_features(chunks); tm["cheap_features"] = clock() - t0
    t0 = clock(); embs = []
    for i in range(0, len(sents), EMB_BATCH):
        t = tokenizer(sents[i:i + EMB_BATCH], return_tensors="np", truncation=True, padding=True, max_length=128)
        embs.append(sess_text.run(None, {"input_ids": t["input_ids"].astype(np.int64),
                                         "attention_mask": t["attention_mask"].astype(np.int64)})[0])
    text = np.concatenate(embs); tm["text_encoder"] = clock() - t0
    t0 = clock(); sel, emb_all = onnx_select(sid, cheap, energy, chunks, select, text, transcript_csv, segs, times)
    tm["select"] = clock() - t0
    # audio: (K, 768) | layer attention (K, N_LAYERS, 768) | frame attention (K, N_FRAMES, N_FLAYERS, 768)
    t0 = clock(); audio = emb_all[sel] if emb_all is not None else onnx_wavlm(chunks[sel]); tm["wavlm"] = clock() - t0
    t0 = clock(); gm = speaker_features(transcript_csv, y_use, sid)[None] if DEP_GM else None  # [T2] [V2]
    tm["speaker_features"] = clock() - t0
    t0 = clock()
    if FINAL == "fusion":
        los = []
        for ks, kc in eval_views(sid, len(text), len(sel)):
            t_v = text[ks]; idx = window_index(len(t_v))
            feeds = {"text": t_v[idx].astype(np.float32), "audio": audio[kc][None].astype(np.float32)}
            if DEP_GM:
                feeds["egemaps"] = gm
            los.append(float(sess_head.run(None, feeds)[1][0]))
        lo = np.mean(los)
    else:
        lt = []
        for ks, _ in eval_views(sid, len(text), len(chunks)):        # text model: views over ALL chunks
            t_v = text[ks]; idx = window_index(len(t_v))
            lt.append(float(sess_thead.run(None, {"text": t_v[idx].astype(np.float32)})[0][0]))
        la = []
        for _, kc in eval_views(sid, len(text), len(sel)):            # audio model: views over K chunks
            feeds = {"audio": audio[kc][None].astype(np.float32)}
            if DEP_GM:
                feeds["egemaps"] = gm
            la.append(float(sess_ahead.run(None, feeds)[0][0]))
        clip = float(logit(1 - LOGIT_CLIP))
        x = np.clip([np.mean(lt), np.mean(la)], -clip, clip)
        lo = float(np.dot(ens["late"]["coef"], x) + ens["late"]["intercept"])
    tm["heads"] = clock() - t0
    p = float(sigmoid(lo))
    emb_kb = len(sel) * (N_FRAMES * N_FLAYERS if DEP_FA != "off" else N_LAYERS if DEP_LAYERED else 1) * 768 * 2 / 1024
    return dict(prob=p, depressed=p >= THR_FIN, n_chunks=len(chunks), n_kept=len(sel),
                uplink_raw_KB=len(sel) * CHUNK_SIZE * 2 / 1024, uplink_emb_KB=emb_kb, timings=tm)


ready = (sess_audio is not None and (DEP["selector"] not in LEARNED or sess_scorer is not None)
         and (DEP["selector"] != "text" or sess_tsel is not None))
if ready:
    lat = []
    for i in range(min(3, len(test_entries))):
        e = test_entries[i]
        if not (os.path.exists(e["wav"]) and os.path.exists(e["tcsv"])):
            print("  skipping E2E demo for", e["sid"], "(raw files not attached)")
            continue
        r_sel = onnx_predict(e["tcsv"], e["wav"], e["sid"], select=True)
        r_all = onnx_predict(e["tcsv"], e["wav"], e["sid"], select=False)
        print(f"  E2E parity speaker {e['sid']}: torch P={p_dep[i]:.4f} | onnx P={r_sel['prob']:.4f} | "
              f"kept {r_sel['n_kept']}/{r_sel['n_chunks']} chunks")
        for tag, r in [("top-k", r_sel), ("all", r_all)]:
            lat.append(dict(sid=e["sid"], pipeline=tag, n_chunks_encoded=r["n_kept"], **r["timings"],
                            total_s=sum(r["timings"].values())))
    if lat:
        lat = pd.DataFrame(lat)
        lat_mean = lat.drop(columns="sid").groupby("pipeline").mean()
        print("\nHost CPU (ONNX Runtime) latency per session, seconds — rerun on the Jetson for the paper:\n",
              lat_mean.round(3).to_string())
        lat.to_csv(os.path.join(OUT_DIR, "edge_latency_host.csv"), index=False)

print("\nDone. Results:", OUT_DIR, "| Deployment bundle:", ONNX_DIR)
