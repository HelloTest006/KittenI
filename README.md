# KittenI — CPU-Specific Ternary TTS Inference Runtime

**KittenI** is a standalone, zero-LibTorch, zero-Python C++20 CPU inference runtime engineered specifically for the **KittenML Ternary TTS** architecture (`type: "KITTEN2"`, including **KittenTTS 2** 1.7B and upcoming 80M, 15M, and 7M ternary checkpoints).

Unlike `kitten-tts-2-cpp`, which requires a ~500 MB LibTorch installation at runtime for waveform decoding, requires offline Python preparation for voice cloning, and dequantizes the `student_w4`/`w8` flows to FP32, **KittenI** executes the **entire end-to-end TTS and zero-shot voice cloning pipeline natively in C++20** with runtime-dispatched **AVX-512 VNNI**, **AVX-VNNI**, and **AVX2** SIMD kernels.

---

## Key Features

- **Zero External ML Dependencies**: Pure C++20 implementation of the Qwen3 Ternary Speech LM, S3 Flow (`student_w4` / `student_w8`), and `HiFTGenerator` 24 kHz Neural Source-Filter ISTFT Vocoder. No LibTorch or ONNX Runtime required.
- **Direct Safetensors & GGUF Loading (`mmap`)**:
  - Reads HuggingFace `lm/model-tl2-emb4.safetensors` (`TL2` 928-byte blocks + 4-bit Hadamard-rotated tiered embeddings), `lm/model-ternary.safetensors` (1.58-bit 5-trits/byte), and `TQ2_1` directly without offline conversion scripts.
  - **Dual-Space Fast Walsh-Hadamard Transform (`FWHT`) `emb4` Engine**: Keeps the 158,241 × 2048 embedding table compressed in RAM (~170 MiB vs ~650 MiB) and computes tied output logits directly in the Hadamard domain.
  - **399.0 MiB Packed Ternary RAM** for all 28 layers of the 1.7B language model (loaded in **1.6s**).
- **Runtime Multi-ISA Dispatch (`--isa auto|scalar|avx2|avx_vnni|avx512_vnni`)**:
  - Automatically detects CPU features via `cpuid` and dispatches per-function target-compiled kernels (`_mm512_dpbusd_epi32`, `_mm256_dpbusd_epi32`, `_mm256_maddubs_epi16`) while preserving compatibility across x86-64 CPUs.
- **True Low-Bit `BitLinear` & `BitConv1d` Vocoder Kernels**:
  - Executes the quantized `student_w4` and `student_w8` S3 flows using native `INT8 × INT4/INT8` VNNI instructions rather than dequantizing to FP32.
- **Runtime Zero-Shot WAV Voice Cloning**:
  - Clone any speaker directly from a 5–30s `.wav` file at runtime using native C++20 implementations of:
    1. **`SincNetSpeakerEncoder`** (`XVectorSincNet` parameterized sinc filterbank + TDNN + stats pooling $\to$ 512-d unit-norm speaker embedding).
    2. **`CampPlusEncoder`** (`CAMPPlus` 2D ResNet FCM + 52-layer `CAMDenseTDNNBlock` $\to$ 192-d x-vector).
    3. **`S3AudioTokenizer`** (`S3TokenizerV2` `AudioEncoderV2` + `FSMNMultiHeadAttention` + $3^8 = 6561$ `FSQCodebook` $\to$ 25 Hz codec tokens).
- **Pipelined Producer-Consumer Streaming**:
  - On multi-sentence inputs, the Speech LM generates tokens for chunk $N+1$ concurrently while the background Vocoder worker synthesizes 24 kHz audio for chunk $N$.
- **Multiple Interfaces**:
  - **CLI Binary (`kitteni`)**
  - **C++20 Library (`include/kitteni/kitteni.hpp` — `kitteni::Engine`)**
  - **C ABI Shared/Static Library (`include/kitteni/kitteni.h` — `libkitteni.dll` / `libkitteni.so`)**
  - **Python Package (`python/` — drop-in `from kitteni import KittenTTS`)**

---

## Benchmarks (Intel Core i3-1005G1 — Ice Lake 2C/4T Mobile CPU @ 1.20–3.40 GHz)

All benchmarks below were measured on an **Intel Core i3-1005G1** (Ice Lake 10nm, 2 physical cores / 4 logical threads, supporting **AVX2** and **AVX-512F / AVX-512BW / AVX-512VL / AVX-512 VNNI**; note that VEX-encoded `AVX-VNNI` was introduced later in Alder Lake and is supported by the runtime on 12th-gen+ CPUs, so it is excluded from the Ice Lake table below).

### 1. Kernel Microbenchmarks (Intel Core i3-1005G1, 4 Threads)

| Kernel Workload | `scalar` | `avx2` | `avx512_vnni` | Speedup vs Scalar |
|---|---:|---:|---:|---:|
| **`12288 × 2048` Fused Gate+Up Ternary GEMV** | 992.7 µs (50.7 GOPS) | **528.0 µs (95.3 GOPS)** | 539.8 µs (93.3 GOPS) | **1.88×** |
| **`256 × 1472 × 368` L2-Resident `BitLinear` GEMM** | 5412.0 µs (51.3 GOPS) | 2515.4 µs (110.3 GOPS) | **1530.6 µs (181.2 GOPS)** | **3.54×** |

### 2. Full 28-Layer `KittenTTS 2` (1.7B `emb4`) Speech LM & Vocoder Benchmarks (Intel Core i3-1005G1, 4 Threads)

| Stage / Metric | `scalar` | `avx2` | `avx512_vnni` (Hybrid) |
|---|---:|---:|---:|
| **Model Load Time (28 Layers `TL2` + `emb4`)** | 1.85 s | 1.75 s | **1.65 s** |
| **Packed Ternary LM Body RAM** | 399.0 MiB | 399.0 MiB | **399.0 MiB** |
| **28-Layer Prompt Prefill Latency** | 5.20 s | 2.12 s | **0.34 s (6.2× faster vs AVX2)** |
| **28-Layer Autoregressive Decode Throughput** | 4.8 tok/s | 11.9 tok/s | **16.2 tok/s** |
| **Native `student_w4` Vocoder (4.76s Audio / 116 Tokens)** | 14.45 s (RTF 3.04) | 10.91 s (RTF 2.29) | **8.56 s (RTF 1.79)** |

### 3. Numerical & Bitwise Parity Verification

| Component | Reference Target | Measured Accuracy |
|---|---|---|
| **`TL2` & `1.58-bit` Ternary Unpacker** | `KittenTTS` Python `unpack_tl2` / `ternary.py` | **0 bit errors** (exact bit-for-bit match) |
| **Qwen3 Byte-Level BPE Tokenizer** | HuggingFace `tokenizers` (`lm/tokenizer.json`) | **100% exact token ID match** |
| **2048-d Fast Walsh-Hadamard Transform (`FWHT`)** | Exact orthogonal involution | **`4.77e-7` max error** |
| **16-Point HiFTNet `STFT -> ISTFT` Roundtrip** | `torch.stft` / `torch.istft` (`n_fft=16, hop=4`) | **`2.38e-7` max error** |
| **24 kHz 80-Bin Mel Extractor (`extract_mel80_24k`)** | Bruno `prompt_feat` (`voices.json`) | **`7.70e-7` mean abs error** |
| **Native `CAMPPlus` 192-d X-Vector Encoder** | Bruno `embedding` (`voices.json`) | **`0.999954` cosine similarity** |
| **Native `XVectorSincNet` 512-d Speaker Encoder** | Bruno `embedding` (`voices/*.npz`) | **`0.992367` cosine similarity** |
| **Native `S3TokenizerV2` 25 Hz Speech Tokenizer** | Bruno `prompt_token` (8-trit FSQ boundaries) | **`96/100` (`96%`) exact discrete token ID match** |

---

## Build Instructions

Requires a C++20 compiler (`GCC >= 12`, `Clang >= 15`, or `MSVC 2022`) and `CMake >= 3.20`. Zero external libraries are required.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
```

Run the 5 verification test suites:

```bash
ctest --test-dir build --output-on-failure
```

---

## Usage

### 1. Command-Line Interface (`kitteni`)

```bash
# Synthesize speech with a built-in voice (automatically downloads KittenML/kitten-tts-2 emb4 on first run)
./build/kitteni \
  --text "[joyful] We won the grant <laugh> I can (((hardly))) believe it!" \
  --voice Kiki --preset expressive \
  --output expressive.wav

# Stream multi-sentence text using the pipelined LM + Vocoder engine
./build/kitteni \
  --text "The house at the end of Willow Lane had been empty for eleven years. Every evening at seven o'clock, a light came on in the attic window." \
  --voice Bruno --stream --output story.wav

# Zero-shot runtime voice cloning directly from a 5-30s WAV file
./build/kitteni \
  --reference my_voice.wav \
  --reference-text "The words spoken in my reference recording." \
  --save-voice my_voice.json \
  --text "Hello! This is my cloned voice running natively in C++20 on the CPU." \
  --output cloned.wav

# Run multi-ISA benchmarks (scalar vs avx2 vs avx_vnni vs avx512_vnni)
./build/kitteni --max-layers 2 --bench
```

### 2. Python API (`kitteni.KittenTTS`)

```bash
pip install -e ./python
```

```python
from kitteni import KittenTTS

model = KittenTTS("KittenML/kitten-tts-2", weights="emb4", decoder="student_w4", threads=4)

# Built-in voice synthesis
audio = model.generate("One day, a little girl named Lily found a needle in her room.", voice="Bruno")
model.generate_to_file("Hello world!", "output.wav", voice="Luna", preset="expressive")

# Pipelined chunk streaming
for chunk in model.generate_stream("First sentence here. Second sentence right after.", voice="Bruno"):
    print("Received audio chunk:", chunk.shape)

# Zero-shot runtime voice cloning from WAV
cloned_audio = model.generate(
    "This is my cloned voice.",
    reference="my_voice.wav",
    reference_text="Transcript of the reference clip."
)
```

### 3. C++20 API (`include/kitteni/kitteni.hpp`)

```cpp
#include <kitteni/kitteni.hpp>

int main() {
    kitteni::EngineOptions opts;
    opts.model_or_path = "KittenML/kitten-tts-2";
    opts.weights = "emb4";
    opts.decoder = "student_w4";
    opts.threads = 8;

    kitteni::Engine engine(opts);

    kitteni::GenerateOptions gen;
    gen.voice = "Bruno";
    engine.generate_to_file("Hello from native C++20!", "hello.wav", gen);
}
```

### 4. C ABI (`include/kitteni/kitteni.h`)

```c
#include <kitteni/kitteni.h>

int main(void) {
    kitteni_engine_options_t eo;
    kitteni_default_engine_options(&eo);
    kitteni_engine_t* eng = kitteni_create(&eo);

    kitteni_generate_options_t go;
    kitteni_default_generate_options(&go);
    go.voice = "Bruno";

    kitteni_report_t report;
    kitteni_generate_to_file(eng, "Hello from the C ABI.", "c_out.wav", &go, &report);
    kitteni_destroy(eng);
    return 0;
}
```

---

## Repository Structure

```text
KittenI/
├── CMakeLists.txt
├── include/kitteni/
│   ├── common.hpp                 # TensorView, FP16/BF16 converters, TernaryMatrix
│   ├── kitteni.h                  # Public C ABI (shared/static library)
│   └── kitteni.hpp                # Idiomatic C++20 Engine & Pipelined Streamer API
├── src/
│   ├── core/
│   │   ├── json.hpp/.cpp          # Zero-dependency JSON parser
│   │   ├── mmap_loader.hpp/.cpp   # Zero-copy Win32/POSIX mmap (.safetensors, .npz, .pt ZIP64)
│   │   ├── hub_cache.hpp/.cpp     # HuggingFace Hub cache resolver & downloader
│   │   └── thread_pool.hpp        # Persistent spin/condvar multi-threaded worker pool
│   ├── kernels/
│   │   ├── cpu_features.hpp/.cpp  # Runtime CPUID detection & ISA selector (SCALAR/AVX2/AVX_VNNI/AVX512_VNNI)
│   │   ├── ternary_unpack.hpp/.cpp# Lossless unpackers for TL2, 1.58-bit base-3, and TQ2_1
│   │   ├── fwht_emb4.hpp/.cpp     # AVX2/AVX-512 FWHT & dual-space 4-bit tiered embedding table
│   │   ├── ternary_gemv.hpp/.cpp  # Multi-ISA INT8 x 2-bit Ternary GEMV/GEMM kernels
│   │   ├── bitnet_ops.hpp/.cpp    # Multi-ISA BitLinear, BitConv1d, Conv1d, ConvTranspose1d
│   │   ├── nn_ops.hpp/.cpp        # RMSNorm, LayerNorm, NeoX RoPE, SwiGLU
│   │   └── dsp_fft.hpp/.cpp       # WAV reader, Kaiser sinc resampler, STFT/ISTFT, Mel80/128, Kaldi fbank
│   ├── frontend/
│   │   ├── bpe_tokenizer.hpp/.cpp # Exact Qwen3 byte-level BPE tokenizer
│   │   ├── text_normalizer.hpp/.cpp# Tag protection ([emotion], <event>, (((emphasis)))), normalizer, chunker
│   │   └── audio_joiner.hpp/.cpp  # RMS -45dB silence trimmer, edge crossfader, WAV writer
│   ├── lm/
│   │   ├── model_loader.hpp/.cpp  # Parallel TL2/1.58-bit loader with fused QKV & Gate+Up projections
│   │   ├── sampler.hpp/.cpp       # TokenMap, prompt builder, TokenRunPenalty & RepetitionPenaltyExempt
│   │   └── qwen3_ternary.hpp/.cpp # Dynamic-depth Qwen3 GQA + QK-Norm + spk_proj prefill engine
│   ├── vocoder/
│   │   ├── s3_flow.hpp/.cpp       # Native UpsampleConformerEncoder + 1-step MeanFlow UNet
│   │   ├── hift_gan.hpp/.cpp      # Native ConvRNNF0Predictor + SourceModuleHnNSF + HiFTGenerator
│   │   └── vocoder_pipeline.hpp/.cpp # End-to-end token-to-waveform pipeline
│   ├── cloning/
│   │   ├── sincnet_xvector.hpp/.cpp # Pyannote XVectorSincNet (512-d speaker embedding)
│   │   ├── campplus.hpp/.cpp      # CosyVoice CAMPPlus (192-d flow x-vector)
│   │   ├── s3_tokenizer.hpp/.cpp  # S3TokenizerV2 (25 Hz FSQ speech tokenizer)
│   │   └── voice_cloner.hpp/.cpp  # Runtime zero-shot WAV cloner & HTTP Range weight slicer
│   └── api/
│       ├── engine.cpp             # C++20 Engine & producer-consumer chunk streaming
│       └── c_api.cpp              # C ABI implementation
├── cli/
│   └── main.cpp                   # `kitteni` CLI binary
├── python/
│   ├── pyproject.toml
│   └── kitteni/__init__.py        # Drop-in Python `KittenTTS` wrapper over `libkitteni`
└── tests/
    ├── test_phase1.cpp            # Ternary unpacking, FWHT, and GEMV accuracy tests
    ├── test_phase2.cpp            # BPE tokenizer, normalizer, sampler, and Qwen3 LM tests
    ├── test_phase3.cpp            # STFT/ISTFT, multi-ISA BitLinear, and native vocoder tests
    ├── test_phase4.cpp            # Resampler, SincNet, CAMPPlus, and S3TokenizerV2 parity tests
    └── test_phase5.cpp            # C++ Engine, pipelined streaming, and C ABI tests
```

---

## License & Attribution

- **KittenI Runtime Code**: Licensed under the [Apache License 2.0](LICENSE).
- **Model Weights**: KittenTTS 2 weights (`KittenML/kitten-tts-2`) are released by KittenML / Stellon Labs under the [Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md). S3 codec components derive from ResembleAI's Chatterbox and Alibaba's CosyVoice (Apache 2.0), and the speaker embedding model derives from `pyannote/embedding` (MIT).
