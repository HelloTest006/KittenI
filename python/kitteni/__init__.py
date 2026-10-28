"""KittenI Python bindings wrapping the native C++20 `libkitteni` shared library.

Provides a drop-in `KittenTTS` class compatible with `kittenml.KittenTTS`:
- `m = KittenTTS("KittenML/kitten-tts-2", weights="emb4", decoder="student_w4", threads=4)`
- `audio = m.generate("Hello world!", voice="Bruno")`
- `for chunk in m.generate_stream(long_text, voice="Luna"): ...`
- `m.generate_to_file("Hello world!", "output.wav", voice="Bruno")`
- `audio = m.generate("Cloned voice.", reference="my_voice.wav", reference_text="...")`
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import sys
from typing import Dict, Iterator, List, Optional

import numpy as np


class _EngineOptionsC(ctypes.Structure):
    _fields_ = [
        ("model_or_path", ctypes.c_char_p),
        ("weights", ctypes.c_char_p),
        ("decoder", ctypes.c_char_p),
        ("cache_dir", ctypes.c_char_p),
        ("isa", ctypes.c_char_p),
        ("threads", ctypes.c_int32),
        ("vocoder_threads", ctypes.c_int32),
        ("max_layers", ctypes.c_int32),
        ("allow_download", ctypes.c_int32),
    ]


class _GenerateOptionsC(ctypes.Structure):
    _fields_ = [
        ("voice", ctypes.c_char_p),
        ("reference_wav", ctypes.c_char_p),
        ("reference_text", ctypes.c_char_p),
        ("preset", ctypes.c_char_p),
        ("max_new_tokens", ctypes.c_int32),
        ("normalize", ctypes.c_int32),
        ("use_reference_prompt", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("min_p", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("repetition_window", ctypes.c_int32),
        ("token_run_penalty", ctypes.c_float),
        ("token_run_grace", ctypes.c_int32),
        ("chunk_chars", ctypes.c_int32),
        ("chunk_min_chars", ctypes.c_int32),
        ("chunk_gap_s", ctypes.c_float),
        ("seed", ctypes.c_int32),
        ("greedy", ctypes.c_int32),
    ]


class _ReportC(ctypes.Structure):
    _fields_ = [
        ("num_chunks", ctypes.c_size_t),
        ("total_generated_tokens", ctypes.c_size_t),
        ("total_audio_tokens", ctypes.c_size_t),
        ("total_samples", ctypes.c_size_t),
        ("lm_seconds", ctypes.c_double),
        ("vocoder_seconds", ctypes.c_double),
        ("wall_seconds", ctypes.c_double),
        ("audio_seconds", ctypes.c_double),
        ("rtf", ctypes.c_double),
    ]


_STREAM_CALLBACK_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_void_p,
)


def _find_library() -> ctypes.CDLL:
    env_lib = os.environ.get("KITTENI_LIBRARY_PATH")
    candidates: List[pathlib.Path] = []
    if env_lib:
        candidates.append(pathlib.Path(env_lib))

    here = pathlib.Path(__file__).resolve().parent
    repo_root = here.parent.parent
    for sub in ("build", "build/Release", "bin", "."):
        for name in ("libkitteni.dll", "kitteni.dll", "libkitteni.so", "libkitteni.dylib"):
            candidates.append(repo_root / sub / name)
            candidates.append(here / name)

    for cand in candidates:
        if cand.is_file():
            if sys.platform == "win32":
                mingw_bin = pathlib.Path(r"C:\ProgramData\mingw64\mingw64\bin")
                if mingw_bin.is_dir() and hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(mingw_bin))
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(cand.parent))
            lib = ctypes.CDLL(str(cand))
            _bind_signatures(lib)
            return lib

    raise FileNotFoundError(
        "Could not locate libkitteni shared library (.dll/.so/.dylib). "
        "Build KittenI with CMake first or set KITTENI_LIBRARY_PATH."
    )


def _bind_signatures(lib: ctypes.CDLL) -> None:
    lib.kitteni_default_engine_options.argtypes = [ctypes.POINTER(_EngineOptionsC)]
    lib.kitteni_default_engine_options.restype = None

    lib.kitteni_default_generate_options.argtypes = [ctypes.POINTER(_GenerateOptionsC)]
    lib.kitteni_default_generate_options.restype = None

    lib.kitteni_last_error.argtypes = []
    lib.kitteni_last_error.restype = ctypes.c_char_p

    lib.kitteni_create.argtypes = [ctypes.POINTER(_EngineOptionsC)]
    lib.kitteni_create.restype = ctypes.c_void_p

    lib.kitteni_destroy.argtypes = [ctypes.c_void_p]
    lib.kitteni_destroy.restype = None

    lib.kitteni_sample_rate.argtypes = [ctypes.c_void_p]
    lib.kitteni_sample_rate.restype = ctypes.c_int32

    lib.kitteni_num_voices.argtypes = [ctypes.c_void_p]
    lib.kitteni_num_voices.restype = ctypes.c_size_t

    lib.kitteni_voice_name.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.kitteni_voice_name.restype = ctypes.c_char_p

    lib.kitteni_generate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(_GenerateOptionsC),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(_ReportC),
    ]
    lib.kitteni_generate.restype = ctypes.c_int

    lib.kitteni_generate_stream.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(_GenerateOptionsC),
        _STREAM_CALLBACK_TYPE,
        ctypes.c_void_p,
        ctypes.POINTER(_ReportC),
    ]
    lib.kitteni_generate_stream.restype = ctypes.c_int

    lib.kitteni_generate_to_file.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(_GenerateOptionsC),
        ctypes.POINTER(_ReportC),
    ]
    lib.kitteni_generate_to_file.restype = ctypes.c_int

    lib.kitteni_free_audio.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.kitteni_free_audio.restype = None


_LIB: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _find_library()
    return _LIB


class KittenTTS:
    """Drop-in CPU-specific ternary TTS inference engine for KittenTTS 2."""

    def __init__(
        self,
        model_name: str = "KittenML/kitten-tts-2",
        cache_dir: Optional[str] = None,
        weights: str = "emb4",
        decoder: str = "student_w4",
        isa: str = "auto",
        threads: int = 0,
        vocoder_threads: int = 0,
        max_layers: int = -1,
        allow_download: bool = True,
    ):
        self._lib = _get_lib()
        opts = _EngineOptionsC()
        self._lib.kitteni_default_engine_options(ctypes.byref(opts))

        b_model = model_name.encode("utf-8")
        b_weights = weights.encode("utf-8")
        b_decoder = decoder.encode("utf-8")
        b_cache = (cache_dir or "").encode("utf-8")
        b_isa = isa.encode("utf-8")

        opts.model_or_path = b_model
        opts.weights = b_weights
        opts.decoder = b_decoder
        opts.cache_dir = b_cache
        opts.isa = b_isa
        opts.threads = int(threads)
        opts.vocoder_threads = int(vocoder_threads)
        opts.max_layers = int(max_layers)
        opts.allow_download = 1 if allow_download else 0

        self._handle = self._lib.kitteni_create(ctypes.byref(opts))
        if not self._handle:
            err = (self._lib.kitteni_last_error() or b"unknown error").decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to initialize KittenI engine: {err}")

        self.last_report: Dict[str, float] = {}

    def __del__(self):
        if getattr(self, "_handle", None) and getattr(self, "_lib", None):
            self._lib.kitteni_destroy(self._handle)
            self._handle = None

    @property
    def sample_rate(self) -> int:
        return int(self._lib.kitteni_sample_rate(self._handle))

    @property
    def available_voices(self) -> List[str]:
        n = self._lib.kitteni_num_voices(self._handle)
        return [
            self._lib.kitteni_voice_name(self._handle, i).decode("utf-8")
            for i in range(n)
        ]

    @property
    def available_decoders(self) -> List[str]:
        return ["default", "student_w4", "student_w8"]

    @property
    def available_weights(self) -> List[str]:
        return ["emb4", "packed", "full"]

    def _build_gen_opts(
        self,
        voice: Optional[str],
        reference: Optional[str],
        reference_text: Optional[str],
        preset: Optional[str],
        max_new_tokens: int,
        normalize: bool,
        use_reference_prompt: Optional[bool],
        temperature: Optional[float],
        top_k: Optional[int],
        top_p: Optional[float],
        min_p: Optional[float],
        advanced: Optional[Dict[str, object]],
    ):
        if voice is not None and reference is not None:
            raise ValueError("give either voice= or reference=, not both")

        opts = _GenerateOptionsC()
        self._lib.kitteni_default_generate_options(ctypes.byref(opts))

        keep_alive = []
        if voice is not None:
            b_v = voice.encode("utf-8")
            keep_alive.append(b_v)
            opts.voice = b_v
        if reference is not None:
            b_r = str(reference).encode("utf-8")
            keep_alive.append(b_r)
            opts.reference_wav = b_r
        if reference_text is not None:
            b_rt = reference_text.encode("utf-8")
            keep_alive.append(b_rt)
            opts.reference_text = b_rt
        if preset is not None:
            b_p = preset.encode("utf-8")
            keep_alive.append(b_p)
            opts.preset = b_p

        opts.max_new_tokens = int(max_new_tokens)
        opts.normalize = 1 if normalize else 0
        if use_reference_prompt is not None:
            opts.use_reference_prompt = 1 if use_reference_prompt else 0
        if temperature is not None:
            opts.temperature = float(temperature)
        if top_k is not None:
            opts.top_k = int(top_k)
        if top_p is not None:
            opts.top_p = float(top_p)
        if min_p is not None:
            opts.min_p = float(min_p)

        adv = advanced or {}
        if "repetition_penalty" in adv and adv["repetition_penalty"] is not None:
            opts.repetition_penalty = float(adv["repetition_penalty"])
        if "repetition_window" in adv and adv["repetition_window"] is not None:
            opts.repetition_window = int(adv["repetition_window"])
        if "token_run_penalty" in adv and adv["token_run_penalty"] is not None:
            opts.token_run_penalty = float(adv["token_run_penalty"])
        if "token_run_grace" in adv and adv["token_run_grace"] is not None:
            opts.token_run_grace = int(adv["token_run_grace"])
        if "chunk_chars" in adv and adv["chunk_chars"] is not None:
            opts.chunk_chars = int(adv["chunk_chars"])
        if "chunk_min_chars" in adv and adv["chunk_min_chars"] is not None:
            opts.chunk_min_chars = int(adv["chunk_min_chars"])
        if "chunk_gap_s" in adv and adv["chunk_gap_s"] is not None:
            opts.chunk_gap_s = float(adv["chunk_gap_s"])
        if "seed" in adv and adv["seed"] is not None:
            opts.seed = int(adv["seed"])
        if "greedy" in adv and adv["greedy"] is not None:
            opts.greedy = 1 if adv["greedy"] else 0

        return opts, keep_alive

    def _record_report(self, rep: _ReportC) -> None:
        self.last_report = {
            "num_chunks": int(rep.num_chunks),
            "total_generated_tokens": int(rep.total_generated_tokens),
            "total_audio_tokens": int(rep.total_audio_tokens),
            "total_samples": int(rep.total_samples),
            "lm_seconds": float(rep.lm_seconds),
            "vocoder_seconds": float(rep.vocoder_seconds),
            "wall_seconds": float(rep.wall_seconds),
            "audio_seconds": float(rep.audio_seconds),
            "rtf": float(rep.rtf),
        }

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        reference: Optional[str] = None,
        reference_text: Optional[str] = None,
        preset: Optional[str] = None,
        max_new_tokens: int = 1000,
        normalize: bool = True,
        use_reference_prompt: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        advanced: Optional[Dict[str, object]] = None,
    ) -> np.ndarray:
        opts, _keep = self._build_gen_opts(
            voice, reference, reference_text, preset, max_new_tokens, normalize,
            use_reference_prompt, temperature, top_k, top_p, min_p, advanced
        )
        pcm_ptr = ctypes.POINTER(ctypes.c_float)()
        num_samples = ctypes.c_size_t(0)
        rep = _ReportC()

        rc = self._lib.kitteni_generate(
            self._handle,
            text.encode("utf-8"),
            ctypes.byref(opts),
            ctypes.byref(pcm_ptr),
            ctypes.byref(num_samples),
            ctypes.byref(rep),
        )
        if rc != 0:
            err = (self._lib.kitteni_last_error() or b"unknown error").decode("utf-8", errors="replace")
            raise RuntimeError(f"kitteni_generate failed: {err}")

        self._record_report(rep)
        n = num_samples.value
        if n == 0 or not pcm_ptr:
            return np.zeros(0, dtype=np.float32)
        try:
            arr = np.ctypeslib.as_array(pcm_ptr, shape=(n,)).copy()
        finally:
            self._lib.kitteni_free_audio(pcm_ptr)
        return arr

    def generate_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        reference: Optional[str] = None,
        reference_text: Optional[str] = None,
        preset: Optional[str] = None,
        max_new_tokens: int = 1000,
        normalize: bool = True,
        use_reference_prompt: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        advanced: Optional[Dict[str, object]] = None,
    ) -> Iterator[np.ndarray]:
        opts, _keep = self._build_gen_opts(
            voice, reference, reference_text, preset, max_new_tokens, normalize,
            use_reference_prompt, temperature, top_k, top_p, min_p, advanced
        )
        chunks: List[np.ndarray] = []

        @_STREAM_CALLBACK_TYPE
        def _cb(idx: int, total: int, pcm_ptr, n_samples: int, _ud) -> int:
            if n_samples > 0 and pcm_ptr:
                chunks.append(np.ctypeslib.as_array(pcm_ptr, shape=(n_samples,)).copy())
            return 1

        rep = _ReportC()
        rc = self._lib.kitteni_generate_stream(
            self._handle,
            text.encode("utf-8"),
            ctypes.byref(opts),
            _cb,
            None,
            ctypes.byref(rep),
        )
        if rc != 0:
            err = (self._lib.kitteni_last_error() or b"unknown error").decode("utf-8", errors="replace")
            raise RuntimeError(f"kitteni_generate_stream failed: {err}")

        self._record_report(rep)
        yield from chunks

    def generate_to_file(
        self,
        text: str,
        output_path: str,
        **kwargs,
    ) -> str:
        opts, _keep = self._build_gen_opts(
            kwargs.get("voice"),
            kwargs.get("reference"),
            kwargs.get("reference_text"),
            kwargs.get("preset"),
            kwargs.get("max_new_tokens", 1000),
            kwargs.get("normalize", True),
            kwargs.get("use_reference_prompt"),
            kwargs.get("temperature"),
            kwargs.get("top_k"),
            kwargs.get("top_p"),
            kwargs.get("min_p"),
            kwargs.get("advanced"),
        )
        rep = _ReportC()
        rc = self._lib.kitteni_generate_to_file(
            self._handle,
            text.encode("utf-8"),
            str(output_path).encode("utf-8"),
            ctypes.byref(opts),
            ctypes.byref(rep),
        )
        if rc != 0:
            err = (self._lib.kitteni_last_error() or b"unknown error").decode("utf-8", errors="replace")
            raise RuntimeError(f"kitteni_generate_to_file failed: {err}")
        self._record_report(rep)
        return str(output_path)
