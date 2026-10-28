#include "core/hub_cache.hpp"
#include "core/mmap_loader.hpp"
#include "frontend/bpe_tokenizer.hpp"
#include "frontend/text_normalizer.hpp"
#include "lm/model_loader.hpp"
#include "lm/sampler.hpp"
#include "lm/qwen3_ternary.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <cstdlib>

using namespace kitteni;

#define REQUIRE(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1); \
        } \
    } while (0)

static void test_bpe_tokenizer_exact_parity(const ResolvedRepo& repo) {
    std::cout << "[Test 1] Verifying Qwen3 BPE Tokenizer exact parity against HuggingFace tokenizers...\n";
    std::string lm_dir = (repo.root_dir / repo.config.value_string("lm_dir", "lm")).string();
    Qwen3BPETokenizer tok;
    tok.load(lm_dir);

    // 1. "{emo: 1}"
    {
        auto ids = tok.encode("{emo: 1}");
        std::vector<int32_t> expected = {90, 6726, 25, 220, 16, 92};
        REQUIRE(ids == expected);
        REQUIRE(tok.decode(ids) == "{emo: 1}");
    }

    // 2. "One day, a little girl named Lily found a needle in her room."
    {
        std::string s = "One day, a little girl named Lily found a needle in her room.";
        auto ids = tok.encode(s);
        std::vector<int32_t> expected = {
            3966, 1899, 11, 264, 2632, 3743, 6941, 47290, 1730, 264, 30309, 304, 1059, 3054, 13
        };
        REQUIRE(ids == expected);
        REQUIRE(tok.decode(ids) == s);
    }

    // 3. "[joyful] We won the grant <laugh> I can (((hardly))) believe it!"
    {
        std::string s = "[joyful] We won the grant <laugh> I can (((hardly))) believe it!";
        auto ids = tok.encode(s);
        std::vector<int32_t> expected = {
            3809, 2253, 1262, 60, 1205, 2765, 279, 13198, 366, 4260, 7443, 29, 358, 646, 11600, 18703, 398, 7705, 4411, 432, 0
        };
        REQUIRE(ids == expected);
        REQUIRE(tok.decode(ids) == s);
    }

    // 4. Bruno reference transcript
    {
        std::string s = "When the Western music overwhelms the Chinese music, it evokes the long history of conflict between Asia and the Western nations that sought to exploit and dominate it.";
        auto ids = tok.encode(s);
        std::vector<int32_t> expected = {
            4498, 279, 10867, 4627, 16783, 1011, 279, 8453, 4627, 11, 432, 3637, 7940, 279, 1293, 3840, 315, 12055, 1948, 13622, 323, 279, 10867, 16675, 429, 16105, 311, 32194, 323, 40736, 432, 13
        };
        REQUIRE(ids == expected);
        REQUIRE(tok.decode(ids) == s);
    }

    // 5. UTF-8 umlauts: "Guten Morgen. Ich wünsche dir einen wunderschönen Tag."
    {
        std::string s = "Guten Morgen. Ich w\xC3\xBCnsche dir einen wundersch\xC3\xB6nen Tag.";
        auto ids = tok.encode(s);
        std::vector<int32_t> expected = {
            38, 13160, 386, 39563, 13, 25861, 289, 66107, 1528, 5419, 15462, 289, 31009, 331, 2956, 12495, 12353, 13
        };
        REQUIRE(ids == expected);
        REQUIRE(tok.decode(ids) == s);
    }

    std::cout << "  All 5 BPE reference strings matched HuggingFace tokenizers 100% bit-for-bit.\n";
}

static void test_normalizer_and_chunker() {
    std::cout << "[Test 2] Testing Text Normalizer, Expression Tag Protection & Sentence Chunker...\n";

    REQUIRE(has_expression_tag("[joyful] We won <laugh> (((hardly)))!"));
    REQUIRE(!has_expression_tag("See section [3] where x < 5."));

    // Verify expression tags survive normalization verbatim while numbers/currencies expand
    std::string raw = "[joyful] We won $12.50 on the 1st try <laugh> I (((really))) mean it";
    std::string norm = normalize_text(raw, true);
    std::cout << "  Normalized: " << norm << "\n";
    REQUIRE(norm.find("[joyful]") != std::string::npos);
    REQUIRE(norm.find("<laugh>") != std::string::npos);
    REQUIRE(norm.find("(((really)))") != std::string::npos);
    REQUIRE(norm.find("twelve dollars and fifty cents") != std::string::npos);
    REQUIRE(norm.find("first") != std::string::npos);
    REQUIRE(norm.back() == '.');

    // Verify non-tag brackets become commas
    std::string citation = "In section [3], x = 5.";
    std::string norm_cit = normalize_text(citation, true);
    std::cout << "  Citation normalized: " << norm_cit << "\n";
    REQUIRE(norm_cit.find("[") == std::string::npos);
    REQUIRE(norm_cit.find("equals") != std::string::npos);

    // Verify sentence chunking and short-chunk folding
    std::string long_text =
        "The house at the end of Willow Lane had been empty for eleven years, and yet every evening, at exactly seven o'clock, a light came on in the attic window. "
        "Nobody in the village spoke about it after dark, though the baker claimed he once saw a tall shadow pacing behind the curtains. "
        "When Lily finally found the brass key beneath the garden stone, her hands were trembling too much to turn it in the lock. "
        "Still, she stepped inside.";
    auto chunks = split_for_synthesis(long_text, 220, 80);
    REQUIRE(chunks.size() >= 2);
    for (size_t i = 0; i < chunks.size(); ++i) {
        std::cout << "  Chunk " << i << " (" << chunks[i].size() << " chars): " << chunks[i] << "\n";
        REQUIRE(chunks[i].size() >= 80);
    }
}

static void test_prompt_and_logits_processors(const ResolvedRepo& repo) {
    std::cout << "[Test 3] Testing TokenMap, Prompt Assembly & Logits Processors...\n";
    TokenMap tm = TokenMap::from_config(repo.config);
    REQUIRE(tm.audio_id_base == 151675);
    REQUIRE(tm.num_audio_tokens == 6561);
    REQUIRE(tm.uses_reference_role_tokens());

    std::vector<int32_t> ref_text = {10, 20};
    std::vector<int32_t> ref_audio = {0, 4299};
    auto prefix = build_reference_prefix(tm, ref_text, ref_audio);
    std::vector<int32_t> expected_prefix = {
        tm.start_id,
        tm.reference_text_start_id, 10, 20, tm.reference_text_end_id,
        tm.reference_speech_start_id, 151675, 151675 + 4299, tm.reference_speech_end_id
    };
    REQUIRE(prefix == expected_prefix);

    std::vector<int32_t> target_text = {30, 40};
    std::vector<int32_t> emo_ids = {90, 92};
    auto prompt = build_generation_prompt(tm, target_text, &prefix, &emo_ids);
    REQUIRE(prompt.front() == tm.start_id);
    REQUIRE(prompt.back() == tm.speech_start_id);

    // Test TokenRunPenalty + RepetitionPenaltyExempt
    std::vector<float> scores(158241, 0.0f);
    int32_t stuck_tok = tm.audio_id_base + 4299; // Silence token
    scores[stuck_tok] = 10.0f;
    scores[tm.speech_end_id] = 10.0f;

    std::vector<int32_t> history(12, stuck_tok); // 12 consecutive repeats (>= grace 10)
    history.push_back(tm.speech_end_id);
    history.pop_back(); // back is stuck_tok with run=12

    SamplingSettings s;
    s.temperature = 1.0f;
    s.top_k = 0;
    s.top_p = 1.0f;
    s.min_p = 0.0f;
    s.repetition_penalty = 1.1f;
    s.repetition_window = 50;
    s.token_run_grace = 10;
    s.token_run_penalty = 1.3f;

    apply_logits_processors(scores, history, 0, tm, s);
    // Expected for stuck_tok:
    // 1. TokenRunPenalty: 10.0 - (12 - 10 + 1) * 1.3 = 10.0 - 3.9 = 6.1
    // 2. RepetitionPenalty: 6.1 / 1.1 = 5.5454545
    REQUIRE(std::fabs(scores[stuck_tok] - (6.1f / 1.1f)) < 1e-5f);
    // speech_end_id is exempt even if in history
    history.push_back(tm.speech_end_id);
    scores[tm.speech_end_id] = 10.0f;
    apply_logits_processors(scores, history, 0, tm, s);
    REQUIRE(std::fabs(scores[tm.speech_end_id] - 10.0f) < 1e-5f);
}

static void test_qwen3_ternary_engine_prefill_and_decode(const ResolvedRepo& repo) {
    std::cout << "[Test 4] Testing Qwen3TernaryEngine prefill + decode step on real emb4 layers...\n";
    KittenTernaryLMWeights lm;
    lm.load(repo, "emb4", 2); // Load 2 layers + full emb4 + spk_proj for fast unit test

    TokenMap tm = TokenMap::from_config(repo.config);
    Qwen3TernaryEngine engine(lm, 4, /*use_int8_act=*/true);

    std::vector<float> spk(512, 0.01f);
    std::vector<int32_t> prompt = {tm.start_id, tm.text_start_id, 3966, 1899, tm.final_seg_id, tm.speech_start_id};

    Qwen3KVCache kv;
    kv.init(lm.config, 64);

    std::vector<float> logits;
    engine.prefill(prompt, spk.data(), kv, logits, &tm, /*speech_decode_only=*/true);
    REQUIRE(kv.cur_seq_len == static_cast<int64_t>(prompt.size()));
    REQUIRE(static_cast<int64_t>(logits.size()) == lm.config.vocab_size);
    REQUIRE(std::isfinite(logits[tm.speech_end_id]));
    REQUIRE(std::isfinite(logits[tm.audio_id_base + 100]));
    REQUIRE(!std::isfinite(logits[0])); // Non-speech token masked when speech_decode_only=true

    // Run 1 decode step
    engine.decode_step(tm.audio_id_base + 100, kv, logits, &tm, /*speech_decode_only=*/true);
    REQUIRE(kv.cur_seq_len == static_cast<int64_t>(prompt.size()) + 1);
    REQUIRE(std::isfinite(logits[tm.audio_id_base + 200]));
    std::cout << "  2-layer prefill + decode_step succeeded; audio logit[151875] = "
              << logits[tm.audio_id_base + 200] << "\n";
}

int main() {
    std::cout << "=== KittenI Phase 2 Unit & Parity Tests ===\n";
    ResolvedRepo repo = resolve_repository("KittenML/kitten-tts-2", "emb4", "", false);
    test_bpe_tokenizer_exact_parity(repo);
    test_normalizer_and_chunker();
    test_prompt_and_logits_processors(repo);
    test_qwen3_ternary_engine_prefill_and_decode(repo);
    std::cout << "=== All Phase 2 Unit Tests PASSED ===\n";
    return 0;
}
