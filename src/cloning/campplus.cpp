#include "cloning/campplus.hpp"
#include "kernels/dsp_fft.hpp"
#include "kernels/bitnet_ops.hpp"
#include "kernels/nn_ops.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace kitteni {

namespace {

std::vector<float> get_f32(const SafetensorsFile& sf, const std::string& key) {
    TensorView tv = sf.get_tensor(key);
    size_t n = tv.numel();
    std::vector<float> out(n);
    std::memcpy(out.data(), tv.data, n * sizeof(float));
    return out;
}

CampPlusEncoder::FoldedBN load_folded_bn(const SafetensorsFile& sf, const std::string& base, bool affine = true) {
    auto mean = get_f32(sf, base + "running_mean");
    auto var  = get_f32(sf, base + "running_var");
    size_t C = mean.size();

    CampPlusEncoder::FoldedBN bn;
    bn.scale.resize(C);
    bn.shift.resize(C);

    if (affine) {
        auto weight = get_f32(sf, base + "weight");
        auto bias   = get_f32(sf, base + "bias");
        for (size_t c = 0; c < C; ++c) {
            float sc = weight[c] / std::sqrt(var[c] + 1e-5f);
            bn.scale[c] = sc;
            bn.shift[c] = bias[c] - mean[c] * sc;
        }
    } else {
        for (size_t c = 0; c < C; ++c) {
            float sc = 1.0f / std::sqrt(var[c] + 1e-5f);
            bn.scale[c] = sc;
            bn.shift[c] = -mean[c] * sc;
        }
    }
    return bn;
}

// 2D Convolution on `[C_in, F_in, T_in]` with kernel `(kH, kW)`, stride `(sF, 1)`, padding `(padF, padT)`
std::vector<float> conv2d_f32(
    const float* x,
    const float* weight,
    int64_t C_in,
    int64_t C_out,
    int64_t F_in,
    int64_t T_in,
    int64_t kF,
    int64_t kT,
    int64_t sF,
    int64_t padF,
    int64_t padT,
    int64_t& F_out
) {
    F_out = (F_in + 2 * padF - kF) / sF + 1;
    std::vector<float> y(static_cast<size_t>(C_out * F_out * T_in), 0.0f);

    for (int64_t co = 0; co < C_out; ++co) {
        for (int64_t ci = 0; ci < C_in; ++ci) {
            const float* w_kern = weight + ((co * C_in + ci) * kF) * kT;
            const float* x_ch = x + ci * (F_in * T_in);

            for (int64_t fo = 0; fo < F_out; ++fo) {
                float* y_row = y.data() + (co * F_out + fo) * T_in;
                for (int64_t kf = 0; kf < kF; ++kf) {
                    int64_t fi = fo * sF - padF + kf;
                    if (fi < 0 || fi >= F_in) continue;
                    const float* x_row = x_ch + fi * T_in;

                    for (int64_t kt = 0; kt < kT; ++kt) {
                        const float wk = w_kern[kf * kT + kt];
                        const int64_t t_off = kt - padT;
                        const int64_t t_start = std::max<int64_t>(0, -t_off);
                        const int64_t t_end   = std::min<int64_t>(T_in, T_in - t_off);
                        for (int64_t t = t_start; t < t_end; ++t) {
                            y_row[t] += wk * x_row[t + t_off];
                        }
                    }
                }
            }
        }
    }
    return y;
}

void apply_bn_relu_inplace(
    float* x,
    int64_t C,
    int64_t spatial_len,
    const CampPlusEncoder::FoldedBN& bn,
    bool apply_relu = true
) {
    for (int64_t c = 0; c < C; ++c) {
        const float sc = bn.scale[c];
        const float sh = bn.shift[c];
        float* row = x + c * spatial_len;
        for (int64_t i = 0; i < spatial_len; ++i) {
            float v = row[i] * sc + sh;
            row[i] = (apply_relu && v < 0.0f) ? 0.0f : v;
        }
    }
}

} // namespace

void CampPlusEncoder::load(const SafetensorsFile& sf, const std::string& prefix) {
    head_conv1_w_ = get_f32(sf, prefix + "head.conv1.weight");
    head_bn1_     = load_folded_bn(sf, prefix + "head.bn1.");

    auto load_resblock2d = [&](const std::string& p, int stride_f, BasicResBlock2D& b) {
        b.stride_f = stride_f;
        b.conv1_w  = get_f32(sf, p + "conv1.weight");
        b.bn1      = load_folded_bn(sf, p + "bn1.");
        b.conv2_w  = get_f32(sf, p + "conv2.weight");
        b.bn2      = load_folded_bn(sf, p + "bn2.");
        b.has_shortcut = (stride_f != 1);
        if (b.has_shortcut) {
            b.shortcut_w  = get_f32(sf, p + "shortcut.0.weight");
            b.shortcut_bn = load_folded_bn(sf, p + "shortcut.1.");
        }
    };

    load_resblock2d(prefix + "head.layer1.0.", 2, head_layer1_[0]);
    load_resblock2d(prefix + "head.layer1.1.", 1, head_layer1_[1]);
    load_resblock2d(prefix + "head.layer2.0.", 2, head_layer2_[0]);
    load_resblock2d(prefix + "head.layer2.1.", 1, head_layer2_[1]);

    head_conv2_w_ = get_f32(sf, prefix + "head.conv2.weight");
    head_bn2_     = load_folded_bn(sf, prefix + "head.bn2.");

    tdnn_w_  = get_f32(sf, prefix + "xvector.tdnn.linear.weight");
    tdnn_bn_ = load_folded_bn(sf, prefix + "xvector.tdnn.nonlinear.batchnorm.");

    auto load_cam_block = [&](const std::string& blk_p, int num_layers, int64_t init_ch, int64_t dil,
                              std::vector<CamDenseLayer>& out_layers) {
        out_layers.resize(static_cast<size_t>(num_layers));
        for (int i = 0; i < num_layers; ++i) {
            CamDenseLayer& l = out_layers[i];
            l.in_ch = init_ch + i * 32;
            l.out_ch = 32;
            l.bn_ch = 128;
            l.kernel_size = 3;
            l.dilation = dil;

            std::string lp = blk_p + "tdnnd" + std::to_string(i + 1) + ".";
            l.nonlinear1_bn      = load_folded_bn(sf, lp + "nonlinear1.batchnorm.");
            l.linear1_w          = get_f32(sf, lp + "linear1.weight");
            l.nonlinear2_bn      = load_folded_bn(sf, lp + "nonlinear2.batchnorm.");
            l.cam_linear_local_w = get_f32(sf, lp + "cam_layer.linear_local.weight");
            l.cam_linear1_w      = get_f32(sf, lp + "cam_layer.linear1.weight");
            l.cam_linear1_b      = get_f32(sf, lp + "cam_layer.linear1.bias");
            l.cam_linear2_w      = get_f32(sf, lp + "cam_layer.linear2.weight");
            l.cam_linear2_b      = get_f32(sf, lp + "cam_layer.linear2.bias");
        }
    };

    load_cam_block(prefix + "xvector.block1.", 12, 128, 1, block1_);
    transit1_.in_ch = 512; transit1_.out_ch = 256;
    transit1_.bn = load_folded_bn(sf, prefix + "xvector.transit1.nonlinear.batchnorm.");
    transit1_.linear_w = get_f32(sf, prefix + "xvector.transit1.linear.weight");

    load_cam_block(prefix + "xvector.block2.", 24, 256, 2, block2_);
    transit2_.in_ch = 1024; transit2_.out_ch = 512;
    transit2_.bn = load_folded_bn(sf, prefix + "xvector.transit2.nonlinear.batchnorm.");
    transit2_.linear_w = get_f32(sf, prefix + "xvector.transit2.linear.weight");

    load_cam_block(prefix + "xvector.block3.", 16, 512, 2, block3_);
    transit3_.in_ch = 1024; transit3_.out_ch = 512;
    transit3_.bn = load_folded_bn(sf, prefix + "xvector.transit3.nonlinear.batchnorm.");
    transit3_.linear_w = get_f32(sf, prefix + "xvector.transit3.linear.weight");

    out_nonlinear_bn_ = load_folded_bn(sf, prefix + "xvector.out_nonlinear.batchnorm.");
    dense_w_          = get_f32(sf, prefix + "xvector.dense.linear.weight");
    dense_bn_         = load_folded_bn(sf, prefix + "xvector.dense.nonlinear.batchnorm.", /*affine=*/false);
}

std::vector<float> CampPlusEncoder::encode_16k(
    const float* wav_16k,
    int64_t num_samples,
    const ThreadPool* pool
) const {
    // 1. Extract Kaldi 80-bin fbank features `[T_fbank, 80]` and permute to `[1, 80, T_fbank]`
    int64_t T = 0;
    std::vector<float> fbank_Tx80 = extract_kaldi_fbank80_16k(wav_16k, num_samples, T);
    if (T <= 0) {
        throw std::invalid_argument("Reference audio too short for CampPlusEncoder");
    }

    int64_t F = 80;
    std::vector<float> x(static_cast<size_t>(F * T));
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t f = 0; f < F; ++f) {
            x[f * T + t] = fbank_Tx80[t * 80 + f];
        }
    }

    // 2. FCM 2D ResNet head:
    int64_t F_next = 0;
    x = conv2d_f32(x.data(), head_conv1_w_.data(), 1, 32, F, T, 3, 3, 1, 1, 1, F_next);
    F = F_next;
    apply_bn_relu_inplace(x.data(), 32, F * T, head_bn1_, true);

    auto run_resblock2d = [&](const BasicResBlock2D& b) {
        int64_t F1 = 0, F2 = 0;
        auto out = conv2d_f32(x.data(), b.conv1_w.data(), 32, 32, F, T, 3, 3, b.stride_f, 1, 1, F1);
        apply_bn_relu_inplace(out.data(), 32, F1 * T, b.bn1, true);
        out = conv2d_f32(out.data(), b.conv2_w.data(), 32, 32, F1, T, 3, 3, 1, 1, 1, F2);
        apply_bn_relu_inplace(out.data(), 32, F2 * T, b.bn2, false);

        if (b.has_shortcut) {
            int64_t Fs = 0;
            auto sc = conv2d_f32(x.data(), b.shortcut_w.data(), 32, 32, F, T, 1, 1, b.stride_f, 0, 0, Fs);
            apply_bn_relu_inplace(sc.data(), 32, Fs * T, b.shortcut_bn, false);
            for (size_t i = 0; i < out.size(); ++i) {
                float v = out[i] + sc[i];
                out[i] = (v > 0.0f) ? v : 0.0f;
            }
        } else {
            for (size_t i = 0; i < out.size(); ++i) {
                float v = out[i] + x[i];
                out[i] = (v > 0.0f) ? v : 0.0f;
            }
        }
        x.swap(out);
        F = F2;
    };

    run_resblock2d(head_layer1_[0]);
    run_resblock2d(head_layer1_[1]);
    run_resblock2d(head_layer2_[0]);
    run_resblock2d(head_layer2_[1]);

    x = conv2d_f32(x.data(), head_conv2_w_.data(), 32, 32, F, T, 3, 3, 2, 1, 1, F_next);
    F = F_next; // F = 10
    apply_bn_relu_inplace(x.data(), 32, F * T, head_bn2_, true);
    // `x` reshaped to `[32 * 10, T] = [320, T]` is already contiguous `[320, T]` in memory!

    // 3. `tdnn`: Conv1d(320 -> 128, k=5, stride=2, pad=2) + BN + ReLU
    int64_t T_tdnn = (T + 2 * 2 - 5) / 2 + 1;
    std::vector<float> td_out(static_cast<size_t>(128 * T_tdnn));
    conv1d_f32(x.data(), tdnn_w_.data(), nullptr, 320, 128, T, 5, 2, 2, 1, td_out.data(), pool);
    apply_bn_relu_inplace(td_out.data(), 128, T_tdnn, tdnn_bn_, true);
    x.swap(td_out);
    T = T_tdnn;
    int64_t cur_ch = 128;

    // 4. Helper for CAMDenseTDNNBlock + TransitLayer
    auto run_dense_block_and_transit = [&](const std::vector<CamDenseLayer>& layers, const TransitBlock& tr) {
        for (const auto& l : layers) {
            // a) nonlinear1 (BN + ReLU) on `x` (`[cur_ch, T]`) -> `linear1` (1x1 conv `cur_ch -> 128`)
            std::vector<float> x_nl1 = x;
            apply_bn_relu_inplace(x_nl1.data(), cur_ch, T, l.nonlinear1_bn, true);

            std::vector<float> bn_feat(static_cast<size_t>(128 * T));
            conv1d_f32(x_nl1.data(), l.linear1_w.data(), nullptr, cur_ch, 128, T, 1, 1, 0, 1, bn_feat.data(), pool);

            // b) nonlinear2 (BN + ReLU) on `bn_feat` (`[128, T]`)
            apply_bn_relu_inplace(bn_feat.data(), 128, T, l.nonlinear2_bn, true);

            // c) CAMLayer:
            //    `y = linear_local(bn_feat)` (Conv1d(128->32, k=3, pad=dilation, dilation=dilation))
            std::vector<float> y(static_cast<size_t>(32 * T));
            conv1d_f32(bn_feat.data(), l.cam_linear_local_w.data(), nullptr,
                       128, 32, T, 3, 1, l.dilation, l.dilation, y.data(), pool);

            //    `context = bn_feat.mean(-1, keepdim=True) + seg_pooling(bn_feat, seg_len=100, ceil_mode=True)`
            const int64_t seg_len = 100;
            const int64_t n_segs = (T + seg_len - 1) / seg_len;
            std::vector<float> context(static_cast<size_t>(128 * T));

            for (int64_t c = 0; c < 128; ++c) {
                const float* row = bn_feat.data() + c * T;
                double g_sum = 0.0;
                for (int64_t t = 0; t < T; ++t) g_sum += row[t];
                float g_mean = static_cast<float>(g_sum / static_cast<double>(T));

                float* ctx_row = context.data() + c * T;
                for (int64_t s = 0; s < n_segs; ++s) {
                    int64_t s_start = s * seg_len;
                    int64_t s_end = std::min(T, s_start + seg_len);
                    // Note: PyTorch F.avg_pool1d(..., ceil_mode=True, count_include_pad=True) divides by kernel_size (100)
                    // unless count_include_pad=False; default count_include_pad is True, wait: on 1D with padding=0,
                    // PyTorch avg_pool1d with padding=0 divides by actual window length `s_end - s_start`!
                    double seg_sum = 0.0;
                    for (int64_t t = s_start; t < s_end; ++t) seg_sum += row[t];
                    float seg_mean = static_cast<float>(seg_sum / static_cast<double>(s_end - s_start));
                    float val = g_mean + seg_mean;
                    for (int64_t t = s_start; t < s_end; ++t) ctx_row[t] = val;
                }
            }

            //    `m = sigmoid(linear2(relu(linear1(context))))`
            std::vector<float> ctx_mid(static_cast<size_t>(64 * T));
            conv1d_f32(context.data(), l.cam_linear1_w.data(), l.cam_linear1_b.data(),
                       128, 64, T, 1, 1, 0, 1, ctx_mid.data(), pool);
            for (float& v : ctx_mid) if (v < 0.0f) v = 0.0f;

            std::vector<float> m_gate(static_cast<size_t>(32 * T));
            conv1d_f32(ctx_mid.data(), l.cam_linear2_w.data(), l.cam_linear2_b.data(),
                       64, 32, T, 1, 1, 0, 1, m_gate.data(), pool);

            for (size_t i = 0; i < y.size(); ++i) {
                float sig = 1.0f / (1.0f + std::exp(-m_gate[i]));
                y[i] *= sig;
            }

            // Concatenate `y` (`[32, T]`) onto `x` (`[cur_ch, T]`) -> `[cur_ch + 32, T]`
            x.insert(x.end(), y.begin(), y.end());
            cur_ch += 32;
        }

        // TransitLayer: BN + ReLU -> Conv1d(in_ch -> out_ch, k=1)
        apply_bn_relu_inplace(x.data(), tr.in_ch, T, tr.bn, true);
        std::vector<float> tr_out(static_cast<size_t>(tr.out_ch * T));
        conv1d_f32(x.data(), tr.linear_w.data(), nullptr, tr.in_ch, tr.out_ch, T, 1, 1, 0, 1, tr_out.data(), pool);
        x.swap(tr_out);
        cur_ch = tr.out_ch;
    };

    run_dense_block_and_transit(block1_, transit1_);
    run_dense_block_and_transit(block2_, transit2_);
    run_dense_block_and_transit(block3_, transit3_);

    // 5. `out_nonlinear`: BN + ReLU on `[512, T]`
    apply_bn_relu_inplace(x.data(), 512, T, out_nonlinear_bn_, true);

    // 6. `StatsPool`: mean and unbiased std over `T` -> `[1024]`
    std::vector<float> stats(1024);
    for (int64_t c = 0; c < 512; ++c) {
        const float* row = x.data() + c * T;
        double sum = 0.0;
        for (int64_t t = 0; t < T; ++t) sum += row[t];
        double mean = sum / static_cast<double>(T);

        double sq_sum = 0.0;
        for (int64_t t = 0; t < T; ++t) {
            double d = static_cast<double>(row[t]) - mean;
            sq_sum += d * d;
        }
        // Notice: `statistics_pooling` in s3gen.py has `std = x.std(dim=dim, unbiased=True)`
        double std_val = (T > 1) ? std::sqrt(sq_sum / static_cast<double>(T - 1)) : 0.0;
        stats[c]       = static_cast<float>(mean);
        stats[c + 512] = static_cast<float>(std_val);
    }

    // 7. `DenseLayer`: Conv1d(1024 -> 192, k=1, bias=False) + BatchNorm1d(192, affine=False)
    std::vector<float> emb(192);
    dense_linear_f32(stats.data(), dense_w_.data(), nullptr, 1024, 192, emb.data());
    for (int i = 0; i < 192; ++i) {
        emb[i] = emb[i] * dense_bn_.scale[i] + dense_bn_.shift[i];
    }
    return emb;
}

} // namespace kitteni
