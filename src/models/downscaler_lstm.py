"""微气候降尺度残差学习（可选模块）——需要 torch。

定位
----
物理内核（src/microclimate.py）给出的是**参数化先验**下的温度/湿度/土壤水场。
当茶园小气候站（杭州市气象局已在西湖龙井产区布设茶叶小气候观测站）数据可用时，
本模块学习"物理内核输出 → 实况"的残差：

    T_obs(t, x) = T_phys(t, x) + f_θ( 地形因子, 区域强迫, 物理内核输出, 时间编码 )

为什么用 LSTM / Temporal Fusion Transformer
------------------------------------------
- 微气候对气象强迫的响应具有**滞后与累积效应**（前几日的辐射与土壤水状态影响当日冠层温度），
  因此需要序列模型而非逐时刻回归；
- 地形因子是**静态协变量**，观测是**动态协变量 + 目标序列**，这正是 TFT 的原生接口；
- 站点稀疏（可能只有 1–3 个站）而样本在时间上很长，属于典型的小样本空间外推问题，
  需要把地形因子作为输入并做时空交叉验证。

使用
----
    pip install torch
    python -m src.models.downscaler_lstm --synthetic      # 合成数据自检（无需真实观测）
    python -m src.models.downscaler_lstm --data observations.npz --epochs 200

数据格式（observations.npz）
    T_phys   (T, n_site)   物理内核在该站的输出气温
    y        (T, n_site)   实况气温
    static   (n_site, k)   站点静态地形因子（高程/坡度/坡向 sin,cos/TWI/SVF）
    forcing  (T, m)        区域强迫（气温、湿度、风速、辐射、降水、土壤水）
缺失值用 NaN 表示，训练时自动跳过。
"""
from __future__ import annotations

import argparse

import numpy as np

try:                                                # pragma: no cover
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception:                                   # noqa: BLE001
    TORCH_OK = False
    nn = object                                     # 占位，保证模块可被导入


if TORCH_OK:

    class DownscalerLSTM(nn.Module):
        """静态地形编码 + 动态序列编码 → 逐时刻残差。"""

        def __init__(self, n_static: int, n_forcing: int, hidden: int = 64,
                     layers: int = 2, dropout: float = 0.15, mc_dropout: bool = True):
            super().__init__()
            self.static_enc = nn.Sequential(
                nn.Linear(n_static, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.lstm = nn.LSTM(input_size=n_forcing + 1,   # +1 为物理内核输出
                                hidden_size=hidden, num_layers=layers,
                                batch_first=True, dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.SiLU(),
                nn.Dropout(dropout if mc_dropout else 0.0), nn.Linear(hidden, 1))
            # 残差尺度约束：物理内核已经解释了大部分方差，残差量级应受限
            self.res_scale = nn.Parameter(torch.tensor(2.0))

        def forward(self, forcing: "torch.Tensor", phys: "torch.Tensor",
                    static: "torch.Tensor") -> "torch.Tensor":
            b, t, _ = forcing.shape
            z = self.static_enc(static).unsqueeze(1).expand(b, t, -1)
            x = torch.cat([forcing, phys.unsqueeze(-1)], dim=-1)
            h, _ = self.lstm(x)
            out = self.head(torch.cat([h, z], dim=-1)).squeeze(-1)
            return phys + self.res_scale * torch.tanh(out)      # 有界残差

        @torch.no_grad()
        def predict_with_uncertainty(self, forcing, phys, static, n_draws: int = 32):
            """MC-Dropout：开启 dropout 做多次前向，给出预测均值与标准差。"""
            self.train()                                       # 打开 dropout
            preds = torch.stack([self.forward(forcing, phys, static) for _ in range(n_draws)])
            self.eval()
            return preds.mean(0), preds.std(0)


def make_windows(forcing: np.ndarray, phys: np.ndarray, y: np.ndarray,
                 static: np.ndarray, seq_len: int = 24, stride: int = 6):
    """切分滑动窗口；任何含 NaN 的窗口直接丢弃。"""
    X, P, S, Y = [], [], [], []
    T = forcing.shape[0]
    for s in range(0, T - seq_len + 1, stride):
        f = forcing[s:s + seq_len]
        p = phys[s:s + seq_len]
        yy = y[s:s + seq_len]
        if np.isnan(f).any() or np.isnan(p).any() or np.isnan(yy).any():
            continue
        for site in range(forcing.shape[1]):
            X.append(f[:, site, :]); P.append(p[:, site]); Y.append(yy[:, site])
            S.append(static[site])
    return (np.asarray(X, dtype=np.float32), np.asarray(P, dtype=np.float32),
            np.asarray(Y, dtype=np.float32), np.asarray(S, dtype=np.float32))


def synthetic_dataset(T: int = 2000, n_site: int = 3, seed: int = 0):
    """合成数据自检：制造一个"物理内核低估阳坡、高估谷地"的残差。"""
    rng = np.random.default_rng(seed)
    static = rng.normal(size=(n_site, 6)).astype(np.float32)
    static[:, 0] = [220.0, 90.0, 40.0][:n_site]            # 高程代理
    forcing = rng.normal(size=(T, n_site, 5)).astype(np.float32)
    forcing[:, :, 0] += 28.0                                # 气温基准
    phys = forcing[:, :, 0] + rng.normal(0, 0.3, (T, n_site)).astype(np.float32)
    bias = (0.02 * (static[:, 0] - 80.0) * -1 + 0.6 * static[:, 1]).astype(np.float32)
    y = phys + bias[None, :] + rng.normal(0, 0.2, (T, n_site)).astype(np.float32)
    return forcing, phys, y, static


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="observations.npz")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", default="outputs/downscaler_lstm.pt")
    args = ap.parse_args()

    if not TORCH_OK:
        print("未安装 torch，跳过残差学习模块（不影响主管线）。"
              "启用方式：pip install torch")
        return

    if args.synthetic or not args.data:
        print("使用合成数据自检……")
        forcing, phys, y, static = synthetic_dataset()
    else:
        d = np.load(args.data)
        forcing, phys, y, static = d["forcing"], d["T_phys"], d["y"], d["static"]

    X, P, Y, S = make_windows(forcing, phys, y, static, seq_len=args.seq_len)
    print(f"样本窗口 {X.shape}，站点数 {S.shape[0]}")
    n = len(X)
    tr = int(n * 0.8)
    dev = "cpu"
    model = DownscalerLSTM(S.shape[1], X.shape[2], hidden=args.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    tX, tP, tS, tY = (torch.tensor(a, device=dev) for a in (X[:tr], P[:tr], S[:tr], Y[:tr]))
    vX, vP, vS, vY = (torch.tensor(a, device=dev) for a in (X[tr:], P[tr:], S[tr:], Y[tr:]))

    bs = 64
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tX))
        tot = 0.0
        for i in range(0, len(tX), bs):
            idx = perm[i:i + bs]
            pred = model(tX[idx], tP[idx], tS[idx])
            loss = torch.nn.functional.huber_loss(pred, tY[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if (ep + 1) % max(1, args.epochs // 6) == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                vp = model(vX, vP, vS)
                rmse_model = float(torch.sqrt(((vp - vY) ** 2).mean()))
                rmse_phys = float(torch.sqrt(((vP - vY) ** 2).mean()))
            print(f"epoch {ep+1:>4}/{args.epochs}  train={tot/len(tX):.4f}  "
                  f"验证 RMSE：物理内核 {rmse_phys:.3f} → 残差订正后 {rmse_model:.3f} ℃")

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "n_static": int(S.shape[1]), "n_forcing": int(X.shape[2]),
                "hidden": args.hidden, "seq_len": args.seq_len}, args.out)
    print(f"模型已保存 {args.out}")


if __name__ == "__main__":
    main()
