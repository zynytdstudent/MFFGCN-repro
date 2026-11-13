"""
MFFGCN: Multimodal Feature Fusion Graph Convolution Network
Reproduction of: Zhang et al., "MFFGCN: Multimodal Feature Fusion Graph
Convolution Network for Radio Map Estimation With Uneven Spatial Sampling",
IEEE TMC, vol. 24, no. 12, Dec. 2025.

Components
----------
1. Dual-encoder U-Net (multimodal feature fusion network)
   - observation encoder : sparse measurement map f_i          (1 channel)
   - environment encoder : building map m_i + Tx map p_i        (2 channels)
   - AFM (adaptive feature fusion module) at every encoder stage
2. Graph convolution refine network
   - patch embedding (4 x stride-2 conv)  -> 16x16 = 256 nodes
   - dynamic KNN graph built on guide features
   - 3 x max-relative graph convolution (MRConv, eq. 8-9)
   - Graph2Image (4 x transposed conv) -> residual map
3. Residual output: y_hat = y_coarse + y_residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# basic blocks
# --------------------------------------------------------------------------
class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class EncStage(nn.Module):
    """Two 3x3 conv-bn-relu blocks, as in the RadioUNet-style U-Net."""

    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNReLU(cin, cout),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class DecStage(nn.Module):
    """Upsample + skip concat + conv."""

    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, 2)
        self.body = nn.Sequential(
            ConvBNReLU(cout + cskip, cout),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        return self.body(torch.cat([x, skip], dim=1))


# --------------------------------------------------------------------------
# AFM: adaptive feature fusion module  (eq. 4-5)
# --------------------------------------------------------------------------
class AFM(nn.Module):
    """
    F_e' = W (x) F_e ,  W = softmax(conv3x3(F_e))
    F    = conv3x3(concat(F_e', F_o))

    mode='softmax' keeps the paper's channel-normalised gate (sum over channels
    equals 1). mode='sigmoid' gives an independent 0..1 gate per channel and
    preserves the feature magnitude; provided as an ablation because a
    channel-softmax gate attenuates the environment branch by 1/C.

    Ablation modes of Table V:
        'concat'  plain channel concatenation, no gate at all
        'add'     element-wise sum of the two branches, then 1x1 conv
        'cross'   cross-attention: the observation branch queries the
                  environment branch, and the resulting context is added to the
                  environment feature before fusion
    """

    def __init__(self, c, mode="softmax"):
        super().__init__()
        self.mode = mode
        self.c = c
        if mode in ("softmax", "sigmoid", "spatial"):
            self.gate = nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
            )
            self.fuse = nn.Sequential(
                nn.Conv2d(2 * c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )
        elif mode == "concat":
            self.fuse = nn.Sequential(
                nn.Conv2d(2 * c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )
        elif mode == "add":
            self.mix = nn.Sequential(
                nn.Conv2d(c, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )
        elif mode == "cross":
            self.q = nn.Conv2d(c, c, 1, bias=False)
            self.k = nn.Conv2d(c, c, 1, bias=False)
            self.v = nn.Conv2d(c, c, 1, bias=False)
            self.fuse = nn.Sequential(
                nn.Conv2d(2 * c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"unknown AFM mode {mode}")

    def forward(self, f_obs, f_env):
        if self.mode == "add":
            return self.mix(f_obs + f_env)
        if self.mode == "concat":
            return self.fuse(torch.cat([f_env, f_obs], dim=1))
        if self.mode == "cross":
            b, c, h, w = f_env.shape
            q = self.q(f_obs).mean(dim=(2, 3)).unsqueeze(1)              # B,1,C
            k = self.k(f_env).flatten(2)                                 # B,C,N
            v = self.v(f_env).flatten(2)                                 # B,C,N
            attn = torch.softmax(q @ k / (c ** 0.5), dim=-1)             # B,1,N
            ctx = (attn @ v.transpose(1, 2)).reshape(b, c, 1, 1)
            return self.fuse(torch.cat([f_env + ctx, f_obs], dim=1))
        g = self.gate(f_env)
        if self.mode == "sigmoid":
            w = torch.sigmoid(g)
        elif self.mode == "spatial":
            # softmax over the spatial extent instead of the channel axis.
            # A channel-wise softmax with C=512 gives every channel a mean
            # weight of 1/C, which effectively deletes the environment branch;
            # normalising over H*W keeps the gate at unit magnitude.
            w = torch.softmax(g.flatten(2), dim=-1).view_as(g)
        else:
            w = torch.softmax(g, dim=1)          # paper eq. (4)
        f_env_w = w * f_env
        return self.fuse(torch.cat([f_env_w, f_obs], dim=1))


# --------------------------------------------------------------------------
# multimodal feature fusion network (dual-encoder U-Net)
# --------------------------------------------------------------------------
class DualEncoderUNet(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256), bottleneck=512, out_ch=1,
                 afm_mode="softmax"):
        super().__init__()
        self.afm_mode = afm_mode
        w = list(widths)
        # observation encoder
        self.enc_obs = nn.ModuleList()
        # environment encoder (2 channels in)
        self.enc_env = nn.ModuleList()
        self.afm = nn.ModuleList()

        cin_o, cin_e = 1, 2
        for c in w:
            self.enc_obs.append(EncStage(cin_o, c))
            self.enc_env.append(EncStage(cin_e, c))
            self.afm.append(AFM(c, afm_mode))
            cin_o, cin_e = c, c

        self.pool = nn.MaxPool2d(2)

        self.enc_obs_b = EncStage(cin_o, bottleneck)
        self.enc_env_b = EncStage(cin_e, bottleneck)
        self.afm_b = AFM(bottleneck, afm_mode)

        # decoder
        ws = list(reversed(w))
        self.dec = nn.ModuleList()
        cin = bottleneck
        for c in ws:
            self.dec.append(DecStage(cin, c, c))
            cin = c
        self.head = nn.Conv2d(cin, out_ch, 1)

    def forward(self, obs, env):
        skips = []
        xo, xe = obs, env
        for i in range(len(self.enc_obs)):
            xo = self.enc_obs[i](xo)
            xe = self.enc_env[i](xe)
            skips.append(self.afm[i](xo, xe))
            if i < len(self.enc_obs) - 1:
                xo = self.pool(xo)
                xe = self.pool(xe)

        xo = self.pool(xo)
        xe = self.pool(xe)
        xo = self.enc_obs_b(xo)
        xe = self.enc_env_b(xe)
        x = self.afm_b(xo, xe)

        for i, dec in enumerate(self.dec):
            x = dec(x, skips[-1 - i])
        return torch.sigmoid(self.head(x))


# --------------------------------------------------------------------------
# graph convolution refine network
# --------------------------------------------------------------------------
class MRConv(nn.Module):
    """
    Max-relative graph convolution (eq. 9):
        g(.) = [x_i , max_{j in N(i)} (x_j - x_i)]
        h(.) = g(.) @ W_update
    Operates on dense node features with a kNN neighbour index tensor.
    """

    def __init__(self, cin, cout, act=True):
        super().__init__()
        self.proj_self = nn.Linear(cin, cout, bias=False)
        self.proj_rel = nn.Linear(cin, cout, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x, knn_idx):
        # x: (B, N, C)   knn_idx: (B, N, K) indices of neighbours
        B, N, C = x.shape
        K = knn_idx.shape[-1]
        idx = knn_idx.reshape(B, N * K, 1).expand(-1, -1, C)
        x_nb = torch.gather(x, 1, idx).reshape(B, N, K, C)   # (B,N,K,C)
        rel = x_nb - x.unsqueeze(2)                          # x_j - x_i
        x_rel = rel.max(dim=2).values                        # (B,N,C)
        out = self.proj_self(x) + self.proj_rel(x_rel)
        out = self.bn(out.transpose(1, 2)).transpose(1, 2)
        return self.act(out)


class GraphRefineNet(nn.Module):
    """
    Patch-level dynamic graph refinement with residual output.

    Input : obs (1ch, sparse, zeros elsewhere), coarse (1ch), mask (1ch)
    Output: residual map (1ch) at full resolution
    """

    def __init__(self, dim=256, k=16, gcn_layers=3, patch=16, img=256,
                 gcn_mode="dynamic"):
        super().__init__()
        self.k = k
        self.patch = patch
        self.dim = dim
        self.gcn_mode = gcn_mode
        self.grid = img // patch                     # 16

        # map branch : re-inject measured high-confidence values
        # input = [observation-filled map, observation mask] -> 2 channels
        self.map_branch = nn.Sequential(
            ConvBNReLU(2, 32),
            ConvBNReLU(32, 32),
        )
        # guide branch : semantic cues from the coarse estimation
        self.guide_branch = nn.Sequential(
            ConvBNReLU(1, 32),
            ConvBNReLU(32, 32),
        )
        self.merge = ConvBNReLU(64, 64)

        # patch embedding: 4 conv stride 2, 256 -> 16
        self.patch_embed = nn.Sequential(
            ConvBNReLU(64, 64, 3, 2),
            ConvBNReLU(64, 128, 3, 2),
            ConvBNReLU(128, 192, 3, 2),
            ConvBNReLU(192, dim, 3, 2),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid * self.grid, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.gcn = nn.ModuleList(
            [MRConv(dim, dim) for _ in range(gcn_layers if gcn_mode != "none" else 0)]
        )

        # Graph2Image: 4 transposed conv (k=4, s=2) + 1 conv 3x3
        self.g2i = nn.Sequential(
            nn.ConvTranspose2d(dim, 192, 4, 2, 1), nn.BatchNorm2d(192), nn.ReLU(True),
            nn.ConvTranspose2d(192, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 1, 3, 1, 1),
        )

    # ---- dynamic graph construction (eq. 6) ------------------------------
    @staticmethod
    def build_knn(feat, k):
        """
        feat: (B, N, C) guide features (used for edge construction only).
        Returns knn_idx (B, N, K) excluding self-loop.
        """
        B, N, C = feat.shape
        f = F.normalize(feat.float(), dim=-1)        # fp32 for stable distance
        dist = torch.cdist(f, f)                     # (B,N,N) euclidean
        kk = min(k + 1, N)
        _, idx = torch.topk(dist, kk, dim=-1, largest=False)
        # drop the self neighbour (first element when distance == 0)
        return idx[:, :, 1:k + 1].contiguous()

    def forward(self, obs, coarse, mask):
        # fuse: measured pixels keep the ground-truth observation
        x_in = obs * mask + coarse * (1.0 - mask)

        f_map = self.map_branch(torch.cat([x_in, mask], dim=1))
        f_gui = self.guide_branch(coarse)
        feat = self.merge(torch.cat([f_map, f_gui], dim=1))

        z = self.patch_embed(feat)                   # (B,dim,16,16)
        B, C, H, W = z.shape
        z = z.flatten(2).transpose(1, 2)             # (B,N,C)
        z = z + self.pos_embed

        if self.gcn_mode == "none":
            pass                                     # ablation: w/o GCN
        elif self.gcn_mode == "static":
            # ablation: w/o Dynamic - edges come from the fixed 16x16 grid
            # positions, so the graph is identical for every sample
            knn_idx = self.build_knn(self.pos_embed.expand(B, -1, -1), self.k)
            for layer in self.gcn:
                z = z + layer(z, knn_idx)
        else:
            knn_idx = self.build_knn(z, self.k)      # semantic-similarity graph
            for layer in self.gcn:
                z = z + layer(z, knn_idx)

        z = z.transpose(1, 2).reshape(B, C, H, W)
        return self.g2i(z)


# --------------------------------------------------------------------------
# full model
# --------------------------------------------------------------------------
class MFFGCN(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256), bottleneck=512,
                 gcn_dim=256, k=16, gcn_layers=3, img=256, patch=16,
                 afm_mode="softmax", gcn_mode="dynamic", residual=True):
        super().__init__()
        self.fusion = DualEncoderUNet(widths, bottleneck, afm_mode=afm_mode)
        self.refine = GraphRefineNet(dim=gcn_dim, k=k,
                                     gcn_layers=gcn_layers,
                                     patch=patch, img=img,
                                     gcn_mode=gcn_mode)
        self.residual = residual

    def forward(self, obs, env, mask):
        coarse = self.fusion(obs, env)
        out = self.refine(obs, coarse, mask)
        if not self.residual:
            # ablation "w/o Residual": the graph branch has to produce the
            # whole map on its own instead of only the correction
            fine = torch.sigmoid(out)
            return coarse, fine, fine - coarse
        fine = coarse + out
        return coarse, fine, out


class SingleEncoderUNet(nn.Module):
    """
    Ablation row "Single Encoder" of Table V: early fusion, one encoder over
    the stacked [sparse observation, building map, Tx map] input.

    Kept in the codebase for two reasons: it is part of the paper's ablation,
    and it serves as a sanity baseline - if this cannot generalise either, the
    problem sits in the data pipeline rather than in the MFFGCN design.
    """

    def __init__(self, widths=(32, 64, 128, 256), bottleneck=512, out_ch=1,
                 in_ch=3):
        super().__init__()
        w = list(widths)
        self.enc = nn.ModuleList()
        cin = in_ch
        for c in w:
            self.enc.append(EncStage(cin, c))
            cin = c
        self.pool = nn.MaxPool2d(2)
        self.bott = EncStage(cin, bottleneck)
        self.dec = nn.ModuleList()
        cin = bottleneck
        for c in reversed(w):
            self.dec.append(DecStage(cin, c, c))
            cin = c
        self.head = nn.Conv2d(cin, out_ch, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            skips.append(x)
            if i < len(self.enc) - 1:
                x = self.pool(x)
        x = self.bott(self.pool(x))
        for i, d in enumerate(self.dec):
            x = d(x, skips[-1 - i])
        return torch.sigmoid(self.head(x))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
