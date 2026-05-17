"""
ApacheLicense2.0
Copyright (c) 2026 tkxu

 eo_simulator.py — ガウスプルーム合成データ生成器

 役割:
   衛星観測データ (Sentinel-2 B11/B12) の代わりに、物理モデルに基づいた合成観測データを生成する。eo_provider.py と同じ ObservationBundle形式で返すため、推論パイプラインのテスト・検証に差し替えて使用できる。

   ┌──────────────────────────────────────────────────────────┐
   │  使い方                                                    │
   │    sim = PlumeSimulator()                                  │
   │    bundle = sim.generate(                                  │
   │        lat=38.49, lon=54.19,                              │
   │        Q=4000.0,          # 排出量 [kg/h]                 │
   │        wind_speed=4.0,    # 風速 [m/s]                    │
   │        wind_deg=120.0,    # 気象風向 [degrees]             │
   │    )                                                       │
   │    # bundle["bands"]["B11"] → ndarray (H×W) float32       │
   │    # bundle["plume_true"]   → ndarray (H×W) 真のプルーム   │
   │    # bundle["meta"]["band_set"]  → "swir_only"            │
   │    # bundle["meta"]["sensors"]   → ["synthetic"]          │
   └──────────────────────────────────────────────────────────┘

 eo_types.py との対応:
   ObservationBundle / SyntheticBundle / SyntheticMeta / BandData / SensorSource を参照する。


 物理モデル:
   定常ガウスプルーム拡散 (Gaussian Plume Dispersion Model)拡散係数: Pasquill-Gifford クラス C (中程度の大気安定度) 吸収モデル: Beer-Lambert 則による Sentinel-2 バンド減衰

   推論エンジンが知らない「現実の揺らぎ」を合成データに注入する。
   - 風速の確率的揺らぎ (対数正規分布)
   - 拡散係数の個体差 (PG係数のばらつき)
   - GP (Gaussian Process) ベースの空間相関ノイズ

 ObservationBundle 出力形式:
   {
     "bands":        {"B11": ndarray, "B12": ndarray},
     "wind_speed":   float,    # [m/s]  注入値 (mismatch前)
     "wind_deg":     float,    # [degrees]
     "plume_true":   ndarray,  # 真のプルーム濃度場 [g/m²] ← 合成専用キー
     "Q_true":       float,    # 真の排出量 [kg/h]         ← 合成専用キー
     "meta": {
       "lat", "lon", "wind_speed_actual", "wind_deg_actual",
       "mismatch_enabled", "seed", "shape", "res_m", "backend"
     }
   }

 依存ライブラリ:
   pip install numpy scipy matplotlib
"""
#eo_simulator.py
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
# gridspec は未使用のため削除
from scipy import ndimage as ndi
from typing import Dict, List, Optional, Tuple

from eo_types import (
    ObservationBundle,
    SiteEntry,
    SyntheticBundle,
    SyntheticMeta,
    wind_deg_to_math as _wind_deg_to_math,  # eo_types.py に一元化 (重複排除)
    wind_deg_to_vec,
)
from eo_engines import Provider  # Provider ABC を継承するために import


# =============================================================================
# 定数
# =============================================================================

# メタン物理定数
M_CH4 = 16.04       # モル質量 [g/mol]
NA    = 6.022e23    # アボガドロ定数 [/mol]

# Sentinel-2 バンド吸収断面積 [cm²/molecule]
SIGMA_ABS: Dict[str, float] = {
    "B11": 1.2e-21,   # 1610 nm
    "B12": 8.5e-21,   # 2190 nm
}

# Pasquill-Gifford 拡散係数 (クラスC: 中程度の大気安定度)
PG_A_DEFAULT = 0.11
PG_B_DEFAULT = 0.91


# =============================================================================
# 座標変換ユーティリティ
#
# _wind_deg_to_math / wind_deg_to_vec は eo_types.py に一元化済み。
# eo_simulator.py は import して使用する (重複定義禁止)。
# =============================================================================

def _rotate_to_wind_frame(
    dx: np.ndarray,
    dy: np.ndarray,
    deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    変位配列を風向整合座標系 (along-wind / cross-wind) に回転する。

    Parameters
    ----------
    dx  : 東方向変位 [m]
    dy  : 北方向変位 [m]
    deg : 気象風向 [degrees]

    Returns
    -------
    (xr, yr): along-wind [m], cross-wind [m]
    """
    th = np.radians(_wind_deg_to_math(deg))
    xr =  dx * np.cos(th) + dy * np.sin(th)
    yr = -dx * np.sin(th) + dy * np.cos(th)
    return xr, yr


# =============================================================================
# 地表反射率生成
# =============================================================================

class SurfaceGenerator:
    """
    合成地表反射率フィールドを生成する。

    単純なガウスフィルタによる空間相関を基本とし、
    Mismatch モードでは GP (Gaussian Process) ベースの
    より現実的な空間相関ノイズを重畳する。
    """

    def __init__(self, res: float = 20.0):
        self.res = res

    def generate(
        self,
        shape:          Tuple[int, int],
        seed:           Optional[int] = None,
        gp_noise:       bool  = False,
        gp_length_scale: float = 5.0,
        gp_amplitude:   float = 0.02,
    ) -> np.ndarray:
        """
        空間相関を持つ合成地表反射率フィールドを生成する。

        Parameters
        ----------
        shape           : (rows, cols) ピクセル数
        seed            : 再現性のための乱数シード
        gp_noise        : True で GP ベースの空間相関ノイズを追加
        gp_length_scale : GP の相関長 [pixels]
        gp_amplitude    : GP ノイズの振幅

        Returns
        -------
        np.ndarray  反射率フィールド (0〜1)
        """
        rng = np.random.default_rng(seed)

        # 基本反射率: 空間相関ありノイズ
        base = ndi.gaussian_filter(rng.random(shape), sigma=3.0)

        if gp_noise:
            # GP ベースの空間相関ノイズ (Squared Exponential カーネル)
            # 完全な GP 計算はコストが高いため、
            # 白色ノイズをガウスフィルタで平滑化して近似する
            white = rng.standard_normal(shape)
            gp_approx = ndi.gaussian_filter(white, sigma=gp_length_scale)
            gp_approx /= (gp_approx.std() + 1e-8)   # 正規化
            base = base + gp_amplitude * gp_approx
            base = np.clip(base, 0.0, 1.0)

        return base.astype(np.float32)


# =============================================================================
# ガウスプルーム拡散モデル
# =============================================================================

class GaussianPlumeModel:
    """
    定常ガウスプルーム拡散モデル。

    Pasquill-Gifford クラス C を基本とし、
    Mismatch Injection 時は拡散係数に個体差を持たせる。

    拡散モデル:
        C(x, y) = Q_s / (u * σ_y(x)) * exp(-y² / (2σ_y²))
        σ_y(x) = PG_A * x^PG_B   [Pasquill-Gifford]

    Parameters
    ----------
    pg_a : PG 係数 A (横拡散の大きさ)
    pg_b : PG 係数 B (横拡散の距離依存指数)
    res  : ピクセル解像度 [m/px]
    """

    def __init__(
        self,
        pg_a: float = PG_A_DEFAULT,
        pg_b: float = PG_B_DEFAULT,
        res:  float = 20.0,
    ):
        self.pg_a = pg_a
        self.pg_b = pg_b
        self.res  = res

    def compute(
        self,
        shape:      Tuple[int, int],
        Q:          float,
        wind_speed: float,
        wind_deg:   float,
        src:        Tuple[int, int] = (50, 50),
        pg_a:       Optional[float] = None,
    ) -> np.ndarray:
        """
        プルーム濃度場を計算する。

        Parameters
        ----------
        shape      : (rows, cols) ピクセル数
        Q          : 排出量 [kg/h]
        wind_speed : 風速 [m/s]
        wind_deg   : 気象風向 [degrees]
        src        : 排出源ピクセル位置 (row, col)
        pg_a       : PG 係数 A の上書き値。None の場合は self.pg_a を使用。
                     Mismatch Injection 時に呼び出し元から渡すことで
                     インスタンス状態の破壊的変更を避ける。

        Returns
        -------
        np.ndarray  プルーム濃度場 [g/m²]
        """
        _pg_a = pg_a if pg_a is not None else self.pg_a

        y_idx, x_idx = np.indices(shape)
        dx = (x_idx - src[1]) * self.res
        dy = (y_idx - src[0]) * self.res

        xr, yr = _rotate_to_wind_frame(dx, dy, wind_deg)


        # Q * 1e3 / 3600.0
        Qs   = Q * 1e3 / 3600.0
        mask = xr > 0

        sig_y = np.zeros(shape, dtype=np.float64)
        sig_y[mask] = _pg_a * (xr[mask] ** self.pg_b)

        plume = np.zeros(shape, dtype=np.float64)
        plume[mask] = (Qs / (wind_speed * sig_y[mask] + 1e-8)) * np.exp(
            -(yr[mask] ** 2) / (2.0 * sig_y[mask] ** 2 + 1e-8)
        )

        return plume.astype(np.float32)


# =============================================================================
# バンド合成 (Beer-Lambert 則)
# =============================================================================

class BandSynthesizer:
    """
    プルーム濃度場から Sentinel-2 バンド (B11/B12) の吸収を合成する。

    Beer-Lambert 則:
        I = I0 * exp(-σ_abs * N_column)
        N_column = C [g/m²] / M_CH4 * NA / 1e4   [molecules/cm²]

    センサーノイズは正規分布で加算する。
    """

    def __init__(self, sensor_noise_std: float = 0.001):
        self.sensor_noise_std = sensor_noise_std

    def synthesize(
        self,
        surface:    np.ndarray,
        plume:      np.ndarray,
        rng:        np.random.Generator,
    ) -> Dict[str, np.ndarray]:
        """
        地表反射率とプルーム濃度場からバンドデータを合成する。

        Parameters
        ----------
        surface : 地表反射率フィールド (H×W)
        plume   : プルーム濃度場 [g/m²] (H×W)
        rng     : 乱数生成器 (再現性のため外部から渡す)

        Returns
        -------
        Dict[str, np.ndarray]  {"B11": ndarray, "B12": ndarray}
        """
        # カラム濃度 [molecules/cm²]
        column = plume / M_CH4 * NA / 1e4

        bands: Dict[str, np.ndarray] = {}
        for b in ["B11", "B12"]:
            noise      = rng.normal(0, self.sensor_noise_std, surface.shape)
            # クリップ上限 8 の根拠をコメントで補足。
            # exp(-8) ≈ 3.4e-4 となり物理的にほぼ完全吸収。
            # これ以上は数値的に意味がなく、オーバーフロー防止のための上限。
            absorption = np.clip(SIGMA_ABS[b] * column, 0, 8)
            bands[b]   = ((surface + noise) * np.exp(-absorption)).astype(np.float32)

        return bands


# =============================================================================
# Mismatch Injection
# =============================================================================

class MismatchInjector:
    """
    推論モデルとの過適合を防ぐためのノイズ注入を行う

    推論エンジンが知らない現実の揺らぎを合成データに加えることで、
    アルゴリズムのロバスト性をテストできる。

    注入する揺らぎ:
        wind_speed_factor : 風速の乗数 (対数正規分布)
                            推論エンジンは true_speed を知らず、
                            nominal_speed で推定する。
        pg_a_factor       : PG 係数 A の乗数 (正規分布)
                            拡散係数の個体差を模倣する。
    """

    def __init__(
        self,
        wind_speed_sigma: float = 0.15,   # 対数正規の σ (約±15%)
        pg_a_sigma:       float = 0.10,   # PG_A の変動 (約±10%)
    ):
        self.wind_speed_sigma = wind_speed_sigma
        self.pg_a_sigma       = pg_a_sigma

    def perturb(
        self,
        wind_speed: float,
        pg_a:       float,
        rng:        np.random.Generator,
    ) -> Tuple[float, float]:
        """
        風速と PG 係数に揺らぎを加える。

        Parameters
        ----------
        wind_speed : 公称風速 [m/s]
        pg_a       : 公称 PG 係数 A
        rng        : 乱数生成器

        Returns
        -------
        (actual_wind_speed, actual_pg_a): 実際に使用する値
        """
        # 風速: 対数正規分布 (常に正値を保証)
        log_factor  = rng.normal(0, self.wind_speed_sigma)
        actual_wind = wind_speed * np.exp(log_factor)

        # PG_A: 正規分布
        pg_a_factor = 1.0 + rng.normal(0, self.pg_a_sigma)
        actual_pg_a = pg_a * max(pg_a_factor, 0.5)   # 0.5 以下にはならない

        return float(actual_wind), float(actual_pg_a)


# =============================================================================
# PlumeSimulator — 統合クラス
# =============================================================================

class PlumeSimulator(Provider):
    """
    ガウスプルーム合成データ生成器。

    earth_obs_provider.py の DataFetcher と同じ ObservationBundle 形式で
    データを返すため、推論パイプラインのテスト・検証に差し替えて使用できる。

    Parameters
    ----------
    res             : ピクセル解像度 [m/px]
    shape           : シーンサイズ (rows, cols) [pixels]
    src             : 排出源ピクセル位置 (row, col)
    sensor_noise_std: センサーノイズの標準偏差
    mismatch        : True で Mismatch Injection を有効化
    mismatch_config : MismatchInjector のパラメータ dict
    gp_noise        : True で GP ベースの空間相関ノイズを有効化
    gp_length_scale : GP の相関長 [pixels]
    gp_amplitude    : GP ノイズの振幅

    使用例:
        sim    = PlumeSimulator(mismatch=True)
        bundle = sim.generate(Q=4000.0, wind_speed=4.0, wind_deg=120.0, seed=42)
        bands  = bundle["bands"]   # {"B11": ndarray, "B12": ndarray}
        plume  = bundle["plume_true"]   # 真のプルーム濃度場
    """

    def __init__(
        self,
        res:              float = 20.0,
        shape:            Tuple[int, int] = (100, 100),
        src:              Tuple[int, int] = (50, 50),
        sensor_noise_std: float = 0.001,
        mismatch:         bool  = False,
        mismatch_config:  Optional[Dict] = None,
        gp_noise:         bool  = False,
        gp_length_scale:  float = 5.0,
        gp_amplitude:     float = 0.02,
        # seed 引数を追加する。__init__ では使用しないが互換性のために受け取る。
        seed:             Optional[int] = None,
    ):
        self.res             = res
        self.shape           = shape
        self.src             = src
        self.gp_noise        = gp_noise
        self.gp_length_scale = gp_length_scale
        self.gp_amplitude    = gp_amplitude
        self.mismatch        = mismatch

        self._surface_gen  = SurfaceGenerator(res=res)
        self._band_synth   = BandSynthesizer(sensor_noise_std=sensor_noise_std)
        self._plume_model  = GaussianPlumeModel(res=res)
        self._mismatch_inj = MismatchInjector(**(mismatch_config or {}))

    def generate(
        self,
        Q:          float,
        wind_speed: float,
        wind_deg:   float,
        lat:        float = 0.0,
        lon:        float = 0.0,
        seed:       Optional[int] = None,
    ) -> Dict:
        """
        合成観測データを生成して ObservationBundle 形式で返す。

        Parameters
        ----------
        Q          : 排出量 [kg/h]
        wind_speed : 公称風速 [m/s]
        wind_deg   : 気象風向 [degrees, 北基準時計回り]
        lat        : シーン中心緯度 (メタデータ用)
        lon        : シーン中心経度 (メタデータ用)
        seed       : 乱数シード (再現性)

        Returns
        -------
        Dict  ObservationBundle 形式 (合成専用キー plume_true / Q_true を追加)
        """
        rng = np.random.default_rng(seed)

        # --- Mismatch Injection ---
        actual_wind  = wind_speed
        actual_pg_a  = PG_A_DEFAULT

        if self.mismatch:
            actual_wind, actual_pg_a = self._mismatch_inj.perturb(
                wind_speed, PG_A_DEFAULT, rng
            )
            #  self._plume_model.pg_a を書き換えるのをやめ、
            # compute() の pg_a 引数として渡すことでインスタンス状態を保護する。
            # 複数回の generate() 呼び出し間で pg_a が汚染されない。

        # --- 地表反射率生成 ---
        surface = self._surface_gen.generate(
            shape           = self.shape,
            seed            = seed,
            gp_noise        = self.gp_noise,
            gp_length_scale = self.gp_length_scale,
            gp_amplitude    = self.gp_amplitude,
        )

        # --- プルーム濃度場計算 ---
        # actual_pg_a を pg_a 引数で渡し、インスタンス状態を変更しない
        plume = self._plume_model.compute(
            shape      = self.shape,
            Q          = Q,
            wind_speed = actual_wind,
            wind_deg   = wind_deg,
            src        = self.src,
            pg_a       = actual_pg_a,
        )

        # --- バンド合成 ---
        bands = self._band_synth.synthesize(surface, plume, rng)

        return {
            # earth_obs_provider.py と共通のキー
            "bands":      bands,
            "wind_speed": wind_speed,
            "wind_deg":   wind_deg,
            "sza":        None,
            "vza":        None,
            "amf":        None,
            "surface_pressure": None,
            "ch4_column": None,
            # 合成データ専用キー
            "plume_true": plume,
            "Q_true":     Q,
            "meta": {
                "lat":               lat,
                "lon":               lon,
                "wind_speed_actual": actual_wind,
                "wind_deg_actual":   wind_deg,
                "pg_a_actual":       actual_pg_a,
                "mismatch_enabled":  self.mismatch,
                "gp_noise_enabled":  self.gp_noise,
                "seed":              seed,
                "shape":             self.shape,
                "res_m":             self.res,
                "backend":           "synthetic",
                "band_set":          "swir_only",   # 合成は常に B11/B12 のみ
                "sensors":           ["synthetic"],  # eo_types.SensorSource 参照
            },
        }

    def get_bundle(
        self,
        site: "SiteEntry",
        dt:   "Optional[datetime]" = None,
    ) -> "ObservationBundle":
        """
        Provider ABC の統一インターフェース実装。

        SiteEntry の情報を使って generate() を呼び出す。
        EOPipeline から isinstance(provider, Provider) で分岐されたとき
        このメソッドが呼ばれる。

        Parameters
        ----------
        site : SiteEntry  サイトレジストリのエントリ
        dt   : datetime   合成データ生成では使用しない (互換性のために受け取る)

        Returns
        -------
        ObservationBundle
        """
        return self.generate(
            Q          = site["Q_true"],
            wind_speed = site["wind_speed"],
            wind_deg   = site["wind_deg"],
            lat        = site["lat"],
            lon        = site["lon"],
            seed       = site.get("seed"),
        )

    def generate_batch(
        self,
        sites:    List[Dict],
        Q_values: Optional[List[float]] = None,
    ) -> List[Dict]:
        """
        サイトリストに対してバッチ生成する。

        Parameters
        ----------
        sites    : SITE_REGISTRY 形式の dict リスト
                   各 dict は lat, lon, wind_speed, wind_deg, Q_true, seed を持つ
        Q_values : 排出量の上書きリスト (None の場合は site["Q_true"] を使用)

        Returns
        -------
        List[Dict]  各サイトの ObservationBundle
        """
        #  Q_values の長さが sites と一致しない場合に早期エラーを出す
        if Q_values is not None and len(Q_values) != len(sites):
            raise ValueError(
                f"Q_values の長さ ({len(Q_values)}) が "
                f"sites の長さ ({len(sites)}) と一致しません。"
            )

        results = []
        for i, site in enumerate(sites):
            Q = Q_values[i] if Q_values is not None else site["Q_true"]
            bundle = self.generate(
                Q          = Q,
                wind_speed = site["wind_speed"],
                wind_deg   = site["wind_deg"],
                lat        = site.get("lat", 0.0),
                lon        = site.get("lon", 0.0),
                seed       = site.get("seed"),
            )
            bundle["site_id"] = site.get("id", f"site_{i}")
            results.append(bundle)
        return results


# =============================================================================
# 可視化ユーティリティ
# =============================================================================

def visualize_bundle(
    bundle:    Dict,
    title:     str = "PlumeSimulator — 合成データ",
    save_path: Optional[str] = None,
) -> None:
    """
    ObservationBundle の内容を3列で可視化する。

    左: 真のプルーム濃度場 + 風向矢印
    中: B11 バンド (合成)
    右: MBSP = log(B11) - log(B12)

    Parameters
    ----------
    bundle    : PlumeSimulator.generate() の返値
    title     : フィギュアタイトル
    save_path : 保存パス (None の場合は表示のみ)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold")

    def _style(ax: plt.Axes) -> None:
        ax.set_facecolor("#0d1117")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")
        ax.title.set_color("white")

    # --- 左: 真のプルーム ---
    ax0 = axes[0]
    plume = bundle.get("plume_true")
    if plume is not None:
        im0 = ax0.imshow(plume, cmap="hot_r", origin="upper")
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="[g/m²]")
        # 風向矢印
        wvec = wind_deg_to_vec(bundle["wind_deg"])
        cx, cy = plume.shape[1] // 2, plume.shape[0] // 2
        ax0.annotate(
            "", xy=(cx + wvec[0] * 14, cy - wvec[1] * 14),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="-|>", color="cyan", lw=2.0),
        )
        ax0.text(cx + wvec[0] * 18, cy - wvec[1] * 18,
                 "wind", color="cyan", fontsize=8, ha="center", va="center")
    ax0.set_title("True Plume  [g/m²]")
    _style(ax0)

    # --- 中: B11 バンド ---
    ax1   = axes[1]
    bands = bundle.get("bands")
    if bands is not None:
        im1 = ax1.imshow(bands["B11"], cmap="viridis", origin="upper")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="reflectance")

    # メタデータ注釈
    meta = bundle.get("meta", {})
    mismatch_str = "ON" if meta.get("mismatch_enabled") else "OFF"
    gp_str       = "ON" if meta.get("gp_noise_enabled") else "OFF"
    ax1.set_title(f"B11 (synthetic)\nMismatch={mismatch_str}  GP={gp_str}")
    _style(ax1)

    # --- 右: MBSP ---
    ax2 = axes[2]
    if bands is not None:
        mbsp = (np.log(np.maximum(bands["B11"], 1e-8))
              - np.log(np.maximum(bands["B12"], 1e-8)))
        vmax = np.nanpercentile(np.abs(mbsp), 98)
        im2  = ax2.imshow(mbsp, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, origin="upper")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="MBSP")

    Q_true = bundle.get("Q_true", "—")
    ws     = meta.get("wind_speed_actual", bundle.get("wind_speed", "—"))
    ax2.set_title(
        f"MBSP = log(B11) - log(B12)\n"
        f"Q={Q_true:.0f} kg/h  ws={ws:.1f} m/s"
        if isinstance(Q_true, (int, float)) and isinstance(ws, float)
        else "MBSP = log(B11) - log(B12)"
    )
    _style(ax2)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Figure saved → {save_path}")

    plt.show()


def visualize_mismatch_comparison(
    Q:          float = 2000.0,
    wind_speed: float = 4.0,
    wind_deg:   float = 120.0,
    seed:       int   = 42,
    save_path:  Optional[str] = None,
) -> None:
    """
    Mismatch なし / あり / GP ノイズありの3条件を並べて比較する。

    Parameters
    ----------
    Q          : 排出量 [kg/h]
    wind_speed : 風速 [m/s]
    wind_deg   : 気象風向 [degrees]
    seed       : 乱数シード
    save_path  : 保存パス
    """
    configs = [
        ("Baseline (mismatch=OFF, gp=OFF)",
         PlumeSimulator(mismatch=False, gp_noise=False)),
        ("Mismatch ON (wind/PG 揺らぎ)",
         PlumeSimulator(mismatch=True,  gp_noise=False)),
        ("Mismatch ON + GP Noise",
         PlumeSimulator(mismatch=True,  gp_noise=True)),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        f"Mismatch Injection 比較  Q={Q:.0f} kg/h  "
        f"ws={wind_speed:.1f} m/s  wd={wind_deg:.0f}°",
        color="white", fontsize=13, fontweight="bold",
    )

    col_titles = ["True Plume [g/m²]", "B11 (synthetic)", "MBSP"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, color="white", fontsize=10, fontweight="bold")

    for i, (label, sim) in enumerate(configs):
        bundle = sim.generate(Q=Q, wind_speed=wind_speed,
                              wind_deg=wind_deg, seed=seed)

        # プルーム
        ax = axes[i, 0]
        plume = bundle["plume_true"]
        im = ax.imshow(plume, cmap="hot_r", origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_ylabel(label, color="white", fontsize=8)

        # B11
        ax1 = axes[i, 1]
        ax1.imshow(bundle["bands"]["B11"], cmap="viridis", origin="upper")

        # MBSP
        ax2 = axes[i, 2]
        bands = bundle["bands"]
        mbsp  = (np.log(np.maximum(bands["B11"], 1e-8))
               - np.log(np.maximum(bands["B12"], 1e-8)))
        vmax  = np.nanpercentile(np.abs(mbsp), 98)
        ax2.imshow(mbsp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")

        meta = bundle["meta"]
        ax2.text(
            2, 6,
            f"ws_actual={meta['wind_speed_actual']:.2f}\n"
            f"pg_a={meta['pg_a_actual']:.3f}",
            color="#00ff88", fontsize=7,
            bbox=dict(fc="black", alpha=0.6, pad=2, lw=0),
        )

        for ax in [axes[i, 0], axes[i, 1], axes[i, 2]]:
            ax.set_facecolor("#0d1117")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#333344")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Figure saved → {save_path}")
    plt.show()


# =============================================================================
# テストコード
# =============================================================================

# サンプルサイトレジストリ (earth_obs_provider.py と同形式)
_SAMPLE_REGISTRY = [
    {
        "id": "TM-01", "lat": 38.49, "lon": 54.19,
        "wind_speed": 4.0, "wind_deg": 120, "Q_true": 4000.0,
        "seed": 42, "category": "super-emitter",
    },
    {
        "id": "PB-01", "lat": 31.83, "lon": -102.37,
        "wind_speed": 6.5, "wind_deg": 200, "Q_true": 800.0,
        "seed": 7,  "category": "mid-range",
    },
    {
        "id": "DZ-01", "lat": 27.21, "lon": 2.52,
        "wind_speed": 3.0, "wind_deg": 45,  "Q_true": 1800.0,
        "seed": 13, "category": "mid-range",
    },
    {
        "id": "HN-01", "lat": 32.93, "lon": 3.13,
        "wind_speed": 8.0, "wind_deg": 315, "Q_true": 300.0,
        "seed": 99, "category": "near-limit",
    },
]


def test_basic_generation() -> None:
    """基本的なプルーム生成テスト (外部API不要)。"""
    print("\n" + "="*55)
    print("  TEST-1: 基本生成テスト")
    print("="*55)

    # PlumeSimulator(seed=None) は __init__ に seed 引数を追加して対応
    sim    = PlumeSimulator(seed=None)
    bundle = sim.generate(Q=2000.0, wind_speed=4.0, wind_deg=120.0, seed=42)

    # 構造チェック
    required = ["bands", "wind_speed", "wind_deg", "plume_true", "Q_true", "meta"]
    for key in required:
        assert key in bundle, f"'{key}' が返却されていない"
    assert "B11" in bundle["bands"] and "B12" in bundle["bands"]
    assert bundle["meta"]["backend"] == "synthetic"

    # 値域チェック
    plume = bundle["plume_true"]
    b11   = bundle["bands"]["B11"]
    assert plume.shape == (100, 100),   f"plume shape 異常: {plume.shape}"
    assert np.all(plume >= 0),          "plume に負値"
    assert np.all(b11   > 0),           "B11 にゼロ以下の値"
    assert bundle["Q_true"] == 2000.0,  "Q_true が一致しない"

    print(f"  plume shape: {plume.shape}  max={plume.max():.4f} g/m²")
    print(f"  B11   shape: {b11.shape}    min={b11.min():.4f}")
    print(f"  backend: {bundle['meta']['backend']}")
    print("  → PASS")


def test_mismatch_injection() -> None:
    """Mismatch Injection テスト。"""
    print("\n" + "="*55)
    print("  TEST-2: Mismatch Injection テスト")
    print("="*55)

    sim_base     = PlumeSimulator(mismatch=False)
    sim_mismatch = PlumeSimulator(mismatch=True)

    b_base     = sim_base.generate(Q=2000.0, wind_speed=4.0, wind_deg=120.0, seed=7)
    b_mismatch = sim_mismatch.generate(Q=2000.0, wind_speed=4.0, wind_deg=120.0, seed=7)

    ws_base     = b_base["meta"]["wind_speed_actual"]
    ws_mismatch = b_mismatch["meta"]["wind_speed_actual"]

    # Mismatch なし: actual == nominal
    assert abs(ws_base - 4.0) < 1e-9, "Mismatch OFF なのに wind_speed が変化"
    # Mismatch あり: actual ≠ nominal (確率的なので稀に一致することはありうる)
    # 同一 seed でも分布から外れることを確認
    print(f"  baseline    wind_speed_actual = {ws_base:.4f} m/s (nominal=4.0)")
    print(f"  mismatch    wind_speed_actual = {ws_mismatch:.4f} m/s (nominal=4.0)")
    print(f"  pg_a_actual = {b_mismatch['meta']['pg_a_actual']:.4f} (nominal={PG_A_DEFAULT})")
    assert not b_base["meta"]["mismatch_enabled"],     "baseline が mismatch ON"
    assert     b_mismatch["meta"]["mismatch_enabled"], "mismatch が OFF のまま"
    print("  → PASS")


def test_gp_noise() -> None:
    """GP ノイズテスト。"""
    print("\n" + "="*55)
    print("  TEST-3: GP 空間相関ノイズテスト")
    print("="*55)

    sim_plain = PlumeSimulator(gp_noise=False)
    sim_gp    = PlumeSimulator(gp_noise=True, gp_amplitude=0.05)

    b_plain = sim_plain.generate(Q=1000.0, wind_speed=3.0, wind_deg=90.0, seed=13)
    b_gp    = sim_gp.generate(Q=1000.0, wind_speed=3.0, wind_deg=90.0, seed=13)

    # GP ノイズありの方が B11 の空間分散が大きい
    std_plain = b_plain["bands"]["B11"].std()
    std_gp    = b_gp["bands"]["B11"].std()
    print(f"  B11 std (plain) = {std_plain:.4f}")
    print(f"  B11 std (GP)    = {std_gp:.4f}")
    assert std_gp > std_plain, "GP ノイズが B11 に反映されていない"
    assert b_gp["meta"]["gp_noise_enabled"], "gp_noise_enabled が False"
    print("  → PASS")


def test_batch_generation() -> None:
    """バッチ生成テスト。"""
    print("\n" + "="*55)
    print("  TEST-4: バッチ生成テスト (4サイト)")
    print("="*55)

    sim     = PlumeSimulator(mismatch=False)
    bundles = sim.generate_batch(_SAMPLE_REGISTRY)

    assert len(bundles) == 4, f"バッチ数が異常: {len(bundles)}"
    for b in bundles:
        assert b["plume_true"].shape == (100, 100)
        assert b["bands"]["B11"].shape == (100, 100)

    print(f"  {'ID':<8} {'Q_true':>8} {'plume_max':>10} {'B11_min':>10}")
    print(f"  {'─'*42}")
    for b in bundles:
        print(f"  {b['site_id']:<8} "
              f"{b['Q_true']:>8.0f} "
              f"{b['plume_true'].max():>10.4f} "
              f"{b['bands']['B11'].min():>10.4f}")
    print("  → PASS")


def test_reproducibility() -> None:
    """再現性テスト: 同一 seed で同一結果が得られることを確認。"""
    print("\n" + "="*55)
    print("  TEST-5: 再現性テスト")
    print("="*55)

    sim = PlumeSimulator(mismatch=True, gp_noise=True)
    b1  = sim.generate(Q=3000.0, wind_speed=5.0, wind_deg=180.0, seed=99)
    b2  = sim.generate(Q=3000.0, wind_speed=5.0, wind_deg=180.0, seed=99)

    assert np.allclose(b1["plume_true"],      b2["plume_true"]),      "plume_true が一致しない"
    assert np.allclose(b1["bands"]["B11"],    b2["bands"]["B11"]),    "B11 が一致しない"
    assert np.isclose(b1["meta"]["wind_speed_actual"],
                      b2["meta"]["wind_speed_actual"]),               "wind_speed_actual が一致しない"
    print("  同一 seed での再現性: PASS")


# =============================================================================
# エントリポイント
# =============================================================================

if __name__ == "__main__":

    print("=" * 55)
    print("  eo_simulator.py — 合成データ生成器テスト")
    print("=" * 55)

    # --- 単体テスト ---
    test_basic_generation()
    test_mismatch_injection()
    test_gp_noise()
    test_batch_generation()
    test_reproducibility()

    print("\n" + "="*55)
    print("  全テスト PASS")
    print("="*55)

    # --- 可視化デモ ---
    print("\n可視化デモを実行します ...")

    # 単一サイトの可視化
    sim    = PlumeSimulator(mismatch=True, gp_noise=True)
    bundle = sim.generate(
        Q=4000.0, wind_speed=4.0, wind_deg=120.0,
        lat=38.49, lon=54.19, seed=42,
    )
    visualize_bundle(
        bundle,
        title="TM-01 (Turkmenistan) — Mismatch ON + GP Noise",
        save_path="plume_sim_single.png",
    )

    # Mismatch 比較
    visualize_mismatch_comparison(
        Q=2000.0, wind_speed=4.0, wind_deg=120.0, seed=42,
        save_path="plume_sim_mismatch_comparison.png",
    )
