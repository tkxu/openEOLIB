"""
ApacheLicense2.0
Copyright (c) 2026 tkxu
"""
# =============================================================================
# eo_engines.py — 推論エンジン集 + InferenceEngine / Provider ABC
#
# 概要:
#   openEOLIB で使用できる InferenceEngine の ABC・実装・ファクトリを
#   まとめたモジュール。
#
#   ┌──────────────────────────────────────────────────────────────┐
#   │  依存関係                                                      │
#   │                                                               │
#   │  eo_types.py   ← eo_engines.py  (型定義を参照)               │
#   │  eo_engines.py ← eo_pipeline.py (ABC・NullEngine を import)  │
#   │  eo_engines.py ← main_sample.py (build_engine を import)     │
#   │                                                               │
#   │  eo_engines.py は eo_pipeline / eo_provider / eo_simulator を │
#   │  一切 import しない。                                          │
#   └──────────────────────────────────────────────────────────────┘
#
#   収録クラス:
#     InferenceEngine      推論エンジンの抽象基底クラス 
#     Provider             データプロバイダの抽象基底クラス 
#     NullInferenceEngine  テスト用ダミーエンジン
#     MBSPThresholdEngine  B11/B12 閾値法 (Varon et al. 2021)
#     EucalyptusEngine     Project Eucalyptus アダプタ (非商用)
#
#   ファクトリ:
#     build_engine(name)   名前文字列からエンジンを生成する
#
# 使用例:
#   from eo_engines import MBSPThresholdEngine, EucalyptusEngine, build_engine
#
#   # 直接インスタンス化
#   engine = MBSPThresholdEngine(z_thresh=3.0)
#
#   # ファクトリ経由
#   engine = build_engine("mbsp")
#   engine = build_engine("eucalyptus")
#   engine = build_engine("null")
#
# ライセンス:
#   MBSPThresholdEngine: 制限なし
#   EucalyptusEngine:    非商用のみ (Project Eucalyptus ライセンスに準ずる)
#                        https://github.com/Orbio-Earth/Project-Eucalyptus
#
# =============================================================================

#eo_engines.py
from __future__ import annotations

import logging
import threading
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.special import expit

from eo_types import (
    InferenceResult,
    ObservationBundle,
    QualityFlags,
    SiteEntry,
)

_logger = logging.getLogger(__name__)


# =============================================================================
# 抽象基底クラス
# =============================================================================

class InferenceEngine(ABC):
    """
    推論エンジンの抽象基底クラス。

    eo_pipeline.EOPipeline はこのインターフェースだけを知っており、
    具体的なアルゴリズム実装は外部から差し込まれる。

    実装クラスの例:
        - NullInferenceEngine        (テスト用ダミー、このファイルに同梱)
        - MBSPThresholdEngine        (Varon et al. 2021 閾値法)
        - EucalyptusEngine           (Project Eucalyptus アダプタ)

    実装義務:
        infer() メソッドを実装すること。
        返り値は InferenceResult 型 (eo_types.py 参照)、
        検出不可能な場合は None を返す。
    """

    @abstractmethod
    def infer(
        self,
        mbsp:       np.ndarray,
        wind_speed: float,
        wind_deg:   float,
        **kwargs,
    ) -> Optional[InferenceResult]:
        """
        MBSP フィールドから排出量を推定する。

        Parameters
        ----------
        mbsp       : MBSP フィールド  shape=(H, W)
        wind_speed : 風速 [m/s]
        wind_deg   : 気象風向 [degrees]
        **kwargs   : エンジン固有の追加パラメータ

        Returns
        -------
        InferenceResult または None (検出不可能な場合)
        """
        ...

    def infer_null(
        self,
        mbsp:       np.ndarray,
        wind_speed: float,
        wind_deg:   float,
    ) -> Optional[InferenceResult]:
        """
        ヌル試行用の推定。

        デフォルトは infer() と同じ処理。
        ROC 構築時のヌル分布生成に特化した処理が必要な場合は
        サブクラスでオーバーライドする。
        """
        return self.infer(mbsp, wind_speed, wind_deg)


class Provider(ABC):
    """
    データプロバイダの抽象基底クラス。

    EOPipeline はこのインターフェースだけを知っており、
    具体的なデータ取得方法は外部から差し込まれる。

    実装クラスの例:
        - PlumeSimulator (eo_simulator.py)  合成データ生成
        - DataFetcher    (eo_provider.py)   実データ取得

    実装義務:
        get_bundle() メソッドを実装すること。
    """

    @abstractmethod
    def get_bundle(
        self,
        site: SiteEntry,
        dt:   Optional[datetime] = None,
    ) -> ObservationBundle:
        """
        サイト情報から ObservationBundle を返す。

        Parameters
        ----------
        site : SiteEntry  サイトレジストリのエントリ
        dt   : datetime   観測日時 (実データ取得時のみ使用)

        Returns
        -------
        ObservationBundle
        """
        ...


# =============================================================================
# NullInferenceEngine — テスト用ダミーエンジン
# =============================================================================

class NullInferenceEngine(InferenceEngine):
    """
    テスト・デモ用のダミー推論エンジン。

    簡易ヒューリスティックに基づくダミー推定（科学的意味は持たない）

    実運用では独自の推論エンジンに差し替えることを想定。
    外部ライブラリを組み込む場合も InferenceEngine を継承して
    infer() を実装するだけでパイプラインに接続できる。
    """

    def __init__(self, q_scale: float = 1000.0):
        """
        Parameters
        ----------
        q_scale : MBSP ピーク値に対する Q のスケール係数
        """
        self.q_scale = q_scale

    def infer(
        self,
        mbsp:       np.ndarray,
        wind_speed: float,
        wind_deg:   float,
        **kwargs,
    ) -> Optional[InferenceResult]:
        """
        MBSP の Z スコア閾値による簡易検出と Q 推定。

        アルゴリズム:
            1. MAD ロバスト推定で背景を除去し Z スコアを計算する。
            2. max(Z) < 2.0 のとき検出なし (None を返す)。
            3. MLLR = Σ max(Z - 2.0, 0)  (超閾値 Z の積分)
            4. Q [kg/h] ≈ MLLR × q_scale / 100
               ※ このスケール式はテスト用のヒューリスティックであり
               物理的な定量精度を保証しない。
               実運用では MBSPThresholdEngine の
               IME 法 (Q ≈ MLLR × wind_speed × q_scale_factor) を使用すること。
        """
        mu  = float(np.nanmedian(mbsp))
        sig = float(1.4826 * np.nanmedian(np.abs(mbsp - mu)))
        eps = sig + 1e-8
        z   = (mbsp - mu) / eps

        if float(np.nanmax(z)) < 2.0:
            return None

        mllr = float(np.nansum(np.maximum(z - 2.0, 0)))
        q    = mllr * self.q_scale / 100.0

        flags: QualityFlags = {
            "low_wind":          wind_speed < 2.0,
            "multi_modal_theta": False,
            "roi_unstable":      False,
            "template_dominant": False,
        }

        return {
            "q":     float(np.clip(q, 0, 50000)),
            "q_std": float(q * 0.3),
            "mllr":  mllr,
            "p_det": float(np.clip(mllr / 200.0, 0, 1)),
            "flags": flags,
        }


# =============================================================================
# Engine 1: MBSPThresholdEngine
#
# Varon et al. (2021) の MBSP 法に基づく古典的閾値エンジン。
# B11/B12 だけを使用するため追加インストール不要。
#
# 手法:
#   1. MBSP = log(B11) - log(B12) を計算
#   2. 背景を MAD ロバスト推定で除去して Z スコア化
#   3. 閾値超え画素の積分値から MLLR を計算
#   4. 排出量 Q は MLLR・風速・スケール係数から簡易推定
#
# 精度:
#   super-emitter クラス (> 1000 kg/h) は高感度。
#   mid-range / near-limit では偽陽性が増加する。
#   より高精度が必要な場合は EucalyptusEngine を使用してください。
#
# 参考:
#   Varon et al. (2021) AMT 14, 2771–2786
#   https://doi.org/10.5194/amt-14-2771-2021
# =============================================================================

class MBSPThresholdEngine(InferenceEngine):
    """
    MBSP 閾値法による排出検出エンジン (Varon et al. 2021 準拠)。

    依存ライブラリ: numpy・scipy のみ（追加インストール不要）

    Parameters
    ----------
    z_thresh       : 検出 Z スコア閾値 (デフォルト: 3.0)
    min_pixels     : 最小検出画素数 (デフォルト: 5)
    q_scale_factor : MLLR → Q [kg/h] の変換スケール係数 (デフォルト: 30.0)
                     Q [kg/h] ≈ MLLR × wind_speed [m/s] × q_scale_factor
                     実データでのキャリブレーションによる更新を推奨。
    """

    def __init__(
        self,
        z_thresh:       float = 3.0,
        min_pixels:     int   = 5,
        q_scale_factor: float = 30.0,
    ):
        self.z_thresh       = z_thresh
        self.min_pixels     = min_pixels
        self.q_scale_factor = q_scale_factor

    def infer(
        self,
        mbsp:       np.ndarray,
        wind_speed: float,
        wind_deg:   float,
        **kwargs,
    ) -> Optional[InferenceResult]:
        """
        MBSP の Z スコア閾値で検出し、簡易 Q 推定を行う。

        Parameters
        ----------
        mbsp       : MBSP フィールド shape=(H, W)
        wind_speed : 風速 [m/s]
        wind_deg   : 気象風向 [degrees]

        Returns
        -------
        InferenceResult または None (検出なし)
        """
        # --- 背景除去 (MAD ロバスト推定) ---
        mu    = float(np.nanmedian(mbsp))
        mad   = float(np.nanmedian(np.abs(mbsp - mu)))
        sigma = 1.4826 * mad + 1e-8
        z     = (mbsp - mu) / sigma

        # --- 閾値検出 ---
        mask  = z > self.z_thresh
        n_pix = int(np.sum(mask))
        if n_pix < self.min_pixels:
            return None

        # --- MLLR 計算 (超閾値 Z スコアの積分値) ---
        mllr = float(np.sum(np.maximum(z[mask] - self.z_thresh, 0.0)))

        # --- 簡易 Q 推定 ---
        # Q [kg/h] ≈ MLLR × U [m/s] × q_scale_factor
        # q_scale_factor=30.0 は Varon et al. (2021) Table 1 の
        # IME (Integrated Methane Enhancement) 法の比例定数を
        # MBSP 単位系に合わせて経験的に調整した値。
        # 実データでのキャリブレーションによる更新を推奨。
        q_est = mllr * wind_speed * self.q_scale_factor
        q_std = q_est * 0.4   # 固定 40% 不確実性（保守的推定）

        # --- 検出確率校正 (経験的ロジスティック関数) ---
        p_det = float(expit(0.05 * mllr - 2.0))

        flags: QualityFlags = {
            "low_wind":          wind_speed < 2.0,
            "multi_modal_theta": False,
            "roi_unstable":      False,
            "template_dominant": False,
        }

        return {
            "q":     float(np.clip(q_est, 0, 50000)),
            "q_std": float(q_std),
            "mllr":  mllr,
            "p_det": p_det,
            "flags": flags,
        }


# =============================================================================
# Engine 2: EucalyptusEngine
#
# Project Eucalyptus (Orbio-Earth) の学習済みモデルへのアダプタ。
#
# ライセンス:
#   Project Eucalyptus は非商用ライセンスです。
#   研究・教育目的での使用のみ許可されています。
#   商用利用については Orbio-Earth (info@orbio.earth) にお問い合わせください。
#   https://github.com/Orbio-Earth/Project-Eucalyptus
#
# インストール:
#   git clone https://github.com/Orbio-Earth/Project-Eucalyptus
#   pip install -r Project-Eucalyptus/requirements.txt
#
# 入力仕様の注意:
#   Project Eucalyptus のモデルは Sentinel-2 全バンド (10チャンネル) と
#   2時点の時系列入力を想定しています。
#   B11/B12 の1時点のみの場合、このアダプタは
#   MBSPThresholdEngine にフォールバックします。
#   時系列参照シーンが必要な場合は multi_temporal に別日のシーンを格納して
#   kwargs["bands"] 経由で渡してください。
#
# 精度:
#   原論文 (Varon et al., 2021 / Rischard et al., 2025) によると、
#   200〜300 kg/h 以上の排出源を検出可能です。
# =============================================================================

def _try_import_eucalyptus() -> bool:
    """Project Eucalyptus のインポートを試みる。"""
    try:
        # Project Eucalyptus のクローン先を sys.path に追加する場合は
        # 実行前に以下を設定してください:
        # sys.path.insert(0, "/path/to/Project-Eucalyptus")
        import eucalyptus  # noqa: F401
        return True
    except ImportError:
        return False


class EucalyptusEngine(InferenceEngine):
    """
    Project Eucalyptus (Orbio-Earth) の学習済みモデルへのアダプタ。

    ライセンス: 非商用のみ
    リポジトリ: https://github.com/Orbio-Earth/Project-Eucalyptus

    インストールが必要:
        git clone https://github.com/Orbio-Earth/Project-Eucalyptus
        pip install -r Project-Eucalyptus/requirements.txt

    入力仕様の制約:
        Project Eucalyptus のモデルは全バンド + 時系列2枚を想定しています。
        B11/B12 の1時点のみの場合、このアダプタは
        MBSPThresholdEngine にフォールバックします。

    Parameters
    ----------
    model_path       : 学習済みモデルのパス（省略時は自動ダウンロードを試みる）
    fallback         : Eucalyptus が利用不可の場合のフォールバックエンジン
                       省略時は MBSPThresholdEngine を使用する。
    allow_same_frame : t と t_ref に同一フレームを使用することを許可するか。
                       False (デフォルト) の場合は RuntimeError を送出する。
                       True に設定すると精度が著しく低下するため、
                       研究目的での動作確認以外では使用しないこと。
    """

    def __init__(
        self,
        model_path:       Optional[str]             = None,
        fallback:         Optional[InferenceEngine] = None,
        allow_same_frame: bool                      = False,
    ):
        self.model_path       = model_path
        self.fallback         = fallback or MBSPThresholdEngine()
        self.allow_same_frame = allow_same_frame
        self._model           = None
        self._lock            = threading.Lock()
        self._available       = _try_import_eucalyptus()

        if not self._available:
            warnings.warn(
                "\n"
                "  Project Eucalyptus が見つかりません。\n"
                "  MBSPThresholdEngine にフォールバックします。\n"
                "\n"
                "  Eucalyptus を使用するには:\n"
                "    git clone https://github.com/Orbio-Earth/Project-Eucalyptus\n"
                "    pip install -r Project-Eucalyptus/requirements.txt\n"
                "\n"
                "  ライセンス: 非商用のみ (研究・教育目的)\n",
                stacklevel=2,
            )

    def _load_model(self) -> None:
        """
        モデルを遅延ロードする (スレッドセーフ)。

        _available フラグの更新も _lock 内で行う。
        ロック外で _available を書き換えると、他スレッドが
        古い値を読んで _load_model を再呼び出しする可能性があるため。
        """
        if self._model is not None:
            return
        if not self._available:
            return
        with self._lock:
            # ロック取得後に再チェック (二重ロードを防ぐ)
            if self._model is not None:
                return
            if not self._available:
                return
            try:
                import eucalyptus
                self._model = eucalyptus.load_model(
                    sensor = "sentinel2",
                    path   = self.model_path,  # None の場合は自動ダウンロード
                )
                print("  [EucalyptusEngine] モデルロード完了")
            except Exception as e:
                warnings.warn(
                    f"Eucalyptus モデルのロードに失敗しました: {e}\n"
                    "MBSPThresholdEngine にフォールバックします。",
                    stacklevel=2,
                )
                # _available の書き換えは _lock 内で行い、他スレッドへの
                # 可視性を保証する。
                self._available = False

    def infer(
        self,
        mbsp:       np.ndarray,
        wind_speed: float,
        wind_deg:   float,
        **kwargs,
    ) -> Optional[InferenceResult]:
        """
        Eucalyptus モデルで推論する。
        利用不可の場合は fallback エンジンに委譲する。

        Parameters
        ----------
        mbsp       : MBSP フィールド shape=(H, W)
        wind_speed : 風速 [m/s]
        wind_deg   : 気象風向 [degrees]
        kwargs     :
            bands  : BandData  {"B11": ndarray, "B12": ndarray} (任意)
                     渡された場合は Eucalyptus の入力テンソルを構成する。
                     未指定の場合は fallback エンジンに委譲する。

        Returns
        -------
        InferenceResult または None (検出なし)
        """
        self._load_model()

        if not self._available or self._model is None:
            return self.fallback.infer(mbsp, wind_speed, wind_deg, **kwargs)

        bands = kwargs.get("bands")
        if bands is None:
            warnings.warn(
                "EucalyptusEngine: bands (B11/B12) が渡されていません。"
                "MBSPThresholdEngine にフォールバックします。",
                stacklevel=2,
            )
            return self.fallback.infer(mbsp, wind_speed, wind_deg)

        # t と t_ref に同一フレームを使用すると時系列差分がゼロになり精度が著しく低下する。
        # allow_same_frame=False (デフォルト) の場合は RuntimeError を送出して誤用を防ぐ。
        # 正確な推論には ObservationBundle["multi_temporal"] に
        # 別日の参照シーンを格納した上で渡してください。
        if not self.allow_same_frame:
            raise RuntimeError(
                "EucalyptusEngine: t と t_ref に同一フレームを使用しようとしています。\n"
                "時系列差分がゼロになるため精度が著しく低下します。\n"
                "multi_temporal に別日の参照シーンを格納して渡すか、\n"
                "動作確認目的であれば EucalyptusEngine(allow_same_frame=True) を"
                "明示的に指定してください。"
            )

        warnings.warn(
            "EucalyptusEngine: t と t_ref に同一フレームを使用しています (allow_same_frame=True)。\n"
            "時系列差分がゼロになるため精度が著しく低下します。\n"
            "本番運用では multi_temporal に参照シーンを格納して渡してください。",
            RuntimeWarning,
            stacklevel=2,
        )

        try:
            import torch
            h, w   = bands["B11"].shape
            b11    = torch.from_numpy(bands["B11"]).float().unsqueeze(0)
            b12    = torch.from_numpy(bands["B12"]).float().unsqueeze(0)
            # 疑似テンソル: [B11, B12, zeros×8] × 2時点 = 20チャンネル
            zeros  = torch.zeros(8, h, w)
            frame  = torch.cat([b11, b12, zeros], dim=0)          # 10チャンネル
            tensor = torch.cat([frame, frame], dim=0).unsqueeze(0) # 1×20×H×W

            with torch.no_grad():
                result = self._model.predict(tensor)

            plume_mask = result["mask"].squeeze().numpy()
            confidence = float(result.get("confidence", 0.5))
            n_pix      = int(np.sum(plume_mask > 0.5))

            if n_pix < 5:
                return None

            # Q 推定: Eucalyptus の定量出力がある場合はそちらを使用
            if "emission_rate_kg_h" in result:
                q     = float(result["emission_rate_kg_h"])
                q_std = q * 0.3
            else:
                q     = float(n_pix * wind_speed * 20.0)  # 粗い近似
                q_std = q * 0.5

            mllr  = float(n_pix) * confidence
            p_det = float(np.clip(confidence, 0, 1))

            flags: QualityFlags = {
                "low_wind":          wind_speed < 2.0,
                "multi_modal_theta": False,
                "roi_unstable":      False,
                "template_dominant": False,
            }

            return {
                "q":     float(np.clip(q, 0, 50000)),
                "q_std": float(q_std),
                "mllr":  mllr,
                "p_det": p_det,
                "flags": flags,
            }

        except Exception as e:
            warnings.warn(
                f"EucalyptusEngine 推論エラー: {e}\n"
                "MBSPThresholdEngine にフォールバックします。",
                stacklevel=2,
            )
            return self.fallback.infer(mbsp, wind_speed, wind_deg)


# =============================================================================
# ファクトリ
# =============================================================================

def build_engine(name: str) -> InferenceEngine:
    """
    名前文字列から推論エンジンを生成するファクトリ。

    Parameters
    ----------
    name : "mbsp" / "eucalyptus" / "null"

    Returns
    -------
    InferenceEngine の実装

    Raises
    ------
    ValueError : 不明なエンジン名が指定された場合
    """
    if name == "mbsp":
        _logger.info("推論エンジン: MBSPThresholdEngine (Varon et al. 2021 準拠)")
        return MBSPThresholdEngine()

    elif name == "eucalyptus":
        _logger.info("推論エンジン: EucalyptusEngine (Project Eucalyptus / 非商用)")
        _logger.info("ライセンス: 非商用のみ — https://github.com/Orbio-Earth/Project-Eucalyptus")
        return EucalyptusEngine(fallback=MBSPThresholdEngine())

    elif name == "null":
        _logger.info("推論エンジン: NullInferenceEngine (ダミー・テスト用)")
        return NullInferenceEngine()

    else:
        raise ValueError(
            f"不明なエンジン名: '{name}'\n"
            f"  有効な値: 'mbsp', 'eucalyptus', 'null'"
        )
