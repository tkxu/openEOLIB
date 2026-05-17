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
#
#   カスタムエンジンの実装サンプルとして参照してください。
#
# =============================================================================

#eo_engines.py
from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import numpy as np
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
        - NullInferenceEngine   (テスト用ダミー、このファイルに同梱)
        - MBSPThresholdEngine   (main_sample.py に定義)
        - CustomEngine          (main_sample.py に定義)

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
               実運用では MBSPThresholdEngine (main_sample.py) の
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
# ファクトリ
# =============================================================================

def build_engine(name: str) -> InferenceEngine:
    """
    名前文字列から推論エンジンを生成するファクトリ。

    eo_engines.py が直接提供するエンジンは NullInferenceEngine のみ。
    MBSPThresholdEngine / EucalyptusEngine / CustomEngine は
    main_sample.py に定義されており、そちらの build_engine() で取得できる。

    Parameters
    ----------
    name : "null"

    Returns
    -------
    InferenceEngine の実装

    Raises
    ------
    ValueError : 不明なエンジン名が指定された場合
    """
    if name == "null":
        _logger.info("推論エンジン: NullInferenceEngine (ダミー・テスト用)")
        return NullInferenceEngine()

    else:
        raise ValueError(
            f"不明なエンジン名: '{name}'\n"
            f"  有効な値: 'null'\n"
            f"  mbsp / eucalyptus / custom は main_sample.py の build_engine() を使用してください。"
        )
