"""
ApacheLicense2.0
Copyright (c) 2026 tkxu

eo_roc.py — ROC 曲線構築モジュール

役割:
    eo_pipeline.EOPipeline から分離した ROC 曲線構築ロジックを提供する。
    パイプライン実行結果 (SiteResultList) とサイトレジストリを受け取り、
    ROC 曲線データ (RocData) を返す。

    ┌──────────────────────────────────────────────────────────────┐
    │  依存関係                                                      │
    │                                                               │
    │  eo_types.py  ← eo_roc.py  (RocData / SiteResult を参照)    │
    │  eo_roc.py    ← eo_pipeline.py  (EOPipeline から呼ばれる)   │
    │  eo_roc.py    ← eo_visualisation.py (RocData を渡す)        │
    │                                                               │
    │  eo_roc.py は eo_pipeline / eo_provider / eo_simulator を    │
    │  一切 import しない。                                          │
    └──────────────────────────────────────────────────────────────┘

アルゴリズム:
    1. SiteResultList から正例 MLLR を収集する。
       inv が None のサイトは MLLR = -∞ として扱う。

    2. ヌル分布 MLLR を生成する。
       NullMllrSampler を用い、サイトごとに null_runs 回のランダム
       MBSP サンプルから NullInferenceEngine 相当の MLLR を計算する。
       ヌルサンプラーは外部から差し替え可能 (NullSampler ABC)。

    3. MLLR_THRESHOLDS をスイープして FPR / TPR を計算し RocData を返す。

    4. AUC はトラペゾイド則で計算する。

使用例:
    from eo_roc      import RocBuilder, NullMllrSampler
    from eo_pipeline import EOPipeline, NullInferenceEngine
    from eo_simulator import PlumeSimulator

    pipeline = EOPipeline(provider=PlumeSimulator(), engine=NullInferenceEngine())
    results  = pipeline.run_all(site_registry)

    builder  = RocBuilder()
    roc      = builder.build(results, site_registry)
    # roc["auc"], roc["fpr"], roc["tpr"] ...

依存ライブラリ:
    pip install numpy
"""

from __future__ import annotations

import warnings
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

from eo_types import (
    RocData,
    SiteEntry,
    SiteResult,
    SiteResultList,
)


# =============================================================================
# 定数
# =============================================================================

# ROC 構築用 MLLR スイープ範囲
# ★ eo_pipeline.py はここから import して使用すること（二重定義禁止）
MLLR_THRESHOLDS = np.linspace(-50, 300, 120)

# デフォルトのヌル試行回数 (サイトあたり)
NULL_RUNS_DEFAULT = 30

# 偽陽性率の FPR 補間グリッド (ROC プロット用)
FPR_GRID = np.linspace(0, 1, 200)


# =============================================================================
# NullSampler — ヌル MLLR 生成の抽象基底クラス
# =============================================================================

class NullSampler(ABC):
    """
    ヌル分布 MLLR を生成する抽象基底クラス。

    RocBuilder に差し込むことで、ヌル MLLR の生成ロジックを
    パイプラインから独立して差し替えられる。

    実装義務:
        sample_null_mllrs() を実装すること。
    """

    @abstractmethod
    def sample_null_mllrs(
        self,
        result:    SiteResult,
        null_runs: int,
        rng:       np.random.Generator,
    ) -> List[float]:
        """
        サイト1件分のヌル MLLR をサンプリングする。

        Parameters
        ----------
        result    : 対象サイトの SiteResult
        null_runs : ヌル試行回数
        rng       : 乱数生成器

        Returns
        -------
        List[float]  長さ null_runs のヌル MLLR リスト
        """
        ...


# =============================================================================
# NullMllrSampler — デフォルト実装
# =============================================================================

class NullMllrSampler(NullSampler):
    """
    MBSP フィールドをランダムシャッフルしてヌル MLLR を生成する。

    プルームシグナルが空間的に集中しているという仮定のもと、
    MBSP を空間シャッフルすることでヌル分布を近似する。

    アルゴリズム:
        1. result["mbsp"] を取得する。
        2. null_runs 回、mbsp をフラット化してシャッフルし、
           元の shape に戻す。
        3. シャッフルされた mbsp に対して _simple_mllr() を適用する。

    Parameters
    ----------
    z_threshold : MLLR 計算用の Z スコア閾値 (デフォルト: 2.0)
    """

    def __init__(self, z_threshold: float = 2.0):
        self.z_threshold = z_threshold

    def sample_null_mllrs(
        self,
        result:    SiteResult,
        null_runs: int,
        rng:       np.random.Generator,
    ) -> List[float]:
        mbsp = result.get("mbsp")
        if mbsp is None or mbsp.size == 0:
            return [float("-inf")] * null_runs

        flat = mbsp.ravel().copy()
        null_mllrs: List[float] = []

        for _ in range(null_runs):
            rng.shuffle(flat)
            shuffled = flat.reshape(mbsp.shape)
            null_mllrs.append(self._simple_mllr(shuffled))

        return null_mllrs

    def _simple_mllr(self, mbsp: np.ndarray) -> float:
        """
        MBSP フィールドから簡易 MLLR を計算する。

        NullInferenceEngine と同等のヒューリスティック。
        MLLR = Σ max(Z - threshold, 0)
        """
        mu  = float(np.nanmedian(mbsp))
        sig = float(1.4826 * np.nanmedian(np.abs(mbsp - mu))) + 1e-8
        z   = (mbsp - mu) / sig
        return float(np.nansum(np.maximum(z - self.z_threshold, 0)))


# =============================================================================
# RocCurveCalculator — FPR/TPR 計算ロジック
# =============================================================================

class RocCurveCalculator:
    """
    正例 MLLR とヌル MLLR から ROC 曲線を計算する。

    MLLR_THRESHOLDS をスイープして各閾値における
    TPR (真陽性率) と FPR (偽陽性率) を計算する。
    """

    def compute(
        self,
        positive_mllrs: List[float],
        null_mllrs:     List[float],
        thresholds:     np.ndarray = MLLR_THRESHOLDS,
    ) -> RocData:
        """
        ROC 曲線データを計算する。

        Parameters
        ----------
        positive_mllrs : プルームありサイトの MLLR リスト (正例)
        null_mllrs     : ヌル試行の MLLR リスト (負例)
        thresholds     : スイープする MLLR 閾値配列

        Returns
        -------
        RocData
        """
        pos = np.array([m for m in positive_mllrs if np.isfinite(m)],
                       dtype=np.float64)
        neg = np.array([m for m in null_mllrs     if np.isfinite(m)],
                       dtype=np.float64)

        n_pos = len(pos)
        n_neg = len(neg)

        if n_pos == 0:
            warnings.warn(
                "正例 MLLR が空です。ROC 曲線を構築できません。"
                "全サイトで推論エンジンが None を返していないか確認してください。",
                stacklevel=3,
            )
            return RocData(
                fpr=np.zeros(len(thresholds)),
                tpr=np.zeros(len(thresholds)),
                auc=0.0,
                positive_mllrs=list(positive_mllrs),
                null_mllrs=list(null_mllrs),
            )

        fpr_arr = np.zeros(len(thresholds))
        tpr_arr = np.zeros(len(thresholds))

        for i, thresh in enumerate(thresholds):
            tpr_arr[i] = float(np.mean(pos >= thresh)) if n_pos > 0 else 0.0
            fpr_arr[i] = float(np.mean(neg >= thresh)) if n_neg > 0 else 0.0

        # FPR の昇順でソートして ROC 曲線を整える
        sort_idx = np.argsort(fpr_arr)
        fpr_sorted = fpr_arr[sort_idx]
        tpr_sorted = tpr_arr[sort_idx]

        # AUC: トラペゾイド則 (NumPy 2.x: trapezoid / 1.x: trapz)
        try:
            auc = float(np.trapezoid(tpr_sorted, fpr_sorted))
        except AttributeError:  # NumPy < 2.0
            auc = float(np.trapz(tpr_sorted, fpr_sorted))

        return RocData(
            fpr=fpr_sorted.astype(np.float64),
            tpr=tpr_sorted.astype(np.float64),
            auc=float(np.clip(auc, 0.0, 1.0)),
            positive_mllrs=positive_mllrs,
            null_mllrs=null_mllrs,
        )


# =============================================================================
# RocBuilder — 公開 API
# =============================================================================

class RocBuilder:
    """
    SiteResultList から ROC 曲線データを構築する。

    eo_pipeline.EOPipeline から build_roc() として呼ばれることを想定するが、
    単独でも使用できる。

    Parameters
    ----------
    null_runs   : サイトあたりのヌル試行回数
    null_sampler: ヌル MLLR 生成器 (省略時は NullMllrSampler)
    seed        : 乱数シード (再現性)
    thresholds  : MLLR スイープ範囲 (省略時は MLLR_THRESHOLDS)

    使用例:
        builder = RocBuilder(null_runs=30, seed=0)
        roc     = builder.build(results, site_registry)
        print(f"AUC = {roc['auc']:.4f}")
    """

    def __init__(
        self,
        null_runs:    int                   = NULL_RUNS_DEFAULT,
        null_sampler: Optional[NullSampler] = None,
        seed:         Optional[int]         = None,
        thresholds:   Optional[np.ndarray]  = None,
    ):
        self.null_runs    = null_runs
        self.null_sampler = null_sampler or NullMllrSampler()
        self.seed         = seed
        self.thresholds   = thresholds if thresholds is not None else MLLR_THRESHOLDS
        self._calculator  = RocCurveCalculator()

    def build(
        self,
        results:       SiteResultList,
        site_registry: List[SiteEntry],
    ) -> RocData:
        """
        ROC 曲線データを構築する。

        Parameters
        ----------
        results       : EOPipeline.run_all() の返値
        site_registry : SiteEntry のリスト (results と同順)

        Returns
        -------
        RocData
        """
        rng = np.random.default_rng(self.seed)

        positive_mllrs = self._collect_positive_mllrs(results)
        null_mllrs     = self._collect_null_mllrs(results, rng)

        return self._calculator.compute(
            positive_mllrs = positive_mllrs,
            null_mllrs     = null_mllrs,
            thresholds     = self.thresholds,
        )

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------

    def _collect_positive_mllrs(self, results: SiteResultList) -> List[float]:
        """
        各サイトの正例 MLLR を収集する。

        inv が None のサイト (検出なし) は -inf として扱い、
        常に「検出失敗」として ROC に反映する。
        """
        mllrs: List[float] = []
        for r in results:
            inv = r.get("inv")
            if inv is not None:
                mllrs.append(float(inv["mllr"]))
            else:
                mllrs.append(float("-inf"))
        return mllrs

    def _collect_null_mllrs(
        self,
        results: SiteResultList,
        rng:     np.random.Generator,
    ) -> List[float]:
        """
        全サイトのヌル MLLR を収集する。

        サイトごとに null_runs 回サンプリングし、フラットなリストにまとめる。
        """
        null_mllrs: List[float] = []
        for r in results:
            sampled = self.null_sampler.sample_null_mllrs(r, self.null_runs, rng)
            null_mllrs.extend(sampled)
        return null_mllrs


# =============================================================================
# モジュールレベル便利関数
# =============================================================================

def build_roc(
    results:       SiteResultList,
    site_registry: List[SiteEntry],
    null_runs:     int            = NULL_RUNS_DEFAULT,
    seed:          Optional[int]  = None,
) -> RocData:
    """
    RocBuilder のデフォルト設定を使ったショートカット関数。

    Parameters
    ----------
    results       : EOPipeline.run_all() の返値
    site_registry : SiteEntry のリスト
    null_runs     : ヌル試行回数
    seed          : 乱数シード

    Returns
    -------
    RocData

    使用例:
        from eo_roc import build_roc
        roc = build_roc(results, site_registry, null_runs=30, seed=0)
    """
    return RocBuilder(null_runs=null_runs, seed=seed).build(results, site_registry)


# =============================================================================
# テストコード (外部API不要)
# 注意: 本番運用では tests/ ディレクトリに分離することを推奨します。
# =============================================================================

def test_roc_curve_calculator() -> None:
    """RocCurveCalculator の単体テスト。"""
    print("\n" + "="*55)
    print("  TEST-1: RocCurveCalculator")
    print("="*55)

    calc = RocCurveCalculator()

    # 完全分離: AUC = 1.0 に近い
    pos  = [200.0, 150.0, 180.0, 220.0]
    null = [-5.0, -10.0, 2.0, -8.0]
    roc  = calc.compute(pos, null)

    assert 0.0 <= roc["auc"] <= 1.0,          f"AUC 範囲外: {roc['auc']}"
    assert len(roc["fpr"]) == len(roc["tpr"]), "fpr/tpr 長不一致"
    assert roc["auc"] > 0.8,                  f"完全分離で AUC={roc['auc']:.3f} が低すぎる"
    print(f"  完全分離 AUC = {roc['auc']:.4f} (期待: > 0.8) → PASS")

    # ランダム: AUC ≈ 0.5
    rng      = np.random.default_rng(0)
    pos_rand = rng.normal(10, 20, 100).tolist()
    neg_rand = rng.normal(10, 20, 100).tolist()
    roc_rand = calc.compute(pos_rand, neg_rand)
    print(f"  ランダム AUC  = {roc_rand['auc']:.4f} (期待: ≈ 0.5)")

    # 正例なし: 警告を出して AUC=0.0 で返る
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        roc_empty = calc.compute([], [1.0, 2.0, 3.0])
    assert roc_empty["auc"] == 0.0, "正例なし時の AUC が 0.0 でない"
    assert len(caught) == 1,        "正例なし時に警告が出ていない"
    print("  正例なし → 警告 + AUC=0.0 → PASS")


def test_null_mllr_sampler() -> None:
    """NullMllrSampler の単体テスト。"""
    print("\n" + "="*55)
    print("  TEST-2: NullMllrSampler")
    print("="*55)

    # ★ numpy はトップレベルでインポート済みのため再 import 不要
    from eo_simulator import PlumeSimulator

    sim    = PlumeSimulator()
    bundle = sim.generate(Q=3000.0, wind_speed=4.0, wind_deg=120.0, seed=42)

    # SiteResult の最小構成を手動で作成
    mbsp = (np.log(np.maximum(bundle["bands"]["B11"], 1e-8))
          - np.log(np.maximum(bundle["bands"]["B12"], 1e-8)))

    dummy_result: SiteResult = {
        "site":           {"id": "TEST", "name": "test", "lat": 0.0, "lon": 0.0,
                           "wind_speed": 4.0, "wind_deg": 120.0,
                           "Q_true": 3000.0, "seed": 42, "category": "mid-range"},
        "mbsp":           mbsp,
        "llr":            np.zeros_like(mbsp),
        "mask_v12":       np.zeros_like(mbsp, dtype=bool),
        "plume_true":     bundle["plume_true"],
        "wvec":           np.array([1.0, 0.0]),
        "post":           0.0,
        "candidates_v42": [],
        "best_candidate": None,
        "inv":            None,
        "phys_result":    None,
        "scoring":        None,
        "flags":          {"low_wind": False, "multi_modal_theta": False,
                           "roi_unstable": False, "template_dominant": False},
    }

    sampler = NullMllrSampler()
    rng     = np.random.default_rng(0)
    nulls   = sampler.sample_null_mllrs(dummy_result, null_runs=20, rng=rng)

    assert len(nulls) == 20,               f"ヌルサンプル数異常: {len(nulls)}"
    assert all(np.isfinite(m) for m in nulls), "ヌル MLLR に inf/nan"
    print(f"  ヌル MLLR: mean={np.mean(nulls):.2f}  std={np.std(nulls):.2f}")
    print(f"  null_runs=20 → len={len(nulls)} → PASS")


def test_roc_builder_integration() -> None:
    """RocBuilder の統合テスト (PlumeSimulator + NullInferenceEngine)。"""
    print("\n" + "="*55)
    print("  TEST-3: RocBuilder 統合テスト")
    print("="*55)

    from eo_simulator import PlumeSimulator
    from eo_pipeline  import EOPipeline, NullInferenceEngine
    from eo_types     import SiteEntry

    registry: List[SiteEntry] = [
        {"id": "TM-01", "name": "Turkmenistan", "lat": 38.49, "lon": 54.19,
         "wind_speed": 4.0, "wind_deg": 120, "Q_true": 4000.0,
         "seed": 42, "category": "super-emitter"},
        {"id": "PB-01", "name": "Permian Basin", "lat": 31.83, "lon": -102.37,
         "wind_speed": 6.5, "wind_deg": 200, "Q_true": 800.0,
         "seed": 7,  "category": "mid-range"},
        {"id": "HN-01", "name": "Hassi R'Mel", "lat": 32.93, "lon": 3.13,
         "wind_speed": 8.0, "wind_deg": 315, "Q_true": 300.0,
         "seed": 99, "category": "near-limit"},
    ]

    pipeline = EOPipeline(
        provider  = PlumeSimulator(),
        engine    = NullInferenceEngine(),
    )
    results = pipeline.run_all(registry)

    builder = RocBuilder(null_runs=10, seed=0)
    roc     = builder.build(results, registry)

    assert "fpr"            in roc
    assert "tpr"            in roc
    assert "auc"            in roc
    assert "positive_mllrs" in roc
    assert "null_mllrs"     in roc
    assert 0.0 <= roc["auc"] <= 1.0,              f"AUC 範囲外: {roc['auc']}"
    assert len(roc["fpr"]) == len(roc["tpr"]),     "fpr/tpr 長不一致"
    assert len(roc["positive_mllrs"]) == len(registry), "正例数がサイト数と不一致"
    assert len(roc["null_mllrs"]) == len(registry) * 10, \
        f"ヌル数異常: {len(roc['null_mllrs'])} (期待: {len(registry)*10})"

    print(f"  AUC = {roc['auc']:.4f}")
    print(f"  正例 MLLR: {[f'{m:.1f}' for m in roc['positive_mllrs']]}")
    print(f"  ヌル数: {len(roc['null_mllrs'])}")
    print("  → PASS")


def test_build_roc_shortcut() -> None:
    """モジュールレベル build_roc() のテスト。"""
    print("\n" + "="*55)
    print("  TEST-4: build_roc() ショートカット関数")
    print("="*55)

    from eo_simulator import PlumeSimulator
    from eo_pipeline  import EOPipeline, NullInferenceEngine
    from eo_types     import SiteEntry

    registry: List[SiteEntry] = [
        {"id": "TM-01", "name": "Turkmenistan", "lat": 38.49, "lon": 54.19,
         "wind_speed": 4.0, "wind_deg": 120, "Q_true": 4000.0,
         "seed": 42, "category": "super-emitter"},
    ]

    pipeline = EOPipeline(provider=PlumeSimulator(), engine=NullInferenceEngine())
    results  = pipeline.run_all(registry)
    roc      = build_roc(results, registry, null_runs=5, seed=42)

    assert 0.0 <= roc["auc"] <= 1.0
    print(f"  AUC = {roc['auc']:.4f} → PASS")


def test_custom_null_sampler() -> None:
    """カスタム NullSampler の差し込みテスト。"""
    print("\n" + "="*55)
    print("  TEST-5: カスタム NullSampler 差し込み")
    print("="*55)

    class ConstantNullSampler(NullSampler):
        """常に固定値を返すテスト用サンプラー。"""
        def sample_null_mllrs(self, result, null_runs, rng):
            return [-999.0] * null_runs

    from eo_simulator import PlumeSimulator
    from eo_pipeline  import EOPipeline, NullInferenceEngine
    from eo_types     import SiteEntry

    registry: List[SiteEntry] = [
        {"id": "TM-01", "name": "Turkmenistan", "lat": 38.49, "lon": 54.19,
         "wind_speed": 4.0, "wind_deg": 120, "Q_true": 4000.0,
         "seed": 42, "category": "super-emitter"},
    ]

    pipeline = EOPipeline(provider=PlumeSimulator(), engine=NullInferenceEngine())
    results  = pipeline.run_all(registry)

    builder = RocBuilder(null_runs=5, null_sampler=ConstantNullSampler(), seed=0)
    roc     = builder.build(results, registry)

    assert all(m == -999.0 for m in roc["null_mllrs"]), \
        "カスタムサンプラーの値が反映されていない"
    # ヌルが全て -999 なので正例 MLLR > -999 なら AUC = 1.0
    pos_valid = [m for m in roc["positive_mllrs"] if np.isfinite(m)]
    if pos_valid and all(m > -999.0 for m in pos_valid):
        assert roc["auc"] == 1.0, f"AUC が 1.0 でない: {roc['auc']}"
    print(f"  カスタムサンプラー (constant=-999) AUC={roc['auc']:.4f} → PASS")


# =============================================================================
# エントリポイント
# =============================================================================

if __name__ == "__main__":
    import logging, sys
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

    print("=" * 55)
    print("  eo_roc.py — ROC 曲線構築モジュールテスト")
    print("=" * 55)

    test_roc_curve_calculator()
    test_null_mllr_sampler()
    test_roc_builder_integration()
    test_build_roc_shortcut()
    test_custom_null_sampler()

    print("\n" + "="*55)
    print("  全テスト PASS")
    print("  使用方法:")
    print("    from eo_roc import build_roc, RocBuilder")
    print("    roc = build_roc(results, site_registry)")
    print("    # または")
    print("    roc = RocBuilder(null_runs=50, seed=0).build(results, registry)")
    print("="*55)
