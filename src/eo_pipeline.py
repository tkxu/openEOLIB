"""
ApacheLicense2.0
Copyright (c) 2026 tkxu
"""
# eo_pipeline.py
from __future__ import annotations

import logging
import warnings
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from eo_types import (
    BandData,
    InferenceResult,
    ObservationBundle,
    QualityFlags,
    RocData,
    SiteEntry,
    SiteResult,
    SiteResultList,
)
from eo_roc import RocBuilder, MLLR_THRESHOLDS


logger = logging.getLogger(__name__)

# === 定数 ===

# ヌル試行回数 (ROC 構築用)
NULL_RUNS_PER_SITE = 30

# デフォルトの空品質フラグ (推論エンジンが None を返した場合に使用)
_DEFAULT_FLAGS: QualityFlags = {
    "low_wind":          False,
    "multi_modal_theta": False,
    "roi_unstable":      False,
    "template_dominant": False,
}


def wind_deg_to_vec(deg: float) -> np.ndarray:
    """気象風向を単位ベクトル [vx=East, vy=North] に変換する。"""
    rad = np.radians(_wind_deg_to_math(deg))
    return np.array([np.cos(rad), np.sin(rad)])


def _wind_deg_to_math(deg: float) -> float:
    """
    気象風向 (北基準時計回り) を数学角 (東基準反時計回り) に変換する。

    全コンポーネントがこの関数を経由することで変換の一貫性を保証する。
    """
    return (270.0 - deg) % 360.0


# === MBSP 計算 (共有ユーティリティ) ===
def compute_mbsp(bands: BandData) -> np.ndarray:
    """
    Multi-Band Spectral Product (MBSP) を計算する。

    MBSP = log(B11) - log(B12)

    B11 (1610nm) と B12 (2190nm) のメタン吸収断面積の差を利用して
    メタンシグナルを強調する対数差分指標。

    Parameters
    ----------
    bands : BandData  {"B11": ndarray, "B12": ndarray}

    Returns
    -------
    np.ndarray  MBSP フィールド  shape=(H, W)  dtype=float32
    """
    return (
        np.log(np.maximum(bands["B11"], 1e-8))
      - np.log(np.maximum(bands["B12"], 1e-8))
    ).astype(np.float32)


# === 推論エンジンの抽象基底クラス (プロトコル定義) ===
class InferenceEngine(ABC):
    """
    推論エンジンの抽象基底クラス。

    obs_pipeline.py はこのインターフェースだけを知っており、
    具体的なアルゴリズム実装は外部から差し込まれる。

    実装クラスの例:
        - NullInferenceEngine        (テスト用ダミー、このファイルに同梱)
        - 外部ライブラリの推論モデル  (将来の連携)

    実装義務:
        infer() メソッドを実装すること。
        返り値は InferenceResult 型 (obs_types.py 参照)、
        検出不可能な場合は None を返す。
    """

    @abstractmethod
    def infer(
        self,
        mbsp:      np.ndarray,
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


# Provider 抽象基底クラスを定義し、統一インターフェースを提供
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


# === テスト用ダミー推論エンジン ===
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

        # 簡易 MLLR: Z スコアの積分値
        mllr = float(np.nansum(np.maximum(z - 2.0, 0)))
        q    = mllr * self.q_scale / 100.0

        flags: QualityFlags = {
            "low_wind":          wind_speed < 2.0,
            "multi_modal_theta": False,
            "roi_unstable":      False,
            "template_dominant": False,
        }

        return InferenceResult(
            q     = float(np.clip(q, 0, 50000)),
            q_std = float(q * 0.3),
            mllr  = mllr,
            p_det = float(np.clip(mllr / 200.0, 0, 1)),
            flags = flags,
        )


# === ObsPipeline — パイプライン本体 ===
class EOPipeline:
    """
    観測データ処理パイプライン。

    サイトレジストリを巡回し、データ取得 → MBSP 計算 →
    推論エンジン呼び出し → 結果集約 → ROC 評価 を実行する。

    推論エンジンは InferenceEngine を実装したオブジェクトとして
    外部から差し込む。eo_pipeline.py はエンジンの内部実装を
    一切知らない。

    Parameters
    ----------
    provider       : ObservationBundle を返すオブジェクト。
                     Provider を継承した DataFetcher (eo_provider.py) または
                     PlumeSimulator (eo_simulator.py) を渡す。
    engine         : InferenceEngine を実装した推論エンジン。
                     未指定の場合は NullInferenceEngine を使用。
    res            : ピクセル解像度 [m/px]
    null_runs      : ROC 構築用ヌル試行回数

    使用例:
        # 合成データ + ダミーエンジンでテスト
        from eo_simulator import PlumeSimulator
        pipeline = EOPipeline(provider=PlumeSimulator())
        results  = pipeline.run_all(site_registry)
        roc      = pipeline.build_roc(results, site_registry)

        from eo_provider import DataFetcher
        pipeline = EOPipeline(
            provider = DataFetcher(),
            engine   = CoreEngine(),
        )
    """

    def __init__(
        self,
        provider:  Provider,
        engine:    Optional[InferenceEngine] = None,
        res:       float = 20.0,
        null_runs: int   = NULL_RUNS_PER_SITE,
    ):
        self.provider  = provider
        self.engine    = engine or NullInferenceEngine()
        self.res       = res
        self.null_runs = null_runs

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------

    def _bundle_to_mbsp(self, bundle: ObservationBundle) -> Optional[np.ndarray]:
        """
        ObservationBundle から MBSP を計算する。

        bands が None の場合 (取得失敗) は None を返す。
        """
        if bundle.get("bands") is None:
            return None
        return compute_mbsp(bundle["bands"])

    def _extract_wind(
        self,
        bundle: ObservationBundle,
        site:   SiteEntry,
    ) -> Tuple[float, float]:
        """
        バンドルから風速・風向を取得する。

        実データ (openEO) では bundle の値を使用する。
        合成データ (synthetic) または取得失敗時は site の公称値にフォールバック。
        合成データの場合は bundle に wind が存在しないのが正常であるため警告は出さない。
        """
        ws  = bundle.get("wind_speed")
        wd  = bundle.get("wind_deg")
        src = bundle.get("meta", {}).get("backend", "unknown")

        if ws is None or wd is None:
            if src != "synthetic":
                warnings.warn(
                    f"[{site['id']}] 風速・風向の取得に失敗。"
                    f"site の公称値にフォールバックします。",
                    stacklevel=2,
                )
            ws = site["wind_speed"]
            wd = site["wind_deg"]

        return float(ws), float(wd)

    def _fetch_bundle(
        self,
        site: SiteEntry,
        dt:   Optional[datetime],
    ) -> ObservationBundle:
        """
        provider の種別を判定して ObservationBundle を取得する。

        Provider 抽象基底クラスを実装したオブジェクトは get_bundle() で統一取得。
        未実装の旧インターフェース (generate / fetch_from_site / fetch) は
        後方互換のためフォールバック対応するが、将来のバージョンで削除予定。
        """
        # Provider ABC を実装している場合は統一インターフェースを使用
        if isinstance(self.provider, Provider):
            return self.provider.get_bundle(site, dt)

        # 後方互換: 旧インターフェースへのフォールバック
        # DeprecationWarning: v2.0 で削除予定。Provider ABC を実装してください。
        if hasattr(self.provider, "generate"):
            warnings.warn(
                f"{type(self.provider).__name__} は Provider ABC を実装していません。"
                "generate() インターフェースは非推奨です。"
                "Provider を継承して get_bundle() を実装してください。"
                "このフォールバックは v2.0 で削除予定です。",
                DeprecationWarning,
                stacklevel=3,
            )
            # PlumeSimulator
            return self.provider.generate(
                Q          = site["Q_true"],
                wind_speed = site["wind_speed"],
                wind_deg   = site["wind_deg"],
                lat        = site["lat"],
                lon        = site["lon"],
                seed       = site.get("seed"),
            )
        elif hasattr(self.provider, "fetch_from_site") and dt is not None:
            warnings.warn(
                f"{type(self.provider).__name__} は Provider ABC を実装していません。"
                "fetch_from_site() インターフェースは非推奨です。"
                "Provider を継承して get_bundle() を実装してください。"
                "このフォールバックは v2.0 で削除予定です。",
                DeprecationWarning,
                stacklevel=3,
            )
            # DataFetcher (earth_obs_provider.py)
            return self.provider.fetch_from_site(site, dt)
        elif hasattr(self.provider, "fetch") and dt is not None:
            warnings.warn(
                f"{type(self.provider).__name__} は Provider ABC を実装していません。"
                "fetch() インターフェースは非推奨です。"
                "Provider を継承して get_bundle() を実装してください。"
                "このフォールバックは v2.0 で削除予定です。",
                DeprecationWarning,
                stacklevel=3,
            )
            return self.provider.fetch(site["lat"], site["lon"], dt)
        else:
            raise ValueError(
                "provider は Provider ABC を継承して get_bundle() を実装するか、"
                "generate() または fetch() / fetch_from_site() を"
                "持つオブジェクトである必要があります。"
            )

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def run_site(
        self,
        site: SiteEntry,
        dt:   Optional[datetime] = None,
    ) -> SiteResult:
        """
        単一サイトのパイプラインを実行する。

        1. provider からデータ取得 (実データ or 合成データ)
        2. MBSP 計算
        3. 推論エンジンに委譲
        4. SiteResult に集約して返す

        Parameters
        ----------
        site : SiteEntry  サイトレジストリのエントリ
        dt   : datetime   観測日時 (実データ取得時のみ使用)

        Returns
        -------
        SiteResult
        """
        # --- データ取得 ---
        bundle: ObservationBundle = self._fetch_bundle(site, dt)

        # --- MBSP 計算 ---
        mbsp = self._bundle_to_mbsp(bundle)
        if mbsp is None:
            warnings.warn(f"[{site['id']}] MBSP の計算に失敗しました。", stacklevel=2)
            h = int(site.get("grid_h", 100))
            w = int(site.get("grid_w", 100))
            mbsp = np.zeros((h, w), dtype=np.float32)

        wind_speed, wind_deg = self._extract_wind(bundle, site)

        # --- 推論エンジンに委譲 ---
        inv_result = self.engine.infer(mbsp, wind_speed, wind_deg)

        flags: QualityFlags = (
            inv_result["flags"]
            if inv_result is not None
            else _DEFAULT_FLAGS.copy()
        )

        # --- plume_true の取得 (合成データのみ) ---
        # bundle.get("plume_true", fallback) はキーが存在して値が None の場合に
        # fallback を返さない。"or" を使って None も確実に fallback する。
        plume_true = bundle.get("plume_true") or np.zeros_like(mbsp)

        # --- 風向ベクトル ---
        wvec = wind_deg_to_vec(wind_deg)

        return SiteResult(
            site       = site,
            mbsp       = mbsp,
            llr        = np.zeros_like(mbsp),  # エンジンが未提供の場合はゼロ
            mask_v12   = np.zeros_like(mbsp, dtype=bool),
            plume_true = plume_true,
            wvec       = wvec,
            post       = float(inv_result["p_det"]) if inv_result else 0.0,
            candidates_v42 = [],
            best_candidate = None,
            inv            = inv_result,
            phys_result    = None,
            scoring        = None,
            flags          = flags,
            gt_match       = False,
            meta           = None,
        )

    def run_all(
        self,
        site_registry: List[SiteEntry],
        dt: Optional[datetime] = None,
    ) -> SiteResultList:
        """
        サイトレジストリ全体を巡回してパイプラインを実行する。

        Parameters
        ----------
        site_registry : SiteEntry のリスト
        dt            : datetime  実データ取得時の観測日時

        Returns
        -------
        SiteResultList  各サイトの SiteResult リスト
        """
        results: SiteResultList = []
        for site in site_registry:
            logger.info("  [%s] %s ...", site["id"], site["name"])
            r      = self.run_site(site, dt=dt)
            status = "OK" if r["inv"] is not None else "NO DETECT"
            q_str  = f"Q={r['inv']['q']:.0f}" if r["inv"] else "—"
            logger.info("%s  %s kg/h", status, q_str)
            results.append(r)
        return results

    def build_roc(
        self,
        results:       SiteResultList,
        site_registry: List[SiteEntry],
        seed:          Optional[int] = None,
    ) -> RocData:
        """
        ROC 曲線データを構築する。

        内部で eo_roc.RocBuilder を使用する。
        ヌル試行回数は EOPipeline(null_runs=...) で指定した値を引き継ぐ。

        Parameters
        ----------
        results       : run_all() の返値
        site_registry : SiteEntry のリスト (results と同順)
        seed          : 乱数シード (再現性)

        Returns
        -------
        RocData
        """
        return RocBuilder(
            null_runs = self.null_runs,
            seed      = seed,
        ).build(results, site_registry)


# =============================================================================
# テストコード
# =============================================================================

# サンプルサイトレジストリ
_SAMPLE_REGISTRY: List[SiteEntry] = [
    SiteEntry(
        id="TM-01", name="Turkmenistan Compressor Station",
        lat=38.49, lon=54.19, wind_speed=4.0, wind_deg=120,
        Q_true=4000.0, seed=42, category="super-emitter",
    ),
    SiteEntry(
        id="PB-01", name="Permian Basin Wellpad",
        lat=31.83, lon=-102.37, wind_speed=6.5, wind_deg=200,
        Q_true=800.0, seed=7, category="mid-range",
    ),
    SiteEntry(
        id="DZ-01", name="Algeria In Salah Gas Field",
        lat=27.21, lon=2.52, wind_speed=3.0, wind_deg=45,
        Q_true=1800.0, seed=13, category="mid-range",
    ),
    SiteEntry(
        id="HN-01", name="Hassi R'Mel Flare Station",
        lat=32.93, lon=3.13, wind_speed=8.0, wind_deg=315,
        Q_true=300.0, seed=99, category="near-limit",
    ),
]


def test_compute_mbsp() -> None:
    """compute_mbsp の単体テスト。"""
    print("\n" + "="*55)
    print("  TEST-1: compute_mbsp")
    print("="*55)

    b11  = np.ones((10, 10), dtype=np.float32) * 0.5
    b12  = np.ones((10, 10), dtype=np.float32) * 0.4
    mbsp = compute_mbsp({"B11": b11, "B12": b12})

    expected = float(np.log(0.5) - np.log(0.4))
    assert mbsp.shape == (10, 10),              "shape 異常"
    assert np.allclose(mbsp, expected, atol=1e-5), "値が期待値と一致しない"
    assert mbsp.dtype == np.float32,            "dtype が float32 でない"
    print(f"  MBSP 値: {mbsp[0,0]:.6f}  期待値: {expected:.6f}")
    print("  → PASS")


def test_null_inference_engine() -> None:
    """NullInferenceEngine の単体テスト。"""
    print("\n" + "="*55)
    print("  TEST-2: NullInferenceEngine")
    print("="*55)

    from eo_simulator import PlumeSimulator
    sim    = PlumeSimulator()
    bundle = sim.generate(Q=3000.0, wind_speed=4.0, wind_deg=120.0, seed=42)
    mbsp   = compute_mbsp(bundle["bands"])

    engine = NullInferenceEngine()
    result = engine.infer(mbsp, 4.0, 120.0)

    if result is not None:
        assert result["q"]     >= 0,      "q が負値"
        assert result["q_std"] >= 0,      "q_std が負値"
        assert 0 <= result["p_det"] <= 1, "p_det が [0,1] 外"
        assert "flags" in result,         "flags が返却されていない"
        print(f"  q={result['q']:.1f}  mllr={result['mllr']:.2f}  "
              f"p_det={result['p_det']:.3f}")
    else:
        print("  検出なし (Q が低すぎる可能性)")
    print("  → PASS")


def test_inference_engine_protocol() -> None:
    """InferenceEngine プロトコルの差し込みテスト。"""
    print("\n" + "="*55)
    print("  TEST-3: InferenceEngine 差し込みプロトコル")
    print("="*55)

    class DummyEngine(InferenceEngine):
        """最小実装のダミーエンジン。"""
        def infer(self, mbsp, wind_speed, wind_deg, **kwargs):
            return InferenceResult(
                q=999.0, q_std=1.0, mllr=50.0, p_det=0.8,
                flags=_DEFAULT_FLAGS.copy(),
            )

    from eo_simulator import PlumeSimulator
    pipeline = EOPipeline(
        provider = PlumeSimulator(),
        engine   = DummyEngine(),
    )

    r = pipeline.run_site(_SAMPLE_REGISTRY[0])
    assert r["inv"] is not None,   "inv が None"
    assert r["inv"]["q"] == 999.0, "カスタムエンジンの q が反映されていない"
    assert r["site"]["id"] == "TM-01"
    print(f"  DummyEngine の q={r['inv']['q']}  → PASS")


def test_run_all() -> None:
    """run_all の統合テスト。"""
    print("\n" + "="*55)
    print("  TEST-4: run_all (4サイト)")
    print("="*55)

    from eo_simulator import PlumeSimulator
    pipeline = EOPipeline(
        provider = PlumeSimulator(),
        engine   = NullInferenceEngine(),
    )
    results = pipeline.run_all(_SAMPLE_REGISTRY)

    assert len(results) == 4, f"結果数が異常: {len(results)}"
    for r in results:
        assert "site"  in r
        assert "mbsp"  in r
        assert "inv"   in r
        assert "flags" in r

    print(f"\n  {'ID':<8} {'Q_true':>8} {'Q_est':>8} {'MLLR':>8} {'p_det':>6}")
    print(f"  {'─'*44}")
    for r in results:
        site = r["site"]
        inv  = r["inv"]
        if inv:
            print(f"  {site['id']:<8} {site['Q_true']:>8.0f} "
                  f"{inv['q']:>8.0f} {inv['mllr']:>8.2f} {inv['p_det']:>6.3f}")
        else:
            print(f"  {site['id']:<8} {site['Q_true']:>8.0f} "
                  f"{'—':>8} {'—':>8} {'—':>6}")
    print("  → PASS")


def test_build_roc() -> None:
    """build_roc の統合テスト。"""
    print("\n" + "="*55)
    print("  TEST-5: build_roc")
    print("="*55)

    from eo_simulator import PlumeSimulator
    pipeline = EOPipeline(
        provider  = PlumeSimulator(),
        engine    = NullInferenceEngine(),
        null_runs = 5,
    )
    results = pipeline.run_all(_SAMPLE_REGISTRY)
    roc     = pipeline.build_roc(results, _SAMPLE_REGISTRY, seed=0)

    assert "fpr" in roc and "tpr" in roc and "auc" in roc
    assert 0.0 <= roc["auc"] <= 1.0,          f"AUC 範囲外: {roc['auc']}"
    assert len(roc["fpr"]) == len(roc["tpr"]), "fpr/tpr 長不一致"
    assert len(roc["positive_mllrs"]) == len(_SAMPLE_REGISTRY)
    print(f"  AUC = {roc['auc']:.4f}")
    print(f"  positive MLLR: {[f'{m:.1f}' for m in roc['positive_mllrs']]}")
    print("  → PASS")


# === エントリポイント ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 55)
    print("  eo_pipeline.py — パイプラインテスト")
    print("=" * 55)
    print("  推論エンジン: NullInferenceEngine (ダミー)")
    print("  データソース: PlumeSimulator (合成)")

    test_compute_mbsp()
    test_null_inference_engine()
    test_inference_engine_protocol()
    test_run_all()
    test_build_roc()

    print("\n" + "="*55)
    print("  全テスト PASS")
    print("  実運用では InferenceEngine を継承した")
    print("  推論エンジンを EOPipeline に差し込んでください。")
    print("="*55)
