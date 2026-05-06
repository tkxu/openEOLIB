"""
ApacheLicense2.0

Copyright (c) 2026 tkxu

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   
 eo_pipeline.py — 観測データ処理パイプライン


 コンポーネント:
   InferenceEngine      推論エンジンの抽象基底クラス (プロトコル定義)
   NullInferenceEngine  テスト用ダミー推論エンジン
   EOPipeline           パイプライン本体

 依存ライブラリ:
   pip install numpy scipy

 関連ファイル:
   eo_types.py       型定義
   eo_provider.py    実データ取得
   eo_simulator.py   合成データ生成

"""
#eo_pipeline.py
from __future__ import annotations

import warnings
import numpy as np
from abc import ABC, abstractmethod
from scipy import ndimage as ndi
from typing import Dict, List, Optional

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


# =============================================================================
# 定数
# =============================================================================

# ROC 構築用 MLLR スイープ範囲
MLLR_THRESHOLDS = np.linspace(-50, 300, 120)

# ヌル試行回数 (ROC 構築用)
NULL_RUNS_PER_SITE = 30

# デフォルトの空品質フラグ (推論エンジンが None を返した場合に使用)
_DEFAULT_FLAGS: QualityFlags = {
    "low_wind":          False,
    "multi_modal_theta": False,
    "roi_unstable":      False,
    "template_dominant": False,
}


# =============================================================================
# MBSP 計算 (共有ユーティリティ)
# =============================================================================

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


# =============================================================================
# 推論エンジンの抽象基底クラス (プロトコル定義)
# =============================================================================

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


# =============================================================================
# テスト用ダミー推論エンジン
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
        """MBSP のピーク値から簡易的な Q を推定する。"""
        mu  = float(np.nanmedian(mbsp))
        sig = float(1.4826 * np.nanmedian(np.abs(mbsp - mu)))
        z   = (mbsp - mu) / (sig + 1e-8)

        if float(np.nanmax(z)) < 2.0:
            return None

        # 簡易 MLLR: Z スコアの積分値
        mllr = float(np.nansum(np.maximum(z - 2.0, 0)))
        q    = mllr / (sig + 1e-8) * self.q_scale / 100.0

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


# =============================================================================
# ObsPipeline — パイプライン本体
# =============================================================================

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
                     DataFetcher (eo_provider.py) または
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
        provider,
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
    ) -> tuple:
        """
        バンドルから風速・風向を取得する。

        実データ (openEO) では bundle の値を使用する。
        合成データ (synthetic) または取得失敗時は site の公称値にフォールバック。
        """
        ws  = bundle.get("wind_speed")
        wd  = bundle.get("wind_deg")
        src = bundle.get("meta", {}).get("backend", "unknown")

        if ws is None or wd is None:
            if src != "synthetic":
                warnings.warn(
                    f"[{site['id']}] 風速・風向の取得に失敗。"
                    f"site の公称値にフォールバックします。",
                    stacklevel=3,
                )
            ws = site["wind_speed"]
            wd = site["wind_deg"]

        return float(ws), float(wd)

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def run_site(
        self,
        site: SiteEntry,
        dt=None,
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
        meta    = site.get("meta", {})
        backend = getattr(self.provider, '__class__', None)

        # provider の種別を自動判定して適切なメソッドを呼ぶ
        if hasattr(self.provider, "generate"):
            # PlumeSimulator
            bundle: ObservationBundle = self.provider.generate(
                Q          = site["Q_true"],
                wind_speed = site["wind_speed"],
                wind_deg   = site["wind_deg"],
                lat        = site["lat"],
                lon        = site["lon"],
                seed       = site.get("seed"),
            )
        elif hasattr(self.provider, "fetch_from_site") and dt is not None:
            # DataFetcher (earth_obs_provider.py)
            bundle = self.provider.fetch_from_site(site, dt)
        elif hasattr(self.provider, "fetch") and dt is not None:
            bundle = self.provider.fetch(site["lat"], site["lon"], dt)
        else:
            raise ValueError(
                "provider は generate() または fetch() / fetch_from_site() を"
                "持つオブジェクトである必要があります。"
            )

        # --- MBSP 計算 ---
        mbsp = self._bundle_to_mbsp(bundle)
        if mbsp is None:
            warnings.warn(f"[{site['id']}] MBSP の計算に失敗しました。", stacklevel=2)
            mbsp = np.zeros((100, 100), dtype=np.float32)

        wind_speed, wind_deg = self._extract_wind(bundle, site)

        # --- 推論エンジンに委譲 ---
        inv_result = self.engine.infer(mbsp, wind_speed, wind_deg)

        flags: QualityFlags = (
            inv_result["flags"]
            if inv_result is not None
            else _DEFAULT_FLAGS.copy()
        )

        # --- plume_true の取得 (合成データのみ) ---
        plume_true = bundle.get("plume_true", np.zeros_like(mbsp))

        # --- 風向ベクトル ---
        from eo_simulator import _wind_deg_to_vec
        wvec = _wind_deg_to_vec(wind_deg)

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
        )

    def run_all(
        self,
        site_registry: List[SiteEntry],
        dt=None,
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
            print(f"  [{site['id']}] {site['name']} ...", end=" ", flush=True)
            r      = self.run_site(site, dt=dt)
            status = "OK" if r["inv"] is not None else "NO DETECT"
            q_str  = f"Q={r['inv']['q']:.0f}" if r["inv"] else "—"
            print(f"{status}  {q_str} kg/h")
            results.append(r)
        return results


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
    roc     = pipeline.build_roc(results, _SAMPLE_REGISTRY)

    assert "fpr" in roc and "tpr" in roc and "auc" in roc
    assert 0.0 <= roc["auc"] <= 1.0, f"AUC 範囲外: {roc['auc']}"
    assert len(roc["fpr"]) == len(MLLR_THRESHOLDS)
    print(f"  AUC = {roc['auc']:.4f}")
    print(f"  positive MLLR: {roc['positive_mllrs']}")
    print("  → PASS")


# =============================================================================
# エントリポイント
# =============================================================================

if __name__ == "__main__":

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
