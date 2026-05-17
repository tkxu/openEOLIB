"""
ApacheLicense2.0

Copyright (c) 2026 tkxu


 eo_report.py — マスターフィギュア統合出力

 役割:
   eo_visualisation.py の各パネル (SpectralPanel / StatisticalPanel / FlagsPanel)
   を組み合わせてマスターフィギュアを生成・保存する。

   パネルの描画ロジックは eo_visualisation.py に置き、
   このファイルはレイアウト制御とファイル保存のみを担当する。

 分割構成:
   eo_visualisation.py      パネル群 + 共通スタイルユーティリティ
   eo_report_builder.py     (このファイル) ReportBuilder のみ

 使用例:
   from eo_report_builder import ReportBuilder

   builder = ReportBuilder(report_title="openEOLIB Demo")
   fig     = builder.build(results, roc=roc, save_path="report.png")


 依存ライブラリ:
   pip install numpy matplotlib

 関連ファイル:
   eo_types.py          SiteResult / RocData 型定義
   eo_visualisation.py  SpectralPanel / StatisticalPanel / FlagsPanel
   eo_pipeline.py       パイプライン実行
"""
#eo_report.py
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional

from eo_types import (
    RocData,
    SiteResultList,
)
from eo_visualisation import (
    THEME_BG,
    THEME_TEXT,
    DEFAULT_DETECT_THRESH,
    SpectralPanel,
    StatisticalPanel,
    FlagsPanel,
)


# =============================================================================
# ReportBuilder — マスターフィギュア統合出力
# =============================================================================

class ReportBuilder:
    """
    パイプライン実行結果からマスターフィギュアを生成・保存する。

    存在するパネルに応じて自動でセクションを構成する。

        Section 1 : SpectralPanel    (常時表示)
        Section 2 : StatisticalPanel (roc が渡された場合)
        Section 3 : FlagsPanel       (flags を持つ結果が存在する場合)

    Parameters
    ----------
    report_title : フィギュアのタイトル文字列
    detect_thresh: MLLR 検出閾値 (StatisticalPanel に伝達)
    """

    def __init__(
        self,
        report_title:  str   = "Validation Report",
        detect_thresh: float = DEFAULT_DETECT_THRESH,
    ):
        self.report_title  = report_title
        self.detect_thresh = detect_thresh

    def build(
        self,
        results:   SiteResultList,
        roc:       Optional[RocData] = None,
        save_path: Optional[str]     = None,
    ) -> plt.Figure:
        """
        マスターフィギュアを生成する。

        Parameters
        ----------
        results   : ObsPipeline.run_all() の返値
        roc       : ObsPipeline.build_roc() の返値 (None の場合は統計パネルを省略)
        save_path : PNG 保存パス (None の場合は表示のみ)

        Returns
        -------
        plt.Figure
        """
        has_roc   = roc is not None
        has_flags = any(r.get("flags") for r in results)

        # --- セクション構成の決定 ---
        sections      = ["spectral"]
        height_ratios = [1.6]
        if has_roc:
            sections.append("statistical")
            height_ratios.append(1.1)
        if has_flags:
            sections.append("flags")
            height_ratios.append(0.5)

        n_sites = len(results)
        fig_h   = 6 * n_sites * height_ratios[0] / 1.6 + sum(
            h * 6 for h in height_ratios[1:]
        )
        fig = plt.figure(figsize=(20, max(fig_h, 10)))
        fig.patch.set_facecolor(THEME_BG)

        # --- タイトル ---
        fig.text(0.5, 0.995, self.report_title,
                 ha="center", va="top", fontsize=14,
                 fontweight="bold", color=THEME_TEXT)
        site_ids = " · ".join(r["site"]["id"] for r in results)

        # [Fix-B2] SiteResult["meta"] は EOPipeline.run_site() が常に None で返すため
        # bundle["meta"]["backend"] をここから取得できない。
        # backend 情報を表示するには EOPipeline がバンドルのメタを
        # SiteResult に引き渡す設計変更が必要（将来対応）。
        # 暫定として provider_mode を反映した bundle["meta"]["backend"] の
        # 代わりに categories のサマリーを表示する。
        if results:
            categories = sorted({r["site"].get("category", "unknown") for r in results})
            subtitle_right = "categories: " + ", ".join(categories)
        else:
            subtitle_right = ""
        fig.text(0.5, 0.988,
                 f"Sites: {site_ids}   |   {subtitle_right}",
                 ha="center", va="top", fontsize=8, color="#aaaacc")

        # --- セクションラベルの y 座標 ---
        section_labels = {
            "spectral":    ("▌ SPECTRAL",     "#4ecdc4"),
            "statistical": ("▌ STATISTICAL",  "#ff6b35"),
            "flags":       ("▌ FLAGS",         "#fbbf24"),
        }
        label_ys = self._calc_label_ys(height_ratios)
        for sec, ly in zip(sections, label_ys):
            label, color = section_labels[sec]
            fig.text(0.01, ly, label, va="top", fontsize=10,
                     fontweight="bold", color=color)

        # --- アウターグリッド ---
        outer = gridspec.GridSpec(
            len(sections), 1, figure=fig,
            top=0.982, bottom=0.02,
            hspace=0.07,
            height_ratios=height_ratios,
        )

        # --- 各パネルの描画 ---
        for idx, sec in enumerate(sections):
            if sec == "spectral":
                SpectralPanel().draw(fig, outer[idx], results)
            elif sec == "statistical" and roc is not None:
                StatisticalPanel().draw(fig, outer[idx], results, roc)
            elif sec == "flags":
                FlagsPanel().draw(fig, outer[idx], results)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  Figure saved → {save_path}")

        return fig

    @staticmethod
    def _calc_label_ys(height_ratios: List[float]) -> List[float]:
        """セクションラベルの y 座標を height_ratios から計算する。"""
        total = sum(height_ratios)
        ys    = []
        cum   = 0.0
        for h in height_ratios:
            ys.append(0.982 - (cum / total) * 0.962)
            cum += h
        return ys


# =============================================================================
# テストコード (外部API不要)
# =============================================================================

def test_report_builder_structure() -> None:
    """ReportBuilder のセクション構成ロジックをテストする。"""
    print("\n" + "="*55)
    print("  TEST-1: ReportBuilder セクション構成")
    print("="*55)

    builder = ReportBuilder()
    ys_1    = builder._calc_label_ys([1.6])
    ys_3    = builder._calc_label_ys([1.6, 1.1, 0.5])

    assert len(ys_1) == 1
    assert len(ys_3) == 3
    assert ys_3[0] > ys_3[1] > ys_3[2], "y 座標が降順でない"
    print(f"  1セクション: y={ys_1}")
    print(f"  3セクション: y={[f'{y:.3f}' for y in ys_3]}")
    print("  → PASS")


def test_full_render() -> None:
    """PlumeSimulator + NullInferenceEngine + ReportBuilder の結合テスト。"""
    print("\n" + "="*55)
    print("  TEST-2: フルレンダリングテスト")
    print("="*55)

    from eo_simulator  import PlumeSimulator
    from eo_pipeline   import EOPipeline
    from eo_engines    import NullInferenceEngine

    registry = [
        {"id": "TM-01", "name": "Turkmenistan", "lat": 38.49, "lon": 54.19,
         "wind_speed": 4.0, "wind_deg": 120, "Q_true": 4000.0,
         "seed": 42, "category": "super-emitter"},
        {"id": "HN-01", "name": "Hassi R'Mel", "lat": 32.93, "lon": 3.13,
         "wind_speed": 8.0, "wind_deg": 315, "Q_true": 300.0,
         "seed": 99, "category": "near-limit"},
    ]

    pipeline = EOPipeline(
        provider  = PlumeSimulator(),
        engine    = NullInferenceEngine(),
        null_runs = 3,
    )
    results = pipeline.run_all(registry)
    roc     = pipeline.build_roc(results, registry)

    builder = ReportBuilder(report_title="TEST render")
    fig     = builder.build(results, roc=roc, save_path=None)

    assert fig is not None
    assert len(fig.axes) > 0, "Axes が生成されていない"
    print(f"  生成 Axes 数: {len(fig.axes)}")
    plt.close(fig)
    print("  → PASS")


# =============================================================================
# エントリポイント
# =============================================================================

if __name__ == "__main__":

    print("=" * 55)
    print("  eo_report_builder.py — ReportBuilder テスト")
    print("=" * 55)

    test_report_builder_structure()
    test_full_render()

    print("\n" + "="*55)
    print("  全テスト PASS")
    print("="*55)

    # --- デモ出力 ---
    print("\nデモレポートを生成します ...")

    from eo_simulator  import PlumeSimulator
    from eo_pipeline   import EOPipeline, NullInferenceEngine

    _DEMO_REGISTRY = [
        {"id": "TM-01", "name": "Turkmenistan Compressor Station",
         "lat": 38.49, "lon": 54.19, "wind_speed": 4.0, "wind_deg": 120,
         "Q_true": 4000.0, "seed": 42, "category": "super-emitter"},
        {"id": "PB-01", "name": "Permian Basin Wellpad",
         "lat": 31.83, "lon": -102.37, "wind_speed": 6.5, "wind_deg": 200,
         "Q_true": 800.0, "seed": 7, "category": "mid-range"},
        {"id": "DZ-01", "name": "Algeria In Salah",
         "lat": 27.21, "lon": 2.52, "wind_speed": 3.0, "wind_deg": 45,
         "Q_true": 1800.0, "seed": 13, "category": "mid-range"},
        {"id": "HN-01", "name": "Hassi R'Mel Flare Station",
         "lat": 32.93, "lon": 3.13, "wind_speed": 8.0, "wind_deg": 315,
         "Q_true": 300.0, "seed": 99, "category": "near-limit"},
    ]

    pipeline = EOPipeline(
        provider  = PlumeSimulator(mismatch=True, gp_noise=True),
        engine    = NullInferenceEngine(),
        null_runs = 10,
    )
    results = pipeline.run_all(_DEMO_REGISTRY)
    roc     = pipeline.build_roc(results, _DEMO_REGISTRY)

    fig = ReportBuilder(
        report_title="openEOLIB Demo Report"
    ).build(
        results,
        roc       = roc,
        save_path = "openEOLIB_report.png",
    )
    plt.show()
