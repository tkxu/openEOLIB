"""
ApacheLicense2.0

Copyright (c) 2026 tkxu


 eo_visualisation.py — 観測データ・パイプライン結果の可視化パネル群

 役割:
   eo_pipeline.py の SiteResult と RocData を受け取り、
   物理・統計の両面からフィギュアを生成する個別パネルを提供する。

   コアアルゴリズム (推論エンジン) への依存をゼロにするため、
   scoring / phys_result は Optional として扱い、
   存在する場合のみ対応するパネルを描画する。

 コンポーネント:
   eo_visualisation.py  (このファイル)
     ダークテーマ定数・共通スタイル関数
     SpectralPanel    空間マップ (プルーム / MBSP)
     StatisticalPanel 統計サマリー (Q推定 / MLLR / ROC / 分布)
     FlagsPanel       品質フラグ可視化 (QualityFlags)

   eo_report_builder.py
     ReportBuilder    マスターフィギュア統合出力

 依存ライブラリ:
   pip install numpy matplotlib

 関連ファイル:
   eo_types.py           SiteResult / RocData 型定義
   eo_pipeline.py        パイプライン実行
   eo_report_builder.py  統合レポート出力
"""
#eo_visualisation.py
from __future__ import annotations

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional

from eo_types import (
    RocData,
    SiteEntry,
    SiteResult,
    SiteResultList,
)


# =============================================================================
# ダークテーマ共通定数
# =============================================================================

_BG          = "#0d1117"
_PANEL_BG    = "#0d1117"
_GRID_COLOR  = "#333344"
_TEXT_WHITE  = "white"

# カテゴリ別カラー
_CATEGORY_COLOR: Dict[str, str] = {
    "super-emitter": "#ff6b35",
    "mid-range":     "#4ecdc4",
    "near-limit":    "#ffe66d",
}

# Tier 別カラー (コアアルゴリズムが scoring を返す場合のみ使用)
_TIER_COLOR: Dict[str, str] = {
    "Tier-A": "#00ff88",
    "Tier-B": "#4ecdc4",
    "Tier-C": "#ffe66d",
    "None":   "#ff4444",
}

# 品質フラグの定義 (FlagsPanel で使用)
_FLAG_KEYS   = ["low_wind", "multi_modal_theta", "roi_unstable", "template_dominant"]
_FLAG_LABELS = ["Low wind", "Multi-modal θ", "ROI unstable", "Template dom."]

# デフォルト検出閾値 (MLLR)
_DETECT_THRESH = 20.0

# =============================================================================
# 公開定数 (eo_report_builder.py など外部モジュールが使用する定数)
#
# _プレフィックス付きの内部定数を外部から直接 import することを避けるため、
# パブリックな名前でエイリアスを定義する。
# 内部コードはそのまま _BG 等を使用し続けることができる。
# =============================================================================

THEME_BG           = _BG          # ダークテーマ背景色
THEME_TEXT          = _TEXT_WHITE  # テキスト色
DEFAULT_DETECT_THRESH = _DETECT_THRESH  # デフォルト MLLR 検出閾値


# =============================================================================
# 共通スタイルユーティリティ
# =============================================================================

def _style_ax(ax: plt.Axes) -> None:
    """全サブプロットに共通のダークスタイルを適用する。"""
    ax.set_facecolor(_PANEL_BG)
    ax.tick_params(colors=_TEXT_WHITE, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID_COLOR)
    ax.yaxis.label.set_color(_TEXT_WHITE)
    ax.xaxis.label.set_color(_TEXT_WHITE)
    ax.title.set_color(_TEXT_WHITE)


def _category_color(site: SiteEntry) -> str:
    """カテゴリに対応するカラーコードを返す。"""
    return _CATEGORY_COLOR.get(site.get("category", ""), "#aaaaaa")


def _wind_arrow(ax: plt.Axes, wvec: np.ndarray, cx: int = 50, cy: int = 50) -> None:
    """フィールド画像上に風向矢印を描画する。"""
    dx =  wvec[0] * 14
    dy = -wvec[1] * 14   # 画像座標は y 軸が反転
    ax.annotate(
        "", xy=(cx + dx, cy + dy), xytext=(cx, cy),
        arrowprops=dict(arrowstyle="-|>", color="cyan", lw=1.8),
    )
    ax.text(cx + dx * 1.20, cy + dy * 1.20, "wind",
            color="cyan", fontsize=6, ha="center", va="center")


def _table_style(tbl, n_rows: int, n_cols: int, tc_list: List[str]) -> None:
    """
    matplotlib テーブルに共通スタイルを適用する。

    Parameters
    ----------
    tbl      : matplotlib Table オブジェクト
    n_rows   : データ行数 (ヘッダー除く)
    n_cols   : 列数
    tc_list  : 各行のテキストカラーリスト (長さ n_rows)
    """
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.55)
    for j in range(n_cols):
        tbl[0, j].set_facecolor("#2a2a3e")
        tbl[0, j].set_text_props(color=_TEXT_WHITE, fontweight="bold")
    for i in range(n_rows):
        fc = "#1a1a2e" if i % 2 == 0 else "#16213e"
        tc = tc_list[i] if i < len(tc_list) else "#aaaaaa"
        for j in range(n_cols):
            tbl[i + 1, j].set_facecolor(fc)
            tbl[i + 1, j].set_text_props(color=tc)



# =============================================================================
# SpectralPanel — 空間マップ (サイト × 2 列)
# =============================================================================

class SpectralPanel:
    """
    サイトごとに空間フィールドを2列で描画する。

        Col 0 : 真のプルーム濃度場 + 風向矢印
        Col 1 : MBSP = log(B11) - log(B12)

    LLR マップ・検出マスクはコアアルゴリズム依存のため省略し、
    MBSP フィールドを代わりに表示する。
    推論エンジンが inv を返した場合は推定 Q を注釈する。
    """

    CMAP_PLUME = "hot_r"
    CMAP_MBSP  = "RdBu_r"

    def draw(
        self,
        fig:     plt.Figure,
        outer_gs,
        results: SiteResultList,
    ) -> None:
        n = len(results)

        #  ヘッダー行(row 0)とサイト行(row 1..n)を分離する。
        #
        # n+1 行グリッドを確保し、row 0 をヘッダー専用行とする。
        #         ヘッダー行の高さ比率を小さく (0.08) 設定してスペースを節約する。
        #         サイトは row 1..n に配置するため、インデックスは inner[i+1, j]。
        inner = gridspec.GridSpecFromSubplotSpec(
            n + 1, 2,
            subplot_spec  = outer_gs,
            hspace        = 0.08,
            wspace        = 0.05,
            height_ratios = [0.08] + [1.0] * n,  # row 0: ヘッダー / row 1..n: サイト
        )

        # --- 列ヘッダー (row 0 専用、サイト行と衝突しない) ---
        for j, title in enumerate(["True plume field [g/m²]", "MBSP  (log B11 − log B12)"]):
            ax = fig.add_subplot(inner[0, j])
            ax.text(0.5, 0.5, title,
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color=_TEXT_WHITE,
                    transform=ax.transAxes)
            ax.axis("off")
            ax.set_facecolor(_PANEL_BG)

        # --- サイト行 (row 1..n) ---
        for i, r in enumerate(results):
            site = r["site"]
            tag  = f"{site['id']}\n{site.get('category', '')}"

            # --- Col 0: 真のプルーム ---
            ax0  = fig.add_subplot(inner[i + 1, 0])
            plume = r.get("plume_true", np.zeros((100, 100)))
            pm   = ax0.imshow(plume, cmap=self.CMAP_PLUME, origin="upper")
            plt.colorbar(pm, ax=ax0, fraction=0.046, pad=0.04, label="[g/m²]")
            wvec = r.get("wvec")
            if wvec is not None:
                _wind_arrow(ax0, wvec)
            ax0.set_ylabel(tag, fontsize=8, color=_TEXT_WHITE)
            ax0.set_xticks([])
            ax0.set_yticks([])
            ax0.set_facecolor(_PANEL_BG)

            # --- Col 1: MBSP ---
            ax1  = fig.add_subplot(inner[i + 1, 1])
            mbsp = r.get("mbsp", np.zeros((100, 100)))
            vmax = float(np.nanpercentile(np.abs(mbsp), 98))
            mm   = ax1.imshow(mbsp, cmap=self.CMAP_MBSP,
                               vmin=-vmax, vmax=vmax, origin="upper")
            plt.colorbar(mm, ax=ax1, fraction=0.046, pad=0.04, label="MBSP")

            # 推定結果の注釈 (inv があれば表示)
            inv = r.get("inv")
            if inv is not None:
                label  = (f"Q={inv['q']:.0f}±{inv['q_std']:.0f} kg/h\n"
                          f"MLLR={inv['mllr']:.1f}")
                tcolor = "#00ff88"
            else:
                label  = "NO DETECT"
                tcolor = "#ff4444"

            ax1.text(2, 6, label, fontsize=7, color=tcolor, va="top",
                     bbox=dict(fc="black", alpha=0.55, pad=2, lw=0))
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_facecolor(_PANEL_BG)


# =============================================================================
# StatisticalPanel — 統計サマリー (2×3 グリッド)
# =============================================================================

class StatisticalPanel:
    """
    クロスサイト統計サマリー。

        A : Q推定値 vs 真値 (エラーバー付き棒グラフ)
        B : MLLR 棒グラフ + 検出閾値線
        C : Q 相対誤差 + 不確実性
        D : ROC 曲線 + AUC
        E : MLLR 分布 (H1 vs H0)
        F : サマリーテーブル

    scoring が存在する場合は Tier 情報を追加表示する。
    """

    def draw(
        self,
        fig:     plt.Figure,
        outer_gs,
        results: SiteResultList,
        roc:     RocData,
    ) -> None:
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=outer_gs, hspace=0.45, wspace=0.38
        )
        self._q_estimate(    fig, inner[0, 0], results)
        self._mllr_bar(      fig, inner[0, 1], results)
        self._q_uncertainty( fig, inner[0, 2], results)
        self._roc(           fig, inner[1, 0], roc)
        self._mllr_dist(     fig, inner[1, 1], roc)
        self._summary_table( fig, inner[1, 2], results)

    # ------------------------------------------------------------------

    def _q_estimate(self, fig, gs, results: SiteResultList) -> None:
        ax = fig.add_subplot(gs)
        ids, q_true_l, q_est_l, q_std_l, colors = [], [], [], [], []
        for r in results:
            ids.append(r["site"]["id"])
            q_true_l.append(r["site"]["Q_true"])
            colors.append(_category_color(r["site"]))
            q_est_l.append(r["inv"]["q"]     if r["inv"] else 0.0)
            q_std_l.append(r["inv"]["q_std"] if r["inv"] else 0.0)

        x, w = np.arange(len(ids)), 0.35
        ax.bar(x - w/2, q_true_l, w, color=colors, alpha=0.45,
               edgecolor="white", lw=0.8, label="Q_true")
        ax.bar(x + w/2, q_est_l,  w, color=colors, alpha=0.9,
               edgecolor="white", lw=0.8, label="Q_est",
               yerr=q_std_l, capsize=4,
               error_kw=dict(ecolor="white", lw=1.2))
        ax.set_xticks(x)
        ax.set_xticklabels(ids, fontsize=8)
        ax.set_ylabel("Q [kg/h]", fontsize=8)
        ax.set_title("A  |  Q estimation vs ground truth",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        _style_ax(ax)

    def _mllr_bar(self, fig, gs, results: SiteResultList) -> None:
        ax = fig.add_subplot(gs)
        ids, mllrs, colors = [], [], []
        for r in results:
            ids.append(r["site"]["id"])
            mllrs.append(r["inv"]["mllr"] if r["inv"] else 0.0)
            colors.append(_category_color(r["site"]))

        x = np.arange(len(ids))
        ax.bar(x, mllrs, color=colors, alpha=0.85, edgecolor="white", lw=0.8)
        ax.axhline(_DETECT_THRESH, color="#ff4444", lw=1.5, ls="--",
                   label=f"thresh={_DETECT_THRESH}")
        ax.set_xticks(x)
        ax.set_xticklabels(ids, fontsize=8)
        ax.set_ylabel("MLLR", fontsize=8)
        ax.set_title("B  |  MLLR per site", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        _style_ax(ax)

    def _q_uncertainty(self, fig, gs, results: SiteResultList) -> None:
        ax = fig.add_subplot(gs)
        ids, rel_err, rel_std, colors = [], [], [], []
        for r in results:
            ids.append(r["site"]["id"])
            q_t = r["site"]["Q_true"]
            colors.append(_category_color(r["site"]))
            if r["inv"] is not None:
                rel_err.append((r["inv"]["q"] - q_t) / q_t * 100.0)
                rel_std.append(r["inv"]["q_std"]      / q_t * 100.0)
            else:
                rel_err.append(np.nan)
                rel_std.append(np.nan)

        x = np.arange(len(ids))
        ax.bar(x, rel_err, color=colors, alpha=0.75, edgecolor="white", lw=0.8)
        ax.errorbar(x, rel_err, yerr=rel_std, fmt="none",
                    ecolor="white", capsize=4, lw=1.2)
        ax.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(ids, fontsize=8)
        ax.set_ylabel("Relative error [%]", fontsize=8)
        ax.set_title("C  |  Q uncertainty (σ_Q / Q_true)",
                     fontsize=9, fontweight="bold")
        _style_ax(ax)

    def _roc(self, fig, gs, roc: RocData) -> None:
        ax = fig.add_subplot(gs)
        ax.plot(roc["fpr"], roc["tpr"], color="#4ecdc4", lw=2.0,
                label=f"AUC = {roc['auc']:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--", label="Random")
        ax.fill_between(roc["fpr"], roc["tpr"], alpha=0.15, color="#4ecdc4")
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title("D  |  ROC curve (MLLR threshold sweep)",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        _style_ax(ax)

    def _mllr_dist(self, fig, gs, roc: RocData) -> None:
        ax   = fig.add_subplot(gs)
        pos  = [m for m in roc["positive_mllrs"] if np.isfinite(m)]
        null = [m for m in roc["null_mllrs"]     if np.isfinite(m)]
        all_vals = pos + null
        if not all_vals:
            _style_ax(ax)
            return
        bins = np.linspace(min(all_vals) - 5, max(all_vals) + 5, 40)
        if pos:
            ax.hist(pos,  bins=bins, color="#ff6b35", alpha=0.7,
                    label="Plume (H1)", density=True)
        if null:
            ax.hist(null, bins=bins, color="#888888", alpha=0.6,
                    label="Null (H0)",  density=True)
        ax.axvline(_DETECT_THRESH, color="#ff4444", lw=1.5, ls="--",
                   label=f"thresh={_DETECT_THRESH}")
        ax.set_xlabel("MLLR", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title("E  |  MLLR distribution: H1 vs H0",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        _style_ax(ax)

    def _summary_table(self, fig, gs, results: SiteResultList) -> None:
        ax   = fig.add_subplot(gs)
        ax.axis("off")

        # scoring があれば Tier 列を追加
        has_scoring = any(r.get("scoring") for r in results)
        cols = (["Site", "Q_true", "Q_est", "Err%", "MLLR", "p_det", "Tier"]
                if has_scoring
                else ["Site", "Q_true", "Q_est", "Err%", "MLLR", "p_det"])

        rows, tc_list = [], []
        for r in results:
            site, inv = r["site"], r["inv"]
            q_t  = site["Q_true"]
            sc   = r.get("scoring")
            tier = sc["tier"] if sc else "—"

            if inv is not None:
                err = f"{(inv['q'] - q_t) / q_t * 100:+.1f}"
                row = [site["id"], f"{q_t:.0f}", f"{inv['q']:.0f}",
                       err, f"{inv['mllr']:.1f}", f"{inv['p_det']:.2f}"]
                if has_scoring:
                    row.append(tier)
                tc_list.append(_TIER_COLOR.get(tier, "#00ff88") if has_scoring
                                else "#00ff88")
            else:
                row = [site["id"], f"{q_t:.0f}", "—", "—", "—", "—"]
                if has_scoring:
                    row.append("—")
                tc_list.append("#ff4444")
            rows.append(row)

        tbl = ax.table(cellText=rows, colLabels=cols,
                       loc="center", cellLoc="center")
        _table_style(tbl, len(rows), len(cols), tc_list)
        ax.set_title("F  |  Summary table",
                     fontsize=9, fontweight="bold", pad=10)


# =============================================================================
# FlagsPanel — 品質フラグ可視化
# =============================================================================

class FlagsPanel:
    """
    推論エンジンが返す QualityFlags をサイト横断で可視化する。

    QualityFlags 型 (obs_types.py) だけに依存するため、
    どの推論エンジンを使っても同じパネルが描画できる。

        J : フラグヒートマップ (サイト × フラグ種別)
        K : p_det 棒グラフ (検出確率)
        L : MLLR 棒グラフ (ROI 安定性の代理指標)
    """

    def draw(
        self,
        fig:     plt.Figure,
        outer_gs,
        results: SiteResultList,
    ) -> None:
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer_gs, hspace=0.30, wspace=0.38
        )
        self._flag_heatmap(fig, inner[0, 0], results)
        self._p_det_bar(   fig, inner[0, 1], results)
        self._mllr_bar(    fig, inner[0, 2], results)

    def _flag_heatmap(self, fig, gs, results: SiteResultList) -> None:
        ax  = fig.add_subplot(gs)
        ids = [r["site"]["id"] for r in results]
        mat = np.zeros((len(results), len(_FLAG_KEYS)))

        for i, r in enumerate(results):
            flags = r.get("flags") or {}
            for j, key in enumerate(_FLAG_KEYS):
                mat[i, j] = 1.0 if flags.get(key, False) else 0.0

        im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=1,
                       aspect="auto", origin="upper")
        ax.set_xticks(range(len(_FLAG_KEYS)))
        ax.set_xticklabels(_FLAG_LABELS, fontsize=7, rotation=20, ha="right")
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels(ids, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Flag fired")
        ax.set_title("J  |  Quality flags", fontsize=9, fontweight="bold")
        ax.set_facecolor(_PANEL_BG)
        ax.title.set_color(_TEXT_WHITE)
        ax.tick_params(colors=_TEXT_WHITE, labelsize=7)

    def _p_det_bar(self, fig, gs, results: SiteResultList) -> None:
        ax = fig.add_subplot(gs)
        ids, p_dets, colors = [], [], []
        for r in results:
            ids.append(r["site"]["id"])
            inv = r.get("inv")
            p_dets.append(float(inv["p_det"]) if inv else 0.0)
            colors.append(_category_color(r["site"]))

        x = np.arange(len(ids))
        ax.bar(x, p_dets, color=colors, alpha=0.85, edgecolor="white", lw=0.8)
        ax.axhline(0.5, color="#ff4444", lw=1.2, ls="--", label="P=0.5")
        ax.set_xticks(x)
        ax.set_xticklabels(ids, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("p_det", fontsize=8)
        ax.set_title("K  |  Detection probability",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        _style_ax(ax)

    def _mllr_bar(self, fig, gs, results: SiteResultList) -> None:
        ax = fig.add_subplot(gs)
        ids, mllrs, colors = [], [], []
        for r in results:
            ids.append(r["site"]["id"])
            inv    = r.get("inv")
            flags  = r.get("flags") or {}
            mllrs.append(float(inv["mllr"]) if inv else 0.0)
            # roi_unstable フラグが立っているサイトを赤で強調
            colors.append(
                "#ff4444" if flags.get("roi_unstable", False)
                else _category_color(r["site"])
            )

        x = np.arange(len(ids))
        ax.bar(x, mllrs, color=colors, alpha=0.85, edgecolor="white", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ids, fontsize=8)
        ax.set_ylabel("MLLR (best ROI)", fontsize=8)
        ax.set_title("L  |  MLLR stability  (red = roi_unstable)",
                     fontsize=9, fontweight="bold")
        _style_ax(ax)


# =============================================================================
# テストコード (外部API不要)
# =============================================================================

def test_style_utils() -> None:
    """スタイルユーティリティの単体テスト。"""
    print("\n" + "="*55)
    print("  TEST-1: スタイルユーティリティ")
    print("="*55)

    from eo_types import SiteEntry
    site: SiteEntry = {
        "id": "TM-01", "name": "test",
        "lat": 0.0, "lon": 0.0,
        "wind_speed": 4.0, "wind_deg": 120.0,
        "Q_true": 4000.0, "seed": 42,
        "category": "super-emitter",
    }
    assert _category_color(site)                           == "#ff6b35"
    assert _category_color({"category": "mid-range"})     == "#4ecdc4"
    assert _category_color({"category": "near-limit"})    == "#ffe66d"
    assert _category_color({"category": "unknown"})       == "#aaaaaa"
    print("  _category_color → PASS")

    assert _TIER_COLOR["Tier-A"] == "#00ff88"
    assert _TIER_COLOR["None"]   == "#ff4444"
    print("  _TIER_COLOR → PASS")

    print("  → PASS")


# =============================================================================
# エントリポイント
# =============================================================================

if __name__ == "__main__":

    print("=" * 55)
    print("  eo_visualisation.py — 可視化パネルモジュールテスト")
    print("=" * 55)

    test_style_utils()

    print("\n" + "="*55)
    print("  全テスト PASS")
    print("  統合レポートのテスト・デモは eo_report.py を参照。")
    print("="*55)
