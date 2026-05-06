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

 eo_visualisation.py — 観測データ・パイプライン結果の可視化

 役割:
   eo_pipeline.py の SiteResult と RocData を受け取り、
   物理・統計の両面からフィギュアを生成する。

   コアアルゴリズム (推論エンジン) への依存をゼロにするため、
   scoring / phys_result は Optional として扱い、
   存在する場合のみ対応するパネルを描画する。

 コンポーネント:
   ダークテーマ定数・共通スタイル関数
   SpectralPanel    空間マップ (プルーム / MBSP)
   StatisticalPanel 統計サマリー (Q推定 / MLLR / ROC / 分布)
   FlagsPanel       品質フラグ可視化 (QualityFlags)
   ReportBuilder    マスターフィギュア統合出力

 ファイル名:


 依存ライブラリ:
   pip install numpy matplotlib

 関連ファイル:
eo_types.py      SiteResult / RocData 型定義
eo_pipeline.py   パイプライン実行
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
        n     = len(results)
        inner = gridspec.GridSpecFromSubplotSpec(
            n, 2, subplot_spec=outer_gs, hspace=0.08, wspace=0.05
        )

        # 列ヘッダー
        for j, title in enumerate(["True plume field [g/m²]", "MBSP  (log B11 − log B12)"]):
            ax = fig.add_subplot(inner[0, j])
            ax.set_title(title, fontsize=9, fontweight="bold",
                         color=_TEXT_WHITE, pad=5)
            ax.axis("off")

        for i, r in enumerate(results):
            site = r["site"]
            tag  = f"{site['id']}\n{site.get('category', '')}"

            # --- Col 0: 真のプルーム ---
            ax0  = fig.add_subplot(inner[i, 0])
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
            ax1  = fig.add_subplot(inner[i, 1])
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
# ReportBuilder — マスターフィギュア統合出力
# =============================================================================

class ReportBuilder:
    """
    パイプライン実行結果からマスターフィギュアを生成・保存する。

    存在するパネルに応じて自動でセクションを構成する。

        Section 1 : SpectralPanel   (常時表示)
        Section 2 : StatisticalPanel (roc が渡された場合)
        Section 3 : FlagsPanel      (flags を持つ結果が存在する場合)

    Parameters
    ----------
    report_title : フィギュアのタイトル文字列
    detect_thresh: MLLR 検出閾値 (StatisticalPanel に伝達)
    """

    def __init__(
        self,
        report_title:  str   = "earth-obs-toolkit — Validation Report",
        detect_thresh: float = _DETECT_THRESH,
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
        fig.patch.set_facecolor(_BG)

        # --- タイトル ---
        fig.text(0.5, 0.995, self.report_title,
                 ha="center", va="top", fontsize=14,
                 fontweight="bold", color=_TEXT_WHITE)
        site_ids = " · ".join(r["site"]["id"] for r in results)
        backend  = (results[0].get("meta", {}).get("backend", "unknown")
                    if results else "unknown")
        fig.text(0.5, 0.988,
                 f"Sites: {site_ids}   |   backend: {backend}",
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


def test_report_builder_structure() -> None:
    """ReportBuilder のセクション構成ロジックをテストする。"""
    print("\n" + "="*55)
    print("  TEST-2: ReportBuilder セクション構成")
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
    print("  TEST-3: フルレンダリングテスト")
    print("="*55)

    from eo_simulator  import PlumeSimulator
    from eo_pipeline   import EOPipeline, NullInferenceEngine
    from eo_types      import SiteEntry

    registry = [
        SiteEntry(id="TM-01", name="Turkmenistan", lat=38.49, lon=54.19,
                  wind_speed=4.0, wind_deg=120, Q_true=4000.0,
                  seed=42, category="super-emitter"),
        SiteEntry(id="HN-01", name="Hassi R'Mel", lat=32.93, lon=3.13,
                  wind_speed=8.0, wind_deg=315, Q_true=300.0,
                  seed=99, category="near-limit"),
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
    print("  eo_visualisation.py — 可視化モジュールテスト")
    print("=" * 55)

    test_style_utils()
    test_report_builder_structure()
    test_full_render()

    print("\n" + "="*55)
    print("  全テスト PASS")
    print("="*55)

    # --- デモ出力 ---
    print("\nデモレポートを生成します ...")

    from eo_simulator  import PlumeSimulator
    from eo_pipeline   import EOPipeline, NullInferenceEngine
    from eo_types      import SiteEntry

    _DEMO_REGISTRY = [
        SiteEntry(id="TM-01", name="Turkmenistan Compressor Station",
                  lat=38.49, lon=54.19, wind_speed=4.0, wind_deg=120,
                  Q_true=4000.0, seed=42, category="super-emitter"),
        SiteEntry(id="PB-01", name="Permian Basin Wellpad",
                  lat=31.83, lon=-102.37, wind_speed=6.5, wind_deg=200,
                  Q_true=800.0, seed=7, category="mid-range"),
        SiteEntry(id="DZ-01", name="Algeria In Salah",
                  lat=27.21, lon=2.52, wind_speed=3.0, wind_deg=45,
                  Q_true=1800.0, seed=13, category="mid-range"),
        SiteEntry(id="HN-01", name="Hassi R'Mel Flare Station",
                  lat=32.93, lon=3.13, wind_speed=8.0, wind_deg=315,
                  Q_true=300.0, seed=99, category="near-limit"),
    ]

    pipeline = EOPipeline(
        provider  = PlumeSimulator(mismatch=True, gp_noise=True),
        engine    = NullInferenceEngine(),
        null_runs = 10,
    )
    results = pipeline.run_all(_DEMO_REGISTRY)
    roc     = pipeline.build_roc(results, _DEMO_REGISTRY)

    ReportBuilder(
        report_title="earth-obs-toolkit — Demo Report"
    ).build(
        results,
        roc       = roc,
        save_path = "eo_toolkit_demo_report.png",
    )
    plt.show()
