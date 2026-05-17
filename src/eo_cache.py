"""
ApacheLicense2.0
Copyright (c) 2026 tkxu

eo_cache.py — キャッシュ + Provenance 統合モジュール

役割:
    DataFetcher が取得した ObservationBundle をキャッシュし、
    同一フィンガープリントのリクエストに対して再計算を省略する。

    Provenance（来歴）とキャッシュを1ファイルに同居させることで、
    インポートパスをシンプルに保つ。

        from eo_cache import ScienceCacheStore, DatasetProvenance, make_input_hash

モジュール構成:
    ┌─────────────────────────────────────────────────────────┐
    │  Section 1 — Provenance                                 │
    │      ProvenanceRecord   単一オペレータの実行記録          │
    │      DatasetProvenance  バンドル全体の来歴               │
    │      make_input_hash()  座標・日時 → SHA-256            │
    │                                                         │
    │  Section 2 — Cache                                      │
    │      CacheStats         ヒット率計測                     │
    │      ScienceCacheStore  ファイルシステムキャッシュ        │
    └─────────────────────────────────────────────────────────┘

ストレージ形式 (現在):
    pickle (.pkl) を使用する。

    NOTE — 将来の移行計画:
        pickle.load() は任意コードを実行できるため、
        信頼できない共有パス (NFS / Docker volume / CI artifact) では
        セキュリティリスクになる。

        将来のリリースで np.savez_compressed (.npz) または zarr への
        移行を予定している。移行後は allow_pickle=False でロードすることで
        このリスクを排除する。

        移行時の変更範囲はこのファイルの Section 2 のみ。
        Section 1 (Provenance) と公開 API は変更しない。

    現時点でのリスク軽減策:
        - キャッシュディレクトリを信頼できるパスにのみ配置する
        - CI 環境ではキャッシュを使わない (cache_dir=None)
        - 本番環境では ScienceCacheStore を直接使わず DataFetcher 経由で制御する


使用例:
    store = ScienceCacheStore(".science_cache")
    key   = provenance.fingerprint()

    cached = store.get(key)
    if cached is not None:
        return cached           # キャッシュヒット

    bundle = ...                # 実際のデータ取得
    store.set(key, bundle)
    return bundle
"""
#eo_cache.py
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)


# =============================================================================
# Section 1 — Provenance
# =============================================================================

@dataclass(frozen=True)
class ProvenanceRecord:
    """
    単一オペレータの実行記録。

    Fields
    ------
    operator  : オペレータ名 (例: "Sentinel2Fetcher.fetch")
    params    : オペレータのパラメータ辞書 (JSON シリアライズ可能な値のみ)
    timestamp : 実行日時 ISO8601 文字列

    注意:
        timestamp は DatasetProvenance.fingerprint() に含まれない。
        詳細は DatasetProvenance.fingerprint() の docstring を参照。
    """
    operator:  str
    params:    Dict[str, Any]
    timestamp: str


@dataclass(frozen=True)
class DatasetProvenance:
    """
    ObservationBundle 全体の来歴。

    Fields
    ------
    inputs_hash     : 入力空間 (lat / lon / dt) の SHA-256 ハッシュ
    lineage         : 適用されたオペレータの実行記録リスト (追加順)
    source_versions : データソース名 → バージョン文字列の辞書
                      例: {"s2": "SENTINEL2_L2A", "era5": "reanalysis-era5-land"}

    Methods
    -------
    fingerprint()   : 全フィールドを統合した SHA-256 フィンガープリントを返す。
                      ScienceCacheStore のキャッシュキーとして使用する。
    with_record()   : lineage に ProvenanceRecord を追加した新しいインスタンスを返す。

    設計判断:
        - float 値を直接ハッシュしない → make_input_hash() で正規化文字列経由
        - DataFrame / ndarray をキーにしない → inputs_hash は文字列ハッシュのみ
        - provenance 自体は pickle 非依存 → json.dumps で決定論的シリアライズ
    """
    inputs_hash:     str
    lineage:         List[ProvenanceRecord] = field(default_factory=list)
    source_versions: Dict[str, str]        = field(default_factory=dict)

    def fingerprint(self) -> str:
        """
        全フィールドを統合した SHA-256 フィンガープリントを返す。

        同一入力・同一 lineage・同一バージョンなら実行日時に関わらず
        常に同じ値を返す。sort_keys=True により辞書キー順の揺らぎを排除する。

        設計判断 — timestamp を fingerprint から除外する理由:
            キャッシュの目的は「同一入力・同一処理・同一バージョン」に対する
            再計算の省略である。実行日時が異なっても処理内容が同じであれば
            キャッシュヒットさせるべきであり、timestamp を含めると
            毎回キャッシュミスになる。

            lineage audit や reproducibility audit で実行時刻を追跡したい場合は
            ProvenanceRecord.timestamp を直接参照すること。
            fingerprint はあくまで「内容の同一性」を表す識別子である。

            参考: timestamp intentionally excluded from fingerprint
                  to preserve deterministic cache identity.

        Returns
        -------
        str  SHA-256 ハッシュ (64文字の16進数)
        """
        payload = {
            "inputs": self.inputs_hash,
            "lineage": [
                # timestamp は除外 (上記 docstring 参照)
                {"op": r.operator, "params": r.params}
                for r in self.lineage
            ],
            "versions": self.source_versions,
        }
        raw = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    def with_record(self, record: ProvenanceRecord) -> "DatasetProvenance":
        """
        lineage に ProvenanceRecord を追加した新しい DatasetProvenance を返す。

        frozen=True のため in-place 変更はできないため、
        新規オブジェクトを生成して返す。
        """
        return DatasetProvenance(
            inputs_hash=self.inputs_hash,
            lineage=list(self.lineage) + [record],
            source_versions=self.source_versions,
        )


def make_input_hash(lat: float, lon: float, dt_iso: str) -> str:
    """
    座標・日時から入力空間ハッシュを生成する。

    float を直接ハッシュせず文字列正規化を経由することで、
    浮動小数点表現の揺らぎを排除する。

    Parameters
    ----------
    lat    : 緯度 [degrees]  小数点6桁に正規化
    lon    : 経度 [degrees]  小数点6桁に正規化
    dt_iso : 観測日時 ISO8601 文字列

    Returns
    -------
    str  SHA-256 ハッシュ (64文字の16進数)
    """
    normalized = f"{lat:.6f}:{lon:.6f}:{dt_iso}"
    return hashlib.sha256(normalized.encode()).hexdigest()


# =============================================================================
# Section 2 — Cache
#
# 将来の移行計画:
#     現在は pickle を使用する。
#     将来のリリースで np.savez_compressed または zarr に移行予定。
#     移行時の変更範囲はこの Section 2 のみ。
#     Section 1 (Provenance) と公開 API (get / set / invalidate / clear / size)
#     は変更しない。
# =============================================================================

@dataclass
class CacheStats:
    """
    キャッシュアクセス統計。

    Fields
    ------
    hits   : キャッシュヒット回数
    misses : キャッシュミス回数
    sets   : キャッシュ書き込み回数
    """
    hits:   int = field(default=0)
    misses: int = field(default=0)
    sets:   int = field(default=0)

    @property
    def hit_rate(self) -> float:
        """ヒット率 [0, 1]。アクセスがなければ 0.0 を返す。"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __str__(self) -> str:
        return (f"CacheStats(hits={self.hits}, misses={self.misses}, "
                f"sets={self.sets}, hit_rate={self.hit_rate:.2%})")


class ScienceCacheStore:
    """
    ファイルシステムベースのキャッシュストア。

    現在の実装は pickle を使用する。
    公開 API (get / set / invalidate / clear / size) は将来の実装変更後も維持する。

    WARNING:
        pickle は任意コードを実行できるため、信頼できないパスから
        ロードすることはセキュリティ上危険である。
        キャッシュディレクトリを信頼できるローカルパスにのみ配置すること。
        将来のリリースで np.savez_compressed に移行することでこのリスクを排除する。

    Parameters
    ----------
    root : キャッシュディレクトリのパス (デフォルト: ".science_cache")

    使用例:
        store = ScienceCacheStore(".science_cache")
        store.set("abc123", bundle)
        bundle = store.get("abc123")
    """

    def __init__(self, root: str = ".science_cache"):
        self.root  = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStats()

    # ------------------------------------------------------------------
    # ストレージバックエンド (将来の移行対象)
    #
    # 以下の _path() / get() / set() が移行対象。
    # invalidate() / clear() / size() はパス変更のみで対応可能。
    # ------------------------------------------------------------------

    def _path(self, key: str) -> Path:
        """
        キャッシュキーからファイルパスを生成する。

        NOTE: np.savez_compressed 移行時は拡張子を .pkl → .npz に変更する。
        """
        return self.root / f"{key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        """
        キャッシュからオブジェクトを取得する。

        NOTE: 将来の移行時はここを np.load(allow_pickle=False) に置き換える。

        Parameters
        ----------
        key : DatasetProvenance.fingerprint() の返値

        Returns
        -------
        キャッシュヒット時はオブジェクト、ミス時は None
        """
        p = self._path(key)
        if not p.exists():
            self.stats.misses += 1
            _logger.debug("[Cache] MISS  key=%s", key[:12])
            return None

        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)  # NOTE: 将来は np.load(allow_pickle=False)
            self.stats.hits += 1
            _logger.info("[Cache] HIT   key=%s", key[:12])
            return obj
        except Exception as e:
            # pickle.UnpicklingError / EOFError を含む全例外をキャッチし、
            # 破損エントリを削除してキャッシュミスとして扱う。
            # KeyboardInterrupt / SystemExit は Exception の外であるため
            # 意図せずブロックされることはない。
            _logger.warning("[Cache] 読み込み失敗 (破損の可能性): key=%s  %s",
                            key[:12], e)
            try:
                p.unlink()
            except OSError:
                pass
            self.stats.misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        """
        オブジェクトをキャッシュに保存する。

        NOTE: 将来の移行時はここを np.savez_compressed に置き換える。
              移行後は value が Dict[str, Any] に制限される。

        Parameters
        ----------
        key   : DatasetProvenance.fingerprint() の返値
        value : 保存するオブジェクト (現在は pickle 可能なもの全般)
        """
        p = self._path(key)
        try:
            with open(p, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)  # NOTE: 将来は np.savez_compressed
            self.stats.sets += 1
            _logger.debug("[Cache] SET   key=%s", key[:12])
        except (OSError, pickle.PicklingError) as e:
            _logger.warning("[Cache] 書き込み失敗: key=%s  %s", key[:12], e)

    def invalidate(self, key: str) -> bool:
        """
        指定キーのキャッシュを削除する。

        Returns
        -------
        bool  削除成功なら True、存在しなければ False
        """
        p = self._path(key)
        if p.exists():
            try:
                p.unlink()
                _logger.info("[Cache] INVALIDATE key=%s", key[:12])
                return True
            except OSError as e:
                _logger.warning("[Cache] 削除失敗: key=%s  %s", key[:12], e)
        return False

    def clear(self) -> int:
        """
        キャッシュディレクトリ内の全 .pkl ファイルを削除する。

        NOTE: np.savez_compressed 移行時は *.pkl → *.npz に変更する。

        Returns
        -------
        int  削除したファイル数
        """
        count = 0
        for p in self.root.glob("*.pkl"):
            try:
                p.unlink()
                count += 1
            except OSError:
                pass
        _logger.info("[Cache] CLEAR  %d files removed", count)
        return count

    def size(self) -> int:
        """キャッシュエントリ数を返す。"""
        return len(list(self.root.glob("*.pkl")))

    def __repr__(self) -> str:
        return (f"ScienceCacheStore(root='{self.root}', "
                f"entries={self.size()}, {self.stats})")
