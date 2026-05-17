"""
ApacheLicense2.0
Copyright (c) 2026 tkxu
"""
# eo_provider.py
from __future__ import annotations

import math
import logging
import os
import sys
import tempfile
import threading
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# eo_era5.py から ERA5Fetcher と _uv_to_wind を取得
# _uv_to_wind は eo_era5.py に一元化されているため、ここでは import のみ
try:
    from eo_era5 import ERA5Fetcher, _uv_to_wind
    _ERA5_AVAILABLE = True
except ImportError:
    warnings.warn(
        "eo_era5.py が見つかりません。ERA5 取得が無効化されます。\n"
        "  eo_era5.py を同じディレクトリに配置してください。",
        stacklevel=2,
    )
    ERA5Fetcher     = None
    _ERA5_AVAILABLE = False

    # eo_era5.py が存在しない場合のフォールバック。
    # eo_era5._uv_to_wind と同一の変換式を使用する。
    # 将来 eo_era5 側の式が変わった場合はこちらも合わせて更新すること。
    def _uv_to_wind(u10: float, v10: float) -> Tuple[float, float]:
        """eo_era5._uv_to_wind のフォールバック実装。変換式は eo_era5 と同一。"""
        import math as _math
        speed = _math.sqrt(u10 ** 2 + v10 ** 2)
        deg   = _math.degrees(_math.atan2(-u10, -v10)) % 360.0
        return speed, deg

# eo_types.py からバンドセット定数を取得
try:
    from eo_types import S2_BAND_SETS
except ImportError:
    S2_BAND_SETS: Dict[str, List[str]] = {
        "swir_only":  ["B11", "B12"],
        "full":       ["B01","B02","B03","B04","B05","B06",
                       "B07","B08","B8A","B09","B11","B12"],
        "eucalyptus": ["B01","B02","B03","B04","B05",
                       "B08","B8A","B09","B11","B12"],
        "custom":     [],
    }

# science サブパッケージ (provenance + cache)
try:
    from eo_engines import Provider  # Provider ABC を継承するために import
    _ENGINES_AVAILABLE = True
except ImportError:
    _ENGINES_AVAILABLE = False
    warnings.warn(
        "eo_engines.py が見つかりません。DataFetcher は Provider ABC を継承しません。\n"
        "  eo_engines.py を同じディレクトリに配置してください。",
        stacklevel=2,
    )
    # フォールバック: Provider が使えない場合はダミー基底クラスを使用
    class Provider:  # type: ignore[no-redef]
        pass

# eo_cache.py からキャッシュ・プロベナンス機能を取得

try:
    from eo_cache import ScienceCacheStore, DatasetProvenance, make_input_hash
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False
    warnings.warn(
        "eo_cache.py が見つかりません。キャッシュ機能は無効化されます。\n"
        "  eo_cache.py を同じディレクトリに配置してください。",
        stacklevel=2,
    )

# --- openEO ---
try:
    import openeo
    _OPENEO_AVAILABLE = True
except ImportError:
    _OPENEO_AVAILABLE = False
    warnings.warn(
        "openeo が未インストールです。\n  pip install openeo rasterio",
        stacklevel=2,
    )

try:
    import rasterio
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False
    warnings.warn(
        "rasterio が未インストールです。\n  pip install rasterio",
        stacklevel=2,
    )

try:
    import xarray as xr
    _XARRAY_AVAILABLE = True
except ImportError:
    _XARRAY_AVAILABLE = False

_logger = logging.getLogger(__name__)

# =============================================================================
# 定数
# =============================================================================

OPENEO_BACKEND   = "https://openeo.dataspace.copernicus.eu"
RES_S2_M         = 20.0    # Sentinel-2 解像度 [m/px]
RES_S5P_DEG      = 0.01    # Sentinel-5P リサンプル解像度 [degrees]
DEFAULT_BBOX_KM  = 20.0    # デフォルト取得範囲の半径 [km]
S2_TIME_DELTA    = 5       # Sentinel-2 前後探索期間 [days]
S5P_TIME_DELTA   = 2       # Sentinel-5P 前後探索期間 [days]
# Sentinel-2 シーン品質チェック定数
# shape collapse 検出: 縦横ともにこのピクセル数を下回った場合は異常とみなす。
# radius_km=20, res=20m のとき理論値は ~2000px。16px は collapse 検出用の最低閾値。
S2_MIN_SHAPE: int = 16

# 有効画素率の最低閾値: B11 の有限値画素が全体のこの割合を下回る場合は
# SCL マスクによる全 NaN または data gap とみなして None を返す。
S2_MIN_VALID_RATIO: float = 0.05

# bands フィールドに含めるべき spectral バンド名のセット
# sza / vza / amf などの非スペクトルキーを除外するために使用する
_SPECTRAL_KEYS: Set[str] = {
    b for bands in S2_BAND_SETS.values() for b in bands
}

# =============================================================================
# ユーティリティ
# =============================================================================

def _lat_lon_to_bbox(
    lat:       float,
    lon:       float,
    radius_km: float = DEFAULT_BBOX_KM,
) -> Tuple[float, float, float, float]:
    """
    中心座標と半径 [km] から openEO 形式の BBox (west, south, east, north) を返す。

    ERA5 の area パラメータ [N, W, S, E] とは順序が異なる点に注意。
    ERA5 向けには eo_era5._lat_lon_to_area() を使用すること。
    """
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _download_to_ndarray(cube: Any, label: str = "") -> np.ndarray:
    """
    openEO DataCube を一時ファイル経由で ndarray (bands, H, W) に変換する。

    try/finally で一時ファイルのリークを防ぐ。
    shape を DEBUG ログに出力することで、backend による spatial collapse を早期検出できる。
    """
    if not _RASTERIO_AVAILABLE:
        raise RuntimeError(
            "rasterio が未インストールです。\n  pip install rasterio"
        )
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        cube.download(tmp_path, format="GTiff")
        with rasterio.open(tmp_path) as src:
            arr = src.read().astype(np.float32)
            _logger.debug(
                "[%s] downloaded shape=%s  (bands, H, W)",
                label or "download", arr.shape,
            )
            return arr
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def _latlon_to_utm_epsg(lat: float, lon: float) -> str:
    """
    緯度経度から適切な UTM EPSG を返す。

    北半球:
        EPSG:32601 - 32660

    南半球:
        EPSG:32701 - 32760
    """
    zone = int((lon + 180) // 6) + 1

    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone

    return f"EPSG:{epsg}"

# =============================================================================
# OpenEO クライアント (スレッドセーフ・シングルトン)
# =============================================================================

class OpenEOClient:
    """
    openEO (CDSE) への接続をシングルトンで管理する。

    threading.Lock でマルチスレッド環境の二重初期化を防ぐ。

    認証方式は環境変数 OPENEO_AUTH_METHOD で切り替える。
    未設定の場合は以下の優先順位で自動選択する:

        1. refresh_token  (デフォルト・非対話的)
           リフレッシュトークンが ~/.config/openeo/ 等に保存されていれば
           ユーザー操作なしで認証が完了する。
           初回または期限切れ時は device フローにフォールバックし、
           認証成功後に新しいリフレッシュトークンを自動保存する。

           環境変数でトークンを直接指定することも可能:
             OPENEO_REFRESH_TOKEN=<token>
           この場合ファイルへの保存・読み込みをスキップして指定値を使用する。

        2. client_credentials  (CI / サービスアカウント向け)
           OPENEO_AUTH_METHOD=client_credentials を設定する場合は
           OPENEO_CLIENT_ID / OPENEO_CLIENT_SECRET も必須。

        3. device  (対話的フォールバック・ブラウザ操作が必要)
           OPENEO_AUTH_METHOD=device を明示指定するか、
           refresh_token フロー失敗時に自動的に使用される。

    初回セットアップ手順:
        OPENEO_AUTH_METHOD を設定しない状態で初回実行すると
        device フローが起動する。ブラウザで認証を完了すると
        リフレッシュトークンが自動保存され、次回以降は非対話的になる。
    """

    _instance: Optional["OpenEOClient"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "OpenEOClient":
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._con = None
                cls._instance = obj
        return cls._instance

    def get(self) -> "openeo.Connection":
        # _con の読み取りと _connect の呼び出しをロックで保護する。
        # ロック外で _con を読むと、別スレッドが _connect 中に
        # None を読んで二重接続が発生する可能性がある。
        with self._lock:
            if self._con is None:
                self._connect()
            return self._con

    def _connect(self) -> None:
        """
        接続と認証を行う。呼び出し元 get() がすでに _lock を保持しているため
        このメソッド自身は再取得しない。
        """
        if not _OPENEO_AVAILABLE:
            raise RuntimeError(
                "openeo が未インストールです。\n  pip install openeo"
            )
        self._con = openeo.connect(OPENEO_BACKEND)

        auth_method = os.environ.get("OPENEO_AUTH_METHOD", "refresh_token")

        if auth_method == "client_credentials":
            self._auth_client_credentials()
        elif auth_method == "device":
            self._auth_device(store_refresh_token=True)
        else:
            # デフォルト: refresh_token → device フォールバック
            self._auth_refresh_token_with_fallback()

    # ------------------------------------------------------------------
    # 認証ヘルパー
    # ------------------------------------------------------------------

    def _auth_refresh_token_with_fallback(self) -> None:
        """
        リフレッシュトークンで認証する。失敗した場合は device フローにフォールバックし、
        成功後に新しいリフレッシュトークンを保存する。

        トークンの取得元 (優先順):
            1. 環境変数 OPENEO_REFRESH_TOKEN (CI / シークレット管理ツール向け)
            2. openEO クライアントが管理するローカルファイル
               (~/.config/openeo/ または OPENEO_CONFIG_HOME 配下)
        """
        env_token = os.environ.get("OPENEO_REFRESH_TOKEN", "")
        if env_token:
            # 環境変数から直接トークンを使用
            _logger.info("[Auth] リフレッシュトークン認証 (OPENEO_REFRESH_TOKEN 環境変数)")
            try:
                self._con.authenticate_oidc_refresh_token(refresh_token=env_token)
                _logger.info("[Auth] リフレッシュトークン認証 成功")
                return
            except Exception as e:
                _logger.warning(
                    "[Auth] OPENEO_REFRESH_TOKEN での認証に失敗: %s\n"
                    "  → device フローにフォールバックします。",
                    e,
                )
        else:
            # ローカルファイルから自動読み込み
            _logger.info("[Auth] リフレッシュトークン認証 (ローカルファイル)")
            try:
                self._con.authenticate_oidc_refresh_token()
                _logger.info("[Auth] リフレッシュトークン認証 成功")
                return
            except Exception as e:
                _logger.info(
                    "[Auth] リフレッシュトークンが見つからないか期限切れです (%s)\n"
                    "  → device フローにフォールバックします。",
                    e,
                )

        # フォールバック: device フロー (store_refresh_token=True で次回以降を非対話的にする)
        _logger.info("[Auth] device フロー (ブラウザ認証)。完了後にリフレッシュトークンを保存します。")
        self._auth_device(store_refresh_token=True)

    def _auth_device(self, store_refresh_token: bool = True) -> None:
        """
        device フロー (対話的ブラウザ認証) を実行する。

        Parameters
        ----------
        store_refresh_token : True の場合、認証成功後にリフレッシュトークンを
                              ローカルファイルに保存する。デフォルト True。
                              次回以降のリフレッシュトークン認証のために必要。
        """
        self._con.authenticate_oidc_device(
            store_refresh_token=store_refresh_token,
        )
        _logger.info(
            "[Auth] device フロー認証 成功 (store_refresh_token=%s)",
            store_refresh_token,
        )

    def _auth_client_credentials(self) -> None:
        """
        client credentials フロー (サービスアカウント認証) を実行する。

        必要な環境変数:
            OPENEO_CLIENT_ID     : クライアント ID
            OPENEO_CLIENT_SECRET : クライアントシークレット
        """
        client_id     = os.environ.get("OPENEO_CLIENT_ID", "")
        client_secret = os.environ.get("OPENEO_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            raise RuntimeError(
                "OPENEO_AUTH_METHOD=client_credentials が指定されていますが、\n"
                "  OPENEO_CLIENT_ID / OPENEO_CLIENT_SECRET が未設定です。"
            )
        self._con.authenticate_oidc_client_credentials(
            client_id=client_id,
            client_secret=client_secret,
        )
        _logger.info("[Auth] client credentials 認証 成功")

    @classmethod
    def reset(cls) -> None:
        """テスト用: シングルトンをリセットして再接続を強制する。"""
        with cls._lock:
            cls._instance = None


def verify_access() -> None:
    """認証済み接続の確認。接続できた場合はアカウント ID を表示する。"""
    con = OpenEOClient().get()
    print(f"  Connection: {con.describe_account()['id']}")


# =============================================================================
# Sentinel-2 フェッチャー
# =============================================================================

class Sentinel2Fetcher:
    """
    openEO 経由で Sentinel-2 L2A のバンドデータを取得する。

    返り値の Dict には spectral バンド (B11 等) のほか
    sza / vza / amf / sza_mean / vza_mean / scene_id が含まれる。
    DataFetcher._build_bundle() では _SPECTRAL_KEYS を使って
    spectral バンドのみを "bands" フィールドに格納する。

    SCL マスクについて:
        SCL クラス 0 (no data)・9 (high-confidence cloud)・10 (thin cirrus) を
        標準マスクとして適用する。

        8 (medium-confidence cloud) は偽陽性が多いため除外しているが、
        精度が求められる用途では _SCL_MASK_CLASSES に追加すること。
    """

    _ANGLE_BANDS = ["sunZenithAngles", "viewZenithMean"]

    # 0: no data
    # 9: high-confidence cloud
    # 10: thin cirrus
    _SCL_MASK_CLASSES = (0, 9, 10)

    def __init__(self, client: OpenEOClient, band_set: str = "swir_only"):
        self.con      = client.get()
        self.band_set = band_set

    @staticmethod
    def _latlon_to_utm_epsg(lat: float, lon: float) -> str:
        """
        緯度経度から適切な UTM EPSG を返す。

        北半球:
            EPSG:32601 - 32660

        南半球:
            EPSG:32701 - 32760
        """
        zone = int((lon + 180) // 6) + 1

        if lat >= 0:
            epsg = 32600 + zone
        else:
            epsg = 32700 + zone

        return f"EPSG:{epsg}"

    def fetch(
        self,
        lat:       float,
        lon:       float,
        dt:        datetime,
        radius_km: float = DEFAULT_BBOX_KM,
    ) -> Optional[Dict]:
        """
        指定座標・日時の Sentinel-2 データを取得する。

        Returns
        -------
        Dict または None
            spectral バンド (band_set に依存) + sza, vza, amf,
            sza_mean, vza_mean, scene_id
        """

        target_spectral = S2_BAND_SETS.get(
            self.band_set,
            S2_BAND_SETS["swir_only"]
        )

        bbox = _lat_lon_to_bbox(lat, lon, radius_km)

        t_start = (dt - timedelta(days=S2_TIME_DELTA)).strftime("%Y-%m-%d")
        t_end   = (dt + timedelta(days=S2_TIME_DELTA)).strftime("%Y-%m-%d")

        try:
            cube = self.con.load_collection(
                "SENTINEL2_L2A",
                spatial_extent={
                    "west":  bbox[0],
                    "south": bbox[1],
                    "east":  bbox[2],
                    "north": bbox[3],
                },
                temporal_extent=[t_start, t_end],
                bands=target_spectral + ["SCL"] + self._ANGLE_BANDS,
            )

            # -------------------------------------------------------------
            # CRS 決定
            #
            # EPSG:4326 + resolution=20 は backend により
            # 「20 degree」と解釈される場合がある。
            #
            # UTM CRS を明示することで、
            # resolution=20 を meters として保証する。
            # -------------------------------------------------------------

            utm_epsg = self._latlon_to_utm_epsg(lat, lon)

            _logger.debug(
                "[S2] using projection=%s lat=%.4f lon=%.4f",
                utm_epsg,
                lat,
                lon,
            )

            # -------------------------------------------------------------
            # SCL mask
            #
            # mask → resample の順にする。
            # -------------------------------------------------------------

            scl = cube.band("SCL")

            scl_mask = (scl == self._SCL_MASK_CLASSES[0])

            for cls in self._SCL_MASK_CLASSES[1:]:
                scl_mask = scl_mask | (scl == cls)

            cube = cube.mask(scl_mask)

            # -------------------------------------------------------------
            # Spatial resampling
            # -------------------------------------------------------------

            cube = cube.resample_spatial(
                resolution=RES_S2_M,
                projection=utm_epsg,
            )

            # -------------------------------------------------------------
            # Temporal reduction
            # -------------------------------------------------------------

            cube = (
                cube
                .filter_bands(target_spectral + self._ANGLE_BANDS)
                .reduce_dimension(
                    dimension="t",
                    reducer="median",
                )
            )

            # -------------------------------------------------------------
            # download
            # -------------------------------------------------------------

            data = _download_to_ndarray(cube, "S2")

        except Exception as e:
            warnings.warn(
                f"[S2] 取得エラー: {e}",
                stacklevel=2,
            )
            return None

        # -------------------------------------------------------------
        # バンド数チェック
        # -------------------------------------------------------------

        n_spectral = len(target_spectral)
        n_expected = n_spectral + 2

        if data.shape[0] < n_expected:
            warnings.warn(
                f"[S2] バンド数不足 "
                f"(取得={data.shape[0]}, 期待={n_expected})",
                stacklevel=2,
            )
            return None

        # -------------------------------------------------------------
        # shape collapse チェック
        # -------------------------------------------------------------

        h, w = data.shape[1], data.shape[2]

        expected_px = int(
            radius_km * 2 * 1000 / RES_S2_M
        )

        _logger.debug(
            "[S2] downloaded shape=%s expected~%dpx",
            data.shape,
            expected_px,
        )

        if h < S2_MIN_SHAPE or w < S2_MIN_SHAPE:
            warnings.warn(
                f"[S2] シーンサイズが異常です "
                f"(shape={data.shape})。\n"
                f"  期待サイズ: ~{expected_px}px\n"
                f"  projection/resolution/backend を確認してください。",
                stacklevel=2,
            )
            return None

        # -------------------------------------------------------------
        # spectral bands
        #
        # NaN を保持したまま clip する
        # -------------------------------------------------------------

        bands = {}

        for i, b in enumerate(target_spectral):

            arr = data[i].astype(np.float32)

            finite_mask = np.isfinite(arr)

            clipped = np.full(
                arr.shape,
                np.nan,
                dtype=np.float32,
            )

            clipped[finite_mask] = np.clip(
                arr[finite_mask],
                1e-4,
                None,
            )

            bands[b] = clipped

        # -------------------------------------------------------------
        # geometry
        # -------------------------------------------------------------

        sza = data[n_spectral]
        vza = data[n_spectral + 1]

        amf = (
            1.0 / np.clip(np.cos(np.radians(sza)), 1e-3, 1.0)
            +
            1.0 / np.clip(np.cos(np.radians(vza)), 1e-3, 1.0)
        )

        amf = np.where(
            np.isfinite(amf),
            amf,
            np.nan,
        )

        # -------------------------------------------------------------
        # valid ratio check
        # -------------------------------------------------------------

        if "B11" not in bands:
            warnings.warn(
                "[S2] B11 が存在しません。",
                stacklevel=2,
            )
            return None

        b11_arr = bands["B11"]

        valid_ratio = float(
            np.mean(np.isfinite(b11_arr))
        )

        _logger.debug(
            "[S2] B11 valid_ratio=%.3f shape=(%d,%d)",
            valid_ratio,
            h,
            w,
        )

        if valid_ratio < S2_MIN_VALID_RATIO:
            warnings.warn(
                f"[S2] 有効画素率が低すぎます "
                f"(valid_ratio={valid_ratio:.3f} "
                f"< {S2_MIN_VALID_RATIO})。\n"
                f"  全雲 / data gap の可能性があります。",
                stacklevel=2,
            )
            return None

        # -------------------------------------------------------------
        # return
        # -------------------------------------------------------------

        return {
            **bands,

            "sza": sza,
            "vza": vza,
            "amf": amf,

            "sza_mean": (
                float(np.nanmean(sza))
                if np.any(np.isfinite(sza))
                else float("nan")
            ),

            "vza_mean": (
                float(np.nanmean(vza))
                if np.any(np.isfinite(vza))
                else float("nan")
            ),

            "scene_id": (
                f"S2_{dt.strftime('%Y%m%d')}_"
                f"{lat:.2f}N_{lon:.2f}E"
            ),
        }


# =============================================================================
# Sentinel-5P フェッチャー
# =============================================================================

class Sentinel5PFetcher:
    """openEO 経由で Sentinel-5P L2 の CH4 カラム濃度を取得する。"""

    def __init__(self, client: OpenEOClient):
        self.con = client.get()

    def fetch(
        self,
        lat:       float,
        lon:       float,
        dt:        datetime,
        radius_km: float = DEFAULT_BBOX_KM,
    ) -> Optional[Dict]:
        """
        指定座標・日時周辺の Sentinel-5P CH4 を取得する。

        Returns
        -------
        Dict または None
            Keys: ch4_column (ndarray [ppb]),
                  ch4_mean, ch4_std (float), valid_frac (float)
        """
        bbox    = _lat_lon_to_bbox(lat, lon, radius_km)
        t_start = (dt - timedelta(days=S5P_TIME_DELTA)).strftime("%Y-%m-%d")
        t_end   = (dt + timedelta(days=S5P_TIME_DELTA)).strftime("%Y-%m-%d")

        try:
            cube = self.con.load_collection(
                "SENTINEL_5P_L2",
                spatial_extent  = {"west": bbox[0], "south": bbox[1],
                                   "east": bbox[2], "north": bbox[3]},
                temporal_extent = [t_start, t_end],
                bands           = ["CH4", "dataMask"],
            )
            cube = cube.resample_spatial(resolution=RES_S5P_DEG, projection="EPSG:4326")
            cube = (
                cube
                .mask(cube.band("dataMask") == 0)
                .filter_bands(["CH4"])
                .reduce_dimension(dimension="t", reducer="median")
            )
            data = _download_to_ndarray(cube, "S5P")

        except Exception as e:
            warnings.warn(f"[S5P] 取得エラー: {e}", stacklevel=2)
            return None

        return {"ch4_column": data[0] if data.ndim == 3 else data}


# =============================================================================
# DataFetcher — 統合データ取得クラス (Lazy Initialization)
# =============================================================================

class DataFetcher(Provider):
    """
    Sentinel-2・ERA5・Sentinel-5P を統合して ObservationBundle を返す。

    Lazy Initialization: openEO への接続は fetch() 呼び出し時に初めて行う。
    3ソースは ThreadPoolExecutor で並列取得 (max_workers に依存)。
    各サブフェッチャーが失敗しても部分結果を返す（サイレント縮退）。

    Parameters
    ----------
    radius_km   : 取得範囲の半径 [km]
    max_retries : ERA5 リトライ回数
    max_workers : 並列スレッド数
    band_set    : 取得バンドセット ("swir_only" / "full" / "eucalyptus")
    cache_dir   : キャッシュディレクトリ (None でキャッシュ無効)
    """

    def __init__(
        self,
        radius_km:   float         = DEFAULT_BBOX_KM,
        max_retries: int           = 2,
        max_workers: int           = 1,
        band_set:    str           = "swir_only",
        cache_dir:   Optional[str] = ".science_cache",
    ):
        self.radius_km   = radius_km
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.band_set    = band_set
        self._s2:   Optional[Sentinel2Fetcher]  = None
        self._s5p:  Optional[Sentinel5PFetcher] = None
        self._era5: Optional[Any]               = None

        self._cache = (
            ScienceCacheStore(cache_dir)
            if _CACHE_AVAILABLE and cache_dir
            else None
        )

    def _ensure_fetchers(self) -> None:
        """初回 fetch() 時に各フェッチャーを遅延生成する。"""
        if self._s2 is None:
            client    = OpenEOClient()
            self._s2  = Sentinel2Fetcher(client, self.band_set)
            self._s5p = Sentinel5PFetcher(client)

        if self._era5 is None and _ERA5_AVAILABLE:
            self._era5 = ERA5Fetcher(max_retries=self.max_retries)

    def fetch(self, lat: float, lon: float, dt: datetime) -> Dict:
        """
        指定座標・日時の ObservationBundle を返す。

        Parameters
        ----------
        lat : 中心緯度 [degrees]
        lon : 中心経度 [degrees]
        dt  : 観測希望日時 (UTC)

        Returns
        -------
        ObservationBundle (Dict)
        """
        self._ensure_fetchers()

        # --- キャッシュ参照 ---
        cache_key = None
        if _CACHE_AVAILABLE and self._cache is not None:
            cache_key = DatasetProvenance(
                inputs_hash     = make_input_hash(lat, lon, dt.isoformat()),
                source_versions = {"s2": "S2", "s5p": "S5P", "era5": "ERA5"},
            ).fingerprint()
            cached = self._cache.get(cache_key)
            if cached is not None:
                _logger.info("[DataFetcher] CACHE HIT key=%s", cache_key[:12])
                return cached

        # --- 並列取得 ---
        tasks: Dict[str, tuple] = {
            "s2":  (self._s2.fetch,  (lat, lon, dt, self.radius_km)),
            "s5p": (self._s5p.fetch, (lat, lon, dt, self.radius_km)),
        }
        if self._era5 is not None:
            tasks["era5"] = (self._era5.fetch, (lat, lon, dt, self.radius_km))

        raw: Dict[str, Optional[Dict]] = {"s2": None, "era5": None, "s5p": None}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(fn, *args): key
                for key, (fn, args) in tasks.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    raw[key] = future.result()
                except Exception as e:
                    # サブフェッチャーが失敗しても他のソースの結果は保持する
                    warnings.warn(
                        f"[DataFetcher] {key} 取得中に例外: {e}",
                        stacklevel=2,
                    )

        bundle = self._build_bundle(raw, lat, lon, dt, cache_key)

        # --- キャッシュ保存 ---
        if self._cache is not None and cache_key is not None:
            self._cache.set(cache_key, bundle)

        return bundle

    def fetch_from_site(self, site: Dict, dt: datetime) -> Dict:
        """SITE_REGISTRY エントリを直接受け取るショートカット。"""
        return self.fetch(site["lat"], site["lon"], dt)

    def get_bundle(
        self,
        site: Dict,
        dt:   Optional[datetime] = None,
    ) -> Dict:
        """
        Provider ABC の統一インターフェース実装。

        EOPipeline から isinstance(provider, Provider) で分岐されたとき
        このメソッドが呼ばれる。dt が None の場合は現在時刻を使用する。

        Parameters
        ----------
        site : SiteEntry  サイトレジストリのエントリ
        dt   : datetime   観測日時 (None の場合は UTC 現在時刻)

        Returns
        -------
        ObservationBundle
        """
        if dt is None:
            dt = datetime.now(tz=timezone.utc)
        return self.fetch_from_site(site, dt)

    def _build_bundle(
        self,
        raw:       Dict[str, Optional[Dict]],
        lat:       float,
        lon:       float,
        dt:        datetime,
        cache_key: Optional[str],
    ) -> Dict:
        """
        各ソースの生取得結果を ObservationBundle 形式に変換する。

        Sentinel-2 の返り値には sza / vza / amf も np.ndarray として含まれるため、
        _SPECTRAL_KEYS フィルタで spectral バンドのみを "bands" に格納する。
        """
        s2   = raw.get("s2")
        era5 = raw.get("era5")
        s5p  = raw.get("s5p")

        # spectral バンドのみ抽出 (sza / vza / amf を除外)
        bands = (
            {k: v for k, v in s2.items()
             if isinstance(v, np.ndarray) and k in _SPECTRAL_KEYS}
            if s2 is not None else None
        )

        return {
            "bands":            bands,
            "sza":              s2.get("sza")              if s2   is not None else None,
            "vza":              s2.get("vza")              if s2   is not None else None,
            "amf":              s2.get("amf")              if s2   is not None else None,
            "wind_speed":       era5.get("wind_speed")     if era5 is not None else None,
            "wind_deg":         era5.get("wind_deg")       if era5 is not None else None,
            "surface_pressure": era5.get("surface_pressure") if era5 is not None else None,
            "ch4_column":       s5p.get("ch4_column")      if s5p  is not None else None,
            "meta": {
                "lat":            lat,
                "lon":            lon,
                "datetime":       dt.isoformat(),
                "band_set":       self.band_set,
                "backend":        "openeo",
                "era5_backend":   era5.get("backend") if era5 is not None else None,
                "provenance_key": cache_key,
                "cloud_pct":      None,
                "s2_scene_id":    s2.get("scene_id") if s2 is not None else None,
                "sza_mean":       s2.get("sza_mean") if s2 is not None else None,
                "vza_mean":       s2.get("vza_mean") if s2 is not None else None,
            },
        }


# =============================================================================
# テストコード   — クラス定義の後、__main__ の前に配置
# =============================================================================

def _print_bundle(label: str, result: Optional[Dict]) -> None:
    """
    ObservationBundle または Dict を整形表示するヘルパー。

    np.nanmin/nanmax を _safe_nanmin/_safe_nanmax で保護する。 
    """
    print(f"\n{'─'*58}")
    print(f"  {label}")
    print(f"{'─'*58}")
    if result is None:
        print("  → 結果なし (取得失敗 or 認証未設定)")
        return
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            print(f"  {key:18s}: ndarray shape={val.shape} "
                  f"dtype={val.dtype} "
                  )
        elif isinstance(val, dict):
            print(f"  {key:18s}:")
            for k2, v2 in val.items():
                if isinstance(v2, np.ndarray):
                    print(f"    {k2:14s}: ndarray shape={v2.shape}")
                else:
                    print(f"    {k2:14s}: {v2}")
        else:
            print(f"  {key:18s}: {val}")


def test_utils() -> None:
    """_uv_to_wind / _lat_lon_to_bbox の単体テスト。"""
    print("\n[TEST-0] Utils")

    # _uv_to_wind
    ws, wd = _uv_to_wind(-1.0, -1.0)
    print(f"  _uv_to_wind(-1, -1): ws={ws:.4f}, wd={wd:.2f}")
    expected_ws = 2.0 ** 0.5
    expected_wd = 45.0
    assert abs(ws - expected_ws) < 1e-4, f"wind_speed 異常: {ws}"
    assert abs(wd - expected_wd) < 0.1,  f"wind_deg 異常: {wd}"
    print("  _uv_to_wind → PASS")

    # _lat_lon_to_bbox
    bbox = _lat_lon_to_bbox(35.0, 135.0, 10.0)
    west, south, east, north = bbox
    print(f"  _lat_lon_to_bbox(35, 135, 10): {bbox}")
    assert west  < 135.0 < east,  "longitude 範囲エラー"
    assert south < 35.0  < north, "latitude 範囲エラー"
    print("  _lat_lon_to_bbox → PASS")

    print("  TEST-0: PASS")


def test_sentinel2(lat: float, lon: float, dt: datetime) -> None:
    """Sentinel2Fetcher の動作テスト (openEO 認証が必要)。"""
    print("\n[TEST-1] Sentinel-2")
    if not _OPENEO_AVAILABLE:
        print("  SKIPPED: openeo 未インストール")
        return
    try:
        fetcher = Sentinel2Fetcher(OpenEOClient())
        res     = fetcher.fetch(lat, lon, dt, radius_km=5.0)
        if res is not None:
            print(f"  OK: keys={list(res.keys())}, sza_mean={res['sza_mean']:.2f}")
        else:
            print("  result = None (取得失敗)")
    except Exception as e:
        print(f"  FAILED: {e}")


def test_era5(lat: float, lon: float, dt: datetime) -> None:
    """ERA5Fetcher の動作テスト (CDS API 認証が必要)。"""
    print("\n[TEST-2] ERA5 (via eo_era5.py)")
    if not _ERA5_AVAILABLE or ERA5Fetcher is None:
        print("  SKIPPED: eo_era5.py が見つかりません")
        return
    try:
        fetcher = ERA5Fetcher()
        res     = fetcher.fetch(lat, lon, dt, radius_km=10.0)
        if res is not None:
            print(
                f"  OK: ws={res['wind_speed']:.2f}  "
                f"wd={res['wind_deg']:.2f}  "
                f"sp={res['surface_pressure']:.1f}"
            )
        else:
            print("  result = None (取得失敗)")
    except Exception as e:
        print(f"  FAILED: {e}")


def test_sentinel5p(lat: float, lon: float, dt: datetime) -> None:
    """Sentinel5PFetcher の動作テスト (openEO 認証が必要)。"""
    print("\n[TEST-3] Sentinel-5P")
    if not _OPENEO_AVAILABLE:
        print("  SKIPPED: openeo 未インストール")
        return
    try:
        fetcher = Sentinel5PFetcher(OpenEOClient())
        res     = fetcher.fetch(lat, lon, dt, radius_km=10.0)
        if res is not None:
            print(f"  OK: ch4_column.shape={res['ch4_column'].shape}")
        else:
            print("  result = None (取得失敗)")
    except Exception as e:
        print(f"  FAILED: {e}")

def test_data_fetcher(
    lat: float    = 38.49,
    lon: float    = 54.19,
    dt:  datetime = datetime(2023, 7, 15, 8, 0, tzinfo=timezone.utc),
) -> Dict:
    """DataFetcher 統合テスト。TM-01 デフォルト。"""
    print("\n" + "="*58)
    print("  TEST-4: DataFetcher 統合テスト (openEO)")
    print(f"  lat={lat}  lon={lon}  dt={dt.isoformat()}")
    print("="*58)

    fetcher = DataFetcher(radius_km=DEFAULT_BBOX_KM, max_retries=3)
    bundle  = fetcher.fetch(lat, lon, dt)
    _print_bundle("ObservationBundle", bundle)

    required = ["bands", "sza", "vza", "amf",
                "wind_speed", "wind_deg", "surface_pressure",
                "ch4_column", "meta"]
    for key in required:
        assert key in bundle, f"ObservationBundle に '{key}' が存在しない"
    assert bundle["meta"]["backend"] == "openeo", "backend が 'openeo' でない"
    print("\n  [ASSERT] 全キー + backend='openeo' → OK")

    if bundle["bands"] is not None:
        b11_valid = bundle["bands"]["B11"][np.isfinite(bundle["bands"]["B11"])]
        if len(b11_valid) > 0:
            assert np.all(b11_valid > 0), "B11 有効画素にゼロ以下の値"
        else:
            warnings.warn("B11 の全画素が SCL マスクにより NaN です。")
        print("  [ASSERT] bands → OK")

    if bundle["amf"] is not None:
        amf_valid = bundle["amf"][np.isfinite(bundle["amf"])]
        if len(amf_valid) > 0:
            assert np.all(amf_valid >= 2.0), f"AMF 異常: min={amf_valid.min():.4f}"
        print("  [ASSERT] AMF → OK")

    if bundle["wind_speed"] is not None:
        assert bundle["wind_speed"] >= 0,           "wind_speed が負値"
        assert 0 <= bundle["wind_deg"] < 360,       "wind_deg 範囲外"
        sp = bundle["surface_pressure"]
        assert sp is None or 50000 <= sp <= 110000, f"surface_pressure 異常: {sp}"
        print("  [ASSERT] wind / surface_pressure → OK")

    return bundle


def test_custom_location(
    lat:       float,
    lon:       float,
    dt:        datetime,
    radius_km: float = DEFAULT_BBOX_KM,
) -> Dict:
    """
    任意の座標・日時でのデータ取得テスト。

    使用例:
        bundle = test_custom_location(
            lat=35.68, lon=139.69,
            dt=datetime(2024, 3, 1, 2, 0, tzinfo=timezone.utc),
        )
    """
    print("\n" + "="*58)
    print("  TEST-5: カスタム座標・日時")
    print(f"  lat={lat}  lon={lon}  dt={dt.isoformat()}")
    print(f"  radius={radius_km} km")
    print("="*58)

    fetcher = DataFetcher(radius_km=radius_km, max_retries=3)
    bundle  = fetcher.fetch(lat, lon, dt)
    _print_bundle("ObservationBundle", bundle)
    return bundle

def search_valid_obs_date(
    lat: float, 
    lon: float, 
    start_dt: datetime, 
    end_dt: datetime, 
    step_days: int = 5,
    min_valid_ratio: float = 0.1
) -> List[datetime]:
    """
    指定期間を走査し、SCLマスクを通過した有効な画素を持つ日時をリストアップする。
    """
    print(f"\n[SEARCH] Searching valid data for Lat:{lat}, Lon:{lon}")
    print(f"         Period: {start_dt.date()} to {end_dt.date()} (Step: {step_days} days)")
    
    # 探索を高速化するため、S2のみに絞り、キャッシュは無効化を推奨
    fetcher = DataFetcher(radius_km=10.0, band_set="swir_only", cache_dir=None)
    valid_dates = []

    current_dt = start_dt
    while current_dt <= end_dt:
        sys.stdout.write(f"\r  Checking: {current_dt.strftime('%Y-%m-%d')} ... ")
        sys.stdout.flush()
        
        # ERA5/S5Pをスキップするため、内部の _s2 のみ直接叩くことも検討可能だが、
        # ここでは汎用性を考え fetch() を使用（※必要に応じて fetcher._era5 = None で無効化）
        bundle = fetcher.fetch(lat, lon, current_dt)
        
        if bundle.get("bands") is not None and "B11" in bundle["bands"]:
            b11 = bundle["bands"]["B11"]
            valid_count = np.count_nonzero(np.isfinite(b11))
            total_count = b11.size
            ratio = valid_count / total_count if total_count > 0 else 0
            
            if ratio >= min_valid_ratio:
                print(f" FOUND! (Valid ratio: {ratio:.1%})")
                valid_dates.append(current_dt)
            else:
                # print(f" Masked (Ratio: {ratio:.1%})") # デバッグ用
                pass
        
        current_dt += timedelta(days=step_days)
    
    print(f"\n[SEARCH] Finished. Found {len(valid_dates)} valid dates.")
    return valid_dates

# =============================================================================
# エントリポイント
# =============================================================================

if __name__ == "__main__":
    print("=" * 58)
    print("  eo_provider.py  統合データ取得層テスト")
    print("=" * 58)

    print("\n[0] 環境チェック")
    print(f"  openeo   : {'OK' if _OPENEO_AVAILABLE   else 'NOT INSTALLED'}")
    print(f"  rasterio : {'OK' if _RASTERIO_AVAILABLE else 'NOT INSTALLED'}")
    print(f"  eo_era5  : {'OK' if _ERA5_AVAILABLE     else 'NOT FOUND'}")
    print(f"  eo_cache : {'OK' if _CACHE_AVAILABLE    else 'NOT FOUND'}")
    print(f"  xarray   : {'OK' if _XARRAY_AVAILABLE   else 'NOT INSTALLED'}")

    if _OPENEO_AVAILABLE:
        try:
            verify_access()
        except Exception:
            pass

    # TEST-0: 外部 API 不要・常時実行
    test_utils()

    # デフォルトターゲット: TM-01 トルクメニスタン
    TARGET_LAT = 38.49
    TARGET_LON = 54.19
    TARGET_DT  = datetime(2023, 7, 15, 8, 0, tzinfo=timezone.utc)

    test_sentinel2(TARGET_LAT, TARGET_LON, TARGET_DT)
    test_era5(TARGET_LAT, TARGET_LON, TARGET_DT)
    test_sentinel5p(TARGET_LAT, TARGET_LON, TARGET_DT)

    print("\n[4] DataFetcher 統合テスト")
    test_data_fetcher(TARGET_LAT, TARGET_LON, TARGET_DT)

    print("\n[5] カスタム座標テスト (PB-01 ペルム盆地)")
    test_custom_location(
        lat       = 31.83,
        lon       = -102.37,
        dt        = datetime(2024, 1, 20, 15, 0, tzinfo=timezone.utc),
        radius_km = 25.0,
    )

    start = datetime(2023, 6, 1, tzinfo=timezone.utc)
    end   = datetime(2023, 6, 5, tzinfo=timezone.utc)
    found = search_valid_obs_date(38.49, 54.19, start, end)

    print("\n" + "="*58)
    print("  テスト完了")
    print("  openEO 未接続の場合 TEST-1〜5 は SKIP されます。")
    print("  接続単体テスト: test_openeo_connection() を呼び出してください。")
    print("="*58)
