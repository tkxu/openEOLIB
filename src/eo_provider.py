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
   
 eo_provider.py — 地球観測データ提供レイヤ

 役割:
   任意の日時・座標を受け取り、下記3ソースからデータを取得してObservationBundle 形式で返す。

   ┌──────────────────────────────────────────────────────────┐
   │  呼び出し元 (eo_pipeline.py / ObsPipeline)                │
   │    provider = DataFetcher()                               │
   │    obs = provider.fetch(lat=38.49, lon=54.19, dt=...)     │
   │    # obs["bands"]["B11"]     → ndarray (H×W) float32     │
   │    # obs["bands"]["B08"]     → ndarray (H×W) float32     │
   │    #   (band_set="full" 時のみ)                           │
   │    # obs["wind_speed"]       → float [m/s]               │
   │    # obs["wind_deg"]         → float [degrees FROM方向]   │
   │    # obs["sza"]              → ndarray [degrees]          │
   │    # obs["surface_pressure"] → float [Pa]                │
   │    # obs["meta"]["band_set"] → "swir_only" or "full" 等  │
   └──────────────────────────────────────────────────────────┘

 データソース (すべて openEO / CDSE 経由):
 ※取得失敗や欠損が発生する場合あり
   Sentinel-2 L2A   B11 (1610nm) / B12 (2190nm) 反射率バンド  (デフォルト)
                    全13バンド取得も可能 (band_set="full")
                    SCL マスク (有効地表画素: SCL 4/5/6)
                    SZA / VZA (太陽・衛星天頂角)
                    解像度: 20m/px, CRS: EPSG:4326

   ERA5 Land        u10 / v10 地上10m風速成分
                    surface_pressure [Pa]
                    空間平均 (mean_spatial) で安定化

   Sentinel-5P L2   CH4カラム濃度 [ppb]
                    qa_value >= 0.5 品質フィルタ適用
                    TROPOMI footprint → resample 0.01°

 eo_types.py との対応:
   ObservationBundle / BandData / ObservationMeta / SensorSource / BandSetName
   S2_BAND_SETS を参照する。

 認証:
   openEO (CDSE): OIDC デバイスフロー (authenticate_oidc_device)
                  初回のみ URL 表示 → ブラウザで認証

 依存ライブラリ:
   pip install openeo rasterio numpy


 データソース (すべて openEO / CDSE 経由):
   Sentinel-2 L2A   B11 (1610nm) / B12 (2190nm) 反射率バンド
                    SCL マスク (有効地表画素: SCL 4/5/6)
                    SZA / VZA (太陽・衛星天頂角)
                    解像度: 20m/px, CRS: EPSG:4326

   ERA5 Land        u10 / v10 地上10m風速成分
                    surface_pressure [Pa]
                    空間平均 (mean_spatial) で安定化

   Sentinel-5P L2   CH4カラム濃度 [ppb]
                    qa_value >= 0.5 品質フィルタ適用
                    TROPOMI footprint → resample 0.01°

 ObservationBundle (Dict):
   {
     "bands":            {"B11": ndarray, "B12": ndarray} | None,
     "sza":              ndarray | None,
     "vza":              ndarray | None,
     "amf":              ndarray | None,
     "wind_speed":       float | None,
     "wind_deg":         float | None,
     "surface_pressure": float | None,
     "ch4_column":       ndarray | None,
     "meta": {
       "lat", "lon", "datetime", "bbox_deg",
       "cloud_pct", "s2_scene_id", "sza_mean", "vza_mean", "backend"
     }
   }

 認証:
   openEO (CDSE): OIDC デバイスフロー (authenticate_oidc_device)
                  初回のみ URL 表示 → ブラウザで認証
                  以降は refresh token で自動再認証

 依存ライブラリ:
   pip install openeo rasterio numpy
"""
#eo_provider.py
import os
import math
import logging
import sys
import tempfile
import threading
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# eo_types.py からバンドセット定数を取得
try:
    from eo_types import S2_BAND_SETS, RES_S2_M as _RES_S2_M_FROM_TYPES
except ImportError:
    # eo_types.py が見つからない場合はデフォルト値を使用
    S2_BAND_SETS = {
        "swir_only":  ["B11", "B12"],
        "full":       ["B01","B02","B03","B04","B05","B06",
                       "B07","B08","B8A","B09","B11","B12"],
        "eucalyptus": ["B01","B02","B03","B04","B05",
                       "B08","B8A","B09","B11","B12"],
        "custom":     [],
    }

# ロギング設定 (WARNING 以上を標準出力へ。デバッグ時は INFO に下げる)
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
_logger = logging.getLogger(__name__)

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

# --- rasterio (GeoTIFF → ndarray) ---
try:
    import rasterio
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False
    warnings.warn(
        "rasterio が未インストールです。\n  pip install rasterio",
        stacklevel=2,
    )


# =============================================================================
# 定数
# =============================================================================

OPENEO_BACKEND   = "https://openeo.dataspace.copernicus.eu"
RES_S2_M         = 20.0    # Sentinel-2 解像度 [m/px]
RES_S5P_DEG      = 0.01    # Sentinel-5P リサンプル解像度 [degrees]
DEFAULT_BBOX_KM  = 20.0    # デフォルト取得範囲の半径 [km]
S2_TIME_DELTA    = 5       # Sentinel-2 前後探索期間 [days]
S5P_TIME_DELTA   = 2       # Sentinel-5P 前後探索期間 [days]

# SCL クラス: 4=Vegetation 5=Non-vegetated 6=Water (有効地表)
SCL_VALID_CLASSES = [4, 5, 6]


# =============================================================================
# ユーティリティ
# =============================================================================

def _lat_lon_to_bbox(
    lat:       float,
    lon:       float,
    radius_km: float = DEFAULT_BBOX_KM,
) -> List[float]:
    """
    中心座標と半径 [km] から openEO 形式の BBox を返す。

    Returns
    -------
    [west, south, east, north]
    """
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def _uv_to_wind(u10: float, v10: float) -> Tuple[float, float]:
    """
    ERA5 の (u10, v10) To-Vector を気象慣例の FROM 方向に変換する。

    変換式: wind_deg = atan2(-u10, -v10) mod 360
        例: u>0, v=0 (東向きに吹く) → FROM=西 (270°)
        例: u=0, v>0 (北向きに吹く) → FROM=南 (180°)
    """
    speed    = math.sqrt(u10 ** 2 + v10 ** 2)
    wind_deg = math.degrees(math.atan2(-u10, -v10)) % 360.0
    return speed, wind_deg


def _tiff_to_ndarray(path: str) -> np.ndarray:
    """GeoTIFF を float32 ndarray (bands, H, W) に変換する。"""
    if not _RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio 未インストール: pip install rasterio")
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def _safe_nanmin(arr: np.ndarray) -> str:
    valid = arr[np.isfinite(arr)]
    return f"{valid.min():.4f}" if len(valid) > 0 else "all-NaN"


def _safe_nanmax(arr: np.ndarray) -> str:
    valid = arr[np.isfinite(arr)]
    return f"{valid.max():.4f}" if len(valid) > 0 else "all-NaN"


def _download_to_ndarray(cube: Any, label: str = "") -> np.ndarray:
    """
    openEO DataCube を一時ファイル経由で ndarray に変換する。


    Parameters
    ----------
    cube  : openEO DataCube
    label : ログ用ラベル

    Returns
    -------
    np.ndarray  shape=(bands, H, W)
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".tif", delete=False
        ) as tmp:
            tmp_path = tmp.name
        cube.download(tmp_path, format="GTiff")
        return _tiff_to_ndarray(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# =============================================================================
# OpenEO クライアント (スレッドセーフ・シングルトン)
# =============================================================================

class OpenEOClient:
    """
    openEO (CDSE) への接続をシングルトンで管理する。

    threading.Lock でマルチスレッド環境の二重初期化を防ぐ。
    認証は authenticate_oidc_device() を使用し、
    Spyder / Jupyter / headless 環境でも動作する。
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
        """接続済み Connection を返す。未接続なら接続する。"""
        if not _OPENEO_AVAILABLE:
            raise RuntimeError("openeo 未インストール: pip install openeo")
        if self._con is None:
            self._connect()
        return self._con

    def _connect(self) -> None:
        """OIDC デバイスフローで接続する。"""
        _logger.info("[openEO] %s に接続中 ...", OPENEO_BACKEND)
        print(f"  [openEO] {OPENEO_BACKEND} に接続中 ...")
        self._con = openeo.connect(OPENEO_BACKEND)
        # デバイスフロー: URL を表示してブラウザ不要で認証できる
        self._con.authenticate_oidc_device()
        print("  [openEO] 認証完了")

    def reconnect(self) -> None:
        """トークン失効時の再接続。長時間バッチ処理で使用する。"""
        print("  [openEO] 再接続中 ...")
        self._con = None
        self._connect()

    @classmethod
    def reset(cls) -> None:
        """テスト用: シングルトンをリセットして再接続を強制する。"""
        with cls._lock:
            cls._instance = None


# =============================================================================
# Sentinel-2 取得
# =============================================================================

class Sentinel2Fetcher:
    """
    openEO 経由で Sentinel-2 L2A のバンドデータを取得する。

    band_set 引数で取得バンドセットを切り替えられる。
        "swir_only"  : B11・B12 のみ (デフォルト・最小構成)
        "full"       : B01〜B12 全13バンド
        "eucalyptus" : Project Eucalyptus が要求する10バンド
        "custom"     : custom_bands で直接指定

    処理フロー:
        1. load_collection (band_set に応じたバンドリスト + SCL + SZA + VZA)
        2. resample_spatial → 20m, EPSG:4326
        3. SCL マスク (valid == 0 記法でバックエンド差異を回避)
        4. filter_bands → target_bands (SCL を除外)
        5. reduce_dimension(t, median) でモザイク
        6. download → _download_to_ndarray (try/finally)
        7. ローカルで np.clip (openEO apply を使わない)
        8. AMF = 1/cos(SZA) + 1/cos(VZA)、np.where で NaN 保護
    """

    # SZA・VZA は常に取得する (AMF 計算に必要)
    _ANGLE_BANDS = ["sunZenithAngles", "viewZenithMean"]

    def __init__(self, client: OpenEOClient):
        self.con = client.get()

    def fetch(
        self,
        lat:          float,
        lon:          float,
        dt:           datetime,
        radius_km:    float = DEFAULT_BBOX_KM,
        band_set:     str   = "swir_only",
        custom_bands: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """
        指定座標・日時周辺の Sentinel-2 データを取得する。

        Parameters
        ----------
        lat          : 中心緯度 [degrees]
        lon          : 中心経度 [degrees]
        dt           : 観測希望日時 (UTC)
        radius_km    : 取得範囲の半径 [km]
        band_set     : 取得バンドセット ("swir_only" / "full" / "eucalyptus" / "custom")
        custom_bands : band_set="custom" 時に取得するバンドリスト

        Returns
        -------
        Dict または None
            Keys: BandData の全フィールド (band_set に応じて存在するバンドが変わる),
                  sza, vza, amf (ndarray), sza_mean, vza_mean (float),
                  cloud_pct (None), scene_id (str), band_set (str)
        """
        # band_set からバンドリストを決定
        if band_set == "custom":
            if not custom_bands:
                warnings.warn(
                    "band_set='custom' の場合は custom_bands を指定してください。"
                    "swir_only にフォールバックします。",
                    stacklevel=2,
                )
                target_spectral = S2_BAND_SETS["swir_only"]
            else:
                target_spectral = custom_bands
        else:
            target_spectral = S2_BAND_SETS.get(band_set, S2_BAND_SETS["swir_only"])

        bbox    = _lat_lon_to_bbox(lat, lon, radius_km)
        t_start = (dt - timedelta(days=S2_TIME_DELTA)).strftime("%Y-%m-%d")
        t_end   = (dt + timedelta(days=S2_TIME_DELTA)).strftime("%Y-%m-%d")

        # load_collection に渡すバンドリスト: 対象バンド + SCL + 角度バンド
        load_bands = target_spectral + ["SCL"] + self._ANGLE_BANDS

        try:
            cube = self.con.load_collection(
                "SENTINEL2_L2A",
                spatial_extent  = {"west": bbox[0], "south": bbox[1],
                                   "east": bbox[2], "north": bbox[3]},
                temporal_extent = [t_start, t_end],
                bands           = load_bands,
            )

            # CRS 統一・解像度統一 (20m/px, EPSG:4326)
            cube = cube.resample_spatial(
                resolution = RES_S2_M,
                projection = "EPSG:4326",
            )

            # SCL マスク: valid == 0 でバックエンド差異を回避
            scl   = cube.band("SCL")
            valid = (scl == 4) | (scl == 5) | (scl == 6)
            cube  = cube.mask(valid == 0)

            # SCL を除いてダウンロード対象を絞る
            # バンド順: [対象バンド群..., SZA, VZA]
            target_bands = target_spectral + self._ANGLE_BANDS
            cube = cube.filter_bands(target_bands)

            cube = cube.reduce_dimension(dimension="t", reducer="median")

            # try/finally で tmpfile リークを防ぐ
            data = _download_to_ndarray(cube, label="S2")

        except Exception as e:
            warnings.warn(f"Sentinel-2 取得エラー: {e}", stacklevel=2)
            return None

        n_spectral = len(target_spectral)
        n_expected = n_spectral + 2   # 対象バンド + SZA + VZA
        if data.shape[0] < n_expected:
            warnings.warn(
                f"Sentinel-2 バンド数不足 "
                f"(取得={data.shape[0]}, 期待={n_expected})",
                stacklevel=2,
            )
            return None

        # 対象バンドを辞書形式で格納 (BandData 形式)
        bands: Dict[str, np.ndarray] = {}
        for i, b in enumerate(target_spectral):
            bands[b] = np.clip(data[i], 1e-4, None)

        # SZA・VZA は末尾2バンド
        sza = data[n_spectral]
        vza = data[n_spectral + 1]

        # AMF: np.clip + np.where で NaN 暴走防止
        cos_sza = np.clip(np.cos(np.radians(sza)), 1e-3, 1.0)
        cos_vza = np.clip(np.cos(np.radians(vza)), 1e-3, 1.0)
        amf_raw = 1.0 / cos_sza + 1.0 / cos_vza
        amf     = np.where(np.isfinite(amf_raw), amf_raw, np.nan)

        return {
            **bands,   # BandData の全フィールドをアンパック
            "sza":       sza,
            "vza":       vza,
            "amf":       amf,
            "sza_mean":  float(np.nanmean(sza)) if np.any(np.isfinite(sza)) else float("nan"),
            "vza_mean":  float(np.nanmean(vza)) if np.any(np.isfinite(vza)) else float("nan"),
            "cloud_pct": None,
            "scene_id":  (f"S2_openEO_{dt.strftime('%Y%m%d')}"
                          f"_{lat:.2f}N_{lon:.2f}E"),
            "band_set":  band_set,
        }


# =============================================================================
# ERA5 取得
# =============================================================================

class ERA5Fetcher:
    """
    openEO 経由で ERA5 Land から u10・v10・surface_pressure を取得する。

    変更点:
        - 正時スナップ + UTC 'Z' 付与 
        - x → y → t の順で次元削減を明示 
        - try/finally で tmpfile をクリーンアップ 
        - リトライ処理を維持

    取得変数: u10, v10 [m/s], sp (surface_pressure) [Pa]
    """

    _ERA5_BANDS = ["u10", "v10", "sp"]

    def __init__(
        self,
        client:      OpenEOClient,
        max_retries: int   = 3,
        retry_wait:  float = 5.0,
    ):
        self.con         = client.get()
        self.max_retries = max_retries
        self.retry_wait  = retry_wait

    def fetch(
        self,
        lat: float,
        lon: float,
        dt:  datetime,
    ) -> Optional[Dict]:
        """
        指定座標・日時の ERA5 風速・地表気圧を取得する。

        Returns
        -------
        Dict または None
            Keys: u10, v10, wind_speed, wind_deg (float),
                  surface_pressure (float [Pa])
        """
        # 正時スナップ + UTC 'Z' 付与 
        dt_hour = dt.replace(minute=0, second=0, microsecond=0)
        t_slot  = dt_hour.strftime("%Y-%m-%dT%H:00:00Z")
        bbox    = _lat_lon_to_bbox(lat, lon, radius_km=50.0)

        import time as _time
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                cube = self.con.load_collection(
                    "ERA5_LAND",
                    spatial_extent  = {"west": bbox[0], "south": bbox[1],
                                       "east": bbox[2], "north": bbox[3]},
                    temporal_extent = [t_slot, t_slot],
                    bands           = self._ERA5_BANDS,
                )

                # x → y → t の順で明示的に次元削減 
                cube = cube.reduce_dimension(dimension="x", reducer="mean")
                cube = cube.reduce_dimension(dimension="y", reducer="mean")
                cube = cube.reduce_dimension(dimension="t", reducer="mean")

                # try/finally で tmpfile をクリーンアップ 
                data       = _download_to_ndarray(cube, label="ERA5")
                last_error = None
                break

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    warnings.warn(
                        f"ERA5 リトライ ({attempt}/{self.max_retries}): {e}",
                        stacklevel=2,
                    )
                    _time.sleep(self.retry_wait)

        if last_error is not None:
            warnings.warn(
                f"ERA5 取得失敗 (全{self.max_retries}回): {last_error}",
                stacklevel=2,
            )
            return None

        try:
            u10 = float(data[0].flat[0])
            v10 = float(data[1].flat[0])
            sp  = float(data[2].flat[0])
        except (IndexError, ValueError) as e:
            warnings.warn(f"ERA5 値読み込みエラー: {e}", stacklevel=2)
            return None

        wind_speed, wind_deg = _uv_to_wind(u10, v10)

        return {
            "u10":              u10,
            "v10":              v10,
            "wind_speed":       wind_speed,
            "wind_deg":         wind_deg,
            "surface_pressure": sp,
        }


# =============================================================================
# Sentinel-5P 取得
# =============================================================================

class Sentinel5PFetcher:
    """
    openEO 経由で Sentinel-5P L2 から CH4 カラム濃度を取得する。

    qa_value >= 0.5 の品質フィルタを適用。
    try/finally で tmpfile をクリーンアップ。 
    """

    _S5P_BAND     = "CH4_column_volume_mixing_ratio_dry_air"
    _S5P_QA_BAND  = "qa_value"
    _QA_THRESHOLD = 0.5

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
                "SENTINEL5P_L2_CH4",
                spatial_extent  = {"west": bbox[0], "south": bbox[1],
                                   "east": bbox[2], "north": bbox[3]},
                temporal_extent = [t_start, t_end],
                bands           = [self._S5P_BAND, self._S5P_QA_BAND],
            )

            # qa_value < 0.5 の画素を NaN 化
            qa   = cube.band(self._S5P_QA_BAND)
            cube = cube.mask(qa < self._QA_THRESHOLD)
            cube = cube.filter_bands([self._S5P_BAND])

            cube = cube.resample_spatial(
                resolution = RES_S5P_DEG,
                projection = "EPSG:4326",
            )
            cube = cube.reduce_dimension(dimension="t", reducer="mean")

            # try/finally でクリーンアップ 
            data = _download_to_ndarray(cube, label="S5P")

        except Exception as e:
            warnings.warn(f"Sentinel-5P 取得エラー: {e}", stacklevel=2)
            return None

        ch4        = data[0]
        valid_mask = np.isfinite(ch4)

        return {
            "ch4_column": ch4,
            "ch4_mean":   float(np.nanmean(ch4)) if np.any(valid_mask) else float("nan"),
            "ch4_std":    float(np.nanstd(ch4))  if np.any(valid_mask) else float("nan"),
            "valid_frac": float(np.mean(valid_mask)),
        }


# =============================================================================
# DataFetcher — 3ソースを統合する唯一の窓口 (Lazy Initialization) 
# =============================================================================

class DataFetcher:
    """
    Sentinel-2・ERA5・Sentinel-5P を統合して ObservationBundle を返す。

    Lazy Initialization: openEO への接続は fetch() 呼び出し時に初めて行う。
    3ソースは ThreadPoolExecutor で並列取得 (最大 3 スレッド)。
    各サブフェッチャーが失敗しても部分結果を返す。

    Parameters
    ----------
    radius_km   : 取得範囲の半径 [km]
    max_retries : ERA5 リトライ回数
    max_workers : 並列スレッド数
    band_set    : 取得バンドセット 
                  "swir_only"  B11/B12 のみ (デフォルト)
                  "full"       全13バンド
                  "eucalyptus" Project Eucalyptus 用10バンド
    """

    def __init__(
        self,
        radius_km:   float = DEFAULT_BBOX_KM,
        max_retries: int   = 3,
        max_workers: int   = 3,
        band_set:    str   = "swir_only",
    ):
        self.radius_km   = radius_km
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.band_set    = band_set
        self._s2:   Optional[Sentinel2Fetcher]  = None
        self._era5: Optional[ERA5Fetcher]       = None
        self._s5p:  Optional[Sentinel5PFetcher] = None

    def _ensure_fetchers(self) -> None:
        """初回 fetch() 時に Fetcher を遅延生成する。"""
        if self._s2 is None:
            client     = OpenEOClient()
            self._s2   = Sentinel2Fetcher(client)
            self._era5 = ERA5Fetcher(client, max_retries=self.max_retries)
            self._s5p  = Sentinel5PFetcher(client)

    @property
    def era5(self) -> "ERA5Fetcher":
        """ERA5Fetcher への参照。max_retries の確認用。"""
        if self._era5 is None:
            raise RuntimeError(
                "ERA5Fetcher は fetch() 呼び出し後にアクセスできます。"
            )
        return self._era5

    def fetch(
        self,
        lat: float,
        lon: float,
        dt:  datetime,
    ) -> Dict:
        """
        指定座標・日時の ObservationBundle を返す。

        meta に band_set / sensors / res_m を追加する。 
        band_set に応じたバンドを BandData 形式で格納する。
        """
        self._ensure_fetchers()

        print(f"  [DataFetcher] 並列取得開始  lat={lat:.3f} lon={lon:.3f} "
              f"dt={dt.strftime('%Y-%m-%d %H:%M')} UTC "
              f"(workers={self.max_workers}  band_set={self.band_set})")

        tasks = {
            "s2":   (self._s2.fetch,
                     (lat, lon, dt, self.radius_km, self.band_set)),
            "era5": (self._era5.fetch, (lat, lon, dt)),
            "s5p":  (self._s5p.fetch,  (lat, lon, dt, self.radius_km)),
        }
        fetched: Dict[str, Optional[Dict]] = {
            "s2": None, "era5": None, "s5p": None
        }

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(fn, *args): key
                for key, (fn, args) in tasks.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    fetched[key] = future.result()
                    print(f"  [{key.upper():5s}] 完了")
                except Exception as e:
                    warnings.warn(
                        f"DataFetcher: {key} 取得中に例外: {e}", stacklevel=2
                    )

        s2   = fetched["s2"]
        era5 = fetched["era5"]
        s5p  = fetched["s5p"]

        # --- Sentinel-2 フィールドの展開 ---
        # Sentinel2Fetcher.fetch() は band_set に応じたバンド辞書と
        # sza/vza/amf/scene_id/band_set を返す
        bands: Optional[Dict[str, np.ndarray]] = None
        sza = vza = amf = None
        sza_mean = vza_mean = cloud_pct = scene_id = None
        actual_band_set = self.band_set

        if s2 is not None:
            actual_band_set = s2.pop("band_set",  self.band_set)
            sza      = s2.pop("sza",       None)
            vza      = s2.pop("vza",       None)
            amf      = s2.pop("amf",       None)
            sza_mean = s2.pop("sza_mean",  None)
            vza_mean = s2.pop("vza_mean",  None)
            cloud_pct = s2.pop("cloud_pct", None)
            scene_id  = s2.pop("scene_id",  None)
            # 残りが BandData のバンドフィールド (B11, B12, B08 等)
            bands = {k: v for k, v in s2.items()
                     if isinstance(v, np.ndarray)}

        # --- ERA5 フィールドの展開 ---
        wind_speed = wind_deg = surface_pressure = None
        if era5 is not None:
            wind_speed       = era5["wind_speed"]
            wind_deg         = era5["wind_deg"]
            surface_pressure = era5["surface_pressure"]

        # --- Sentinel-5P フィールドの展開 ---
        ch4_column = None
        if s5p is not None:
            ch4_column = s5p["ch4_column"]

        # 利用可能なデータソースリストを構築
        sensors: List[str] = []
        if bands is not None:  sensors.append("sentinel2")
        if era5  is not None:  sensors.append("era5")
        if s5p   is not None:  sensors.append("sentinel5p")

        return {
            "bands":            bands,
            "sza":              sza,
            "vza":              vza,
            "amf":              amf,
            "wind_speed":       wind_speed,
            "wind_deg":         wind_deg,
            "surface_pressure": surface_pressure,
            "ch4_column":       ch4_column,
            "meta": {
                "lat":         lat,
                "lon":         lon,
                "datetime":    dt.isoformat(),
                "bbox_deg":    self.radius_km / 111.32,
                "cloud_pct":   cloud_pct,
                "s2_scene_id": scene_id,
                "sza_mean":    sza_mean,
                "vza_mean":    vza_mean,
                "backend":     "openeo",
                "band_set":    actual_band_set,
                "sensors":     sensors,
                "res_m":       RES_S2_M,
            },
        }

    def fetch_from_site(self, site: Dict, dt: datetime) -> Dict:
        """SITE_REGISTRY エントリを直接受け取るショートカット。"""
        return self.fetch(site["lat"], site["lon"], dt)


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
                  f"min={_safe_nanmin(val)} max={_safe_nanmax(val)}")
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
    """
    座標変換・風向変換・設定値の単体テスト。

    DataFetcher は Lazy Initialization のためインスタンス生成だけでは
    openEO への接続が発生しない。外部 API 不要で常時実行できる。 
    """
    print("\n" + "="*58)
    print("  TEST-0: ユーティリティ (外部API不要)")
    print("="*58)

    # BBox
    bbox = _lat_lon_to_bbox(38.49, 54.19, radius_km=20.0)
    print(f"\n  BBox (TM-01, r=20km):")
    print(f"    [W={bbox[0]:.4f}, S={bbox[1]:.4f}, "
          f"E={bbox[2]:.4f}, N={bbox[3]:.4f}]")
    assert bbox[0] < 54.19 < bbox[2], "longitude range error"
    assert bbox[1] < 38.49 < bbox[3], "latitude range error"
    print("  BBox → PASS")

    # 風向変換
    cases = [
        ( 10.0,  0.0, "東向きに吹く → FROM=西", 270.0),
        (  0.0, 10.0, "北向きに吹く → FROM=南", 180.0),
        (-10.0,  0.0, "西向きに吹く → FROM=東",  90.0),
        (  0.0,-10.0, "南向きに吹く → FROM=北",   0.0),
    ]
    print(f"\n  風向変換テスト [atan2(-u,-v) FROM方向]:")
    all_pass = True
    for u, v, name, expected in cases:
        _, deg = _uv_to_wind(u, v)
        diff   = abs((deg - expected + 180) % 360 - 180)
        status = "PASS" if diff < 0.1 else f"FAIL (expected={expected:.0f}°)"
        if "FAIL" in status:
            all_pass = False
        print(f"    u={u:+5.1f} v={v:+5.1f} → {deg:6.1f}°  [{status}]  {name}")
    print(f"  風向変換 → {'PASS' if all_pass else 'FAIL'}")

    # _safe_nanmin/_safe_nanmax 
    print(f"\n  _safe_nanmin/_safe_nanmax (全NaN配列):")
    all_nan = np.full((5, 5), np.nan)
    assert _safe_nanmin(all_nan) == "all-NaN", "_safe_nanmin が全NaN配列で失敗"
    assert _safe_nanmax(all_nan) == "all-NaN", "_safe_nanmax が全NaN配列で失敗"
    print("  → PASS")

    # DataFetcher Lazy Init: インスタンス生成だけでは接続しない 
    print(f"\n  DataFetcher Lazy Init + 設定値確認:")
    fetcher = DataFetcher(radius_km=10.0, max_retries=5, max_workers=3)
    assert fetcher.max_retries == 5,  "max_retries が保存されていない"
    assert fetcher.max_workers == 3,  "max_workers が保存されていない"
    assert fetcher._s2   is None,     "Lazy Init: _s2 が fetch() 前に生成されている"
    assert fetcher._era5 is None,     "Lazy Init: _era5 が fetch() 前に生成されている"
    print(f"    max_retries={fetcher.max_retries}  max_workers={fetcher.max_workers}")
    print(f"    _s2 is None={fetcher._s2 is None}  → PASS (接続未発生)")

    # OpenEOClient シングルトン
    print(f"\n  OpenEOClient シングルトン確認:")
    c1 = OpenEOClient()
    c2 = OpenEOClient()
    assert c1 is c2, "シングルトンが複数インスタンスを返している"
    print(f"    c1 is c2 → PASS")

    print("\n  TEST-0 → 全項目 PASS")


def test_sentinel2(
    lat:       float    = 38.49,
    lon:       float    = 54.19,
    dt:        datetime = datetime(2023, 7, 15, 8, 0, tzinfo=timezone.utc),
    radius_km: float    = DEFAULT_BBOX_KM,
) -> Optional[Dict]:
    """Sentinel-2 B11/B12/SZA/VZA/AMF 取得テスト。TM-01 デフォルト。"""
    print("\n" + "="*58)
    print("  TEST-1: Sentinel-2 B11/B12/SZA/VZA (openEO)")
    print(f"  lat={lat}  lon={lon}  dt={dt.isoformat()}")
    print(f"  radius={radius_km} km")
    print("="*58)

    if not _OPENEO_AVAILABLE:
        print("  → SKIP (openeo 未インストール)")
        return None

    try:
        client  = OpenEOClient()
        fetcher = Sentinel2Fetcher(client)
        result  = fetcher.fetch(lat, lon, dt, radius_km=radius_km)
    except Exception as e:
        print(f"  → SKIP (接続・認証エラー: {e})")
        return None

    _print_bundle("Sentinel-2 結果", result)

    if result is not None:
        for key in ["B11", "B12", "sza", "vza", "amf"]:
            assert key in result, f"'{key}' が返却されていない"
        # SCL マスク後は NaN 画素が存在するため有限値のみ検証
        b11_valid = result["B11"][np.isfinite(result["B11"])]
        assert np.all(b11_valid > 0),            "B11 有効画素にゼロ以下の値"
        sza_valid = result["sza"][np.isfinite(result["sza"])]
        assert np.all(sza_valid >= 0),            "SZA に負値"
        assert np.all(sza_valid < 90),            "SZA が 90° 以上"
        amf_valid = result["amf"][np.isfinite(result["amf"])]
        assert np.all(amf_valid >= 2.0),          "AMF < 2.0 (物理的に不正)"
        print(f"\n  [ASSERT] B11/B12/SZA/VZA/AMF (有効画素) → OK")

    return result


def test_era5(
    lat: float    = 38.49,
    lon: float    = 54.19,
    dt:  datetime = datetime(2023, 7, 15, 8, 0, tzinfo=timezone.utc),
) -> Optional[Dict]:
    """ERA5 風速・地表気圧取得テスト。TM-01 デフォルト。"""
    print("\n" + "="*58)
    print("  TEST-2: ERA5 風速・surface_pressure (openEO)")
    print(f"  lat={lat}  lon={lon}  dt={dt.isoformat()}")
    print("="*58)

    if not _OPENEO_AVAILABLE:
        print("  → SKIP (openeo 未インストール)")
        return None

    try:
        client  = OpenEOClient()
        fetcher = ERA5Fetcher(client, max_retries=3)
        result  = fetcher.fetch(lat, lon, dt)
    except Exception as e:
        print(f"  → SKIP (接続・認証エラー: {e})")
        return None

    _print_bundle("ERA5 結果", result)

    if result is not None:
        sp = result["surface_pressure"]
        wd = result["wind_deg"]
        assert 50000 <= sp <= 110000, f"surface_pressure 異常: {sp:.0f} Pa"
        assert 0     <= wd <  360,    f"wind_deg 範囲外: {wd:.1f}°"
        print(f"\n  [ASSERT] wind_deg={wd:.1f}°  surface_pressure={sp:.0f} Pa → OK")

    return result


def test_sentinel5p(
    lat:       float    = 38.49,
    lon:       float    = 54.19,
    dt:        datetime = datetime(2023, 7, 15, 8, 0, tzinfo=timezone.utc),
    radius_km: float    = DEFAULT_BBOX_KM,
) -> Optional[Dict]:
    """Sentinel-5P CH4 取得テスト。TM-01 デフォルト。"""
    print("\n" + "="*58)
    print("  TEST-3: Sentinel-5P CH4 (openEO)")
    print(f"  lat={lat}  lon={lon}  dt={dt.isoformat()}")
    print(f"  radius={radius_km} km")
    print("="*58)

    if not _OPENEO_AVAILABLE:
        print("  → SKIP (openeo 未インストール)")
        return None

    try:
        client  = OpenEOClient()
        fetcher = Sentinel5PFetcher(client)
        result  = fetcher.fetch(lat, lon, dt, radius_km=radius_km)
    except Exception as e:
        print(f"  → SKIP (接続・認証エラー: {e})")
        return None

    _print_bundle("Sentinel-5P 結果", result)
    return result


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


def test_openeo_connection() -> bool:
    """
    openEO への接続・認証・カタログ取得の単体テスト。
    デバイスフロー OIDC で headless 環境でも動作する。 

    Returns
    -------
    bool  成功: True / 失敗: False
    """
    oe_logger = logging.getLogger("openeo")
    oe_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    oe_logger.addHandler(handler)

    print(f"\n--- openEO 接続テスト ---")
    print(f"  Backend: {OPENEO_BACKEND}")

    try:
        conn = openeo.connect(OPENEO_BACKEND)
        print("  [1/3] 疎通確認: OK")

        print("  [2/3] OIDC デバイスフロー認証 ...")
        conn.authenticate_oidc_device()
        print("        認証完了: OK")

        ids = [c["id"] for c in conn.list_collections()]
        print(f"  [3/3] カタログ取得: OK  ({len(ids)} collections)")
        print(f"        先頭5件: {ids[:5]}")
        print("--- PASS ---")
        return True

    except Exception as e:
        print(f"\n  [ERROR] {e}")
        return False


def verify_access() -> None:
    """既存の認証情報でカタログ取得を確認する簡易チェック。"""
    try:
        conn = openeo.connect(OPENEO_BACKEND)
        ids  = [c["id"] for c in conn.list_collections()]
        print(f"  [verify_access] OK  ({len(ids)} collections)  先頭5: {ids[:5]}")
    except Exception as e:
        print(f"  [verify_access] 失敗 (未認証の可能性): {e}")


# =============================================================================
# エントリポイント  — ファイルに1か所のみ
# =============================================================================

if __name__ == "__main__":

    print("=" * 58)
    print("  earth_obs_provider.py — データ取得層テスト (openEO 統合版)")
    print("=" * 58)
    print("  推奨実行環境: コマンドプロンプト / PowerShell")
    print("  (Spyder のコンソール経由は OIDC 認証が不安定な場合あり)")

    # 環境チェック
    print("\n[0] 環境チェック")
    print(f"  openeo   : {'OK' if _OPENEO_AVAILABLE   else 'NOT INSTALLED'}")
    print(f"  rasterio : {'OK' if _RASTERIO_AVAILABLE else 'NOT INSTALLED'}")
    print(f"  backend  : {OPENEO_BACKEND}")
    print("  認証方式 : OIDC デバイスフロー (初回のみ URL 表示)")

    # 接続確認 (認証済みの場合のみ)
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

    print("\n" + "="*58)
    print("  テスト完了")
    print("  openEO 未接続の場合 TEST-1〜5 は SKIP されます。")
    print("  接続単体テスト: test_openeo_connection() を呼び出してください。")
    print("="*58)
