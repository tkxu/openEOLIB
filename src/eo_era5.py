"""
ApacheLicense2.0
Copyright (c) 2026 tkxu
"""
# eo_era5.py
from __future__ import annotations

import math
import os
import tempfile
import time
import warnings
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import cdsapi
    _CDSAPI_AVAILABLE = True
except ImportError:
    _CDSAPI_AVAILABLE = False

try:
    import xarray as xr
    _XARRAY_AVAILABLE = True
except ImportError:
    _XARRAY_AVAILABLE = False

# =============================================================================
# 定数
# =============================================================================

DEFAULT_RADIUS_KM = 10.0
ERA5_DATASET      = "reanalysis-era5-land"
ERA5_VARIABLES    = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
]

# =============================================================================
# モジュールレベルユーティリティ
# _uv_to_wind は eo_provider.py が from eo_era5 import _uv_to_wind で使用する
# =============================================================================

def _lat_lon_to_area(
    lat:       float,
    lon:       float,
    radius_km: float,
) -> List[float]:
    """
    中心座標と半径から ERA5 area パラメータ [N, W, S, E] を返す。

    ERA5 の area 順は [North, West, South, East] であることに注意。
    eo_provider._lat_lon_to_bbox の (west, south, east, north) とは順序が異なる。
    """
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
    return [lat + dlat, lon - dlon, lat - dlat, lon + dlon]


def _uv_to_wind(u10: float, v10: float) -> Tuple[float, float]:
    """
    ERA5 の (u10, v10) To-Vector を気象慣例の FROM 方向に変換する。

    変換式: wind_deg = atan2(-u10, -v10) mod 360
        例: u>0, v=0 (東向きに吹く) → FROM=西 (270°)
        例: u=0, v>0 (北向きに吹く) → FROM=南 (180°)

    Returns
    -------
    (wind_speed [m/s], wind_deg [degrees])
    """
    speed = math.sqrt(u10 ** 2 + v10 ** 2)
    deg   = math.degrees(math.atan2(-u10, -v10)) % 360.0
    return speed, deg


# =============================================================================
# ERA5Fetcher
# =============================================================================

class ERA5Fetcher:
    """
    cdsapi を使用して ERA5 Land から風速・地表気圧を取得する。

    fetch() が公開 API。内部は以下のプライベートメソッドに分割されている:
        _download()    : CDS API へのリクエスト送信 + NetCDF ダウンロード
        _load_point()  : NetCDF から指定座標の点データを抽出
        _format()      : 生データ (u10/v10/sp) を ObservationBundle 形式に変換

    リトライ戦略:
        指数バックオフ (retry_wait × 2^attempt)。
        全 max_retries 回失敗した場合は None を返す。

    Parameters
    ----------
    max_retries : 最大リトライ回数 (デフォルト: 5)
    retry_wait  : 初回リトライ待機秒数 (デフォルト: 5.0)
    """

    def __init__(
        self,
        max_retries: int   = 5,
        retry_wait:  float = 5.0,
    ):
        if not _CDSAPI_AVAILABLE:
            raise RuntimeError(
                "cdsapi が未インストールです。\n"
                "  pip install cdsapi"
            )
        if not _XARRAY_AVAILABLE:
            raise RuntimeError(
                "xarray が未インストールです。\n"
                "  pip install xarray"
            )
        self.max_retries = max_retries
        self.retry_wait  = retry_wait
        self.client      = cdsapi.Client(quiet=True, progress=False)

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def fetch(
        self,
        lat:       float,
        lon:       float,
        dt:        datetime,
        radius_km: float = DEFAULT_RADIUS_KM,
    ) -> Optional[Dict]:
        """
        指定座標・日時の ERA5 値を取得する。

        Parameters
        ----------
        lat       : 中心緯度 [degrees]
        lon       : 中心経度 [degrees]
        dt        : 観測希望日時 (UTC)
        radius_km : 取得範囲の半径 [km]

        Returns
        -------
        Dict または None (全リトライ失敗時)
            Keys: u10, v10, wind_speed, wind_deg, surface_pressure, backend
        """
        dt_hour = dt.replace(minute=0, second=0, microsecond=0)
        area    = _lat_lon_to_area(lat, lon, radius_km)

        for attempt in range(self.max_retries):
            tmp_path = None
            try:
                tmp_path = self._download(area, dt_hour)
                raw      = self._load_point(tmp_path, lat, lon)
                return self._format(raw)

            except Exception as e:
                warnings.warn(
                    f"[ERA5 retry {attempt + 1}/{self.max_retries}] {e}",
                    stacklevel=2,
                )
                # 最終リトライ後はスリープ不要
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_wait * (2 ** attempt))

            finally:
                # ダウンロード失敗時も一時ファイルを確実に削除する
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        return None

    # ------------------------------------------------------------------
    # プライベートメソッド
    # ------------------------------------------------------------------

    def _download(
        self,
        area:    List[float],
        dt_hour: datetime,
    ) -> str:
        """
        CDS API にリクエストを送信し、NetCDF を一時ファイルに保存する。

        Returns
        -------
        str : 一時ファイルのパス
              呼び出し元 fetch() の finally で削除される。
        """
        fd, tmp_path = tempfile.mkstemp(suffix=".nc")
        os.close(fd)

        request = {
            "variable": ERA5_VARIABLES,
            "year":     dt_hour.strftime("%Y"),
            "month":    dt_hour.strftime("%m"),
            "day":      dt_hour.strftime("%d"),
            "time":     dt_hour.strftime("%H:00"),
            "area":     area,
            "format":   "netcdf",
        }

        result = self.client.retrieve(ERA5_DATASET, request)
        result.download(tmp_path)

        # CDS 新API (2024年移行後) はNetCDFをZIPに包んで返す場合がある。
        # ZIPと判定された場合は展開し、内部の最初の .nc ファイルに差し替える。
        if zipfile.is_zipfile(tmp_path):
            fd_nc, nc_path = tempfile.mkstemp(suffix=".nc")
            os.close(fd_nc)
            with zipfile.ZipFile(tmp_path, "r") as zf:
                nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                if not nc_names:
                    os.remove(nc_path)
                    raise RuntimeError(
                        f"ZIPアーカイブ内に .nc ファイルが見つかりません: {zf.namelist()}"
                    )
                with zf.open(nc_names[0]) as src, open(nc_path, "wb") as dst:
                    dst.write(src.read())
            os.remove(tmp_path)
            tmp_path = nc_path

        return tmp_path


    def _load_point(
        self,
        path: str,
        lat:  float,
        lon:  float,
    ) -> Dict:
        """
        NetCDF ファイルから指定座標の最近傍点データを抽出する。

        time 次元が存在する場合は最初のタイムステップを選択する。
        これは fetch() が 1 時刻分のみリクエストするため通常は
        1 タイムステップだが、API の返値が複数時刻を含む場合への
        安全策として明示的に isel(time=0) を適用する。

        Returns
        -------
        Dict : {"u10": float, "v10": float, "sp": float}
        """
        with xr.open_dataset(path, engine="netcdf4") as ds:
        
            if "time" in ds.dims:
                ds = ds.isel(time=0)
        
            lat_vals = ds["latitude"].values
            lon_vals = ds["longitude"].values
        
            lat_idx = abs(lat_vals - lat).argmin()
            lon_idx = abs(lon_vals - lon).argmin()
        
            pt = ds.isel(latitude=lat_idx, longitude=lon_idx)
        
            return {
                "u10": float(pt["u10"].item()),
                "v10": float(pt["v10"].item()),
                "sp":  float(pt["sp"].item()),
            }

    def _format(self, raw: Dict) -> Dict:
        """
        生データ (u10/v10/sp) を ObservationBundle 互換形式に変換する。

        Returns
        -------
        Dict
            Keys: u10, v10, wind_speed, wind_deg, surface_pressure, backend
        """
        wind_speed, wind_deg = _uv_to_wind(raw["u10"], raw["v10"])
        return {
            "u10":              raw["u10"],
            "v10":              raw["v10"],
            "wind_speed":       wind_speed,
            "wind_deg":         wind_deg,
            "surface_pressure": raw["sp"],
            "backend":          "cdsapi_stable",
        }


# =============================================================================
# テストユーティリティ
# =============================================================================

def _print_result(result: Optional[Dict]) -> None:
    """取得結果を整形表示するヘルパー。"""
    print("\n" + "=" * 60)
    print("  ERA5 Result")
    print("=" * 60)

    if result is None:
        print("  result = None")
        return

    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.6f}")
        else:
            print(f"  {k:20s}: {v}")


def test_uv_to_wind() -> None:
    """_uv_to_wind の変換方向テスト。"""
    print("\n" + "=" * 60)
    print("  TEST-1  _uv_to_wind")
    print("=" * 60)

    test_cases = [
        # u10,   v10,  expected_deg,  description
        ( 10.0,  0.0,  270.0, "eastward wind  → FROM west  (270°)"),
        (-10.0,  0.0,   90.0, "westward wind  → FROM east  ( 90°)"),
        (  0.0, 10.0,  180.0, "northward wind → FROM south (180°)"),
        (  0.0,-10.0,    0.0, "southward wind → FROM north (  0°)"),
    ]

    all_pass = True

    for u10, v10, expected, label in test_cases:
        speed, deg = _uv_to_wind(u10, v10)
        diff       = abs((deg - expected + 180) % 360 - 180)
        passed     = diff < 0.1
        if not passed:
            all_pass = False
        print(
            f"  u10={u10:+6.1f}  v10={v10:+6.1f}"
            f"  → speed={speed:6.3f}  deg={deg:7.2f}"
            f"  [{'PASS' if passed else 'FAIL'}]  {label}"
        )

    assert all_pass, "_uv_to_wind に失敗したケースがあります"
    print("\n  RESULT: PASS")


def test_area_conversion() -> None:
    """_lat_lon_to_area の範囲テスト。"""
    print("\n" + "=" * 60)
    print("  TEST-2  _lat_lon_to_area")
    print("=" * 60)

    area                    = _lat_lon_to_area(38.49, 54.19, 10.0)
    north, west, south, east = area

    print(f"  north = {north:.6f}")
    print(f"  west  = {west:.6f}")
    print(f"  south = {south:.6f}")
    print(f"  east  = {east:.6f}")

    assert north > south,  f"north ({north:.4f}) <= south ({south:.4f})"
    assert east  > west,   f"east ({east:.4f}) <= west ({west:.4f})"
    assert north > 38.49,  "north が中心緯度より小さい"
    assert south < 38.49,  "south が中心緯度より大きい"
    assert east  > 54.19,  "east が中心経度より小さい"
    assert west  < 54.19,  "west が中心経度より大きい"

    print("\n  RESULT: PASS")


def test_era5_fetch() -> None:
    """ERA5Fetcher.fetch() の統合テスト (CDS API 接続が必要)。"""
    print("\n" + "=" * 60)
    print("  TEST-3  ERA5Fetcher.fetch()")
    print("=" * 60)

    if not _CDSAPI_AVAILABLE or not _XARRAY_AVAILABLE:
        print("  SKIPPED: cdsapi または xarray が未インストール")
        return

    try:
        fetcher = ERA5Fetcher(max_retries=5, retry_wait=5.0)
    except RuntimeError as e:
        print(f"  SKIPPED: {e}")
        return

    dt = datetime(2023, 7, 15, 8, 0)

    result = fetcher.fetch(lat=38.49, lon=54.19, dt=dt, radius_km=10.0)
    _print_result(result)

    # CDS API 未認証・ネットワーク不可の場合は SKIP（後続テストをブロックしない）
    if result is None:
        warnings.warn(
            "ERA5 fetch が None を返しました。"
            "CDS API の認証設定 (~/.cdsapirc) とネットワーク接続を確認してください。",
            stacklevel=2,
        )
        print("\n  RESULT: SKIPPED (None returned)")
        return

    assert "u10"              in result, "u10 が返却されていない"
    assert "v10"              in result, "v10 が返却されていない"
    assert "wind_speed"       in result, "wind_speed が返却されていない"
    assert "wind_deg"         in result, "wind_deg が返却されていない"
    assert "surface_pressure" in result, "surface_pressure が返却されていない"
    assert "backend"          in result, "backend が返却されていない"

    assert 0.0    <= result["wind_deg"]         <  360.0,  f"wind_deg 範囲外: {result['wind_deg']}"
    assert         result["wind_speed"]         >= 0.0,    f"wind_speed が負値: {result['wind_speed']}"
    assert 50000.0 <= result["surface_pressure"] <= 110000.0, \
        f"surface_pressure 異常: {result['surface_pressure']}"

    print("\n  RESULT: PASS")


# =============================================================================
# エントリポイント
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  eo_era5.py  standalone test")
    print("=" * 60)

    print("\n  Environment")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  cdsapi  : {'OK' if _CDSAPI_AVAILABLE  else 'NG (pip install cdsapi)'}")
    print(f"  xarray  : {'OK' if _XARRAY_AVAILABLE  else 'NG (pip install xarray)'}")

    try:
        import netCDF4
        print("  netCDF4 : OK")
    except Exception:
        print("  netCDF4 : NG (pip install netCDF4)")

    try:
        test_uv_to_wind()
        test_area_conversion()
        test_era5_fetch()

        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n  Interrupted")

    except Exception as e:
        print("\n" + "=" * 60)
        print("  TEST FAILED")
        print("=" * 60)
        print(f"\n  {type(e).__name__}: {e}")
        raise
