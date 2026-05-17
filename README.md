# openEOLIB

Sentinel-2・ERA5・Sentinel-5P の衛星観測データを取得し、メタンガス漏洩検出の検証パイプラインを構築するための Python ライブラリです。

推論エンジンを外部から差し込む設計になっており、独自の排出量推定アルゴリズムと組み合わせて使用できます。

---

## 特徴

- **実データ取得**：openEO (CDSE) 経由で Sentinel-2・ERA5・Sentinel-5P を一括取得
- **全バンド対応**：B11/B12 のみ (デフォルト) から全13バンド・Eucalyptus セットまで切り替え可能
- **合成データ生成**：ガウスプルーム拡散モデルによる物理的に妥当な合成観測データ
- **Mismatch Injection**：ロバスト性を検証
- **差し込み可能な推論エンジン**：`InferenceEngine` を実装すれば任意のアルゴリズムを接続可能
- **可視化**：プルーム空間場・MBSP・MLLR 分布・ROC 曲線・品質フラグを統合レポートとして出力

---

## ファイル構成

```
/
├── eo_types.py           モジュール間インターフェースの型定義 (TypedDict)
├── eo_engines.py         InferenceEngine / Provider ABC + NullInferenceEngine
├── eo_pipeline.py        観測データ処理パイプライン
├── eo_provider.py        実データ取得層 (openEO / CDSE)
├── eo_era5.py            ERA5 風速・地表気圧取得 (cdsapi)
├── eo_simulator.py       ガウスプルーム合成データ生成器
├── eo_roc.py             ROC 曲線構築モジュール
├── eo_cache.py           キャッシュ + Provenance 管理
├── eo_visualisation.py   可視化パネル群
├── eo_report.py          マスターフィギュア統合出力 (ReportBuilder)
└── examples/
    └── main_sample.py    エンドツーエンドサンプル
                          (MBSPThresholdEngine  / CustomEngine を含む)
```

推論エンジンの実装（排出量逆推定アルゴリズム）は本リポジトリには含まれていません。
`InferenceEngine` を継承したクラスを独自に実装して差し込んでください（後述）。

### モジュール依存関係

```
eo_types.py                 唯一の共有インターフェース定義
    ↑
eo_engines.py               InferenceEngine / Provider ABC + NullInferenceEngine
    ↑
    ├── eo_pipeline.py       パイプライン (ROC 構築は eo_roc に委譲)
    │       ↑
    │   eo_roc.py            ROC 曲線構築
    │
    ├── eo_provider.py       実データ取得
    │       ↑
    │   eo_era5.py           ERA5 取得
    │   eo_cache.py          キャッシュ + Provenance
    │
    └── eo_simulator.py      合成データ生成

eo_visualisation.py         可視化パネル群 (SpectralPanel / StatisticalPanel / FlagsPanel)
    ↑
eo_report.py                ReportBuilder (パネルを組み合わせてレポート出力)

examples/main_sample.py     MBSPThresholdEngine / CustomEngine
                            build_engine() / build_provider() を定義
```

---

## 依存ライブラリ

```bash
# 共通 (合成データ・ROC・可視化)
pip install numpy scipy matplotlib

# 実データ取得 (--provider real)
pip install openeo rasterio xarray cdsapi
```

| ライブラリ | 用途 |
|---|---|
| `numpy` | 数値計算 |
| `scipy` | 空間フィルタ・ロジスティック校正 |
| `matplotlib` | 可視化 |
| `openeo` | CDSE への接続・Sentinel-2 / S5P データ取得 |
| `rasterio` | GeoTIFF → ndarray 変換 |
| `xarray` | ERA5 NetCDF 読み込み |
| `cdsapi` | ERA5 取得 (Copernicus CDS API) |

Python バージョン: **3.8 以上**

---

## 認証の設定

Sentinel-2・ERA5・Sentinel-5P のすべてのデータ取得に [Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu/) のアカウントが必要です。

初回実行時に OIDC デバイスフロー認証が自動で起動します。

```python
from eo_provider import DataFetcher
fetcher = DataFetcher()
# 初回 fetch() 呼び出し時に URL が表示されてブラウザ認証が行われます
```

認証方式は環境変数 `OPENEO_AUTH_METHOD` で切り替えられます。

| 変数値 | 動作 |
|---|---|
| 未設定 (デフォルト) | リフレッシュトークン → デバイスフロー自動フォールバック |
| `client_credentials` | CI / サービスアカウント向け (`OPENEO_CLIENT_ID` / `OPENEO_CLIENT_SECRET` も必要) |
| `device` | 対話的デバイスフローを明示指定 |

---

## クイックスタート

### 合成データで動作確認する（認証不要）

```python
from eo_simulator  import PlumeSimulator
from eo_pipeline   import EOPipeline
from eo_engines    import NullInferenceEngine
from eo_report     import ReportBuilder

# サイトを定義する (TypedDict 形式)
registry = [
    {
        "id": "TM-01", "name": "Turkmenistan Compressor Station",
        "lat": 38.49, "lon": 54.19,
        "wind_speed": 4.0, "wind_deg": 120,
        "Q_true": 4000.0, "seed": 42, "category": "super-emitter",
    },
]

# パイプラインを構築する
pipeline = EOPipeline(
    provider = PlumeSimulator(mismatch=True, gp_noise=True),
    engine   = NullInferenceEngine(),
)

# 実行
results = pipeline.run_all(registry)
roc     = pipeline.build_roc(results, registry)

# レポートを生成する
ReportBuilder().build(results, roc=roc, save_path="report.png")
```

### 実データで実行する

```python
from datetime import datetime, timezone
from eo_provider import DataFetcher
from eo_pipeline import EOPipeline
from eo_engines  import NullInferenceEngine

pipeline = EOPipeline(
    provider = DataFetcher(band_set="swir_only"),  # or "full", "eucalyptus"
    engine   = NullInferenceEngine(),              # 独自エンジンに差し替えてください
)

results = pipeline.run_all(
    registry,
    dt = datetime(2023, 7, 15, 8, 0, tzinfo=timezone.utc),
)
```

---

## バンドセットの選択

Sentinel-2 の取得バンドを `band_set` パラメータで切り替えられます。

| band_set | 取得バンド | 用途 |
|---|---|---|
| `"swir_only"` | B11・B12 | メタン検出の最小構成（デフォルト） |
| `"full"` | B01〜B12 全13バンド | 土地被覆分類・雲マスク強化 |
| `"eucalyptus"` | 10バンド (B01〜B05, B08, B8A, B09, B11, B12) | Project Eucalyptus DL モデル入力 |

```python
fetcher = DataFetcher(band_set="full")
# bundle["bands"]["B08"] → NIR バンドが利用可能
```

---

## 推論エンジンの差し込み方

`InferenceEngine` (`eo_engines.py`) を継承して `infer()` を実装するだけで、任意の排出量推定アルゴリズムをパイプラインに接続できます。

```python
from eo_engines import InferenceEngine
from eo_types   import InferenceResult, QualityFlags
import numpy as np

class MyEngine(InferenceEngine):
    def infer(self, mbsp, wind_speed, wind_deg, **kwargs):
        # ここに独自アルゴリズムを実装する
        flags: QualityFlags = {
            "low_wind":          wind_speed < 2.0,
            "multi_modal_theta": False,
            "roi_unstable":      False,
            "template_dominant": False,
        }
        return {
            "q": 3500.0, "q_std": 200.0,
            "mllr": 120.0, "p_det": 0.92,
            "flags": flags,
        }

from eo_pipeline  import EOPipeline
from eo_simulator import PlumeSimulator

pipeline = EOPipeline(
    provider = PlumeSimulator(),
    engine   = MyEngine(),
)
```

### 組み込みエンジン

`examples/main_sample.py` に以下のエンジンが実装されています。

| エンジン | クラス | 説明 |
|---|---|---|
| `mbsp` | `MBSPThresholdEngine` | B11/B12 閾値法 (Varon et al. 2021)・追加インストール不要 |
| `eucalyptus` | `EucalyptusEngine` | Project Eucalyptus アダプタ・**非商用のみ** |
| `null` | `NullInferenceEngine` | ダミーエンジン・テスト用 (`eo_engines.py`) |
| `custom` | `CustomEngine` | カスタム実装サンプル |

```python
from examples.main_sample import MBSPThresholdEngine, build_engine

engine = MBSPThresholdEngine(z_thresh=3.0, q_scale_factor=30.0)
# または
engine = build_engine("mbsp")   # "mbsp" / "eucalyptus" / "null" / "custom"
```

---

## Mismatch Injection について

`PlumeSimulator` は推論エンジンが知らない「現実の揺らぎ」を合成データに注入する機能を持っています。

```python
from eo_simulator import PlumeSimulator, visualize_mismatch_comparison

# 比較可視化
visualize_mismatch_comparison(
    Q=2000.0, wind_speed=4.0, wind_deg=120.0, seed=42,
    save_path="mismatch_comparison.png",
)
```

---

## ObservationBundle 形式

`eo_provider.py` と `eo_simulator.py` は同じ形式でデータを返します。
`meta["backend"]` でデータ源を識別できます。

```python
{
    "bands": {
        "B11": ndarray,   # 必須
        "B12": ndarray,   # 必須
        "B08": ndarray,   # band_set="full" 時のみ
        # ...
    },
    "wind_speed":       float | None,
    "wind_deg":         float | None,
    "sza":              ndarray | None,  # 実データのみ
    "vza":              ndarray | None,  # 実データのみ
    "amf":              ndarray | None,  # 実データのみ
    "surface_pressure": float | None,    # 実データのみ (ERA5)
    "ch4_column":       ndarray | None,  # 実データのみ (Sentinel-5P)
    "plume_true":       ndarray | None,  # 合成データのみ
    "Q_true":           float | None,    # 合成データのみ
    "meta": {
        "backend":  "openeo" | "synthetic",
        "band_set": "swir_only" | "full" | "eucalyptus" | "custom",
        "sensors":  ["sentinel2", "era5", ...],
        "res_m":    20.0,
        # ...
    }
}
```

型定義の詳細は [`eo_types.py`](./eo_types.py) を参照してください。

---

## テストの実行

各モジュールは `if __name__ == "__main__":` ブロックに単体テストを内蔵しています。
外部 API 接続が不要なテストは常時実行できます。

```bash
# 型定義テスト（常時実行可）
python eo_types.py

# 合成データ生成テスト（常時実行可）
python eo_simulator.py

# パイプラインテスト（常時実行可）
python eo_pipeline.py

# ROC 曲線テスト（常時実行可）
python eo_roc.py

# キャッシュ・Provenance テスト（常時実行可）
python eo_cache.py

# 可視化テスト（常時実行可）
python eo_visualisation.py

# レポート出力テスト（常時実行可）
python eo_report.py

# ERA5 取得テスト（cdsapi 認証が必要）
python eo_era5.py

# 実データ取得テスト（openEO 認証が必要）
python eo_provider.py
```

---

## examples の実行

```bash
# MBSP 閾値法 + 合成データ（すぐ動く・認証不要）
python examples/main_sample.py --provider synthetic

# MBSP 閾値法 + 実データ（openEO 認証が必要）
python examples/main_sample.py --engine mbsp --provider real --date 2023-07-15

# カスタムエンジン + 合成データ
python examples/main_sample.py --engine custom --provider synthetic

```

| オプション | 有効値 | デフォルト |
|---|---|---|
| `--engine` | `mbsp` / `eucalyptus` / `null` / `custom` | `mbsp` |
| `--provider` | `real` / `synthetic` | `real` |
| `--band-set` | `swir_only` / `full` / `eucalyptus` | `swir_only` |
| `--date` | `YYYY-MM-DD` | `2023-07-15` |
| `--time` | `HH:MM` (UTC) | `08:00` |
| `--null-runs` | 整数 | `10` |
| `--no-save` | フラグ | レポート画像を保存しない |

---

## ライセンス

```
Apache License 2.0
```

---

## 関連情報

- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)
- [openEO Python Client](https://open-eo.github.io/openeo-python-client/)
- [Sentinel-2 バンド仕様](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2)
- [ERA5 データ説明](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
- [Sentinel-5P / TROPOMI](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-5p)
- [Project Eucalyptus (非商用)](https://github.com/Orbio-Earth/Project-Eucalyptus)
- [Varon et al. (2021) MBSP 法](https://doi.org/10.5194/amt-14-2771-2021)
