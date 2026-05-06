# openEOLIB

Sentinel-2・ERA5・Sentinel-5P の衛星観測データを取得し、ガウスプルーム合成データによる検証パイプラインを構築するためのPythonライブラリです。

推論エンジンを外部から差し込む設計になっており、独自の排出量推定アルゴリズムと組み合わせて使用できます。

---

## 特徴

- **実データ取得**：openEO (CDSE) 経由で Sentinel-2・ERA5・Sentinel-5P を一括取得
- **全バンド対応**：B11/B12 のみ (デフォルト) から全13バンド・Eucalyptus セットまで切り替え可能
- **合成データ生成**：ガウスプルーム拡散モデルによる物理的に妥当な合成観測データ
- **Mismatch Injection**：風速揺らぎ・拡散係数個体差・GP 空間相関ノイズを注入してロバスト性を検証
- **差し込み可能な推論エンジン**：`InferenceEngine` を実装すれば任意のアルゴリズムを接続可能
- **可視化**：プルーム空間場・MBSP・MLLR 分布・ROC 曲線・品質フラグを統合レポートとして出力

---

## ファイル構成

```
/
├── eo_types.py           モジュール間インターフェースの型定義 (TypedDict)
├── eo_provider.py        実データ取得層 (openEO / CDSE)
├── eo_simulator.py       ガウスプルーム合成データ生成器
├── eo_pipeline.py        観測データ処理パイプライン
├── eo_visualisation.py   可視化モジュール
└── examples/
    └── main_sample.py    使い方サンプル
```

推論エンジンの実装（排出量逆推定アルゴリズム）は本リポジトリには含まれていません。
`InferenceEngine` を継承したクラスを独自に実装して差し込んでください（後述）。

---

## 依存ライブラリ

```bash
pip install openeo rasterio numpy scipy matplotlib
```

| ライブラリ | 用途 |
|---|---|
| `openeo` | CDSE への接続・データ取得 |
| `rasterio` | GeoTIFF → ndarray 変換 |
| `numpy` | 数値計算 |
| `scipy` | 空間フィルタ・最適化 |
| `matplotlib` | 可視化 |

Python バージョン: **3.8 以上**

---

## 認証の設定

Sentinel-2・ERA5・Sentinel-5P のすべてのデータ取得に[Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu/) のアカウントが必要です。

初回実行時に OIDC デバイスフロー認証が自動で起動します。

```python
from eo_provider import DataFetcher
fetcher = DataFetcher()
# 初回 fetch() 呼び出し時に URL が表示されてブラウザ認証が行われます
```


---

## クイックスタート

### 合成データで動作確認する（認証不要）

```python
from eo_simulator    import PlumeSimulator
from eo_pipeline     import EOPipeline, NullInferenceEngine
from eo_types        import SiteEntry
from eo_visualisation import ReportBuilder

# サイトを定義する
registry = [
    SiteEntry(
        id="TM-01", name="Turkmenistan Compressor Station",
        lat=38.49, lon=54.19,
        wind_speed=4.0, wind_deg=120,
        Q_true=4000.0, seed=42, category="super-emitter",
    ),
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
from eo_pipeline  import EOPipeline, NullInferenceEngine

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
| `"eucalyptus"` | 10バンド | Project Eucalyptus DL モデル入力 |

```python
fetcher = DataFetcher(band_set="full")
# bundle["bands"]["B08"] → NIR バンドが利用可能
```

---

## 推論エンジンの差し込み方

`InferenceEngine` を継承して `infer()` を実装するだけで、
任意の排出量推定アルゴリズムをパイプラインに接続できます。

```python
from eo_pipeline import InferenceEngine
from eo_types    import InferenceResult, QualityFlags

class MyEngine(InferenceEngine):
    def infer(self, mbsp, wind_speed, wind_deg, **kwargs):
        # ここに独自アルゴリズムを実装する
        flags: QualityFlags = {
            "low_wind": wind_speed < 2.0,
            "multi_modal_theta": False,
            "roi_unstable": False,
            "template_dominant": False,
        }
        return InferenceResult(
            q=3500.0, q_std=200.0,
            mllr=120.0, p_det=0.92,
            flags=flags,
        )

from eo_pipeline  import EOPipeline
from eo_simulator import PlumeSimulator

pipeline = EOPipeline(
    provider = PlumeSimulator(),
    engine   = MyEngine(),
)
```

---


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
        ...
    },
    "wind_speed":       float | None,
    "wind_deg":         float | None,
    "sza":              ndarray | None,  # 実データのみ
    "surface_pressure": float | None,    # 実データのみ
    "ch4_column":       ndarray | None,  # 実データのみ
    "plume_true":       ndarray | None,  # 合成データのみ
    "Q_true":           float | None,    # 合成データのみ
    "meta": {
        "backend":  "openeo" | "synthetic",
        "band_set": "swir_only" | "full" | "eucalyptus",
        "sensors":  ["sentinel2", "era5", ...],
        "res_m":    20.0,
        ...
    }
}
```

型定義の詳細は [`eo_types.py`](./eo_types.py) を参照してください。

---

## モジュール依存関係

```
eo_types.py            ← 唯一の共有インターフェース定義
    ↑ import
    ├── eo_provider.py      実データ取得
    ├── eo_simulator.py     合成データ生成
    ├── eo_pipeline.py      パイプライン
    └── eo_visualisation.py 可視化

eo_pipeline.py
    ├── import: eo_types.py
    ├── import: eo_simulator.py  (ヌル試行生成)
    └── 外部注入: InferenceEngine
```

---

## テストの実行

```bash
# 型定義テスト（常時実行可）
python eo_types.py

# 合成データ生成テスト（常時実行可）
python eo_simulator.py

# パイプラインテスト（常時実行可）
python eo_pipeline.py

# 可視化テスト（常時実行可）
python eo_visualisation.py

# 実データ取得テスト（openEO 認証が必要）
python eo_provider.py
```

---

## examples の実行

```bash
# MBSP 閾値法 + 合成データ（すぐ動く・認証不要）
python examples/main_sample.py

# MBSP 閾値法 + 実データ
python examples/main_sample.py --engine mbsp --provider real --date 2023-07-15

# Project Eucalyptus（要ライセンス確認）
git clone https://github.com/Orbio-Earth/Project-Eucalyptus
pip install -r Project-Eucalyptus/requirements.txt
python examples/main_sample.py --engine eucalyptus
```

---

## ライセンス
```
   Copyright 2026 tkxu

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
