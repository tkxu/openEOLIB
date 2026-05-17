"""
ApacheLicense2.0
Copyright (c) 2026 tkxu
"""
#eo_types.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import numpy as np

# Required は Python 3.11+ で typing に追加された。
# 3.8〜3.10 では typing_extensions から取得する。
try:
    from typing import Required  # type: ignore[attr-defined]  # Python 3.11+
except ImportError:
    from typing_extensions import Required  # type: ignore[assignment]


# =============================================================================
# センサ種別・バンドセット定数
# =============================================================================

# データソース種別
# ObservationMeta["backend"] で使用する。
SensorSource = Literal[
    "openeo",       # Sentinel-2 / ERA5 / S5P (openEO / CDSE 経由)
    "synthetic",    # PlumeSimulator による合成データ
    "ground_insitu",# 地上インサイチュ計測 (固定局・移動局)
    "ground_gnss",  # GNSS 気圧・気温センサ
    "ground_lidar", # 地上ライダー (風速・乱流プロファイル)
    "openaq",       # OpenAQ 公開大気質データ
    "custom",       # 上記以外のカスタムソース
]

# Sentinel-2 バンドセット名
# eo_provider.py の Sentinel2Fetcher(band_set=...) に渡す。
BandSetName = Literal[
    "swir_only",    # B11・B12 のみ (デフォルト・最小構成)
    "full",         # B01〜B12 全13バンド
    "eucalyptus",   # Project Eucalyptus が要求する10バンド
                    # B01,B02,B03,B04,B05,B08,B08A,B09,B11,B12
    "custom",       # 呼び出し元が bands リストを直接指定
]

# バンドセット定数: BandSetName → バンド名リスト
# eo_provider.py の load_collection() に渡す bands 引数と対応する。
S2_BAND_SETS: Dict[str, List[str]] = {
    "swir_only":  ["B11", "B12"],
    "full":       ["B01", "B02", "B03", "B04", "B05", "B06",
                   "B07", "B08", "B8A", "B09", "B11", "B12"],
    "eucalyptus": ["B01", "B02", "B03", "B04", "B05",
                   "B08", "B8A", "B09", "B11", "B12"],
    "custom":     [],  # 呼び出し元が別途指定
}

# Sentinel-2 バンドの波長・解像度情報 (参照用)
S2_BAND_INFO: Dict[str, Dict] = {
    "B01": {"wavelength_nm": 443,  "res_m": 60,  "desc": "coastal aerosol"},
    "B02": {"wavelength_nm": 490,  "res_m": 10,  "desc": "blue"},
    "B03": {"wavelength_nm": 560,  "res_m": 10,  "desc": "green"},
    "B04": {"wavelength_nm": 665,  "res_m": 10,  "desc": "red"},
    "B05": {"wavelength_nm": 705,  "res_m": 20,  "desc": "red edge 1"},
    "B06": {"wavelength_nm": 740,  "res_m": 20,  "desc": "red edge 2"},
    "B07": {"wavelength_nm": 783,  "res_m": 20,  "desc": "red edge 3"},
    "B08": {"wavelength_nm": 842,  "res_m": 10,  "desc": "NIR broad"},
    "B8A": {"wavelength_nm": 865,  "res_m": 20,  "desc": "NIR narrow"},
    "B09": {"wavelength_nm": 940,  "res_m": 60,  "desc": "water vapour"},
    "B11": {"wavelength_nm": 1610, "res_m": 20,  "desc": "SWIR 1 (CH4 sensitive)"},
    "B12": {"wavelength_nm": 2190, "res_m": 20,  "desc": "SWIR 2 (CH4 reference)"},
}


# =============================================================================
# Layer 0 — サイトレジストリ
# =============================================================================

class SiteEntry(TypedDict, total=False):
    """
    サイトレジストリの1エントリ。

    SITE_REGISTRY リストの各要素に対応する。
    earth_obs_provider.py・plume_simulator.py・protocol.py が共通で参照する。
    """
    id:         Required[str]    # サイト識別子 (例: "TM-01")
    name:       Required[str]    # サイト正式名称
    lat:        Required[float]  # 中心緯度 [degrees]
    lon:        Required[float]  # 中心経度 [degrees]
    wind_speed: Required[float]  # 公称風速 [m/s]
    wind_deg:   Required[float]  # 公称気象風向 [degrees, 北基準時計回り FROM方向]
    Q_true:     Required[float]  # 真の排出量 [kg/h]  (検証用)
    seed:       Required[int]    # 乱数シード  (合成データの再現性用)
    category:   Required[str]    # カテゴリ ("super-emitter" / "mid-range" / "near-limit")
    # MBSP 計算失敗時のフォールバックシーンサイズ [pixels]
    # 未指定の場合は EOPipeline 内でデフォルト値 (100) を使用する。
    grid_h:     int              # シーン高さ [px]  (オプション)
    grid_w:     int              # シーン幅  [px]  (オプション)


# =============================================================================
# Layer 2 — 観測データ (ObservationBundle)
# =============================================================================

class BandData(TypedDict, total=False):
    """
    Sentinel-2 バンドデータ。

    全13バンドをオプションフィールドとして定義する。
    B11・B12 はメタン検出の基本バンドであり Required[] で必須指定する。
    それ以外は band_set に応じて存在する場合のみ格納される。

    後方互換性:
        既存の {"B11": arr, "B12": arr} 形式はそのまま動作する。
        B11/B12 のみを必要とするエンジンは他のバンドを無視すればよい。

    Required[] の効果:
        total=False のため全フィールドはデフォルトでオプションだが、
        B11/B12 に Required[] を付与することで mypy / pyright が
        これらのキーの欠落を静的エラーとして検出できる。
        ランタイムの型強制は行われない点に注意。

    解像度の注意:
        B01/B09 は 60m、B02/B03/B04/B08 は 10m、他は 20m が原解像度。
        eo_provider.py は resample_spatial で 20m に統一して返す。

    shape: すべて (H, W)  dtype: float32  値域: [0, 1] 反射率
    """
    # B11/B12 に Required[] を付与し、静的型チェッカーが欠落を検出できるようにする。
    #   変更前: B11: np.ndarray  (total=False のため実質オプション扱い)
    #   変更後: B11: Required[np.ndarray]  (mypy / pyright で必須として検査される)
    #   後方互換: {"B11": arr, "B12": arr} 形式の既存コードはそのまま動作する。

    # --- SWIR バンド (メタン検出の基本・必須) ---
    B11: Required[np.ndarray]   # 1610 nm  SWIR 1  CH4 に感度あり  20m
    B12: Required[np.ndarray]   # 2190 nm  SWIR 2  CH4 の参照バンド  20m

    # --- 可視・近赤外バンド (雲マスク・土地被覆・DL モデル入力) ---
    B01: np.ndarray   #  443 nm  coastal aerosol  60m → 20m リサンプル
    B02: np.ndarray   #  490 nm  blue   10m → 20m リサンプル
    B03: np.ndarray   #  560 nm  green  10m → 20m リサンプル
    B04: np.ndarray   #  665 nm  red    10m → 20m リサンプル
    B05: np.ndarray   #  705 nm  red edge 1  20m
    B06: np.ndarray   #  740 nm  red edge 2  20m
    B07: np.ndarray   #  783 nm  red edge 3  20m
    B08: np.ndarray   #  842 nm  NIR broad   10m → 20m リサンプル
    B8A: np.ndarray   #  865 nm  NIR narrow  20m
    B09: np.ndarray   #  940 nm  water vapour  60m → 20m リサンプル


class MultiTemporalBands(TypedDict):
    """
    時系列バンドデータ。

    Project Eucalyptus・CH4Net 等の時系列入力モデルが要求する形式。
    t（対象時刻）と t_ref（参照時刻）の2時点を保持する。

    Fields
    ------
    t     : 対象時刻のバンドデータ (メタンプルームが存在する可能性がある時刻)
    t_ref : 参照時刻のバンドデータ (クリーンなシーン)
    dt_days: t と t_ref の間隔 [days]
    """
    t:       BandData    # 対象時刻 (t)
    t_ref:   BandData    # 参照時刻 (t-1 または別日)
    dt_days: float       # 2時点間の日数差


class ObservationMeta(TypedDict, total=False):
    """
    観測メタデータ。

    ObservationBundle["meta"] に格納される。

    backend フィールドでデータ源を識別する (SensorSource 型):
        "openeo"      : eo_provider.py 経由の実データ
        "synthetic"   : eo_simulator.py 経由の合成データ
        "ground_*"    : 各種地上センサ
        "openaq"      : OpenAQ 公開データ
        "custom"      : その他

    band_set フィールドで取得済みバンドセットを識別する (BandSetName 型):
        "swir_only"   : B11・B12 のみ (デフォルト)
        "full"        : 全13バンド
        "eucalyptus"  : Project Eucalyptus 用10バンド
        "custom"      : 個別指定

    sensors フィールドで利用可能なデータソースをリストする:
        例: ["sentinel2", "era5", "sentinel5p", "ground_insitu"]

    Optional[float] と total=False の使い分け:
        キーが存在することは保証されるがゼロ・None になりうるフィールドは
        Required[Optional[float]] で明示する。
        キー自体が存在しない場合もあるフィールドは Optional[float] のまま。
    """
    # --- 必須フィールド ---
    lat:         Required[float]        # 中心緯度 [degrees]
    lon:         Required[float]        # 中心経度 [degrees]
    backend:     Required[SensorSource] # データソース種別 (SensorSource リテラル)

    # --- 実データのみ (キー自体が存在しない場合もある) ---
    datetime:    str              # 観測日時 ISO8601
    bbox_deg:    float            # 取得範囲の半径 [degrees]
    cloud_pct:   Required[Optional[float]]  # 雲量 [%] (SCL 使用時は None)
    s2_scene_id: Required[Optional[str]]    # Sentinel-2 シーン識別子
    sza_mean:    Required[Optional[float]]  # 太陽天頂角の空間平均 [degrees]
    vza_mean:    Required[Optional[float]]  # 衛星天頂角の空間平均 [degrees]

    # --- 拡張フィールド ---
    band_set:    BandSetName  # 取得済みバンドセット (BandSetName リテラル)
    sensors:     List[str]    # 利用可能データソースのリスト
    res_m:       float        # 統一解像度 [m/px] (デフォルト: 20.0)
    provenance_key:  Required[Optional[str]]  # DatasetProvenance.fingerprint() の返値
    era5_backend:    Required[Optional[str]]  # ERA5 バックエンド識別子 (例: "cdsapi_stable")



class ObservationBundle(TypedDict, total=False):
    """
    観測データの統合コンテナ。

    eo_provider.py (DataFetcher.fetch) と
    eo_simulator.py (PlumeSimulator.generate) が同じ形式で返す。
    meta["backend"] でデータ源を識別する。

    拡張フィールド一覧:
        bands            Sentinel-2 バンド (B11/B12 は論理的必須)
        multi_temporal   時系列バンドデータ (DL モデル用)

    後方互換性:
        既存コードは bands["B11"] / bands["B12"] のみを参照すれば動作する。
        新フィールドは存在しない場合 None として扱う。
    """
    # --- 共通フィールド ---
    bands:             Optional[BandData]           # Sentinel-2 バンド
    wind_speed:        Optional[float]              # 風速 [m/s]
    wind_deg:          Optional[float]              # 気象風向 [degrees]
    meta:              ObservationMeta              # メタデータ

    # --- 時系列拡張 (DL モデル用) ---
    multi_temporal:    Optional[MultiTemporalBands] # 時系列バンドデータ
                                                    # Project Eucalyptus 等が使用

    # --- 実データ専用 (openEO 経由) ---
    sza:               Optional[np.ndarray]  # 太陽天頂角 [degrees]  shape=(H,W)
    vza:               Optional[np.ndarray]  # 衛星天頂角 [degrees]  shape=(H,W)
    amf:               Optional[np.ndarray]  # Air Mass Factor       shape=(H,W)
    surface_pressure:  Optional[float]       # 地表気圧 [Pa]  ERA5 由来
    ch4_column:        Optional[np.ndarray]  # CH4 カラム濃度 [ppb]  shape=(H,W)
                                             # Sentinel-5P / TROPOMI 由来


    # --- 合成データ専用 (eo_simulator 経由) ---
    plume_true:        Optional[np.ndarray]  # 真のプルーム濃度場 [g/m²]  shape=(H,W)
    Q_true:            Optional[float]       # 真の排出量 [kg/h]


class SyntheticMeta(ObservationMeta, total=False):
    """
    合成データ専用メタデータ。ObservationMeta を拡張する。

    eo_simulator.py (PlumeSimulator) が生成するバンドルのメタデータ。
    Mismatch Injection の実際の注入値を記録する。
    backend は常に "synthetic"。
    """
    wind_speed_actual:  float          # mismatch 後の実際の風速 [m/s]
    wind_deg_actual:    float          # 実際の風向 [degrees]
    pg_a_actual:        float          # 実際の PG 係数 A
    mismatch_enabled:   bool           # Mismatch Injection の有効/無効
    gp_noise_enabled:   bool           # GP 空間相関ノイズの有効/無効
    seed:               Optional[int]  # 乱数シード
    shape:              Tuple[int, int]# シーンサイズ (rows, cols)
    res_m:              float          # ピクセル解像度 [m/px]


class SyntheticBundle(ObservationBundle):
    """
    合成データ専用の ObservationBundle。

    eo_simulator.py が生成する。ObservationBundle との互換性を保ちつつ
    plume_true・Q_true・SyntheticMeta を持つことを型で明示する。

    Required[] を使用して必須フィールドを明示する。
    ObservationBundle は total=False であるため、
    サブクラスで必須にしたいフィールドは Required[] で個別に指定する。
    Python 3.8〜3.10 では typing_extensions が必要。
    """
    plume_true: Required[np.ndarray]    # 真のプルーム濃度場 [g/m²]  shape=(H,W)  (必須)
    Q_true:     Required[float]         # 真の排出量 [kg/h]            (必須)
    meta:       Required[SyntheticMeta]


# =============================================================================
# Layer 4 — 推論・スコアリング結果
# =============================================================================

class QualityFlags(TypedDict):
    """
    推論エンジン (OperationalInferenceV14_1) が返す品質フラグ。

    フラグが True の場合、推定結果の信頼性が低下している可能性がある。
    """
    low_wind:          bool   # 風速 < 2.0 m/s (拡散モデルの精度低下)
    multi_modal_theta: bool   # θスイープが多峰的 (風向の一意性低下)
    roi_unstable:      bool   # ROI バリアント間の MLLR 標準偏差 > 10
    template_dominant: bool   # H0 構造テンプレートが優勢 (地表ノイズの疑い)


class InferenceResult(TypedDict):
    """
    SLVEA 逆推定エンジンの出力。

    OperationalInferenceV14_1.infer() が返す。
    コアアルゴリズムの出力であるため、このファイルでは型定義のみを行い
    実装は非公開リポジトリに置く。
    """
    q:     float          # 推定排出量 [kg/h]
    q_std: float          # Q の不確実性 (全分散の法則による)
    mllr:  float          # Marginal Log-Likelihood Ratio
    p_det: float          # 検出確率 [0, 1]  (MLLR から校正)
    flags: QualityFlags   # 品質フラグ


class PhysResult(TypedDict):
    """
    物理整合性検証 (PhysicalValidator) の出力。

    V42 §10 の WindAlign / DownwindSNR に対応する。
    """
    wind_speed:    float   # 風速 [m/s]
    wind_align:    float   # PCA 主軸と風向の整合度 [0, 1]
    downwind_snr:  float   # ダウンウィンド方向の SNR [0, 1]
    low_wind_mode: bool    # 低風速モード (wind_speed < u10_min)


class ScoringResult(TypedDict):
    """
    DetScoreEngine の出力。

    V42 §11 の Tier 判定・det_score に対応する。
    """
    det_score: float   # 総合検出スコア [0, 1]
    tier:      str     # Tier 判定 ("Tier-A" / "Tier-B" / "Tier-C" / "None")
    z_norm:    float   # Z スコアの非線形正規化値 [0, 1]
    mllr_norm: float   # MLLR の正規化値 [0, 1]


# =============================================================================
# Layer 5 — パイプライン実行結果
# =============================================================================

class SiteResult(TypedDict, total=False):
    """
    ValidationProtocol.run_site() の出力。

    1サイト分のパイプライン全結果を格納する。
    visualisation.py はこの型だけを知っていれば描画できる。

    total=False: 候補が検出されない場合は best_candidate 等が None になる。
    """
    # --- 入力 ---
    site:           SiteEntry             # サイトエントリ

    # --- 観測データ ---
    mbsp:           np.ndarray            # MBSP フィールド shape=(H,W)
    llr:            np.ndarray            # LLR マップ shape=(H,W)
    mask_v12:       np.ndarray            # V12 LLR マスク shape=(H,W) bool
    plume_true:     np.ndarray            # 真のプルーム濃度場 shape=(H,W)
    wvec:           np.ndarray            # 風向単位ベクトル shape=(2,)

    # --- 検出結果 ---
    post:           float                 # 検出事後確率 [0, 1]
    candidates_v42: List[CandidateDict]   # 形状候補リスト
    best_candidate: Optional[CandidateDict]  # 代表候補 (最高 z_mean)

    # --- 逆推定・スコアリング ---
    inv:            Optional[InferenceResult]   # SLVEA 逆推定結果
    phys_result:    Optional[PhysResult]        # 物理整合性検証結果
    scoring:        Optional[ScoringResult]     # Tier 判定結果
    flags:          QualityFlags                # 推論品質フラグ

    # --- 検証用 ---
    gt_match:       bool                  # Ground truth 一致フラグ (検証用)
    meta:           Optional[Dict[str, Any]]  # パイプライン実行時の追加メタデータ


class RocData(TypedDict):
    """
    ROC 曲線データ。

    ValidationProtocol.build_roc_data() が返す。
    """
    fpr:             np.ndarray    # 偽陽性率配列 shape=(N,)
    tpr:             np.ndarray    # 真陽性率配列 shape=(N,)
    auc:             float         # AUC 値
    positive_mllrs:  List[float]   # プルームありの MLLR スコアリスト
    null_mllrs:      List[float]   # ヌル分布の MLLR スコアリスト


# =============================================================================
# 型エイリアス (可読性のための別名)
# =============================================================================

# モジュールをまたぐ「バンドデータ単体」の型エイリアス
BandDict = BandData

# SiteResult のリスト (eo_pipeline.py の run_all の返り値)
SiteResultList = List[SiteResult]

# バンドセット定数へのショートカット (from eo_types import BAND_SETS)
BAND_SETS = S2_BAND_SETS


# =============================================================================
# 風向変換ユーティリティ
#
# 複数モジュール (eo_pipeline / eo_simulator) で同一ロジックが必要なため、
# eo_types.py に一元化して重複定義を排除する。
# 依存モジュール側では以下のように import すること:
#   from eo_types import wind_deg_to_math, wind_deg_to_vec
# =============================================================================

def wind_deg_to_math(deg: float) -> float:
    """
    気象風向 (北基準時計回り) を数学角 (東基準反時計回り) に変換する。

    全コンポーネントがこの関数を経由することで変換の一貫性を保証する。
    """
    return (270.0 - deg) % 360.0


def wind_deg_to_vec(deg: float) -> "np.ndarray":
    """気象風向を単位ベクトル [vx=East, vy=North] に変換する。"""
    rad = np.radians(wind_deg_to_math(deg))
    return np.array([np.cos(rad), np.sin(rad)])


# =============================================================================
# テストコード (外部API不要)
# =============================================================================

def test_sensor_source_and_band_sets() -> None:
    """SensorSource・BandSetName・S2_BAND_SETS の構造テスト。"""
    print("\n" + "="*58)
    print("  TEST-1: SensorSource / BandSetName / S2_BAND_SETS")
    print("="*58)

    # S2_BAND_SETS の構造
    assert "swir_only"  in S2_BAND_SETS
    assert "full"       in S2_BAND_SETS
    assert "eucalyptus" in S2_BAND_SETS
    assert "B11" in S2_BAND_SETS["swir_only"]
    assert "B12" in S2_BAND_SETS["swir_only"]
    assert len(S2_BAND_SETS["full"]) == 12
    assert len(S2_BAND_SETS["eucalyptus"]) == 10

    # BAND_SETS エイリアス
    assert BAND_SETS is S2_BAND_SETS

    # S2_BAND_INFO の波長確認
    assert S2_BAND_INFO["B11"]["wavelength_nm"] == 1610
    assert S2_BAND_INFO["B12"]["wavelength_nm"] == 2190
    assert S2_BAND_INFO["B11"]["res_m"]         == 20

    print(f"  swir_only  バンド数: {len(S2_BAND_SETS['swir_only'])}")
    print(f"  full       バンド数: {len(S2_BAND_SETS['full'])}")
    print(f"  eucalyptus バンド数: {len(S2_BAND_SETS['eucalyptus'])}")
    print("  → PASS")


def test_band_data_full() -> None:
    """BandData の全バンド対応テスト。"""
    print("\n" + "="*58)
    print("  TEST-2: BandData 全バンド対応")
    print("="*58)

    H, W = 50, 50

    # swir_only: 後方互換 (B11/B12 が Required なので最低限これで OK)
    bands_swir: BandData = {
        "B11": np.zeros((H, W), dtype=np.float32),
        "B12": np.zeros((H, W), dtype=np.float32),
    }
    assert "B11" in bands_swir
    assert "B12" in bands_swir
    print("  swir_only (後方互換) → PASS")

    # full: 全13バンド
    bands_full: BandData = {b: np.zeros((H, W), dtype=np.float32)
                             for b in S2_BAND_SETS["full"]}
    assert len([k for k in bands_full if k.startswith("B")]) == 12
    print(f"  full (12バンド): {list(bands_full.keys())} → PASS")

    # eucalyptus: 10バンド
    bands_euca: BandData = {b: np.zeros((H, W), dtype=np.float32)
                             for b in S2_BAND_SETS["eucalyptus"]}
    assert len(bands_euca) == 10
    print(f"  eucalyptus (10バンド) → PASS")


def test_multi_temporal_bands() -> None:
    """MultiTemporalBands のテスト。"""
    print("\n" + "="*58)
    print("  TEST-3: MultiTemporalBands (時系列)")
    print("="*58)

    H, W = 50, 50
    base_band: BandData = {
        "B11": np.zeros((H, W), dtype=np.float32),
        "B12": np.zeros((H, W), dtype=np.float32),
    }
    mt: MultiTemporalBands = {
        "t":       base_band,
        "t_ref":   base_band,
        "dt_days": 10.0,
    }
    assert "t"     in mt
    assert "t_ref" in mt
    assert mt["dt_days"] == 10.0
    print(f"  t / t_ref / dt_days={mt['dt_days']} days → PASS")


def test_ground_sensor_data() -> None:
    """GroundSensorData・GroundSensorMeta のテスト。"""
    print("\n" + "="*58)
    print("  TEST-4: GroundSensorData / GroundSensorMeta")
    print("="*58)

    # 地上 CH4 センサ (インサイチュ)
    gs: GroundSensorData = {
        "ch4_ppm":       2.15,
        "wind_speed":    3.8,
        "wind_deg":      125.0,
        "temperature_k": 298.0,
        "pressure_pa":   101325.0,
        "timestamp":     "2023-07-15T08:05:00Z",
        "averaging_sec": 60,
    }
    assert gs["ch4_ppm"]   == 2.15
    assert gs["wind_speed"] == 3.8
    print(f"  ch4_ppm={gs['ch4_ppm']} ppm  wind={gs['wind_speed']} m/s → PASS")

    # ライダー風速プロファイル
    gs_lidar: GroundSensorData = {
        "wind_profile": np.array([3.5, 4.0, 4.8, 5.2]),
        "alt_levels_m": np.array([10.0, 50.0, 100.0, 200.0]),
        "timestamp":    "2023-07-15T08:00:00Z",
    }
    assert gs_lidar["wind_profile"].shape == (4,)
    print(f"  lidar wind_profile shape={gs_lidar['wind_profile'].shape} → PASS")

    # メタデータ
    gm: GroundSensorMeta = {
        "sensor_type":      "ground_insitu",
        "sensor_id":        "SITE-TM-01-GS01",
        "sensor_lat":       38.490,
        "sensor_lon":       54.192,
        "sensor_alt_m":     2.0,
        "dist_to_site_m":   250.0,
        "data_source_url":  "https://example.com/api/sensors/TM-01",
        "available_fields": ["ch4_ppm", "wind_speed", "wind_deg",
                             "temperature_k", "pressure_pa"],
        "notes":            "Picarro G2301 CRDS analyzer",
    }
    assert gm["sensor_type"]    == "ground_insitu"
    assert gm["dist_to_site_m"] == 250.0
    print(f"  sensor_id={gm['sensor_id']}  dist={gm['dist_to_site_m']} m → PASS")


def test_observation_bundle_extended() -> None:
    """拡張 ObservationBundle のテスト。"""
    print("\n" + "="*58)
    print("  TEST-5: ObservationBundle (全フィールド)")
    print("="*58)

    H, W = 100, 100

    meta: ObservationMeta = {
        "lat":         38.49,
        "lon":         54.19,
        "datetime":    "2023-07-15T08:00:00Z",
        "bbox_deg":    0.18,
        "cloud_pct":   None,
        "s2_scene_id": "S2_openEO_20230715",
        "sza_mean":    32.5,
        "vza_mean":     4.1,
        "backend":     "openeo",
        "band_set":    "full",
        "sensors":     ["sentinel2", "era5", "sentinel5p", "ground_insitu"],
        "res_m":       20.0,
        "provenance_key": None,
        "era5_backend":   None,
    }

    # 全バンド
    bands_full: BandData = {b: np.zeros((H, W), dtype=np.float32)
                             for b in S2_BAND_SETS["full"]}

    # 時系列
    base_band: BandData = {"B11": np.zeros((H, W)), "B12": np.zeros((H, W))}
    mt: MultiTemporalBands = {"t": base_band, "t_ref": base_band, "dt_days": 10.0}

    bundle: ObservationBundle = {
        "bands":              bands_full,
        "wind_speed":         4.0,
        "wind_deg":           120.0,
        "meta":               meta,
        "multi_temporal":     mt,
        "sza":                np.zeros((H, W), dtype=np.float32),
        "vza":                np.zeros((H, W), dtype=np.float32),
        "amf":                np.full((H, W), 2.0, dtype=np.float32),
        "surface_pressure":   101325.0,
        "ch4_column":         np.zeros((H, W), dtype=np.float32),
        "plume_true":         None,
        "Q_true":             None,
    }

    assert bundle["meta"]["backend"]   == "openeo"
    assert bundle["meta"]["band_set"]  == "full"
    assert "sentinel2" in bundle["meta"]["sensors"]
    assert "ground_insitu" in bundle["meta"]["sensors"]
    assert bundle["multi_temporal"]["dt_days"] == 10.0
    assert len(bundle["bands"]) == 12

    print(f"  backend={bundle['meta']['backend']}")
    print(f"  band_set={bundle['meta']['band_set']}  "
          f"bands={len(bundle['bands'])}バンド")
    print(f"  sensors={bundle['meta']['sensors']}")
    print(f"  multi_temporal dt={bundle['multi_temporal']['dt_days']} days")
    print("  → PASS")


def test_typeddict_structure() -> None:
    """既存の TypedDict 構造テスト（後方互換性確認）。"""
    print("\n" + "="*58)
    print("  TEST-6: 後方互換性 (既存コードとの互換)")
    print("="*58)

    # --- SiteEntry ---
    site: SiteEntry = {
        "id": "TM-01", "name": "Turkmenistan Compressor Station",
        "lat": 38.49, "lon": 54.19,
        "wind_speed": 4.0, "wind_deg": 120.0,
        "Q_true": 4000.0, "seed": 42, "category": "super-emitter",
    }
    assert site["Q_true"] == 4000.0
    print("  SiteEntry → PASS")

    # --- SiteEntry with grid_h / grid_w ---
    site_with_grid: SiteEntry = {**site, "grid_h": 128, "grid_w": 128}
    assert site_with_grid["grid_h"] == 128
    print("  SiteEntry (grid_h/grid_w) → PASS")

    # --- QualityFlags ---
    flags: QualityFlags = {
        "low_wind": False, "multi_modal_theta": True,
        "roi_unstable": False, "template_dominant": False,
    }
    assert flags["multi_modal_theta"] is True
    print("  QualityFlags → PASS")

    # --- InferenceResult ---
    inv: InferenceResult = {
        "q": 3800.0, "q_std": 120.0, "mllr": 145.3,
        "p_det": 0.94, "flags": flags,
    }
    assert inv["q"] == 3800.0
    print("  InferenceResult → PASS")

    # --- RocData ---
    roc: RocData = {
        "fpr": np.linspace(0, 1, 10), "tpr": np.linspace(0, 1, 10),
        "auc": 0.87, "positive_mllrs": [145.3], "null_mllrs": [-12.1],
    }
    assert roc["auc"] == 0.87
    print("  RocData → PASS")

    # --- BandDict エイリアス (後方互換) ---
    bd: BandDict = {"B11": np.zeros((10, 10)), "B12": np.zeros((10, 10))}
    assert "B11" in bd
    print("  BandDict エイリアス → PASS")

    print("\n  全テスト PASS")


def print_type_summary() -> None:
    """型定義の一覧をコンソールに表示する。"""
    types_info = [
        ("─── センサ種別・定数 ───", ""),
        ("SensorSource",       "データソース種別リテラル (openeo / synthetic / ground_* 等)"),
        ("BandSetName",        "バンドセット名リテラル (swir_only / full / eucalyptus)"),
        ("S2_BAND_SETS",       "バンドセット定数辞書  {name: [band, ...]}"),
        ("S2_BAND_INFO",       "バンド波長・解像度情報辞書"),
        ("─── Layer 0 ───", ""),
        ("SiteEntry",          "サイトレジストリの1エントリ (grid_h/grid_w 追加)"),
        ("─── Layer 2: 観測データ ───", ""),
        ("BandData",           "Sentinel-2 全13バンド (B11/B12はRequired必須)"),
        ("MultiTemporalBands", "時系列バンドデータ t / t_ref (DL モデル用)"),
        ("ObservationMeta",    "観測メタデータ (backend / band_set / sensors)"),
        ("ObservationBundle",  "統合データコンテナ (衛星+ERA5+S5P+地上センサ)"),
        ("SyntheticMeta",      "合成データ専用メタ (ObservationMeta を拡張)"),
        ("SyntheticBundle",    "合成データ専用バンドル (plume_true / Q_true 必須)"),
        ("─── Layer 4: 推論結果 ───", ""),
        ("QualityFlags",       "推論品質フラグ (low_wind / roi_unstable 等)"),
        ("InferenceResult",    "逆推定結果 (q / q_std / mllr / p_det / flags)"),
        ("PhysResult",         "物理整合性検証 (wind_align / downwind_snr)"),
        ("ScoringResult",      "Tier 判定結果 (det_score / tier / z_norm)"),
        ("─── Layer 5: パイプライン ───", ""),
        ("SiteResult",         "run_site() の全出力 (1サイト分)"),
        ("RocData",            "ROC 曲線データ (fpr / tpr / auc)"),
    ]
    print("\n" + "="*65)
    print("  eo_types.py — 型定義一覧")
    print("="*65)
    for name, desc in types_info:
        if desc == "":
            print(f"\n  {name}")
        else:
            print(f"  {name:<24} {desc}")
    print("="*65)


# =============================================================================
# エントリポイント
# =============================================================================

if __name__ == "__main__":

    print_type_summary()
    test_sensor_source_and_band_sets()
    test_band_data_full()
    test_multi_temporal_bands()
    test_observation_bundle_extended()
    test_typeddict_structure()

    print("\n" + "="*58)
    print("  全テスト PASS")
    print("  静的型チェック: mypy eo_types.py")
    print("  ※ TypedDict は実行時の型強制を行いません。")
    print("    mypy / pyright で静的解析してください。")
    print("="*58)
