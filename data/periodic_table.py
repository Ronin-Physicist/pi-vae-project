import torch
from typing import Dict, Optional
import mendeleev

def _get_normalized_features(atomic_num: int) -> Optional[torch.Tensor]:
    """
    単一の原子番号についてmendeleevからデータを取得し、正規化計算を行う。
    必須の物理量が1つでも欠損している場合は None を返す純粋関数。
    """
    try:
        elem = mendeleev.element(atomic_num)
        
        group = elem.group_id
        period = elem.period
        
        # 価電子数は関数として取得
        nvalence = elem.nvalence()
        
        en_pauling = elem.en_pauling
        
        # 共有結合半径 (Pyykkoスケールが最も一般的に埋まっている)
        covalent_radius = elem.covalent_radius_pyykko 
        
        # 欠損値 (None) の厳格なチェック (No Padding)
        if None in (group, period, nvalence, en_pauling, covalent_radius):
            return None
            
        # Min-Max 正規化 (ハードコーディングされた数学的契約)
        norm_group = (float(group) - 1.0) / 17.0
        norm_period = (float(period) - 1.0) / 6.0
        norm_nvalence = (float(nvalence) - 1.0) / 7.0
        norm_en = (float(en_pauling) - 0.7) / 3.3
        norm_radius = (float(covalent_radius) - 30.0) / 195.0
        
        features = [norm_group, norm_period, norm_nvalence, norm_en, norm_radius]
        
        # 計算グラフに流し込めるよう float32 でキャスト
        return torch.tensor(features, dtype=torch.float32)
        
    except Exception:
        # mendeleev内部での想定外のエラー（未定義の巨大原子など）は安全に握り潰す
        return None

def build_periodic_table_cache(max_atomic_num: int = 54) -> Dict[int, torch.Tensor]:
    """
    1からmax_atomic_num（デフォルトはXe=54まで）の元素についてループを回し、
    正規化された5次元特徴量ベクトルを持つ辞書を構築する。
    内部で _get_normalized_features を呼び出し、None が返ってきた原子はスキップする。
    """
    cache = {}
    for atomic_num in range(1, max_atomic_num + 1):
        features = _get_normalized_features(atomic_num)
        if features is not None:
            cache[atomic_num] = features
            
    return cache
