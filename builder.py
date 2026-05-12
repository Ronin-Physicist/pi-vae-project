import torch
from typing import Dict, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # RDKitのC++レベルの警告・エラー出力を完全にミュートする

def _smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """SMILES文字列からRDKitのMolオブジェクトを生成し、明示的な水素を付加する"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 立体化学や双極子モーメント計算に不可欠なため水素を明示的に付加
        mol = Chem.AddHs(mol)
        return mol
    except Exception:
        return None

def _generate_3d_coords(mol: Chem.Mol, seed: int) -> Optional[torch.Tensor]:
    """ETKDGアルゴリズムを用いて決定論的に3D配座を生成する"""
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        res = AllChem.EmbedMolecule(mol, params)
        if res == -1:
            return None
        
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()
        coords = [conf.GetAtomPosition(i) for i in range(num_atoms)]
        x = torch.tensor([[p.x, p.y, p.z] for p in coords], dtype=torch.float32)
        return x
    except Exception:
        return None

def _compute_gasteiger_charges(mol: Chem.Mol) -> Optional[torch.Tensor]:
    """Gasteiger法を用いて部分電荷を計算する"""
    try:
        AllChem.ComputeGasteigerCharges(mol)
        
        charges = []
        for atom in mol.GetAtoms():
            # '_GasteigerCharge' プロパティを取得
            charge = atom.GetDoubleProp('_GasteigerCharge')
            charges.append([charge])
            
        charges_tensor = torch.tensor(charges, dtype=torch.float32)
        
        # 計算不能な原子によりNaNやInfが発生した場合は安全にNoneを返す
        if torch.isnan(charges_tensor).any() or torch.isinf(charges_tensor).any():
            return None
            
        return charges_tensor
    except Exception:
        return None

def _extract_edge_index(mol: Chem.Mol) -> torch.Tensor:
    """PyTorch Geometric互換の無向グラフエッジインデックスを生成する"""
    src = []
    dst = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # 無向グラフとして両方向のエッジを登録
        src.extend([i, j])
        dst.extend([j, i])
        
    # 結合が一つもない単原子分子（メタンの水素除去版など、通常あり得ないが）のフォールバック
    if not src:
        return torch.empty((2, 0), dtype=torch.long)
        
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index

def _extract_node_features(mol: Chem.Mol, pt_cache: Dict[int, torch.Tensor]) -> Optional[torch.Tensor]:
    """周期表キャッシュから各原子の特徴量を引き当てる"""
    features = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num not in pt_cache:
            # キャッシュに存在しない（想定外の）原子番号の場合はフェイルセーフ
            return None
        features.append(pt_cache[atomic_num])
        
    # 個々のTensor [5] をスタックして [N, 5] のTensorにする
    h = torch.stack(features)
    return h

def smiles_to_pure_graph(
    smiles: str, 
    periodic_table_cache: Dict[int, torch.Tensor], 
    seed: int = 42
) -> Optional[Dict[str, torch.Tensor]]:
    """
    SMILES文字列から、EGNN入力および物理量計算に必要なテンソル辞書を構築する。
    途中のRDKit処理で失敗した場合は、例外を投げずに None を返す。
    """
    mol = _smiles_to_mol(smiles)
    if mol is None:
        return None
        
    x = _generate_3d_coords(mol, seed)
    if x is None:
        return None
        
    charges = _compute_gasteiger_charges(mol)
    if charges is None:
        return None
        
    # Node Featuresの抽出（未知の原子が含まれていればここで弾く）
    h = _extract_node_features(mol, periodic_table_cache)
    if h is None:
        return None
        
    edge_index = _extract_edge_index(mol)
    
    return {
        "x": x,
        "h": h,
        "edge_index": edge_index,
        "charges": charges
    }
