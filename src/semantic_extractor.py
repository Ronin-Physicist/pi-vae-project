import os
import json
import google.generativeai as genai
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io

# ==========================================
# Phase 1: The Gatekeeper (API & RDKit Setup)
# ==========================================

# ※実行前に環境変数 GEMINI_API_KEY を設定してください
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Gemini 1.5 Proの初期化（Flashでも可能ですが、より深い意味解釈のためにProを推奨）
# 構造化出力（JSON）を強制するために response_schema を使用します
model = genai.GenerativeModel('gemini-1.5-pro')

def smiles_to_image(smiles: str) -> Image.Image:
    """RDKitを用いてSMILESから2D画像（PIL Image）を生成する"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # 描画オプション（AIが見やすいように水素を明示するなど、今後チューニングの余地あり）
    img_bytes = Draw.MolToImage(mol, size=(400, 400), returnPNG=True)
    return Image.open(io.BytesIO(img_bytes))

# ==========================================
# Phase 2: The Architect (Semantic Extraction)
# ==========================================

class SemanticVector(BaseModel):
    """Geminiに出力させるJSONのスキーマ定義"""
    rod_like: float = Field(description="棒状・剛直な直線性を持つ度合い (0.0 - 1.0)")
    y_shaped: float = Field(description="Y字型・分岐によるパッキング阻害の度合い (0.0 - 1.0)")
    spherical: float = Field(description="球状・丸まりやすさの度合い (0.0 - 1.0)")
    flexible_chain: float = Field(description="柔軟な鎖状・ふにゃふにゃ度合い (0.0 - 1.0)")
    
def get_semantic_vector(image: Image.Image) -> dict:
    """Gemini APIを叩き、ノルム1のSoft Labelベクトルを取得する"""
    
    # E-GNNの文脈をLLMに注入する「プロンプト・エンジニアリング」
    prompt = """
    あなたはマテリアルズ・インフォマティクスの専門家です。
    提供された画像は化学分子の2D構造式です。
    
    このタスクの目的は、この分子のマクロな形状（ゲシュタルト）を評価し、
    後続の等変グラフニューラルネットワーク（E-GNN）のGlobal Featureとして
    使用するための「セマンティック・ベクトル」を作成することです。
    
    以下の4つのカテゴリに対する該当度合いを、連続値で評価してください。
    重要: 4つの値の合計が必ず 1.0 になるように（確率分布のように）出力してください。
    
    1. rod_like: ガチガチに固まった棒状
    2. y_shaped: Y字型の分岐
    3. spherical: 丸まった球状
    4. flexible_chain: 柔軟に曲がる鎖状（PMIが同じでもrod_likeと区別してください）
    """
    
    response = model.generate_content(
        [prompt, image],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=SemanticVector,
            temperature=0.2 # 幻覚を抑え、決定論的な評価に近づけるため低めに設定
        )
    )
    
    # JSON文字列をパース
    raw_vector = json.loads(response.text)
    
    # 【泥臭いエンジニアリングの防衛線】
    # LLMは「合計1.0にして」と言っても計算ミスすることがあるため、
    # Python側で確実にL1正規化（ノルム1）を行う
    total = sum(raw_vector.values())
    normalized_vector = {k: round(v / total, 3) for k, v in raw_vector.items()}
    
    return normalized_vector

# ==========================================
# Phase 3: The Experiment (PoC Execution)
# ==========================================

if __name__ == "__main__":
    # テストケース：剛直な棒 vs 柔軟な棒 vs Y字 vs 球状
    test_cases = {
        "Rigid_Rod (剛直な棒)": "C1=CC=C(C=C1)C2=CC=C(C=C2)C3=CC=C(C=C3)", # p-テルフェニル
        "Flexible_Chain (柔軟な鎖)": "CCCCCCCCCCCC", # ドデカン
        "Y_Shaped (Y字型)": "CC(C)(C)C1=CC(=CC(=C1)C(C)(C)C)C(C)(C)C", # 1,3,5-トリ-tert-ブチルベンゼン
        "Spherical (球状っぽい)": "C12C3C4C1C5C4C3C25" # キューバン（2D描画だとどう見えるかテスト）
    }
    
    for name, smiles in test_cases.items():
        print(f"--- Testing: {name} ---")
        try:
            # 1. RDKitで画像化
            img = smiles_to_image(smiles)
            # 2. Geminiでセマンティックラベル抽出
            vector = get_semantic_vector(img)
            
            print(f"SMILES: {smiles}")
            print(f"Semantic Vector: {json.dumps(vector, indent=2)}\n")
            
        except Exception as e:
            print(f"Error during processing {name}: {e}\n")


