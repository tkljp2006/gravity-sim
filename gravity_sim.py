"""
模擬宇宙 - Step 1: 地球の重力シミュレーション + AI連携
=========================================================
構造:
  ユーザーの質問 → simulate() で物理計算 → AIが自然言語で回答

必要なライブラリのインストール:
  pip install anthropic

使い方:
  python gravity_sim.py
"""

import math
import anthropic
import json


# ── 物理エンジン本体 ──────────────────────────────────────────

def simulate(mass_kg: float, height_m: float, drag_coeff: float = 0.0, dt: float = 0.001) -> dict:
    """
    地球表面での自由落下シミュレーション

    引数:
        mass_kg    : 物体の質量 (kg)
        height_m   : 初期高度 (m)
        drag_coeff : 空気抵抗係数 (0 = 真空, 0.47 = 球体の目安)
        dt         : タイムステップ (秒)

    戻り値:
        dict: シミュレーション結果
    """
    G = 9.81          # 重力加速度 (m/s²)
    RHO = 1.225       # 空気密度 (kg/m³) ※標準大気
    AREA = 0.01       # 断面積 (m²) ※固定値、今後可変にする予定

    h = height_m
    v = 0.0
    t = 0.0
    trajectory = []

    while h > 0:
        # 空気抵抗による力 (F = 0.5 * Cd * ρ * A * v²)
        drag_force = 0.5 * drag_coeff * RHO * AREA * v * v
        # 合力から加速度を計算 (F = ma → a = F/m)
        a = G - (drag_force / mass_kg)
        # 速度・位置を更新 (オイラー法)
        v += a * dt
        h -= v * dt
        t += dt

        # 0.1秒ごとに記録
        if round(t * 10) % 1 == 0:
            trajectory.append({
                "t": round(t, 2),
                "height": round(max(h, 0), 3),
                "velocity": round(v, 3),
                "acceleration": round(a, 3),
            })

    # 理論値 (空気抵抗なし)
    theoretical_time = math.sqrt(2 * height_m / G)
    theoretical_v    = G * theoretical_time

    return {
        "mass_kg": mass_kg,
        "initial_height_m": height_m,
        "drag_coefficient": drag_coeff,
        "fall_time_s": round(t, 3),
        "impact_velocity_ms": round(v, 2),
        "kinetic_energy_j": round(0.5 * mass_kg * v * v, 2),
        "theoretical_time_s": round(theoretical_time, 3),
        "theoretical_velocity_ms": round(theoretical_v, 2),
        "trajectory_sample": trajectory[:5],  # 最初の5点だけ渡す(トークン節約)
    }


# ── AI連携レイヤー ────────────────────────────────────────────

def ask_physics(question: str) -> str:
    """
    自然言語の質問 → 物理シミュレーション → AI回答

    引数:
        question: ユーザーの質問文

    戻り値:
        AIによる回答テキスト
    """
    client = anthropic.Anthropic()

    # Step 1: AIに質問からパラメータを抽出させる
    extract_prompt = f"""
以下の質問から物理シミュレーションのパラメータを抽出し、JSONのみで返してください。
余分なテキスト・マークダウンは不要です。

質問: {question}

出力形式:
{{
  "mass_kg": <数値>,
  "height_m": <数値>,
  "drag_coeff": <数値 0.0〜1.0、明記なければ0.0>
}}
"""

    param_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{"role": "user", "content": extract_prompt}]
    )

    raw = param_response.content[0].text.strip()

    try:
        params = json.loads(raw)
    except json.JSONDecodeError:
        return "パラメータの抽出に失敗しました。質問をより具体的にしてください。"

    # Step 2: 物理エンジンでシミュレーション実行
    result = simulate(
        mass_kg=params.get("mass_kg", 1.0),
        height_m=params.get("height_m", 100.0),
        drag_coeff=params.get("drag_coeff", 0.0),
    )

    # Step 3: AIがシミュレーション結果を自然言語で回答
    answer_prompt = f"""
あなたは物理シミュレーションエンジンと連携したAIアシスタントです。
以下のシミュレーション結果を使って、ユーザーの質問に日本語で分かりやすく答えてください。

ユーザーの質問: {question}

シミュレーション結果:
{json.dumps(result, ensure_ascii=False, indent=2)}

回答のポイント:
- 主要な数値（落下時間・衝突速度）を明確に伝える
- 理論値との差がある場合はその理由も説明する
- 専門用語は噛み砕いて説明する
"""

    answer_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": answer_prompt}]
    )

    return answer_response.content[0].text


# ── メイン ────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  模擬宇宙 v0.1 — 重力シミュレーター")
    print("  'quit' で終了")
    print("=" * 50)

    # サンプル質問で動作確認
    sample_questions = [
        "100kgの物体を200mの高さから落としたら何秒で着地しますか？",
        "10kgのボールを空気抵抗あり(係数0.47)で50mから落としたときの衝突速度は？",
    ]

    print("\n【サンプル実行】")
    for q in sample_questions:
        print(f"\n質問: {q}")
        print("計算中...")
        answer = ask_physics(q)
        print(f"回答: {answer}")
        print("-" * 40)

    # 対話モード
    print("\n【対話モード】質問を入力してください")
    while True:
        question = input("\n質問 > ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("終了します。")
            break
        if not question:
            continue
        print("計算中...")
        print(f"回答: {ask_physics(question)}")


if __name__ == "__main__":
    main()
