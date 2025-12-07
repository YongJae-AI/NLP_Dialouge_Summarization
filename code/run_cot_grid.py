"""
CoT/encoder 템플릿 조합 실험 실행 스크립트.
각 조합마다 full_pipeline을 호출해 학습+추론+제출 파일을 생성한다.
lr=3e-5 고정.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

encoder_keys = [
    "enc_guide_block",
    "enc_role_preserve",
    "enc_3step_extract",
]

cot_keys = [
    "cot_standard_reasoning",
    "cot_dialogue_decomp",
    "cot_hierarchical",
    "cot_reflexion",
    "cot_light_planning",
]

CONFIG = "config/config_kobart_style_prompt.yaml"  # KoBART 기본 설정 사용 (lr=3e-5)


def run_one(enc_key: str, cot_key: str):
    run_name = f"enc={enc_key}_cot={cot_key}"
    cmd = [
        "python",
        str(ROOT / "code" / "full_pipeline.py"),
        "--config",
        str(ROOT / CONFIG),
        "--run_name",
        run_name,
        "--enc_key",
        enc_key,
        "--cot_key",
        cot_key,
    ]
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    for enc_key in encoder_keys:
        for cot_key in cot_keys:
            run_one(enc_key, cot_key)


if __name__ == "__main__":
    main()
