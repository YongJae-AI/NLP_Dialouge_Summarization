import os
import subprocess
import sys


def main() -> None:
    base_env = os.environ.copy()

    # PYTHONPATH 유지 (외부에서 PYTHONPATH=src 로 실행했다면 그대로 전달)
    python_exe = sys.executable

    commands = [
        [python_exe, "code/one_off/run_kobart_bs8.py"],
        [python_exe, "code/one_off/run_kobart_bs4.py"],
    ]

    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=base_env)
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}: {' '.join(cmd)}")
            break


if __name__ == "__main__":
    main()

