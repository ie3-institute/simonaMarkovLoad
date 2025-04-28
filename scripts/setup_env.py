import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]):
    print("›", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    run(["poetry", "install"])
    run(["poetry", "run", "pre-commit", "install"])


if __name__ == "__main__":
    main()
