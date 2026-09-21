"""Print the environment used for a run (recorded in results/env.json)."""
import json, os, platform, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def main():
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine()}
    for mod in ("numpy", "scipy", "sympy", "statsmodels"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception as exc:                       # pragma: no cover
            info[mod] = f"not installed ({exc})"
    os.makedirs(OUT, exist_ok=True)
    json.dump(info, open(os.path.join(OUT, "env.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
