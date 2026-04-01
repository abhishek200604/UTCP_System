"""
setup.py — UTCP System Installer
Installs all dependencies with styled terminal output.

Usage:  python setup.py
"""

import subprocess
import sys
import time

# ── ANSI colour codes ──
class C:
    RST  = "\033[0m"
    BOLD = "\033[1m"
    DIM  = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED  = "\033[91m"
    BLUE = "\033[94m"
    MAG  = "\033[95m"
    WHITE = "\033[97m"


# Packages to install (must match requirements.txt)
PACKAGES = [
    ("streamlit",     ">=1.24.0",  "🌐", "Web Dashboard Framework"),
    ("pandas",        ">=1.5.0",   "🐼", "Data Manipulation"),
    ("numpy",         ">=1.23.0",  "🔢", "Numerical Computing"),
    ("scikit-learn",  ">=1.2.0",   "🧠", "Machine Learning Models"),
    ("matplotlib",    ">=3.6.0",   "📊", "Static Plotting"),
    ("seaborn",       ">=0.12.0",  "🎨", "Statistical Visualization"),
    ("plotly",        ">=5.13.0",  "📈", "Interactive Charts & Maps"),
    ("joblib",        ">=1.2.0",   "💾", "Model Serialization"),
]



def _emoji_extra(text):
    import unicodedata
    extra = 0
    for ch in text:
        if unicodedata.east_asian_width(ch) in ('W', 'F'):
            extra += 1
        elif ord(ch) > 0xFFFF:
            extra += 1
    return extra


def _cbox(text, width):
    pad = width - len(text) - _emoji_extra(text)
    left = pad // 2
    right = pad - left
    return ' ' * left + text + ' ' * right


def _header():
    W = 52
    print()
    print(f"  {C.CYAN}{C.BOLD}╔{'═' * W}╗{C.RST}")
    print(f"  {C.CYAN}{C.BOLD}║{' ' * W}║{C.RST}")
    print(f"  {C.CYAN}{C.BOLD}║{_cbox('UTCP System  —  Setup', W)}║{C.RST}")
    print(f"  {C.CYAN}{C.BOLD}║{_cbox('Installing Dependencies', W)}║{C.RST}")
    print(f"  {C.CYAN}{C.BOLD}║{' ' * W}║{C.RST}")
    print(f"  {C.CYAN}{C.BOLD}╚{'═' * W}╝{C.RST}")
    print()


def install_package(name, version_spec):
    """Install a single package quietly, return (success, version_installed)."""
    pkg = f"{name}{version_spec}"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "-q", "--disable-pip-version-check"],
        capture_output=True,
        text=True,
    )
    # Get installed version
    ver_result = subprocess.run(
        [sys.executable, "-m", "pip", "show", name],
        capture_output=True,
        text=True,
    )
    version = "?"
    for line in ver_result.stdout.splitlines():
        if line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()
            break
    return result.returncode == 0, version


def main():
    _header()
    t0 = time.time()

    print(f"  {C.DIM}{'─' * 68}{C.RST}")
    print(f"  {C.BOLD}  {'Package':<18}{'Description':<28}{'Version':>10}  Status{C.RST}")
    print(f"  {C.DIM}{'─' * 68}{C.RST}")

    success_count = 0
    fail_count = 0

    for name, ver_spec, emoji, desc in PACKAGES:
        ok, version = install_package(name, ver_spec)
        if ok:
            success_count += 1
            status = f"{C.GREEN}✅{C.RST}"
            ver_str = f"{C.CYAN}{version:>10}{C.RST}"
        else:
            fail_count += 1
            status = f"{C.RED}❌{C.RST}"
            ver_str = f"{C.RED}{'FAILED':>10}{C.RST}"

        print(
            f"  {emoji} {C.WHITE}{name:<17}{C.RST}"
            f"{C.DIM}{desc:<28}{C.RST}"
            f"{ver_str}  {status}"
        )

    print(f"  {C.DIM}{'─' * 68}{C.RST}")
    elapsed = time.time() - t0
    print()

    # ── Summary ──
    W = 52
    if fail_count == 0:
        print(f"  {C.GREEN}{C.BOLD}╔{'═' * W}╗{C.RST}")
        print(f"  {C.GREEN}{C.BOLD}║{' ' * W}║{C.RST}")
        print(f"  {C.GREEN}{C.BOLD}║{_cbox('ALL DEPENDENCIES INSTALLED', W)}║{C.RST}")
        print(f"  {C.GREEN}{C.BOLD}║{_cbox(f'{success_count} packages  ·  {elapsed:.1f}s', W)}║{C.RST}")
        print(f"  {C.GREEN}{C.BOLD}║{' ' * W}║{C.RST}")
        print(f"  {C.GREEN}{C.BOLD}╚{'═' * W}╝{C.RST}")
    else:
        print(f"  {C.RED}{C.BOLD}╔{'═' * W}╗{C.RST}")
        print(f"  {C.RED}{C.BOLD}║{' ' * W}║{C.RST}")
        print(f"  {C.RED}{C.BOLD}║{_cbox('SOME PACKAGES FAILED', W)}║{C.RST}")
        print(f"  {C.RED}{C.BOLD}║{_cbox(f'{fail_count} failed  ·  {success_count} ok  ·  {elapsed:.1f}s', W)}║{C.RST}")
        print(f"  {C.RED}{C.BOLD}║{' ' * W}║{C.RST}")
        print(f"  {C.RED}{C.BOLD}╚{'═' * W}╝{C.RST}")

    print()
    print(f"  {C.MAG}💡 Next steps:{C.RST}")
    print(f"     {C.DIM}1.{C.RST} {C.CYAN}python train_model.py{C.RST}    {C.DIM}— Train the ML model{C.RST}")
    print(f"     {C.DIM}2.{C.RST} {C.CYAN}streamlit run app.py{C.RST}     {C.DIM}— Launch the dashboard{C.RST}")
    print()


if __name__ == "__main__":
    main()
