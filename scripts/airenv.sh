# Put aiecc and Peano where aircc can find them (design/07-environment.md §4).
# Usage: source scripts/airenv.sh   [python-executable]
_py="${1:-$(dirname "${BASH_SOURCE[0]:-$0}")/../.venv/bin/python}"
SP=$("$_py" -c "import sysconfig;print(sysconfig.get_paths()['purelib'])") || return 1
[ -d "$SP/mlir_air/bin" ] || { echo "airenv: no mlir_air in $SP" >&2; return 1; }
export PATH="$SP/mlir_air/bin:$SP/mlir_aie/bin:$PATH"
export PEANO_INSTALL_DIR="$SP/llvm-aie"
unset _py
