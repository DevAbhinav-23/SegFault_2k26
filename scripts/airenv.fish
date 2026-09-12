# Put aiecc and Peano where aircc can find them (design/07-environment.md §4).
# Usage: source scripts/airenv.fish   [python-executable]
set -l py (test -n "$argv[1]"; and echo $argv[1]; or echo (status dirname)/../.venv/bin/python)
set -l SP ($py -c "import sysconfig;print(sysconfig.get_paths()['purelib'])"); or return 1
test -d "$SP/mlir_air/bin"; or begin; echo "airenv: no mlir_air in $SP" >&2; return 1; end
set -gx PATH $SP/mlir_air/bin $SP/mlir_aie/bin $PATH
set -gx PEANO_INSTALL_DIR $SP/llvm-aie
