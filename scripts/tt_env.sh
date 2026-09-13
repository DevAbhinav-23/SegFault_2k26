# The Tenstorrent functional-simulator environment (design/PROGRESS-TT.md §2).
#
# Usage:  source scripts/tt_env.sh        # builds .venv-tt on first use, then exports
#         .venv-tt/bin/python -m pytest -m requires_ttsim tests/tt
#
# Why a second venv: the ttnn 0.78.0 wheel pins `numpy<2` and the project pins `numpy==2.5.3`,
# so the two cannot share `.venv`. `.venv-tt` is git-ignored; every artefact it is built from is
# checksummed in vendor/tt/SHA256SUMS (the same R-13 discipline as vendor/wheels).
#
# The three environment facts are the gate report's, verbatim (tt-probe/REPRO.sh):
#   TT_METAL_SIMULATOR        the ttsim shared object; its directory must also hold
#                             soc_descriptor.yaml, under exactly that name
#   TT_METAL_SLOW_DISPATCH_MODE=1   no fast-dispatch firmware exists for the simulator
#   TT_METAL_HOME             must be UNSET: the wheel is self-rooted, and a stale value
#                             sends tt_metal looking for a source tree that is not there

_tt_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
_tt_venv="$_tt_root/.venv-tt"
_tt_cache="$_tt_root/vendor/tt"

if [ ! -x "$_tt_venv/bin/python" ]; then
    ( set -e
      cd "$_tt_cache"
      sha256sum -c SHA256SUMS
      uv venv --python 3.12 "$_tt_venv"
      uv pip install --python "$_tt_venv/bin/python" \
         "$_tt_cache"/ttnn-0.78.0-*.whl pytest
      _sp=$("$_tt_venv/bin/python" -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
      # sfpi is the device C++ toolchain. It is NOT in the wheel; tt_metal looks for it at
      # exactly <site-packages>/ttnn/runtime/sfpi (gate report G-TT0).
      rm -rf "$_sp/ttnn/runtime/sfpi"
      tar -xJf "$_tt_cache"/sfpi_7.75.1_x86_64_debian.txz -C "$_sp/ttnn/runtime"
      # The repo itself: `pip install -e .` would drag in numpy==2.5.3 and break ttnn.
      echo "$_tt_root" > "$_sp/segfault_repo.pth"
    ) || { echo "tt_env: building $_tt_venv failed" >&2; unset _tt_root _tt_venv _tt_cache; return 1; }
fi

export TT_METAL_SIMULATOR="$_tt_cache/libttsim_wh.so"
export TT_METAL_SLOW_DISPATCH_MODE=1
unset TT_METAL_HOME
unset _tt_root _tt_venv _tt_cache
