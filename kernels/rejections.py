"""The three demo rejections (design/03-lld-M8-kernels-demo.md §3.5).

Each is a two-line edit of a working schedule. The surface (Person A M1/M2)
does not exist yet, so each raises NotImplementedError; docstrings carry the
expected message shapes from §5.3.
"""


def bad_skew(target):
    """L2-CAUSALITY: dependence (0, 1) is not carried forward by the schedule.

    Expected message shape (§5.3)::

        L2-CAUSALITY: dependence (0, 1) is not carried forward by the schedule
          in clause: skew(time=(ax.i,))
          ...
          because:   Ssigma = [e_i]; for the dependence d = (0, +1, -7) ...
                     Ssigma . d = 0, and causality needs Ssigma . d >= 1 ...
          fix:       skew(time=(ax.i, ax.j0)) ...
    """
    raise NotImplementedError("surface: Person A M1/M2")


def bad_stationary(target):
    """STATIONARITY: operand C cannot be stationary under this placement.

    Expected message shape (§5.3)::

        STATIONARITY: operand C cannot be stationary under this placement
          in clause: stationary("C")
          ...
          because:   ker M_C = span{e_k} ... ker Spi = span{e_i1, e_j0, ...}
                     span{e_k} is not contained in ker Spi ...
          fix:       place(px=ax.i0, py=ax.j0) ... or stationary("B") ...
    """
    raise NotImplementedError("surface: Person A M1/M2")


def bad_capacity(target):
    """L1-CAPACITY: the per-core working set is 86 016 bytes vs 65 536 budget.

    Expected message shape (§5.3)::

        L1-CAPACITY: the per-core working set is 86 016 bytes against a
        65 536 byte budget
          in clause: double_buffer("A", "B")
          ...
          because:   acc [96, 96] f32 = 36 864 / a doubled -> 24 576 /
                     b doubled -> 24 576 / total 86 016 > 65 536 ...
          fix:       tile(ax.k, 16) instead of 32 ... or drop double_buffer ...
    """
    raise NotImplementedError("surface: Person A M1/M2")
