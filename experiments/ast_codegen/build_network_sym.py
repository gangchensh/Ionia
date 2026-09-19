#!/usr/bin/env python3
"""
Build a patched *_sym.ml file for a given network.

We take the fully patched alexnet_sym.ml as the generic base (it contains
symbolic nat/fin, path-indexed vwt/vrd, functional mk_vec/v_nth, etc.),
then append any additional generic helpers and the network-specific helpers
from the pure extracted *_comm.ml produced by Ionia/Extract_Comms.v.

Duplicate type definitions and duplicate top-level let bindings in the extra
slice are skipped, because the base already contains patched versions of the
generic helpers (e.g. vrd/vwt).  Only helpers that are new or different for
the target network are kept.

Usage:
  python3 build_network_sym.py <network> <pure_extracted_comm.ml>
"""

import sys
import os
import re

NETWORK_RANGES = {
    "alexnet": {
        # The new base alexnet_sym.ml already contains all generic helpers and
        # all alexnet-specific helpers except the final alexnet_comm AST.  We
        # therefore append only the top-level alexnet_comm definition.
        "pure_extra_start": 1676,   # let alexnet_comm = ...
        "pure_end": None,           # end of file
    },
    "lenet5": {
        "pure_extra_start": 1172,   # reLU1d_comm
        "pure_end": None,
    },
}

TYPE_DECL_RE = re.compile(r"^type\s+('?[A-Za-z_][A-Za-z0-9_']*\s+)*([A-Za-z_][A-Za-z0-9_']*)\b")
LET_DECL_RE = re.compile(r"^let\s+(rec\s+)?([A-Za-z_][A-Za-z0-9_']*)\b")
MODULE_RE = re.compile(r"^module\s+([A-Za-z_][A-Za-z0-9_']*)\b")
INCLUDE_RE_TMPL = r"^include\s+{}\s*$"


def collect_names(lines, regex, group_index=1):
    """Return the set of names declared in the given OCaml lines."""
    names = set()
    for line in lines:
        m = regex.match(line)
        if m:
            names.add(m.group(group_index))
    return names


def is_toplevel_start(line):
    """True if line starts a new top-level construct."""
    s = line.strip()
    if s == "":
        return True
    if line.startswith("("):
        return True
    if line.startswith("let ") or line.startswith("let\t"):
        return True
    if line.startswith("type ") or line.startswith("type\t"):
        return True
    return False


def collect_module_names(lines):
    """Return the set of module names declared in the given OCaml lines."""
    names = set()
    for line in lines:
        m = MODULE_RE.match(line)
        if m:
            names.add(m.group(1))
    return names


def find_module_end(lines, start_idx):
    """
    Find the index of the matching 'end' for a module that starts at start_idx.
    We assume the module body is introduced by 'struct' (possibly on the next
    line) and that the closing 'end' is indented at most one space.
    """
    n = len(lines)
    # Find the 'struct' or 'sig' keyword that opens the module body.
    body_start = start_idx
    for i in range(start_idx, min(n, start_idx + 5)):
        if re.search(r"\b(struct|sig)\b", lines[i]):
            body_start = i
            break
    # Search forward for the closing 'end' at low indentation.
    for i in range(body_start + 1, n):
        line = lines[i]
        stripped = line.strip()
        if stripped == 'end':
            # Closing end is aligned with the module keyword (<=1 leading space).
            leading = len(line) - len(line.lstrip(' '))
            if leading <= 1:
                return i
    return None


def replace_module(base_lines, extra_lines, module_name):
    """
    If both base_lines and extra_lines contain a module named module_name,
    replace the base copy with the extra copy and remove the extra copy from
    extra_lines.  Returns (new_base_lines, new_extra_lines).
    """
    base_idx = None
    for i, line in enumerate(base_lines):
        if re.match(rf"^module\s+{re.escape(module_name)}\b", line):
            base_idx = i
            break
    extra_idx = None
    for i, line in enumerate(extra_lines):
        if re.match(rf"^module\s+{re.escape(module_name)}\b", line):
            extra_idx = i
            break
    if base_idx is None or extra_idx is None:
        return base_lines, extra_lines

    base_end = find_module_end(base_lines, base_idx)
    extra_end = find_module_end(extra_lines, extra_idx)
    if base_end is None or extra_end is None:
        print(f"Warning: could not find end of module {module_name}; skipping replacement")
        return base_lines, extra_lines

    new_base = base_lines[:base_idx] + extra_lines[extra_idx:extra_end + 1] + base_lines[base_end + 1:]
    new_extra = extra_lines[:extra_idx] + extra_lines[extra_end + 1:]
    return new_base, new_extra


def filter_duplicates(extra_lines, base_type_names, base_let_names, base_module_names):
    """Drop type/let/module declarations from extra_lines that already exist in the base."""
    filtered = []
    skip_kind = None  # ('type', name), ('let', name), or ('module', name)
    pending_include = None  # module name whose 'include X' should be dropped

    for line in extra_lines:
        # Drop a trailing 'include X' that followed a skipped module.
        if pending_include is not None:
            if re.match(r"^include\s+" + re.escape(pending_include) + r"\s*$", line):
                pending_include = None
                continue
            pending_include = None

        mm = MODULE_RE.match(line)
        if mm:
            name = mm.group(1)
            if name in base_module_names:
                skip_kind = ('module', name)
                continue
            else:
                skip_kind = None

        tm = TYPE_DECL_RE.match(line)
        if tm:
            name = tm.group(2)
            if name in base_type_names:
                skip_kind = ('type', name)
                continue
            else:
                skip_kind = None

        lm = LET_DECL_RE.match(line)
        if lm:
            name = lm.group(2)
            if name in base_let_names:
                skip_kind = ('let', name)
                continue
            else:
                skip_kind = None

        if skip_kind is not None:
            if skip_kind[0] == 'module':
                if line.strip() == 'end' and not line[0].isspace():
                    pending_include = skip_kind[1]
                    skip_kind = None
                continue
            elif is_toplevel_start(line):
                skip_kind = None
            else:
                continue

        filtered.append(line)
    return filtered


def main():
    if len(sys.argv) != 3:
        print("Usage: build_network_sym.py <network> <pure_extracted_comm.ml>")
        sys.exit(1)

    network, pure_path = sys.argv[1:3]
    cfg = NETWORK_RANGES.get(network)
    if cfg is None:
        print(f"Unknown network: {network}")
        sys.exit(1)

    base_path = "alexnet_sym.ml"
    out_path = f"{network}_sym.ml"

    if not os.path.exists(base_path):
        print(f"Base file not found: {base_path}")
        sys.exit(1)

    with open(base_path, 'r') as f:
        base_lines = f.readlines()

    # Find where the base's own alexnet_comm AST starts; we drop from there.
    # The marker can be either a Coq-style '(** val ... *)' comment or the
    # 'let alexnet_comm =' binding itself.
    drop_from = None
    comm_val_marker = "(** val alexnet_comm : comm **)"
    comm_let_marker = "let alexnet_comm ="
    for i, line in enumerate(base_lines):
        if line.startswith(comm_val_marker) or line.startswith(comm_let_marker):
            drop_from = i
            break
    if drop_from is None:
        print("Could not find alexnet_comm marker in base file")
        sys.exit(1)

    with open(pure_path, 'r') as f:
        pure_lines = f.readlines()

    start = cfg["pure_extra_start"] - 1   # 0-based
    end = cfg["pure_end"]  # None means to end of file
    if end is not None:
        end -= 1

    extra_lines = pure_lines[start:end]

    base_prefix = base_lines[:drop_from]

    # The Z module in some extractions contains extra functions (e.g. min)
    # that are missing from the alexnet base.  Replace the base copy with the
    # more complete pure copy before doing any other filtering.
    base_prefix, extra_lines = replace_module(base_prefix, extra_lines, "Z")

    base_type_names = collect_names(base_prefix, TYPE_DECL_RE, group_index=2)
    base_let_names = collect_names(base_prefix, LET_DECL_RE, group_index=2)
    base_module_names = collect_module_names(base_prefix)
    extra_lines = filter_duplicates(extra_lines, base_type_names, base_let_names, base_module_names)

    with open(out_path, 'w') as f:
        f.writelines(base_prefix)
        f.writelines(extra_lines)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
