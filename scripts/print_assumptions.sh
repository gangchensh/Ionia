#!/usr/bin/env bash
# Axiom audit: run Print Assumptions on the key theorems of the release.
# Usage (from the release root, after `make -f Makefile.coq`):
#   ./scripts/print_assumptions.sh
#
# Scope note: this audits the dependency closure of the key theorems
# listed in print_assumptions.v (core rules, parallel safety, all
# optimization rules, all 12 network theorems, the MiniTransformer
# faithful bridge, the A/B bridges).  src/Ionia additionally contains a
# few historical `Axiom` declarations from the route-B development that
# are NOT in any audited theorem's dependency closure (any dependence
# would show up in the output below):
#   - src/Ionia/CNN_CodeGen.v: CGexpvar_gen_oob, writeNat_str2fin,
#     writeNat_fin2nat_str2fin, writeNat_add, writeNat_sub, writeNat_mul,
#     writeNat_div, writeNat_mod (code-generation conventions for the
#     route-B printer experiments);
#   - src/Ionia/CNN_Examples_Base.v: Conv2d_equiv_concrete_4x4,
#     Conv3d_equiv_concrete_1x4x4, Conv4d_equiv_concrete_1x1x4x4,
#     ConvTranspose2d_equiv_concrete_2x2,
#     ConvTranspose3d_equiv_concrete_1x2x2,
#     ConvTranspose4d_equiv_concrete_1x1x2x2, Conv4d_p1_k3_s2_equiv,
#     Conv4d_p3_k7_s2_equiv, Conv4d_p0_k1_s2_equiv (concrete-shape
#     operator equivalences, superseded by the proved general forms);
#   - src/Ionia/CNN_Examples_LeNet5.v: lenet5_equiv (the original route-B
#     LeNet-5 statement, superseded by LeNet5_DPIA.A_LeNet5).
set -euo pipefail
cd "$(dirname "$0")/.."

COQC="${COQC:-coqc}"
OUT=scripts/print_assumptions_output.txt

"$COQC" -R src/Ionia Ionia scripts/print_assumptions.v > "$OUT" 2>&1 || {
    echo "coqc failed; see $OUT"; exit 1; }

echo "=== Axioms referenced (unique) ==="
grep -E '^[A-Za-z_][A-Za-z0-9_.]* :' "$OUT" | sort -u || true
echo
echo "=== Full output in $OUT ==="
if grep -qi 'admitted\|Axiom ' "$OUT"; then
    echo "WARNING: non-standard axiom or Admitted detected — inspect $OUT"
    exit 2
fi
echo "OK: only standard library axioms are used."
