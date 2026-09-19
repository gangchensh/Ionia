#!/usr/bin/env python3
"""Patch alexnet_par_sym.ml so that the expression-level Padding2d reads of
Conv4d_ast_comm / Maxpool4d_pad_ast_comm evaluate correctly at SYMBOLIC
indices.

Background: the extracted padding2d decides between the zero padding and the
array read with `Nat.ltb_spec0`.  The symbolic `ltb` (transform_to_symbolic.py)
defaults to True for non-concrete indices, so every padded read collapses to
`exp_default` (0.0) — wrong for the loop-variable indices of generated code.

The patch rewrites the Num case to build REif (a native DataExp.rexp
constructor) conditionals, which the printer (comm_to_c_alexnet_par.ml) emits
as C ternaries — deferring the decision to run time.  This lives in the
untrusted symbolic/printing layer and is backed by the NumPy numeric
validation (validate_alexnet_par.py)."""

import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "alexnet_par_sym.ml"

with open(PATH) as f:
    src = f.read()

# The DataExp copy of acc_default builds the "none" path as a Coq-string
# literal, but [path]'s [Var] takes a native OCaml string.
src, n = re.subn(
    r'Obj\.magic \(Var \(String[\s\S]*?EmptyString\)+',
    'Obj.magic (Var "none")',
    src,
)
if n:
    print(f"{PATH}: acc_default 'none' path literal -> native string ({n})")

# The Nat-module sub keeps the concrete-only shape in this extraction (the
# shared transform's regex expects different indentation), so patch it here:
# symbolic operands must yield a symbolic [Sub] instead of crashing.
SUB_OLD = '''  let rec sub n m =
    match n with
    | O -> n
    | S k -> (match m with
              | O -> n
              | S l -> sub k l)'''
SUB_NEW = '''  let rec sub n m =
    match (n, m) with
    | (O, _) -> O
    | (x, O) -> x
    | (S k, S l) -> sub k l
    | _ -> Sub (n, m)'''
if SUB_OLD in src:
    src = src.replace(SUB_OLD, SUB_NEW)
    print(f"{PATH}: Nat-module sub made symbolic-aware")

OLD_START = "let padding2d n1 n2 d p e ="
OLD_END = "\n\n(** val padding3d"

i = src.index(OLD_START)
j = src.index(OLD_END, i)
old_block = src[i:j]

NEW_BLOCK = '''let padding2d n1 n2 d p e =
  mkvSeq (Ary ((add (mul (S (S O)) p) n2),
    (fle_self (add (mul (S (S O)) p) n2)), d)) (add (mul (S (S O)) p) n1)
    (fun i ->
    mkvSeq d (add (mul (S (S O)) p) n2) (fun j ->
      let i_val = fin2nat (add (mul (S (S O)) p) n1) i in
      let j_val = fin2nat (add (mul (S (S O)) p) n2) j in
      match d with
       | Num ->
         (* Symbolic indices: emit REif conditionals (printed as C ternaries)
            instead of deciding the bounds check statically. *)
         Obj.magic (REif ((BLt (i_val, p)), (Obj.magic (exp_default d)),
           REif ((BLt (i_val, (add p n1))),
             REif ((BLt (j_val, p)), (Obj.magic (exp_default d)),
               REif ((BLt (j_val, (add p n2))),
                 (Obj.magic (idx n2 d
                   (idx n1 (Ary (n2, (fle_self n2), d)) e
                     (nat2fin n1 (Nat.sub i_val p)))
                   (nat2fin n2 (Nat.sub j_val p)))),
                 (Obj.magic (exp_default d)))),
             (Obj.magic (exp_default d)))))
       | Ary (_, _, _) ->
         (match Nat.ltb_spec0 i_val p with
          | ReflectT -> exp_default d
          | ReflectF ->
            (match Nat.ltb_spec0 i_val (add p n1) with
             | ReflectT ->
               (match Nat.ltb_spec0 j_val p with
                | ReflectT -> exp_default d
                | ReflectF ->
                  (match Nat.ltb_spec0 j_val (add p n2) with
                   | ReflectT ->
                     idx n2 d
                       (idx n1 (Ary (n2, (fle_self n2), d)) e
                         (nat2fin n1 (Nat.sub i_val p)))
                       (nat2fin n2 (Nat.sub j_val p))
                   | ReflectF -> exp_default d))
             | ReflectF -> exp_default d))))'''

if "REif" not in old_block:
    src = src[:i] + NEW_BLOCK + src[j:]
    print(f"{PATH}: padding2d patched (Num -> REif conditionals)")
else:
    print(f"{PATH}: padding2d already patched")

with open(PATH, "w") as f:
    f.write(src)
