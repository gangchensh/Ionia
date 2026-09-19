(* Demo_Comms.v — Ionia tutorial demo operators (demo/README.md).
 *
 * Concrete instantiations of three route-A operators, extracted to OCaml
 * (demo_comm.ml) and consumed by the CUDA printer demo_to_cuda.ml:
 *
 *   demo_dot_comm  : Linear2d batch=1 m=8 outnum=1  — a dot product with bias
 *                    (the Linear2d cell is exactly
 *                     dot x w = sum (mkvSeq (fun i => mul x|[i] w|[i])));
 *                    correctness: Operators_Sem_Equiv.Linear2d_sem,
 *                    Bridge_AlexNet.Linear2d_sem_single.
 *   demo_relu_comm : ReLU4d 1x2x4x4 (CNN_Examples_Base_Ops.v, CG world — the
 *                    variant the code-generation pipeline consumes);
 *                    correctness: CNN_Examples_Base_Ops.ReLU4d_equiv_sem,
 *                    route-A twin Ops_Elementwise.A_ReLU4d.
 *   demo_conv_comm : Conv4d_p1_k3 1x1x8x8 -> 2 filters (same materialised-
 *                    padding variant as the AlexNet pipeline);
 *                    correctness: Ops_Conv.A_Conv4d (general pad_idx2d form),
 *                    Bridge_AlexNet.Conv4d_sem_single (Conv4d_ast_comm form).
 *
 * No proofs are done in this file; it only instantiates proved operators.
 *)

From Stdlib Require Import String.
Open Scope string_scope.

Require Import Ionia.CNN_Examples.
Require Import Ionia.CNN_CodeGen.

(* -------------------- Demo 0: dot product (Linear2d 1x8 -> 1x1) --------- *)

Definition dot_x : exp (ary '{1} (ary '{8} num)) := vrd "x".
Definition dot_w : exp (ary '{1} (ary '{8} num)) := vrd "w".
Definition dot_b : exp (ary '{1} num) := vrd "b".
Definition dot_output : acc (ary '{1} (ary '{1} num)) := vwt "output".

Definition demo_dot_comm : comm :=
  @CNN_Examples_Base_Ops.Linear2d_comm 1 8 1 dot_w dot_b dot_x dot_output.

(* -------------------- Demo 1: ReLU4d 1x2x4x4 ---------------------------- *)

Definition relu_input : exp (ary '{1} (ary '{2} (ary '{4} (ary '{4} num)))) :=
  vrd "input".
Definition relu_output : acc (ary '{1} (ary '{2} (ary '{4} (ary '{4} num)))) :=
  vwt "output".

Definition demo_relu_comm : comm :=
  @CNN_Examples_Base_Ops.ReLU4d_comm 1 2 4 4 relu_input relu_output.

(* -------------------- Demo 2: Conv4d p=1 k=3 s=1, 1x1x8x8 -> 1x2x8x8 ---- *)

Definition conv_input : exp (ary '{1} (ary '{1} (ary '{8} (ary '{8} num)))) :=
  vrd "input".
Definition conv_w : exp (ary '{2} (ary '{1} (ary '{3} (ary '{3} num)))) :=
  vrd "conv_w".
Definition conv_b : exp (ary '{2} num) := vrd "conv_b".
Definition conv_output : acc (ary '{1} (ary '{2} (ary '{8} (ary '{8} num)))) :=
  vwt "output".

Definition demo_conv_comm : comm :=
  @CNN_Examples_Base.Conv4d_p1_k3_comm 1 1 2 8 conv_input conv_w conv_b conv_output.

(* -------------------- Extraction ---------------------------------------- *)

Extraction "demo_comm.ml" demo_dot_comm demo_relu_comm demo_conv_comm.
