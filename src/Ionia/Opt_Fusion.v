(* G0 optimization-rule layer: fusion rules (route A / DPIA rule layer).

   Builds three families of fusion rewriting rules over the semantic model
   of Ionia.DPIA_Rules (comm := env -> env,
   env := path -> option (exp num)); all are Leibniz equations or
   observation-point equations, all closed with Qed, relying only on the
   standard classical axioms (functional extensionality / proof
   irrelevance / classic).

   1. map fusion (most general form): element-wise producer-consumer
      buffer elimination [map_fusion] -- the two-phase command that first
      uses mkvSeql to write f into a temporary buffer and then reads it
      back element-wise with bind_read, rewriting the output through g,
      is equivalent to directly computing
      A(a, mkvSeq (fun i => g (f i))),
      provided the temporary buffer is fresh (StructUndefined), the output
      and buffer paths are disjoint, and only the output path is observed
      (in_acc p a).
      Key lemmas:
      - [read_exp_mkvSeql]: after the producer loop finishes, reading
        cell i yields exactly f i (via
        Proof_CodeGen.access_seqfor_ary_eq_general, the seqfor
        generalization of the access_assign family on which
        LeNet5_DPIA.bind_read_replace rests);
      - [consumer_fusion]: the bind_read inside the consumer loop is
        replaced cell-wise by the direct value (loop-level generalization
        of the bind_read_replace family);
      - [access_assign_disjoint] / [seqfor_read_exp_stable]: the
        consumer's writes to the output do not disturb reads of the
        temporary buffer (disjointness premise made explicit).
   2. Conv4d + ReLU4d fusion (application level, route-A operators):
      [A_Conv4d_ReLU4d]: A(output, ReLU4d (Conv4d p s bias w input))
        = ConvReLU4d_fused_comm p s bias w input output,
      the fused kernel inlines the ReLU into the convolution output
      assignment (the continuation of the suml reduction carries the if
      branch), eliminating the intermediate feature-map buffer;
      [Conv4d_ReLU4d_buffer_elim] / [Conv4d_ReLU4d_comm_buffer_elim]:
      the two-phase form "write temporary buffer + bind_read read and
      rewrite" = the fused kernel.
   3. General mkvSeq chain-fusion rule: [mkvSeq_chain_fusion] /
      [A_mkvSeq_chain] -- two consecutive element-wise transformations
      = one composed transformation.
   Extra: [lenet5_conv1_relu1_fusion] -- the equation before/after fusing
      the LeNet-5 conv1+relu1 fragment (Conv3d_comm writing a temporary
      buffer + bind_read + ReLU3d_comm == the direct value-passing write
      A(output, ReLU3d (Conv3d ...))).

   Note: this file Requires LeNet5_Functional (whose LeNet5_Utils
   registers "|" as a keyword, which breaks parsing of the [| ] /
   [|{ }] notations), so the whole file follows the
   Ops_Elementwise / Ops_Conv convention of explicit idx / idxAcc. *)

Require Import Logic.FunctionalExtensionality.
Require Import List Reals String Nat Bool Arith Lra.
Require Import Coq.Logic.ProofIrrelevance.
Require Import Eqdep_dec.
Require Import Coq.Logic.Classical_Prop.
Import ListNotations.

Require Import Ionia.DPIA_Rules.
Require Import Ionia.Proof_CodeGen.
Import Proof_CodeGen.
Require Import Ionia.LeNet5_Functional.
Require Import Ionia.LeNet5_Imperative.
Require Import Ionia.LeNet5_DPIA.
Require Import Ionia.Ops_Elementwise.
Require Import Ionia.Ops_Conv.
From Stdlib Require Import Lia.

Set Implicit Arguments.

(* -------------------------------------------------------------------------- *)
(* 1. mkvSeq chain fusion (function layer): two consecutive element-wise  *)
(*    transformations = one composed transformation                            *)
(* -------------------------------------------------------------------------- *)

(* Building mkvSeq f first and then reading it back element-wise to apply g is equal to the single composition mkvSeq (g . f). *)
Lemma mkvSeq_chain_fusion : forall (n:nat) (d:data)
    (f : fin n -> exp d) (g : exp d -> exp d),
  mkvSeq (fun i => g (idx (mkvSeq f) i)) = mkvSeq (fun i => g (f i)).
Proof.
  intros n d f g. f_equal.
  apply functional_extensionality. intro i.
  destruct n.
  - inversion i; lia.
  - f_equal. rewrite idx_mkvSeq by lia. reflexivity.
Qed.

(* A-rule form: can be used directly as a rewrite rule. *)
Lemma A_mkvSeq_chain : forall (n:nat) (d:data)
    (f : fin n -> exp d) (g : exp d -> exp d) (a : acc (ary '{n} d)),
  A(a, mkvSeq (fun i => g (idx (mkvSeq f) i))) = A(a, mkvSeq (fun i => g (f i))).
Proof.
  intros. rewrite mkvSeq_chain_fusion. reflexivity.
Qed.

(* -------------------------------------------------------------------------- *)
(* 2. map fusion: element-wise producer-consumer buffer elimination        *)
(* -------------------------------------------------------------------------- *)

(* Writing to an accumulator b disjoint from c does not change reads of c
   (access level).  This is the access-level generalization of
   assign_not_in_acc. *)
Lemma access_assign_disjoint : forall (d2:data) (c : acc d2) (d1:data) (b : acc d1)
    (e : exp d1) (rho : env),
  (forall q, in_acc q b -> ~ in_acc q c) ->
  access (assign b e rho) c = access rho c.
Proof.
  intros d2; induction d2 as [| n0 m d2' IHd]; intros c d1 b e rho Hdisj.
  - rewrite !access_num. apply assign_not_in_acc. intro Hc.
    exact (Hdisj c Hc eq_refl).
  - rewrite !access_ary. apply mkoptvec_dependence. intro i.
    apply IHd. intros q Hb Hci.
    exact (Hdisj q Hb (ex_intro _ i Hci)).
Qed.

(* read_exp is the default wrapper of access: if the accesses are equal,
   the read_exp are equal.  Note that read_exp is a fixpoint recursing on
   the structure of d; for a generic d the fix does not reduce, so one
   must case-split on d before unfolding. *)
Lemma read_exp_congr : forall (d:data) (rho1 rho2 : env) (a : acc d),
  access rho1 a = access rho2 a -> read_exp rho1 a = read_exp rho2 a.
Proof.
  intros d; destruct d; intros rho1 rho2 a H; unfold read_exp; rewrite H; reflexivity.
Qed.

(* Preservation of disjoint writes at the read_exp level. *)
Lemma read_exp_assign_disjoint : forall (d2:data) (c : acc d2) (d1:data) (b : acc d1)
    (e : exp d1) (rho : env),
  (forall q, in_acc q b -> ~ in_acc q c) ->
  read_exp (assign b e rho) c = read_exp rho c.
Proof.
  intros d2 c d1 b e rho Hdisj.
  apply read_exp_congr. apply access_assign_disjoint. exact Hdisj.
Qed.

(* If every step command preserves reads of t, so does the whole seqfor. *)
Lemma seqfor_read_exp_stable : forall (m:nat) (c : fin m -> comm) (d:data)
    (t : acc d) (rho : env),
  (forall i rho', read_exp (c i rho') t = read_exp rho' t) ->
  read_exp (seqfor c rho) t = read_exp rho t.
Proof.
  intros m; induction m as [| m' IHm]; intros c d t rho H.
  - reflexivity.
  - rewrite seqfor_Sn. unfold seq.
    rewrite H. apply IHm. intros i rho'. apply H.
Qed.

(* read_exp is the default wrapper of access: when access hits Some, the read returns it directly. *)
Lemma read_exp_of_access : forall (d:data) (rho : env) (a : acc d) (e : exp d),
  access rho a = Some e -> read_exp rho a = e.
Proof.
  intros d; destruct d; intros rho a e H; unfold read_exp; rewrite H; reflexivity.
Qed.

(* After the producer loop (mkvSeql writing f cell by cell into the
   temporary buffer rooted at p) finishes, reading cell i yields exactly
   f i.  This is the loop-level generalization of the read_exp_assign
   lemma on which bind_read_replace (LeNet5_DPIA:274) rests; the core is
   given by access_seqfor_ary_eq_general (Proof_CodeGen). *)
Lemma read_exp_mkvSeql : forall (n:nat) (d:data) (f : fin n -> exp d)
    (p : path) (rho : env) (i : fin n),
  n > 0 ->
  read_exp (mkvSeql (fun j o => A(o, f j)) (@acc' (ary '{n} d) p) rho)
           (idxAcc (@acc' (ary '{n} d) p) i)
  = f i.
Proof.
  intros n d f p rho i Hn. apply read_exp_of_access.
  replace (idxAcc (@acc' (ary '{n} d) p) i) with (@acc' d (AryIdx i p)).
  2: { unfold idxAcc. rewrite acc'_ary. rewrite v_idx_mk_vec. reflexivity. }
  unfold mkvSeql, A. cbv beta iota.
  assert (Hprod : seqfor (fun j : fin n => assign (idxAcc (@acc' (ary '{n} d) p) j) (f j))
                = seqfor (fun j : fin n => assign (@acc' d (AryIdx j p)) (idx (mkvSeq f) j))).
  { apply seqfor_extensionality. intro j. cbv beta iota.
    unfold idxAcc. rewrite acc'_ary. rewrite v_idx_mk_vec.
    rewrite idx_mkvSeq by lia. reflexivity. }
  rewrite Hprod.
  replace (f i) with (idx (mkvSeq f) i).
  2: { apply idx_mkvSeq. exact Hn. }
  apply (@access_seqfor_ary_eq_general d p rho n n (fun j : fin n => j) (mkvSeq f)).
  - intros j k Hjk. exact Hjk.
  - intros p1 p2 e' x' H12. apply access_assign_neq. exact H12.
  - intros p1 e' x'. apply access_assign_eq. reflexivity.
Qed.

(* Consumer-loop fusion: if the read-side cell i is already f i in rho,
   and every write-side cell is path-disjoint from every read-side cell,
   then the loop that "reads back cell-wise with bind_read and rewrites
   through g" is equivalent to the loop that writes g (f i) directly.
   This is the loop-level generalization of the bind_read_replace family. *)
Lemma consumer_fusion : forall (n:nat) (d:data)
    (f : fin n -> exp d) (g : exp d -> exp d)
    (tacc wacc : fin n -> acc d) (rho : env),
  (forall i, read_exp rho (tacc i) = f i) ->
  (forall i j q, in_acc q (wacc i) -> ~ in_acc q (tacc j)) ->
  seqfor (fun i => bind_read (tacc i) (fun v => wacc i |:= g v)) rho
  = seqfor (fun i => wacc i |:= g (f i)) rho.
Proof.
  intros n; induction n as [| n' IHn]; intros d f g tacc wacc rho Hread Hdisj.
  - reflexivity.
  - rewrite !seqfor_Sn. cbv beta. unfold seq.
    assert (HIH : seqfor (fun i : fin n' => bind_read (tacc FS[i]) (fun v => wacc FS[i] |:= g v)) rho
                = seqfor (fun i : fin n' => wacc FS[i] |:= g (f FS[i])) rho).
    { apply (IHn d (fun i => f FS[i]) g (fun i => tacc FS[i]) (fun i => wacc FS[i]) rho).
      - intro i. apply Hread.
      - intros i j q Hw Hc. exact (Hdisj FS[i] FS[j] q Hw Hc). }
    rewrite HIH.
    unfold bind_read. cbv beta.
    assert (Hst : read_exp (seqfor (fun i : fin n' => wacc FS[i] |:= g (f FS[i])) rho) (tacc '[n'])
                = read_exp rho (tacc '[n'])).
    { apply seqfor_read_exp_stable. intros i rho'. cbv beta.
      apply read_exp_assign_disjoint.
      intros q Hw Hc. exact (Hdisj FS[i] '[n'] q Hw Hc). }
    rewrite Hst. rewrite (Hread '[n']). reflexivity.
Qed.

(* The command equation of map fusion (shared producer-prefix form):
   the two-phase program = the direct loop writing g (f i). *)
Lemma map_fusion_seq : forall (n:nat) (d:data)
    (f : fin n -> exp d) (g : exp d -> exp d)
    (tmp_path : path) (a : acc (ary '{n} d)),
  n > 0 ->
  (forall q, in_acc q a -> ~ in_acc q (@acc' (ary '{n} d) tmp_path)) ->
  seq (mkvSeql (fun i o => A(o, f i)) (@acc' (ary '{n} d) tmp_path))
      (seqfor (fun i => bind_read (idxAcc (@acc' (ary '{n} d) tmp_path) i)
                              (fun v => idxAcc a i |:= g v)))
  = seq (mkvSeql (fun i o => A(o, f i)) (@acc' (ary '{n} d) tmp_path))
        (mkvSeql (fun i o => A(o, g (f i))) a).
Proof.
  intros n d f g tmp_path a Hn Hdisj.
  apply functional_extensionality. intro rho. unfold seq.
  assert (Hcons :
    seqfor (fun i : fin n => bind_read (idxAcc (@acc' (ary '{n} d) tmp_path) i)
                               (fun v => idxAcc a i |:= g v))
           (mkvSeql (fun i o => A(o, f i)) (@acc' (ary '{n} d) tmp_path) rho)
    = seqfor (fun i : fin n => idxAcc a i |:= g (f i))
             (mkvSeql (fun i o => A(o, f i)) (@acc' (ary '{n} d) tmp_path) rho)).
  { apply (@consumer_fusion n d f g
             (fun i => idxAcc (@acc' (ary '{n} d) tmp_path) i)
             (fun i => idxAcc a i)).
    - intro i. apply (@read_exp_mkvSeql n d f tmp_path rho i Hn).
    - intros i j q Hw Hc. apply (Hdisj q).
      + exists i. exact Hw.
      + exists j. exact Hc. }
  rewrite Hcons. reflexivity.
Qed.

(* The map fusion rule (most general form): the two-phase program with
   new, provided the temporary buffer is fresh, the output and buffer
   paths are disjoint, and only the output path p is observed, is
   equivalent to A(a, mkvSeq (fun i => g (f i))).
   Note: since only the output path is observed, assign_pure_independent
   gives the conclusion directly; the StructUndefined (freshness) premise
   is kept explicit as required -- it is needed when observing the whole
   environment or when cleaning up the new. *)
Theorem map_fusion : forall (n:nat) (d:data)
    (f : fin n -> exp d) (g : exp d -> exp d)
    (tmp_path : path) (a : acc (ary '{n} d)) (rho : env) (p : path),
  n > 0 ->
  in_acc p a ->
  @StructUndefined (ary '{n} d) (@acc' (ary '{n} d) tmp_path) rho ->
  (forall q, in_acc q a -> ~ in_acc q (@acc' (ary '{n} d) tmp_path)) ->
  new (fun tmp : acc (ary '{n} d) =>
        seq (mkvSeql (fun i o => A(o, f i)) tmp)
            (seqfor (fun i => bind_read (idxAcc tmp i)
                              (fun v => idxAcc a i |:= g v))))
      (@acc' (ary '{n} d) tmp_path) rho p
  = A(a, mkvSeq (fun i => g (f i))) rho p.
Proof.
  intros n d f g tmp_path a rho p Hn Hp Hfresh Hdisj.
  unfold new. cbv beta.
  rewrite (@map_fusion_seq n d f g tmp_path a Hn Hdisj).
  rewrite <- (@AmkvSeq n d (fun i : fin n => g (f i)) a).
  unfold seq, A. apply assign_pure_independent. exact Hp.
Qed.

(* -------------------------------------------------------------------------- *)
(* 3. Conv4d + ReLU4d fusion (route-A application level)                      *)
(* -------------------------------------------------------------------------- *)

(* The per-output-cell command of the fused kernel: the continuation of
   the suml reduction directly carries the ReLU branch, and the
   convolution result is no longer written to an intermediate feature-map
   buffer. *)
Definition ConvReLU4dCell_comm {batch m1 m2 n1 n2 channel filernum : nat} (p s : nat)
  (bias : exp (ary '{filernum} num))
  (w : exp (ary '{filernum} (ary '{channel} (ary '{n1} (ary '{n2} num)))))
  (input : exp (ary '{batch} (ary '{channel} (ary '{m1} (ary '{m2} num)))))
  (b : fin batch)
  (cout : fin filernum)
  (i0 : fin ((m1 - n1 + 2 * p) / s + 1))
  (j0 : fin ((m2 - n2 + 2 * p) / s + 1))
  (output_cell : acc num)
  : comm :=
  suml (mkvSeq (fun c : fin channel =>
        sum (mkvSeq (fun i1 : fin n1 =>
          sum (mkvSeq (fun j1 : fin n2 =>
            mul (pad_idx2d p (idx (idx input b) c)
                           (fin2nat i0 * s + fin2nat i1)
                           (fin2nat j0 * s + fin2nat j1))
                (idx (idx (idx (idx w cout) c) i1) j1)))))))
       (fun r : exp num =>
          output_cell |:= (if leExp (add r (idx bias cout)) zero
                             then zero
                             else add r (idx bias cout)))
       output_cell.

Definition ConvReLU4d_fused_comm {batch m1 m2 n1 n2 channel filernum : nat} (p s : nat)
  (bias : exp (ary '{filernum} num))
  (w : exp (ary '{filernum} (ary '{channel} (ary '{n1} (ary '{n2} num)))))
  (input : exp (ary '{batch} (ary '{channel} (ary '{m1} (ary '{m2} num)))))
  (output : acc (ary '{batch} (ary '{filernum}
                (ary '{((m1 - n1 + 2 * p) / s + 1)}
                 (ary '{((m2 - n2 + 2 * p) / s + 1)} num)))))
  : comm :=
  mkvSeql (fun b ob => mkvSeql (fun cout oc =>
    mkvSeql (fun i0 o0 => mkvSeql (fun j0 o1 =>
      ConvReLU4dCell_comm p s bias w input b cout i0 j0 o1) o0) oc) ob) output.

(* Key scalar-level lemma: the direct assignment of relu (sum + bias) =
   the suml reduction skeleton with relu done in the final continuation.
   Since C(sum xs, c) is by definition c (sum xs), this goes directly
   through DPIA_Rules.C_sum_post_acc. *)
Lemma A_relu_add_sum : forall (n:nat) (xs : exp (ary '{n} num)) (bc : exp num)
    (a : acc num),
  A(a, if leExp (add (sum xs) bc) zero then zero else add (sum xs) bc)
  = suml xs (fun r : exp num =>
      a |:= (if leExp (add r bc) zero then zero else add r bc)) a.
Proof.
  intros n xs bc a.
  change (A(a, if leExp (add (sum xs) bc) zero then zero else add (sum xs) bc))
    with (C(sum xs, fun r : exp num =>
            a |:= (if leExp (add r bc) zero then zero else add r bc))).
  apply C_sum_post_acc.
Qed.

(* Application-level fusion rule: the direct assignment of
   ReLU4d . Conv4d = the fused-kernel command.  The proof shape follows
   Ops_Conv.A_Conv4d: AmkvSeq x4 + seqfor_extensionality to descend to
   the output cell, idx_mkvSeq x4 to unroll the mkvSeq chain of Conv4d
   into Conv4dCell, and finally A_relu_add_sum to inline the relu into
   the suml continuation. *)
Theorem A_Conv4d_ReLU4d : forall (batch m1 m2 n1 n2 channel filernum p s : nat)
  (Hb : batch > 0) (Hf : filernum > 0)
  (bias : exp (ary '{filernum} num))
  (w : exp (ary '{filernum} (ary '{channel} (ary '{n1} (ary '{n2} num)))))
  (input : exp (ary '{batch} (ary '{channel} (ary '{m1} (ary '{m2} num)))))
  (output : acc (ary '{batch} (ary '{filernum}
                (ary '{((m1 - n1 + 2 * p) / s + 1)}
                 (ary '{((m2 - n2 + 2 * p) / s + 1)} num))))),
  A(output, ReLU4d (Conv4d p s bias w input))
  = ConvReLU4d_fused_comm p s bias w input output.
Proof.
  intros batch m1 m2 n1 n2 channel filernum p s Hb Hf bias w input output.
  unfold ReLU4d, ConvReLU4d_fused_comm, ConvReLU4dCell_comm, Conv4d.
  rewrite AmkvSeq. apply seqfor_extensionality. intro b.
  rewrite AmkvSeq. apply seqfor_extensionality. intro cout.
  rewrite AmkvSeq. apply seqfor_extensionality. intro i0.
  rewrite AmkvSeq. apply seqfor_extensionality. intro j0.
  cbv beta.
  rewrite !idx_mkvSeq by lia.
  unfold Conv4dCell.
  rewrite A_relu_add_sum. reflexivity.
Qed.

(* The two-phase form (A writing the temporary buffer + bind_read
   reading it back and then A-ReLU4d) = the fused kernel:
   bind_read_replace replaces the read by value passing, then the
   application-level rule applies. *)
Theorem Conv4d_ReLU4d_buffer_elim : forall (batch m1 m2 n1 n2 channel filernum p s : nat)
  (Hb : batch > 0) (Hf : filernum > 0)
  (bias : exp (ary '{filernum} num))
  (w : exp (ary '{filernum} (ary '{channel} (ary '{n1} (ary '{n2} num)))))
  (input : exp (ary '{batch} (ary '{channel} (ary '{m1} (ary '{m2} num)))))
  (output : acc (ary '{batch} (ary '{filernum}
                (ary '{((m1 - n1 + 2 * p) / s + 1)}
                 (ary '{((m2 - n2 + 2 * p) / s + 1)} num)))))
  (tmp_path : path),
  seq (A(@acc' (ary '{batch} (ary '{filernum}
                (ary '{((m1 - n1 + 2 * p) / s + 1)}
                 (ary '{((m2 - n2 + 2 * p) / s + 1)} num)))) tmp_path,
        Conv4d p s bias w input))
      (bind_read (@acc' (ary '{batch} (ary '{filernum}
                (ary '{((m1 - n1 + 2 * p) / s + 1)}
                 (ary '{((m2 - n2 + 2 * p) / s + 1)} num)))) tmp_path)
                 (fun e => A(output, ReLU4d e)))
  = seq (A(@acc' (ary '{batch} (ary '{filernum}
                (ary '{((m1 - n1 + 2 * p) / s + 1)}
                 (ary '{((m2 - n2 + 2 * p) / s + 1)} num)))) tmp_path,
        Conv4d p s bias w input))
        (ConvReLU4d_fused_comm p s bias w input output).
Proof.
  intros batch m1 m2 n1 n2 channel filernum p s Hb Hf bias w input output tmp_path.
  rewrite bind_read_replace.
  rewrite A_Conv4d_ReLU4d by assumption.
  reflexivity.
Qed.

(* The two-phase form in the generated-code shape (Conv4d_comm writing
   the buffer + bind_read + ReLU4d_comm reading and rewriting) = the
   fused kernel: first fold back to the A form via A_Conv4d, then
   value-passing replacement, then inline the ReLU. *)
Theorem Conv4d_ReLU4d_comm_buffer_elim : forall (batch m1 m2 n1 n2 channel filernum p s : nat)
  (Hb : batch > 0) (Hf : filernum > 0)
  (bias : exp (ary '{filernum} num))
  (w : exp (ary '{filernum} (ary '{channel} (ary '{n1} (ary '{n2} num)))))
  (input : exp (ary '{batch} (ary '{channel} (ary '{m1} (ary '{m2} num)))))
  (output : acc (ary '{batch} (ary '{filernum}
                (ary '{((m1 - n1 + 2 * p) / s + 1)}
                 (ary '{((m2 - n2 + 2 * p) / s + 1)} num)))))
  (tmp_path : path),
  seq (Conv4d_comm p s bias w input
        (@acc' (ary '{batch} (ary '{filernum}
                (ary '{((m1 - n1 + 2 * p) / s + 1)}
                 (ary '{((m2 - n2 + 2 * p) / s + 1)} num)))) tmp_path))
      (bind_read (@acc' (ary '{batch} (ary '{filernum}
                (ary '{((m1 - n1 + 2 * p) / s + 1)}
                 (ary '{((m2 - n2 + 2 * p) / s + 1)} num)))) tmp_path)
                 (fun e => ReLU4d_comm e output))
  = seq (Conv4d_comm p s bias w input
          (@acc' (ary '{batch} (ary '{filernum}
                (ary '{((m1 - n1 + 2 * p) / s + 1)}
                 (ary '{((m2 - n2 + 2 * p) / s + 1)} num)))) tmp_path))
        (ConvReLU4d_fused_comm p s bias w input output).
Proof.
  intros batch m1 m2 n1 n2 channel filernum p s Hb Hf bias w input output tmp_path.
  rewrite <- A_Conv4d. rewrite bind_read_replace. rewrite <- A_ReLU4d.
  rewrite A_Conv4d_ReLU4d by assumption. reflexivity.
Qed.

(* -------------------------------------------------------------------------- *)
(* 4. Extra: the before/after-fusion equation of the LeNet-5 conv1+relu1    *)
(*    fragment                                                                *)
(* -------------------------------------------------------------------------- *)

(* Before fusion: Conv3d_comm writes the temporary feature-map buffer,
   which is read back with bind_read and then ReLU3d_comm writes the
   output; after fusion: the convolution result is passed directly by
   value (the "inlining" under the value semantics), and the right-hand
   side of the equation is exactly the DPIA normal form of the first
   layer of lenet5_comm after bind_read_replace. *)
Example lenet5_conv1_relu1_fusion : forall
  (input : exp (ary '{1} (ary '{28} (ary '{28} num))))
  (c1_w : exp (ary '{6} (ary '{1} (ary '{5} (ary '{5} num)))))
  (c1_b : exp (ary '{6} num))
  (tmp_path : path)
  (output : acc (ary '{6} (ary '{28} (ary '{28} num)))),
  seq (Conv3d_comm (p:=2) H5 c1_w c1_b input
        (@acc' (ary '{6} (ary '{28} (ary '{28} num))) tmp_path))
      (bind_read (@acc' (ary '{6} (ary '{28} (ary '{28} num))) tmp_path)
                 (fun e => ReLU3d_comm e output))
  = seq (A(@acc' (ary '{6} (ary '{28} (ary '{28} num))) tmp_path,
           Conv3d (p:=2) H5 c1_w c1_b input))
        (A(output, ReLU3d (Conv3d (p:=2) H5 c1_w c1_b input))).
Proof.
  intros. rewrite <- A_Conv3d. rewrite bind_read_replace. rewrite <- A_ReLU3d.
  reflexivity.
Qed.

(* -------------------------------------------------------------------------- *)
(* Axiom audit                                                                 *)
(* -------------------------------------------------------------------------- *)

Print Assumptions mkvSeq_chain_fusion.
Print Assumptions A_mkvSeq_chain.
Print Assumptions map_fusion.
Print Assumptions A_Conv4d_ReLU4d.
Print Assumptions Conv4d_ReLU4d_buffer_elim.
Print Assumptions Conv4d_ReLU4d_comm_buffer_elim.
Print Assumptions lenet5_conv1_relu1_fusion.
