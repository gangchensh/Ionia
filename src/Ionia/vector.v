(*
This version fix the problem of  idx_2d_to_1d and idx_1d_to_2d definition.
This problem make the correctness proof failed.
*)

(* # =================================================================== 
   # Vector Type
   #
   # Introduction
   #
   # We build a vector type based on pair.
   #
   # Two basic function: mk_vec v_idx
   #
   #   mk_vec : create a vector with a function from index to element
   #
   #   v_idx : access a element of array
   #
   # We define other functions on vector with the two functions above.
   #
   #   v_map v_reduce v_zip v_split v_join v_unzip1 v_unzip2
   #
   # There are some properties between these functions.
   #
   #  mk_vec_bij_fin: forall (m:nat) (x0 : A) (v : vector m),
   #    mk_vec (v_nth x0 v) = v.
   #
   #  v_nth_mk_vec: forall (n:nat) (f:fin n -> A) (x0:A) (i:fin n),
   #    i < n -> v_nth x0 (mk_vec f) i = f (nat2fin n i).
   #
   #  v_nth_map: forall (A B:Set) (n:nat) (f: A->B) (v:vector A n) (x0:A) (x1:B)(i:fin n),
   #    i < n -> v_nth x1 (v_map x0 f v) i = f (v_nth x0 v i).
   #
   #  v_map_zip : forall n (A B C:Set) (x0:A) (x1:B) (f:A->B->C) (v1:vector A n) (v2:vector B n),
   #    v_map (x0,x1) (cur f) (v_zip x0 x1 v1 v2) = mk_vec (fun i => f (v_nth x0 v1 i) (v_nth x1 v2 i)).
   #
   #  v_map_fg : forall n (A B C:Set) (x0:A) (x1:B) (f:A->B) (g:B->C) (v1:vector A n),
   #    v_map x1 g (v_map x0 f v1) = v_map x0 (fun a => g (f a)) v1.
   #
   #  v_map_map_split: forall (m n : nat) (A B:Set) (x0:vector A n) (x1:A) (f : A -> B) (v : vector A (m*n)),
   #    v_map x0 (fun v0 => v_map x1 f v0) (v_split x1 v) =
   #    mk_vec (fun i => mk_vec (fun j => f (v_idx x1 v (nat2fin (m*n) (i * n + j))))).
   #
   #  v_map_join : forall (m n : nat) (A B:Set) (x0:A) (x1: vector A n) (f : A -> B) (v : vector (vector A n) m),
   #    v_map x0 f (v_join x0 x1 v) 
   #    = mk_vec (fun i => f (v_idx x0 (v_idx x1 v (nat2fin m (i/n))) (nat2fin n (i mod n)))).
   #
   #  v_map_join_split: forall [m n:nat] (A B:Set) (f:A -> B) (x0:A)(x1:vector A n) (x2: B) (x3: vector B n)(v:vector A (m*n)),
   #    v_map x0 f v = v_join x2 x3 (v_map x1 (v_map x0 f) (v_split x0 v)).
   #
   #  v_split_join: forall (A:Set) (m n:nat) (x0:A) (x1:vector A n) (v:vector (vector A n) m),
   #    v_split x0 (v_join x0 x1 v) = v.
   #
   #  v_zip_map: forall (A B:Set) (n:nat) (f g: A -> B) (x0:A) (x1:B) (a b: vector A n),
   #    v_zip x1 x1 (v_map x0 f a) (v_map x0 g b) = v_map (x0,x0) (fun xy => (f (fst xy),g (snd xy))) (v_zip x0 x0 a b).
   #
   #  v_unzip1_mk_vec: forall (A B:Set) n (x0:A*B) f g,
   #    v_unzip1 x0 (mk_vec (fun i => (f i, g i))) = mk_vec (fun i:fin n => f i).
   #
   #  v_unzip2_mk_vec: forall (A B:Set) n (x0:A*B)f g,
   #    v_unzip2 x0 (mk_vec (fun i => (f i, g i))) = mk_vec (fun i: fin n => g i).
   #   
   # =================================================================== *)

Require Import Bool.Bool Logic.FunctionalExtensionality Nat.
Require Export finType.
Require Import Lia.
Require Import EqdepFacts.
Require Import Coq.Logic.ProofIrrelevance.
Require Import Compare_dec.

Set Implicit Arguments.
Import Nat.

Ltac fun_ext:= apply functional_extensionality; intros.

Notation "[ x1 , .. , xn ]" := (.. (tt,x1) .., xn). 

(* Boolean-expression simplification.
Normalizes equalities over Boolean values (true or false).
Converts Boolean natural-number comparisons (<? and =?) into mathematical
form (e.g. <, =, <>).
Splits Boolean conjunctions (&&) into simpler sub-hypotheses.
It recurses (via repeat and boolsimpl) so that every simplifiable
hypothesis is processed.

1. Symmetry handling
| [ H : (true = _) |- _ ] => symmetry in H
| [ H : (false = _) |- _ ] => symmetry in H
These two lines handle hypotheses H that are equalities with true or
false on the left-hand side.

symmetry in H reverses the equality H, e.g. turning true = x into
x = true, so that subsequent tactics can handle it more easily.

2. Natural-number comparison handling
| [ H : (_ <? _ = true)%nat |- _ ]  => apply Nat.ltb_lt in H
| [ H : (_ <? _ = false)%nat |- _ ] => apply Nat.ltb_nlt in H
| [ H : (_ =? _ = true)%nat |- _ ]  => apply Nat.eqb_eq in H
| [ H : (_ =? _ = false)%nat |- _ ] => apply Nat.eqb_neq in H
These lines handle hypotheses involving natural-number comparisons.

Nat.ltb_lt turns x <? y = true into x < y.
Nat.ltb_nlt turns x <? y = false into ~ (x < y).
Nat.eqb_eq turns x =? y = true into x = y.
Nat.eqb_neq turns x =? y = false into x <> y.
3. Boolean conjunction handling
| [ H : (_ && _ = true) |- _ ] => apply andb_true_iff in H;
                                  destruct H; boolsimpl
| [ H : (_ && _ = false) |- _ ] => apply andb_false_iff in H;
                                   destruct H; boolsimpl
These lines handle hypotheses involving Boolean conjunction (&&).

andb_true_iff turns x && y = true into x = true /\ y = true, and destruct H
then splits it into two hypotheses.
andb_false_iff turns x && y = false into x = false \/ y = false, and
destruct H then splits it into two subgoals.

After the split, boolsimpl calls itself recursively and keeps simplifying
the new hypotheses.


*)
Ltac boolsimpl :=
  repeat match goal with
         | [ H : (true = _) |- _ ] => symmetry in H
         | [ H : (false = _) |- _ ] => symmetry in H
         | [ H : (_ <? _ = true)%nat |- _ ] => apply Nat.ltb_lt in H
         | [ H : (_ <? _ = false)%nat |- _ ] => apply Nat.ltb_nlt in H
         | [ H : (_ =? _ = true)%nat |- _ ] => apply Nat.eqb_eq in H
         | [ H : (_ =? _ = false)%nat |- _ ] => apply Nat.eqb_neq in H

         | [ H : (_ && _ = true) |- _ ] => apply andb_true_iff in H;
                                           destruct H; boolsimpl
         | [ H : (_ && _ = false) |- _ ] => apply andb_false_iff in H;
                                            destruct H; boolsimpl
  end.

(*
This Ltac defines an automation tactic named [all] that handles many
kinds of proof goals.  It simplifies goals or applies suitable tactics
via pattern matching and recursive self-calls.  A detailed explanation
of each part:

1. Handling universal quantifiers (forall)
| |- forall _, _ => intros
If the current goal is a universal quantifier (forall), introduce the
variable with intros.

For example, the goal forall x, P x is simplified to P x, with x
introduced into the context.

2. Handling universal quantifiers in hypotheses
| H:forall _,_=_ |- _ => try (rewrite H; clear H)
If hypothesis H is a universally quantified equation (e.g.
forall x, f x = g x), try to apply it to the current goal with
rewrite H.

Then clear H to avoid reusing it.
The try ensures the tactic does not abort even if rewrite fails.

3. Handling natural-number equality tests (=?)
| |- context[ (?x =? ?m) ] =>
    let E := fresh "E" in
    destruct (x =? m) eqn:E; subst;
    boolsimpl; try lia
If the goal contains a natural-number equality test (x =? m), then:

destruct (x =? m) performs case analysis, producing two subgoals:
x =? m = true and x =? m = false.
eqn:E records the result of the test in a new hypothesis E.
subst tries to substitute known equalities.
The boolsimpl tactic (defined above) is called to simplify Boolean
expressions.
try lia tries to discharge any arithmetic goals.

4. Handling tuple equality
| |- ( _ , _) = ( _ , _)  => f_equal; repeat all
If the goal is an equality of two tuples (e.g. (a, b) = (c, d)), then:
  f_equal splits the goal into the subgoals a = c and b = d.
  repeat all recursively calls the all tactic on those subgoals.

5. Handling function equality
| |- (fun _ => _ ) = (fun _ => _)  => fun_ext
If the goal is an equality of two functions (e.g.
fun x => f x = fun x => g x), call the fun_ext tactic.
fun_ext is an application of the functional extensionality axiom; it
usually turns the goal into forall x, f x = g x.

6. Default case
| _ => auto with core ;
      auto with fcore;
      autorewrite with fcore;
      autorewrite with core
If none of the above cases match, try the following tactics:

auto with core: automated proof using the lemmas of the core database.
auto with fcore: automated proof using the lemmas of the custom fcore
database.
autorewrite with fcore: automated rewriting with the fcore rewrite rules.
autorewrite with core: automated rewriting with the core rewrite rules.

Summary
The all tactic is a multi-purpose automation tactic handling many kinds
of proof goals.  Its main features:

Handling universal quantifiers and equalities in hypotheses.
Splitting natural-number equality tests.
Handling tuple and function equalities.
Falling back to automated tactics and rewrite rules.

The tactic is deliberately general and applies in many situations,
especially proofs involving equalities, universal quantifiers and
functional extensionality.

*)
Ltac all:=
  match goal with
  | |- forall _, _ => intros
  | H:forall _,_=_ |- _ => try (rewrite H; clear H)
  | |- context[ (?x =? ?m) ] => 
      let E := fresh "E" in
      destruct (x =? m) eqn:E;subst;
      boolsimpl; try lia
  | |- ( _ , _) = ( _ , _)  => f_equal; repeat all
  | |- (fun _ => _ ) = (fun _ => _)  => fun_ext
  | _ => auto with core ;
  auto with fcore;
  autorewrite with fcore;
  autorewrite with core
  end.

(*
This Ltac defines an automation tactic named [f_simpl], mainly used to
simplify goals involving finite sets (fin), natural-number comparisons
(<?) and Boolean equalities.  It works via pattern matching, assertion
generation and repeated application of lemmas.  A detailed explanation
of each part:

1. Try unfolding ltS
2. Handling hypotheses of type fin (S n) and fle (S n)
match goal with
| x: fin (S ?n) |- _ => assert (H:x < S n); auto with fcore
| x: fle (S ?n) |- _ => assert (H:x <= S n); auto with fcore
| _ => idtac
end;
If the context has a hypothesis x : fin (S n) (x is an element of the
finite set S n), generate a new hypothesis H : x < S n and try to prove
it with auto with fcore.
If the context has a hypothesis x : fle (S n) (x is an upper-bounded
element of the finite set S n), generate a new hypothesis H : x <= S n
and try to prove it with auto with fcore.
If nothing matches, do nothing (idtac).


3. Handling Boolean equalities
repeat match goal with
       | |- false = _ => symmetry
       | |- true = _ => symmetry
       | |- _ <? _ = true => try apply Nat.ltb_lt; f_simpl
       | |- _ <? _ = false => try apply Nat.ltb_ge; f_simpl
       end;
This is a loop (repeat) that repeatedly processes Boolean equalities in
the goal:
If the goal is false = _ or true = _, use symmetry to reverse the
equality (e.g. turning false = x into x = false).
If the goal is _ <? _ = true, try applying Nat.ltb_lt to turn it into
the natural-number less-than relation (x < y), then recurse with f_simpl.
If the goal is _ <? _ = false, try applying Nat.ltb_ge to turn it into
the natural-number greater-or-equal relation (x >= y), then recurse with
f_simpl.

4. Try applying specific lemmas
try (apply fin_eq');
try (apply fin_neq');
try (apply fle_eq);
try (apply fle2nat_fle_self);

5. Automated rewriting and proving
autorewrite with fcore; auto with fcore;
try lia;
Use autorewrite with fcore to rewrite the goal with the rewrite rules of
the fcore database.
Use auto with fcore to try to prove the goal automatically.
Use try lia to try to discharge linear-arithmetic goals.

Summary
The main features of the f_simpl tactic:

Handling hypotheses about finite sets (fin) and upper bounds (fle),
generating new inequality hypotheses.
Simplifying Boolean equalities and natural-number comparisons (<?).
Applying specific lemmas (such as fin_eq', fle_eq) to simplify goals.
Finishing proofs with automated rewriting and proving tactics
(autorewrite, auto and lia).

This tactic is suited to proofs involving finite sets, natural-number
comparisons and Boolean equalities, and considerably reduces the amount
of manual work.
*)
Ltac f_simpl:=
  try unfold ltS;
  match goal with
  | x: fin (S ?n) |- _ => assert (H:x<S n);auto with fcore
  | x: fle (S ?n) |- _ => assert (H:x<=S n);auto with fcore
  | _ => idtac
  end;
  repeat match goal with
         | |- false = _ => symmetry
         | |- true = _ => symmetry
         | |- _ <? _ = true => try apply Nat.ltb_lt;f_simpl
         | |- _ <? _ = false => try apply Nat.ltb_ge;f_simpl
         end;
  try (apply fin_eq');
  try (apply fin_neq');
  try (apply fle_eq);
  try (apply fle2nat_fle_self);
  autorewrite with fcore;auto with fcore;
  try lia.
  
  
  
Ltac flt x m:=
  assert (x< S m);auto with fcore.

Section VectorType.

Variable A:Set.

(* definition of vector *)
Fixpoint vector (n : nat) : Set :=
  match n with
  | O => unit
  | S n => ((vector n) * A)%type
  end.

Fixpoint mk_vec {m:nat} : (fin m -> A) -> vector m:=
  match m with
  | 0 => fun ( _:_ ) => tt :vector 0
  | S m' => fun ( u: fin (S m') -> A) =>
      (mk_vec (fun i => u (fin2finS i)),
       u (exist _ m' (Nat.lt_succ_diag_r m')))
  end.

Fixpoint v_nth {n : nat} (x0 : A) : (vector n) -> nat -> A 
  := match n with
  | O => fun (_ : vector O) (_ : nat) => x0
  | S n' => fun (v : vector (S n')) (i : nat) => 
     if i =? n' then snd v else v_nth x0 (fst v) i
  end.

Definition v_idx {n: nat} (x0 : A) : (vector n) -> fin n -> A :=
  fun v i => v_nth x0 v i.

Section Facts.

(** One can prove that for 0 <= i < n, the value of v_nth does not depend on the default value *)
Lemma v_nth_x0_eq :
  forall (n i : nat) (v : vector n) (x0 x1 : A),
  0 < n -> i < n -> v_nth x0 v i = v_nth x1 v i.
Proof.
  intros n i v x0 x1 H H0. destruct n. lia. (* rule out the case v : vector 0 *) 
  induction n as [| n' IHn]. (* prove by induction starting from n = 1 *)
  - assert (i = 0). lia. subst; auto.
  - destruct v. simpl. destruct (i =? S n') eqn : E. (* case split: is i the last element of v : vector (S n)? *)
    + auto.
    + boolsimpl. apply IHn; lia.
Qed.

(** One can prove that for 0 < n, the value of v_idx does not depend on the default value *) 
Lemma v_idx_x0_eq :
  forall (n : nat) (i : fin n) (v : vector n) (x0 x1 : A),
  v_idx x0 v i = v_idx x1 v i.
Proof.
  intros n i v x0 x1. unfold v_idx. apply v_nth_x0_eq.
  - destruct n; try lia. exfalso. apply fin0_empty; auto. 
  - apply fin_proof.
Qed.
 
Fixpoint vector_eqb (A_eqb : A -> A -> bool) (n : nat) : vector n -> vector n -> bool :=  
  match n with  
  | O => fun _ _ => true (* vectors of length 0 are always equal *)  
  | S n' => fun v1 v2 =>  
      let (v1_tail, v1_head) := v1 in  
      let (v2_tail, v2_head) := v2 in  
      vector_eqb A_eqb n' v1_tail v2_tail && A_eqb v1_head v2_head  
  end.

Lemma vector_eqb_refl :  
  forall (A_eqb : A -> A -> bool),  
  (forall a, A_eqb a a = true) ->  
  forall (n : nat) (v : vector n),  
    vector_eqb A_eqb n v v = true.
Proof.  
  intros A_eqb A_eqb_refl.  
  induction n as [| n' IHn].  
  - (* Base case: n = 0 *)  
    intros v.  
    simpl. (* vector_eqb always returns true *)  
    reflexivity.  
  - (* Inductive case: n = S n' *)  
    intros v.  
    simpl. (* unfold the definition of vector_eqb *)  
    destruct v as [v_tail v_head]. (* split the vector into tail and head *)  
    rewrite IHn. (* apply the induction hypothesis to the tail vector *)  
    rewrite A_eqb_refl. (* apply reflexivity of A_eqb to the head element *)  
    reflexivity.  
Qed.

Lemma vector_eqb_sym :  
  forall (A_eqb : A -> A -> bool),  
  (forall a b, A_eqb a b = A_eqb b a) ->  
  forall (n : nat) (v1 v2 : vector n),  
    vector_eqb A_eqb n v1 v2 = vector_eqb A_eqb n v2 v1.
Proof.  
  intros A_eqb A_eqb_sym.  
  induction n as [| n' IHn].  
  - (* Base case: n = 0 *)  
    intros v1 v2.  
    simpl. (* vector_eqb always returns true *)  
    reflexivity.  
  - (* Inductive case: n = S n' *)  
    intros v1 v2.  
    simpl. (* unfold the definition of vector_eqb *)  
    destruct v1 as [v1_tail v1_head].  
    destruct v2 as [v2_tail v2_head]. (* split the vector into tail and head *)  
    rewrite IHn. (* apply the induction hypothesis to the tail vector *)  
    rewrite A_eqb_sym. (* apply symmetry of A_eqb to the head element *)  
    reflexivity.  
Qed.

Lemma vector_eqb_trans :  
  forall (A_eqb : A -> A -> bool),  
  (forall a b c, A_eqb a b = true -> A_eqb b c = true -> A_eqb a c = true) ->  
  forall (n : nat) (v1 v2 v3 : vector n),  
    vector_eqb A_eqb n v1 v2 = true ->  
    vector_eqb A_eqb n v2 v3 = true ->  
    vector_eqb A_eqb n v1 v3 = true.
Proof.  
  intros  A_eqb A_eqb_trans.  
  induction n as [| n' IHn].  
  - (* Base case: n = 0 *)  
    intros v1 v2 v3.  
    simpl. (* vector_eqb always returns true *)  
    intros _ _. (* the hypotheses are satisfied *)  
    reflexivity.  
  - (* Inductive case: n = S n' *)  
    intros v1 v2 v3.  
    simpl. (* unfold the definition of vector_eqb *)  
    destruct v1 as [v1_tail v1_head].  
    destruct v2 as [v2_tail v2_head].  
    destruct v3 as [v3_tail v3_head]. (* split the vector into tail and head *)  
    intros H12 H23. (* hypotheses: both vector_eqb comparisons (tail and head) are true *)  
    apply andb_true_iff in H12 as [H12_tail H12_head]. (* split H12 into tail and head parts *)  
    apply andb_true_iff in H23 as [H23_tail H23_head]. (* split H23 into tail and head parts *)  
    apply andb_true_iff. (* the goal is to show that both tail and head are true *)  
    split.  
    + (* tail: recurse with IHn *)  
      apply IHn with (v2 := v2_tail);auto.  
    + (* head: use transitivity of A_eqb *)  
      apply A_eqb_trans with (b := v2_head); assumption.  
Qed.

Lemma vector_eqb_correct :  
  forall (A_eqb : A -> A -> bool),  
  (forall a1 a2, A_eqb a1 a2 = true <-> a1 = a2) ->  
  forall (n : nat) (v1 v2 : vector n),  
    vector_eqb A_eqb n v1 v2 = true <-> v1 = v2.  
Proof.  
  intros A_eqb A_eqb_correct.  
  induction n as [| n' IHn].  
  - (* Base case: n = 0 *)  
    intros v1 v2.  
    simpl. (* vector_eqb always returns true for n = 0 *)  
    split.
    +  destruct v1,v2.  reflexivity.
    + reflexivity. (* unit type has only one value, so v1 = v2 *)  
  - (* Inductive case: n = S n' *)  
    intros v1 v2.  
    simpl. (* Expand vector_eqb for n = S n' *)  
    destruct v1 as [v1_tail v1_head].  
    destruct v2 as [v2_tail v2_head].  
    split.  
    + (* -> direction: vector_eqb = true implies v1 = v2 *)  
      intros H.  
      apply andb_true_iff in H as [H_tail H_head]. (* Split the conjunction *)  
      apply IHn in H_tail. (* Apply the induction hypothesis to the tails *)  
      apply A_eqb_correct in H_head. (* Apply the correctness of A_eqb to the heads *)  
      subst. (* Substitute equalities *)  
      reflexivity.  
    + (* <- direction: v1 = v2 implies vector_eqb = true *)  
      intros H.  
      inversion H. (* Decompose the equality v1 = v2 *)  
      apply IHn in H1. (* Apply the induction hypothesis to the tails *)  
      apply A_eqb_correct in H2. (* Apply the correctness of A_eqb to the heads *)  
      assert(A_eqb_refl: forall (a:A), A_eqb a a = true).
      intro a.  rewrite A_eqb_correct. auto.
      rewrite vector_eqb_refl. rewrite A_eqb_refl. auto.
      apply A_eqb_refl.  
Qed.  

(* Decidability of vector_eqb *)  
Lemma vector_eqb_dec :   
  forall (A_eqb : A -> A -> bool) (n : nat) (v1 v2 : vector n),  
  {vector_eqb A_eqb n v1 v2 = true} + {vector_eqb A_eqb n v1 v2 = false}.  
Proof.  
  intros A_eqb n.  
  induction n as [| n' IHn'].  
  - (* Base case: n = 0 *)  
    intros v1 v2.  
    simpl. (* vector_eqb A_eqb 0 v1 v2 = true *)  
    left. reflexivity.  

  - (* Inductive case: n = S n' *)  
    intros v1 v2.  
    simpl. (* vector_eqb A_eqb (S n') v1 v2 = vector_eqb A_eqb n' (fst v1) (fst v2) && A_eqb (snd v1) (snd v2) *)  
    destruct (IHn' (fst v1) (fst v2)) as [H_tail | H_tail];  
    destruct (A_eqb (snd v1) (snd v2)) eqn:H_head;
    destruct v1,v2; simpl in H_head,H_tail.  rewrite H_tail, H_head.
    + (* Both tail and head are equal *)  
      left.   reflexivity.  
    + (* Tail is equal, but head is not *)  
      right. rewrite H_tail, H_head. reflexivity.  
    + (* Tail is not equal, but head is equal *)  
      right. rewrite H_tail, H_head. reflexivity.  
    + (* Both tail and head are not equal *)  
      right. rewrite H_tail, H_head. reflexivity.  
Qed.


(* If f1 = f2 pointwise, then mk_vec f1 = mk_vec f2. *)
Lemma mk_vec_eq :  
  forall (m : nat) (f1 f2 : fin m -> A),  
    (forall i : fin m, f1 i = f2 i) ->  
    mk_vec f1 = mk_vec f2.
Proof.  
  intros m f1 f2 H.  
  induction m as [| m' IH].  
  - (* Base case: m = 0 *)  
    simpl. reflexivity. (* `mk_vec` for `m = 0` always returns `tt`, so the result is trivial *)  
  - (* Inductive case: m = S m' *)  
    simpl.  
    f_equal.  
    + (* Prove equality of the "rest" of the vector *)  
      apply IH.  
      intros i.  
      specialize (H (fin2finS i)).  
      assumption.  
    + (* Prove equality of the last element *)  
      specialize (H (exist _ m' (Nat.lt_succ_diag_r m'))).  
      assumption.  
Qed.

(* When constructing a vector A (S m), the result of mk_vec equals:
the vector formed by the first m elements: mk_vec (fun i => u (fin2finS i))
plus the last element: u (exist ...), i.e. u m.
This restates the definition of mk_vec, but as a lemma it provides a
basis for inductive reasoning, making it easy to rewrite mk_vec u in
later proofs.
*)
Lemma mk_vec_Sn : forall (m:nat) (u : fin (S m) -> A),
  mk_vec u = (mk_vec (fun (i:fin m) => u (fin2finS i)),
  u (exist _ m (Nat.lt_succ_diag_r m))).
Proof. auto. Qed.

Lemma v_nth_Sn_simpl : forall m (a: vector m) b x x0,
  @v_nth (S m) x0 (a,b) x = if x =? m then b else v_nth x0 a x.
Proof. auto. Qed.

Lemma mk_vec_Sn_x0 : forall (m:nat) (x0:A),
  mk_vec(fun i:fin (S m) => x0) = (mk_vec (fun i => x0), x0).
Proof. auto. Qed.

(* #[export] Hint Rewrite mk_vec_Sn v_nth_Sn_simpl: core. *)

Lemma v_nth_Sn : forall m (a: vector m) b x x0,
  x < m -> @v_nth (S m) x0 (a,b) x = v_nth x0 a x.
Proof. 
  repeat all. simpl. repeat all.
Qed.

Lemma v_nth_x : forall x (a: vector x) b x0,
  @v_nth (S x) x0 (a,b) x = b.
Proof. 
  simpl. repeat all. 
Qed.

Lemma v_nth_0: forall (m:nat) (x0 e: A) (v : vector m),
  m <> 0 -> @v_nth (S m) x0 (v,e) 0 = v_nth x0 v 0.
Proof. 
  intros. simpl. destruct m. exfalso. apply H. auto. auto.  
Qed.

Lemma v_nth_tt : forall (a: vector 0) x x0,
  v_nth x0 a x = x0.
Proof. 
  intros. simpl in a. simpl. auto.
Qed.



Hint Rewrite v_nth_x v_nth_Sn_simpl: core.

Ltac v_mk_simpl:=
  match goal with
  | |- @mk_vec (S _) _ = @mk_vec (S _) _ => f_equal;fun_ext;v_mk_simpl
  | _ => idtac;
  rewrite ?mk_vec_Sn 
  end.

Ltac v_nth_simpl:=
  autorewrite with core;
  rewrite ?v_nth_Sn_simpl;auto with fcore; autorewrite with fcore;try lia.


Lemma v_nth_x_fin : forall x (a: vector x) b x0 (i: fin (S x)),
  fin2nat i = x -> @v_nth (S x) x0 (a,b) x = b.
Proof.
  repeat all.
Qed.

Hint Rewrite v_nth_x_fin: core.

Lemma mk_vec_bij_fin: forall (m:nat) (x0 : A) (v : vector m),
  mk_vec (v_nth x0 v) = v.
Proof. 
  induction m,v;auto. v_mk_simpl; repeat all. rewrite <- IHm with(x0:=x0).
  induction m; auto.  v_mk_simpl. repeat all. flt x m. lia.
Qed.

Lemma fin2finS_nat2fin : forall (n i:nat) (H:i<S n) (Hi_lt_n:i<n),
   fin2finS (nat2fin Hi_lt_n) = nat2fin H.
Proof. intros. unfold fin2finS. unfold nat2fin. (* [ | [ | i]] = [ | i] *)
 simpl. apply exist_eq.
Qed.

(* Indexing element i of the vector mk_vec f is equivalent to computing the value at index i directly from the function f. *)
Lemma v_nth_mk_vec: forall  (n:nat) (i:nat) (f:fin n -> A) (x0:A) (H : i < n) ,
   v_nth x0 (mk_vec f) i = f (nat2fin  H).
Proof.
  induction n; intros.
  - exfalso. apply (Nat.nlt_0_r i). exact H.
  - v_mk_simpl. v_nth_simpl. all.
    + f_equal.
      apply fin_eq with (n:=S n)(i1:=exist (fun i : nat => i < S n) n (Nat.lt_succ_diag_r n)).
      f_simpl.
    + assert (Hi_lt_n: i < n).
      { apply (proj1 (Nat.lt_succ_r i n)) in H.
        destruct (le_lt_eq_dec i n H) as [Hlt | Heq].
        - exact Hlt.
        - exfalso. apply E. exact Heq. }
      specialize(IHn i (fun i0 : fin n => f (fin2finS i0)) x0 Hi_lt_n).
      rewrite IHn. f_equal. apply fin2finS_nat2fin .
Qed.

Lemma fin2finS_eq : forall (n':nat) (i_val:nat) (i_proof : i_val < S n'),
  exist (fun x => x < S (S n')) i_val (Nat.lt_lt_succ_r _ _ i_proof) =
  fin2finS (exist (fun x => x < S n') i_val i_proof).
Proof.
  intros n' i_val i_proof.
  (* unfold the definition of fin2finS *)
  unfold fin2finS.
  (* the goal is now exist _ i_val (Nat.lt_lt_succ_r _ _ i_proof) = exist _ i_val (Nat.lt_lt_succ_r _ _ i_proof) *)
  reflexivity. (* the goal holds *)
Qed.

Lemma v_idx_mk_vec :  
  forall (n : nat) (i : fin n) (f : fin n -> A) (x0 : A),  
    v_idx x0 (mk_vec f) i = f i.  
Proof.  
  intros n [k pf] f x0.    unfold v_idx.  
  (* We now invoke the lemma that says “the nth element of (mk_vec f)   
     at index k is f (exist k pf)”. *)  
  specialize (v_nth_mk_vec f x0 pf) as H.  
  (* That lemma might look like: [v_nth (mk_vec f) k = f (exist k pf)]. *)  
  simpl. rewrite H. reflexivity.  
Qed.

End Facts.
End VectorType.

Notation "F[ i | n ]" := (@nat2fin n i).
Notation "F[ i ]" := (nat2fin i).

#[export] Hint Rewrite v_nth_mk_vec mk_vec_bij_fin :core.

Arguments mk_vec {A} {m}.
Arguments v_nth {A} {n}.

#[export] Hint Rewrite (* mk_vec_Sn v_nth_Sn_simpl *) v_nth_x:core.

Section VecFun.

Definition cur {A B C:Set} (f: A -> B -> C) :=
  fun xy : A * B => f (fst xy) (snd xy).

Definition uncur {A B C:Set} (f: A*B -> C) :=
  fun x y => f (x,y).
  
Definition v_make {A : Set} (n:nat) (x0: A): vector A n:=
  mk_vec (fun i:fin n => x0).

(* mapping one vector to another *)
Definition v_map {A B : Set} {n : nat} (x0:A) (f : A -> B) (v:vector A n) : vector B n :=
  mk_vec (fun x => f (v_idx x0 v x)).

(* fold a vector to an element from left to right *)
Fixpoint v_reduce {A B : Set} {n : nat} (f : A -> B -> B) (init_val:B)
  : (vector A n) -> B :=
  match n with
  | O => fun (v : vector A 0) => init_val
  | S n' => fun (v : vector A (S n')) => 
    f (snd v) (v_reduce f init_val (fst v))
  end.

Compute v_reduce Nat.add 0 ([0, 1, 2, 3] : vector nat 4). (* = 3+(2+(1+(0+0))) = 6 *)

(* iterate i from n-1 to 0. *)
Fixpoint n_reduce_aux {B:Set} (f : nat->B->B) (a:B) (i:nat) : B :=
  match i with
  | 0 => a
  | S i' => @n_reduce_aux B f (f i' a) i'
  end.



Definition n_reduce {B:Set} {n:nat} (f:nat->B->B) (a:B) : B :=
  @n_reduce_aux B f a n.

Compute @n_reduce nat 4 Nat.add 0. (* = 0+(1+(2+(3+0)))) = 6 *)

Lemma n_reduce_aux_base :
  forall {B : Set} {n : nat} (f : nat -> B -> B) (a : B),
    @n_reduce_aux B f a 0 = a.
Proof.  intros B f a.  reflexivity. Qed.

Lemma n_reduce_aux_rec :
  forall {B : Set} {n : nat} (f : nat -> B -> B) (a : B) (i : nat),
    @n_reduce_aux B f a (S i) = @n_reduce_aux B f (f i a) i.
Proof.  intros B f a i. simpl. reflexivity. Qed.

Lemma n_reduce_aux_init :
  forall (f : nat->bool) (i:nat) (a:bool) ,
     n_reduce_aux (fun i y => y && f i) a i = true -> a=true.
Proof. 
  induction i. 
+ simpl. auto. 
+ intro a. simpl. intro H. specialize(IHi (a && f i)).
  assert(a && f i = true). { apply IHi. exact H. } 
  apply andb_prop in H0. tauto.
Qed.

(* return true if for all i< n, f i = true. *)
Definition n_forall {n:nat} (f : nat -> bool) : bool :=
    @n_reduce bool n (fun (i:nat) (y:bool) => y && f i) true.

Lemma n_forall_r  {n:nat} (f : nat -> bool) : 
  (forall i : fin n, f i = true) -> @n_forall n f = true.
Proof. unfold n_forall, n_reduce. induction n.
+ intro H.  simpl. reflexivity.
+ intro H. simpl.
  assert(Hn: f n = true). {
   specialize(H (fin_last n)). exact H.
  }  
  rewrite Hn. apply IHn.
  intro i.  apply (H (fin2finS i)).
Qed.


Lemma n_forall_l  {n:nat} (f : nat -> bool) : 
  @n_forall n f = true -> forall (i:fin n), f i = true.
Proof.
  unfold n_forall, n_reduce. induction n.
+ intros. inversion i. inversion  H0.
+ intros H i.                            (* Introduce hypotheses *)
  simpl in H.
  destruct i.
  destruct (eq_nat_dec x n) as [-> | Hneq].
  - pose(g:=fun i y => y && f i).
    (* f [ | n] = true *)
    apply (n_reduce_aux_init f n). apply H.
  - assert(x < n) by lia.
    assert(H1: f n = true).
    apply (n_reduce_aux_init f n). apply H.
    rewrite H1 in H.
    specialize(IHn H).
    specialize(IHn (nat2fin H0)). exact IHn.
Qed.

Lemma n_forall_correct  {n:nat} (f : nat -> bool) : 
  (forall i : fin n, f i = true) <-> @n_forall n f = true.
Proof. split; [ apply n_forall_r | apply n_forall_l ]. Qed.


(* convert a function over fin n and B to a function on nat. *)
Definition finf2natf2 {B : Set} {n : nat} (f : fin n -> B -> B) : nat -> B -> B :=
  fun (i : nat) (a : B) =>
    match lt_dec i n with
    | left H => f (exist _ i H) a (* i < n, so use f with the valid index *)
    | right _ => a (* i >= n, return the accumulator unchanged *)
    end.

(* correctness of finf2natf: (finf2natf f) = f. *)
Lemma finf2natf2_agree_valid :
  forall {B : Set} {n : nat} (f : fin n -> B -> B) (i : fin n) (a : B),
    finf2natf2 f (fin2nat i) a = f i a.
Proof.
  intros B n f i a.
  unfold finf2natf2.
  destruct i as [i H]. (* Destruct i into its underlying nat and proof H : i < n *)
  destruct ( lt_dec (@fin2nat n (@exist nat (fun i0 : nat => Peano.lt i0 n) i H)) n) as [H' | H']. (* Case analysis on i < n *)
  - (* Case: i < n *)
    f_equal. simpl. apply exist_eq.
  - (* Case: i >= n *)
    exfalso. (* Contradiction: H states i < n, but H' states i >= n *)
    apply H'. (* H:i<n |- [ | i] < n *)
    exact H.
Qed.

(* convert a single fin n argument function to a function over nat. *)
Definition finf2natf1 {B:Set} {n:nat} (f : fin n->B) (a:B) : nat->B :=
  fun (i : nat) =>
    match lt_dec i n with
    | left H => f (exist _ i H) (* i < n, so use f with the valid index *)
    | right _ => a (* i >= n, return the accumulator unchanged *)
    end.

(* change n_reduce to i_reduce by declaring f over fin n. *)
Definition i_reduce_aux {B:Set} {n:nat} (f:fin n->B->B) (a:B) (i:fin n) : B :=
  @n_reduce_aux B (finf2natf2 f) a i .   

Definition i_reduce {B:Set} {n:nat} (f:fin n->B->B) (a:B): B :=
  @n_reduce B n (finf2natf2 f) a .   

Lemma i_reduce_aux_eq_n_reduce :
  forall {B : Set} {n : nat} (f : fin n -> B -> B) (a : B) (i : fin n),
    @i_reduce_aux B n f a i = @n_reduce_aux B (finf2natf2 f) a (fin2nat i).
Proof. intros B n f a i. reflexivity. Qed.

Lemma i_reduce_eq_n_reduce :
  forall {B : Set} {n : nat} (f : fin n -> B -> B) (a : B) ,
    @i_reduce B n f a = @n_reduce B n (finf2natf2 f) a.
Proof. unfold i_reduce, n_reduce. intros B n f a. reflexivity. Qed.

(* return true if for all i< n, f i = true. *)
Definition i_forall {n:nat} (f : fin n -> bool) : bool :=
    @i_reduce bool n (fun (i:fin n) (y:bool) => y && f i) true.

(* alternative definition of i_forall using n_forall. *)
Definition i_forall' {n:nat} (f : fin n -> bool) : bool :=
    @n_forall n (finf2natf1 f true).

(* equivalence of two definitions of i_forall's *)
Lemma i_forall_equiv :  forall (n:nat) (f : fin n -> bool),
    i_forall f = i_forall' f.
Proof.
  intros n f.
  unfold i_forall, i_forall', i_reduce, n_forall, finf2natf1.
  (* Simplify the definitions *)
  assert (H: forall i : fin n, 
           finf2natf2 (fun (i : fin n) (y : bool) => y && f i) (fin2nat i) true  
          = true && f i).
  {
    intros i.
    unfold finf2natf2.
    destruct (lt_dec (fin2nat i) n) as [H | H].
    - (* Case: fin2nat i < n *)
      simpl. f_equal. destruct i. simpl. apply exist_eq.
    - (* Case: fin2nat i >= n *)
      exfalso. apply H. apply fin_proof.
  }
  (* Use the assertion to rewrite the goal *)
  unfold n_reduce. unfold finf2natf2. 
  f_equal. fun_ext. fun_ext.
  destruct(lt_dec x n).
  - reflexivity.
  - rewrite andb_true_r.  reflexivity. 
Qed.

Lemma i_forall_r  {n:nat} (f : fin n -> bool) : 
  (forall i : fin n, f i = true) -> @i_forall n f = true.
Proof. rewrite i_forall_equiv. unfold i_forall'. 
rewrite<- n_forall_correct. intros H i.
unfold finf2natf1. 
destruct(lt_dec i n).
+ specialize(H i). destruct i. simpl.
  rewrite<- H.  f_equal.  apply exist_eq. 
+ reflexivity.
Qed.

Lemma i_forall_l  {n:nat} (f : fin n -> bool) : 
  @i_forall n f = true -> forall (i:fin n), f i = true.
Proof. rewrite i_forall_equiv. unfold i_forall'. 
rewrite<- n_forall_correct. intros H i.
specialize(H i). unfold finf2natf1 in H.
destruct(lt_dec i n).
+ destruct i. simpl in H.
  rewrite<- H.  f_equal.  apply exist_eq. 
+ destruct i. destruct n0. exact l.
Qed.

Theorem i_forall_correct  {n:nat} (f : fin n -> bool) : 
  (forall i : fin n, f i = true) <-> @i_forall n f = true.
Proof. split; [ apply i_forall_r | apply i_forall_l ]. Qed.

(* if f a = None then false else true. *)
Definition opt2b {A B:Set} (f:A->option B) (a:A) :=
   match f a with
    | None => false
    | Some _ => true
   end.

(* extend f: fin i -> A to a function on fin (S i) -> A. *)
Definition predf {A:Set} (m:nat) (f : fin (S m) -> A)
  : (fin m -> A) := fun (i : fin m) => f (fin2finS i).

Lemma predf_ok  {A:Set} (m:nat) (f : fin (S m) -> A): forall (i:fin m),
    predf f i = f (fin2finS i).
Proof. intro i. unfold predf. reflexivity. Qed.

Fixpoint v_max {A: Set} {n : nat} (f : A -> A -> bool) (init_val:A)
  : (vector A n) -> A :=
  match n with
  | O => fun (v : vector A 0) => init_val
  | S n' => fun (v : vector A (S n')) => 
            if (f (snd v) (v_max f init_val (fst v))) 
            then (snd v) 
            else (v_max f init_val (fst v))
  end.

(* combine two vector into a vector of pairs *)
Definition v_zip {A B:Set} {n:nat} (x0:A) (x1:B) (v1:vector A n) (v2:vector B n) : vector (A*B) n:=
  mk_vec (fun x => (v_idx x0 v1 x,v_idx x1 v2 x)).

Definition v_unzip1 {A B:Set} {n:nat} (x0:A*B) (v: vector (A*B) n): vector A n :=
  mk_vec (fun x => fst (v_idx x0 v x)).

Definition v_unzip2 {A B:Set} {n:nat} (x0:A*B) (v: vector (A*B) n): vector B n :=
  mk_vec (fun x => snd (v_idx x0 v x)).

Lemma i_n_plus_j_lt_m_n : forall m n i j : nat,  
  i < m ->  
  j < n ->  
  i * n + j < m * n.  
Proof.  
  intros m n i j Hi Hj.  
  (* step 1: show i*n + j < i*n + n *)  
  apply Nat.lt_le_trans with (m := i * n + n).  
  - apply Nat.add_lt_mono_l.  
    exact Hj.          (* j < n ⇒ i*n + j < i*n + n *)  
  - (* step 2: show i*n + n <= m*n *)  
    (* rewrite i*n + n as (i+1)*n *)  
    replace (i * n + n) with ((i + 1) * n) by lia.  
    (* because i < m => i+1 <= m => (i+1)*n <= m*n *)  
    apply Nat.mul_le_mono_r.  
    lia.  
Qed.


Definition idx_2d_to_1d {m n : nat} (i : fin m) (j : fin n) : fin (m * n) :=
  let i' := fin2nat i in let j' := fin2nat j in
  let idx := i' * n + j' in
  let H : idx < m * n := i_n_plus_j_lt_m_n (fin_proof i) (fin_proof j) in
      exist _ idx H.

(*
v_split transforms a vector of length m * n into a nested vector of dimensions m x n. 
For a given flat vector v and indices i : fin m and j : fin n, the element at position (i, j) in the resulting matrix is:
result[i][j]=v[idx_2d_to_1d(i,j)]
*)
Definition v_split {A:Set} {m n:nat} (x0 : A) (v:vector A (m*n)) : vector (vector A n) m :=
  mk_vec (fun (i:fin m) => mk_vec (fun (j:fin n) => 
      @v_idx A (m*n) x0 v (idx_2d_to_1d i j))).
(*
Definition v_split {A:Set} {m n:nat} (x0 : A) (v:vector A (m*n)) : vector (vector A n) m :=
  mk_vec (fun (i:fin m) => mk_vec (fun (j:fin n) => 
      @v_idx A (m*n) x0 v (idx_2d_to_1d i j))).
*)
 
Fixpoint vec_length {A : Set} {n : nat} : (vector A n) -> nat :=
  match n with
  | O    => fun (v:vector A 0) => 0
  | S n' => fun (v:vector A (S n')) =>
             let (hd,tl) := v in S (vec_length hd)                
  end.

Lemma vec_length_correct : forall (A : Set) (n : nat) (v : vector A n),
  vec_length v = n.
Proof.
  intros A n v.
  induction n as [| n' IH].
  - (* Case n = 0 *)
    destruct v. reflexivity.
  - (* Case n = S n' *)
    destruct v as [hd tl].
    simpl. rewrite IH.  reflexivity.
Qed.

Lemma vec_length_mk_vec :  
  forall (A : Set) (n : nat) (f : fin n -> A),  
    vec_length (mk_vec f) = n.  
Proof.  
  intros A n.  
  induction n as [| n' IH]; intros f.  
  - (* Base case: n = 0 *)  
    (* By definition, [mk_vec f] is [tt : unit], so [vec_length] is 0. *)  
    simpl. reflexivity.  
  - (* Inductive case: n = S n' *)  
    (* Typically, [mk_vec f] is [(mk_vec f', x)] for a suitable [f']. *)  
    simpl.  
    (* The [simpl] or [unfold] will reveal [vec_length (fst (mk_vec f))]. *)  
    rewrite IH.  
    reflexivity.  
Qed.

Lemma v_split_length_outer : forall A m n x0 (v : vector A (m*n)),  
  vec_length (v_split x0 v) = m.  
Proof.  
  intros A m n x0 v.  
  unfold v_split.  
  (* By definition, v_split is mk_vec with domain fin m,   
     so its length is m. *)  
  rewrite vec_length_mk_vec.  
  reflexivity.  
Qed.

Definition default_vec {A : Set} (n : nat) (x0 : A) : vector A n :=  
  mk_vec (fun _ => x0).
  
Lemma v_split_length_inner :  
  forall (A : Set) (m n : nat) (x0 : A)  
         (v : vector A (m * n)) (v0 : vector A n) (i : fin m),  
    vec_length (v_idx v0 (v_split x0 v) i) = n.  
Proof.  
  intros A m n x0 v v' i.  
  unfold v_split.           (* Expands to [mk_vec (fun i => mk_vec (fun j => ...))]. *)  
  rewrite v_idx_mk_vec.     (* Indexing into [mk_vec f] gives [f i]. *)  
  (* Now the result is [mk_vec (fun j => v_idx A (m * n) x0 v (idx_2d_to_1d i j))],  
     which is a vector of length [n]. *)  
  apply vec_length_mk_vec.  
Qed.

(* The elements of the original vector should be correctly mapped to the corresponding positions in the resulting matrix *)
Lemma v_split_element_correct :  
  forall (A : Set) (m n : nat) (x0 : A)  
         (v : vector A (m * n)) (v0 : vector A n) (i : fin m) (j : fin n),  
    v_idx x0 (v_idx v0 (v_split x0 v) i) j  
    = v_idx x0 v (idx_2d_to_1d i j).  
Proof.  
  intros A m n x0 v v0 i j.  
  unfold v_split.   
  (* v_split x0 v = mk_vec (fun i => mk_vec (fun j => v_idx x0 v (idx_2d_to_1d i j))) *)  
  rewrite v_idx_mk_vec.  
  (* Rewriting the outer call to [v_idx] eliminates the outer mk_vec   
     and leaves us with [mk_vec (fun j => v_idx x0 v (idx_2d_to_1d i j))]. *)  
  rewrite v_idx_mk_vec.  
  (* This eliminates the inner mk_vec at index j, yielding [v_idx x0 v (idx_2d_to_1d i j)]. *)  
  reflexivity.  
Qed.

(* Index Mapping Correctness *)
Lemma idx_2d_to_1d_correct :  forall m n (i : fin m) (j : fin n),  
    idx_2d_to_1d i j < m * n.  
Proof.  
  intros m n [i Hi] [j Hj].  
  (* Now [i] and [j] are natural numbers with proofs [i < m] and [j < n]. *)  
  unfold idx_2d_to_1d.  
  (* Typically, idx_2d_to_1d i j = i * n + j. *)  
  simpl.  
  (* Since i < m and j < n, we want to show i*n + j < m*n. *)  
  apply i_n_plus_j_lt_m_n ; assumption.
Qed.

(* v_split on edge cases, such as when m = 0 or n = 0 *)
Lemma v_split_empty_m :  
  forall (A : Set) (n : nat) (x0 : A) (v : vector A (0 * n)),  
    v_split x0 v = mk_vec (fun (i : fin 0) =>  
      mk_vec (fun (j : fin n) => x0)).
Proof. intros. simpl. destruct v. unfold v_split.  simpl.  reflexivity.
Qed. 
 
Lemma mk_vec_ext :  
  forall (A : Set) (n : nat) (f g : fin n -> A),  
    (forall i : fin n, f i = g i) ->  
    mk_vec f = mk_vec g.  
Proof.  
  intros A n f g Heq.  
  (* If mk_vec is literally "f", then they’re equal by functional extensionality. *)  
  apply functional_extensionality_dep in Heq.
  rewrite Heq. reflexivity.  
Qed.

Lemma v_split_empty_n :  
  forall (A : Set) (m : nat) (x0 : A) (v : vector A (m * 0)),  
    v_split x0 v = mk_vec (fun (i : fin m) =>  
      mk_vec (fun (j : fin 0) => x0)).  
Proof.  
  intros A m x0 v.  
  unfold v_split.  
  (* v_split x0 v = mk_vec (fun (i : fin m) =>  
       mk_vec (fun (j : fin 0) => v_idx x0 v (idx_2d_to_1d i j))). *)  

  (* Since the inner index j : fin 0 doesn't exist, both sides  
     will have an empty inner vector. We show they are equal  
     by pointwise equality (outer mk_vec). *)  

  apply mk_vec_ext.  (* Vector extensionality: compare elements at each i. *)  
  intros i.  
  apply mk_vec_ext.  (* Compare elements at each j. *)  
  intros j.  
  (* Here j : fin 0, which is impossible. *)  
  inversion j.  (* Contradiction, so the equality holds vacuously. *)  
  inversion H.
Qed. 


Lemma div_lt_3var : forall x m n, x < m * n -> n > 0 -> x / n < m.
Proof.
  intros x m n Hx Hn.
  apply Div0.div_lt_upper_bound ;  lia.
Qed.

(* prove x mod n < n *)
Lemma mod_upper_bound : forall x n, n > 0 -> x mod n < n.
Proof.
  intros x n Hn.
  apply Nat.mod_upper_bound; lia.
Qed.

(* auxiliary lemma: fin 0 is an empty type *)
Lemma fin_0_absurd : forall {A : Type} (x : fin 0), A.
Proof.
  intros A x.
  destruct x. lia.  (* fin 0 has no constructors, so this is a contradiction *) 
Qed.

(* convert m*0 to 0, then call fin_0_absurd. *)
Definition fin_mult_0_absurd (m : nat) (x : fin (m*0)) : False :=  
  fin_0_absurd (eq_rect (m*0) (fun t => fin t) x 0 (Nat.mul_0_r m)).

Definition idx_1d_to_2d {m n : nat} : fin (m * n) -> fin m * fin n :=  
  match n as n0 return fin (m * n0) -> fin m * fin n0 with  
  | 0 =>  
      fun (x : fin (m * 0)) =>  
        False_rect (fin m * fin 0) (fin_mult_0_absurd m x)  
  | S n' =>  
      fun (x : fin (m * S n')) =>  
        match x with  
        | exist _ i Hi =>  
            let row := i / (S n') in  
            let col := i mod (S n') in  

            let Hn_neq_0 : S n' <> 0 := Nat.neq_succ_0 n' in  

            (* use eq_rect to turn Hi : i < m*(S n') into i < (S n')*m *)  
            let Hi_rewritten :=  
              eq_rect (m * S n') (fun z => i < z) Hi ((S n') * m)  
                      (Nat.mul_comm m (S n'))  
            in  
            (* now Hi_rewritten : i < (S n') * m, so div_lt_upper_bound applies *)  
            let Hr : row < m :=  
              Div0.div_lt_upper_bound i (S n') m Hi_rewritten  
            in  
            let Hc : col < S n' := Nat.mod_upper_bound i (S n') Hn_neq_0 in  

            (exist _ row Hr, exist _ col Hc)  
        end  
  end.

Lemma idx_1d_to_2d_correct : forall m n (x : fin (m * n)),  
  let (col, row) := idx_1d_to_2d x in col < m /\ row < n.
Proof.
  intros m n x.  
  (* Destructure the pair returned by idx_1d_to_2d. *)  
  destruct (idx_1d_to_2d x) as [col row].  
  (* Because col has type fin m, it’s guaranteed that `col < m`.   
     Similarly, row has type fin n, so `row < n`. *)  
  split; [exact (proj2_sig col) | exact (proj2_sig row)].  
Qed.

Lemma div_eq_fst_divmodS : forall i m, div i (S m) = fst (divmod i m 0 m).
Proof. intros. reflexivity. Qed.

Lemma div_eq_fst_divmod0 : forall i , div i 0 = 0.
Proof. intros. reflexivity. Qed.

Lemma div_lt_n : forall (m' n i : nat),  
  i < S m' * n ->  i / S m' < n.  
Proof.  
  intros m' n i H.  
  destruct n.  
  - (* n = 0 → premise i < S m'*0 is impossible for i:nat *)  
    exfalso; lia.  
  - (* n = S n' *)  
    apply Div0.div_lt_upper_bound.  
    + (* positivity: 0 < S m' *)  
      lia.  
Qed.

    

Lemma exist_equal :  
  forall (A : Type) (P : A -> Prop) (x y : A) (H:x=y) (px : P x) (py : P y),  
     exist P x px = exist P y py.  
Proof.  
  intros A P x y H. rewrite H.
  intros px py. f_equal.                  (* it suffices to show px = py *)  
  apply proof_irrelevance.  (* px = py follows from proof irrelevance *)  
Qed.  

(*************************************************************)  
(* 1. idx_1d_to_2d (idx_2d_to_1d col row) = (col, row)      *)  
(************************************************************)  

Lemma idx_1d_to_2d_after_idx_2d_to_1d :  
  forall (m n : nat) (row : fin m) (col : fin n),  
    idx_1d_to_2d (idx_2d_to_1d row col) = (row, col).  
Proof. 
  intros m n [col_val col_pf] [row_val row_pf].  
  unfold idx_2d_to_1d.  
  simpl.  
  (* Case analysis on n *)  
  destruct n as [| n'].  
  - (* n = 0 ⇒ contradiction, because col < 0 cannot happen *)  
    exfalso. lia.  
  - (* n = S n' ⇒ use the div / mod characterization *)  
    destruct (exist (fun i => i < m * S n')  
         (col_val * (S n') + row_val) (i_n_plus_j_lt_m_n col_pf row_pf))
      as [idx idx_pf] eqn:E.
    unfold idx_1d_to_2d.
    inversion E. (* H0 : col_val * n + row_val = idx |- *)
    clear E. 
    (* unfold the computation of idx_1d_to_2d *)
    assert (Hrow : (col_val * S n' + row_val) / S n' = col_val).
    {
      rewrite Nat.add_comm. (* Swap col_val * S n' and row_val *)
      rewrite Nat.div_add. (* Use Nat.div_add_exact *)
      - rewrite Nat.div_small.
        + reflexivity; assumption.
        + assumption.
      - lia. 
    }   
      assert (Hcol : idx mod (S n') = row_val).
     {
      rewrite <- H0. rewrite Nat.add_comm.
      rewrite Div0.mod_add.
      - rewrite Nat.mod_small.
        + reflexivity.
        + assumption.
     }
    f_equal.
    assert (Hr: idx / (S n') = col_val).
    {  rewrite <- H0. apply Hrow. }

    * (* [ | idx / S n'] = [ | col_val] *)
      apply exist_equal. apply Hr.
    * (* [ | idx mod S n'] = [ | row_val] *)
      apply exist_equal. apply Hcol.
Qed.

Lemma idx_2d_to_1d_after_idx_1d_to_2d :  
  forall (m n : nat) (idx : fin (m * n)),  
    idx_2d_to_1d (fst (idx_1d_to_2d idx)) (snd (idx_1d_to_2d idx)) = idx.
Proof.
  intros m n [idx idx_pf].
  unfold idx_1d_to_2d.
  (* Case analysis on n *)
  destruct n as [| n'].
  - (* n = 0 ⇒ contradiction, because idx < m * 0 cannot happen *)
    exfalso. rewrite Nat.mul_0_r in idx_pf. lia.
  - (* n = S n' ⇒ use the div / mod characterization *)
    (* Decompose idx_1d_to_2d *)
    set (row := idx / (S n')).
    set (col := idx mod (S n')).
    assert (Hrow : row < m).
    {
      apply Div0.div_lt_upper_bound.
      - rewrite Nat.mul_comm in idx_pf.
        assumption.
    }
    assert (Hcol : col < S n').
    {
      apply Nat.mod_upper_bound.
      apply Nat.neq_succ_0.
    }
    (* Now compute idx_2d_to_1d *)
    unfold idx_2d_to_1d. simpl.
    (* Show that row * S n' + col = idx *)
    apply exist_equal.
  (* Now the goal is:
     S n' * ((S n' * row + col) / n') + (n' - (S n' * row + col) mod n') =
     S n' * row + col
  *)
  unfold row, col. rewrite Nat.mul_comm.
  rewrite <- Nat.div_mod_eq with (x := idx) (y := S n').
  reflexivity.
Qed.

(* Takes a nested vector v : vector (vector A n) m and flattens it into a vector of type vector A (m * n). *)
Definition v_join {A:Set} {m n:nat} (x0:A) 
  (v:vector (vector A n) m) : vector A (m * n) :=
  mk_vec (fun (x:fin (m * n)) =>
    let (i, j) := idx_1d_to_2d x in  (* convert the 1-D index x to a 2-D index (i, j) *)
    v_idx x0 (v_nth (default_vec n x0) v i) j         (* extract the element from the nested vector *)
  ).
(*
Definition v_join {A:Set} {m n:nat} (x0:A) (x1:vector A n) 
  (v:vector (vector A n) m) : vector A (m * n):=
  mk_vec (fun x => v_idx x0 (v_nth x1 v F[ x/n | m ]) F[ x mod n | n]).
*)
 
Lemma v_nth_last : forall A n (v : vector A n) (a x0 : A),
  v_nth x0 ((v,a):(vector A (S n)) )
      (fin2nat (exist _ n (Nat.lt_succ_diag_r n))) = a.
Proof.
  intros A n v a x0.
  (* Unfold v_nth *)
  unfold v_nth.
  (* Simplify fin2nat (exist _ n (Div0.lt_succ_diag_r n)) to n *)
  simpl.
  (* Simplify the if-then-else expression *)
  destruct (n =? n) eqn:H.
  - (* Case: n = n *)
    reflexivity. 
  - (* Case: n <> n *)
    apply Nat.eqb_neq in H. contradiction.
Qed.

Theorem vector_eq_v_nth : forall A n (v1 v2 : vector A n) x0,
  v1 = v2 <-> (forall i : fin n, v_nth x0 v1 i = v_nth x0 v2 i).
Proof.
intros H.
induction n as [| n' IH].
+ (* Case n = 0 *)
  - destruct v1, v2. intro. split; reflexivity.
+ (* Case n = S n' *)
  - destruct v1 as [v1' a1], v2 as [v2' a2]. intro x0.
    split.
    * intro Hva. inversion Hva. intro i.
      subst a1 v1'. reflexivity.
    * intro H0.  (* (v1', a1) = (v2', a2) *)
      assert(a1=a2).
      {       
        specialize (H0 (exist _ n' (Nat.lt_succ_diag_r n'))).
        unfold v_idx in H0. 
        repeat rewrite v_nth_last with (n:=n') in H0. exact H0. 
      }
     assert(v1'=v2') as Hv.
     {
       rewrite IH with (x0:=x0). intros i.
       specialize (H0 (fin2finS i)).
       simpl in H0.
      (* Case analysis on i =? n' *)
        destruct (i =? n') eqn:H2 in H0.
        - (* Case: i = n' *)
          (* Rewrite H0 using the fact that i = n' *)
          rewrite Nat.eqb_eq in H2.  
          destruct i. simpl in H2. lia.
        - (* Case: i <> n' *) exact H0. 
      }
      rewrite H1, Hv. reflexivity.
Qed.

Lemma vector_eq_v_idx : forall A n (v1 v2 : vector A n) x0,
  v1 = v2 <-> (forall i : fin n, v_idx x0 v1 i = v_idx x0 v2 i).
Proof. unfold v_idx. apply vector_eq_v_nth. Qed.

Lemma v_join_unfold : forall (A : Set) (m n : nat) (x0 : A)
  (v : vector (vector A n) m) (x : fin (m * n)),
    v_nth x0 (v_join x0 v) x =
    let '(i, j) := idx_1d_to_2d x in
        v_nth x0 (v_nth (default_vec n x0) v i) j.
Proof.
  intros A m n x0 v x.
  unfold v_join. (* v_join x0 v = mk_vec (fun x => ...) *)
  simpl.
  (* 1) decompose x into a natural number i together with its boundedness proof H *)
  set (i := fin2nat x).
  set (H := fin_proof x). (* i < m*n *)
  (* 2) call v_nth_mk_vec, passing (i := i)(H := H) explicitly *)
  rewrite v_nth_mk_vec with (H:=H).
  assert (F_H_eq_x : F[H] = x). {
  (* unfold the definition of F[H] *)
  unfold nat2fin.
  (* the goal is now exist _ x H = x *)
  destruct x as [x' H'].
  (* since x is exist _ x' H' and H and H' are the same proof, they are directly equal *)
  reflexivity.
  }
  (* replace F[H] with x *)
  rewrite F_H_eq_x. reflexivity.
Qed.

Lemma v_nth_v_split_nats :
  forall (A : Set) (m n : nat) (x0 : A) (v : vector A (m*n))
   (i j : nat) (Hi : i < m) (Hj : j < n),
    v_nth x0 (v_nth (default_vec n x0)  (v_split x0 v) i) j 
  = v_nth x0 v (i*n + j).
Proof.
  intros A m n x0 v i j Hi Hj.
  unfold v_split; simpl.
  (* v_split x0 v = mk_vec (λ i, mk_vec (λ j, v_idx x0 v (idx_2d_to_1d i j))). *)

  (* step 1: take the i-th element of the outer mk_vec *)
  (* using the lemma v_nth_mk_vec: v_nth x0 (mk_vec f) i = f (exist (fun i0 => i0 < length) i Hi) *)
  rewrite v_nth_mk_vec with (i := i) (H := Hi). simpl.
  (* step 2: take the j-th element of the result (i.e. of mk_vec (fun j => ...)) *)
  rewrite v_nth_mk_vec with (i := j) (H := Hj). simpl.
  (* v_idx x0 v (idx_2d_to_1d ...) => v_nth x0 v (i*n + j) *)
  reflexivity.
Qed.

Lemma v_nth_v_split :
  forall (A : Set) (m n : nat) (x0 : A) (v : vector A (m * n)) (i : fin m) (j : fin n),
    v_nth x0 (v_nth (default_vec n x0) (v_split x0 v) i) j =
    v_nth x0 v (idx_2d_to_1d i j).
Proof.
  intros A m n x0 v i j.
  destruct i,j. apply v_nth_v_split_nats.
  exact l. exact l0.
Qed.

Lemma v_split_round_trip :
  forall (A : Set) (m n : nat) (x0 : A) (v : vector A (m * n)),
    v_join x0 (v_split x0 v) = v.
Proof.
  intros A m n x0 v.
  rewrite vector_eq_v_nth with (x0:=x0). (* turn vector equality into pointwise equality *)
  (* v_join is generated with mk_vec; first unfold mk_vec, *)
  (* then idx_1d_to_2d idx yields the index pair (i, j) *)
  intro idx. remember (idx_1d_to_2d idx) as ij eqn:E.
  destruct ij as [i j]. rewrite v_join_unfold . rewrite <- E.
  (* v_split x0 v is also generated with mk_vec; each i corresponds to a "subvector" *)
  (* each subvector is again generated with mk_vec; each j corresponds to the element of v at index idx_2d_to_1d i j *)
  (* so a single simplification step shows the extracted element equals the original vector at idx_2d_to_1d i j *)

  (* now use idx_2d_to_1d_after_idx_1d_to_2d: from E we know idx = idx_2d_to_1d i j *)
  rewrite <- (idx_2d_to_1d_after_idx_1d_to_2d m n idx).
  rewrite<- E. simpl.
  apply v_nth_v_split.
Qed.

Lemma pow_0_mul_1: forall i j,
  i ^ O * j = j.
Proof. intros. rewrite Nat.pow_0_r. apply Nat.mul_1_l.  Qed.

Lemma pow_Sn_mul : forall n i j,
  i ^ S n * j = i * (i ^ n * j).
Proof. intros. rewrite Nat.pow_succ_r'. rewrite Nat.mul_assoc. auto. Qed.

Fixpoint v_iterate {A:Set} {i j:nat} (n: nat) (f:forall m, vector A (i*m) -> vector A m) 
  :vector A ((i^n)*j) -> vector A j :=
  match n with
  | O => fun (v:vector A (i^O*j)) => eq_rec (i^O*j) (vector A) v j (pow_0_mul_1 i j) 
  | S n' => fun v => v_iterate n' f (f ((i^ n')*j) 
    (eq_rec (i ^ S n' * j) (vector A) v (i * (i ^ n' * j)) (pow_Sn_mul n' i j)))
  end.

(* Lemma to combine ~ (val < m) and val < m + n into val - m < n *)
Lemma offset_proof_concise : forall val m n,
  val < m + n -> ~ (val < m) -> val - m < n.
Proof.  intros val m n H_lt H_ge. lia. Qed.

Definition v_concat {A : Set} {m n : nat} (x0 : A)  
           (v1 : vector A m) (v2 : vector A n) : vector A (m + n) :=  
  mk_vec (fun (i : fin (m + n)) =>  
    let val := fin2nat i in  
    match Nat.ltb_spec0 val m with (* if val < m *)
    | ReflectT _ H_lt => v_idx x0 v1 (nat2fin H_lt)
    | ReflectF _ H_ge =>  
        (* Use offset_proof_concise to directly prove val - m < n *)
        let H_lt_sub := @offset_proof_concise val m n (fin_proof i) H_ge in
        v_idx x0 v2 (nat2fin H_lt_sub)
    end
  ).
(*
Definition v_concat {A:Set} {m n: nat} (x0:A) (v1:vector A m) (v2:vector A n): vector A (m+n) :=
  mk_vec (fun (i:fin (m+n)) => 
    if i<?m then (v_idx x0 v1 F[ i ] ) else (v_idx x0 v2 F[ i-m ])).
*)

Definition v_padl {A : Set} {n : nat} (x0 : A) (k : nat) (v : vector A n) : vector A (k + n) :=
  mk_vec (fun (i : fin (k + n)) =>
    let val := fin2nat i in
    match Nat.ltb_spec0 val k with (* if val < k *)
    | ReflectT _ H_lt => x0
    | ReflectF _ H_ge =>
        (* Use offset_proof_concise to directly prove val - k < n *)
        let H_lt_sub := @offset_proof_concise val k n (fin_proof i) H_ge in
        v_idx x0 v (nat2fin H_lt_sub)
    end
  ).
(*
Definition v_padl {A:Set} {n:nat} (x0:A) (k:nat) (v:vector A n): vector A (k+n):=
  mk_vec (fun i => if i<?k then x0 else v_idx x0 v F[ i-k ]).
*)

Definition v_padr {A : Set} {n : nat} (x0 : A) (k : nat) (v : vector A n) : vector A (n + k) :=
  mk_vec (fun (i : fin (n + k)) =>
    let val := fin2nat i in
    match Nat.ltb_spec0 val n with (* if val < n *)
    | ReflectT _ H_lt => v_idx x0 v (nat2fin H_lt)
    | ReflectF _ H_ge => x0
    end
  ). 
(*
Definition v_padr {A:Set} {n:nat} (x0:A) (k:nat) (v:vector A n) : vector A (n+k):=
  mk_vec (fun i => if i<?n then v_idx x0 v F[ i ] else x0).
*)

(* v_idx_sub *)
Definition v_idx_sub {A : Set} {n : nat} (x0 : A) (v : vector A n) 
  {k : nat} (i : fin k) (H : k <= n) : A :=
  let val := fin2nat i in
  let H_lt : i<n := Nat.lt_le_trans (fin2nat i) k n (fin_proof i) H in
    v_idx x0 v (nat2fin H_lt).

(** One can prove that for 0 < n, the value of v_idx_sub does not depend on the default value *)
Lemma v_idx_sub_x0_eq :
  forall {A : Set} (n k : nat) (v : vector A n) (x0 x1 : A) (i : fin k) (H : k <= n),
  v_idx_sub x0 v i H = v_idx_sub x1 v i H.
Proof.
  intros A n k v x0 x1 i H. apply v_idx_x0_eq.
Qed.

(* take the first k elements of a vector v of length n, producing a new vector of length k *)
Definition v_truncl {A : Set} {n : nat} (x0 : A) (k : nat) 
  (H_le : k <= n) (v : vector A n) : vector A k :=
     mk_vec (fun (i : fin k) => v_idx_sub x0 v i H_le).
(*
Definition v_truncl {A:Set} {n:nat} (x0:A) (k:nat) (v:vector A n) : vector A k :=
  mk_vec (fun i => v_idx x0 v F[ i ]).
*)

Definition v_idx_sub_offset {A : Set} {n : nat} (x0 : A) (v : vector A n) 
  {k : nat} (i : fin k) (offset : nat) (H : k + offset <= n) : A :=
  let val := fin2nat i + offset in
  let H_lt_k_offset : val < k + offset := 
    proj1 (Nat.add_lt_mono_r (fin2nat i) k offset) (fin_proof i) in
  let H_lt_n : val < n := Nat.lt_le_trans val (k+offset) n H_lt_k_offset H in
    v_idx x0 v (nat2fin H_lt_n). 

Lemma offset_add_le {n k : nat} (H_le : k <= n) : k + (n-k) <= n.
Proof. rewrite Nat.add_comm. lia. Qed.

(* take the last k elements of a vector v of length n, producing a new vector of length k *)
Definition v_truncr {A : Set} {n : nat} (x0 : A) (k : nat) 
  (H_le : k <= n) (v : vector A n) : vector A k :=
    mk_vec (fun (i : fin k) =>
        v_idx_sub_offset x0 v i (n-k) (offset_add_le H_le)).
(*
Definition v_truncr {A:Set} {n:nat} (x0:A) (k:nat) (v:vector A n) : vector A k :=
  mk_vec (fun i => v_idx x0 v F[ i+n-k ]).
*)


Definition v_trans {A:Set} {n m:nat} (x0:A)(v:vector (vector A m) n) : vector (vector A n) m :=
  mk_vec (fun i:fin m => mk_vec (fun j:fin n => v_idx x0 (v_idx (default_vec m x0) v j) i)).



End VecFun.

#[export] Hint Unfold v_map v_zip v_unzip1 v_unzip2 cur uncur v_idx v_split v_join
  v_concat v_padl v_padr v_truncl v_truncr v_trans : core.

Lemma v_reduce_Sn: forall {A B : Set} {n : nat} (f : A -> B -> B) (init_val:B) (a:vector A n)(b:A),
  v_reduce f init_val ((a,b):vector A (S n)) = f b (v_reduce f init_val a).
Proof. auto. Qed.

Lemma v_max_Sn: forall {A : Set} {n:nat} (f: A -> A -> bool) (v:vector A n) (e x0:A),
  @v_max A (S n) f (@v_nth A n x0 v 0) (v, e) = 
  if f e (v_max f (v_nth x0 v 0) v) then e else v_max f (v_nth x0 v 0) v.
Proof.
  intros. simpl. auto.
Qed.

Lemma fin_nat2fin_eq : forall {n : nat} (x : fin n) (H : fin2nat x < n),
  F[ H] = x. (* nat2fin H = x.*)
Proof.
  intros n x H.  destruct x as [x Hx].  unfold nat2fin.  simpl.
  apply exist_eq.
Qed.

Lemma v_unzip1_mk_vec: forall (A B:Set) n (x0:A*B) f g,
  v_unzip1 x0 (mk_vec (fun i => (f i, g i))) = 
  mk_vec (fun i:fin n => f i).
Proof.
  intros. unfold v_unzip1.  f_equal. fun_ext.
  destruct x. 
  rewrite v_idx_mk_vec. reflexivity.
Qed.

Lemma v_unzip2_mk_vec: forall (A B:Set) n (x0:A*B)f g,
  v_unzip2 x0 (mk_vec (fun i => (f i, g i))) = 
  mk_vec (fun i: fin n => g i).
Proof.
  intros. unfold v_unzip2.  f_equal. fun_ext.
(* snd (v_nth x0 (mk_vec (fun i : fin (S n) => (f i, g i))) x) = g x *)
  assert(H : x<S n). destruct x. simpl. lia.
  rewrite v_idx_mk_vec. reflexivity.
Qed.

Lemma v_nth_map: forall (A B:Set) (n:nat) (f: A->B) (v:vector A n) (x0:A) (x1:B)(i:fin n),
  v_nth x1 (v_map x0 f v) i = f (v_nth x0 v i).
Proof.
  intros. unfold v_map.
  rewrite v_nth_mk_vec with (H:=fin_proof i). f_equal. 
Qed.

(* cur f applied to element i of the zipped vector v_zip x0 x1 v1 v2 is equivalent to f applied to the i-th elements of v1 and v2 respectively. *)
Lemma v_nth_v_zip : forall {A B C : Set} (f : A -> B -> C) 
  (x0 : A) (x1 : B) (n : nat)
  (v1 : vector A n) (v2 : vector B n) (i : fin n),
  cur f (v_nth (x0, x1) (v_zip x0 x1 v1 v2) i) =
  f (v_nth x0 v1 i) (v_nth x1 v2 i).
Proof.
  intros A B C f x0 x1 n v1 v2 i.
  unfold cur. f_equal. 
+ unfold v_zip. destruct i.
  rewrite v_nth_mk_vec with (H:=l). reflexivity. 
+ unfold v_zip. destruct i.
  rewrite v_nth_mk_vec with (H:=l). reflexivity. 
Qed.

Lemma v_map_zip : forall n (A B C:Set) (x0:A) (x1:B) (f:A->B->C)
  (v1:vector A n) (v2:vector B n),
  v_map (x0,x1) (cur f) (v_zip x0 x1 v1 v2) = 
  mk_vec (fun i => f (v_nth x0 v1 i) (v_nth x1 v2 i)).
Proof.
  intros. unfold v_map. f_equal. fun_ext. 
  unfold v_idx. apply v_nth_v_zip.
Qed.

Lemma mk_vec_zip : forall n (A B C:Set) (x0:A) (x1:B) (f:A->B->C) 
  (g: fin n-> fin n) (v1:vector A n) (v2:vector B n),
    mk_vec (fun i => cur f (v_nth (x0,x1) (v_zip x0 x1 v1 v2) (g i))) 
  = mk_vec (fun i => f (v_nth x0 v1 (g i))(v_nth x1 v2 (g i))).
Proof.
  intros. unfold v_map. f_equal. fun_ext. 
  unfold v_idx. apply v_nth_v_zip.
Qed. 

Lemma mk_vec2_zip : forall m n (A B C:Set) (x0:A) (x1:B) (f:A->B->C) 
  (g: fin m -> fin n -> fin (m*n)) (v1:vector A (m*n)) (v2:vector B (m*n)),
    mk_vec (fun i => mk_vec (fun j => 
              cur f (v_nth (x0,x1) (v_zip x0 x1 v1 v2) (g i j)))) 
  = mk_vec (fun i => mk_vec (fun j => 
              f (v_nth x0 v1 (g i j))(v_nth x1 v2 (g i j)))).
Proof.
  intros. unfold v_map. f_equal. fun_ext. f_equal. fun_ext.  
  apply v_nth_v_zip.
Qed.

Lemma v_map_fg : forall n (A B C:Set) (x0:A) (x1:B) (f:A->B) (g:B->C) (v1:vector A n),
  v_map x1 g (v_map x0 f v1) = v_map x0 (fun a => g (f a)) v1.
Proof.
  intros. unfold v_map. f_equal. fun_ext. f_equal.
  rewrite v_idx_mk_vec. reflexivity.
Qed.

Lemma idx_lt_mn : forall (m n x x0 : nat),
  x < m ->  x0 < n -> x * n + x0 < m * n.
Proof.
  intros m n x x0 Hxm Hx0.
  (* case split on n: if n = 0 then x0 < 0 is a contradiction, so only the case n = S n' needs handling *)
  destruct n as [| n'].
  - inversion Hx0.  (* x0 < 0 is impossible for natural numbers; contradiction directly *)
  - (* now n = S n' > 0 *)
    (* step 1: from x0 < S n' conclude x*n + x0 < x*n + S n' *)
    eapply Nat.lt_le_trans.
    + apply Nat.add_lt_mono_l. exact Hx0.
    (* step 2: rewrite x*(S n') + (S n') as (S n')*(x+1) *)
    + rewrite <- Nat.mul_succ_l.
      assert(S x <= m) by lia.
      (* step 3: from x < m we get x+1 <= m; multiplying by (S n') > 0 gives (S n')*(x+1) <= (S n')*m *)
      apply Nat.mul_le_mono_pos_r.
      * apply Nat.lt_0_succ.
      * exact H.
Qed.

Lemma idx_lt_mn_fin : forall (m n : nat) (i : fin m) (j : fin n),
  fin2nat i * n + fin2nat j < m * n.
Proof. intros. destruct i,j. simpl.  apply idx_lt_mn; auto. Qed.


(* splitting v, mapping, and recombining is equivalent to computing each element's value directly by index. *)
Lemma v_map_map_split: forall (m n : nat) (A B:Set) (x0:vector A n) 
  (x1:A) (f : A -> B) (v : vector A (m*n)),
  v_map x0 (fun v0 => v_map x1 f v0) (v_split x1 v) =
  mk_vec (fun i => mk_vec (fun j => 
    f (v_idx x1 v (nat2fin (idx_lt_mn_fin  i j ))))).
Proof.
  intros. unfold v_map. f_equal. fun_ext. f_equal. fun_ext.
  f_equal. unfold v_split. 
  rewrite v_idx_mk_vec.
  rewrite v_idx_mk_vec.
  f_equal. unfold idx_2d_to_1d .
  apply exist_equal. reflexivity.
Qed.

(*
Example
Suppose m = 2, n = 3.  v = [a, b, c, d, e, f].  f is a function from
elements of type A to type B.

Left-hand side of the equation:
Split v into [[a, b, c], [d, e, f]].
Apply v_map x1 f to each subvector, giving
[[f a, f b, f c], [f d, f e, f f]].

Right-hand side of the equation:
For i = 0, j = 0, compute f (v_idx x1 v (nat2fin (idx_lt_mn_fin 0 0))) = f a.
For i = 0, j = 1, compute f (v_idx x1 v (nat2fin (idx_lt_mn_fin 0 1))) = f b.

And so on, finally giving [[f a, f b, f c], [f d, f e, f f]].
*)
 
Lemma div_lt_m : forall {m n : nat} (i : fin (m * n)),
  fin2nat i / n < m.
Proof.
  intros m n [i Hi]. simpl.
  apply Div0.div_lt_upper_bound. lia.
Qed.

Lemma mod_lt_n : forall {m n : nat} (i : fin (m * n)),
  fin2nat i mod n < n.
Proof.
  intros m n [i Hi]. destruct n. lia.
  apply Nat.mod_upper_bound. lia.
Qed.

Lemma default_vec_nth : forall {A : Set} {n : nat} (x0 : A) (i : fin n),
  v_nth x0 (default_vec n x0) i = x0.
Proof.
  intros A n x0 i. induction n as [ | n' IH].
  - (* Case n = 0 *) destruct i. lia. 
  - (* Case n = S n' *)
    destruct i as [i Hi]. simpl. destruct (i =? n') eqn:Heq.
    + apply Nat.eqb_eq in Heq. subst. reflexivity.
    + apply Nat.eqb_neq in Heq.
    assert (H : i < n') by lia.
    rewrite v_nth_mk_vec with (H:=H). reflexivity.
Qed.

(* accessing an element in a nested vector by first converting a 1D index to a 2D index
  (using idx_1d_to_2d) is equivalent to directly computing the 2D indices using division 
  and modulo operations. 
Proof Sketch
1) Unfold idx_1d_to_2d: Extract the 2D indices (i, j) from the 1D index x.
2) Define row and col: Compute row = x / S n and col = x mod S n.
3) Prove bounds for row and col:
  Use Nat.div_lt_upper_bound to prove row < S m.
  Use Nat.mod_upper_bound to prove col < S n.
4) Simplify the goal:
  Replace x / S n with row and x mod S n with col.
5) Replace F[ div_lt_m (S m) [ | x] H] and F[ mod_lt_n (S m) [ | x] H]:
  Replace these terms with [ | row] and [ | col], respectively, using exist_equal to 
  prove equality of the existential terms.
6) Use v_nth_independent_of_default:
  Show that v_nth x0 (v_nth (default_vec (S n) x0) v [ | row]) [ | col] is equal to 
  v_nth x0 (v_nth x1 v [ | row]) [ | col] by using the lemma v_nth_independent_of_default.
7) Conclude by reflexivity:
*)
Lemma v_nth_2d_eq : forall (m n : nat) (A : Set) (x0 : A)
  (x1 : vector A (S n)) (v : vector (vector A (S n)) (S m))
  (x : fin (S m * S n)),
  (let '(i, j) := idx_1d_to_2d x in
    v_nth x0 (v_nth (default_vec (S n) x0) v i) j) =
  v_nth x0 (v_nth x1 v F[ div_lt_m x]) F[ mod_lt_n x].
Proof.
  intros m n A x0 x1 v x.
  (* Unfold idx_1d_to_2d to extract the pair (i, j) *)
  unfold idx_1d_to_2d.  destruct x as [x Hx]. 
  (* Extract the row and column indices *)
  set (row := x / S n).  set (col := x mod S n).
  (* Prove that row < S m and col < S n *)
  assert (Hrow : row < S m).
  { unfold row.   apply Div0.div_lt_upper_bound. lia. }
  assert (Hcol : col < S n).
  {   apply Nat.mod_upper_bound. lia. } 
(*
v_nth x0 (v_nth (default_vec (S n) x0) v [ | row]) [ | col] =
v_nth x0 (v_nth x1 v F[ div_lt_m (S m) [ | x] H]) F[ mod_lt_n (S m) [ | x] H]
*)
  replace (F[ div_lt_m [ | x]]) with
 (@exist nat (fun i : nat => lt i (S m)) row
              (Nat.Private_NDivProp.div_lt_upper_bound x 
                 (S n) (S m) (Nat.neq_succ_0 n)
                 (@eq_rect nat (mul (S m) (S n)) (fun z : nat => lt x z) Hx
                    (mul (S n) (S m)) (Nat.mul_comm (S m) (S n))))) .
  replace (F[ mod_lt_n [ | x]]) with
    (@exist nat (fun i : nat => lt i (S n)) col
           (Nat.mod_upper_bound x (S n) (Nat.neq_succ_0 n))).
  - (* v_nth x0 (v_nth (default_vec (S n) x0) v [ | row]) [ | col] =
       v_nth x0 (v_nth x1 v [ | row]) [ | col] *)
    rewrite v_nth_x0_eq with (x1:=x1). reflexivity. lia. auto.
  - (* [ | col] = F[ mod_lt_n (S m) [ | x] H] *)
    apply exist_equal. unfold col.  reflexivity.
  - (* [ | row] = F[ div_lt_m (S m) [ | x] H] *)
    apply exist_equal. unfold row.  reflexivity.
Qed.

(*
 mapping a function f over a joined nested vector is equivalent to constructing 
 a new vector where each element is obtained by applying f to the corresponding 
 element in the nested vector v. 
*)
Lemma v_map_join : forall (m n : nat) (A B : Set) (x0 : A)
  (f : A -> B) (x1:vector A n) (v : vector (vector A n) m)
  (H:n <> 0) , (* Add this precondition *)
  v_map x0 f (v_join x0 v) =
  mk_vec (fun (i : fin (m * n)) => f (v_idx x0
    (v_idx x1 v (F[div_lt_m i]))
    F[ mod_lt_n i])).
Proof.
  autounfold.
  intros. induction m;auto.
  induction n. 
+ unfold v_map.  f_equal. fun_ext.
  assert(x < S m * 0). destruct x. exact l.
  rewrite (Nat.mul_0_r (S m)) in H0. inversion H0.
+ unfold v_map. f_equal. fun_ext. f_equal. unfold v_join.
  rewrite v_nth_mk_vec with (H:=(fin_proof x)). unfold v_idx.
  assert(H1:S n <> 0). lia. assert (F[fin_proof x] = x). 
  apply nat2fin_fin2nat. rewrite H0. apply v_nth_2d_eq.
Qed. 

(*
Lemma v_map_join_split: forall [m n:nat] (A B:Set) (f:A -> B) 
  (x0:A)(x1:vector A n) (x2: B) (x3: vector B n)(v:vector A (m*n)),
  v_map x0 f v = v_join x2 x3 (v_map x1 (v_map x0 f) (v_split x0 v)).
*)

(* mapping a function f over a flat vector v is equivalent to splitting v into 
  a nested vector, mapping f over the nested structure, and then joining it back
  into a flat vector. 
*)
Lemma v_map_join_split: forall [m n:nat] (A B:Set) (f:A -> B) 
  (x0:A)(x1:vector A n) (x2: B) (v:vector A (m*n)),
  v_map x0 f v = v_join x2 (v_map x1 (v_map x0 f) (v_split x0 v)).
Proof.
  intros. rewrite v_map_map_split. unfold v_map. unfold v_join.
  f_equal. fun_ext. 
  unfold idx_1d_to_2d. destruct n.
+ simpl. destruct x.  inversion l; lia.
+ destruct x. 
  assert(H : x / S n < m).  { apply Div0.div_lt_upper_bound; lia.   }
  rewrite v_nth_mk_vec with (H:=H).
  assert(H1 : x mod S n < S n).  { apply Nat.mod_upper_bound; lia.   }
  rewrite v_idx_mk_vec. 
  f_equal. f_equal.
  (* x = F[ idx_lt_mn_fin F[ l] [ | x4]] *)
  destruct x.
  - (* [ | 0] = F[ idx_lt_mn_fin F[ H] [ | 0 mod S n]] *)
     apply exist_equal. simpl. lia.
  - (* [ | S x] = F[ idx_lt_mn_fin F[ H] [ | S x mod S n]] *)
     apply exist_equal. 
    (* S x = F[ H] * S n + [ | S x mod S n] *)
    rewrite Nat.mul_comm.
    apply Nat.div_mod with (x := S x) (y := S n). auto.    
Qed.

Lemma v_nth_fin2finS0 : forall {A : Set} {n : nat} (x0 : A)
  (v : vector A n) (a : A) (x : fin n),
  @v_nth A (S n) x0 (v, a) (fin2finS x) = v_nth x0 v x.
Proof.
  intros A n x0 v a x.
  (* Unfold v_nth for the vector (v, a) *)
  unfold v_nth.
  assert (fin2nat (fin2finS x) = fin2nat x) as H_eq.
  {
    (* fin2finS preserves the underlying natural number *)
    simpl.
    reflexivity.
  }
  (* Rewrite using H_eq *)
  rewrite H_eq.
  (* Case analysis on whether fin2nat x = n *)
  destruct (fin2nat x =? n) eqn:H_case.
  - (* Case: fin2nat x = n *)
    (* This is impossible because x : fin n implies fin2nat x < n *)
    apply Nat.eqb_eq in H_case.
    destruct x as [val H_lt].
    simpl in H_case. subst. lia.
  - (* Case: fin2nat x <> n *)
    destruct x. simpl. reflexivity.
Qed.

Lemma v_nth_fin2finS : forall {A : Set} {n : nat} (x0 : A) 
  (v : vector A n) (a : A) (x : fin n),
  @v_nth A (S n) x0 (v, a) (fin2finS (fin2finS x)) 
= @v_nth A n x0 v (fin2finS x).
Proof.
  intros A n x0 v a x.
  (* Unfold v_nth for the vector (v, a) *)
  unfold v_nth.
  (* Simplify the index fin2finS (fin2finS x) *)
  assert (fin2nat (fin2finS (fin2finS x)) = fin2nat x) as H_eq.
  {
    (* fin2finS preserves the underlying natural number *)
    simpl.
    reflexivity.
  }
  (* Rewrite using H_eq *)
  rewrite H_eq.
  (* Case analysis on whether fin2nat x = n *)
  destruct (fin2nat x =? n) eqn:H_case.
  - (* Case: fin2nat x = n *)
    (* This is impossible because x : fin n implies fin2nat x < n *)
    apply Nat.eqb_eq in H_case.
    destruct x as [val H_lt].
    simpl in H_case. lia.
  - (* Case: fin2nat x <> n *)
    reflexivity.
Qed.

Lemma v_reduce_map: forall [n:nat] (A B C:Set) (g:A -> B) 
    (f: B -> C -> C) (i:C) (x0:A) (v:vector A n) ,
  v_reduce f i (v_map x0 g v) = v_reduce (fun x a => f (g x) a) i v.
Proof.
  autounfold. induction n,v;auto.
 simpl. f_equal.
  f_equal. unfold v_idx. simpl.   rewrite Nat.eqb_refl. auto.
  rewrite <-IHn with (x0:=x0) (v:=v). f_equal. induction n;auto.
  f_equal. simpl. unfold v_map. simpl. f_equal.
+  f_equal.   
  fun_ext. f_equal. assert (x< S n);f_simpl.
  unfold v_idx. apply @v_nth_fin2finS with (n:=S n).
Qed.

Lemma v_join_split: forall (A:Set) (m n:nat) (x0:A) (v:vector A (m*n)),
  v_join x0 (v_split x0 v) = v.
Proof. apply v_split_round_trip. Qed.

Lemma mk_vec_tt: forall (m:nat) (a: vector unit m),
  mk_vec (fun _ : fin m => tt) = a.
Proof.
  induction m,a;auto.
  simpl. f_equal. apply IHm with (a:=v). induction u;auto.
Qed.

Lemma v_split_join : forall (A : Set) (m n : nat) (x0 : A) (v : vector (vector A n) m),
  v_split x0 (v_join x0 v) = v.
Proof.
  intros A m n x0 v.
  (* Use functional extensionality to show equality of nested vectors *)
  rewrite vector_eq_v_nth with (x0 := (default_vec n x0)).
  intro idx.
  (* Unfold v_split and v_join *)
  unfold v_split, v_join.
  destruct idx.
  (* Simplify using v_nth and mk_vec *)
  rewrite v_nth_mk_vec with (H:=l).
  rewrite vector_eq_v_nth with (x0:=x0). intro i.
  destruct i.
  rewrite v_nth_mk_vec with (H:=l0).
  rewrite v_idx_mk_vec.
  rewrite idx_1d_to_2d_after_idx_2d_to_1d.
  unfold v_idx.
(*
v_nth x0 (v_nth (default_vec n x0) v F[ l]) F[ l0] =
v_nth x0 (v_nth (default_vec n x0) v [ | x]) [ | x1]
*)
  reflexivity.
Qed.

Lemma v_zip_map: forall (A B:Set) (n:nat) (f g: A -> B) 
  (x0:A) (x1:B) (a b: vector A n),
    v_zip x1 x1 (v_map x0 f a) (v_map x0 g b) = 
    v_map (x0,x0) (fun xy => (f (fst xy),g (snd xy))) (v_zip x0 x0 a b).
Proof.
  intros. autounfold.  induction n. 
+ auto. 
+ f_equal. fun_ext.  destruct x.
  rewrite ?v_nth_mk_vec with (H:=l). f_simpl.
Qed.

Lemma v_concat_0_l: forall {A:Set} {n: nat} (x0:A) (v1:vector A 0) (v2: vector A n),
  v_concat x0 v1 v2 = v2.
Proof.
  induction n. induction v1,v2;auto.
  unfold v_concat,v_idx. intros. simpl (0+S n).
  rewrite <- mk_vec_bij_fin with (x0:=x0)(A:= A)(m:= S n).
  f_equal. fun_ext. replace (x<?0) with false.
  f_equal. destruct (Nat.ltb_spec0 x 0 ).
+ inversion l. 
+ f_equal. simpl. lia.
+ auto. 
Qed.  

Lemma v_truncl_0:forall {A:Set} {m: nat} (x0:A) (v1:vector A m),
  v_truncl x0 (Nat.le_0_l m:0<=m)  v1 = tt. 
Proof. auto. Qed.

Lemma v_truncl_n:forall {A:Set} {n: nat} (x0:A) (v1:vector A n),
  v_truncl x0 (Nat.le_refl n:n<=n) v1 = v1.
Proof.
  unfold v_truncl,v_idx. induction n. induction v1. auto.
  intros. rewrite <-mk_vec_bij_fin with(x0:=x0).
  f_equal.
Qed.

(* truncating a vector by its full length leaves the vector unchanged. *)
Lemma v_truncr_n :  
  forall {A : Set} {n : nat} (x0 : A) (v1 : vector A n),  
    v_truncr x0 (Nat.le_refl n : n <= n) v1 = v1.  
Proof.  
  intros A n x0 v1.  
  (* Show two vectors are equal by comparing their nth elements. *)  
  rewrite vector_eq_v_nth with (x0 := x0).  (* or: apply vector_extensionality if it’s available *)  
  intros i.  
  unfold v_truncr; cbn.               (* reveals mk_vec’s body *)  
  destruct i.
  rewrite v_nth_mk_vec with (H:=l).               (* selects the i-th element of mk_vec *)  
  unfold v_idx_sub_offset. simpl.            (* offset is (n-n) = 0 *)  
  unfold v_idx. f_equal.
(*
F[ Nat.lt_le_trans (x + (n - n)) (n + (n - n)) n
     (proj1 (Nat.add_lt_mono_r x n (n - n)) l)
     (offset_add_le (Nat.le_refl n))] = x
*)
  cbn in *. lia.
Qed.  

Lemma v_nth_v_concat : forall {A : Set} {m n : nat} (x0 : A)
  (v1 : vector A m) (v2 : vector A n) (i : nat) (H:i < (m + n)),
  v_nth x0 (v_concat x0 v1 v2) i =
  if i <? m then v_nth x0 v1 i else v_nth x0 v2 (i - m).
Proof.
  intros A m n x0 v1 v2 i H.
  (* Unfold v_concat *)
  unfold v_concat.
  (* Simplify using v_nth and mk_vec *)
  rewrite v_nth_mk_vec with (H := H).
  (* Case analysis on whether i < m *)
  destruct (i <? m) eqn:Hi.
  - (* Case i < m *)
    apply Nat.ltb_lt in Hi.
    destruct (Nat.ltb_spec0 F[ H] m ).
    + reflexivity.
    + destruct n0. simpl. exact Hi.
  - (* Case i >= m *)
    apply Nat.ltb_ge in Hi.
    destruct (Nat.ltb_spec0 F[ H] m ).
    + simpl in l. lia.
    + simpl in n0. simpl. unfold v_idx. f_equal. 
Qed.


Lemma v_truncl_concat : forall {A : Set} {m n : nat} (x0 : A)
  (v1 : vector A m) (v2 : vector A n),
  v_truncl x0 (Nat.le_add_r m n) (v_concat x0 v1 v2) = v1.
Proof.
  intros A m n x0 v1 v2.
  (* Use functional extensionality to show equality of vectors *)
  rewrite vector_eq_v_nth with (x0 := x0).
  intro i.
  (* Unfold v_truncl and v_concat *)
  unfold v_truncl. 
  (* Simplify using v_nth and mk_vec *)
  rewrite v_nth_mk_vec with (H := fin_proof i).
  unfold v_idx_sub. unfold v_idx. simpl. 
  (* Simplify the goal using v_nth_v_concat *)
  rewrite v_nth_v_concat.
  destruct i. simpl. 
  apply Nat.ltb_lt in l. rewrite l. reflexivity.
  destruct i. simpl. lia. 
Qed.

Lemma v_nth_padr : forall {A : Set} {n : nat} (x0 : A) (k : nat) (v : vector A n) (i : fin n),
  v_nth x0 (v_padr x0 k v) i = v_nth x0 v i.
Proof.
  intros A n x0 k v i.
  (* Unfold the definition of v_padr *)
  unfold v_padr. destruct i. simpl. 
  assert(x < n+k) by lia.
  rewrite v_nth_mk_vec with (H := H).
  destruct (Nat.ltb_spec0 F[ H] n).
+ reflexivity.
+ destruct n0. simpl. exact l.
Qed.

(*
Lemma v_truncl_padr: forall {A:Set} {n:nat} (x0:A) (k:nat) (v:vector A n),
  v_truncl x0 n (v_padr x0 k v) = v.
*)
Lemma v_truncl_padr: forall {A:Set} {n:nat} (x0:A) (k:nat) (v:vector A n),
  v_truncl x0 (Nat.le_add_r n k:n<=n+k) (v_padr x0 k v) = v.
Proof.
  intros A  n x0 k v. 
  (* Use functional extensionality to show equality of vectors *)
  rewrite vector_eq_v_nth with (x0 := x0).
  intro i.
  (* Unfold v_truncl and v_concat *)
  unfold v_truncl. 
  (* Simplify using v_nth and mk_vec *)
  rewrite v_nth_mk_vec with (H := fin_proof i).
  unfold v_idx_sub. unfold v_idx. simpl.
  apply v_nth_padr.
Qed.

Lemma v_trans_trans:forall {A:Set} {m n: nat} (x0:A)(v:vector (vector A m) n),
  v_trans x0 (v_trans x0 v) = v.
Proof.
  intros A m n x0 v. unfold v_trans.
  rewrite vector_eq_v_nth with (x0:=default_vec m x0). intro i.
  rewrite v_nth_mk_vec with (H := fin_proof i).
  rewrite vector_eq_v_nth with (x0:=x0). intro i0.
  rewrite v_nth_mk_vec with (H := fin_proof i0).
  unfold v_idx. simpl. 
  rewrite v_nth_mk_vec with (H := fin_proof i0).
  rewrite v_nth_mk_vec with (H := fin_proof i).
  simpl. f_equal.  
Qed.

Module Type EType.

Parameter A :Set.
Parameter Zero One : A.
Parameter opp : A -> A.
Parameter add mul: A->A->A.

Infix " + " := add.
Infix " * " := mul.

Parameter add_comm : forall x y , x + y = y + x.
Parameter add_assoc : forall x y z , x + y + z = x + (y + z).
Parameter add_zero_l : forall x , Zero + x  = x.
Parameter mul_add_distr_l : forall x y z, (x+y)*z = x*z + y*z.
Parameter mul_add_distr_r : forall x y z, x*(y+z) = x*y + x*z.
Parameter mul_assoc : forall x y z, x * y * z = x * (y * z).
Parameter mul_zero_l : forall x , Zero * x = Zero.
Parameter mul_one_l : forall x , One * x = x.
Parameter mul_comm : forall x y, x * y = y * x.

End EType.

Module VecMat (E:EType).

Definition A:=E.A.

Definition Zero := E.Zero.

Definition add := E.add.

Definition mul := E.mul.

Definition v_zero {n:nat} := v_make n Zero.

Definition v_add {n:nat} (v1 v2: vector A n) : vector A n :=
  v_map (Zero,Zero) (fun xy: A * A  => add (fst xy) (snd xy)) (v_zip Zero Zero v1 v2).

Definition product {n:nat} (v1 v2: vector A n) : A := 
  v_reduce add Zero (v_map (Zero,Zero) (fun xy: A * A  => mul (fst xy) (snd xy)) 
    (v_zip Zero Zero v1 v2)).

#[export] Hint Unfold v_make v_zero v_add product : core.

Lemma v_add_comm : forall n (v1 v2: vector A n),
  v_add v1 v2 = v_add v2 v1.
Proof.
  autounfold. intros.
  f_equal. fun_ext. destruct x.
  rewrite ?v_nth_mk_vec with (H:=l). f_simpl.
  unfold fst,snd. rewrite E.add_comm. auto.
Qed.

Lemma v_add_assoc: forall n (v1 v2 v3: vector A n),
  v_add (v_add v1 v2) v3 = v_add v1 (v_add v2 v3).
Proof.
  autounfold. intros.
  f_equal. fun_ext. destruct x.
  rewrite ?v_nth_mk_vec with (H:=l). f_simpl.
  unfold fst,snd. rewrite E.add_assoc. auto. 
Qed.

Lemma v_add_zero_l: forall n (v: vector A n),
  v_add v_zero v = v.
Proof.
  autounfold. 
  intros. rewrite <- mk_vec_bij_fin with (x0:=Zero).
  f_equal. fun_ext. destruct x.
  rewrite ?v_nth_mk_vec with (H:=l). f_simpl.
  simpl. rewrite E.add_zero_l. reflexivity.
Qed.

Lemma v_add_zero_r: forall n (v: vector A n),
  v_add v v_zero = v.
Proof. intros. rewrite v_add_comm. apply v_add_zero_l. 
Qed.

Lemma product_comm : forall n (v1 v2: vector A n),
  product v1 v2 = product v2 v1.
Proof. 
  autounfold. 
  intros. f_equal. f_equal. fun_ext. destruct x.
  rewrite ?v_nth_mk_vec with (H:=l). f_simpl. simpl. 
  rewrite E.mul_comm. auto.
Qed.

Lemma product_zero_l : forall n (v: vector A n),
  product v_zero v = Zero.
Proof.
  autounfold. induction n;auto.
  induction v. rewrite mk_vec_Sn. autorewrite with core.
  rewrite v_reduce_Sn. simpl. 
  assert(H: n =? n = true) by apply Nat.eqb_refl. rewrite H.
  simpl. rewrite E.mul_zero_l. rewrite E.add_zero_l.
  symmetry. rewrite <-IHn with (v:=a) at 1.
  f_equal. f_equal. fun_ext. 
  destruct x. simpl. 
  assert(H1: x =? n = false). { apply Nat.eqb_neq, Nat.lt_neq; assumption. }
  rewrite H1.
  rewrite ?v_nth_mk_vec with (H:=l). f_simpl. simpl. 
  rewrite H1. reflexivity.
Qed.

Lemma product_zero_r : forall n (v: vector A n),
  product v v_zero = Zero.
Proof. intros.
  rewrite product_comm. apply product_zero_l.
Qed.

End VecMat.
  




