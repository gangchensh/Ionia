(* # =================================================================== 
   # version 0.2a FinType 2025, modified from finType 2025
   #
   # nat2fin outputs fin n with an argument (H:n>0) :
   # Definition nat2fin {n : nat} (H : n > 0) (i : nat) : fin n :=
   #
   # Introduction
   #
   # We build two types of the finite set of natural numbers with sig
   #
   # fin type is designed for the index of array :
   #
   #   fin n = { i : nat | i < n }. When n = 0, fin 0 IS EMPTY. (2025)
   #
   # fle type is designed for the verification of some rules about array :
   #
   #   fle n = { i : nat | i <= n }.
   #
   # The conversions between nat and the two types above are built by 
   #
   #   fin2nat, fle2nat, nat2fin, nat2fle.
   #
   # There are some properties between these functions.
   #
   #   fin2nat_nat2fin : forall n i , i < n -> fin2nat(nat2fin n i) = i.
   # 
   #   fin_eq : forall (n:nat) (i1 i2:fin n), fin2nat i1 = fin2nat i2 -> i1 = i2.
   #
   #   nat2fin_fin2nat : forall n (i:fin n) , nat2fin n (fin2nat i) = i
   #
   #   fin2nat_exist: forall m x p, @fin2nat (S m) (exist (fun i : nat => i < (S m)) x p) = x.
   #
   #   fle_eq : forall (n:nat) (i1 i2:fle n), fle2nat i1 = fle2nat i2 -> i1 = i2.
   #
   #   fle2nat_exist: forall m x p, @fle2nat m (exist (fun i : nat => i <= m) x p) = x.
   # =================================================================== *)

Require Export Arith Lia Reals.
Require Import Coq.Logic.ProofIrrelevance.

Set Implicit Arguments.

Ltac inv H := intros; inversion H.


Lemma Req_dec_T_refl : forall r : R, Req_dec_T r r = left eq_refl.  
Proof.  
  intros r.  
  (* Since Req_dec_T is opaque, we cannot unfold it directly. *)  
  (* However, we know that Req_dec_T r r must return left eq_refl because r = r is provable. *)  
  destruct (Req_dec_T r r) as [H1 | H2].  
  - (* Case: left H *)  
    (* H is a proof of r = r, which is eq_refl. *)  
    f_equal. apply proof_irrelevance. (* This proves that H = eq_refl. *)  
  - (* Case: right H *)  
    (* This case is impossible because r = r is always true. *)  
    contradiction. 
Qed.  


(* the proof of m < n i is unique *)
Lemma lt_unique : forall m n (p1 p2: m < n),
  p1 = p2.
Proof. intros. unfold lt in p1,p2. apply le_unique. Qed.

(* define the generic lemma exist_eq *)
Lemma exist_eq : forall {B:Set} {P : B -> Prop} (x : B) (p1 p2 : P x),
   exist P x p1 = exist P x p2.
Proof.
  intros B P x p1 p2.
  f_equal. (* use f_equal to split the goal into equality of the value part and the proof part *)
  apply proof_irrelevance. 
Qed.

Notation " [ | n ]" := (exist _ n _).

Lemma match_Some_None  {A:Set} : forall (a:option A) (e:A), 
  a <> None -> match a with Some v => Some e | None => None end = Some e.
Proof. intros a e H. destruct a. auto. contradiction. Qed.


Section FiniteSetLt.

Local Open Scope nat_scope.

(* define fin n as the natural numbers smaller than n *)
Definition fin (n : nat) : Set := {i : nat | i < n}.

(* convert fin n to nat *)
Definition fin2nat {n : nat} (i : fin n) : nat :=
  match i with
  | exist _ i _ => i
  end.

Lemma fin_correct (n:nat) : forall (i:fin n), fin2nat i < n.
Proof. intro. destruct i. exact l. Qed.

(* convert fin n to fin (S n) *)
Definition fin2finS {n : nat} (i : fin n) : fin (S n) :=
  exist _ (fin2nat i) (Nat.lt_lt_succ_r _ _ (proj2_sig i)).

Notation "FS[ i ]" := (fin2finS i).

Lemma finS2fin_aux : forall (i n : nat),
  i < S n -> i <> n -> i < n.
Proof. lia. Qed.

(* convert fin (S n) to fin n *)
Definition finS2fin {n : nat} (i : fin (S n)) (H : fin2nat i <> n) : fin n :=
  exist _ (fin2nat i) (finS2fin_aux (fin_correct i) H).


(* prove that elements of fin n are smaller than S n *)
Lemma fin_finS_eq : forall (n : nat) (i : fin n), fin2nat i < S n.
Proof.
  intros n i. destruct i as [i Hi]. simpl. apply Nat.lt_lt_succ_r. assumption.
Qed.

Coercion fin2nat : fin >-> nat.

(* define an instance of fin 3 *)
Example fin3_example : fin 3 := exist _ 2 (Nat.lt_succ_diag_r 2).

(* extract the natural number from the fin 3 instance *)
Example fin2nat_example : fin2nat fin3_example = 2.
Proof. reflexivity. Qed.

(* convert the fin 3 instance to fin 4 *)
Example fin2finS_example : fin2finS fin3_example = exist _ 2 (Nat.lt_lt_succ_r _ _ (Nat.lt_succ_diag_r 2)).
Proof. reflexivity. Qed.

(* define the zero element of fin n *)
Definition fin0H {n : nat} (H : 0 < n) : fin n :=
  exist _ 0 H.

Example fin3_zero : fin 3 := fin0H (Nat.lt_0_succ 2).

(* prove that fin 0 is empty *)
Lemma fin0_empty : fin 0 -> False.
Proof.
  intros [i Hi]. (* decompose the element of fin 0 *)
  simpl in Hi. (* Hi : i < 0 *)
  apply Nat.nlt_0_r in Hi. (* i < 0 is impossible *)
  assumption.
Qed.

(* prove that fin 1 contains only the element 0 *)
Lemma fin1_elem :
  forall (i : fin 1), fin2nat i = 0.
Proof.
  intros [i Hi]. (* decompose the element of fin 1 *) simpl. lia.
Qed.

(* prove that fin 1 is unique (has exactly one inhabitant) *)
Lemma fin1_unique :
  forall (i i' : fin 1), i = i'.
Proof.
  intros [i Hi] [i' Hi']. (* decompose the elements i, i' : fin 1 *)
  assert (i = 0). lia. assert (i' = 0). lia. (* infer the values of i and i' *)
  subst. f_equal. apply proof_irrelevance.
Qed.

(* automatically generate a proof of 0 < n, returning option (fin n) *)
Definition fin0_auto {n : nat} : option (fin n) :=
  match n with
  | 0 => None (* return None when n = 0 *)
  | S n' => Some (exist _ 0 (Nat.lt_0_succ n')) (* return Some fin_0 when n > 0 *)
  end.

Example fin3_zero_auto : option (fin 3) := fin0_auto.

(* define fin_0: return Some of the zero element of fin n (if it exists), otherwise None. *)
Definition fin_0 (n:nat)  := @fin0_auto n.

Definition fin0n (n:nat) : fin (S n) := exist _ 0 (Nat.lt_0_succ n).

Definition nat2fin {n : nat} (i : nat) (H : i < n) : fin n :=
  exist _ i H.

Definition fin_last (n:nat) : fin (S n) := nat2fin (Nat.lt_succ_diag_r n).

(*
Compute fin_0 3.
Compute @nat2fin 3 1.
Compute @nat2fin 5 2. (* [ | 2] *)
Compute [ | 2].
*)

(* The relationship between fin2nat and nat2fin *)
Lemma fin2nat_nat2fin : forall n i (H:i < n),
    fin2nat(@nat2fin n i H) = i.
Proof. intros. simpl. reflexivity. Qed.

(* extract the proof from fin n *)
Definition fin_proof {n : nat} (i : fin n) : fin2nat i < n :=
  match i with
  | exist _ _ H => H
  end.

(* i+1:fin (S n) *)
Definition finS {n:nat} (i:fin n): fin (S n) :=
  @nat2fin (S n) (S (fin2nat i)) 
    (proj1 (Nat.succ_lt_mono  i n) (fin_proof i)).

(* define the proposition: i is not the 0 of fin (S n) *)
Definition fin_not_zero {n : nat} (i : fin (S n)) : Prop :=
  fin2nat i <> 0.

(* define pred_lt: from a proof of i < S n, construct a proof of pred i < n *)
Lemma pred_lt {n : nat} {i : nat} (H : i < S n) : 0<i -> pred i < n.
Proof. lia. Qed. 

Definition finPred {n : nat} (i : fin (S n)) (H:0 < fin2nat i ) : fin n :=
   @nat2fin n (pred (fin2nat i)) (pred_lt  (fin_proof i) H).

Lemma finPred_type_correct {n : nat} (i : fin (S n)) (H : 0 < fin2nat i) :
  fin2nat (finPred i H) < n.
Proof. apply fin_correct. Qed.

(* i-1 :fin n *)
Lemma finPred_correct {n : nat} (i : nat) (H1:i < S n) (H2:0<i) :
   fin2nat (finPred (@nat2fin (S n) i H1) H2) = i-1.
Proof. simpl. lia. Qed. 

Lemma finPred_predecessor {n : nat} (i : fin (S n)) (H : 0 < fin2nat i) :
  fin2nat (finPred i H) = fin2nat i - 1.
Proof.  simpl. lia. Qed.

(* (1- (1+ i)) = i *)
Lemma finPred_inverse {n : nat} (i : fin n) :
  finPred (finS i) (Nat.lt_0_succ (fin2nat i)) = i.
Proof.
  destruct i as [i Hi].
  unfold finS, finPred.
  simpl. unfold nat2fin.
  apply exist_eq.
Qed.

Lemma finPred_inverse2 {n : nat} (i : fin (S n)) (H : 0 < fin2nat i) :  
  S (fin2nat (finPred i H)) = fin2nat i.  
Proof.  
  unfold finPred. simpl.  
  rewrite Nat.succ_pred; auto. lia.  
Qed.

Lemma finPred_correct2 {n : nat} (i : fin (S n)) (H1:i < S n) (H2:0<i) :
   fin2nat (finPred (@nat2fin (S n) i H1) H2) = i-1.
Proof. simpl. lia. Qed. 


Lemma finPredAux : forall {n:nat} (i:fin (S n)) , 0< fin2nat i ->
  pred (fin2nat i) < n.
Proof.
  destruct i. simpl. intros. lia. 
Qed.

(* i+m:fin n *)
Definition finSn {n:nat} (i:fin n) (m:nat) : fin (n+m).
Proof.
  (* decompose i *)
  destruct i as [i0 Hi].
  (* construct the proof for i + m *)
  apply (exist _ (i0 + m)).
  (* use Nat.add_lt_mono_r to prove i0 + m < n + m *)
  apply Nat.add_lt_mono_r.
  exact Hi.
Defined.


(* the values of fin n equal then two (fin n) equal *)
Lemma fin_eq : forall (n:nat) (i1 i2:fin n),
  fin2nat i1 = fin2nat i2 <-> i1 = i2.
Proof. 
  intros n i1 i2. (* introduce variables and hypotheses *)
  split. (* split the <-> into the two directions *)
  - (* left to right: i1 = i2 -> i1 = i2 *)
    intros H. (* assume fin2nat i1 = fin2nat i2 *)
    destruct i1 as [i1 Hi1]. (* decompose i1 *)
    destruct i2 as [i2 Hi2]. (* decompose i2 *)
    simpl in H. (* simplify H *)
    subst i2. (* replace i2 by i1 *)
    f_equal. (* reduce the goal to Hi1 = Hi2 *)
    apply proof_irrelevance. (* use proof irrelevance *)
  - (* right to left *)
    intros H. (* assume i1 = i2 *)
    subst i1. (* replace i1 by i2 *)
    reflexivity. (* the equality holds directly *)
Qed.

Lemma fin2finS_finS2fin : forall {n : nat} (i : fin n) (H : fin2nat i <> n),
  finS2fin FS[i] H = i.
Proof.
  intros. apply fin_eq. simpl. auto.
Qed.

Lemma finS2fin_fin2finS : forall {n : nat} (i : fin (S n)) (H : fin2nat i <> n),
  FS[finS2fin i H] = i.
Proof.
  intros. apply fin_eq. simpl. auto.
Qed.

Lemma fin_eq': forall (n:nat) (i1 i2:fin n),
  fin2nat i1 = fin2nat i2 -> i1 = i2.
Proof.
  apply fin_eq.
Qed.

Lemma fin_neq : forall n (i1 i2:fin n),
  fin2nat i1 <> fin2nat i2 <-> i1 <> i2.
Proof.
  intros n i1 i2. (* introduce variables and hypotheses *)
  split. (* split the <-> into the two directions *)
  - (* left to right *)
    intros H1 H2. (* assume fin2nat i1 <> fin2nat i2 and i1 = i2 *)
    apply H1. (* apply H1 *)
    rewrite H2. (* replace i1 by i2 *)
    reflexivity. (* the equality holds directly *)
  - (* right to left *)
    intros H1 H2. (* assume i1 <> i2 and fin2nat i1 = fin2nat i2 *)
    apply H1. (* apply H1 *)
    apply fin_eq. (* apply the fin_eq lemma *)
    exact H2. (* use the hypothesis fin2nat i1 = fin2nat i2 *)
Qed.

Lemma fin_neq' : forall n (i1 i2:fin n),
  fin2nat i1 <> fin2nat i2 <-> i1<> i2.
Proof.
  apply fin_neq.
Qed.

Definition ltS (m:nat): fin (S m):=
  exist _ m (Nat.lt_succ_diag_r m).


Lemma finltSn: forall n (i:fin (S n)),
  i < S n.
Proof. induction i. simpl. assumption.  Qed.

Lemma fin2nat_exist: forall m x p,
  @fin2nat (S m) (exist (fun i : nat => i < (S m)) x p) = x.
Proof. simpl. auto. Qed.

(* The relationship between fin2nat and nat2fin *)
Lemma nat2fin_fin2nat : forall (n : nat) (i : fin n),
  @nat2fin n (fin2nat i) (fin_proof i) = i.
Proof.
(*
forall (n : nat) (i : fin n),
@eq (fin n) (@nat2fin n (@fin2nat n i) (@fin_proof n i)) i
*)
  intros n i.
  (* destruct i *)
  destruct i as [i Hi].
  (* unfold the definitions of nat2fin and fin2nat *)
  unfold nat2fin, fin2nat, fin_proof.
  (* simplify *)
  simpl.
  f_equal.
Qed.

Lemma finS_pf : forall (n:nat) (i:fin n), i < S n.
Proof. intros n i.
  exact (Nat.lt_lt_succ_r _ _ (fin_proof i)).
Defined.

Lemma finSS_pf : forall (n:nat) (i:fin n), S i < S n.
Proof.
  intros n i. 
  rewrite<- Nat.succ_lt_mono.
  exact (fin_proof i).
Defined.

Lemma nat2finSi_fin2nat : forall (n : nat) (i : fin n),
  @nat2fin (S n) (S (fin2nat i)) (finSS_pf i) = finS i.
Proof.
  intros n i.
  (* destruct i *)
  destruct i as [i Hi].
  (* unfold the definitions of nat2fin and fin2nat *)
  unfold nat2fin, fin2nat, finS_pf, fin_proof, finS.
  (* simplify *)
  simpl. unfold nat2fin. (* [ | i] = [ | S i] *)
  apply exist_eq.
Qed.

Lemma fin_lt: forall n (x:fin n), n > 0 -> x < n.
Proof.
  induction n. lia. induction x. intros. simpl. lia.
Qed.

Lemma nat2fin_fin2finS : forall m x (i: fin x),
  S x <= m -> @nat2fin m (fin2finS i) = @nat2fin m i.
Proof. intros. simpl. auto. Qed.

Lemma fin2nat_fin2finS: forall(n:nat) (x:fin n) ,
  fin2nat(fin2finS x) = fin2nat x.
Proof.
  intros. unfold fin2finS. rewrite fin2nat_exist. auto.
Qed.

Lemma fin_le : forall m (i : fin m), i <= m.
Proof.
  intros m i.
  destruct i as [i_val i_proof]. (* decompose i into i_val and i_proof *)
  simpl. (* simplify the goal, replacing i by i_val *)
  apply Nat.lt_le_incl. (* use Nat.lt_le_incl to turn i_val < m into i_val <= m *)
  assumption. (* use i_proof as the hypothesis *)
Qed.

Lemma fin_lt': forall n m (i: fin n), n < m-> i < m.
Proof.
  intros. destruct H. destruct i.
+ simpl. lia.
+ destruct i. simpl.   lia.
Qed.

Definition fin0: option (fin 0) := None.

Definition fin_eq_dec {n : nat} (x y : fin n) : {x = y} + {x <> y}.
Proof.
  destruct x as [i Hi]. (* Destruct x into its components: i and Hi *)
  destruct y as [j Hj]. (* Destruct y into its components: j and Hj *)
  destruct (Nat.eq_dec i j) as [Hij | Hij]. (* Decide if i = j *)
  - (* Case 1: i = j *)
    left. (* Prove x = y *)
    subst j. (* Replace j with i *)
    f_equal. (* Prove the equality of the proofs *)
    apply proof_irrelevance. (* Proof irrelevance for Prop *)
  - (* Case 2: i <> j *)
    right. (* Prove x <> y *)
    intros Hxy. (* Assume x = y *)
    inversion Hxy. (* Derive a contradiction *)
    contradiction. (* i <> j contradicts Hij *)
Defined.


(* forall i:fin (S n), P i -> forall i: fin n, P i.  *)
Lemma finP_monotone : forall (n : nat) (P : fin (S n) -> Prop),
  (forall i : fin (S n), P i) -> (forall i : fin n, P (fin2finS i)).
Proof. intros. apply (H (fin2finS i)). Qed.

(* forall i:fin (S n), f i = v -> forall i: fin n, f i = v.  *)
Lemma finf_monotone {A:Set} : forall (n:nat) (f : fin (S n)->A) (v:A),
  (forall i : fin (S n), f i = v) -> 
  (forall i : fin n,     f (fin2finS i) = v).
Proof. intros. apply (H (fin2finS i)). Qed.

(** 2025.5.13 **)

(* injectivity proof of fin2nat *)
Lemma fin2nat_inj : forall n (i j : fin n), 
  fin2nat i = fin2nat j -> i = j.
Proof.
  intros n i j H. (* introduce variables and hypotheses *)
  apply fin_eq. (* apply fin_eq: fin2nat i = fin2nat j <-> i = j *)
  exact H.
Qed.


(* the fin2nat value of fin_last is n *)
Lemma fin_last_correct : forall n, fin2nat (fin_last n) = n.
Proof. 
  intros. unfold fin_last, nat2fin. 
  destruct (le_lt_dec n n).
  - simpl. reflexivity.
  - lia. (* impossible case *)
Qed.

Lemma fin_neq_last_implies_neq_n :
  forall n (i : fin (S n)),
    i <> fin_last n -> fin2nat i <> n.
Proof.
  intros n i H_ineq.
  (* use contradiction *)
  unfold not; intros H_eq.
  (* by fin_eq, equality of fin2nat means equality of the elements *)
  assert (i = fin_last n).
  {
    apply fin_eq. (* fin2nat i = fin2nat (fin_last n) -> i = fin_last n *)
    rewrite H_eq, fin_last_correct. 
    reflexivity.
  }
  contradiction.
Qed.

(*
Definition: convert an element of fin (S n) to fin n (excluding the
maximum-value case).
Condition: when i is not the last element (i.e. fin2nat i <> n).
*)
Definition fin2finS_inv {n} (i : fin (S n)) (H : fin2nat i <> n) : fin n :=
  finS2fin i H. (* directly reuse finS2fin from this file *)

Lemma fin2nat_fin2finS_inv :
  forall n (i : fin (S n)) (H : fin2nat i <> n),
    fin2nat (fin2finS_inv i H) = fin2nat i.
Proof.
  intros n i H.
  unfold fin2finS_inv. (* unfold the definition of fin2finS_inv *)
  unfold finS2fin.     (* unfold the definition of finS2fin *)
  simpl.               (* simplify the fin2nat and exist structure *)
  reflexivity.         (* fin2nat (exist _ (fin2nat i) _) = fin2nat i *)
Qed.

(** case analysis on finite sets *)
Lemma fin_case : 
  forall n (i : fin (S n)),
    (exists j : fin n, i = fin2finS j) \/ i = nat2fin (Nat.lt_succ_diag_r n).
Proof.
  intros. destruct (fin2nat i =? n)%nat eqn:E.
  - right. apply fin2nat_inj. rewrite fin2nat_nat2fin.
    apply Nat.eqb_eq in E. lia.
  - left. rewrite Nat.eqb_neq in E.    
    exists (fin2finS_inv i E). 
    apply fin2nat_inj. (* i = FS[ fin2finS_inv i E] *)
    reflexivity.
Qed.

End FiniteSetLt.


#[export] Hint Rewrite fin2nat_nat2fin fin_eq nat2fin_fin2finS
  fin2nat_fin2finS fin2nat_exist nat2fin_fin2nat: fcore.

#[export] Hint Resolve fin_finS_eq finltSn fin_lt fin_le fin_lt'
  fin2nat_exist fin2nat_fin2finS fin2nat_nat2fin fin_eq nat2fin_fin2finS : fcore .

Section FiniteSetLe.

(* define the finite domain fle n, the natural numbers less than or equal to n *)
Definition fle (n:nat) := { i:nat | i <= n }.

(* convert fle n to nat *)
Definition fle2nat {n:nat} (i:fle n) : nat :=
  match i with
  | exist _ i _ => i
  end.

Lemma fle_correct (n:nat) : forall (i:fle n), fle2nat i <= n.
Proof. intro. destruct i. exact l. Qed.

(* define the maximum value of fle n *)
Definition fle_self (n:nat) : fle n :=
  exist _ n (Nat.le_refl n).

(* define the minimum value of fle n *)
Definition fle_0 (n:nat) : fle n :=
  exist _ 0 (Peano.le_0_n n).

(* extract the proof from fle n *)
Definition fle_proof {n : nat} (i : fle n) : fle2nat i <= n :=
  match i with
  | exist _ _ H => H
  end.

Lemma fle2nat_fle_self: forall n,
  fle2nat (fle_self n) = n.
Proof.
  unfold fle_self. simpl. auto.
Qed. 

Lemma fle_eq : forall (n:nat) (i1 i2:fle n),
  fle2nat i1 = fle2nat i2 -> i1 = i2.
Proof.
  induction i1,i2.
  simpl. intros. subst. f_equal. apply le_unique.
Qed. 

Definition nat2fle (n:nat) (i:nat) :fle n :=
  match (le_lt_dec i n) with
  | left p => exist _ i p
  | right p => fle_0 n
  end.

Coercion fle2nat : fle >-> nat.

Lemma le_lt_trans: forall m n p,
  m < n -> n <= p -> m < p.
Proof. lia. Qed.

Lemma flePredAux : forall (i:nat) ,0 < i -> pred i < i.
Proof. intros. lia. Qed. 

(* i - 1 : fin i*)
Definition flePred {m:nat} (i:fle m) : option (fin i) :=
  match (fle2nat i) with
  | O => None
  | S i0 => Some (exist _ i0 (flePredAux (Nat.lt_0_succ i0) ))
  end.

Lemma flePred'Aux : forall m (i:fle m), pred i <= m.
Proof.
  induction i. simpl. lia.
Qed.

Definition flePred' {m:nat} (i:fle m) :fle m :=
  exist _ (pred i) (flePred'Aux i).

Lemma fle2nat_exist: forall m x p,
  @fle2nat m (exist (fun i : nat => i <= m) x p) = x.
Proof. simpl. auto. Qed.

End FiniteSetLe.

(* Lemma: fle 0 has a unique element *)  
Lemma fle_0_unique : forall (m1 m2 : fle 0), m1 = m2.  
Proof.  
  intros [i1 p1] [i2 p2]. (* Decompose m1 and m2 into their components *)  
  assert (i1 = 0). (* Prove that i1 must be 0 *)  
  {  
    apply Nat.le_0_r. (* i1 <= 0 implies i1 = 0 *)  
    exact p1.  
  }  
  assert (i2 = 0). (* Prove that i2 must be 0 *)  
  {  
    apply Nat.le_0_r. (* i2 <= 0 implies i2 = 0 *)  
    exact p2.  
  }  
  subst. (* Replace i1 and i2 with 0 *)  
  (* Now we need to show that the proofs p1 and p2 are equal *)  
  f_equal.
  apply proof_irrelevance.
Qed.

Definition fle0 : fle 0 := exist _ 0 (le_n 0).
(* Definition fle0 : fle 0 := [ | 0]. *)
Lemma fle0_unique : forall (m:fle 0), m = fle0.
Proof.
  intros [i p]. (* Decompose n into its components (i, p) *)  
  unfold fle0. simpl. (* Simplify the goal *)  
  assert (i = 0). (* Prove that i must be 0 *)  
  {  
    apply Nat.le_0_r. (* i <= 0 implies i = 0 *)  
    exact p.  
  }  
  subst. (* Replace i with 0 in the goal *)  
  f_equal. (* Reduce the goal to proving p = le_n 0 *)  
  apply proof_irrelevance. (* Proofs of 0 <= 0 are unique *)  
Qed.

#[export] Hint Rewrite fle_eq fle2nat_exist fle2nat_fle_self : fcore.

#[export] Hint Resolve fle_eq fle2nat_exist fle2nat_fle_self: fcore.

Lemma fin_exp_fle_aux : forall (i m n : nat),
  i < m -> m <= n -> i < n.
Proof. lia. Qed.

Definition fin_exp_fle {n} (m : fle n) (i : fin m)  :=
  nat2fin (fin_exp_fle_aux (fin_proof i) (fle_proof m)).

Lemma fin_exp_fle_eq :
  forall {n} (m1 m2 : fle n) (i1 : fin m1) (i2 : fin m2),
  (fin2nat i1) = (fin2nat i2)-> fin_exp_fle m1 i1 = fin_exp_fle m2 i2.
Proof.
  intros. apply fin_eq. auto.
Qed.

Lemma fin2nat_fin_exp_fle : forall {n} (m : fle n) (i : fin m),
  fin2nat (fin_exp_fle m i) = fin2nat i.
Proof.
  intros. unfold fin_exp_fle. apply fin2nat_nat2fin.
Qed.

Require Import Ascii String.

Section AuxFun.

Definition natToDigit (n : nat) : ascii :=
  match n with
    | 1 => "1"
    | 2 => "2"
    | 3 => "3"
    | 4 => "4"
    | 5 => "5"
    | 6 => "6"
    | 7 => "7"
    | 8 => "8"  
    | 9 => "9"
    | _ => "0"
  end.

Fixpoint writeNatAux (time n : nat) (acc : string) : string :=
  let acc' := String (natToDigit (n mod 10)) acc in
  match time with
    | 0 => acc'
    | S time' =>
      match n / 10 with
        | 0 => acc'
        | n' => writeNatAux time' n' acc'
      end
  end.

Definition writeNat (n : nat) : string :=
  writeNatAux n n "".

End AuxFun.

Search ( nat2fin ?m ?i = nat2fin ?m ?i').

