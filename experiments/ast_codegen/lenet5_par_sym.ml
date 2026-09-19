
type __ = Obj.t
let __ = let rec f _ = Obj.repr f in Obj.repr f

type unit0 =
| Tt

type bool =
| True
| False

type nat =
| O
| S of nat
| Var of string
| Add of nat * nat
| Mul of nat * nat
| Sub of nat * nat
| Div of nat * nat
| Mod of nat * nat

let rec nat_is_concrete = function
  | O -> true
  | S n -> nat_is_concrete n
  | _ -> false

let rec nat_to_int = function
  | O -> 0
  | S n -> 1 + nat_to_int n
  | _ -> 0

let int_to_nat i =
  let rec go = function
    | 0 -> O
    | n -> S (go (n - 1))
  in
  if i <= 0 then O else go i

let rec writeNatSym n =
  match n with
  | O -> "0"
  | S O -> "1"
  | S (S O) -> "2"
  | S (S (S O)) -> "3"
  | S (S (S (S O))) -> "4"
  | S (S (S (S (S O)))) -> "5"
  | S (S (S (S (S (S O))))) -> "6"
  | S (S (S (S (S (S (S O)))))) -> "7"
  | S (S (S (S (S (S (S (S O))))))) -> "8"
  | S (S (S (S (S (S (S (S (S O)))))))) -> "9"
  | S n' when nat_is_concrete n -> string_of_int (nat_to_int n)
  | Var s -> s
  | Add (a, b) -> "(" ^ writeNatSym a ^ " + " ^ writeNatSym b ^ ")"
  | Mul (a, b) -> "(" ^ writeNatSym a ^ " * " ^ writeNatSym b ^ ")"
  | Sub (a, b) -> "(" ^ writeNatSym a ^ " - " ^ writeNatSym b ^ ")"
  | Div (a, b) -> "(" ^ writeNatSym a ^ " / " ^ writeNatSym b ^ ")"
  | Mod (a, b) -> "(" ^ writeNatSym a ^ " % " ^ writeNatSym b ^ ")"
  | _ -> "NAT"



type ('a, 'b) prod = 'a * 'b

(** val fst : ('a1, 'a2) prod -> 'a1 **)

let fst = function
| (x, _) -> x

(** val snd : ('a1, 'a2) prod -> 'a2 **)

let snd = function
| (_, y) -> y

type comparison =
| Eq
| Lt
| Gt

(** val compOpp : comparison -> comparison **)

let compOpp = function
| Eq -> Eq
| Lt -> Gt
| Gt -> Lt

type 'a sig0 = 'a
  (* singleton inductive, whose constructor was exist *)

type sumbool =
| Left
| Right

type 'a sumor =
| Inleft of 'a
| Inright

(** val pred : nat -> nat **)

let pred n = match n with
| O -> n
| S u -> u

module Coq__1 = struct
 (** val add : nat -> nat -> nat **)

 let rec add n m =
   match (n, m) with
   | (O, x) | (x, O) -> x
   | (S p, S q) -> S (S (add p q))
   | (S p, x) -> S (add p x)
   | (x, S q) -> S (add x q)
   | _ -> Add (n, m)
end
include Coq__1

(** val mul : nat -> nat -> nat **)

let rec mul n m =
  match (n, m) with
  | (O, _) | (_, O) -> O
  | (S O, x) | (x, S O) -> x
  | (S p, S q) -> add (S q) (mul p (S q))
  | _ -> Mul (n, m)

(** val eqb : bool -> bool -> bool **)

let eqb b1 b2 =
  match b1 with
  | True -> b2
  | False -> (match b2 with
              | True -> False
              | False -> True)

module Nat =
 struct
  (** val add : nat -> nat -> nat **)

  let rec add n m =
    match (n, m) with
    | (O, x) | (x, O) -> x
    | (S p, S q) -> S (S (add p q))
    | (S p, x) -> S (add p x)
    | (x, S q) -> S (add x q)
    | _ -> Add (n, m)

  (** val mul : nat -> nat -> nat **)

  let rec mul n m =
    match (n, m) with
    | (O, _) | (_, O) -> O
    | (S O, x) | (x, S O) -> x
    | (S p, S q) -> add (S q) (mul p (S q))
    | _ -> Mul (n, m)

  (** val sub : nat -> nat -> nat **)

  let rec sub n m =
    match n with
    | O -> n
    | S k -> (match m with
              | O -> n
              | S l -> sub k l)

  (** val eqb : nat -> nat -> bool **)

  let rec eqb n m =
    match (n, m) with
    | (O, O) -> True
    | (S n', S m') -> eqb n' m'
    | _ when nat_is_concrete n && nat_is_concrete m -> False
    | _ -> False

  (** val pow : nat -> nat -> nat **)

  let rec pow n = function
  | O -> S O
  | S m0 -> mul n (pow n m0)

  (** val divmod : nat -> nat -> nat -> nat -> (nat, nat) prod **)

  let rec divmod x y q0 u =
    match x with
    | O -> (q0, u)
    | S x' ->
      (match u with
       | O -> divmod x' y (S q0) y
       | S u' -> divmod x' y q0 u')

  (** val div : nat -> nat -> nat **)

  let div x y =
  match (x, y) with
  | (_, O) -> O
  | _ when nat_is_concrete x && nat_is_concrete y ->
      let q, _ = divmod x (pred y) O (pred y) in q
  | _ -> Div (x, y)

  (** val modulo : nat -> nat -> nat **)

  let modulo x y =
  match y with
  | O -> x
  | _ when nat_is_concrete x && nat_is_concrete y ->
      sub (pred y) (snd (divmod x (pred y) O (pred y)))
  | _ -> Mod (x, y)
 end

type positive =
| XI of positive
| XO of positive
| XH

type z =
| Z0
| Zpos of positive
| Zneg of positive

module Pos =
 struct
  (** val succ : positive -> positive **)

  let rec succ = function
  | XI p -> XO (succ p)
  | XO p -> XI p
  | XH -> XO XH

  (** val add : positive -> positive -> positive **)

  let rec add x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> XO (add_carry p q0)
       | XO q0 -> XI (add p q0)
       | XH -> XO (succ p))
    | XO p ->
      (match y with
       | XI q0 -> XI (add p q0)
       | XO q0 -> XO (add p q0)
       | XH -> XI p)
    | XH -> (match y with
             | XI q0 -> XO (succ q0)
             | XO q0 -> XI q0
             | XH -> XO XH)

  (** val add_carry : positive -> positive -> positive **)

  and add_carry x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> XI (add_carry p q0)
       | XO q0 -> XO (add_carry p q0)
       | XH -> XI (succ p))
    | XO p ->
      (match y with
       | XI q0 -> XO (add_carry p q0)
       | XO q0 -> XI (add p q0)
       | XH -> XO (succ p))
    | XH ->
      (match y with
       | XI q0 -> XI (succ q0)
       | XO q0 -> XO (succ q0)
       | XH -> XI XH)

  (** val pred_double : positive -> positive **)

  let rec pred_double = function
  | XI p -> XI (XO p)
  | XO p -> XI (pred_double p)
  | XH -> XH

  type mask =
  | IsNul
  | IsPos of positive
  | IsNeg

  (** val mul : positive -> positive -> positive **)

  let rec mul x y =
    match x with
    | XI p -> add y (XO (mul p y))
    | XO p -> XO (mul p y)
    | XH -> y

  (** val compare_cont : comparison -> positive -> positive -> comparison **)

  let rec compare_cont r x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> compare_cont r p q0
       | XO q0 -> compare_cont Gt p q0
       | XH -> Gt)
    | XO p ->
      (match y with
       | XI q0 -> compare_cont Lt p q0
       | XO q0 -> compare_cont r p q0
       | XH -> Gt)
    | XH -> (match y with
             | XH -> r
             | _ -> Lt)

  (** val compare : positive -> positive -> comparison **)

  let compare =
    compare_cont Eq

  (** val iter_op : ('a1 -> 'a1 -> 'a1) -> positive -> 'a1 -> 'a1 **)

  let rec iter_op op p a =
    match p with
    | XI p0 -> op a (iter_op op p0 (op a a))
    | XO p0 -> iter_op op p0 (op a a)
    | XH -> a

  (** val to_nat : positive -> nat **)

  let to_nat x =
    iter_op Coq__1.add x (S O)

  (** val of_succ_nat : nat -> positive **)

  let rec of_succ_nat = function
  | O -> XH
  | S x -> succ (of_succ_nat x)
 end

module Coq_Pos =
 struct
  (** val succ : positive -> positive **)

  let rec succ = function
  | XI p -> XO (succ p)
  | XO p -> XI p
  | XH -> XO XH

  (** val add : positive -> positive -> positive **)

  let rec add x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> XO (add_carry p q0)
       | XO q0 -> XI (add p q0)
       | XH -> XO (succ p))
    | XO p ->
      (match y with
       | XI q0 -> XI (add p q0)
       | XO q0 -> XO (add p q0)
       | XH -> XI p)
    | XH -> (match y with
             | XI q0 -> XO (succ q0)
             | XO q0 -> XI q0
             | XH -> XO XH)

  (** val add_carry : positive -> positive -> positive **)

  and add_carry x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> XI (add_carry p q0)
       | XO q0 -> XO (add_carry p q0)
       | XH -> XI (succ p))
    | XO p ->
      (match y with
       | XI q0 -> XO (add_carry p q0)
       | XO q0 -> XI (add p q0)
       | XH -> XO (succ p))
    | XH ->
      (match y with
       | XI q0 -> XI (succ q0)
       | XO q0 -> XO (succ q0)
       | XH -> XI XH)

  (** val pred_double : positive -> positive **)

  let rec pred_double = function
  | XI p -> XI (XO p)
  | XO p -> XI (pred_double p)
  | XH -> XH

  type mask = Pos.mask =
  | IsNul
  | IsPos of positive
  | IsNeg

  (** val succ_double_mask : mask -> mask **)

  let succ_double_mask = function
  | IsNul -> IsPos XH
  | IsPos p -> IsPos (XI p)
  | IsNeg -> IsNeg

  (** val double_mask : mask -> mask **)

  let double_mask = function
  | IsPos p -> IsPos (XO p)
  | x0 -> x0

  (** val double_pred_mask : positive -> mask **)

  let double_pred_mask = function
  | XI p -> IsPos (XO (XO p))
  | XO p -> IsPos (XO (pred_double p))
  | XH -> IsNul

  (** val sub_mask : positive -> positive -> mask **)

  let rec sub_mask x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> double_mask (sub_mask p q0)
       | XO q0 -> succ_double_mask (sub_mask p q0)
       | XH -> IsPos (XO p))
    | XO p ->
      (match y with
       | XI q0 -> succ_double_mask (sub_mask_carry p q0)
       | XO q0 -> double_mask (sub_mask p q0)
       | XH -> IsPos (pred_double p))
    | XH -> (match y with
             | XH -> IsNul
             | _ -> IsNeg)

  (** val sub_mask_carry : positive -> positive -> mask **)

  and sub_mask_carry x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> succ_double_mask (sub_mask_carry p q0)
       | XO q0 -> double_mask (sub_mask p q0)
       | XH -> IsPos (pred_double p))
    | XO p ->
      (match y with
       | XI q0 -> double_mask (sub_mask_carry p q0)
       | XO q0 -> succ_double_mask (sub_mask_carry p q0)
       | XH -> double_pred_mask p)
    | XH -> IsNeg

  (** val sub : positive -> positive -> positive **)

  let sub x y =
    match sub_mask x y with
    | IsPos z0 -> z0
    | _ -> XH

  (** val mul : positive -> positive -> positive **)

  let rec mul x y =
    match x with
    | XI p -> add y (XO (mul p y))
    | XO p -> XO (mul p y)
    | XH -> y

  (** val compare_cont : comparison -> positive -> positive -> comparison **)

  let rec compare_cont r x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> compare_cont r p q0
       | XO q0 -> compare_cont Gt p q0
       | XH -> Gt)
    | XO p ->
      (match y with
       | XI q0 -> compare_cont Lt p q0
       | XO q0 -> compare_cont r p q0
       | XH -> Gt)
    | XH -> (match y with
             | XH -> r
             | _ -> Lt)

  (** val compare : positive -> positive -> comparison **)

  let compare =
    compare_cont Eq

  (** val iter_op : ('a1 -> 'a1 -> 'a1) -> positive -> 'a1 -> 'a1 **)

  let rec iter_op op p a =
    match p with
    | XI p0 -> op a (iter_op op p0 (op a a))
    | XO p0 -> iter_op op p0 (op a a)
    | XH -> a

  (** val to_nat : positive -> nat **)

  let to_nat x =
    iter_op Coq__1.add x (S O)

  (** val size_nat : positive -> nat **)

  let rec size_nat = function
  | XI p0 -> S (size_nat p0)
  | XO p0 -> S (size_nat p0)
  | XH -> S O

  (** val ggcdn :
      nat -> positive -> positive -> (positive, (positive, positive) prod)
      prod **)

  let rec ggcdn n a b =
    match n with
    | O -> (XH, ((a, b)))
    | S n0 ->
      (match a with
       | XI a' ->
         (match b with
          | XI b' ->
            (match compare a' b' with
             | Eq -> (a, ((XH, XH)))
             | Lt ->
               let (g, p) = ggcdn n0 (sub b' a') a in
               let (ba, aa) = p in
               (g, ((aa, (add aa (XO ba)))))
             | Gt ->
               let (g, p) = ggcdn n0 (sub a' b') b in
               let (ab, bb) = p in
               (g, (((add bb (XO ab)), bb))))
          | XO b0 ->
            let (g, p) = ggcdn n0 a b0 in
            let (aa, bb) = p in (g, ((aa, (XO bb))))
          | XH -> (XH, ((a, XH))))
       | XO a0 ->
         (match b with
          | XI _ ->
            let (g, p) = ggcdn n0 a0 b in
            let (aa, bb) = p in (g, (((XO aa), bb)))
          | XO b0 -> let (g, p) = ggcdn n0 a0 b0 in ((XO g), p)
          | XH -> (XH, ((a, XH))))
       | XH -> (XH, ((XH, b))))

  (** val ggcd :
      positive -> positive -> (positive, (positive, positive) prod) prod **)

  let ggcd a b =
    ggcdn (Coq__1.add (size_nat a) (size_nat b)) a b

  (** val of_nat : nat -> positive **)

  let rec of_nat = function
  | O -> XH
  | S x -> (match x with
            | O -> XH
            | S _ -> succ (of_nat x))
 end

module Z =
 struct
  (** val double : z -> z **)

  let double = function
  | Z0 -> Z0
  | Zpos p -> Zpos (XO p)
  | Zneg p -> Zneg (XO p)

  (** val succ_double : z -> z **)

  let succ_double = function
  | Z0 -> Zpos XH
  | Zpos p -> Zpos (XI p)
  | Zneg p -> Zneg (Pos.pred_double p)

  (** val pred_double : z -> z **)

  let pred_double = function
  | Z0 -> Zneg XH
  | Zpos p -> Zpos (Pos.pred_double p)
  | Zneg p -> Zneg (XI p)

  (** val pos_sub : positive -> positive -> z **)

  let rec pos_sub x y =
    match x with
    | XI p ->
      (match y with
       | XI q0 -> double (pos_sub p q0)
       | XO q0 -> succ_double (pos_sub p q0)
       | XH -> Zpos (XO p))
    | XO p ->
      (match y with
       | XI q0 -> pred_double (pos_sub p q0)
       | XO q0 -> double (pos_sub p q0)
       | XH -> Zpos (Pos.pred_double p))
    | XH ->
      (match y with
       | XI q0 -> Zneg (XO q0)
       | XO q0 -> Zneg (Pos.pred_double q0)
       | XH -> Z0)

  (** val add : z -> z -> z **)

  let add x y =
    match x with
    | Z0 -> y
    | Zpos x' ->
      (match y with
       | Z0 -> x
       | Zpos y' -> Zpos (Pos.add x' y')
       | Zneg y' -> pos_sub x' y')
    | Zneg x' ->
      (match y with
       | Z0 -> x
       | Zpos y' -> pos_sub y' x'
       | Zneg y' -> Zneg (Pos.add x' y'))

  (** val opp : z -> z **)

  let opp = function
  | Z0 -> Z0
  | Zpos x0 -> Zneg x0
  | Zneg x0 -> Zpos x0

  (** val sub : z -> z -> z **)

  let sub m n =
    add m (opp n)

  (** val mul : z -> z -> z **)

  let mul x y =
    match x with
    | Z0 -> Z0
    | Zpos x' ->
      (match y with
       | Z0 -> Z0
       | Zpos y' -> Zpos (Pos.mul x' y')
       | Zneg y' -> Zneg (Pos.mul x' y'))
    | Zneg x' ->
      (match y with
       | Z0 -> Z0
       | Zpos y' -> Zneg (Pos.mul x' y')
       | Zneg y' -> Zpos (Pos.mul x' y'))

  (** val compare : z -> z -> comparison **)

  let compare x y =
    match x with
    | Z0 -> (match y with
             | Z0 -> Eq
             | Zpos _ -> Lt
             | Zneg _ -> Gt)
    | Zpos x' -> (match y with
                  | Zpos y' -> Pos.compare x' y'
                  | _ -> Gt)
    | Zneg x' ->
      (match y with
       | Zneg y' -> compOpp (Pos.compare x' y')
       | _ -> Lt)

  (** val max : z -> z -> z **)

  let max n m =
    match compare n m with
    | Lt -> m
    | _ -> n

  (** val to_nat : z -> nat **)

  let to_nat = function
  | Zpos p -> Pos.to_nat p
  | _ -> O

  (** val of_nat : nat -> z **)

  let of_nat = function
  | O -> Z0
  | S n0 -> Zpos (Pos.of_succ_nat n0)

  (** val to_pos : z -> positive **)

  let to_pos = function
  | Zpos p -> p
  | _ -> XH

  (** val sgn : z -> z **)

  let sgn = function
  | Z0 -> Z0
  | Zpos _ -> Zpos XH
  | Zneg _ -> Zneg XH

  (** val abs : z -> z **)

  let abs = function
  | Zneg p -> Zpos p
  | x -> x

  (** val ggcd : z -> z -> (z, (z, z) prod) prod **)

  let ggcd a b =
    match a with
    | Z0 -> ((abs b), ((Z0, (sgn b))))
    | Zpos a0 ->
      (match b with
       | Z0 -> ((abs a), (((sgn a), Z0)))
       | Zpos b0 ->
         let (g, p) = Coq_Pos.ggcd a0 b0 in
         let (aa, bb) = p in
         ((Zpos g), (((Zpos aa), (Zpos bb))))
       | Zneg b0 ->
         let (g, p) = Coq_Pos.ggcd a0 b0 in
         let (aa, bb) = p in
         ((Zpos g), (((Zpos aa), (Zneg bb)))))
    | Zneg a0 ->
      (match b with
       | Z0 -> ((abs a), (((sgn a), Z0)))
       | Zpos b0 ->
         let (g, p) = Coq_Pos.ggcd a0 b0 in
         let (aa, bb) = p in
         ((Zpos g), (((Zneg aa), (Zpos bb))))
       | Zneg b0 ->
         let (g, p) = Coq_Pos.ggcd a0 b0 in
         let (aa, bb) = p in
         ((Zpos g), (((Zneg aa), (Zneg bb)))))
 end

(** val pow_pos : ('a1 -> 'a1 -> 'a1) -> 'a1 -> positive -> 'a1 **)

let rec pow_pos rmul x = function
| XI i0 -> let p = pow_pos rmul x i0 in rmul x (rmul p p)
| XO i0 -> let p = pow_pos rmul x i0 in rmul p p
| XH -> x

(** val z_lt_dec : z -> z -> sumbool **)

let z_lt_dec x y =
  match Z.compare x y with
  | Lt -> Left
  | _ -> Right

(** val z_lt_ge_dec : z -> z -> sumbool **)

let z_lt_ge_dec =
  z_lt_dec

(** val z_lt_le_dec : z -> z -> sumbool **)

let z_lt_le_dec =
  z_lt_ge_dec

type q = { qnum : z; qden : positive }

(** val qplus : q -> q -> q **)

let qplus x y =
  { qnum = (Z.add (Z.mul x.qnum (Zpos y.qden)) (Z.mul y.qnum (Zpos x.qden)));
    qden = (Coq_Pos.mul x.qden y.qden) }

(** val qmult : q -> q -> q **)

let qmult x y =
  { qnum = (Z.mul x.qnum y.qnum); qden = (Coq_Pos.mul x.qden y.qden) }

(** val qopp : q -> q **)

let qopp x =
  { qnum = (Z.opp x.qnum); qden = x.qden }

(** val qminus : q -> q -> q **)

let qminus x y =
  qplus x (qopp y)

(** val qinv : q -> q **)

let qinv x =
  match x.qnum with
  | Z0 -> { qnum = Z0; qden = XH }
  | Zpos p -> { qnum = (Zpos x.qden); qden = p }
  | Zneg p -> { qnum = (Zneg x.qden); qden = p }

(** val qlt_le_dec : q -> q -> sumbool **)

let qlt_le_dec x y =
  z_lt_le_dec (Z.mul x.qnum (Zpos y.qden)) (Z.mul y.qnum (Zpos x.qden))

(** val qarchimedean : q -> positive **)

let qarchimedean q0 =
  let { qnum = qnum0; qden = _ } = q0 in
  (match qnum0 with
   | Zpos p -> Coq_Pos.add p XH
   | _ -> XH)

(** val qpower_positive : q -> positive -> q **)

let qpower_positive =
  pow_pos qmult

(** val qpower : q -> z -> q **)

let qpower q0 = function
| Z0 -> { qnum = (Zpos XH); qden = XH }
| Zpos p -> qpower_positive q0 p
| Zneg p -> qinv (qpower_positive q0 p)

(** val qred : q -> q **)

let qred q0 =
  let { qnum = q1; qden = q2 } = q0 in
  let (r1, r2) = snd (Z.ggcd q1 (Zpos q2)) in
  { qnum = r1; qden = (Z.to_pos r2) }

(** val qabs : q -> q **)

let qabs x =
  let { qnum = n; qden = d } = x in { qnum = (Z.abs n); qden = d }

(** val pos_log2floor_plus1 : positive -> positive **)

let rec pos_log2floor_plus1 = function
| XI p' -> Coq_Pos.succ (pos_log2floor_plus1 p')
| XO p' -> Coq_Pos.succ (pos_log2floor_plus1 p')
| XH -> XH

(** val qbound_lt_ZExp2 : q -> z **)

let qbound_lt_ZExp2 q0 =
  match q0.qnum with
  | Z0 -> Zneg (XO (XO (XO (XI (XO (XI (XI (XI (XI XH)))))))))
  | Zpos p ->
    Z.pos_sub (Coq_Pos.succ (pos_log2floor_plus1 p))
      (pos_log2floor_plus1 q0.qden)
  | Zneg _ -> Z0

(** val qbound_ltabs_ZExp2 : q -> z **)

let qbound_ltabs_ZExp2 q0 =
  qbound_lt_ZExp2 (qabs q0)

type cReal = { seq : (z -> q); scale : z }

(** val inject_Q : q -> cReal **)

let inject_Q q0 =
  { seq = (fun _ -> q0); scale = (qbound_ltabs_ZExp2 q0) }

(** val cReal_plus_seq : cReal -> cReal -> z -> q **)

let cReal_plus_seq x y n =
  qred (qplus (x.seq (Z.sub n (Zpos XH))) (y.seq (Z.sub n (Zpos XH))))

(** val cReal_plus_scale : cReal -> cReal -> z **)

let cReal_plus_scale x y =
  Z.add (Z.max x.scale y.scale) (Zpos XH)

(** val cReal_plus : cReal -> cReal -> cReal **)

let cReal_plus x y =
  { seq = (cReal_plus_seq x y); scale = (cReal_plus_scale x y) }

(** val cReal_opp_seq : cReal -> z -> q **)

let cReal_opp_seq x n =
  qopp (x.seq n)

(** val cReal_opp_scale : cReal -> z **)

let cReal_opp_scale x =
  x.scale

(** val cReal_opp : cReal -> cReal **)

let cReal_opp x =
  { seq = (cReal_opp_seq x); scale = (cReal_opp_scale x) }

(** val cReal_mult_seq : cReal -> cReal -> z -> q **)

let cReal_mult_seq x y n =
  qmult (x.seq (Z.sub (Z.sub n y.scale) (Zpos XH)))
    (y.seq (Z.sub (Z.sub n x.scale) (Zpos XH)))

(** val cReal_mult_scale : cReal -> cReal -> z **)

let cReal_mult_scale x y =
  Z.add x.scale y.scale

(** val cReal_mult : cReal -> cReal -> cReal **)

let cReal_mult x y =
  { seq = (cReal_mult_seq x y); scale = (cReal_mult_scale x y) }

(** val sig_forall_dec : (nat -> sumbool) -> nat sumor **)

let sig_forall_dec _ =
  failwith "AXIOM TO BE REALIZED (Stdlib.Reals.ClassicalDedekindReals.sig_forall_dec)"

(** val lowerCutBelow : (q -> bool) -> q **)

let lowerCutBelow f =
  let s =
    sig_forall_dec (fun n ->
      let b = f (qopp { qnum = (Z.of_nat n); qden = XH }) in
      (match b with
       | True -> Right
       | False -> Left))
  in
  (match s with
   | Inleft s0 -> qopp { qnum = (Z.of_nat s0); qden = XH }
   | Inright -> assert false (* absurd case *))

(** val lowerCutAbove : (q -> bool) -> q **)

let lowerCutAbove f =
  let s =
    sig_forall_dec (fun n ->
      let b = f { qnum = (Z.of_nat n); qden = XH } in
      (match b with
       | True -> Left
       | False -> Right))
  in
  (match s with
   | Inleft s0 -> { qnum = (Z.of_nat s0); qden = XH }
   | Inright -> assert false (* absurd case *))

type dReal = (q -> bool)

(** val dRealQlim_rec : (q -> bool) -> nat -> nat -> q **)

let rec dRealQlim_rec f n = function
| O -> assert false (* absurd case *)
| S n0 ->
  let b =
    f
      (qplus (lowerCutBelow f) { qnum = (Z.of_nat n0); qden =
        (Coq_Pos.of_nat (S n)) })
  in
  (match b with
   | True ->
     qplus (lowerCutBelow f) { qnum = (Z.of_nat n0); qden =
       (Coq_Pos.of_nat (S n)) }
   | False -> dRealQlim_rec f n n0)

(** val dRealAbstr : cReal -> dReal **)

let dRealAbstr x =
  let q1 = x.seq (Zpos XH) in
  let q2 = x.seq (Zpos (XO XH)) in
  if q1 = q2 then
    (fun q0 ->
      match qlt_le_dec q0 q1 with
      | Left -> True
      | Right -> False)
  else
    let h = fun q0 n ->
      let s =
        qlt_le_dec
          (qplus q0
            (qpower { qnum = (Zpos (XO XH)); qden = XH } (Z.opp (Z.of_nat n))))
          (x.seq (Z.opp (Z.of_nat n)))
      in
      (match s with
       | Left -> Right
       | Right -> Left)
    in
    (fun q0 ->
    match sig_forall_dec (h q0) with
    | Inleft _ -> True
    | Inright -> False)

(** val dRealQlim : dReal -> nat -> q **)

let dRealQlim x n =
  let s = lowerCutAbove x in
  let s0 = qarchimedean (qminus s (lowerCutBelow x)) in
  dRealQlim_rec x n (mul (S n) (Coq_Pos.to_nat s0))

(** val dRealQlimExp2 : dReal -> nat -> q **)

let dRealQlimExp2 x n =
  dRealQlim x (pred (Nat.pow (S (S O)) n))

(** val cReal_of_DReal_seq : dReal -> z -> q **)

let cReal_of_DReal_seq x n =
  dRealQlimExp2 x (Z.to_nat (Z.opp n))

(** val cReal_of_DReal_scale : dReal -> z **)

let cReal_of_DReal_scale x =
  qbound_lt_ZExp2
    (qplus (qabs (cReal_of_DReal_seq x (Zneg XH))) { qnum = (Zpos (XO XH));
      qden = XH })

(** val dRealRepr : dReal -> cReal **)

let dRealRepr x =
  { seq = (cReal_of_DReal_seq x); scale = (cReal_of_DReal_scale x) }

module type RbaseSymbolsSig =
 sig
  type coq_R

  val coq_Rabst : cReal -> coq_R

  val coq_Rrepr : coq_R -> cReal

  val coq_R0 : coq_R

  val coq_R1 : coq_R

  val coq_Rplus : coq_R -> coq_R -> coq_R

  val coq_Rmult : coq_R -> coq_R -> coq_R

  val coq_Ropp : coq_R -> coq_R
 end

module RbaseSymbolsImpl =
 struct
  type coq_R = dReal

  (** val coq_Rabst : cReal -> dReal **)

  let coq_Rabst =
    dRealAbstr

  (** val coq_Rrepr : dReal -> cReal **)

  let coq_Rrepr =
    dRealRepr

  (** val coq_Rquot1 : __ **)

  let coq_Rquot1 =
    __

  (** val coq_Rquot2 : __ **)

  let coq_Rquot2 =
    __

  (** val coq_R0 : coq_R **)

  let coq_R0 =
    coq_Rabst (inject_Q { qnum = Z0; qden = XH })

  (** val coq_R1 : coq_R **)

  let coq_R1 =
    coq_Rabst (inject_Q { qnum = (Zpos XH); qden = XH })

  (** val coq_Rplus : coq_R -> coq_R -> coq_R **)

  let coq_Rplus x y =
    coq_Rabst (cReal_plus (coq_Rrepr x) (coq_Rrepr y))

  (** val coq_Rmult : coq_R -> coq_R -> coq_R **)

  let coq_Rmult x y =
    coq_Rabst (cReal_mult (coq_Rrepr x) (coq_Rrepr y))

  (** val coq_Ropp : coq_R -> coq_R **)

  let coq_Ropp x =
    coq_Rabst (cReal_opp (coq_Rrepr x))

  type coq_Rlt = __

  (** val coq_R0_def : __ **)

  let coq_R0_def =
    __

  (** val coq_R1_def : __ **)

  let coq_R1_def =
    __

  (** val coq_Rplus_def : __ **)

  let coq_Rplus_def =
    __

  (** val coq_Rmult_def : __ **)

  let coq_Rmult_def =
    __

  (** val coq_Ropp_def : __ **)

  let coq_Ropp_def =
    __

  (** val coq_Rlt_def : __ **)

  let coq_Rlt_def =
    __
 end

type ascii =
| Ascii of bool * bool * bool * bool * bool * bool * bool * bool

(** val eqb0 : ascii -> ascii -> bool **)

let eqb0 a b =
  let Ascii (a0, a1, a2, a3, a4, a5, a6, a7) = a in
  let Ascii (b0, b1, b2, b3, b4, b5, b6, b7) = b in
  (match match match match match match match eqb a0 b0 with
                                       | True -> eqb a1 b1
                                       | False -> False with
                                 | True -> eqb a2 b2
                                 | False -> False with
                           | True -> eqb a3 b3
                           | False -> False with
                     | True -> eqb a4 b4
                     | False -> False with
               | True -> eqb a5 b5
               | False -> False with
         | True -> eqb a6 b6
         | False -> False with
   | True -> eqb a7 b7
   | False -> False)

type coq_string =
| EmptyString
| String of ascii * coq_string

let ascii_to_char (Ascii (b0, b1, b2, b3, b4, b5, b6, b7)) =
  Char.chr ((match b0 with True -> 1 | False -> 0) +
            (match b1 with True -> 2 | False -> 0) +
            (match b2 with True -> 4 | False -> 0) +
            (match b3 with True -> 8 | False -> 0) +
            (match b4 with True -> 16 | False -> 0) +
            (match b5 with True -> 32 | False -> 0) +
            (match b6 with True -> 64 | False -> 0) +
            (match b7 with True -> 128 | False -> 0))

let bool_of_int_bit code i = if code land (1 lsl i) <> 0 then True else False

let ascii_of_char c =
  let code = Char.code c in
  Ascii (bool_of_int_bit code 0, bool_of_int_bit code 1, bool_of_int_bit code 2,
         bool_of_int_bit code 3, bool_of_int_bit code 4, bool_of_int_bit code 5,
         bool_of_int_bit code 6, bool_of_int_bit code 7)

let rec string_to_coq_string s =
  let n = String.length s in
  let rec go i =
    if i >= n then EmptyString
    else String (ascii_of_char (String.get s i), go (i + 1))
  in
  go 0

let rec coq_string_to_string = function
  | EmptyString -> ""
  | String (a, s) -> String.make 1 (ascii_to_char a) ^ coq_string_to_string s



(** val eqb1 : string -> string -> bool **)

let rec eqb1 s1 s2 =
  match s1 with
  | EmptyString ->
    (match s2 with
     | EmptyString -> True
     | String (_, _) -> False)
  | String (c1, s1') ->
    (match s2 with
     | EmptyString -> False
     | String (c2, s2') ->
       (match eqb0 c1 c2 with
        | True -> eqb1 s1' s2'
        | False -> False))

(** val append : string -> string -> string **)

let rec append s1 s2 =
  match s1 with
  | EmptyString -> s2
  | String (c, s1') -> String (c, (append s1' s2))

type fin = nat

(** val fin2nat : nat -> fin -> nat **)

let fin2nat _ i =
  i

(** val fin2finS : nat -> fin -> fin **)

let fin2finS =
  fin2nat

(** val nat2fin : nat -> nat -> fin **)

let nat2fin _ i =
  i

type fle = nat

(** val fle2nat : nat -> fle -> nat **)

let fle2nat _ i =
  i

(** val fle_self : nat -> fle **)

let fle_self n =
  n

(** val natToDigit : nat -> ascii **)

let natToDigit = function
| O -> Ascii (False, False, False, False, True, True, False, False)
| S n0 ->
  (match n0 with
   | O -> Ascii (True, False, False, False, True, True, False, False)
   | S n1 ->
     (match n1 with
      | O -> Ascii (False, True, False, False, True, True, False, False)
      | S n2 ->
        (match n2 with
         | O -> Ascii (True, True, False, False, True, True, False, False)
         | S n3 ->
           (match n3 with
            | O -> Ascii (False, False, True, False, True, True, False, False)
            | S n4 ->
              (match n4 with
               | O ->
                 Ascii (True, False, True, False, True, True, False, False)
               | S n5 ->
                 (match n5 with
                  | O ->
                    Ascii (False, True, True, False, True, True, False, False)
                  | S n6 ->
                    (match n6 with
                     | O ->
                       Ascii (True, True, True, False, True, True, False,
                         False)
                     | S n7 ->
                       (match n7 with
                        | O ->
                          Ascii (False, False, False, True, True, True,
                            False, False)
                        | S n8 ->
                          (match n8 with
                           | O ->
                             Ascii (True, False, False, True, True, True,
                               False, False)
                           | S _ ->
                             Ascii (False, False, False, False, True, True,
                               False, False))))))))))

(** val writeNatAux : nat -> nat -> string -> string **)

let rec writeNatSym n =
  match n with
  | O -> "0"
  | S O -> "1"
  | S (S O) -> "2"
  | S (S (S O)) -> "3"
  | S (S (S (S O))) -> "4"
  | S (S (S (S (S O)))) -> "5"
  | S (S (S (S (S (S O))))) -> "6"
  | S (S (S (S (S (S (S O)))))) -> "7"
  | S (S (S (S (S (S (S (S O))))))) -> "8"
  | S (S (S (S (S (S (S (S (S O)))))))) -> "9"
  | S n' when nat_is_concrete n -> string_of_int (nat_to_int n)
  | Var s -> s
  | Add (a, b) -> "(" ^ writeNatSym a ^ " + " ^ writeNatSym b ^ ")"
  | Mul (a, b) -> "(" ^ writeNatSym a ^ " * " ^ writeNatSym b ^ ")"
  | Sub (a, b) -> "(" ^ writeNatSym a ^ " - " ^ writeNatSym b ^ ")"
  | Div (a, b) -> "(" ^ writeNatSym a ^ " / " ^ writeNatSym b ^ ")"
  | Mod (a, b) -> "(" ^ writeNatSym a ^ " % " ^ writeNatSym b ^ ")"
  | _ -> "NAT"


type 'a vector = __

(** val mk_vec : nat -> (fin -> 'a1) -> 'a1 vector **)

let rec mk_vec m x =
  match m with
  | O -> Obj.magic (fun _ -> Obj.magic ())
  | S m' -> Obj.magic (fun i -> x i)

(** val v_nth : nat -> 'a1 -> 'a1 vector -> nat -> 'a1 **)

let rec v_nth n x0 x x1 =
  match n with
  | O -> x0
  | S n' -> (Obj.magic x : fin -> __) x1

(** val v_idx : nat -> 'a1 -> 'a1 vector -> fin -> 'a1 **)

let v_idx n x0 v i =
  v_nth n x0 v (fin2nat n i)

type data =
| Num
| Ary of nat * fle * data

type path =
| Var of string
| AryIdx of nat * fin * path

type bexp =
| BTrue
| BFalse
| BLe of rexp * rexp
| BGt of rexp * rexp
| BLt of nat * nat
| BEq of nat * nat
| BLeb of nat * nat
| BAnd of bexp * bexp
| BNot of bexp
and rexp =
| REvar of path
| REconst of RbaseSymbolsImpl.coq_R * string
| REopp of rexp
| REadd of rexp * rexp
| REsub of rexp * rexp
| REmul of rexp * rexp
| REdiv of rexp * rexp
| REdiv_n of rexp * nat
| REscal of RbaseSymbolsImpl.coq_R * rexp
| REpow of rexp * nat
| REexp of rexp
| REln of rexp
| REsqrt of rexp
| REif of bexp * rexp * rexp

type exp = __

type acc = __

(** val mkvSeq : data -> nat -> (fin -> exp) -> exp **)

let mkvSeq _ =
  mk_vec

(** val exp_default : data -> exp **)

let rec exp_default = function
| Num ->
  Obj.magic (REconst (RbaseSymbolsImpl.coq_R0, ("0")))
| Ary (n0, n, d0) -> mkvSeq d0 (fle2nat n0 n) (fun _ -> exp_default d0)

(** val idx : nat -> data -> exp -> fin -> exp **)

let idx n t v i =
  v_idx (fle2nat n (fle_self n)) (exp_default t) v i

(** val zero : exp **)

let zero =
  Obj.magic (REconst (RbaseSymbolsImpl.coq_R0, ("0")))

(** val add0 : exp -> exp -> exp **)

let add0 x y =
  Obj.magic (REadd ((Obj.magic x), (Obj.magic y)))

(** val mul0 : exp -> exp -> exp **)

let mul0 x y =
  Obj.magic (REmul ((Obj.magic x), (Obj.magic y)))

(** val div0 : exp -> exp -> exp **)

let div0 x y =
  Obj.magic (REdiv ((Obj.magic x), (Obj.magic y)))

(** val leExp : exp -> exp -> bexp **)

let leExp x y =
  BLe ((Obj.magic x), (Obj.magic y))

(** val gtExp : exp -> exp -> bexp **)

let gtExp x y =
  BGt ((Obj.magic x), (Obj.magic y))

(** val pExp : exp -> exp **)

let pExp x =
  Obj.magic (REexp (Obj.magic x))

(** val vrd : data -> coq_string -> exp **)

let rec vrd d s =
  let rec vrd' d p =
    match d with
    | Num ->
      (match eqb1 s (String ((Ascii (False, False, False, False, True, True, False, False)), EmptyString)) with
       | True -> Obj.magic (REconst (RbaseSymbolsImpl.coq_R0, ("0")))
       | False ->
         match eqb1 s (String ((Ascii (True, False, False, False, True, True, False, False)), EmptyString)) with
         | True -> Obj.magic (REconst (RbaseSymbolsImpl.coq_R1, ("1")))
         | False -> Obj.magic (REvar p))
    | Ary (n0, n, d') -> mk_vec (fle2nat n0 n) (fun i -> vrd' d' (AryIdx (n, i, p)))
  in
  vrd' d (Var (coq_string_to_string s))

(** val vwt : data -> coq_string -> acc **)

let rec vwt d s =
  let rec vwt' d p =
    match d with
    | Num -> Obj.magic p
    | Ary (n0, n, d') -> mk_vec (fle2nat n0 n) (fun i -> vwt' d' (AryIdx (n, i, p)))
  in
  vwt' d (Var (coq_string_to_string s))

(** val mkvAcc : data -> nat -> (fin -> acc) -> acc **)

let mkvAcc _ =
  mk_vec

(** val acc_default : data -> acc **)

let rec acc_default = function
| Num ->
  vwt Num (String ((Ascii (False, True, True, True, False, True, True,
    False)), (String ((Ascii (True, True, True, True, False, True, True,
    False)), (String ((Ascii (False, True, True, True, False, True, True,
    False)), (String ((Ascii (True, False, True, False, False, True, True,
    False)), EmptyString))))))))
| Ary (n0, n, d0) -> mkvAcc d0 (fle2nat n0 n) (fun _ -> acc_default d0)

(** val idxAcc : nat -> data -> acc -> fin -> acc **)

let idxAcc n t v i =
  v_idx (fle2nat n (fle_self n)) (acc_default t) v i

type comm =
| Skip
| If of bexp * (comm, comm) prod
| Seq of (comm, comm) prod
| Assign of data * (acc, exp) prod
| New of data * ((acc, exp) prod -> comm)
| Loopacc of data * (coq_string -> comm)
| Seqfor of nat * (fin -> comm)
| Parfor of nat * data * acc * (fin -> acc -> comm)

(** val flatten_idx : nat -> nat -> nat -> fin -> fin -> fin -> fin **)

let flatten_idx channel m1 m2 c i j =
  nat2fin (mul (mul channel m1) m2)
    (add (add (mul (mul (fin2nat channel c) m1) m2) (mul (fin2nat m1 i) m2))
      (fin2nat m2 j))

(** val lenet5_input : exp **)

let lenet5_input =
  vrd (Ary ((S O), (fle_self (S O)), (Ary ((S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    O)))))))))))))))))))))))))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))))))),
    (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S O)))))))))))))))))))))))))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))))))),
    Num)))))) (String ((Ascii (True, False, False, True, False, True, True,
    False)), (String ((Ascii (False, True, True, True, False, True, True,
    False)), (String ((Ascii (False, False, False, False, True, True, True,
    False)), (String ((Ascii (True, False, True, False, True, True, True,
    False)), (String ((Ascii (False, False, True, False, True, True, True,
    False)), EmptyString))))))))))

(** val lenet5_kc1 : exp **)

let lenet5_kc1 =
  vrd (Ary ((S (S (S (S (S (S O)))))), (fle_self (S (S (S (S (S (S O))))))),
    (Ary ((S O), (fle_self (S O)), (Ary ((S (S (S (S (S O))))),
    (fle_self (S (S (S (S (S O)))))), (Ary ((S (S (S (S (S O))))),
    (fle_self (S (S (S (S (S O)))))), Num)))))))) (String ((Ascii (True,
    True, False, True, False, True, True, False)), (String ((Ascii (True,
    True, False, False, False, True, True, False)), (String ((Ascii (True,
    False, False, False, True, True, False, False)), EmptyString))))))

(** val lenet5_bc1 : exp **)

let lenet5_bc1 =
  vrd (Ary ((S (S (S (S (S (S O)))))), (fle_self (S (S (S (S (S (S O))))))),
    Num)) (String ((Ascii (False, True, False, False, False, True, True,
    False)), (String ((Ascii (True, True, False, False, False, True, True,
    False)), (String ((Ascii (True, False, False, False, True, True, False,
    False)), EmptyString))))))

(** val lenet5_kc2 : exp **)

let lenet5_kc2 =
  vrd (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    O)))))))))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      O))))))))))))))))),
    (Ary ((S (S (S (S (S (S O)))))), (fle_self (S (S (S (S (S (S O))))))),
    (Ary ((S (S (S (S (S O))))), (fle_self (S (S (S (S (S O)))))), (Ary ((S
    (S (S (S (S O))))), (fle_self (S (S (S (S (S O)))))), Num)))))))) (String
    ((Ascii (True, True, False, True, False, True, True, False)), (String
    ((Ascii (True, True, False, False, False, True, True, False)), (String
    ((Ascii (False, True, False, False, True, True, False, False)),
    EmptyString))))))

(** val lenet5_bc2 : exp **)

let lenet5_bc2 =
  vrd (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    O)))))))))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      O))))))))))))))))),
    Num)) (String ((Ascii (False, True, False, False, False, True, True,
    False)), (String ((Ascii (True, True, False, False, False, True, True,
    False)), (String ((Ascii (False, True, False, False, True, True, False,
    False)), EmptyString))))))

(** val lenet5_fc1_w : exp **)

let lenet5_fc1_w =
  vrd (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S
    O)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S
      O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
    (Ary
    ((mul
       (mul (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S O))))))))))))))))
         (S (S (S (S (S O))))))
       (S (S (S (S (S O)))))),
    (fle_self
      (mul
        (mul (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
          O)))))))))))))))) (S (S (S (S (S O))))))
        (S (S (S (S (S O))))))),
    Num)))) (String ((Ascii (False, True, True, False, False, True, True,
    False)), (String ((Ascii (True, True, False, False, False, True, True,
    False)), (String ((Ascii (True, False, False, False, True, True, False,
    False)), (String ((Ascii (True, True, True, True, True, False, True,
    False)), (String ((Ascii (True, True, True, False, True, True, True,
    False)), EmptyString))))))))))

(** val lenet5_fc1_b : exp **)

let lenet5_fc1_b =
  vrd (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S
    O)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S
      O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
    Num)) (String ((Ascii (False, True, True, False, False, True, True,
    False)), (String ((Ascii (True, True, False, False, False, True, True,
    False)), (String ((Ascii (True, False, False, False, True, True, False,
    False)), (String ((Ascii (True, True, True, True, True, False, True,
    False)), (String ((Ascii (False, True, False, False, False, True, True,
    False)), EmptyString))))))))))

(** val lenet5_fc2_w : exp **)

let lenet5_fc2_w =
  vrd (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    O)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
    (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S
    O)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S
      O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
    Num)))) (String ((Ascii (False, True, True, False, False, True, True,
    False)), (String ((Ascii (True, True, False, False, False, True, True,
    False)), (String ((Ascii (False, True, False, False, True, True, False,
    False)), (String ((Ascii (True, True, True, True, True, False, True,
    False)), (String ((Ascii (True, True, True, False, True, True, True,
    False)), EmptyString))))))))))

(** val lenet5_fc2_b : exp **)

let lenet5_fc2_b =
  vrd (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    O)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
    Num)) (String ((Ascii (False, True, True, False, False, True, True,
    False)), (String ((Ascii (True, True, False, False, False, True, True,
    False)), (String ((Ascii (False, True, False, False, True, True, False,
    False)), (String ((Ascii (True, True, True, True, True, False, True,
    False)), (String ((Ascii (False, True, False, False, False, True, True,
    False)), EmptyString))))))))))

(** val lenet5_fc3_w : exp **)

let lenet5_fc3_w =
  vrd (Ary ((S (S (S (S (S (S (S (S (S (S O)))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), (Ary ((S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S
    O)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
    Num)))) (String ((Ascii (False, True, True, False, False, True, True,
    False)), (String ((Ascii (True, True, False, False, False, True, True,
    False)), (String ((Ascii (True, True, False, False, True, True, False,
    False)), (String ((Ascii (True, True, True, True, True, False, True,
    False)), (String ((Ascii (True, True, True, False, True, True, True,
    False)), EmptyString))))))))))

(** val lenet5_fc3_b : exp **)

let lenet5_fc3_b =
  vrd (Ary ((S (S (S (S (S (S (S (S (S (S O)))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), Num)) (String
    ((Ascii (False, True, True, False, False, True, True, False)), (String
    ((Ascii (True, True, False, False, False, True, True, False)), (String
    ((Ascii (True, True, False, False, True, True, False, False)), (String
    ((Ascii (True, True, True, True, True, False, True, False)), (String
    ((Ascii (False, True, False, False, False, True, True, False)),
    EmptyString))))))))))

(** val lenet5_output : acc **)

let lenet5_output =
  vwt (Ary ((S (S (S (S (S (S (S (S (S (S O)))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), Num)) (String
    ((Ascii (True, True, True, True, False, True, True, False)), (String
    ((Ascii (True, False, True, False, True, True, True, False)), (String
    ((Ascii (False, False, True, False, True, True, True, False)), (String
    ((Ascii (False, False, False, False, True, True, True, False)), (String
    ((Ascii (True, False, True, False, True, True, True, False)), (String
    ((Ascii (False, False, True, False, True, True, True, False)),
    EmptyString))))))))))))

(** val lenet5_flatten_comm : exp -> acc -> comm **)

let lenet5_flatten_comm input output =
  Seqfor ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S O)))))))))))))))),
    (fun c -> Seqfor ((S (S (S (S (S O))))), (fun i -> Seqfor ((S (S (S (S (S
    O))))), (fun j -> Assign (Num, (((idxAcc
       (mul
         (mul (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           O)))))))))))))))) (S (S (S (S (S O))))))
         (S (S (S (S (S O))))))
       Num output
       (flatten_idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         O)))))))))))))))) (S (S (S (S (S O))))) (S (S (S (S (S O))))) c i j)), (idx (S (S (S (S (S O))))) Num
      (idx (S (S (S (S (S O))))) (Ary ((S (S (S (S (S O))))),
        (fle_self (S (S (S (S (S O)))))), Num))
        (idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
          O)))))))))))))))) (Ary ((S (S (S (S (S O))))),
          (fle_self (S (S (S (S (S O)))))), (Ary ((S (S (S (S (S O))))),
          (fle_self (S (S (S (S (S O)))))), Num)))) input c)
        i)
      j))))))))))

(** val lenet5_conv1_par_comm : exp -> exp -> exp -> acc -> comm **)

let lenet5_conv1_par_comm input w bias output =
  Parfor ((S (S (S (S (S (S O)))))), (Ary ((S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    O)))))))))))))))))))))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
    (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S O)))))))))))))))))))))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
    Num)))), output, (fun num o -> Seqfor ((S (S (S (S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    O)))))))))))))))))))))))))))), (fun i0 -> Seqfor ((S (S (S (S (S (S (S (S
    (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
    O)))))))))))))))))))))))))))), (fun j0 -> New (Num, (fun acc_tmp ->
    let (acc_a, acc_v) = acc_tmp in
    Seq (((Assign (Num, ((acc_a, zero)))), (Seq (((Seqfor ((S
    O), (fun c -> Seqfor ((S (S (S (S (S O))))), (fun i1 -> Seqfor ((S (S (S
    (S (S O))))), (fun j1 -> Assign (Num, ((acc_a, (add0
      (mul0
        (idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
          (S (S (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))))))
          Num
          (idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
            (S (S (S (S (S (S (S (S (S (S (S
            O)))))))))))))))))))))))))))))))) (Ary ((S (S (S (S (S (S (S (S
            (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
            (S (S O)))))))))))))))))))))))))))))))),
            (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
              (S (S (S (S (S (S (S (S (S (S (S (S (S (S
              O))))))))))))))))))))))))))))))))),
            Num))
            (idx (S O) (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
              (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
              O)))))))))))))))))))))))))))))))),
              (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
                (S (S (S (S (S (S (S (S (S (S (S (S (S (S
                O))))))))))))))))))))))))))))))))),
              (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
              (S (S (S (S (S (S (S (S (S (S (S (S (S
              O)))))))))))))))))))))))))))))))),
              (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
                (S (S (S (S (S (S (S (S (S (S (S (S (S (S
                O))))))))))))))))))))))))))))))))),
              Num)))) input c)
            (nat2fin (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
              (S (S (S (S (S (S (S (S (S (S (S (S (S
              O))))))))))))))))))))))))))))))))
              (add
                (fin2nat (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
                  (S (S (S (S (S (S (S (S (S (S (S
                  O)))))))))))))))))))))))))))) i0)
                (fin2nat (S (S (S (S (S O))))) i1))))
          (nat2fin (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
            (S (S (S (S (S (S (S (S (S (S (S (S (S
            O))))))))))))))))))))))))))))))))
            (add
              (fin2nat (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
                (S (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))
                j0)
              (fin2nat (S (S (S (S (S O))))) j1))))
        (idx (S (S (S (S (S O))))) Num
          (idx (S (S (S (S (S O))))) (Ary ((S (S (S (S (S O))))),
            (fle_self (S (S (S (S (S O)))))), Num))
            (idx (S O) (Ary ((S (S (S (S (S O))))),
              (fle_self (S (S (S (S (S O)))))), (Ary ((S (S (S (S (S O))))),
              (fle_self (S (S (S (S (S O)))))), Num))))
              (idx (S (S (S (S (S (S O)))))) (Ary ((S O), (fle_self (S O)),
                (Ary ((S (S (S (S (S O))))),
                (fle_self (S (S (S (S (S O)))))), (Ary ((S (S (S (S (S
                O))))), (fle_self (S (S (S (S (S O)))))), Num)))))) w num)
              c)
            i1)
          j1))
      acc_v))))))))))), (Assign (Num, (((idxAcc (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S O)))))))))))))))))))))))))))) Num
       (idxAcc (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S O)))))))))))))))))))))))))))) (Ary ((S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S O)))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
         Num)) o i0)
       j0), (add0 acc_v (idx (S (S (S (S (S (S O)))))) Num bias num)))))))))))))))))))

(** val lenet5_conv2_par_comm : exp -> exp -> exp -> acc -> comm **)

let lenet5_conv2_par_comm input w bias output =
  Parfor ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S O)))))))))))))))),
    (Ary ((S (S (S (S (S (S (S (S (S (S O)))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), (Ary ((S (S (S (S
    (S (S (S (S (S (S O)))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), Num)))), output,
    (fun num o -> Seqfor ((S (S (S (S (S (S (S (S (S (S O)))))))))),
    (fun i0 -> Seqfor ((S (S (S (S (S (S (S (S (S (S O)))))))))), (fun j0 ->
    New (Num, (fun acc_tmp ->
    let (acc_a, acc_v) = acc_tmp in
    Seq (((Assign (Num, ((acc_a, zero)))), (Seq (((Seqfor ((S
    (S (S (S (S (S O)))))), (fun c -> Seqfor ((S (S (S (S (S O))))),
    (fun i1 -> Seqfor ((S (S (S (S (S O))))), (fun j1 -> Assign (Num, ((acc_a, (add0
      (mul0
        (idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S O)))))))))))))) Num
          (idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S O)))))))))))))) (Ary
            ((S (S (S (S (S (S (S (S (S (S (S (S (S (S O)))))))))))))),
            (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S
              O))))))))))))))),
            Num))
            (idx (S (S (S (S (S (S O)))))) (Ary ((S (S (S (S (S (S (S (S (S
              (S (S (S (S (S O)))))))))))))),
              (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S
                O))))))))))))))),
              (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S
              O)))))))))))))),
              (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S
                O))))))))))))))),
              Num)))) input c)
            (nat2fin (S (S (S (S (S (S (S (S (S (S (S (S (S (S
              O))))))))))))))
              (add (fin2nat (S (S (S (S (S (S (S (S (S (S O)))))))))) i0)
                (fin2nat (S (S (S (S (S O))))) i1))))
          (nat2fin (S (S (S (S (S (S (S (S (S (S (S (S (S (S O))))))))))))))
            (add (fin2nat (S (S (S (S (S (S (S (S (S (S O)))))))))) j0)
              (fin2nat (S (S (S (S (S O))))) j1))))
        (idx (S (S (S (S (S O))))) Num
          (idx (S (S (S (S (S O))))) (Ary ((S (S (S (S (S O))))),
            (fle_self (S (S (S (S (S O)))))), Num))
            (idx (S (S (S (S (S (S O)))))) (Ary ((S (S (S (S (S O))))),
              (fle_self (S (S (S (S (S O)))))), (Ary ((S (S (S (S (S O))))),
              (fle_self (S (S (S (S (S O)))))), Num))))
              (idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
                O)))))))))))))))) (Ary ((S (S (S (S (S (S O)))))),
                (fle_self (S (S (S (S (S (S O))))))), (Ary ((S (S (S (S (S
                O))))), (fle_self (S (S (S (S (S O)))))), (Ary ((S (S (S (S
                (S O))))), (fle_self (S (S (S (S (S O)))))), Num)))))) w num)
              c)
            i1)
          j1))
      acc_v))))))))))), (Assign (Num, (((idxAcc (S (S (S (S (S (S (S (S (S (S O)))))))))) Num
       (idxAcc (S (S (S (S (S (S (S (S (S (S O)))))))))) (Ary ((S (S (S (S (S
         (S (S (S (S (S O)))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), Num)) o i0)
       j0), (add0 acc_v
      (idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S O))))))))))))))))
        Num bias num)))))))))))))))))))

(** val reLU3d_par_comm : nat -> nat -> nat -> exp -> acc -> comm **)

let reLU3d_par_comm filernum m1 m2 input output =
  Parfor (filernum, (Ary (m1, (fle_self m1), (Ary (m2, (fle_self m2),
    Num)))), output, (fun num o -> Seqfor (m1, (fun i0 -> Seqfor (m2,
    (fun i1 -> If
    ((leExp
       (idx m2 Num
         (idx m1 (Ary (m2, (fle_self m2), Num))
           (idx filernum (Ary (m1, (fle_self m1), (Ary (m2, (fle_self m2),
             Num)))) input num)
           i0)
         i1)
       zero),
    (((Assign (Num, (((idxAcc m2 Num (idxAcc m1 (Ary (m2, (fle_self m2), Num)) o i0) i1), zero)))), (Assign (Num, (((idxAcc m2 Num (idxAcc m1 (Ary (m2, (fle_self m2), Num)) o i0) i1), (idx m2 Num
      (idx m1 (Ary (m2, (fle_self m2), Num))
        (idx filernum (Ary (m1, (fle_self m1), (Ary (m2, (fle_self m2),
          Num)))) input num)
        i0)
      i1))))))))))))))

(** val lenet5_pool1_par_comm : exp -> acc -> comm **)

let lenet5_pool1_par_comm input output =
  Parfor ((S (S (S (S (S (S O)))))), (Ary ((S (S (S (S (S (S (S (S (S (S (S
    (S (S (S O)))))))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S O))))))))))))))),
    (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S O)))))))))))))),
    (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S O))))))))))))))),
    Num)))), output, (fun num o -> Seqfor ((S (S (S (S (S (S (S (S (S (S (S
    (S (S (S O)))))))))))))), (fun i0 -> Seqfor ((S (S (S (S (S (S (S (S (S
    (S (S (S (S (S O)))))))))))))), (fun j0 -> New (Num, (fun tmp ->
    let (w, v) = tmp in
    Seq (((Assign (Num, ((w, (idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S O)))))))))))))))))))))))))))) Num
      (idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
        (S (S (S (S (S (S O)))))))))))))))))))))))))))) (Ary ((S (S (S (S (S
        (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
        O)))))))))))))))))))))))))))),
        (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
          (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
        Num))
        (idx (S (S (S (S (S (S O)))))) (Ary ((S (S (S (S (S (S (S (S (S (S (S
          (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
          O)))))))))))))))))))))))))))),
          (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
            (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
          (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
          (S (S (S (S (S (S (S (S O)))))))))))))))))))))))))))),
          (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
            (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
          Num)))) input num)
        (nat2fin (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
          (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))
          (add
            (mul
              (fin2nat
                (Nat.div (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
                  (S (S (S (S (S (S (S (S (S (S (S
                  O)))))))))))))))))))))))))))) (S (S O)))
                i0)
              (S (S O)))
            (fin2nat (S (S O)) (nat2fin (S (S O)) O)))))
      (nat2fin (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
        (S (S (S (S (S (S (S O))))))))))))))))))))))))))))
        (add
          (mul
            (fin2nat
              (Nat.div (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
                (S (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))
                (S (S O)))
              j0)
            (S (S O)))
          (fin2nat (S (S O)) (nat2fin (S (S O)) O))))))))), (Seq (((Seqfor ((S (S O)), (fun i1 -> Seqfor ((S (S O)), (fun j1 ->
    If
    ((gtExp
       (idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S O)))))))))))))))))))))))))))) Num
         (idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S O)))))))))))))))))))))))))))) (Ary ((S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S O)))))))))))))))))))))))))))),
           (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
             (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
           Num))
           (idx (S (S (S (S (S (S O)))))) (Ary ((S (S (S (S (S (S (S (S (S (S
             (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
             O)))))))))))))))))))))))))))),
             (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
               (S (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
             (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
             (S (S (S (S (S (S (S (S (S O)))))))))))))))))))))))))))),
             (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
               (S (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
             Num)))) input num)
           (nat2fin (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
             (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))
             (add
               (mul
                 (fin2nat
                   (Nat.div (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
                     (S (S (S (S (S (S (S (S (S (S (S (S
                     O)))))))))))))))))))))))))))) (S (S O)))
                   i0)
                 (S (S O)))
               (fin2nat (S (S O)) i1))))
         (nat2fin (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))
           (add
             (mul
               (fin2nat
                 (Nat.div (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
                   (S (S (S (S (S (S (S (S (S (S (S
                   O)))))))))))))))))))))))))))) (S (S O)))
                 j0)
               (S (S O)))
             (fin2nat (S (S O)) j1))))
       v),
    (((Assign (Num, ((w, (idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
      (S (S (S (S (S O)))))))))))))))))))))))))))) Num
      (idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
        (S (S (S (S (S (S O)))))))))))))))))))))))))))) (Ary ((S (S (S (S (S
        (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
        O)))))))))))))))))))))))))))),
        (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
          (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
        Num))
        (idx (S (S (S (S (S (S O)))))) (Ary ((S (S (S (S (S (S (S (S (S (S (S
          (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
          O)))))))))))))))))))))))))))),
          (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
            (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
          (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
          (S (S (S (S (S (S (S (S O)))))))))))))))))))))))))))),
          (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
            (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
          Num)))) input num)
        (nat2fin (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
          (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))
          (add
            (mul
              (fin2nat
                (Nat.div (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
                  (S (S (S (S (S (S (S (S (S (S (S
                  O)))))))))))))))))))))))))))) (S (S O)))
                i0)
              (S (S O)))
            (fin2nat (S (S O)) i1))))
      (nat2fin (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
        (S (S (S (S (S (S (S O))))))))))))))))))))))))))))
        (add
          (mul
            (fin2nat
              (Nat.div (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
                (S (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))
                (S (S O)))
              j0)
            (S (S O)))
          (fin2nat (S (S O)) j1)))))))), (Assign (Num, ((w, v)))))))))))), (Assign (Num, (((idxAcc (S (S (S (S (S (S (S (S (S (S (S (S (S (S O)))))))))))))) Num
       (idxAcc (S (S (S (S (S (S (S (S (S (S (S (S (S (S O)))))))))))))) (Ary
         ((S (S (S (S (S (S (S (S (S (S (S (S (S (S O)))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S O))))))))))))))),
         Num)) o i0)
       j0), v)))))))))))))))))

(** val lenet5_pool2_par_comm : exp -> acc -> comm **)

let lenet5_pool2_par_comm input output =
  Parfor ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S O)))))))))))))))),
    (Ary ((S (S (S (S (S O))))), (fle_self (S (S (S (S (S O)))))), (Ary ((S
    (S (S (S (S O))))), (fle_self (S (S (S (S (S O)))))), Num)))), output,
    (fun num o -> Seqfor ((S (S (S (S (S O))))), (fun i0 -> Seqfor ((S (S (S
    (S (S O))))), (fun j0 -> New (Num, (fun tmp ->
    let (w, v) = tmp in
    Seq (((Assign (Num, ((w, (idx (S (S (S (S (S (S (S (S (S (S O)))))))))) Num
      (idx (S (S (S (S (S (S (S (S (S (S O)))))))))) (Ary ((S (S (S (S (S (S
        (S (S (S (S O)))))))))),
        (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), Num))
        (idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
          O)))))))))))))))) (Ary ((S (S (S (S (S (S (S (S (S (S O)))))))))),
          (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), (Ary ((S (S
          (S (S (S (S (S (S (S (S O)))))))))),
          (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), Num)))) input
          num)
        (nat2fin (S (S (S (S (S (S (S (S (S (S O))))))))))
          (add
            (mul
              (fin2nat
                (Nat.div (S (S (S (S (S (S (S (S (S (S O)))))))))) (S (S O)))
                i0)
              (S (S O)))
            (fin2nat (S (S O)) (nat2fin (S (S O)) O)))))
      (nat2fin (S (S (S (S (S (S (S (S (S (S O))))))))))
        (add
          (mul
            (fin2nat
              (Nat.div (S (S (S (S (S (S (S (S (S (S O)))))))))) (S (S O)))
              j0)
            (S (S O)))
          (fin2nat (S (S O)) (nat2fin (S (S O)) O))))))))), (Seq (((Seqfor ((S (S O)), (fun i1 -> Seqfor ((S (S O)), (fun j1 ->
    If
    ((gtExp
       (idx (S (S (S (S (S (S (S (S (S (S O)))))))))) Num
         (idx (S (S (S (S (S (S (S (S (S (S O)))))))))) (Ary ((S (S (S (S (S
           (S (S (S (S (S O)))))))))),
           (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), Num))
           (idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
             O)))))))))))))))) (Ary ((S (S (S (S (S (S (S (S (S (S
             O)))))))))),
             (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), (Ary ((S
             (S (S (S (S (S (S (S (S (S O)))))))))),
             (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), Num))))
             input num)
           (nat2fin (S (S (S (S (S (S (S (S (S (S O))))))))))
             (add
               (mul
                 (fin2nat
                   (Nat.div (S (S (S (S (S (S (S (S (S (S O)))))))))) (S (S
                     O)))
                   i0)
                 (S (S O)))
               (fin2nat (S (S O)) i1))))
         (nat2fin (S (S (S (S (S (S (S (S (S (S O))))))))))
           (add
             (mul
               (fin2nat
                 (Nat.div (S (S (S (S (S (S (S (S (S (S O)))))))))) (S (S O)))
                 j0)
               (S (S O)))
             (fin2nat (S (S O)) j1))))
       v),
    (((Assign (Num, ((w, (idx (S (S (S (S (S (S (S (S (S (S O)))))))))) Num
      (idx (S (S (S (S (S (S (S (S (S (S O)))))))))) (Ary ((S (S (S (S (S (S
        (S (S (S (S O)))))))))),
        (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), Num))
        (idx (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
          O)))))))))))))))) (Ary ((S (S (S (S (S (S (S (S (S (S O)))))))))),
          (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), (Ary ((S (S
          (S (S (S (S (S (S (S (S O)))))))))),
          (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), Num)))) input
          num)
        (nat2fin (S (S (S (S (S (S (S (S (S (S O))))))))))
          (add
            (mul
              (fin2nat
                (Nat.div (S (S (S (S (S (S (S (S (S (S O)))))))))) (S (S O)))
                i0)
              (S (S O)))
            (fin2nat (S (S O)) i1))))
      (nat2fin (S (S (S (S (S (S (S (S (S (S O))))))))))
        (add
          (mul
            (fin2nat
              (Nat.div (S (S (S (S (S (S (S (S (S (S O)))))))))) (S (S O)))
              j0)
            (S (S O)))
          (fin2nat (S (S O)) j1)))))))), (Assign (Num, ((w, v)))))))))))), (Assign (Num, (((idxAcc (S (S (S (S (S O))))) Num
       (idxAcc (S (S (S (S (S O))))) (Ary ((S (S (S (S (S O))))),
         (fle_self (S (S (S (S (S O)))))), Num)) o i0)
       j0), v)))))))))))))))))

(** val linear1d_par_comm : nat -> nat -> exp -> exp -> exp -> acc -> comm **)

let linear1d_par_comm m outnum w b input output =
  Parfor (outnum, Num, output, (fun out o -> Loopacc (Num, (fun tmp ->
    let w_acc = vwt Num tmp in
    let v = vrd Num tmp in
    Seq (((Assign (Num, ((w_acc, zero)))), (Seq (((Seqfor (m,
    (fun i -> Assign (Num, ((w_acc, (add0
      (mul0 (idx m Num input i)
        (idx m Num (idx outnum (Ary (m, (fle_self m), Num)) w out) i))
      v))))))), (Assign (Num, ((o, (add0 v (idx outnum Num b out)))))))))))))))

(** val reLU1d_par_comm : nat -> exp -> acc -> comm **)

let reLU1d_par_comm n input output =
  Parfor (n, Num, output, (fun i o -> If ((leExp (idx n Num input i) zero),
    (((Assign (Num, ((o, zero)))), (Assign (Num, ((o, (idx n Num input i))))))))))

(** val softmax1d_par_comm : nat -> exp -> acc -> comm **)

let softmax1d_par_comm n input output =
  Loopacc (Num, (fun tmp ->
    let w = vwt Num tmp in
    let v = vrd Num tmp in
    Seq (((Assign (Num, ((w, zero)))), (Seq (((Seqfor (n,
    (fun i -> Assign (Num, ((w, (add0 (pExp (idx n Num input i)) v))))))), (Parfor (n, Num, output,
    (fun j o -> Assign (Num, ((o, (div0 (pExp (idx n Num input j)) v))))))))))))))

(** val lenet5_par_comm : comm **)

let lenet5_par_comm =
  Seq (((lenet5_conv1_par_comm lenet5_input lenet5_kc1 lenet5_bc1
       (vwt (Ary ((S (S (S (S (S (S O)))))),
         (fle_self (S (S (S (S (S (S O))))))), (Ary ((S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         O)))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
         (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S O)))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
         Num)))))) (String ((Ascii (True, True, False, False, False, True,
         True, False)), (String ((Ascii (True, False, False, False, True,
         True, False, False)), EmptyString)))))), (Seq (((reLU3d_par_comm (S (S (S (S (S (S O)))))) (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       O)))))))))))))))))))))))))))) (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       O))))))))))))))))))))))))))))
       (vrd (Ary ((S (S (S (S (S (S O)))))),
         (fle_self (S (S (S (S (S (S O))))))), (Ary ((S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         O)))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
         (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S O)))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
         Num)))))) (String ((Ascii (True, True, False, False, False, True,
         True, False)), (String ((Ascii (True, False, False, False, True,
         True, False, False)), EmptyString)))))
       (vwt (Ary ((S (S (S (S (S (S O)))))),
         (fle_self (S (S (S (S (S (S O))))))), (Ary ((S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         O)))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
         (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S O)))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
         Num)))))) (String ((Ascii (False, True, False, False, True, True,
         True, False)), (String ((Ascii (True, False, False, False, True,
         True, False, False)), EmptyString)))))), (Seq (((lenet5_pool1_par_comm
       (vrd (Ary ((S (S (S (S (S (S O)))))),
         (fle_self (S (S (S (S (S (S O))))))), (Ary ((S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         O)))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
         (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S O)))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S O))))))))))))))))))))))))))))),
         Num)))))) (String ((Ascii (False, True, False, False, True, True,
         True, False)), (String ((Ascii (True, False, False, False, True,
         True, False, False)), EmptyString)))))
       (vwt (Ary ((S (S (S (S (S (S O)))))),
         (fle_self (S (S (S (S (S (S O))))))), (Ary ((S (S (S (S (S (S (S (S
         (S (S (S (S (S (S O)))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S O))))))))))))))),
         (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S O)))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S O))))))))))))))),
         Num)))))) (String ((Ascii (False, False, False, False, True, True,
         True, False)), (String ((Ascii (True, False, False, False, True,
         True, False, False)), EmptyString)))))), (Seq (((lenet5_conv2_par_comm
       (vrd (Ary ((S (S (S (S (S (S O)))))),
         (fle_self (S (S (S (S (S (S O))))))), (Ary ((S (S (S (S (S (S (S (S
         (S (S (S (S (S (S O)))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S O))))))))))))))),
         (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S O)))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S O))))))))))))))),
         Num)))))) (String ((Ascii (False, False, False, False, True, True,
         True, False)), (String ((Ascii (True, False, False, False, True,
         True, False, False)), EmptyString)))))
       lenet5_kc2 lenet5_bc2
       (vwt (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         O)))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           O))))))))))))))))),
         (Ary ((S (S (S (S (S (S (S (S (S (S O)))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), (Ary ((S (S (S
         (S (S (S (S (S (S (S O)))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), Num))))))
         (String ((Ascii (True, True, False, False, False, True, True,
         False)), (String ((Ascii (False, True, False, False, True, True,
         False, False)), EmptyString)))))), (Seq (((reLU3d_par_comm (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       O)))))))))))))))) (S (S (S (S (S (S (S (S (S (S O)))))))))) (S (S (S
       (S (S (S (S (S (S (S O))))))))))
       (vrd (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         O)))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           O))))))))))))))))),
         (Ary ((S (S (S (S (S (S (S (S (S (S O)))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), (Ary ((S (S (S
         (S (S (S (S (S (S (S O)))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), Num))))))
         (String ((Ascii (True, True, False, False, False, True, True,
         False)), (String ((Ascii (False, True, False, False, True, True,
         False, False)), EmptyString)))))
       (vwt (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         O)))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           O))))))))))))))))),
         (Ary ((S (S (S (S (S (S (S (S (S (S O)))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), (Ary ((S (S (S
         (S (S (S (S (S (S (S O)))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), Num))))))
         (String ((Ascii (False, True, False, False, True, True, True,
         False)), (String ((Ascii (False, True, False, False, True, True,
         False, False)), EmptyString)))))), (Seq (((lenet5_pool2_par_comm
       (vrd (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         O)))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           O))))))))))))))))),
         (Ary ((S (S (S (S (S (S (S (S (S (S O)))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), (Ary ((S (S (S
         (S (S (S (S (S (S (S O)))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), Num))))))
         (String ((Ascii (False, True, False, False, True, True, True,
         False)), (String ((Ascii (False, True, False, False, True, True,
         False, False)), EmptyString)))))
       (vwt (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         O)))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           O))))))))))))))))),
         (Ary ((S (S (S (S (S O))))), (fle_self (S (S (S (S (S O)))))), (Ary
         ((S (S (S (S (S O))))), (fle_self (S (S (S (S (S O)))))), Num))))))
         (String ((Ascii (False, False, False, False, True, True, True,
         False)), (String ((Ascii (False, True, False, False, True, True,
         False, False)), EmptyString)))))), (Seq (((lenet5_flatten_comm
       (vrd (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         O)))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           O))))))))))))))))),
         (Ary ((S (S (S (S (S O))))), (fle_self (S (S (S (S (S O)))))), (Ary
         ((S (S (S (S (S O))))), (fle_self (S (S (S (S (S O)))))), Num))))))
         (String ((Ascii (False, False, False, False, True, True, True,
         False)), (String ((Ascii (False, True, False, False, True, True,
         False, False)), EmptyString)))))
       (vwt (Ary
         ((mul
            (mul (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
              O)))))))))))))))) (S (S (S (S (S O))))))
            (S (S (S (S (S O)))))),
         (fle_self
           (mul
             (mul (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
               O)))))))))))))))) (S (S (S (S (S O))))))
             (S (S (S (S (S O))))))),
         Num)) (String ((Ascii (False, True, True, False, False, True, True,
         False)), (String ((Ascii (False, False, True, True, False, True,
         True, False)), (String ((Ascii (True, False, False, False, False,
         True, True, False)), (String ((Ascii (False, False, True, False,
         True, True, True, False)), EmptyString)))))))))), (Seq (((linear1d_par_comm
       (mul
         (mul (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           O)))))))))))))))) (S (S (S (S (S O))))))
         (S (S (S (S (S O))))))
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S
       O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
       lenet5_fc1_w lenet5_fc1_b
       (vrd (Ary
         ((mul
            (mul (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
              O)))))))))))))))) (S (S (S (S (S O))))))
            (S (S (S (S (S O)))))),
         (fle_self
           (mul
             (mul (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
               O)))))))))))))))) (S (S (S (S (S O))))))
             (S (S (S (S (S O))))))),
         Num)) (String ((Ascii (False, True, True, False, False, True, True,
         False)), (String ((Ascii (False, False, True, True, False, True,
         True, False)), (String ((Ascii (True, False, False, False, False,
         True, True, False)), (String ((Ascii (False, False, True, False,
         True, True, True, False)), EmptyString)))))))))
       (vwt (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S
         O)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S
           O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
         Num)) (String ((Ascii (False, True, True, False, False, True, True,
         False)), (String ((Ascii (True, True, False, False, False, True,
         True, False)), (String ((Ascii (True, False, False, False, True,
         True, False, False)), EmptyString)))))))), (Seq (((reLU1d_par_comm (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S
       O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
       (vrd (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S
         O)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S
           O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
         Num)) (String ((Ascii (False, True, True, False, False, True, True,
         False)), (String ((Ascii (True, True, False, False, False, True,
         True, False)), (String ((Ascii (True, False, False, False, True,
         True, False, False)), EmptyString)))))))
       (vwt (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S
         O)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S
           O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
         Num)) (String ((Ascii (False, True, True, False, False, True, True,
         False)), (String ((Ascii (True, True, False, False, False, True,
         True, False)), (String ((Ascii (True, False, False, False, True,
         True, False, False)), (String ((Ascii (True, True, True, True, True,
         False, True, False)), (String ((Ascii (False, True, False, False,
         True, True, True, False)), (String ((Ascii (True, False, True,
         False, False, True, True, False)), (String ((Ascii (False, False,
         True, True, False, True, True, False)), (String ((Ascii (True,
         False, True, False, True, True, True, False)),
         EmptyString)))))))))))))))))), (Seq (((linear1d_par_comm (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S
       O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
       lenet5_fc2_w lenet5_fc2_b
       (vrd (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S
         O)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S
           O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
         Num)) (String ((Ascii (False, True, True, False, False, True, True,
         False)), (String ((Ascii (True, True, False, False, False, True,
         True, False)), (String ((Ascii (True, False, False, False, True,
         True, False, False)), (String ((Ascii (True, True, True, True, True,
         False, True, False)), (String ((Ascii (False, True, False, False,
         True, True, True, False)), (String ((Ascii (True, False, True,
         False, False, True, True, False)), (String ((Ascii (False, False,
         True, True, False, True, True, False)), (String ((Ascii (True,
         False, True, False, True, True, True, False)),
         EmptyString)))))))))))))))))
       (vwt (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         O)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
         Num)) (String ((Ascii (False, True, True, False, False, True, True,
         False)), (String ((Ascii (True, True, False, False, False, True,
         True, False)), (String ((Ascii (False, True, False, False, True,
         True, False, False)), EmptyString)))))))), (Seq (((reLU1d_par_comm (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
       (vrd (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         O)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
         Num)) (String ((Ascii (False, True, True, False, False, True, True,
         False)), (String ((Ascii (True, True, False, False, False, True,
         True, False)), (String ((Ascii (False, True, False, False, True,
         True, False, False)), EmptyString)))))))
       (vwt (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         O)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
         Num)) (String ((Ascii (False, True, True, False, False, True, True,
         False)), (String ((Ascii (True, True, False, False, False, True,
         True, False)), (String ((Ascii (False, True, False, False, True,
         True, False, False)), (String ((Ascii (True, True, True, True, True,
         False, True, False)), (String ((Ascii (False, True, False, False,
         True, True, True, False)), (String ((Ascii (True, False, True,
         False, False, True, True, False)), (String ((Ascii (False, False,
         True, True, False, True, True, False)), (String ((Ascii (True,
         False, True, False, True, True, True, False)),
         EmptyString)))))))))))))))))), (Seq (((linear1d_par_comm (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
       O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
       (S (S (S (S (S (S (S (S (S (S O)))))))))) lenet5_fc3_w lenet5_fc3_b
       (vrd (Ary ((S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
         O)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S (S
           O))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))),
         Num)) (String ((Ascii (False, True, True, False, False, True, True,
         False)), (String ((Ascii (True, True, False, False, False, True,
         True, False)), (String ((Ascii (False, True, False, False, True,
         True, False, False)), (String ((Ascii (True, True, True, True, True,
         False, True, False)), (String ((Ascii (False, True, False, False,
         True, True, True, False)), (String ((Ascii (True, False, True,
         False, False, True, True, False)), (String ((Ascii (False, False,
         True, True, False, True, True, False)), (String ((Ascii (True,
         False, True, False, True, True, True, False)),
         EmptyString)))))))))))))))))
       (vwt (Ary ((S (S (S (S (S (S (S (S (S (S O)))))))))),
         (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), Num)) (String
         ((Ascii (False, True, True, False, False, True, True, False)),
         (String ((Ascii (True, True, False, False, False, True, True,
         False)), (String ((Ascii (True, True, False, False, True, True,
         False, False)), EmptyString)))))))), (softmax1d_par_comm (S (S (S (S (S (S (S (S (S (S O))))))))))
      (vrd (Ary ((S (S (S (S (S (S (S (S (S (S O)))))))))),
        (fle_self (S (S (S (S (S (S (S (S (S (S O))))))))))), Num)) (String
        ((Ascii (False, True, True, False, False, True, True, False)),
        (String ((Ascii (True, True, False, False, False, True, True,
        False)), (String ((Ascii (True, True, False, False, True, True,
        False, False)), EmptyString)))))))
      lenet5_output))))))))))))))))))))))))))))))))))))

(** val writeR : RbaseSymbolsImpl.coq_R -> string **)

let writeR r =
  if r == RbaseSymbolsImpl.coq_R0 then "0.0"
  else if r == RbaseSymbolsImpl.coq_R1 then "1.0"
  else "0.0"

(** val vstr : nat -> string **)

let vstr cnt = "v" ^ writeNatSym cnt

(** val istr : nat -> string **)

let istr cnt = "i" ^ writeNatSym cnt

(** val cG_data : data -> string -> string **)

let rec cG_data d s =
  match d with
  | Num -> "float " ^ s
  | Ary (n0, n, d') -> cG_data d' (s ^ "[" ^ writeNatSym (fle2nat n0 n) ^ "]")

(** val cnt_var : comm -> nat **)

let rec cnt_var = function
  | Skip -> O
  | If (_, (c1, c2)) -> add (cnt_var c1) (cnt_var c2)
  | Seq (c1, c2) -> add (cnt_var c1) (cnt_var c2)
  | Assign _ -> O
  | New _ -> S O
  | Loopacc _ -> O
  | Seqfor _ -> S O
  | Parfor _ -> S O
