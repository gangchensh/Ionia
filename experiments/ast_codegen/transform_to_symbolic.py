#!/usr/bin/env python3
"""
Transform an extracted Ionia .ml file to use symbolic natural numbers.

This allows loop-variable names (introduced by str2fin) to flow through
index arithmetic so that the generated C code contains i0, i1, ... instead
of concrete constants.

The input is the *default* Coq extraction format (Coq inductive pairs with
Pair constructor, Coq inductive strings with EmptyString/String constructors).
The output uses native OCaml pairs, native OCaml strings, and symbolic nats.
"""

import re
import sys

INFILE = sys.argv[1]
OUTFILE = sys.argv[2]

NAT_HELPERS = '''
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

'''

COQ_STRING_HELPERS = '''
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

'''

with open(INFILE, 'r') as f:
    src = f.read()

# 1. Extend the nat type with symbolic constructors.
src = src.replace(
    'type nat =\n| O\n| S of nat',
    '''type nat =
| O
| S of nat
| Var of string
| Add of nat * nat
| Mul of nat * nat
| Sub of nat * nat
| Div of nat * nat
| Mod of nat * nat
''' + NAT_HELPERS
)

# 2. Coq strings are extracted as an inductive type.  Insert helpers to convert
#    them to native OCaml strings where needed (REconst, Var names).
src = src.replace(
    'type string =\n| EmptyString\n| String of ascii * string',
    'type coq_string =\n| EmptyString\n| String of ascii * coq_string\n' + COQ_STRING_HELPERS
)

# Val comments that mention the Coq string type also need the renamed type.
src = src.replace(
    '(** val vrd : data -> string -> exp **)',
    '(** val vrd : data -> coq_string -> exp **)'
)
src = src.replace(
    '(** val vwt : data -> string -> acc **)',
    '(** val vwt : data -> coq_string -> acc **)'
)

# 3. Coq prod is extracted as an algebraic Pair constructor.  Turn it into a
#    native OCaml pair, and update fst/snd destructors.
src = src.replace(
    "type ('a, 'b) prod =\n| Pair of 'a * 'b",
    "type ('a, 'b) prod = 'a * 'b"
)

def unfold_pair_calls(text):
    """Replace every 'Pair (expr1, expr2)' with '(expr1, expr2)'.

    Handles nested Pair calls and parentheses inside the arguments.
    """
    result = []
    i = 0
    n = len(text)
    while i < n:
        if (text[i:i+4] == 'Pair' and
            (i + 4 >= n or text[i+4] == '(' or text[i+4].isspace()) and
            (i == 0 or (not text[i-1].isalnum() and text[i-1] != '_'))):
            # Locate the opening parenthesis.
            j = i + 4
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j] == '(':
                # Find the matching comma at depth 0 inside the argument list.
                depth = 1
                k = j + 1
                while k < n and depth > 0:
                    if text[k] == '(':
                        depth += 1
                    elif text[k] == ')':
                        depth -= 1
                    k += 1
                # k now points just after the closing ')'.
                inner = text[j+1:k-1]
                # Split inner into two arguments at the first comma at depth 0.
                depth = 0
                comma = -1
                for idx, ch in enumerate(inner):
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                    elif ch == ',' and depth == 0:
                        comma = idx
                        break
                if comma >= 0:
                    arg1 = inner[:comma].strip()
                    arg2 = inner[comma+1:].strip()
                    result.append('(' + arg1 + ', ' + arg2 + ')')
                    i = k
                    continue
        result.append(text[i])
        i += 1
    return ''.join(result)

# Unfold Pair constructor applications.  We iterate because nested Pair
# applications may appear.
old_src = None
while old_src != src:
    old_src = src
    src = unfold_pair_calls(src)

# fst/snd over native pairs.
src = re.sub(
    r'let fst = function\n\| Pair \(x, _\) -> x',
    'let fst = function\n| (x, _) -> x',
    src
)
src = re.sub(
    r'let snd = function\n\| Pair \(_, y\) -> y',
    'let snd = function\n| (_, y) -> y',
    src
)

# 4. Replace primitive nat operations with symbolic-aware versions.
# add (global / included Coq__1)
src = re.sub(
    r'let rec add n m =\n  match n with\n  \| O -> m\n  \| S p -> S \(add p m\)',
    '''let rec add n m =
  match (n, m) with
  | (O, x) | (x, O) -> x
  | (S p, S q) -> S (S (add p q))
  | (S p, x) -> S (add p x)
  | (x, S q) -> S (add x q)
  | _ -> Add (n, m)''',
    src
)

# Coq__1.add (one-space indentation inside the generated module)
src = src.replace(
    ''' let rec add n m =
   match n with
   | O -> m
   | S p -> S (add p m)''',
    ''' let rec add n m =
   match (n, m) with
   | (O, x) | (x, O) -> x
   | (S p, S q) -> S (S (add p q))
   | (S p, x) -> S (add p x)
   | (x, S q) -> S (add x q)
   | _ -> Add (n, m)'''
)

# mul
src = re.sub(
    r'let rec mul n m =\n  match n with\n  \| O -> O\n  \| S p -> add m \(mul p m\)',
    '''let rec mul n m =
  match (n, m) with
  | (O, _) | (_, O) -> O
  | (S O, x) | (x, S O) -> x
  | (S p, S q) -> add (S q) (mul p (S q))
  | _ -> Mul (n, m)''',
    src
)

# sub
src = re.sub(
    r'let rec sub n m =\n  match n with\n  \| O -> n\n  \| S k -> \(match m with\n            \| O -> n\n            \| S l -> sub k l\)',
    '''let rec sub n m =
  match (n, m) with
  | (O, _) -> O
  | (x, O) -> x
  | (S k, S l) -> sub k l
  | _ -> Sub (n, m)''',
    src
)

# Nat.add (inside module Nat)
src = src.replace(
    '''  let rec add n m =
    match n with
    | O -> m
    | S p -> S (add p m)''',
    '''  let rec add n m =
    match (n, m) with
    | (O, x) | (x, O) -> x
    | (S p, S q) -> S (S (add p q))
    | (S p, x) -> S (add p x)
    | (x, S q) -> S (add x q)
    | _ -> Add (n, m)'''
)

# Nat.mul
src = src.replace(
    '''  let rec mul n m =
    match n with
    | O -> O
    | S p -> add m (mul p m)''',
    '''  let rec mul n m =
    match (n, m) with
    | (O, _) | (_, O) -> O
    | (S O, x) | (x, S O) -> x
    | (S p, S q) -> add (S q) (mul p (S q))
    | _ -> Mul (n, m)'''
)

# Nat.sub
src = src.replace(
    '''  let rec sub n m =
    match n with
    | O -> n
    | S k -> (match m with
             | O -> n
             | S l -> sub k l)''',
    '''  let rec sub n m =
    match (n, m) with
    | (O, _) -> O
    | (x, O) -> x
    | (S k, S l) -> sub k l
    | _ -> Sub (n, m)'''
)

# Nat.eqb
src = src.replace(
    '''  let rec eqb n m =
    match n with
    | O -> (match m with
            | O -> True
            | S _ -> False)
    | S n' -> (match m with
               | O -> False
               | S m' -> eqb n' m')''',
    '''  let rec eqb n m =
    match (n, m) with
    | (O, O) -> True
    | (S n', S m') -> eqb n' m'
    | _ when nat_is_concrete n && nat_is_concrete m -> False
    | _ -> False'''
)

# Nat.leb
src = src.replace(
    '''  let rec leb n m =
    match n with
    | O -> True
    | S n' -> (match m with
               | O -> False
               | S m' -> leb n' m')''',
    '''  let rec leb n m =
    match (n, m) with
    | (O, _) -> True
    | (_, O) -> False
    | (S n', S m') -> leb n' m'
    | _ -> True'''
)

# Nat.ltb
src = re.sub(
    r'let ltb n m =\s+leb \(S n\) m',
    '''let rec ltb n m =
    match (n, m) with
    | (_, O) -> False
    | (O, S _) -> True
    | (S n', S m') -> ltb n' m'
    | _ -> True''',
    src
)

# Nat.div
src = re.sub(
    r'let div x y = match y with\n  \| O -> y\n  \| S y\' -> fst \(divmod x y\' O y\'\)',
    '''let div x y =
  match (x, y) with
  | (_, O) -> O
  | _ when nat_is_concrete x && nat_is_concrete y ->
      let q, _ = divmod x (pred y) O (pred y) in q
  | _ -> Div (x, y)''',
    src
)

# Nat.modulo
src = re.sub(
    r'let modulo x = function\n  \| O -> x\n  \| S y\' -> sub y\' \(snd \(divmod x y\' O y\'\)\)',
    '''let modulo x y =
  match y with
  | O -> x
  | _ when nat_is_concrete x && nat_is_concrete y ->
      sub (pred y) (snd (divmod x (pred y) O (pred y)))
  | _ -> Mod (x, y)''',
    src
)

# 5. Replace the writeNatAux / writeNat block with a symbolic printer.
WRITE_NAT_SYM = '''let rec writeNatSym n =
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

'''
src = re.sub(
    r'let rec writeNatAux.*?let writeNat n =\n  writeNatAux n n (""|EmptyString)\n',
    WRITE_NAT_SYM,
    src,
    flags=re.DOTALL
)

# Redirect any remaining references to the old writeNat to writeNatSym.
src = re.sub(r'\bwriteNat\b', 'writeNatSym', src)

# 6. str2fin returns a symbolic Var.
src = re.sub(
    r'let str2fin =\n  failwith "AXIOM TO BE REALIZED \(Ionia\.CNN_CodeGen\.str2fin\)"',
    'let str2fin _ s = Var s',
    src
)

# Some realized bases already stub str2fin as a constant O; override that too.
src = re.sub(
    r'let str2fin _ _ =\n  O\n',
    'let str2fin _ s = (Var s : fin)\n',
    src
)

# 7. Represent vectors as functions (fin -> 'a) so that symbolic loop
#    indices can be threaded through indexing operations.
src = src.replace(
    '''let rec mk_vec m x =
  match m with
  | O -> Obj.magic Tt
  | S m' ->
    Obj.magic (((mk_vec m' (fun i -> x (fin2finS m' i))), (x m')))''',
    '''let rec mk_vec m x =
  match m with
  | O -> Obj.magic (fun _ -> Obj.magic ())
  | S m' -> Obj.magic (fun i -> x i)'''
)

# v_nth becomes function application.
src = src.replace(
    '''let rec v_nth n x0 x x1 =
  match n with
  | O -> x0
  | S n' ->
    (match Nat.eqb x1 n' with
     | True -> snd (Obj.magic x)
     | False -> v_nth n' x0 (fst (Obj.magic x)) x1)''',
    '''let rec v_nth n x0 x x1 =
  match n with
  | O -> x0
  | S n' -> (Obj.magic x : fin -> __) x1'''
)

# 8. Rebuild vrd/vwt so that array elements carry indexed paths, and convert
#    Coq inductive strings to native strings where the AST expects them.
VRD_REPLACEMENT = r'''(** val vrd : data -> coq_string -> exp **)

let rec vrd d s =
  let rec vrd' d p =
    match d with
    | Num ->
      (match eqb1 s (String ((Ascii (False, False, False, False, True, True, False, False)), EmptyString)) with
       | True -> Obj.magic (REconst (RbaseSymbolsImpl.coq_R0, (String ((Ascii (False, False, False, False, True, True, False, False)), EmptyString))))
       | False ->
         match eqb1 s (String ((Ascii (True, False, False, False, True, True, False, False)), EmptyString)) with
         | True -> Obj.magic (REconst (RbaseSymbolsImpl.coq_R1, (String ((Ascii (True, False, False, False, True, True, False, False)), EmptyString))))
         | False -> Obj.magic (REvar p))
    | Ary (n0, n, d') -> mk_vec (fle2nat n0 n) (fun i -> vrd' d' (AryIdx (n, i, p)))
  in
  vrd' d (Var (coq_string_to_string s))'''

# We match the common shape of the extracted vrd, which uses eqb1 and the
# String/Ascii constructors for "0" and "1".  The Ascii argument list may span
# multiple lines, so we use a permissive ([\s\S]+?) pattern.
COQ_STRING_ZERO = r'String\s*\(\s*\(Ascii\s*\(\s*[\s\S]+?\)\s*\)\s*,\s*EmptyString\s*\)'
# The exact shape of the extracted vrd varies (line breaks inside Ascii;
# newer sources build indexed names via string append instead of recursing
# on the same name), so we replace everything up to the next declaration.
src = re.sub(
    r'\(\*\* val vrd : data -> coq_string -> exp \*\*\)\n\nlet rec vrd d s =[\s\S]+?(?=\n\n\(\*\* val vwt)',
    VRD_REPLACEMENT,
    src
)

VWT_REPLACEMENT = r'''(** val vwt : data -> coq_string -> acc **)

let rec vwt d s =
  let rec vwt' d p =
    match d with
    | Num -> Obj.magic p
    | Ary (n0, n, d') -> mk_vec (fle2nat n0 n) (fun i -> vwt' d' (AryIdx (n, i, p)))
  in
  vwt' d (Var (coq_string_to_string s))'''

src = re.sub(
    r'\(\*\* val vwt : data -> coq_string -> acc \*\*\)\n\nlet rec vwt d s =[\s\S]+?(?=\n\n\(\*\* val mkvAcc)',
    VWT_REPLACEMENT,
    src
)

# Convert Coq-string arguments of REconst into native string literals.
# The second argument may be parenthesised and may span multiple lines.
ZERO_COQ_STRING = r'String\s*\(\s*\(Ascii\s*\(\s*False\s*,\s*False\s*,\s*False\s*,\s*False\s*,\s*True\s*,\s*True\s*,\s*False\s*,\s*False\s*\)\s*\)\s*,\s*EmptyString\s*\)'
ONE_COQ_STRING = r'String\s*\(\s*\(Ascii\s*\(\s*True\s*,\s*False\s*,\s*False\s*,\s*False\s*,\s*True\s*,\s*True\s*,\s*False\s*,\s*False\s*\)\s*\)\s*,\s*EmptyString\s*\)'
src = re.sub(
    r'(REconst\s*\([^,]+,\s*\(?\s*)' + ZERO_COQ_STRING + r'(\s*\)?\s*\))',
    r'\1"0"\2',
    src,
    flags=re.DOTALL
)
src = re.sub(
    r'(REconst\s*\([^,]+,\s*\(?\s*)' + ONE_COQ_STRING + r'(\s*\)?\s*\))',
    r'\1"1"\2',
    src,
    flags=re.DOTALL
)

# 9. safe_idx_nat: for symbolic indices, assume in bounds.
src = re.sub(
    r'let safe_idx_nat n t e i =\n  match Nat\.ltb_spec0 i n with\n  \| ReflectT -> idx n t e \(nat2fin n i\)\n  \| ReflectF -> exp_default t',
    '''let safe_idx_nat n t e i =
  match Nat.ltb_spec0 i n with
  | ReflectT -> idx n t e (nat2fin n i)
  | ReflectF -> if nat_is_concrete i then exp_default t else idx n t e (nat2fin n i)''',
    src
)

# 10. Keep sig_forall_dec as a real function (rather than a partially-applied
#     failwith thunk) so backtraces retain the caller when debugging.
src = src.replace(
    '''let sig_forall_dec =
  failwith "AXIOM TO BE REALIZED (Stdlib.Reals.ClassicalDedekindReals.sig_forall_dec)"''',
    '''let sig_forall_dec _ =
  failwith "AXIOM TO BE REALIZED (Stdlib.Reals.ClassicalDedekindReals.sig_forall_dec)"'''
)

# 11. Avoid the un-realisable ClassicalDedekindReals.sig_forall_dec axiom when
#     abstracting trivial constant cReals (e.g. inject_Q used for coq_R0/coq_R1).
src = src.replace(
    '''let dRealAbstr x =
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
  | Inright -> False)''',
    '''let dRealAbstr x =
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
    | Inright -> False)'''
)

# 11. The streaming printer expects REvar to carry a path, not a raw string.
src = src.replace(
    'type rexp =\n| REvar of string',
    'type rexp =\n| REvar of path'
)
# Newer extractions define rexp mutually with bexp.
src = src.replace(
    'and rexp =\n| REvar of string',
    'and rexp =\n| REvar of path'
)
# loopacc carries a Coq-string temporary name (vwt/vrd expect coq_string).
src = src.replace(
    '| Loopacc of data * (string -> comm)',
    '| Loopacc of data * (coq_string -> comm)'
)

# Destruction of REvar in the printer: apply the path to an empty pslist.
src = re.sub(
    r'\| REvar x -> x\n',
    '| REvar x -> applyps_path x []\n',
    src
)

# If the extracted file does not contain a writeR stub (newer Coq extractions
# inline writeNat/writeR and omit their definitions), provide one here.  It is
# appended at the end so that the RbaseSymbolsImpl module is already in scope.
if 'let writeR' not in src:
    src += '''
(** val writeR : RbaseSymbolsImpl.coq_R -> string **)

let writeR r =
  if r == RbaseSymbolsImpl.coq_R0 then "0.0"
  else if r == RbaseSymbolsImpl.coq_R1 then "1.0"
  else "0.0"
'''

if 'let vstr' not in src:
    src += '''
(** val vstr : nat -> string **)

let vstr cnt = "v" ^ writeNatSym cnt

(** val istr : nat -> string **)

let istr cnt = "i" ^ writeNatSym cnt
'''

if 'let cG_data' not in src:
    src += '''
(** val cG_data : data -> string -> string **)

let rec cG_data d s =
  match d with
  | Num -> "float " ^ s
  | Ary (n0, n, d') -> cG_data d' (s ^ "[" ^ writeNatSym (fle2nat n0 n) ^ "]")
'''

if 'let cnt_var' not in src:
    src += '''
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
'''

with open(OUTFILE, 'w') as f:
    f.write(src)

print(f"Wrote {OUTFILE}")
