(* Ionia Ventus V2a: comm AST -> OpenCL C (conservative subset) printer.
   Vertical slice: Ionia comm AST -> OpenCL kernels -> Ventus spike simulation.

   Mapping (Ventus V2a adaptation, section 3):
   - top-level Parfor (n, ...)        -> one work-item per index:
                                         int gid = get_global_id(0); if (gid < n) { body }
   - reduction pattern                -> workgroup-per-output lowering:
     Parfor (nwg, Num, out, g -> Loopacc t; t=init; Seqfor(m, j -> t=comb(term,t)); out[g]=t)
       => lane-strided serial accumulation (j = ik + lane, 32 lanes)
          + __local LDS tree reduction with barrier(CLK_LOCAL_MEM_FENCE)
          + lane 0 writes out[wg]    (no shuffle on Ventus: LDS is the only
                                      cross-lane channel)
   - nested Parfor (one level)        -> V3 two-level parallelism: outer index
                                         per workgroup (get_group_id), inner index
                                         lane-strided over 32 lanes (get_local_id)
   - deeper nested Parfor / Seqfor    -> serial per-work-item for loop
   - Loopacc / New                    -> private (per-work-item) declarations
   - statement If                     -> if/else
   - REif expression                  -> ternary (cond ? a : b)  [masked loads]

   Conservative subset choices (OpenCL-CTS compiler pass rate on Ventus is
   26/66, so the generated code avoids risky features):
   - scalar float only (no vector types in this slice),
   - no exp/log/sqrt/pow emission needed by the slice operators,
   - barriers only at workgroup-uniform points,
   - single dimension grids, workgroup size fixed at 32 (= 1 Ventus warp).

   Recognizer coverage (see report): Parfor, Seqfor, Loopacc, New, Assign,
   If, Skip, Seq; rexp: REvar/REconst/REadd/REsub/REmul/REdiv/REopp/REscal/
   REif/REdiv_n/REpow/REexp/REln/REsqrt (printers exist; slice uses the first
   9); bexp: BLe/BGt/BLt/BEq/BLeb/BAnd/BNot/BTrue/BFalse.

   Usage: comm_to_opencl <output_dir> [--lenet5|--alexnet|--resnet18]
   Default mode writes the 6 V2a slice kernels; --lenet5 writes the 13-kernel
   LeNet-5 whole-net chain (V3); --alexnet writes the 20-kernel AlexNet chain
   (V4, 48x48 input, channels /4, fc 1024/1024/256); --resnet18 writes the
   69-kernel ResNet-18 chain (V4, 40x40 input, channels /4, fc 256, plus
   BN/Add/GAP kernels).  Writes
   <output_dir>/<kname>.cl per kernel
   plus <output_dir>/manifest.txt with lines:
     kname gws lws (in|out):pname:count ...                                     *)

(* Lenet5_sym shadows [bool] with True/False; alias the host bool first. *)
type host_bool = bool

open Lenet5_sym

let buf_add = Buffer.add_string
let nn = int_to_nat

(* ---------- paths ---------- *)

let rec applyps_to_buf buf v = function
  | [] -> buf_add buf v
  | v0 :: tl -> applyps_to_buf buf (v ^ "[" ^ writeNatSym v0 ^ "]") tl

let rec applyps_path_to_buf buf p ps =
  match p with
  | Var s -> applyps_to_buf buf s ps
  | AryIdx (n, i, p') -> applyps_path_to_buf buf p' ((fin2nat n i) :: ps)

let rec path_to_string = function
  | Var s -> s
  | AryIdx (n, i, p') ->
      path_to_string p' ^ "[" ^ writeNatSym (fin2nat n i) ^ "]"

(* ---------- expressions (rexp / bexp) ---------- *)

let rec rexp_to_buf buf = function
  | REvar p -> applyps_path_to_buf buf p []
  | REconst (_, s) -> buf_add buf s
  | REopp e' -> buf_add buf "(- "; rexp_to_buf buf e'; buf_add buf ")"
  | REadd (e1, e2) ->
      buf_add buf "("; rexp_to_buf buf e1; buf_add buf " + "; rexp_to_buf buf e2; buf_add buf ")"
  | REsub (e1, e2) ->
      buf_add buf "("; rexp_to_buf buf e1; buf_add buf " - "; rexp_to_buf buf e2; buf_add buf ")"
  | REmul (e1, e2) ->
      buf_add buf "("; rexp_to_buf buf e1; buf_add buf " * "; rexp_to_buf buf e2; buf_add buf ")"
  | REdiv (e1, e2) ->
      buf_add buf "("; rexp_to_buf buf e1; buf_add buf ") / ("; rexp_to_buf buf e2; buf_add buf ")"
  | REdiv_n (e', n) ->
      buf_add buf "(("; rexp_to_buf buf e'; buf_add buf ") / (float)("; buf_add buf (writeNatSym n); buf_add buf "))"
  | REscal (r, e') ->
      buf_add buf "("; buf_add buf (writeR r); buf_add buf " * "; rexp_to_buf buf e'; buf_add buf ")"
  | REpow (e', n) ->
      buf_add buf "(pow("; rexp_to_buf buf e'; buf_add buf ", (float)"; buf_add buf (writeNatSym n); buf_add buf "))"
  | REexp e' -> buf_add buf "(exp("; rexp_to_buf buf e'; buf_add buf "))"
  | REln e' -> buf_add buf "(log("; rexp_to_buf buf e'; buf_add buf "))"
  | REsqrt e' -> buf_add buf "(sqrt("; rexp_to_buf buf e'; buf_add buf "))"
  | REif (b, e1, e2) ->
      buf_add buf "(("; bexp_to_buf buf b; buf_add buf ") ? (";
      rexp_to_buf buf e1; buf_add buf ") : ("; rexp_to_buf buf e2; buf_add buf "))"

and bexp_to_buf buf = function
  | BTrue -> buf_add buf "1"
  | BFalse -> buf_add buf "0"
  | BLe (e1, e2) ->
      buf_add buf "("; rexp_to_buf buf e1; buf_add buf " <= "; rexp_to_buf buf e2; buf_add buf ")"
  | BGt (e1, e2) ->
      buf_add buf "("; rexp_to_buf buf e1; buf_add buf " > "; rexp_to_buf buf e2; buf_add buf ")"
  | BLt (i, n) ->
      buf_add buf "("; buf_add buf (writeNatSym i); buf_add buf " < "; buf_add buf (writeNatSym n); buf_add buf ")"
  | BEq (i, n) ->
      buf_add buf "("; buf_add buf (writeNatSym i); buf_add buf " == "; buf_add buf (writeNatSym n); buf_add buf ")"
  | BLeb (i, n) ->
      buf_add buf "("; buf_add buf (writeNatSym i); buf_add buf " <= "; buf_add buf (writeNatSym n); buf_add buf ")"
  | BNot b ->
      buf_add buf "(!"; bexp_to_buf buf b; buf_add buf ")"
  | BAnd (b1, b2) ->
      buf_add buf "("; bexp_to_buf buf b1; buf_add buf " && "; bexp_to_buf buf b2; buf_add buf ")"

(* ---------- acceptors / exps with data shape ---------- *)

let rec exp_to_buf buf d ps e =
  match d with
  | Num -> rexp_to_buf buf (Obj.magic e)
  | Ary (_, _, d') ->
      (match ps with
       | [] -> buf_add buf "/* CG_exp: array expression without index */"
       | v0 :: ps' -> exp_to_buf buf d' ps' ((Obj.magic e : fin -> __) v0))

let rec path_is_none = function
  | Var "none" -> true
  | AryIdx (_, _, p') -> path_is_none p'
  | _ -> false

let rec acc_is_default_none d a =
  match d with
  | Num -> path_is_none (Obj.magic a)
  | Ary (n0, n, d') ->
      let m = fle2nat n0 n in
      (match m with
       | O -> false
       | _ -> acc_is_default_none d' ((Obj.magic a : fin -> __) (nat2fin m O)))

let rec acc_to_buf buf d ps a =
  match d with
  | Num ->
      let p = (Obj.magic a : path) in
      if path_is_none p then
        buf_add buf "/* default acceptor skipped */"
      else
        applyps_path_to_buf buf p ps
  | Ary (_, _, d') ->
      (match ps with
       | [] -> buf_add buf "/* CG_acc: array acceptor without index */"
       | v0 :: ps' -> acc_to_buf buf d' ps' ((Obj.magic a : fin -> __) v0))

(* ---------- commands ---------- *)

let next_id = ref 0

let rec comm_to_buf buf cnt c =
  match c with
  | Skip -> buf_add buf "/* skip */"
  | If (b, (c1, c2)) ->
      buf_add buf "if ("; bexp_to_buf buf b; buf_add buf ") {\n  ";
      comm_to_buf buf cnt c1;
      buf_add buf "\n} else {\n  ";
      comm_to_buf buf cnt c2;
      buf_add buf "\n}"
  | Seq (c1, c2) ->
      comm_to_buf buf cnt c1;
      buf_add buf "\n";
      comm_to_buf buf (add (cnt_var c1) cnt) c2
  | Assign (d, (a, e)) ->
      if acc_is_default_none d a then
        buf_add buf "/* default assignment skipped */ ;"
      else (
        acc_to_buf buf d [] a;
        buf_add buf " = ";
        exp_to_buf buf d [] e;
        buf_add buf ";"
      )
  | New (d, p) ->
      (* private per-work-item buffer (conservative: no pooling, kernel scope) *)
      let vname = "v" ^ string_of_int !next_id in
      incr next_id;
      buf_add buf (cG_data d vname); buf_add buf ";\n";
      let cs = string_to_coq_string vname in
      comm_to_buf buf (add cnt (S O)) (p ((vwt d cs), (vrd d cs)))
  | Loopacc (d, p) ->
      let name = "t" ^ string_of_int !next_id in
      incr next_id;
      buf_add buf (cG_data d name); buf_add buf ";\n";
      comm_to_buf buf (add cnt (S O)) (p (string_to_coq_string name))
  | Seqfor (n, p) ->
      buf_add buf "for (int "; buf_add buf (istr cnt); buf_add buf " = 0; ";
      buf_add buf (istr cnt); buf_add buf " < "; buf_add buf (writeNatSym n);
      buf_add buf "; "; buf_add buf (istr cnt); buf_add buf " += 1) {\n  ";
      let idx = (Var (istr cnt) : nat) in
      comm_to_buf buf (add cnt (S O)) (p idx);
      buf_add buf "\n}"
  | Parfor (n, d, a, p) ->
      (* deeper-nested Parfor (level 3+): conservative fallback to a serial
         per-work-item loop.  Level-2 nesting is handled in emit_top. *)
      buf_add buf "/* nested Parfor -> serial loop (conservative) */\n";
      buf_add buf "for (int "; buf_add buf (istr cnt); buf_add buf " = 0; ";
      buf_add buf (istr cnt); buf_add buf " < "; buf_add buf (writeNatSym n);
      buf_add buf "; "; buf_add buf (istr cnt); buf_add buf " += 1) {\n  ";
      let idx = (Var (istr cnt) : nat) in
      comm_to_buf buf (add cnt (S O)) (p idx (idxAcc n d a idx));
      buf_add buf "\n}"

(* ---------- reduction recognizer + LDS tree lowering ---------- *)

let wg_size = 32  (* one Ventus warp = 32 lanes; LDS tree needs a power of 2 *)

(* emit the lowered reduction kernel body; returns unit.
   [post] is an optional extra summand for the final write
   (fc layers: out = reduced_sum + bias). *)
let emit_reduction buf nwg accg tname init m f is_max post =
  buf_add buf "  /* reduction: lane-strided serial accumulate + LDS tree */\n";
  buf_add buf "  int wg = get_group_id(0);\n";
  buf_add buf "  int lane = get_local_id(0);\n";
  buf_add buf "  __local float lds["; buf_add buf (string_of_int wg_size); buf_add buf "];\n";
  buf_add buf "  if (wg < "; buf_add buf (writeNatSym nwg); buf_add buf ") {\n";
  buf_add buf "    float "; buf_add buf tname; buf_add buf ";\n";
  buf_add buf "    "; buf_add buf tname; buf_add buf " = ";
  exp_to_buf buf Num [] init; buf_add buf ";\n";
  buf_add buf "    for (int ik = 0; ik < "; buf_add buf (writeNatSym m);
  buf_add buf "; ik += "; buf_add buf (string_of_int wg_size); buf_add buf ") {\n";
  buf_add buf "      int jred = ik + lane;\n";
  buf_add buf "      if (jred < "; buf_add buf (writeNatSym m); buf_add buf ") {\n        ";
  comm_to_buf buf O (f ((Var "jred" : nat)));
  buf_add buf "\n      }\n    }\n";
  buf_add buf "    lds[lane] = "; buf_add buf tname; buf_add buf ";\n";
  buf_add buf "    barrier(CLK_LOCAL_MEM_FENCE);\n";
  buf_add buf "    for (int s = "; buf_add buf (string_of_int (wg_size / 2));
  buf_add buf "; s > 0; s >>= 1) {\n";
  buf_add buf "      if (lane < s) {\n";
  if is_max then
    buf_add buf "        lds[lane] = (lds[lane + s] > lds[lane]) ? lds[lane + s] : lds[lane];\n"
  else
    buf_add buf "        lds[lane] = lds[lane] + lds[lane + s];\n";
  buf_add buf "      }\n";
  buf_add buf "      barrier(CLK_LOCAL_MEM_FENCE);\n";
  buf_add buf "    }\n";
  buf_add buf "    if (lane == 0) {\n      ";
  applyps_path_to_buf buf (Obj.magic accg) [];
  buf_add buf " = ";
  (match post with
   | None -> buf_add buf "lds[0];\n    }\n"
   | Some e ->
       buf_add buf "(lds[0] + "; rexp_to_buf buf e; buf_add buf ");\n    }\n");
  buf_add buf "  }\n"

(* recognize  Parfor (nwg, Num, out, g accg ->
       Loopacc Num t.  t = init; Seqfor (m, j -> t = comb(term(g,j), t)); accg = t)
   with comb = + (sum) or REif(BGt) (max).  Returns true when lowered. *)
let try_lower_reduction buf nwg d a p =
  match d with
  | Num ->
      let g = (Var "wg" : nat) in
      let accg = idxAcc nwg d a g in
      (match p g accg with
       | Loopacc (Num, q) ->
           let tname = "tred" in
           (match q (string_to_coq_string tname) with
            | Seq (Assign (Num, (tacc, init)),
                   Seq (Seqfor (m, f), Assign (Num, (oacc, oexp)))) ->
                let ts = path_to_string (Obj.magic tacc) in
                let out_ok =
                  path_to_string (Obj.magic oacc) = path_to_string (Obj.magic accg) in
                let oexp_ok, post =
                  match (Obj.magic oexp : rexp) with
                  | REvar rp -> (path_to_string rp = ts, None)
                  | REadd (REvar rp, e) when path_to_string rp = ts -> (true, Some e)
                  | REadd (e, REvar rp) when path_to_string rp = ts -> (true, Some e)
                  | _ -> (false, None) in
                if out_ok && oexp_ok then
                  (match f ((Var "jred" : nat)) with
                   | Assign (Num, (tacc2, e2)) ->
                       if path_to_string (Obj.magic tacc2) <> ts then false
                       else
                         let e2r = (Obj.magic e2 : rexp) in
                         (match e2r with
                          | REadd (_, REvar rp) when path_to_string rp = ts ->
                              emit_reduction buf nwg accg tname init m f false post; true
                          | REadd (REvar rp, _) when path_to_string rp = ts ->
                              emit_reduction buf nwg accg tname init m f false post; true
                          | REif (BGt (_, REvar rp), _, REvar rp2)
                            when path_to_string rp = ts && path_to_string rp2 = ts ->
                              emit_reduction buf nwg accg tname init m f true post; true
                          | _ -> false)
                   | _ -> false)
                else false
            | _ -> false)
       | _ -> false)
  | Ary _ -> false

(* ---------- top level: Parfor -> work-item mapping ---------- *)

let emit_top buf c =
  match c with
  | Parfor (n, d, a, p) ->
      if try_lower_reduction buf n d a p then ()
      else begin
        let g = (Var "wg" : nat) in
        match p g (idxAcc n d a g) with
        | Parfor (n2, d2, a2, p2) ->
            (* V3: nested Parfor -> two-level parallelism.
               Outer index -> workgroup (get_group_id), inner index ->
               lane-strided loop over the 32 lanes (get_local_id).
               Used by conv/pool layers: output channel x pixel. *)
            buf_add buf "  /* nested Parfor -> workgroup x lane two-level */\n";
            buf_add buf "  int wg = get_group_id(0);\n";
            buf_add buf "  int lane = get_local_id(0);\n";
            buf_add buf "  if (wg < "; buf_add buf (writeNatSym n); buf_add buf ") {\n";
            buf_add buf "    for (int ll = lane; ll < ";
            buf_add buf (writeNatSym n2);
            buf_add buf "; ll += "; buf_add buf (string_of_int wg_size);
            buf_add buf ") {\n  ";
            let l = (Var "ll" : nat) in
            comm_to_buf buf O (p2 l (idxAcc n2 d2 a2 l));
            buf_add buf "\n    }\n  }\n"
        | _ ->
            buf_add buf "  /* Parfor -> one work-item per index */\n";
            buf_add buf "  int gid = get_global_id(0);\n";
            buf_add buf "  if (gid < "; buf_add buf (writeNatSym n); buf_add buf ") {\n  ";
            let p_arg = (Var "gid" : nat) in
            comm_to_buf buf O (p p_arg (idxAcc n d a p_arg));
            buf_add buf "\n  }\n"
      end
  | _ ->
      buf_add buf "  /* no top-level Parfor: serial kernel (single work-item) */\n";
      buf_add buf "  if (get_global_id(0) == 0) {\n  ";
      comm_to_buf buf O c;
      buf_add buf "\n  }\n"

(* ---------- kernel specs ---------- *)

type param = string * int * host_bool (* name, element count, is_output *)

type kspec = {
  kname : string;
  gws : int;
  lws : int;
  params : param list;
  body : comm;
}

let vrd_f n s = vrd (Ary (nn n, fle_self (nn n), Num)) (string_to_coq_string s)
let vwt_f n s = vwt (Ary (nn n, fle_self (nn n), Num)) (string_to_coq_string s)

(* relu1d: out[i] = max(x[i], 0)   (Parfor form of the extracted reLU1d_comm) *)
let seq2 a b = Seq (a, b)

let relu1d_par n =
  let nN = nn n in
  let x = vrd_f n "x" in
  let out = vwt_f n "out" in
  Parfor (nN, Num, out, fun i acci ->
    let xi = idx nN Num x i in
    If (leExp xi zero, (Assign (Num, (acci, zero)), Assign (Num, (acci, xi)))))

(* add1d: out[i] = x[i] + y[i] *)
let add1d_par n =
  let nN = nn n in
  let x = vrd_f n "x" and y = vrd_f n "y" in
  let out = vwt_f n "out" in
  Parfor (nN, Num, out, fun i acci ->
    Assign (Num, (acci, add0 (idx nN Num x i) (idx nN Num y i))))

(* sum: out[wg] = sum_{j<m} x[wg*m + j]   (workgroup reduction) *)
let sum_wg_comm nwg m =
  let total = nwg * m in
  let x = vrd_f total "x" in
  let out = vwt_f nwg "out" in
  Parfor (nn nwg, Num, out, fun g accg ->
    Loopacc (Num, fun tmp ->
      let t = vwt Num tmp and tv = vrd Num tmp in
      let body j = Assign (Num, (t, add0 (idx (nn total) Num x (add (mul g (nn m)) j)) tv)) in
      seq2 (Assign (Num, (t, zero)))
        (seq2 (Seqfor (nn m, body)) (Assign (Num, (accg, tv))))))

(* max: out[wg] = max_{j<m} x[wg*m + j]   (workgroup reduction, REif form) *)
let max_wg_comm nwg m =
  let total = nwg * m in
  let x = vrd_f total "x" in
  let out = vwt_f nwg "out" in
  Parfor (nn nwg, Num, out, fun g accg ->
    Loopacc (Num, fun tmp ->
      let t = vwt Num tmp and tv = vrd Num tmp in
      let term j = idx (nn total) Num x (add (mul g (nn m)) j) in
      let body j =
        let tj = term j in
        Assign (Num, (t, Obj.magic (REif (gtExp tj tv,
                                          (Obj.magic tj : rexp),
                                          (Obj.magic tv : rexp))))) in
      seq2 (Assign (Num, (t, term O)))
        (seq2 (Seqfor (nn m, body)) (Assign (Num, (accg, tv))))))

(* conv2d (small): in CxHxW, kernel FxCxKxK, pad P, stride 1 ->
   out[f,i,j] = b[f] + sum_{c,ki,kj} w[f,c,ki,kj] * x[c, i+ki-P, j+kj-P]
   with zero padding via masked (ternary) loads.  Flat layouts. *)
let conv2d_comm fN cN hN wN kN pN =
  let out_n = fN * hN * wN in
  let out = vwt_f out_n "out" in
  Parfor (nn out_n, Num, out, fun o acco ->
    let f = Nat.div o (nn (hN * wN)) in
    let i = Nat.modulo (Nat.div o (nn wN)) (nn hN) in
    let j = Nat.modulo o (nn wN) in
    Loopacc (Num, fun tmp ->
      let t = vwt Num tmp and tv = vrd Num tmp in
      let bias = (Obj.magic (REvar (AryIdx (O, f, Var "b"))) : exp) in
      let body c ki kj =
        let ii = add i ki and jj = add j kj in
        let in_r = BAnd (BLeb (nn pN, ii), BLeb (ii, nn (pN + hN - 1))) in
        let in_c = BAnd (BLeb (nn pN, jj), BLeb (jj, nn (pN + wN - 1))) in
        let cond = BAnd (in_r, in_c) in
        let row = Sub (ii, nn pN) and col = Sub (jj, nn pN) in
        let x_idx = add (add (mul c (nn (hN * wN))) (mul row (nn wN))) col in
        let w_idx = add (add (mul f (nn (cN * kN * kN))) (mul c (nn (kN * kN))))
                      (add (mul ki (nn kN)) kj) in
        let masked = REif (cond, REvar (AryIdx (O, x_idx, Var "x")),
                           (Obj.magic zero : rexp)) in
        let term = REmul (REvar (AryIdx (O, w_idx, Var "w")), masked) in
        Assign (Num, (t, add0 (Obj.magic term) tv)) in
      seq2 (Assign (Num, (t, bias)))
        (seq2 (Seqfor (nn cN, fun c ->
                 Seqfor (nn kN, fun ki ->
                   Seqfor (nn kN, fun kj -> body c ki kj))))
           (Assign (Num, (acco, tv))))))

let specs = [
  { kname = "relu1d"; gws = 1024; lws = 32;
    params = [("x", 1024, false); ("out", 1024, true)];
    body = relu1d_par 1024 };
  { kname = "add1d"; gws = 1024; lws = 32;
    params = [("x", 1024, false); ("y", 1024, false); ("out", 1024, true)];
    body = add1d_par 1024 };
  { kname = "sum256"; gws = 32; lws = 32;
    params = [("x", 256, false); ("out", 1, true)];
    body = sum_wg_comm 1 256 };
  { kname = "sum4096"; gws = 512; lws = 32;
    params = [("x", 4096, false); ("out", 16, true)];
    body = sum_wg_comm 16 256 };
  { kname = "max256"; gws = 32; lws = 32;
    params = [("x", 256, false); ("out", 1, true)];
    body = max_wg_comm 1 256 };
  { kname = "conv2d"; gws = 256; lws = 32;
    params = [("x", 128, false); ("w", 72, false); ("b", 4, false); ("out", 256, true)];
    body = conv2d_comm 4 2 8 8 3 1 };
]

(* ================= LeNet-5 whole-net chain (V3) ================= *)
(* Mirrors [LeNet5_Par.lenet5_par_comm]: 13 kernels over flat float
   buffers.  conv/pool use the two-level nested-Parfor lowering
   (output channel x pixel); fc layers use the workgroup-reduction
   lowering with bias folded into the accumulator init; relu/flatten
   are flat gid maps; softmax is a single-work-item serial kernel
   (10 elements; its division parfor is kept sequential, justified by
   [DPIA_Rules.parfor_seqfor] / [LeNet5_Par.softmax1d_par_eval]). *)

let scs = string_to_coq_string

(* 2-D acceptor over a flat buffer; the per-element acceptors are never
   printed (bodies assign through manual flat paths), they only need to
   survive idxAcc during lowering. *)
let vwt_2d n1 n2 s =
  vwt (Ary (nn n1, fle_self (nn n1), Ary (nn n2, fle_self (nn n2), Num))) (scs s)

(* manual flat acceptor path  name[idx]  (idx may be symbolic) *)
let flat_acc name idx = (Obj.magic (AryIdx (O, idx, Var name)) : acc)
let flat_rd name idx = (Obj.magic (REvar (AryIdx (O, idx, Var name))) : exp)

(* conv (valid, stride 1): two-level Parfor, channel x pixel.
   out[f,i,j] = sum_{c,ki,kj} w[f,c,ki,kj] * x[c,i+ki,j+kj];  then + b[f]. *)
let conv2l_comm xn wn bn on fN cN hIn wIn kN =
  let ho = hIn - kN + 1 and wo = wIn - kN + 1 in
  let npix = ho * wo in
  let out = vwt_2d fN npix on in
  Parfor (nn fN, Ary (nn npix, fle_self (nn npix), Num), out, fun f accf ->
    Parfor (nn npix, Num, accf, fun p _accp ->
      let i = Nat.div p (nn wo) and j = Nat.modulo p (nn wo) in
      let accp = flat_acc on (add (mul f (nn npix)) p) in
      Loopacc (Num, fun tmp ->
        let t = vwt Num tmp and tv = vrd Num tmp in
        let body c ki kj =
          let ii = add i ki and jj = add j kj in
          let x_idx = add (add (mul c (nn (hIn * wIn))) (mul ii (nn wIn))) jj in
          let w_idx = add (add (mul f (nn (cN * kN * kN))) (mul c (nn (kN * kN))))
                        (add (mul ki (nn kN)) kj) in
          Assign (Num, (t, add0 (mul0 (flat_rd wn w_idx) (flat_rd xn x_idx)) tv)) in
        seq2 (Assign (Num, (t, zero)))
          (seq2 (Seqfor (nn cN, fun c ->
                   Seqfor (nn kN, fun ki ->
                     Seqfor (nn kN, fun kj -> body c ki kj))))
            (Assign (Num, (accp, add0 tv (flat_rd bn f))))))))

(* maxpool 2x2 stride 2: two-level Parfor, channel x pixel. *)
let pool2l_comm xn on fN hN wN =
  let ho = hN / 2 and wo = wN / 2 in
  let npix = ho * wo in
  let out = vwt_2d fN npix on in
  Parfor (nn fN, Ary (nn npix, fle_self (nn npix), Num), out, fun f accf ->
    Parfor (nn npix, Num, accf, fun p _accp ->
      let i = Nat.div p (nn wo) and j = Nat.modulo p (nn wo) in
      let accp = flat_acc on (add (mul f (nn npix)) p) in
      let xat ii jj =
        flat_rd xn (add (add (mul f (nn (hN * wN)))
                          (mul (add (mul (nn 2) i) ii) (nn wN)))
                      (add (mul (nn 2) j) jj)) in
      New (Num, fun tmp ->
        let (wacc, v) = tmp in
        let upd ii jj =
          If (gtExp (xat ii jj) v,
              (Assign (Num, (wacc, xat ii jj)), Assign (Num, (wacc, v)))) in
        seq2 (Assign (Num, (wacc, xat O O)))
          (seq2 (Seqfor (nn 2, fun i1 -> Seqfor (nn 2, fun j1 -> upd i1 j1)))
            (Assign (Num, (accp, v)))))))

(* relu / flatten: flat gid map *)
let relu1d_named xn on n =
  Parfor (nn n, Num, vwt_f n on, fun i acci ->
    let xi = flat_rd xn i in
    If (leExp xi zero, (Assign (Num, (acci, zero)), Assign (Num, (acci, xi)))))

let copy1d_named xn on n =
  Parfor (nn n, Num, vwt_f n on, fun i acci ->
    Assign (Num, (acci, flat_rd xn i)))

(* fc: workgroup reduction over the dot product; the final write adds the
   bias (out = t + b[g]), mirroring [Linear1d_par_comm]. *)
let fc_wg_comm xn wn bn on nwg m =
  let out = vwt_f nwg on in
  Parfor (nn nwg, Num, out, fun g accg ->
    Loopacc (Num, fun tmp ->
      let t = vwt Num tmp and tv = vrd Num tmp in
      let body j =
        Assign (Num, (t, add0 (mul0 (flat_rd xn j)
                                    (flat_rd wn (add (mul g (nn m)) j))) tv)) in
      seq2 (Assign (Num, (t, zero)))
        (seq2 (Seqfor (nn m, body))
          (Assign (Num, (accg, add0 tv (flat_rd bn g)))))))

(* softmax: serial single-work-item kernel (n = 10) *)
let softmax_comm xn on n =
  let input = vrd_f n xn in
  let out = vwt_f n on in
  Loopacc (Num, fun tmp ->
    let t = vwt Num tmp and tv = vrd Num tmp in
    seq2 (Assign (Num, (t, zero)))
      (seq2 (Seqfor (nn n, fun i ->
               Assign (Num, (t, add0 (pExp (idx (nn n) Num input i)) tv))))
        (Seqfor (nn n, fun j ->
           Assign (Num, (idxAcc (nn n) Num out j,
                         div0 (pExp (idx (nn n) Num input j)) tv))))))

(* round up to a multiple of the 32-lane workgroup *)
let gws_of n = ((n + wg_size - 1) / wg_size) * wg_size

let lenet5_specs = [
  { kname = "conv1"; gws = 6 * wg_size; lws = wg_size;
    params = [("input", 1024, false); ("kc1", 150, false); ("bc1", 6, false);
              ("c1", 4704, true)];
    body = conv2l_comm "input" "kc1" "bc1" "c1" 6 1 32 32 5 };
  { kname = "relu1"; gws = gws_of 4704; lws = wg_size;
    params = [("c1", 4704, false); ("r1", 4704, true)];
    body = relu1d_named "c1" "r1" 4704 };
  { kname = "pool1"; gws = 6 * wg_size; lws = wg_size;
    params = [("r1", 4704, false); ("p1", 1176, true)];
    body = pool2l_comm "r1" "p1" 6 28 28 };
  { kname = "conv2"; gws = 16 * wg_size; lws = wg_size;
    params = [("p1", 1176, false); ("kc2", 2400, false); ("bc2", 16, false);
              ("c2", 1600, true)];
    body = conv2l_comm "p1" "kc2" "bc2" "c2" 16 6 14 14 5 };
  { kname = "relu2"; gws = gws_of 1600; lws = wg_size;
    params = [("c2", 1600, false); ("r2", 1600, true)];
    body = relu1d_named "c2" "r2" 1600 };
  { kname = "pool2"; gws = 16 * wg_size; lws = wg_size;
    params = [("r2", 1600, false); ("p2", 400, true)];
    body = pool2l_comm "r2" "p2" 16 10 10 };
  { kname = "flatten"; gws = gws_of 400; lws = wg_size;
    params = [("p2", 400, false); ("flat", 400, true)];
    body = copy1d_named "p2" "flat" 400 };
  { kname = "fc1"; gws = 120 * wg_size; lws = wg_size;
    params = [("flat", 400, false); ("fc1_w", 48000, false);
              ("fc1_b", 120, false); ("fc1o", 120, true)];
    body = fc_wg_comm "flat" "fc1_w" "fc1_b" "fc1o" 120 400 };
  { kname = "relu3"; gws = gws_of 120; lws = wg_size;
    params = [("fc1o", 120, false); ("fc1r", 120, true)];
    body = relu1d_named "fc1o" "fc1r" 120 };
  { kname = "fc2"; gws = 84 * wg_size; lws = wg_size;
    params = [("fc1r", 120, false); ("fc2_w", 10080, false);
              ("fc2_b", 84, false); ("fc2o", 84, true)];
    body = fc_wg_comm "fc1r" "fc2_w" "fc2_b" "fc2o" 84 120 };
  { kname = "relu4"; gws = gws_of 84; lws = wg_size;
    params = [("fc2o", 84, false); ("fc2r", 84, true)];
    body = relu1d_named "fc2o" "fc2r" 84 };
  { kname = "fc3"; gws = 10 * wg_size; lws = wg_size;
    params = [("fc2r", 84, false); ("fc3_w", 840, false);
              ("fc3_b", 10, false); ("fc3o", 10, true)];
    body = fc_wg_comm "fc2r" "fc3_w" "fc3_b" "fc3o" 10 84 };
  { kname = "softmax"; gws = wg_size; lws = wg_size;
    params = [("fc3o", 10, false); ("output", 10, true)];
    body = softmax_comm "fc3o" "output" 10 };
]

(* ================= V4 shared builders (AlexNet / ResNet-18) ================= *)

let sub0 x y = Obj.magic (REsub ((Obj.magic x), (Obj.magic y)))
let sqrt0 x = Obj.magic (REsqrt (Obj.magic x))
let eps_exp : exp =
  Obj.magic (REconst (RbaseSymbolsImpl.coq_R0, "1e-05f"))

(* conv with zero padding pN and stride sN: two-level Parfor, channel x
   pixel.  out[f,i,j] = b[f] + sum_{c,ki,kj} w[f,c,ki,kj] *
   xpad[c, i*sN+ki, j*sN+kj], the padded read lowered to a masked
   (ternary) load.  Reduction order (c, ki, kj), bias added at the end —
   mirrors [Conv4d_comm] / [Conv4d_3x3_s2_comm] etc. *)
let convps2l_comm xn wn bn on fN cN hIn wIn kN pN sN =
  let ho = (hIn - kN + 2 * pN) / sN + 1 in
  let wo = (wIn - kN + 2 * pN) / sN + 1 in
  let npix = ho * wo in
  let out = vwt_2d fN npix on in
  Parfor (nn fN, Ary (nn npix, fle_self (nn npix), Num), out, fun f accf ->
    Parfor (nn npix, Num, accf, fun p _accp ->
      let i = Nat.div p (nn wo) and j = Nat.modulo p (nn wo) in
      let accp = flat_acc on (add (mul f (nn npix)) p) in
      Loopacc (Num, fun tmp ->
        let t = vwt Num tmp and tv = vrd Num tmp in
        let body c ki kj =
          let ii = add (mul (nn sN) i) ki and jj = add (mul (nn sN) j) kj in
          let in_r = BAnd (BLeb (nn pN, ii), BLeb (ii, nn (pN + hIn - 1))) in
          let in_c = BAnd (BLeb (nn pN, jj), BLeb (jj, nn (pN + wIn - 1))) in
          let cond = BAnd (in_r, in_c) in
          let row = Sub (ii, nn pN) and col = Sub (jj, nn pN) in
          let x_idx = add (add (mul c (nn (hIn * wIn))) (mul row (nn wIn))) col in
          let w_idx = add (add (mul f (nn (cN * kN * kN))) (mul c (nn (kN * kN))))
                        (add (mul ki (nn kN)) kj) in
          let masked = REif (cond, REvar (AryIdx (O, x_idx, Var xn)),
                             (Obj.magic zero : rexp)) in
          let term = REmul (REvar (AryIdx (O, w_idx, Var wn)), masked) in
          Assign (Num, (t, add0 (Obj.magic term) tv)) in
        seq2 (Assign (Num, (t, zero)))
          (seq2 (Seqfor (nn cN, fun c ->
                   Seqfor (nn kN, fun ki ->
                     Seqfor (nn kN, fun kj -> body c ki kj))))
            (Assign (Num, (accp, add0 tv (flat_rd bn f))))))))

(* maxpool k x k, stride sN, zero padding pN: two-level Parfor, channel x
   pixel.  out = max(0, window) — mirrors [Maxpool4d_pad_comm] whose [maxl]
   reduction starts at [zero] over the zero-padded window. *)
let poolksp2l_comm xn on fN hN wN kN sN pN =
  let ho = (hN - kN + 2 * pN) / sN + 1 in
  let wo = (wN - kN + 2 * pN) / sN + 1 in
  let npix = ho * wo in
  let out = vwt_2d fN npix on in
  Parfor (nn fN, Ary (nn npix, fle_self (nn npix), Num), out, fun f accf ->
    Parfor (nn npix, Num, accf, fun p _accp ->
      let i = Nat.div p (nn wo) and j = Nat.modulo p (nn wo) in
      let accp = flat_acc on (add (mul f (nn npix)) p) in
      Loopacc (Num, fun tmp ->
        let t = vwt Num tmp and tv = vrd Num tmp in
        let body ki kj =
          let ii = add (mul (nn sN) i) ki and jj = add (mul (nn sN) j) kj in
          let cond =
            BAnd (BAnd (BLeb (nn pN, ii), BLeb (ii, nn (pN + hN - 1))),
                  BAnd (BLeb (nn pN, jj), BLeb (jj, nn (pN + wN - 1)))) in
          let row = Sub (ii, nn pN) and col = Sub (jj, nn pN) in
          let x_idx = add (add (mul f (nn (hN * wN))) (mul row (nn wN))) col in
          let term = (Obj.magic
            (REif (cond, REvar (AryIdx (O, x_idx, Var xn)),
                   (Obj.magic zero : rexp))) : exp) in
          Assign (Num, (t, Obj.magic (REif (gtExp term tv,
                         (Obj.magic term : rexp), (Obj.magic tv : rexp))))) in
        seq2 (Assign (Num, (t, zero)))
          (seq2 (Seqfor (nn kN, fun ki -> Seqfor (nn kN, fun kj -> body ki kj)))
            (Assign (Num, (accp, tv)))))))

(* BatchNorm (inference mode, given per-channel statistics; route-B
   semantics of resnet18_streaming_output.c):
   y[c,i,j] = (x - mean[c]) / sqrt(var[c] + eps) * gamma[c] + beta[c].
   Flat gid map over C*npix. *)
let bn_comm xn mn vn gn zn on cN npix =
  let n = cN * npix in
  Parfor (nn n, Num, vwt_f n on, fun i acci ->
    let c = Nat.div i (nn npix) in
    let xi = flat_rd xn i in
    let denom = sqrt0 (add0 (flat_rd vn c) eps_exp) in
    Assign (Num, (acci,
      add0 (mul0 (div0 (sub0 xi (flat_rd mn c)) denom) (flat_rd gn c))
           (flat_rd zn c))))

(* residual add: out[i] = x[i] + y[i] (flat gid map) *)
let add1d_named xn yn on n =
  Parfor (nn n, Num, vwt_f n on, fun i acci ->
    Assign (Num, (acci, add0 (flat_rd xn i) (flat_rd yn i))))

(* global average pooling: out[c] = (sum_p x[c,p]) / npix; one work-item
   per channel with a serial accumulation loop (npix is tiny: 4 at V4
   sizes).  Mirrors [GlobalAvgPool4d_comm]. *)
let gap_comm xn on cN npix =
  Parfor (nn cN, Num, vwt_f cN on, fun c accc ->
    Loopacc (Num, fun tmp ->
      let t = vwt Num tmp and tv = vrd Num tmp in
      let body p =
        Assign (Num, (t, add0 (flat_rd xn (add (mul c (nn npix)) p)) tv)) in
      seq2 (Assign (Num, (t, zero)))
        (seq2 (Seqfor (nn npix, body))
          (Assign (Num, (accc,
            Obj.magic (REdiv_n ((Obj.magic tv : rexp), nn npix))))))))

(* ================= AlexNet whole-net chain (V4) ================= *)
(* Mirrors [AlexNet_DPIA.AlexNet]: 5 conv (padded/strided) + relu, 3
   maxpool k3 s2, flatten, 3 fc + softmax.  Reduced for the spike budget:
   input 48x48 (minimum size keeping every stage >= 1x1 under k11/s4 +
   2x k3/s2 + 3x k3), channels /4 (16/48/96/64/64), fc 1024/1024/256.
   Layer sizes: c1=11x11, p1=5x5, c2=5x5, p2=2x2, c3=c4=c5=2x2, p5=1x1,
   flat=64. *)

let alexnet_specs =
  let conv name xn wn bn on fN cN h k p s =
    let ho = (h - k + 2 * p) / s + 1 in
    { kname = name; gws = fN * wg_size; lws = wg_size;
      params = [(xn, cN * h * h, false); (wn, fN * cN * k * k, false);
                (bn, fN, false); (on, fN * ho * ho, true)];
      body = convps2l_comm xn wn bn on fN cN h h k p s } in
  let pool name xn on fN h =
    let ho = (h - 3) / 2 + 1 in
    { kname = name; gws = fN * wg_size; lws = wg_size;
      params = [(xn, fN * h * h, false); (on, fN * ho * ho, true)];
      body = poolksp2l_comm xn on fN h h 3 2 0 } in
  let relu name xn on n =
    { kname = name; gws = gws_of n; lws = wg_size;
      params = [(xn, n, false); (on, n, true)];
      body = relu1d_named xn on n } in
  let fc name xn wn bn on nwg m =
    { kname = name; gws = nwg * wg_size; lws = wg_size;
      params = [(xn, m, false); (wn, nwg * m, false); (bn, nwg, false);
                (on, nwg, true)];
      body = fc_wg_comm xn wn bn on nwg m } in
  [ conv "conv1" "input"  "c1_w" "c1_b" "c1" 16 3  48 11 2 4;
    relu "relu1" "c1"  "r1"  (16 * 121);
    pool "pool1" "r1"  "p1"  16 11;
    conv "conv2" "p1"   "c2_w" "c2_b" "c2" 48 16 5  5 2 1;
    relu "relu2" "c2"  "r2"  (48 * 25);
    pool "pool2" "r2"  "p2"  48 5;
    conv "conv3" "p2"   "c3_w" "c3_b" "c3" 96 48 2  3 1 1;
    relu "relu3" "c3"  "r3"  (96 * 4);
    conv "conv4" "r3"   "c4_w" "c4_b" "c4" 64 96 2  3 1 1;
    relu "relu4" "c4"  "r4"  (64 * 4);
    conv "conv5" "r4"   "c5_w" "c5_b" "c5" 64 64 2  3 1 1;
    relu "relu5" "c5"  "r5"  (64 * 4);
    pool "pool5" "r5"  "p5"  64 2;
    { kname = "flatten"; gws = gws_of 64; lws = wg_size;
      params = [("p5", 64, false); ("flat", 64, true)];
      body = copy1d_named "p5" "flat" 64 };
    fc  "fc1"  "flat" "f1_w" "f1_b" "fc1o" 1024 64;
    relu "relu6" "fc1o" "fc1r" 1024;
    fc  "fc2"  "fc1r" "f2_w" "f2_b" "fc2o" 1024 1024;
    relu "relu7" "fc2o" "fc2r" 1024;
    fc  "fc3"  "fc2r" "f3_w" "f3_b" "fc3o" 256 1024;
    { kname = "softmax"; gws = wg_size; lws = wg_size;
      params = [("fc3o", 256, false); ("output", 256, true)];
      body = softmax_comm "fc3o" "output" 256 };
  ]

(* ================= ResNet-18 whole-net chain (V4) ================= *)
(* Mirrors [ResNet18_DPIA.resnet18]: stem conv 7x7 p3 s2 -> BN -> ReLU ->
   maxpool k3 s2 p1 -> 8 basic blocks (layers 2/2/2/2; blocks 1 of layers
   2-4 have a 1x1 s2 projection shortcut) -> GAP -> fc -> softmax.
   BN follows the route-B parameterized semantics (given mean/var, the
   form extracted into resnet18_streaming_output.c and validated by
   validate_resnet18.py).  Reduced for the spike budget: input 40x40,
   channels /4 (16/32/64/128), fc 256.  Layer sizes: 20x20 -> pool 10x10
   -> 10, 5, 3, 2. *)

let resnet18_specs =
  let conv name xn wn bn on fN cN hIn ho k p s =
    { kname = name; gws = fN * wg_size; lws = wg_size;
      params = [(xn, cN * hIn * hIn, false); (wn, fN * cN * k * k, false);
                (bn, fN, false); (on, fN * ho * ho, true)];
      body = convps2l_comm xn wn bn on fN cN hIn hIn k p s } in
  let bn name xn pfx on cN npix =
    { kname = name; gws = gws_of (cN * npix); lws = wg_size;
      params = [(xn, cN * npix, false);
                (pfx ^ "m", cN, false); (pfx ^ "v", cN, false);
                (pfx ^ "g", cN, false); (pfx ^ "b", cN, false);
                (on, cN * npix, true)];
      body = bn_comm xn (pfx ^ "m") (pfx ^ "v") (pfx ^ "g") (pfx ^ "b")
               on cN npix } in
  let relu name xn on n =
    { kname = name; gws = gws_of n; lws = wg_size;
      params = [(xn, n, false); (on, n, true)];
      body = relu1d_named xn on n } in
  let addk name xn yn on n =
    { kname = name; gws = gws_of n; lws = wg_size;
      params = [(xn, n, false); (yn, n, false); (on, n, true)];
      body = add1d_named xn yn on n } in
  (* one basic block: 7 kernels, +2 (shortcut conv + bn) when [proj] *)
  let block pfx xn cin cout hin hout stride proj =
    let npix = hout * hout in
    let c1 = conv (pfx ^ "_c1") xn (pfx ^ "_w1") (pfx ^ "_b1") (pfx ^ "_c1o")
               cout cin hin hout 3 1 stride in
    let b1 = bn (pfx ^ "_bn1") (pfx ^ "_c1o") (pfx ^ "_bn1") (pfx ^ "_bn1o")
               cout npix in
    let r1 = relu (pfx ^ "_r1") (pfx ^ "_bn1o") (pfx ^ "_r1o") (cout * npix) in
    let c2 = conv (pfx ^ "_c2") (pfx ^ "_r1o") (pfx ^ "_w2") (pfx ^ "_b2")
               (pfx ^ "_c2o") cout cout hout hout 3 1 1 in
    let b2 = bn (pfx ^ "_bn2") (pfx ^ "_c2o") (pfx ^ "_bn2") (pfx ^ "_bn2o")
               cout npix in
    let sc =
      if proj then
        [ conv (pfx ^ "_sc") xn (pfx ^ "_scw") (pfx ^ "_scb") (pfx ^ "_sco")
            cout cin hin hout 1 0 stride;
          bn (pfx ^ "_sbn") (pfx ^ "_sco") (pfx ^ "_sbn") (pfx ^ "_sbno")
            cout npix ]
      else [] in
    let id = if proj then pfx ^ "_sbno" else xn in
    let ad = addk (pfx ^ "_add") (pfx ^ "_bn2o") id (pfx ^ "_addo")
               (cout * npix) in
    let r2 = relu (pfx ^ "_r2") (pfx ^ "_addo") (pfx ^ "_r2o") (cout * npix) in
    [c1; b1; r1; c2; b2] @ sc @ [ad; r2] in
  let stem =
    [ conv "s_conv" "input" "sc1_w" "sc1_b" "s_c1o" 16 3 40 20 7 3 2;
      bn "s_bn" "s_c1o" "s_bn" "s_bn1o" 16 400;
      relu "s_relu" "s_bn1o" "s_r1o" (16 * 400);
      { kname = "s_pool"; gws = 16 * wg_size; lws = wg_size;
        params = [("s_r1o", 16 * 400, false); ("spool", 16 * 100, true)];
        body = poolksp2l_comm "s_r1o" "spool" 16 20 20 3 2 1 } ] in
  let tail =
    [ { kname = "gap"; gws = gws_of 128; lws = wg_size;
        params = [("l4b2_r2o", 128 * 4, false); ("gapo", 128, true)];
        body = gap_comm "l4b2_r2o" "gapo" 128 4 };
      { kname = "fc"; gws = 256 * wg_size; lws = wg_size;
        params = [("gapo", 128, false); ("fc_w", 256 * 128, false);
                  ("fc_b", 256, false); ("fco", 256, true)];
        body = fc_wg_comm "gapo" "fc_w" "fc_b" "fco" 256 128 };
      { kname = "softmax"; gws = wg_size; lws = wg_size;
        params = [("fco", 256, false); ("output", 256, true)];
        body = softmax_comm "fco" "output" 256 } ] in
  stem
  @ block "l1b1" "spool"    16 16 10 10 1 false
  @ block "l1b2" "l1b1_r2o" 16 16 10 10 1 false
  @ block "l2b1" "l1b2_r2o" 16 32 10 5  2 true
  @ block "l2b2" "l2b1_r2o" 32 32 5  5  1 false
  @ block "l3b1" "l2b2_r2o" 32 64 5  3  2 true
  @ block "l3b2" "l3b1_r2o" 64 64 3  3  1 false
  @ block "l4b1" "l3b2_r2o" 64 128 3 2  2 true
  @ block "l4b2" "l4b1_r2o" 128 128 2 2 1 false
  @ tail

let write_kernel dir spec =
  let buf = Buffer.create 4096 in
  buf_add buf "// AUTO-GENERATED by comm_to_opencl.ml (Ionia Ventus V2a) -- do not edit\n";
  buf_add buf "__kernel void "; buf_add buf spec.kname; buf_add buf "(";
  let first = ref true in
  List.iter (fun (pname, _, _) ->
    if !first then first := false else buf_add buf ", ";
    buf_add buf "__global float *"; buf_add buf pname) spec.params;
  buf_add buf ") {\n";
  next_id := 0;
  emit_top buf spec.body;
  buf_add buf "}\n";
  let oc = open_out (Filename.concat dir (spec.kname ^ ".cl")) in
  Buffer.output_buffer oc buf; close_out oc

let write_manifest dir specs =
  let oc = open_out (Filename.concat dir "manifest.txt") in
  List.iter (fun spec ->
    Printf.fprintf oc "%s %d %d" spec.kname spec.gws spec.lws;
    List.iter (fun (pname, cnt, is_out) ->
      Printf.fprintf oc " %s:%s:%d" (if is_out then "out" else "in") pname cnt)
      spec.params;
    Printf.fprintf oc "\n") specs;
  close_out oc

let main () =
  let dir = if Array.length Sys.argv > 1 then Sys.argv.(1) else "." in
  let specs =
    if Array.length Sys.argv > 2 then
      (match Sys.argv.(2) with
       | "--lenet5" -> lenet5_specs
       | "--alexnet" -> alexnet_specs
       | "--resnet18" -> resnet18_specs
       | _ -> specs)
    else specs in
  write_manifest dir specs;
  List.iter (write_kernel dir) specs;
  Printf.printf "comm_to_opencl: wrote %d kernels + manifest to %s\n"
    (List.length specs) dir

let _ = main ()
