(* Streaming C-code printer for the extracted comm AST.
   Avoids building the entire output string in memory by writing
   directly to a Buffer/out_channel.

   Memory control:
   - Every `New` buffer is declared at file scope with `static`, so it is
     not allocated on the call stack.
   - A simple lifetime-aware pool reclaims buffers when their lexical scope
     ends and hands them back to later `New`s of the exact same C type.
     This keeps the static memory footprint close to the maximum live set
     instead of the sum of all temporary buffers. *)

open Minitransformer_faithful_sym

let buf_add = Buffer.add_string

let rec applyps_to_buf buf v = function
  | [] -> buf_add buf v
  | v0 :: tl -> applyps_to_buf buf (v ^ "[" ^ writeNatSym v0 ^ "]") tl

let rec applyps_path_to_buf buf p ps =
  match p with
  | Var s -> applyps_to_buf buf s ps
  | AryIdx (n, i, p') -> applyps_path_to_buf buf p' ((fin2nat n i) :: ps)

let rec rexp_to_buf buf = function
  | REvar p -> applyps_path_to_buf buf p []
  (* Integer-looking literals must print with a decimal point: the faithful
     AST builds 1/2 and 44715/1000000 as REdiv of two REconsts, and C would
     evaluate (1)/((1+1)) and (44715)/(1000000) as INTEGER division (== 0). *)
  | REconst (_, s) ->
      buf_add buf
        (if s <> "" && String.for_all (fun c -> (c >= '0' && c <= '9')) s
         then s ^ ".0" else s)
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
      buf_add buf "(pow("; rexp_to_buf buf e'; buf_add buf ", "; buf_add buf (writeNatSym n); buf_add buf "))"
  | REexp e' ->
      buf_add buf "exp("; rexp_to_buf buf e'; buf_add buf ")"
  | REln e' ->
      buf_add buf "(log("; rexp_to_buf buf e'; buf_add buf "))"
  | REsqrt e' ->
      buf_add buf "sqrt("; rexp_to_buf buf e'; buf_add buf ")"
  (* The faithful AST's causal mask is an expression-level conditional
     (route A's CausalMask2d leaf); the old AST used a comm-level If, so the
     shared printer never needed this case.  Emit a C ternary. *)
  | REif (b, e1, e2) ->
      buf_add buf "("; bexp_to_buf buf b; buf_add buf " ? ";
      rexp_to_buf buf e1; buf_add buf " : "; rexp_to_buf buf e2;
      buf_add buf ")"

and bexp_to_buf buf = function
  | BTrue -> buf_add buf "true"
  | BFalse -> buf_add buf "false"
  | BLe (e1, e2) ->
      rexp_to_buf buf e1; buf_add buf " <= "; rexp_to_buf buf e2
  | BGt (e1, e2) ->
      rexp_to_buf buf e1; buf_add buf " > "; rexp_to_buf buf e2
  | BLt (i, n) ->
      buf_add buf (writeNatSym i); buf_add buf " < "; buf_add buf (writeNatSym n)
  | BEq (i, n) ->
      buf_add buf (writeNatSym i); buf_add buf " == "; buf_add buf (writeNatSym n)
  | BLeb (i, n) ->
      buf_add buf (writeNatSym i); buf_add buf " <= "; buf_add buf (writeNatSym n)
  | BNot b ->
      buf_add buf "!("; bexp_to_buf buf b; buf_add buf ")"
  | BAnd (b1, b2) ->
      bexp_to_buf buf b1; buf_add buf " && "; bexp_to_buf buf b2

let rec exp_to_buf buf d ps e =
  match d with
  | Num -> rexp_to_buf buf (Obj.magic e)
  | Ary (n0, n, d') ->
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
  | Ary (n0, n, d') ->
      (match ps with
       | [] -> buf_add buf "/* CG_acc: array acceptor without index */"
       | v0 :: ps' -> acc_to_buf buf d' ps' ((Obj.magic a : fin -> __) v0))

let flush_threshold = max_int

(* Buffer reuse planning. *)
let buffer_key d = cG_data d ""

let comm_to_file c path =
  let oc = open_out path in
  let buf = Buffer.create 65536 in
  let decl_buf = Buffer.create 4096 in
  let flush () =
    if Buffer.length buf >= flush_threshold then
      (Buffer.output_buffer oc buf; Buffer.clear buf)
  in
  let free_pool : (string, string list) Hashtbl.t = Hashtbl.create 64 in
  let declared : (string, unit) Hashtbl.t = Hashtbl.create 128 in
  let next_id = ref 0 in
  let fresh_count = ref 0 in
  let reuse_count = ref 0 in

  let alloc_buffer d =
    let key = buffer_key d in
    try
      match Hashtbl.find free_pool key with
      | [] -> raise Not_found
      | name :: rest ->
          Hashtbl.replace free_pool key rest;
          incr reuse_count;
          (name, false)
    with Not_found ->
      let name = "v" ^ string_of_int !next_id in
      incr next_id;
      Hashtbl.add declared name ();
      incr fresh_count;
      (name, true)
  in

  let release_buffer d name =
    let key = buffer_key d in
    let rest =
      try Hashtbl.find free_pool key with Not_found -> [] in
    Hashtbl.replace free_pool key (name :: rest)
  in

  let rec comm_to_buf_stream c cnt =
    flush ();
    match c with
    | Skip -> buf_add buf "/* skip */"
    | If (b, (c1, c2)) ->
        buf_add buf "if ("; bexp_to_buf buf b; buf_add buf ")\n  { ";
        comm_to_buf_stream c1 cnt;
        buf_add buf "}\nelse\n  { ";
        comm_to_buf_stream c2 cnt;
        buf_add buf "}"
    | Seq (c1, c2) ->
        comm_to_buf_stream c1 cnt;
        buf_add buf "\n";
        comm_to_buf_stream c2 (add (cnt_var c1) cnt)
    | Assign (d, (a, e)) ->
        if acc_is_default_none d a then
          buf_add buf "/* default assignment skipped */ ;"
        else (
          acc_to_buf buf d [] a;
          buf_add buf " = ";
          exp_to_buf buf d [] e;
          buf_add buf " ;"
        )
    | New (d, p) ->
        let (vname, fresh) = alloc_buffer d in
        if fresh then (
          buf_add decl_buf "static ";
          buf_add decl_buf (cG_data d vname);
          buf_add decl_buf ";\n"
        );
        let cs = string_to_coq_string vname in
        comm_to_buf_stream (p ((vwt d cs), (vrd d cs))) (add cnt (S O));
        release_buffer d vname
    | Loopacc (d, p) ->
        (* A scoped accumulator temporary: declare it locally (per lexical
           scope) and run the body with vwt/vrd of that name. *)
        let name = "t" ^ string_of_int !next_id in
        incr next_id;
        buf_add buf (cG_data d name); buf_add buf ";\n";
        comm_to_buf_stream (p (string_to_coq_string name)) (add cnt (S O))
    | Seqfor (n, p) ->
        buf_add buf "for(int "; buf_add buf (istr cnt); buf_add buf "=0; ";
        buf_add buf (istr cnt); buf_add buf "<"; buf_add buf (writeNatSym n);
        buf_add buf "; "; buf_add buf (istr cnt); buf_add buf " += 1) \n  { ";
        let idx = (Var (istr cnt) : nat) in
        comm_to_buf_stream (p idx) (add cnt (S O));
        buf_add buf "}"
    | Parfor (n, d, a, p) ->
        buf_add buf "#pragma omp parallel for\nfor (int "; buf_add buf (istr cnt);
        buf_add buf "=0; "; buf_add buf (istr cnt); buf_add buf " < ";
        buf_add buf (writeNatSym n); buf_add buf "; "; buf_add buf (istr cnt);
        buf_add buf " += 1) \n  { ";
        let p_arg = (Var (istr cnt) : nat) in
        let a_arg = idxAcc n d a p_arg in
        comm_to_buf_stream (p p_arg a_arg) (add cnt (S O));
        buf_add buf "}"
  in
  comm_to_buf_stream c O;
  Buffer.output_buffer oc decl_buf;
  Buffer.output_buffer oc buf;
  close_out oc;
  Printf.printf "Buffers: %d fresh, %d reused (pool size %d)\n"
    !fresh_count !reuse_count (Hashtbl.length free_pool)

let main () =
  let t0 = Unix.gettimeofday () in
  comm_to_file minitransformer_faithful_comm "minitransformer_faithful_streaming_output.c";
  let t1 = Unix.gettimeofday () in
  let len = (Unix.stat "minitransformer_faithful_streaming_output.c").Unix.st_size in
  Printf.printf "Streaming: wrote %d bytes in %.2f s\n" len (t1 -. t0)

let _ = main ()
