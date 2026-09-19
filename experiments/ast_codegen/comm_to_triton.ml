(* Triton backend generator (G2).
   Phase 1: LeNet-5.  Phase 2 (2026-08-28): AlexNet + ResNet-18 — adds
   padded/strided convolution (pad_idx2d -> masked tl.load, other=0.0),
   maxpool with padding, BatchNorm (inference-mode per-channel affine),
   residual Add and global average pooling.
   Reuses the generic AST recognizers from comm_to_cuda_alexnet.ml /
   comm_to_cuda_resnet18.ml (dot-product / max-reduce / softmax /
   elementwise-relu / flatten / conv-with-pad / avg-reduce) and emits
   shape-specialised Triton Python kernels plus a forward() launch chain.
   Input: the network's sequential AST (parallelism is introduced by the
   backend: output elements map to Triton programs).  The three sym modules
   share identical extracted types (verified by diff), so the alexnet /
   resnet18 comm values are brought in through Obj.magic.
   Usage: ./comm_to_triton [lenet5|alexnet|resnet18]  ->  <net>_triton.py *)

type ocaml_bool = bool
open Lenet5_sym

let buf_add = Buffer.add_string

let rec path_base_name = function
  | Var s -> s
  | AryIdx (_, _, p') -> path_base_name p'

type map_kind = ReluMap | IdentityMap

type relu_mode = NoReLU | WithReLU

type bn_params = {
  mean : path;
  var : path;
  gamma : path;
  beta : path;
  eps : rexp;
}

type elementwise_op =
  | IdentityOp
  | ReluOp
  | AddOp of path
  | BNOp of {
      mean : path;
      var : path;
      gamma : path;
      beta : path;
      eps : rexp;
    }
  | DropoutOp of rexp

type compute_pattern =
  | DotProduct of {
      out_shape : int list;
      red_dims : int list;
      lhs : path;
      rhs : path;
      bias : path option;
      output : path;
      pad_input : string option;
      relu : relu_mode;
      bn : bn_params option;
      add : path option;
      pad_amt : int;              (* zero padding on each spatial side *)
      stride : int;               (* convolution stride *)
      in_hw : (int * int) option; (* original (pre-pad) input H x W *)
    }
  | MaxReduce of {
      out_shape : int list;
      red_dims : int list;
      input : path;
      output : path;
    }
  | MaxPoolPad of {
      out_shape : int list;
      k : int;
      stride : int;
      pad_amt : int;
      in_h : int;
      in_w : int;
      input : path;
      output : path;
    }
  | Softmax of {
      out_shape : int list;
      features : int;
      input : path;
      output : path;
    }
  | ElementwiseMap of {
      out_shape : int list;
      input : path;
      output : path;
      map_kind : map_kind;
    }
  | ElementwiseChain of {
      out_shape : int list;
      input : path;
      output : path;
      ops : (path * elementwise_op) list;
    }
  | Reshape of {
      out_shape : int list;
      in_shape : int list;
      input : path;
      output : path;
    }
  | AvgReduce of {
      out_shape : int list;
      red_dims : int list;
      input : path;
      output : path;
    }
  | GlobalAvgReduce of {
      out_shape : int list;
      in_shape : int list;
      input : path;
      output : path;
    }
  | MatMul of {
      out_shape : int list;
      red_dim : int;
      lhs : path;
      rhs : path;
      output : path;
    }
  | GlobalAvgLinear of {
      out_shape : int list;
      in_shape : int list;
      input : path;
      weight : path;
      bias : path option;
      output : path;
      relu : relu_mode;
    }

(* ---------------------------------------------------------------------- *)
(* Small AST pattern matcher: recognise linear2d_comm and emit a kernel.  *)

let rec nat_of_int n =
  if n <= 0 then O else S (nat_of_int (n - 1))

let rec nat_to_int' = function
  | O -> 0
  | S n -> 1 + nat_to_int' n
  | _ -> failwith "nat_to_int': non-concrete nat"

let path_dims p =
  let rec aux acc = function
    | Var _ -> acc
    | AryIdx (n, _, p') -> aux (nat_to_int' n :: acc) p'
  in
  aux [] p

let rec path_equal p1 p2 =
  match p1, p2 with
  | Var s1, Var s2 -> s1 = s2
  | AryIdx (n1, i1, p1'), AryIdx (n2, i2, p2') ->
      writeNatSym (fin2nat n1 i1) = writeNatSym (fin2nat n2 i2) && path_equal p1' p2'
  | _ -> false

let path_of_rexp e =
  match (Obj.magic e : rexp) with
  | REvar p -> p
  | _ -> failwith "path_of_rexp: not a variable"

let path_of_acc a =
  (Obj.magic a : path)

let rexp_is_zero e =
  match (Obj.magic e : rexp) with
  | REconst (_, "0") -> true
  | _ -> false

let rexp_is_acc acc_path e =
  match (Obj.magic e : rexp) with
  | REvar p -> path_equal p acc_path
  | _ -> false

let rec collect_seqfor_nest c =
  let rec collect dims c =
    match c with
    | Seqfor (n, p) ->
        (* descend with index 0 to discover structure *)
        collect (n :: dims) (p (nat2fin n O))
    | _ -> (List.rev dims, c)
  in
  let dims, _ = collect [] c in
  let rec apply c dims_remaining is =
    match dims_remaining, c, is with
    | n :: ns, Seqfor (_, p), i :: is -> apply (p i) ns is
    | [], _, [] -> c
    | _ -> failwith "collect_seqfor_nest: index mismatch"
  in
  Some (dims, fun is -> apply c (List.rev dims) is)

let match_new_acc acc_path c =
  match c with
  | New (Num, p_new) ->
      let acc_a = Obj.magic acc_path in
      let acc_v = Obj.magic (REvar acc_path) in
      Some (acc_a, acc_v, p_new (acc_a, acc_v))
  | Loopacc (Num, p_new) ->
      (* Rocq 9.0 extraction of the scoped accumulator: the body is
         parameterised by a symbolic name; instantiate it with the
         accumulator base name so paths match [acc_path]. *)
      let acc_a = Obj.magic acc_path in
      let acc_v = Obj.magic (REvar acc_path) in
      Some (acc_a, acc_v, p_new (string_to_coq_string (path_base_name acc_path)))
  | _ -> None

let rec find_first_assign_to target = function
  | Assign (Num, (a, e)) ->
      if path_equal (path_of_acc a) target then Some e else None
  | Seq (c1, c2) ->
      (match find_first_assign_to target c1 with
       | Some _ as r -> r
       | None -> find_first_assign_to target c2)
  | If (_, (c1, c2)) ->
      (match find_first_assign_to target c1 with
       | Some _ as r -> r
       | None -> find_first_assign_to target c2)
  | Seqfor (n, p) ->
      find_first_assign_to target (p (nat2fin n O))
  | New (Num, p_new) ->
      let dummy_a = Obj.magic target in
      let dummy_v = Obj.magic (REvar target) in
      find_first_assign_to target (p_new (dummy_a, dummy_v))
  | _ -> None

let rec find_last_assign c =
  match c with
  | Assign (Num, (a, e)) -> Some (a, e)
  | Seq (_, c2) -> find_last_assign c2
  | If (_, (c1, c2)) ->
      (match find_last_assign c2 with
       | Some _ as r -> r
       | None -> find_last_assign c1)
  | Seqfor (n, p) ->
      find_last_assign (p (nat2fin n O))
  | New (Num, p_new) ->
      let dummy_a = Obj.magic (Var "dummy") in
      let dummy_v = Obj.magic (REvar (Var "dummy")) in
      find_last_assign (p_new (dummy_a, dummy_v))
  | _ -> None

type reduce_op = AddReduceOp | MulReduceOp | MaxReduceOp | MinReduceOp

let match_reduction_update acc_path c =
  match find_first_assign_to acc_path c with
  | None -> None
  | Some rhs ->
      match (Obj.magic rhs : rexp) with
      | REadd (e1, e2) when rexp_is_acc acc_path e1 -> Some (AddReduceOp, e2)
      | REadd (e1, e2) when rexp_is_acc acc_path e2 -> Some (AddReduceOp, e1)
      | REmul (e1, e2) when rexp_is_acc acc_path e1 -> Some (MulReduceOp, e2)
      | REmul (e1, e2) when rexp_is_acc acc_path e2 -> Some (MulReduceOp, e1)
      | _ -> None

let rexp_as_path e =
  match (Obj.magic e : rexp) with
  | REvar p -> Some p
  | _ -> None


let rec seq_to_list = function
  | Seq (c1, c2) -> c1 :: seq_to_list c2
  | c -> [c]

let recognize_dot_product c =
  match collect_seqfor_nest c with
  | None -> None
  | Some (out_dims, apply) ->
      let out_shape = List.map nat_to_int' out_dims in
      let dummy_indices = List.map (fun n -> nat2fin n O) out_dims in
      let inner = apply dummy_indices in
      match match_new_acc (Var "acc") inner with
      | None -> None
      | Some (_, _, body) ->
          (match seq_to_list body with
           | [init_stmt; red_loop; final_stmt] ->
               (match find_first_assign_to (Var "acc") init_stmt with
                | Some e when rexp_is_zero e ->
                    (match match_reduction_update (Var "acc") red_loop with
                     | Some (AddReduceOp, red_e) ->
                         (match (Obj.magic red_e : rexp) with
                          | REmul (e1, e2) ->
                              (match rexp_as_path e1, rexp_as_path e2 with
                               | Some lhs, Some rhs ->
                                   let red_dims =
                                     match collect_seqfor_nest red_loop with
                                     | Some (dims, _) -> List.map nat_to_int' dims
                                     | None -> []
                                   in
                                   (match find_last_assign body with
                                    | Some (out_a, final_e) ->
                                        let output = path_of_acc out_a in
                                        (match (Obj.magic final_e : rexp) with
                                         | REadd (acc_e, bias_e)
                                           when rexp_is_acc (Var "acc") acc_e ->
                                             (match rexp_as_path bias_e with
                                              | Some bias_p ->
                                                  Some (DotProduct {
                                                    out_shape; red_dims; lhs; rhs;
                                                    bias = Some bias_p;
                                                    output;
                                                    pad_input = None;
                                                    relu = NoReLU;
                                                    bn = None;
                                                    add = None;
                                                    pad_amt = 0;
                                                    stride = 1;
                                                    in_hw = None;
                                                  })
                                              | None -> None)
                                         | _ when rexp_is_acc (Var "acc") final_e ->
                                             Some (DotProduct {
                                               out_shape; red_dims; lhs; rhs;
                                               bias = None;
                                               output;
                                               pad_input = None;
                                               relu = NoReLU;
                                               bn = None;
                                               add = None;
                                               pad_amt = 0;
                                               stride = 1;
                                               in_hw = None;
                                             })
                                         | _ -> None)
                                    | None -> None)
                               | _, _ -> None)
                          | _ -> None)
                     | _ -> None)
                | _ -> None)
           | _ -> None)

let rec match_max_update best_path c =
  match c with
  | If (BGt (input_e, acc_e),
        (Assign (Num, (a_true, e_true)), false_c)) ->
      (* The false branch is either [best = best] (mini_alexnet/LeNet form)
         or [Skip] (ResNet-18 form). *)
      let false_ok =
        match false_c with
        | Skip -> true
        | Assign (Num, (a_false, e_false)) ->
            (match rexp_as_path e_false with
             | Some false_p ->
                 path_equal (path_of_acc a_false) best_path
                 && rexp_is_acc best_path (REvar false_p)
             | None -> false)
        | _ -> false
      in
      (match rexp_as_path input_e, rexp_as_path e_true with
       | Some input_p, Some true_p ->
           if false_ok
              && rexp_is_acc best_path acc_e
              && path_equal (path_of_acc a_true) best_path
              && path_base_name input_p = path_base_name true_p then
             Some input_p
           else None
       | _ -> None)
  | Seq (c1, c2) ->
      (match match_max_update best_path c1 with
       | Some _ as r -> r
       | None -> match_max_update best_path c2)
  | If (_, (c1, c2)) ->
      (match match_max_update best_path c1 with
       | Some _ as r -> r
       | None -> match_max_update best_path c2)
  | Seqfor (n, p) ->
      match_max_update best_path (p (nat2fin n O))
  | _ -> None

let recognize_max_reduce c =
  match collect_seqfor_nest c with
  | None -> None
  | Some (out_dims, apply) ->
      let out_shape = List.map nat_to_int' out_dims in
      let dummy_indices = List.map (fun n -> nat2fin n O) out_dims in
      let inner = apply dummy_indices in
      match match_new_acc (Var "best") inner with
      | None -> None
      | Some (_, _, body) ->
          (match seq_to_list body with
           | [init_stmt; red_loop; final_stmt] ->
               (match find_first_assign_to (Var "best") init_stmt with
                | Some init_e ->
                    (match rexp_as_path init_e with
                     | Some input_p ->
                         (match match_max_update (Var "best") red_loop with
                          | None -> None
                          | Some input_p2 ->
                              if path_base_name input_p = path_base_name input_p2 then
                                let red_dims =
                                  match collect_seqfor_nest red_loop with
                                  | Some (dims, _) -> List.map nat_to_int' dims
                                  | None -> []
                                in
                                (match find_last_assign body with
                                 | Some (out_a, final_e)
                                   when rexp_is_acc (Var "best") final_e ->
                                     Some (MaxReduce {
                                       out_shape; red_dims; input = input_p;
                                       output = path_of_acc out_a;
                                     })
                                 | _ -> None)
                              else None)
                     | None -> None)
                | _ -> None)
           | _ -> None)

let recognize_softmax c =
  match collect_seqfor_nest c with
  | None -> None
  | Some (out_dims, apply) ->
      let out_shape = List.map nat_to_int' out_dims in
      let dummy_indices = List.map (fun n -> nat2fin n O) out_dims in
      let inner = apply dummy_indices in
      match match_new_acc (Var "sum") inner with
      | None -> None
      | Some (_, _, body) ->
          (match seq_to_list body with
           | [init_stmt; red_loop; div_loop] ->
               (match find_first_assign_to (Var "sum") init_stmt with
                | Some e when rexp_is_zero e ->
                    (match match_reduction_update (Var "sum") red_loop with
                     | Some (AddReduceOp, red_e) ->
                         (match (Obj.magic red_e : rexp) with
                          | REexp input_e ->
                              (match rexp_as_path input_e with
                               | Some input_p ->
                                   (match collect_seqfor_nest div_loop with
                                    | Some (div_dims, div_apply) ->
                                        let div_dummy = List.map (fun n -> nat2fin n O) div_dims in
                                        let div_inner = div_apply div_dummy in
                                        (match find_last_assign div_inner with
                                         | Some (out_a, div_e) ->
                                             (match (Obj.magic div_e : rexp) with
                                              | REdiv (exp_e, acc_e)
                                                when rexp_is_acc (Var "sum") acc_e ->
                                                  (match (Obj.magic exp_e : rexp) with
                                                   | REexp input_e2 ->
                                                       (match rexp_as_path input_e2 with
                                                        | Some input_p2 ->
                                                            if path_base_name input_p = path_base_name input_p2 then
                                                              let features =
                                                                List.fold_left ( * ) 1 (List.map nat_to_int' div_dims)
                                                              in
                                                              let batch =
                                                                match out_shape with [b] -> b | _ -> failwith "softmax: bad batch"
                                                              in
                                                              Some (Softmax {
                                                                out_shape = [batch; features]; features;
                                                                input = input_p;
                                                                output = path_of_acc out_a;
                                                              })
                                                            else None
                                                        | None -> None)
                                                   | _ -> None)
                                              | _ -> None)
                                         | _ -> None)
                                    | _ -> None)
                               | None -> None)
                          | _ -> None)
                     | _ -> None)
                | _ -> None)
           | _ -> None)

let recognize_elementwise_relu c =
  match collect_seqfor_nest c with
  | None -> None
  | Some (out_dims, apply) ->
      let out_shape = List.map nat_to_int' out_dims in
      let dummy_indices = List.map (fun n -> nat2fin n O) out_dims in
      let inner = apply dummy_indices in
      match inner with
      | If (BLe (input_e, zero_e),
            (Assign (Num, (out_a_true, zero_true)),
             Assign (Num, (out_a_false, input_e2)))) ->
          if rexp_is_zero zero_e && rexp_is_zero zero_true then
            (match rexp_as_path input_e, rexp_as_path input_e2 with
             | Some input_p, Some input_p2 ->
                 if path_base_name input_p = path_base_name input_p2 then
                   let out_p = path_of_acc out_a_true in
                   if path_equal out_p (path_of_acc out_a_false) then
                     Some (ElementwiseMap {
                       out_shape; input = input_p; output = out_p; map_kind = ReluMap;
                     })
                   else None
                 else None
             | _ -> None)
          else None
      | _ -> None

let recognize_elementwise_add c =
  match collect_seqfor_nest c with
  | None -> None
  | Some (out_dims, apply) ->
      let out_shape = List.map nat_to_int' out_dims in
      let dummy_indices = List.map (fun n -> nat2fin n O) out_dims in
      let inner = apply dummy_indices in
      match inner with
      | Assign (Num, (out_a, e)) ->
          (match (Obj.magic e : rexp) with
           | REadd (e1, e2) ->
               (match rexp_as_path e1, rexp_as_path e2 with
                | Some p1, Some p2 ->
                    let out_p = path_of_acc out_a in
                    Some (ElementwiseChain {
                      out_shape; input = p1; output = out_p;
                      ops = [(out_p, AddOp p2)];
                    })
                | _ -> None)
           | _ -> None)
      | _ -> None

let match_bn_expr e =
  match (Obj.magic e : rexp) with
  | REadd (REmul (norm_e, gamma_e), beta_e) ->
      (match (Obj.magic norm_e : rexp), rexp_as_path gamma_e, rexp_as_path beta_e with
       | REdiv (x_e, sqrt_e), Some gamma_p, Some beta_p ->
           (match (Obj.magic x_e : rexp), (Obj.magic sqrt_e : rexp) with
            | REsub (input_e, mean_e), REsqrt (var_eps_e) ->
                (match rexp_as_path input_e, rexp_as_path mean_e with
                 | Some input_p, Some mean_p ->
                     (match (Obj.magic var_eps_e : rexp) with
                      | REadd (var_e, eps_e) ->
                          (match rexp_as_path var_e with
                           | Some var_p ->
                               Some (input_p, mean_p, var_p, gamma_p, beta_p, eps_e)
                           | None -> None)
                      | _ -> None)
                 | _ -> None)
            | _ -> None)
       | _ -> None)
  | _ -> None

let recognize_elementwise_bn c =
  match collect_seqfor_nest c with
  | None -> None
  | Some (out_dims, apply) ->
      let out_shape = List.map nat_to_int' out_dims in
      let dummy_indices = List.map (fun n -> nat2fin n O) out_dims in
      let inner = apply dummy_indices in
      match inner with
      | Assign (Num, (out_a, e)) ->
          (match match_bn_expr e with
           | Some (input_p, mean_p, var_p, gamma_p, beta_p, eps_e) ->
               let out_p = path_of_acc out_a in
               Some (ElementwiseChain {
                 out_shape; input = input_p; output = out_p;
                 ops = [(out_p, BNOp { mean = mean_p; var = var_p;
                                       gamma = gamma_p; beta = beta_p; eps = eps_e })];
               })
           | None -> None)
      | _ -> None

let recognize_elementwise_dropout c =
  match collect_seqfor_nest c with
  | None -> None
  | Some (out_dims, apply) ->
      let out_shape = List.map nat_to_int' out_dims in
      let dummy_indices = List.map (fun n -> nat2fin n O) out_dims in
      let inner = apply dummy_indices in
      match inner with
      | Assign (Num, (out_a, e)) ->
          (match (Obj.magic e : rexp) with
           | REmul (input_e, scale_e) ->
               (match rexp_as_path input_e with
                | Some input_p ->
                    let out_p = path_of_acc out_a in
                    Some (ElementwiseChain {
                      out_shape; input = input_p; output = out_p;
                      ops = [(out_p, DropoutOp scale_e)];
                    })
                | None -> None)
           | _ -> None)
      | _ -> None

let first_assign c =
  let rec aux = function
    | Assign (Num, (a, e)) -> Some (a, e)
    | Seq (c1, _) -> aux c1
    | _ -> None
  in
  aux c

let rec find_original_input pad_path comm =
  match comm with
  | Assign (Num, (a, e)) ->
      (match (Obj.magic e : rexp) with
       | REvar p ->
           let acc_base = path_base_name (path_of_acc a) in
           let exp_base = path_base_name p in
           let pad_base = path_base_name pad_path in
           if acc_base = pad_base && exp_base <> pad_base then
             Some exp_base
           else None
       | _ -> None)
  | Seq (c1, c2) ->
      (match find_original_input pad_path c1 with
       | Some _ as r -> r
       | None -> find_original_input pad_path c2)
  | If (_, (c1, c2)) ->
      (match find_original_input pad_path c1 with
       | Some _ as r -> r
       | None -> find_original_input pad_path c2)
  | Seqfor (n, p_for) ->
      let size = nat_to_int' n in
      let rec try_indices k =
        if k >= size then None
        else
          match find_original_input pad_path (p_for (nat2fin n (nat_of_int k))) with
          | Some _ as r -> r
          | None -> try_indices (k + 1)
      in
      try_indices 0
  | _ -> None

(* ---------------------------------------------------------------------- *)
(* Phase 2: padding / stride / BN / GAP support (ported from               *)
(* comm_to_cuda_resnet18.ml).                                              *)

let rec data_dims = function
  | Num -> []
  | Ary (n, _, d) -> nat_to_int' n :: data_dims d

let infer_stride in_h in_w k pad out_h out_w =
  let try_s s =
    (in_h + 2 * pad - k) / s + 1 = out_h && (in_w + 2 * pad - k) / s + 1 = out_w
  in
  match List.filter try_s [1; 2; 3; 4] with
  | [s] -> s
  | ss ->
      failwith (Printf.sprintf
        "infer_stride: ambiguous/no stride %s for in=%dx%d k=%d pad=%d out=%dx%d"
        (String.concat "," (List.map string_of_int ss)) in_h in_w k pad out_h out_w)

(* Same as find_original_input but returns the full path (the pad-fill loops
   read the original input; we need its dims, not just its base name). *)
let rec find_original_input_path pad_path comm =
  match comm with
  | Assign (Num, (a, e)) ->
      (match (Obj.magic e : rexp) with
       | REvar p ->
           let acc_base = path_base_name (path_of_acc a) in
           let exp_base = path_base_name p in
           let pad_base = path_base_name pad_path in
           if acc_base = pad_base && exp_base <> pad_base then
             Some p
           else None
       | _ -> None)
  | Seq (c1, c2) ->
      (match find_original_input_path pad_path c1 with
       | Some _ as r -> r
       | None -> find_original_input_path pad_path c2)
  | If (_, (c1, c2)) ->
      (match find_original_input_path pad_path c1 with
       | Some _ as r -> r
       | None -> find_original_input_path pad_path c2)
  | Seqfor (n, p_for) ->
      let size = nat_to_int' n in
      let rec try_indices k =
        if k >= size then None
        else
          match find_original_input_path pad_path (p_for (nat2fin n (nat_of_int k))) with
          | Some _ as r -> r
          | None -> try_indices (k + 1)
      in
      try_indices 0
  | _ -> None

(* Convolution component wrapped in a zero-filled pad scratch buffer
   (AlexNet conv4d_p1_k3_comm; ResNet-18 stem conv7x7/s2/p3 and all 3x3
   block convs).  The Triton kernel reads the ORIGINAL input with masked
   loads instead of materialising the padded buffer. *)
let recognize_conv_with_pad c =
  match c with
  | New (pad_data, p_new) ->
      let pad_acc = vwt pad_data (string_to_coq_string "pad") in
      let pad_exp = vrd pad_data (string_to_coq_string "pad") in
      (match p_new (pad_acc, pad_exp) with
       | Seq (padding_init, convolution_comm) ->
           (match find_original_input_path (Var "pad") padding_init with
            | Some input_p ->
                let input_var = path_base_name input_p in
                (match recognize_dot_product convolution_comm with
                 | Some (DotProduct dp) ->
                     (match dp.out_shape, path_dims input_p, data_dims pad_data with
                      | [_; _; out_h; out_w], [_; _; in_h; in_w], [_; _; pad_h; pad_w]
                        when pad_h >= in_h && pad_w >= in_w ->
                          let pad = (pad_h - in_h) / 2 in
                          (match path_dims dp.rhs with
                           | [_; _; k; _] ->
                               let stride = infer_stride in_h in_w k pad out_h out_w in
                               Some (DotProduct { dp with pad_input = Some input_var;
                                                          pad_amt = pad;
                                                          stride;
                                                          in_hw = Some (in_h, in_w) })
                           | _ -> None)
                      | _ -> None)
                 | _ -> None)
            | None -> None)
       | _ -> None)
  | _ -> None

(* Bare convolution without a padding buffer (LeNet-5 convs, ResNet-18 1x1
   projection shortcuts): pad = 0, stride inferred from input/output
   geometry.  A conv is recognised by its 4D weight tensor — the 1x1
   shortcut's reduction collapses to a single [ci] loop (the k=1 loops are
   simplified away in the AST). *)
let recognize_direct_conv c =
  match recognize_dot_product c with
  | Some (DotProduct dp) when dp.pad_input = None ->
      (match path_dims dp.lhs with
       | [_; _; in_h; in_w] | [_; in_h; in_w] ->
           (match dp.out_shape, path_dims dp.rhs with
            | ( [_; _; out_h; out_w] | [_; out_h; out_w] ),
              [_; _; k; _] ->
                let stride = infer_stride in_h in_w k 0 out_h out_w in
                (* normalise red_dims to [ci; k; k] so downstream treats
                   1x1 convs like ordinary convs *)
                let red_dims =
                  match dp.red_dims with
                  | [_; _; _] -> dp.red_dims
                  | [ci] when k = 1 -> [ci; 1; 1]
                  | _ -> dp.red_dims
                in
                if List.length red_dims <> 3 then None
                else
                  Some (DotProduct { dp with red_dims; pad_amt = 0; stride;
                                             in_hw = Some (in_h, in_w) })
            | _ -> None)
       | _ -> None)
  | _ -> None

(* Maxpool with a zero-filled pad scratch buffer (ResNet-18 stem
   maxPool4d_pad_comm, k=3 s=2 p=1).  The AST zero-fills the pad region and
   zero-initialises the accumulator, so masked loads use other=0.0. *)
let recognize_maxpool_pad c =
  match c with
  | New (pad_data, p_new) ->
      let pad_acc = vwt pad_data (string_to_coq_string "pad") in
      let pad_exp = vrd pad_data (string_to_coq_string "pad") in
      (match p_new (pad_acc, pad_exp) with
       | Seq (padding_init, pool_comm) ->
           (match find_original_input_path (Var "pad") padding_init with
            | Some input_p ->
                (match path_dims input_p, data_dims pad_data with
                 | [_; _; in_h; in_w], [_; _; pad_h; pad_w]
                   when pad_h >= in_h && pad_w >= in_w ->
                     let pad = (pad_h - in_h) / 2 in
                     (match collect_seqfor_nest pool_comm with
                      | None -> None
                      | Some (out_dims, apply) ->
                          let out_shape = List.map nat_to_int' out_dims in
                          let dummy_indices = List.map (fun n -> nat2fin n O) out_dims in
                          let inner = apply dummy_indices in
                          (match match_new_acc (Var "best") inner with
                           | None -> None
                           | Some (_, _, body) ->
                               (match seq_to_list body with
                                | [init_stmt; red_loop; final_stmt] ->
                                    (match find_first_assign_to (Var "best") init_stmt with
                                     | Some init_e when rexp_is_zero init_e ->
                                         (match match_max_update (Var "best") red_loop with
                                          | None -> None
                                          | Some _pad_read ->
                                              let red_dims =
                                                match collect_seqfor_nest red_loop with
                                                | Some (dims, _) -> List.map nat_to_int' dims
                                                | None -> []
                                              in
                                              (match red_dims, out_shape, find_last_assign body with
                                               | [k; k2], ([_; _; h_out; w_out] as out_shape), Some (out_a, final_e)
                                                 when k = k2 && rexp_is_acc (Var "best") final_e ->
                                                   let stride = infer_stride in_h in_w k pad h_out w_out in
                                                   Some (MaxPoolPad {
                                                     out_shape; k; stride;
                                                     pad_amt = pad;
                                                     in_h; in_w;
                                                     input = input_p;
                                                     output = path_of_acc out_a;
                                                   })
                                               | _ -> None))
                                     | _ -> None)
                                | _ -> None)))
                 | _ -> None)
            | None -> None)
       | _ -> None)
  | _ -> None

(* Average reduction: acc = sum(input) over red_dims, final assign divides
   by the reduction count (REdiv by constant or REdiv_n). *)
let recognize_avg_reduce c =
  match collect_seqfor_nest c with
  | None -> None
  | Some (out_dims, apply) ->
      let out_shape = List.map nat_to_int' out_dims in
      let dummy_indices = List.map (fun n -> nat2fin n O) out_dims in
      let inner = apply dummy_indices in
      match match_new_acc (Var "acc") inner with
      | None -> None
      | Some (_, _, body) ->
          (match seq_to_list body with
           | [init_stmt; red_loop; final_stmt] ->
               (match find_first_assign_to (Var "acc") init_stmt with
                | Some e when rexp_is_zero e ->
                    (match match_reduction_update (Var "acc") red_loop with
                     | Some (AddReduceOp, red_e) ->
                         (match rexp_as_path red_e with
                          | Some input_p ->
                              let red_dims =
                                match collect_seqfor_nest red_loop with
                                | Some (dims, _) -> List.map nat_to_int' dims
                                | None -> []
                              in
                              let count = List.fold_left ( * ) 1 red_dims in
                              (match find_last_assign body with
                               | Some (out_a, final_e) ->
                                   let final_is_div_by_count =
                                     match (Obj.magic final_e : rexp) with
                                     | REdiv (acc_e, count_e)
                                       when rexp_is_acc (Var "acc") acc_e ->
                                         (match (Obj.magic count_e : rexp) with
                                          | REconst (_, s) -> s = string_of_int count
                                          | _ -> false)
                                     | REdiv_n (acc_e, n) ->
                                         rexp_is_acc (Var "acc") acc_e
                                         && nat_to_int' n = count
                                     | _ -> false
                                   in
                                   if final_is_div_by_count then
                                     Some (AvgReduce {
                                       out_shape; red_dims;
                                       input = input_p;
                                       output = path_of_acc out_a;
                                     })
                                   else None
                               | None -> None)
                          | None -> None)
                     | _ -> None)
                | _ -> None)
           | _ -> None)

let recognize_global_avg_reduce c =
  match recognize_avg_reduce c with
  | Some (AvgReduce { out_shape = ([_; _] as out_shape); red_dims; input; output }) ->
      let in_dims = path_dims input in
      if List.length in_dims = 4 then
        let h = List.nth in_dims 2 in
        let w = List.nth in_dims 3 in
        if h * w = List.fold_left ( * ) 1 red_dims then
          Some (GlobalAvgReduce { out_shape; in_shape = in_dims; input; output })
        else None
      else None
  | _ -> None

let recognize_flatten c =
  (* AlexNet's flatten is a 4-deep seqfor nest (batch x channel x h x w)
     whose body assigns into a 2D output via a computed flat index; the
     mini_alexnet variant is an unrolled Seq of assignments.  Descend
     through any seqfor nest with dummy indices before looking for the
     first assignment. *)
  let body =
    match collect_seqfor_nest c with
    | Some (dims, apply) ->
        let dummy_indices = List.map (fun n -> nat2fin n O) dims in
        apply dummy_indices
    | None -> c
  in
  match first_assign body with
  | Some (out_a, in_e) ->
      (match rexp_as_path in_e with
       | Some in_p ->
           let out_p = path_of_acc out_a in
           let out_dims = path_dims out_p in
           let in_dims = path_dims in_p in
           if List.length out_dims = 2 && List.length in_dims = 4 then
             let batch = List.nth out_dims 0 in
             let flat_len = List.nth out_dims 1 in
             let channels = List.nth in_dims 1 in
             let h = List.nth in_dims 2 in
             let w = List.nth in_dims 3 in
             if flat_len = channels * h * w then
               Some (Reshape {
                 out_shape = [batch; flat_len];
                 in_shape = in_dims;
                 input = in_p;
                 output = out_p;
               })
             else None
           else None
       | None -> None)
  | _ -> None


(* ---------------------------------------------------------------------- *)
(* Triton backend: Python kernel emission + forward launch chain.          *)

let pattern_output = function
  | DotProduct { output; _ } -> output
  | MaxReduce { output; _ } -> output
  | MaxPoolPad { output; _ } -> output
  | Softmax { output; _ } -> output
  | ElementwiseMap { output; _ } -> output
  | ElementwiseChain { output; _ } -> output
  | Reshape { output; _ } -> output
  | GlobalAvgReduce { output; _ } -> output
  | _ -> failwith "pattern_output: unsupported pattern"

let pattern_kind = function
  | DotProduct { red_dims = [_; _; _]; pad_input = Some _; _ } -> "conv+pad"
  | DotProduct { red_dims = [_; _; _]; stride; _ } ->
      if stride > 1 then "conv+s" else "conv"
  | DotProduct _ -> "linear"
  | MaxReduce _ -> "maxpool"
  | MaxPoolPad _ -> "maxpool+pad"
  | Softmax _ -> "softmax"
  | ElementwiseMap _ -> "relu"
  | ElementwiseChain { ops; _ } ->
      (match ops with
       | [(_, BNOp _)] -> "bn"
       | [(_, AddOp _)] -> "add"
       | _ -> "chain")
  | Reshape _ -> "reshape"
  | GlobalAvgReduce _ -> "gap"
  | _ -> "other"

let shape_numel dims = List.fold_left ( * ) 1 dims

(* Decode a flat output offset into per-dimension index vars i0..i{nd-1}. *)
let py_decode dims =
  let nd = List.length dims in
  let b = Buffer.create 256 in
  if nd = 1 then buf_add b "    i0 = offs\n"
  else begin
    buf_add b (Printf.sprintf "    i%d = offs %% %d\n" (nd - 1) (List.nth dims (nd - 1)));
    if nd = 2 then
      buf_add b (Printf.sprintf "    i0 = offs // %d\n" (List.nth dims 1))
    else begin
      buf_add b (Printf.sprintf "    t%d = offs // %d\n" (nd - 2) (List.nth dims (nd - 1)));
      for k = nd - 2 downto 1 do
        buf_add b (Printf.sprintf "    i%d = t%d %% %d\n" k k (List.nth dims k));
        if k = 1 then
          buf_add b (Printf.sprintf "    i0 = t1 // %d\n" (List.nth dims 1))
        else
          buf_add b (Printf.sprintf "    t%d = t%d // %d\n" (k - 1) k (List.nth dims k))
      done
    end
  end;
  Buffer.contents b

(* Row-major flattened offset expression from per-dim index expressions. *)
let rec py_offset dims ivars =
  match dims, ivars with
  | [_], [v] -> v
  | _ :: ds, v :: vs ->
      "(" ^ v ^ " * " ^ string_of_int (shape_numel ds) ^ " + " ^ py_offset ds vs ^ ")"
  | _ -> "0"

let py_load ptr off =
  "tl.load(" ^ ptr ^ " + " ^ off ^ ", mask=mask, other=0.0)"

type triton_kernel = {
  tk_name : string;
  tk_src : string;
  tk_args : string list;   (* input tensor argument names, in order *)
  tk_out : string;
  tk_out_shape : int list;
  tk_block : int;
  tk_grid1 : ocaml_bool;         (* true: a single program covers the whole tensor *)
}

type gen_item =
  | GenKernel of triton_kernel
  | GenReshape of string * string * int  (* out, in, numel *)

let py_kernel_header b name args =
  buf_add b ("@triton.jit\ndef " ^ name ^ "(");
  buf_add b (String.concat ", " (List.map (fun a -> a ^ "_ptr") args));
  buf_add b ", BLOCK: tl.constexpr):\n"

let emit_conv_triton name out_shape lhs_dims rhs_dims x w b_name o ~pad ~stride =
  let nd = List.length out_shape in
  let co_v = "i" ^ string_of_int (nd - 3)
  and ho_v = "i" ^ string_of_int (nd - 2)
  and wo_v = "i" ^ string_of_int (nd - 1) in
  let ci, h, wdt =
    match lhs_dims with
    | [ci; h; wdt] -> (ci, h, wdt)
    | [_; ci; h; wdt] -> (ci, h, wdt)
    | _ -> failwith ("emit_conv_triton: bad input dims for " ^ name)
  in
  let k =
    match rhs_dims with
    | [_; _; k; _] -> k
    | _ -> failwith ("emit_conv_triton: bad weight dims for " ^ name)
  in
  let ho, wo =
    match List.rev out_shape with
    | wo :: ho :: _ -> (ho, wo)
    | _ -> failwith ("emit_conv_triton: bad output dims for " ^ name)
  in
  if ho <> (h + 2 * pad - k) / stride + 1 || wo <> (wdt + 2 * pad - k) / stride + 1 then
    failwith (Printf.sprintf
      "emit_conv_triton: geometry mismatch for %s (in=%dx%d k=%d pad=%d s=%d out=%dx%d)"
      name h wdt k pad stride ho wo);
  (* Index expressions for the source position; keep the phase-1 text for
     the pad=0/stride=1 case so lenet5_triton.py stays byte-identical. *)
  let idx_str v dk =
    let base =
      if stride = 1 then v else "(" ^ v ^ " * " ^ string_of_int stride ^ ")"
    in
    let s = base ^ " + " ^ dk in
    if pad = 0 then "(" ^ s ^ ")"
    else "(" ^ s ^ " - " ^ string_of_int pad ^ ")"
  in
  let b = Buffer.create 4096 in
  py_kernel_header b name ((match b_name with Some _ -> ["x"; "w"; "b"] | None -> ["x"; "w"]) @ ["out"]);
  buf_add b "    pid = tl.program_id(0)\n";
  buf_add b "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n";
  buf_add b (Printf.sprintf "    mask = offs < %d\n" (shape_numel out_shape));
  buf_add b (py_decode out_shape);
  buf_add b "    acc = tl.zeros([BLOCK], dtype=tl.float32)\n";
  buf_add b (Printf.sprintf "    for ci in range(%d):\n" ci);
  buf_add b (Printf.sprintf "        for di in range(%d):\n" k);
  buf_add b (Printf.sprintf "            for dj in range(%d):\n" k);
  let ih = idx_str ho_v "di" and iw = idx_str wo_v "dj" in
  let xvars =
    (if nd = 4 then ["i0"] else [])
    @ ["ci"; ih; iw]
  in
  let wvars = [co_v; "ci"; "di"; "dj"] in
  if pad > 0 then begin
    (* pad_idx2d -> masked load: out-of-range reads return 0.0 *)
    buf_add b ("                ih = " ^ ih ^ "\n");
    buf_add b ("                iw = " ^ iw ^ "\n");
    buf_add b (Printf.sprintf
      "                xm = mask & (ih >= 0) & (ih < %d) & (iw >= 0) & (iw < %d)\n" h wdt);
    buf_add b ("                xv = tl.load(x_ptr + "
      ^ py_offset lhs_dims ((if nd = 4 then ["i0"] else []) @ ["ci"; "ih"; "iw"])
      ^ ", mask=xm, other=0.0)\n")
  end else
    buf_add b ("                xv = " ^ py_load "x_ptr" (py_offset lhs_dims xvars) ^ "\n");
  buf_add b ("                wv = " ^ py_load "w_ptr" (py_offset rhs_dims wvars) ^ "\n");
  buf_add b "                acc += xv * wv\n";
  (match b_name with
   | Some _ -> buf_add b ("    acc += " ^ py_load "b_ptr" co_v ^ "\n")
   | None -> ());
  buf_add b "    tl.store(out_ptr + offs, acc, mask=mask)\n";
  GenKernel { tk_name = name; tk_src = Buffer.contents b;
              tk_args = (match b_name with Some bn -> [x; w; bn] | None -> [x; w]);
              tk_out = o; tk_out_shape = out_shape; tk_block = 128;
              tk_grid1 = false }

let emit_linear_triton name out_shape lhs_dims rhs_dims in_features x w b_name o =
  let nd = List.length out_shape in
  let f_v = "i" ^ string_of_int (nd - 1) in
  let b = Buffer.create 2048 in
  py_kernel_header b name ((match b_name with Some _ -> ["x"; "w"; "b"] | None -> ["x"; "w"]) @ ["out"]);
  buf_add b "    pid = tl.program_id(0)\n";
  buf_add b "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n";
  buf_add b (Printf.sprintf "    mask = offs < %d\n" (shape_numel out_shape));
  buf_add b (py_decode out_shape);
  buf_add b "    acc = tl.zeros([BLOCK], dtype=tl.float32)\n";
  buf_add b (Printf.sprintf "    for k in range(%d):\n" in_features);
  let xoff =
    match lhs_dims with
    | [_] -> "k"
    | [_; _] -> py_offset lhs_dims ["i0"; "k"]
    | _ -> failwith ("emit_linear_triton: bad input dims for " ^ name)
  in
  (match lhs_dims with
   | [_] ->
       (* scalar load, always in bounds; broadcasts against the w vector *)
       buf_add b ("        xv = tl.load(x_ptr + " ^ xoff ^ ")\n")
   | _ ->
       buf_add b ("        xv = " ^ py_load "x_ptr" xoff ^ "\n"));
  buf_add b ("        wv = " ^ py_load "w_ptr" (py_offset rhs_dims [f_v; "k"]) ^ "\n");
  buf_add b "        acc += xv * wv\n";
  (match b_name with
   | Some _ -> buf_add b ("    acc += " ^ py_load "b_ptr" f_v ^ "\n")
   | None -> ());
  buf_add b "    tl.store(out_ptr + offs, acc, mask=mask)\n";
  GenKernel { tk_name = name; tk_src = Buffer.contents b;
              tk_args = (match b_name with Some bn -> [x; w; bn] | None -> [x; w]);
              tk_out = o; tk_out_shape = out_shape; tk_block = 128;
              tk_grid1 = false }

let emit_relu_triton name out_shape x o =
  let total = shape_numel out_shape in
  let b = Buffer.create 512 in
  py_kernel_header b name ["x"; "out"];
  buf_add b "    pid = tl.program_id(0)\n";
  buf_add b "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n";
  buf_add b (Printf.sprintf "    mask = offs < %d\n" total);
  buf_add b "    xv = tl.load(x_ptr + offs, mask=mask, other=0.0)\n";
  buf_add b "    tl.store(out_ptr + offs, tl.maximum(xv, 0.0), mask=mask)\n";
  GenKernel { tk_name = name; tk_src = Buffer.contents b; tk_args = [x];
              tk_out = o; tk_out_shape = out_shape; tk_block = 256;
              tk_grid1 = false }

let emit_maxpool_triton name out_shape in_dims x o =
  let nd = List.length out_shape in
  let c_v = "i" ^ string_of_int (nd - 3)
  and ho_v = "i" ^ string_of_int (nd - 2)
  and wo_v = "i" ^ string_of_int (nd - 1) in
  let h, wdt =
    match List.rev in_dims with
    | wdt :: h :: _ -> (h, wdt)
    | _ -> failwith ("emit_maxpool_triton: bad input dims for " ^ name)
  in
  let ho, wo =
    match List.rev out_shape with
    | wo :: ho :: _ -> (ho, wo)
    | _ -> failwith ("emit_maxpool_triton: bad output dims for " ^ name)
  in
  if ho <= 0 || wo <= 0 || h mod ho <> 0 || wdt mod wo <> 0
     || h / ho <> wdt / wo then
    failwith ("emit_maxpool_triton: bad pooling dims for " ^ name);
  let k = h / ho in
  if k <> 2 then
    failwith ("emit_maxpool_triton: only 2x2 stride-2 pooling supported, got k="
              ^ string_of_int k ^ " for " ^ name);
  let base = (if nd = 4 then ["i0"] else []) @ [c_v] in
  let off plus_h plus_w =
    py_offset in_dims
      (base @ [ "(" ^ string_of_int k ^ " * " ^ ho_v ^ (if plus_h then " + 1" else "") ^ ")";
                "(" ^ string_of_int k ^ " * " ^ wo_v ^ (if plus_w then " + 1" else "") ^ ")" ])
  in
  let b = Buffer.create 1024 in
  py_kernel_header b name ["x"; "out"];
  buf_add b "    pid = tl.program_id(0)\n";
  buf_add b "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n";
  buf_add b (Printf.sprintf "    mask = offs < %d\n" (shape_numel out_shape));
  buf_add b (py_decode out_shape);
  buf_add b ("    x00 = " ^ py_load "x_ptr" (off false false) ^ "\n");
  buf_add b ("    x01 = " ^ py_load "x_ptr" (off false true) ^ "\n");
  buf_add b ("    x10 = " ^ py_load "x_ptr" (off true false) ^ "\n");
  buf_add b ("    x11 = " ^ py_load "x_ptr" (off true true) ^ "\n");
  buf_add b "    m = tl.maximum(tl.maximum(x00, x01), tl.maximum(x10, x11))\n";
  buf_add b "    tl.store(out_ptr + offs, m, mask=mask)\n";
  GenKernel { tk_name = name; tk_src = Buffer.contents b; tk_args = [x];
              tk_out = o; tk_out_shape = out_shape; tk_block = 256;
              tk_grid1 = false }

let emit_softmax_triton name features x o =
  let rec np2 p = if p >= features then p else np2 (2 * p) in
  let block = max 2 (np2 1) in
  let b = Buffer.create 512 in
  py_kernel_header b name ["x"; "out"];
  buf_add b "    offs = tl.arange(0, BLOCK)\n";
  buf_add b (Printf.sprintf "    mask = offs < %d\n" features);
  buf_add b "    xv = tl.load(x_ptr + offs, mask=mask, other=float('-inf'))\n";
  buf_add b "    ev = tl.exp(xv)\n";
  buf_add b "    s = tl.sum(ev, axis=0)\n";
  buf_add b "    tl.store(out_ptr + offs, ev / s, mask=mask)\n";
  GenKernel { tk_name = name; tk_src = Buffer.contents b; tk_args = [x];
              tk_out = o; tk_out_shape = [features]; tk_block = block;
              tk_grid1 = true }

(* Elementwise chain kernel: BatchNorm (inference-mode per-channel affine),
   residual Add, ReLU — one kernel per recognised chain.  BN parameters are
   per-channel vectors indexed by the channel dim (i1 for 4D/2D tensors);
   eps is either an external scalar tensor (REvar) or a literal. *)
let emit_chain_triton name out_shape x o ops =
  let nd = List.length out_shape in
  let ch_v = if nd >= 2 then "i1" else "i0" in
  let args = Buffer.create 256 in
  buf_add args x;
  let add_arg s = buf_add args ("; " ^ s) in
  List.iter (fun (_, op) ->
    match op with
    | AddOp p -> add_arg (path_base_name p)
    | BNOp { mean; var; gamma; beta; eps } ->
        add_arg (path_base_name mean);
        add_arg (path_base_name var);
        add_arg (path_base_name gamma);
        add_arg (path_base_name beta);
        (match (Obj.magic eps : rexp) with
         | REvar p -> add_arg (path_base_name p)
         | _ -> ())
    | _ -> ()) ops;
  let arg_list = String.split_on_char ';' (Buffer.contents args)
                 |> List.map String.trim in
  let b = Buffer.create 2048 in
  py_kernel_header b name (arg_list @ ["out"]);
  buf_add b "    pid = tl.program_id(0)\n";
  buf_add b "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n";
  buf_add b (Printf.sprintf "    mask = offs < %d\n" (shape_numel out_shape));
  buf_add b (py_decode out_shape);
  buf_add b ("    xv = tl.load(" ^ x ^ "_ptr + offs, mask=mask, other=0.0)\n");
  List.iter (fun (_, op) ->
    match op with
    | AddOp p ->
        buf_add b ("    yv = tl.load(" ^ path_base_name p
          ^ "_ptr + offs, mask=mask, other=0.0)\n");
        buf_add b "    xv = xv + yv\n"
    | ReluOp ->
        buf_add b "    xv = tl.maximum(xv, 0.0)\n"
    | BNOp { mean; var; gamma; beta; eps } ->
        let ld p = "tl.load(" ^ path_base_name p ^ "_ptr + " ^ ch_v
                   ^ ", mask=mask, other=0.0)" in
        let ep =
          match (Obj.magic eps : rexp) with
          | REconst (_, s) -> s
          | REvar p -> "tl.load(" ^ path_base_name p ^ "_ptr)"
          | _ -> failwith ("emit_chain_triton: bad eps for " ^ name)
        in
        buf_add b ("    mv = " ^ ld mean ^ "\n");
        buf_add b ("    vv = " ^ ld var ^ "\n");
        buf_add b ("    gv = " ^ ld gamma ^ "\n");
        buf_add b ("    bv = " ^ ld beta ^ "\n");
        buf_add b ("    xv = (xv - mv) / tl.sqrt(vv + (" ^ ep ^ ")) * gv + bv\n")
    | IdentityOp -> ()
    | DropoutOp _ -> failwith ("emit_chain_triton: dropout not supported for " ^ name)) ops;
  buf_add b "    tl.store(out_ptr + offs, xv, mask=mask)\n";
  GenKernel { tk_name = name; tk_src = Buffer.contents b;
              tk_args = arg_list;
              tk_out = o; tk_out_shape = out_shape; tk_block = 256;
              tk_grid1 = false }

(* Maxpool over a zero-padded region (AST semantics: pad cells and the
   accumulator init are 0, so masked loads use other=0.0). *)
let emit_maxpool_pad_triton name out_shape k stride pad in_h in_w x o =
  let nd = List.length out_shape in
  let c_v = "i" ^ string_of_int (nd - 3)
  and ho_v = "i" ^ string_of_int (nd - 2)
  and wo_v = "i" ^ string_of_int (nd - 1) in
  let b = Buffer.create 1024 in
  py_kernel_header b name ["x"; "out"];
  buf_add b "    pid = tl.program_id(0)\n";
  buf_add b "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n";
  buf_add b (Printf.sprintf "    mask = offs < %d\n" (shape_numel out_shape));
  buf_add b (py_decode out_shape);
  buf_add b "    acc = tl.zeros([BLOCK], dtype=tl.float32)\n";
  buf_add b (Printf.sprintf "    for di in range(%d):\n" k);
  buf_add b (Printf.sprintf "        for dj in range(%d):\n" k);
  let in_dims =
    match out_shape with
    | [b; c; _; _] -> [b; c; in_h; in_w]
    | [c; _; _] -> [c; in_h; in_w]
    | _ -> failwith ("emit_maxpool_pad_triton: bad output dims for " ^ name)
  in
  let base = (if nd = 4 then ["i0"] else []) @ [c_v] in
  buf_add b (Printf.sprintf
    "            ih = %s * %d + di - %d\n" ho_v stride pad);
  buf_add b (Printf.sprintf
    "            iw = %s * %d + dj - %d\n" wo_v stride pad);
  buf_add b (Printf.sprintf
    "            xm = mask & (ih >= 0) & (ih < %d) & (iw >= 0) & (iw < %d)\n" in_h in_w);
  buf_add b ("            v = tl.load(x_ptr + "
    ^ py_offset in_dims (base @ ["ih"; "iw"])
    ^ ", mask=xm, other=0.0)\n");
  buf_add b "            acc = tl.maximum(acc, v)\n";
  buf_add b "    tl.store(out_ptr + offs, acc, mask=mask)\n";
  GenKernel { tk_name = name; tk_src = Buffer.contents b; tk_args = [x];
              tk_out = o; tk_out_shape = out_shape; tk_block = 256;
              tk_grid1 = false }

(* Global average pooling: out[b,c] = mean over the HxW plane. *)
let emit_global_avg_triton name out_shape in_shape x o =
  let h, w =
    match List.rev in_shape with
    | w :: h :: _ -> (h, w)
    | _ -> failwith ("emit_global_avg_triton: bad input dims for " ^ name)
  in
  let b = Buffer.create 1024 in
  py_kernel_header b name ["x"; "out"];
  buf_add b "    pid = tl.program_id(0)\n";
  buf_add b "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n";
  buf_add b (Printf.sprintf "    mask = offs < %d\n" (shape_numel out_shape));
  buf_add b (py_decode out_shape);
  buf_add b "    acc = tl.zeros([BLOCK], dtype=tl.float32)\n";
  buf_add b (Printf.sprintf "    for i in range(%d):\n" h);
  buf_add b (Printf.sprintf "        for j in range(%d):\n" w);
  buf_add b ("            acc += " ^ py_load "x_ptr"
    (py_offset in_shape ["i0"; "i1"; "i"; "j"]) ^ "\n");
  buf_add b (Printf.sprintf "    acc = acc / %d.0\n" (h * w));
  buf_add b "    tl.store(out_ptr + offs, acc, mask=mask)\n";
  GenKernel { tk_name = name; tk_src = Buffer.contents b; tk_args = [x];
              tk_out = o; tk_out_shape = out_shape; tk_block = 128;
              tk_grid1 = false }

(* softmax1d_comm has no outer seqfor nest (the batch dim is absent in the
   LeNet-5 AST), so the generic recognize_softmax does not apply; this is
   the same matching logic specialised to a bare Loopacc body. *)
let recognize_softmax1d c =
  match match_new_acc (Var "acc") c with
  | Some (_, _, body) ->
      (match seq_to_list body with
       | [init_stmt; red_loop; div_loop] ->
           (match find_first_assign_to (Var "acc") init_stmt with
            | Some e when rexp_is_zero e ->
                (match match_reduction_update (Var "acc") red_loop with
                 | Some (AddReduceOp, red_e) ->
                     (match (Obj.magic red_e : rexp) with
                      | REexp input_e ->
                          (match rexp_as_path input_e with
                           | Some input_p ->
                               (match collect_seqfor_nest div_loop with
                                | Some (div_dims, div_apply) ->
                                    let dummy = List.map (fun n -> nat2fin n O) div_dims in
                                    (match find_last_assign (div_apply dummy) with
                                     | Some (out_a, div_e) ->
                                         (match (Obj.magic div_e : rexp) with
                                          | REdiv (exp_e, acc_e)
                                            when rexp_is_acc (Var "acc") acc_e ->
                                              (match (Obj.magic exp_e : rexp) with
                                               | REexp input_e2 ->
                                                   (match rexp_as_path input_e2 with
                                                    | Some input_p2 ->
                                                        if path_base_name input_p = path_base_name input_p2 then
                                                          let features =
                                                            shape_numel (List.map nat_to_int' div_dims) in
                                                          Some (Softmax {
                                                            out_shape = [features]; features;
                                                            input = input_p;
                                                            output = path_of_acc out_a;
                                                          })
                                                        else None
                                                    | None -> None)
                                               | _ -> None)
                                          | _ -> None)
                                     | None -> None)
                                | None -> None)
                           | None -> None)
                      | _ -> None)
                 | _ -> None)
            | _ -> None)
       | _ -> None)
  | None -> None

(* Generic reshape (e.g. flatten): a seqfor nest whose body copies input
   elements into an output buffer of equal total size. *)
let recognize_reshape_generic c =
  let body =
    match collect_seqfor_nest c with
    | Some (dims, apply) ->
        let dummy = List.map (fun n -> nat2fin n O) dims in
        apply dummy
    | None -> c
  in
  match first_assign body with
  | Some (out_a, in_e) ->
      (match rexp_as_path in_e with
       | Some in_p ->
           let out_p = path_of_acc out_a in
           let in_dims = path_dims in_p and out_dims = path_dims out_p in
           if in_dims <> [] && out_dims <> []
              && shape_numel in_dims = shape_numel out_dims then
             Some (Reshape {
               out_shape = out_dims; in_shape = in_dims;
               input = in_p; output = out_p;
             })
           else None
       | None -> None)
  | None -> None

let try_safely f c = try f c with _ -> None

(* Network-general matcher chain: pad-wrapped conv/maxpool first (the pad
   scratch buffer must stay atomic), then bare dot-products (direct conv or
   linear), reductions, elementwise chains, reshape, GAP. *)
let try_match_any c =
  match try_safely recognize_conv_with_pad c with
  | Some _ as r -> r
  | None ->
  match try_safely recognize_maxpool_pad c with
  | Some _ as r -> r
  | None ->
  match try_safely recognize_direct_conv c with
  | Some _ as r -> r
  | None ->
  match try_safely recognize_dot_product c with
  | Some (DotProduct { pad_input = None; _ } as pat) -> Some pat
  | _ ->
  match try_safely recognize_max_reduce c with
  | Some pat -> Some pat
  | _ ->
  match try_safely recognize_softmax c with
  | Some pat -> Some pat
  | _ ->
  match try_safely recognize_softmax1d c with
  | Some pat -> Some pat
  | _ ->
  match try_safely recognize_elementwise_relu c with
  | Some pat -> Some pat
  | _ ->
  match try_safely recognize_elementwise_bn c with
  | Some pat -> Some pat
  | _ ->
  match try_safely recognize_elementwise_add c with
  | Some pat -> Some pat
  | _ ->
  match try_safely recognize_global_avg_reduce c with
  | Some pat -> Some pat
  | _ ->
  match try_safely recognize_flatten c with
  | Some pat -> Some pat
  | _ -> try_safely recognize_reshape_generic c

let gen_of_pattern i = function
  | DotProduct { out_shape; red_dims; lhs; rhs; bias; output;
                 pad_input; pad_amt; stride; in_hw; _ } ->
      let o = path_base_name output in
      let name = "k" ^ string_of_int i ^ "_" ^ o in
      let b_name = match bias with Some p -> Some (path_base_name p) | None -> None in
      (match red_dims with
       | [_; _; _] ->
           (* For a padded conv the kernel reads the ORIGINAL input with
              masked loads; its dims come from in_hw (pad scratch dims would
              otherwise be baked in). *)
           let x, x_dims =
             match pad_input, in_hw with
             | Some v, Some (ih, iw) ->
                 (match path_dims lhs with
                  | [b; ci; _; _] -> (v, [b; ci; ih; iw])
                  | [ci; _; _] -> (v, [ci; ih; iw])
                  | _ -> failwith ("gen_of_pattern: bad conv lhs dims for " ^ name))
             | _ -> (path_base_name lhs, path_dims lhs)
           in
           emit_conv_triton name out_shape x_dims (path_dims rhs)
             x (path_base_name rhs) b_name o ~pad:pad_amt ~stride
       | [in_features] ->
           emit_linear_triton name out_shape (path_dims lhs) (path_dims rhs)
             in_features (path_base_name lhs) (path_base_name rhs) b_name o
       | _ -> failwith ("gen_of_pattern: bad reduction dims for " ^ name))
  | MaxReduce { out_shape; input; output; _ } ->
      let o = path_base_name output in
      emit_maxpool_triton ("k" ^ string_of_int i ^ "_" ^ o) out_shape
        (path_dims input) (path_base_name input) o
  | MaxPoolPad { out_shape; k; stride; pad_amt; in_h; in_w; input; output } ->
      let o = path_base_name output in
      emit_maxpool_pad_triton ("k" ^ string_of_int i ^ "_" ^ o) out_shape
        k stride pad_amt in_h in_w (path_base_name input) o
  | Softmax { features; input; output; _ } ->
      let o = path_base_name output in
      emit_softmax_triton ("k" ^ string_of_int i ^ "_" ^ o) features
        (path_base_name input) o
  | ElementwiseMap { out_shape; input; output; map_kind = ReluMap } ->
      let o = path_base_name output in
      emit_relu_triton ("k" ^ string_of_int i ^ "_" ^ o) out_shape
        (path_base_name input) o
  | ElementwiseChain { out_shape; input; output; ops } ->
      let o = path_base_name output in
      emit_chain_triton ("k" ^ string_of_int i ^ "_" ^ o) out_shape
        (path_base_name input) o ops
  | GlobalAvgReduce { out_shape; in_shape; input; output } ->
      let o = path_base_name output in
      emit_global_avg_triton ("k" ^ string_of_int i ^ "_" ^ o) out_shape
        in_shape (path_base_name input) o
  | Reshape { input; output; in_shape; _ } ->
      GenReshape (path_base_name output, path_base_name input, shape_numel in_shape)
  | _ -> failwith "gen_of_pattern: unsupported pattern"

let py_header net =
  "# Auto-generated by comm_to_triton (OCaml) from the Coq-extracted "
  ^ net ^ " AST\n\
   # Coq-extracted AST -> pattern recognition -> Triton kernels. Do not edit.\n\
   import torch\n\
   import triton\n\
   import triton.language as tl\n\n\n"

let main () =
  let net = if Array.length Sys.argv > 1 then Sys.argv.(1) else "lenet5" in
  (* The three sym modules are extracted from the same Coq sources and have
     identical type definitions (verified by diff); Obj.magic is used only
     to satisfy OCaml's nominal typing across modules. *)
  let comm, flatten_all =
    match net with
    | "lenet5" -> (lenet5_comm, false)
    | "alexnet" -> ((Obj.magic Alexnet_sym.alexnet_comm : comm), false)
    | "resnet18" -> ((Obj.magic Resnet18_sym.resnet18_comm : comm), true)
    | _ -> failwith "usage: comm_to_triton [lenet5|alexnet|resnet18]"
  in
  (* ResNet-18 wraps each BasicBlock in a nested Seq; flatten all Seq levels
     so every operator becomes its own component.  New-wrapped pad buffers
     stay atomic (the Seq sits under the New). *)
  let rec flatten_seqs = function
    | Seq (c1, c2) -> flatten_seqs c1 @ flatten_seqs c2
    | c -> [c]
  in
  let components =
    if flatten_all then flatten_seqs comm else seq_to_list comm
  in
  Printf.printf "%s: %d top-level components\n" net (List.length components);
  let items =
    List.mapi (fun i c ->
      match try_match_any c with
      | Some p ->
          Printf.printf "component %2d: %-11s output = %s\n" i
            (pattern_kind p) (path_base_name (pattern_output p));
          gen_of_pattern i p
      | None ->
          failwith ("could not match " ^ net ^ " component " ^ string_of_int i)
    ) components
  in
  let out_file = net ^ "_triton.py" in
  let oc = open_out out_file in
  output_string oc (py_header net);
  List.iter (function
    | GenKernel tk -> output_string oc (tk.tk_src ^ "\n\n")
    | GenReshape _ -> ()) items;
  (* forward(): buffer allocation + launch chain, in original AST order *)
  let produced = Hashtbl.create 32 in
  let externs = ref [] in
  let first_extern = ref None in
  let last_out = ref "" in
  let n_kernels = ref 0 in
  let note_input s =
    if not (Hashtbl.mem produced s) && not (List.mem s !externs) then begin
      externs := !externs @ [s];
      (match !first_extern with
       | None -> first_extern := Some s
       | Some _ -> ())
    end
  in
  let body = Buffer.create 4096 in
  List.iter (fun item ->
    match item with
    | GenKernel tk ->
        List.iter note_input tk.tk_args;
        let dev =
          match !first_extern with
          | Some s -> s ^ ".device"
          | None -> failwith "no external inputs"
        in
        let shape_str =
          match tk.tk_out_shape with
          | [d] -> string_of_int d
          | ds -> "(" ^ String.concat ", " (List.map string_of_int ds) ^ ")"
        in
        (* Skip re-allocation when the output buffer already exists: ResNet-18
           block ReLUs write in-place into the residual-add buffer. *)
        if not (Hashtbl.mem produced tk.tk_out) then
          buf_add body ("    " ^ tk.tk_out ^ " = torch.empty(" ^ shape_str
            ^ ", dtype=torch.float32, device=" ^ dev ^ ")\n");
        Hashtbl.replace produced tk.tk_out ();
        let grid =
          if tk.tk_grid1 then "(1,)"
          else "(triton.cdiv(" ^ string_of_int (shape_numel tk.tk_out_shape)
               ^ ", " ^ string_of_int tk.tk_block ^ "),)"
        in
        buf_add body ("    " ^ tk.tk_name ^ "[" ^ grid ^ "]("
          ^ String.concat ", " (tk.tk_args @ [tk.tk_out])
          ^ ", BLOCK=" ^ string_of_int tk.tk_block ^ ")\n");
        incr n_kernels;
        last_out := tk.tk_out
    | GenReshape (out, inn, numel) ->
        note_input inn;
        buf_add body ("    " ^ out ^ " = " ^ inn ^ ".reshape("
          ^ string_of_int numel ^ ")\n");
        Hashtbl.replace produced out ();
        last_out := out
  ) items;
  output_string oc ("def forward(" ^ String.concat ", " !externs ^ "):\n");
  Buffer.output_buffer oc body;
  output_string oc ("    return " ^ !last_out ^ "\n");
  close_out oc;
  Printf.printf "wrote %s: %d kernels, %d externals (%s)\n"
    out_file !n_kernels (List.length !externs) (String.concat ", " !externs)

let _ = main ()
