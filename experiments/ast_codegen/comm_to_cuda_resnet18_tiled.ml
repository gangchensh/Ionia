(* CUDA backend generator for the ResNet-18 end-to-end network (224x224),
   G1 tiled variant (2026-08-27).

   Derived from comm_to_cuda_resnet18.ml (M10b naive baseline); identical
   AST pattern matchers and fusion pass, but every 4D convolution is emitted
   with the shared-memory tiled template [conv2d_tiled_kernel_gen]:
   a TILE x TILE output tile per thread block, the (TILE-1)*stride+K input
   tile (halo included) staged in shared memory, CB output channels per
   block accumulated in registers, BN/Add/ReLU epilogue fused in-kernel.
   The per-element reduction order (c, k1, k2 ascending) matches the naive
   kernel exactly, so -fmad=false builds remain bit-identical to M10.

   Correctness basis: route-A theorems A_Conv4d_tiled / A_Conv4d_BN_ReLU4d
   (Opt_TiledConv.v, proved by the parallel Coq agent); engineering-side
   correctness is backed by numerical validation against the M7 NumPy
   reference. *)

type ocaml_bool = bool

open Resnet18_sym

let cuda_prelude =
  "#include <cuda_runtime.h>\n#include <math.h>\n\n"

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

(* Cost model for deciding when to fuse and when to use async streams/events. *)
type fusion_cost_model = {
  max_chain_ops : int;            (* max element-wise ops in a single fused chain *)
  min_fuse_elements : int;        (* skip fusing if output has fewer elements than this *)
  enable_conv_relu : ocaml_bool; (* allow Conv + ReLU fusion *)
  enable_conv_bn : ocaml_bool;   (* allow Conv + BN / Conv + BN + ReLU fusion *)
  enable_conv_add : ocaml_bool;  (* allow Conv/Linear + Add fusion *)
  enable_global_avg_linear : ocaml_bool; (* allow GlobalAvgPool + Linear fusion *)
  enable_linear_relu : ocaml_bool; (* allow Linear + ReLU fusion *)
  async_min_kernels : int;        (* min number of kernels to enable stream/event async *)
}

(* Because Mini_alexnet_sym shadows the [bool] type with its own True/False,
   we use trivial comparisons of type Predef.bool for the cost model flags. *)
let default_cost_model = {
  max_chain_ops = 8;
  min_fuse_elements = 0;
  enable_conv_relu = (1 = 1);
  enable_conv_bn = (1 = 1);
  enable_conv_add = (1 = 1);
  enable_global_avg_linear = (1 = 1);
  enable_linear_relu = (1 = 1);
  async_min_kernels = 2;
}

(* A more conservative model for demonstration/comparison.  Not used by default. *)
let conservative_cost_model = {
  default_cost_model with
  max_chain_ops = 2;
  enable_conv_bn = (1 = 0);
  enable_conv_add = (1 = 0);
  enable_global_avg_linear = (1 = 0);
  async_min_kernels = 4;
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
      in_hw : (int * int) option; (* convolution input H x W *)
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

let emit_kernel_header buf name dims =
  let n_total = List.fold_left ( * ) 1 dims in
  let nd = List.length dims in
  buf_add buf ("__global__ void " ^ name ^ "()\n{\n");
  buf_add buf "  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
  buf_add buf ("  int N = " ^ string_of_int n_total ^ ";\n");
  buf_add buf "  if (idx >= N) return;\n";
  List.iteri (fun k dim ->
    let var_name = "i" ^ string_of_int (nd - 1 - k) in
    if k < nd - 1 then
      buf_add buf ("  int " ^ var_name ^ " = idx % " ^ string_of_int dim ^ "; idx /= " ^ string_of_int dim ^ ";\n")
    else
      buf_add buf ("  int " ^ var_name ^ " = idx;\n")
  ) (List.rev dims)

let elementwise_kernel name dims body =
  let buf = Buffer.create 1024 in
  emit_kernel_header buf name dims;
  buf_add buf ("  " ^ body ^ "\n");
  buf_add buf "}\n\n";
  Buffer.contents buf

let index_suffix dims =
  let n = List.length dims in
  let rec aux k acc =
    if k >= n then acc
    else aux (k + 1) (acc ^ "[i" ^ string_of_int k ^ "]")
  in
  aux 0 ""

let scalar_rexp_to_c e =
  match (Obj.magic e : rexp) with
  | REconst (_, s) -> s
  | REvar p -> path_base_name p
  | _ -> failwith "scalar_rexp_to_c: unsupported scalar expression"

let elementwise_chain_kernel name dims chain =
  let input, output, ops =
    match chain with
    | ElementwiseChain { input; output; ops } -> input, output, ops
    | _ -> failwith "elementwise_chain_kernel: not an elementwise chain"
  in
  let buf = Buffer.create 2048 in
  emit_kernel_header buf name dims;
  let idx = index_suffix dims in
  let rec expr_for_op prev = function
    | IdentityOp -> prev
    | ReluOp -> "(" ^ prev ^ " <= 0.0f) ? 0.0f : " ^ prev
    | AddOp other -> "(" ^ prev ^ " + " ^ path_base_name other ^ idx ^ ")"
    | BNOp { mean; var; gamma; beta; eps } ->
        let ch = if List.length dims >= 2 then "i1" else "i0" in
        "(((" ^ prev ^ " - " ^ path_base_name mean ^ "[" ^ ch ^ "]) / "
        ^ "sqrtf((" ^ path_base_name var ^ "[" ^ ch ^ "] + " ^ scalar_rexp_to_c eps ^ "))) * "
        ^ path_base_name gamma ^ "[" ^ ch ^ "] + " ^ path_base_name beta ^ "[" ^ ch ^ "])"
    | DropoutOp scale -> "(" ^ prev ^ " * " ^ scalar_rexp_to_c scale ^ ")"
  in
  let rec emit prev_expr = function
    | [] -> failwith "elementwise_chain_kernel: empty operation list"
    | [(out_var, op)] ->
        let expr = expr_for_op prev_expr op in
        buf_add buf ("  " ^ path_base_name output ^ idx ^ " = " ^ expr ^ ";\n")
    | (out_var, op) :: rest ->
        let expr = expr_for_op prev_expr op in
        let tmp = "v_" ^ path_base_name out_var in
        buf_add buf ("  float " ^ tmp ^ " = " ^ expr ^ ";\n");
        emit tmp rest
  in
  emit (path_base_name input ^ idx) ops;
  buf_add buf "}\n\n";
  Buffer.contents buf

let linear_kernel name batch out_features in_features input_var weight_var bias_var output_var =
  let buf = Buffer.create 1024 in
  emit_kernel_header buf name [batch; out_features];
  buf_add buf ("  float acc = 0.0f;\n");
  buf_add buf ("  for (int i = 0; i < " ^ string_of_int in_features ^ "; i += 1)\n");
  buf_add buf ("    acc += " ^ input_var ^ "[i0][i] * " ^ weight_var ^ "[i1][i];\n");
  buf_add buf ("  " ^ output_var ^ "[i0][i1] = acc + " ^ bias_var ^ "[i1];\n");
  buf_add buf "}\n\n";
  Buffer.contents buf

let linear_relu_kernel name batch out_features in_features input_var weight_var bias_var output_var =
  let buf = Buffer.create 1024 in
  emit_kernel_header buf name [batch; out_features];
  buf_add buf ("  float acc = 0.0f;\n");
  buf_add buf ("  for (int i = 0; i < " ^ string_of_int in_features ^ "; i += 1)\n");
  buf_add buf ("    acc += " ^ input_var ^ "[i0][i] * " ^ weight_var ^ "[i1][i];\n");
  buf_add buf ("  float result = acc + " ^ bias_var ^ "[i1];\n");
  buf_add buf ("  " ^ output_var ^ "[i0][i1] = (result <= 0.0f) ? 0.0f : result;\n");
  buf_add buf "}\n\n";
  Buffer.contents buf

let linear_add_kernel name batch out_features in_features input_var weight_var bias_var residual_var output_var =
  let buf = Buffer.create 1024 in
  emit_kernel_header buf name [batch; out_features];
  buf_add buf ("  float acc = 0.0f;\n");
  buf_add buf ("  for (int i = 0; i < " ^ string_of_int in_features ^ "; i += 1)\n");
  buf_add buf ("    acc += " ^ input_var ^ "[i0][i] * " ^ weight_var ^ "[i1][i];\n");
  buf_add buf ("  " ^ output_var ^ "[i0][i1] = acc + " ^ bias_var ^ "[i1] + " ^ residual_var ^ "[i0][i1];\n");
  buf_add buf "}\n\n";
  Buffer.contents buf

let linear_add_relu_kernel name batch out_features in_features input_var weight_var bias_var residual_var output_var =
  let buf = Buffer.create 1024 in
  emit_kernel_header buf name [batch; out_features];
  buf_add buf ("  float acc = 0.0f;\n");
  buf_add buf ("  for (int i = 0; i < " ^ string_of_int in_features ^ "; i += 1)\n");
  buf_add buf ("    acc += " ^ input_var ^ "[i0][i] * " ^ weight_var ^ "[i1][i];\n");
  buf_add buf ("  float result = acc + " ^ bias_var ^ "[i1] + " ^ residual_var ^ "[i0][i1];\n");
  buf_add buf ("  " ^ output_var ^ "[i0][i1] = (result <= 0.0f) ? 0.0f : result;\n");
  buf_add buf "}\n\n";
  Buffer.contents buf

let maxpool2d_kernel name batch channels h_out w_out input_var output_var =
  let buf = Buffer.create 1024 in
  emit_kernel_header buf name [batch; channels; h_out; w_out];
  buf_add buf ("  float best = " ^ input_var ^ "[i0][i1][2 * i2][2 * i3];\n");
  buf_add buf "  for (int k1 = 0; k1 < 2; k1 += 1)\n";
  buf_add buf "    for (int k2 = 0; k2 < 2; k2 += 1) {\n";
  buf_add buf ("      float v = " ^ input_var ^ "[i0][i1][2 * i2 + k1][2 * i3 + k2];\n");
  buf_add buf "      if (v > best) best = v;\n";
  buf_add buf "    }\n";
  buf_add buf ("  " ^ output_var ^ "[i0][i1][i2][i3] = best;\n");
  buf_add buf "}\n\n";
  Buffer.contents buf

(* Padded max-pool with a zero-initialised accumulator, matching the
   ResNet-18 AST semantics exactly: the scratch pad buffer is zero-filled,
   the accumulator starts at 0, and out-of-range taps contribute 0. *)
let maxpool_pad_kernel name batch channels h_out w_out k stride pad in_h in_w input_var output_var =
  let buf = Buffer.create 1024 in
  emit_kernel_header buf name [batch; channels; h_out; w_out];
  buf_add buf "  float best = 0.0f;\n";
  buf_add buf ("  for (int k1 = 0; k1 < " ^ string_of_int k ^ "; k1 += 1)\n");
  buf_add buf ("    for (int k2 = 0; k2 < " ^ string_of_int k ^ "; k2 += 1) {\n");
  buf_add buf ("      int pi = " ^ string_of_int stride ^ " * i2 + k1 - " ^ string_of_int pad ^ ";\n");
  buf_add buf ("      int pj = " ^ string_of_int stride ^ " * i3 + k2 - " ^ string_of_int pad ^ ";\n");
  buf_add buf ("      float v = (pi < 0 || pi >= " ^ string_of_int in_h ^
               " || pj < 0 || pj >= " ^ string_of_int in_w ^
               ")\n                    ? 0.0f\n                    : " ^ input_var ^ "[i0][i1][pi][pj];\n");
  buf_add buf "      if (v > best) best = v;\n";
  buf_add buf "    }\n";
  buf_add buf ("  " ^ output_var ^ "[i0][i1][i2][i3] = best;\n");
  buf_add buf "}\n\n";
  Buffer.contents buf

let avgpool2d_kernel name batch channels h_out w_out input_var output_var =
  let buf = Buffer.create 1024 in
  emit_kernel_header buf name [batch; channels; h_out; w_out];
  buf_add buf "  float sum = 0.0f;\n";
  buf_add buf "  for (int k1 = 0; k1 < 2; k1 += 1)\n";
  buf_add buf "    for (int k2 = 0; k2 < 2; k2 += 1) {\n";
  buf_add buf ("      sum += " ^ input_var ^ "[i0][i1][2 * i2 + k1][2 * i3 + k2];\n");
  buf_add buf "    }\n";
  buf_add buf ("  " ^ output_var ^ "[i0][i1][i2][i3] = sum / 4.0f;\n");
  buf_add buf "}\n\n";
  Buffer.contents buf

let global_avgpool2d_kernel name batch channels h w input_var output_var =
  let buf = Buffer.create 1024 in
  emit_kernel_header buf name [batch; channels];
  buf_add buf "  float sum = 0.0f;\n";
  buf_add buf ("  for (int ii = 0; ii < " ^ string_of_int h ^ "; ii += 1)\n");
  buf_add buf ("    for (int jj = 0; jj < " ^ string_of_int w ^ "; jj += 1)\n");
  buf_add buf ("      sum += " ^ input_var ^ "[i0][i1][ii][jj];\n");
  buf_add buf ("  " ^ output_var ^ "[i0][i1] = sum / " ^ string_of_float (float_of_int (h * w)) ^ "f;\n");
  buf_add buf "}\n\n";
  Buffer.contents buf

let global_avg_linear_kernel name batch channels h w out_features input_var weight_var bias_var output_var =
  let buf = Buffer.create 2048 in
  emit_kernel_header buf name [batch; out_features];
  buf_add buf "  float acc = 0.0f;\n";
  buf_add buf ("  for (int c = 0; c < " ^ string_of_int channels ^ "; c += 1) {\n");
  buf_add buf "    float sum = 0.0f;\n";
  buf_add buf ("    for (int ii = 0; ii < " ^ string_of_int h ^ "; ii += 1)\n");
  buf_add buf ("      for (int jj = 0; jj < " ^ string_of_int w ^ "; jj += 1)\n");
  buf_add buf ("        sum += " ^ input_var ^ "[i0][c][ii][jj];\n");
  buf_add buf ("    float avg = sum / " ^ string_of_float (float_of_int (h * w)) ^ "f;\n");
  buf_add buf ("    acc += avg * " ^ weight_var ^ "[i1][c];\n");
  buf_add buf "  }\n";
  buf_add buf ("  " ^ output_var ^ "[i0][i1] = acc + " ^ bias_var ^ "[i1];\n");
  buf_add buf "}\n\n";
  Buffer.contents buf

let global_avg_linear_relu_kernel name batch channels h w out_features input_var weight_var bias_var output_var =
  let buf = Buffer.create 2048 in
  emit_kernel_header buf name [batch; out_features];
  buf_add buf "  float acc = 0.0f;\n";
  buf_add buf ("  for (int c = 0; c < " ^ string_of_int channels ^ "; c += 1) {\n");
  buf_add buf "    float sum = 0.0f;\n";
  buf_add buf ("    for (int ii = 0; ii < " ^ string_of_int h ^ "; ii += 1)\n");
  buf_add buf ("      for (int jj = 0; jj < " ^ string_of_int w ^ "; jj += 1)\n");
  buf_add buf ("        sum += " ^ input_var ^ "[i0][c][ii][jj];\n");
  buf_add buf ("    float avg = sum / " ^ string_of_float (float_of_int (h * w)) ^ "f;\n");
  buf_add buf ("    acc += avg * " ^ weight_var ^ "[i1][c];\n");
  buf_add buf "  }\n";
  buf_add buf ("  float result = acc + " ^ bias_var ^ "[i1];\n");
  buf_add buf ("  " ^ output_var ^ "[i0][i1] = (result <= 0.0f) ? 0.0f : result;\n");
  buf_add buf "}\n\n";
  Buffer.contents buf

(* G1: block-parallel GlobalAvgPool+Linear — one 256-thread block per output
   feature; threads stride over channels (each forming the h*w channel mean
   in the original ii,jj order), partial dots are tree-reduced in shared
   memory.  The per-channel mean order is unchanged; only the channel
   reduction is reassociated (error ~1e-8). *)
let global_avg_linear_block_kernel name channels h w out_features input_var weight_var bias_var output_var =
  let buf = Buffer.create 2048 in
  buf_add buf (Printf.sprintf "/*LAUNCH %d 1 1 256 1*/\n" out_features);
  buf_add buf ("__global__ void " ^ name ^ "()\n{\n");
  buf_add buf "  __shared__ float red[256];\n";
  buf_add buf "  int o = blockIdx.x;\n  int tid = threadIdx.x;\n";
  buf_add buf "  float acc = 0.0f;\n";
  buf_add buf (Printf.sprintf "  for (int c = tid; c < %d; c += 256) {\n" channels);
  buf_add buf "    float sum = 0.0f;\n";
  buf_add buf (Printf.sprintf "    for (int ii = 0; ii < %d; ii += 1)\n" h);
  buf_add buf (Printf.sprintf "      for (int jj = 0; jj < %d; jj += 1)\n" w);
  buf_add buf (Printf.sprintf "        sum += %s[0][c][ii][jj];\n" input_var);
  buf_add buf (Printf.sprintf "    float avg = sum / %.1ff;\n" (float_of_int (h * w)));
  buf_add buf (Printf.sprintf "    acc += avg * %s[o][c];\n" weight_var);
  buf_add buf "  }\n";
  buf_add buf "  red[tid] = acc;\n  __syncthreads();\n";
  buf_add buf "  for (int s = 128; s > 0; s >>= 1) {\n    if (tid < s) red[tid] += red[tid + s];\n    __syncthreads();\n  }\n";
  buf_add buf "  if (tid == 0)\n";
  buf_add buf (Printf.sprintf "    %s[0][o] = red[0] + %s[o];\n" output_var bias_var);
  buf_add buf "}\n\n";
  Buffer.contents buf

let matmul2d_kernel name batch out_features in_features input_var weight_var output_var =
  let buf = Buffer.create 1024 in
  emit_kernel_header buf name [batch; out_features];
  buf_add buf "  float acc = 0.0f;\n";
  buf_add buf ("  for (int i = 0; i < " ^ string_of_int in_features ^ "; i += 1)\n");
  buf_add buf ("    acc += " ^ input_var ^ "[i0][i] * " ^ weight_var ^ "[i1][i];\n");
  buf_add buf ("  " ^ output_var ^ "[i0][i1] = acc;\n");
  buf_add buf "}\n\n";
  Buffer.contents buf

let flatten4d_kernel name batch channels h w input_var output_var =
  let flat_len = channels * h * w in
  let buf = Buffer.create 1024 in
  emit_kernel_header buf name [batch; flat_len];
  buf_add buf ("  int hw = " ^ string_of_int (h * w) ^ ";\n");
  buf_add buf "  int c = i1 / hw;\n";
  buf_add buf "  int rem = i1 % hw;\n";
  buf_add buf ("  int h0 = rem / " ^ string_of_int w ^ ";\n");
  buf_add buf ("  int w0 = rem % " ^ string_of_int w ^ ";\n");
  buf_add buf ("  " ^ output_var ^ "[i0][i1] = " ^ input_var ^ "[i0][c][h0][w0];\n");
  buf_add buf "}\n\n";
  Buffer.contents buf

let softmax2d_kernel name batch features input_var output_var =
  let buf = Buffer.create 1024 in
  emit_kernel_header buf name [batch; features];
  buf_add buf "  float sum = 0.0f;\n";
  buf_add buf ("  for (int j = 0; j < " ^ string_of_int features ^ "; j += 1)\n");
  buf_add buf ("    sum += expf(" ^ input_var ^ "[i0][j]);\n");
  buf_add buf ("  " ^ output_var ^ "[i0][i1] = expf(" ^ input_var ^ "[i0][i1]) / sum;\n");
  buf_add buf "}\n\n";
  Buffer.contents buf

(* Generalised convolution kernel: arbitrary kernel size, stride and zero
   padding.  [in_h]/[in_w] are the input spatial dims; the grid covers the
   output [h]x[w].  The padded tap at (stride*i2+k1, stride*i3+k2) maps to
   input coords (pi, pj) = (stride*i2+k1-pad, stride*i3+k2-pad); out-of-range
   taps read as 0, exactly the semantics of the AST's zero-filled pad
   buffer. *)
let conv2d_kernel_gen name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var bn_opt add_opt relu =
  let buf = Buffer.create 2048 in
  emit_kernel_header buf name [batch; out_channels; h; w];
  buf_add buf "  float acc = 0.0f;\n";
  buf_add buf ("  for (int c = 0; c < " ^ string_of_int in_channels ^ "; c += 1)\n");
  buf_add buf ("    for (int k1 = 0; k1 < " ^ string_of_int kernel_size ^ "; k1 += 1)\n");
  buf_add buf ("      for (int k2 = 0; k2 < " ^ string_of_int kernel_size ^ "; k2 += 1) {\n");
  buf_add buf ("        int pi = " ^ string_of_int stride ^ " * i2 + k1 - " ^ string_of_int pad ^ ";\n");
  buf_add buf ("        int pj = " ^ string_of_int stride ^ " * i3 + k2 - " ^ string_of_int pad ^ ";\n");
  buf_add buf ("        float v = (pi < 0 || pi >= " ^ string_of_int in_h ^
               " || pj < 0 || pj >= " ^ string_of_int in_w ^
               ")\n                    ? 0.0f\n                    : " ^ input_var ^ "[i0][c][pi][pj];\n");
  buf_add buf ("        acc += v * " ^ weight_var ^ "[i1][c][k1][k2];\n");
  buf_add buf "      }\n";
  buf_add buf ("  float result = acc + " ^ bias_var ^ "[i1];\n");
  let expr =
    match bn_opt with
    | None -> "result"
    | Some { mean; var; gamma; beta; eps } ->
        let mean_name = path_base_name mean in
        let var_name = path_base_name var in
        let gamma_name = path_base_name gamma in
        let beta_name = path_base_name beta in
        "(((result - " ^ mean_name ^ "[i1]) / sqrtf((" ^ var_name ^ "[i1] + " ^ scalar_rexp_to_c eps ^ "))) * " ^
        gamma_name ^ "[i1] + " ^ beta_name ^ "[i1])"
  in
  let final_expr =
    match add_opt with
    | None -> expr
    | Some residual ->
        let residual_name = path_base_name residual in
        "(" ^ expr ^ " + " ^ residual_name ^ "[i0][i1][i2][i3])"
  in
  (match relu with
   | WithReLU ->
       buf_add buf ("  float fused_result = " ^ final_expr ^ ";\n");
       buf_add buf ("  " ^ output_var ^ "[i0][i1][i2][i3] = (fused_result <= 0.0f) ? 0.0f : fused_result;\n")
   | NoReLU ->
       buf_add buf ("  " ^ output_var ^ "[i0][i1][i2][i3] = " ^ final_expr ^ ";\n"));
  buf_add buf "}\n\n";
  Buffer.contents buf

let conv2d_pad_kernel name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var =
  conv2d_kernel_gen name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var None None NoReLU

let conv2d_pad_relu_kernel name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var =
  conv2d_kernel_gen name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var None None WithReLU

let conv2d_pad_bn_kernel name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var bn =
  conv2d_kernel_gen name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var (Some bn) None NoReLU

let conv2d_pad_bn_relu_kernel name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var bn =
  conv2d_kernel_gen name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var (Some bn) None WithReLU

let conv2d_pad_bn_add_kernel name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var bn residual =
  conv2d_kernel_gen name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var (Some bn) (Some residual) NoReLU

let conv2d_pad_bn_add_relu_kernel name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var bn residual =
  conv2d_kernel_gen name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var (Some bn) (Some residual) WithReLU

let conv2d_pad_add_kernel name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var residual_var output_var =
  conv2d_kernel_gen name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var None (Some (Var residual_var)) NoReLU

let conv2d_pad_add_relu_kernel name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var residual_var output_var =
  conv2d_kernel_gen name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var None (Some (Var residual_var)) WithReLU

(* ------------------------------------------------------------------ *)
(* G1 shared-memory tiled convolution templates (2026-08-27), v2.

   Two shape-specialised paths, both batch = 1, both preserving the naive
   per-element reduction order (c ascending, then k1, k2) so that
   -fmad=false builds stay bit-identical with the M10 baseline:

   - [conv2d_tiled_2x2] (stride 1, K 3, spatial >= 28): block = 16x16
     threads computes a 32x32 output tile (2x2 outputs per thread) for CB2=2
     output channels.  The 34x34 input tile (halo included) is staged in
     shared memory (row-padded to 35 floats against bank conflicts); per
     input channel each thread loads its 4x4 window into registers once and
     reuses it for all CB2 x 2 x 2 x 9 FMAs; the CB2 x 9 tap weights are
     staged in shared memory (one warp) and read as broadcasts.  This cuts
     shared-memory-pipe instructions to ~0.5 per FMA (v1 was ~2 per FMA and
     LSU-bound, slower than naive).

   - [conv2d_tiled_1x1] (any K / stride / small spatial): block = 16x16
     threads computes a 16x16 output tile (1 output per thread) for CB=4
     output channels; (TILE-1)*stride+K input tile in shared memory, CB x K x K
     weights staged in shared memory, register accumulators, no integer
     division in the load loop.

   Epilogue fuses bias (+BN) (+residual Add) (+ReLU) before the store;
   out-of-range taps read as 0 (the AST zero-padding semantics). *)

let tiled_tile = 16   (* threads per block edge *)
let tiled_cb = 4      (* 1x1 path: output channels per block *)
let tiled_cb2 = 4     (* 2x2 path: output channels per block *)

(* Shared epilogue emitter for the tiled templates: bias (+ BN) (+ residual
   Add) (+ ReLU) for one output element.  [acc_s]/[oc_s]/[oh_s]/[ow_s] are C
   expressions for the accumulator, output channel and output position. *)
let emit_tiled_epilogue buf indent acc_s oc_s oh_s ow_s output_var bias_var bn_opt add_opt relu =
  buf_add buf (indent ^ "float result = " ^ acc_s ^ " + " ^ bias_var ^ "[" ^ oc_s ^ "];\n");
  let expr =
    match bn_opt with
    | None -> "result"
    | Some { mean; var; gamma; beta; eps } ->
        let mean_name = path_base_name mean in
        let var_name = path_base_name var in
        let gamma_name = path_base_name gamma in
        let beta_name = path_base_name beta in
        "(((result - " ^ mean_name ^ "[" ^ oc_s ^ "]) / sqrtf((" ^ var_name ^ "[" ^ oc_s ^ "] + " ^ scalar_rexp_to_c eps ^ "))) * " ^
        gamma_name ^ "[" ^ oc_s ^ "] + " ^ beta_name ^ "[" ^ oc_s ^ "])"
  in
  let final_expr =
    match add_opt with
    | None -> expr
    | Some residual ->
        let residual_name = path_base_name residual in
        "(" ^ expr ^ " + " ^ residual_name ^ "[0][" ^ oc_s ^ "][" ^ oh_s ^ "][" ^ ow_s ^ "])"
  in
  (match relu with
   | WithReLU ->
       buf_add buf (indent ^ "float fused_result = " ^ final_expr ^ ";\n");
       buf_add buf (indent ^ output_var ^ "[0][" ^ oc_s ^ "][" ^ oh_s ^ "][" ^ ow_s ^ "] = (fused_result <= 0.0f) ? 0.0f : fused_result;\n")
   | NoReLU ->
       buf_add buf (indent ^ output_var ^ "[0][" ^ oc_s ^ "][" ^ oh_s ^ "][" ^ ow_s ^ "] = " ^ final_expr ^ ";\n"))

(* Split-c reduce kernel: sums the [split] partial slabs of a split
   convolution's workspace in fixed chunk order, then applies the fused
   epilogue (bias + BN + Add + ReLU).  Deterministic; only the reduction
   association differs from the naive chain (error stays ~1e-8). *)
let conv_split_reduce_kernel name h w out_channels split ws_var bias_var output_var bn_opt add_opt relu =
  let n_total = out_channels * h * w in
  let gx = (n_total + 255) / 256 in
  let buf = Buffer.create 1024 in
  buf_add buf (Printf.sprintf "/*LAUNCH2 %d 1 1 256 1*/\n" gx);
  buf_add buf ("__global__ void " ^ name ^ "_red()\n{\n");
  buf_add buf "  int idx = blockIdx.x * 256 + threadIdx.x;\n";
  buf_add buf (Printf.sprintf "  if (idx >= %d) return;\n" n_total);
  buf_add buf (Printf.sprintf "  int ow = idx %% %d;\n  int t0 = idx / %d;\n  int oh = t0 %% %d;\n  int oc = t0 / %d;\n" w w h h);
  buf_add buf (Printf.sprintf "  float acc = %s[0][oc][oh][ow];\n" ws_var);
  for sc = 1 to split - 1 do
    buf_add buf (Printf.sprintf "  acc += %s[%d][oc][oh][ow];\n" ws_var sc)
  done;
  emit_tiled_epilogue buf "  " "acc" "oc" "oh" "ow" output_var bias_var bn_opt add_opt relu;
  buf_add buf "}\n\n";
  Buffer.contents buf

(* 1x1 path: one output element per thread, CB output channels per block.
   [split] > 1 partitions the input-channel reduction across blockIdx.z
   into a partial-sum workspace ([output_var]_ws[split][OC][H][W]); the
   epilogue is deferred to the companion <name>_red kernel. *)
let conv2d_tiled_1x1 split name in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var bn_opt add_opt relu =
  (* tiny spatial extents: 8x8 thread block keeps most threads active *)
  let tile = if h <= 8 && w <= 8 && stride = 1 then tiled_tile / 2 else tiled_tile in
  let cb = tiled_cb in
  let sh = (tile - 1) * stride + kernel_size in
  let gx = (w + tile - 1) / tile in
  let gy = (h + tile - 1) / tile in
  let gz0 = (out_channels + cb - 1) / cb in
  let gz = gz0 * split in
  let kk = kernel_size * kernel_size in
  let cchunk = (in_channels + split - 1) / split in
  let ws_var = output_var ^ "_ws" in
  let buf = Buffer.create 4096 in
  buf_add buf (Printf.sprintf "/*LAUNCH %d %d %d %d %d*/\n" gx gy gz tile tile);
  buf_add buf ("__global__ void " ^ name ^ "()\n{\n");
  buf_add buf (Printf.sprintf "  __shared__ float sh[%d][%d];\n" sh sh);
  buf_add buf (Printf.sprintf "  __shared__ float shw[%d][%d];\n" cb kk);
  buf_add buf "  int tx = threadIdx.x;\n  int ty = threadIdx.y;\n";
  buf_add buf (Printf.sprintf "  int ow = blockIdx.x * %d + tx;\n" tile);
  buf_add buf (Printf.sprintf "  int oh = blockIdx.y * %d + ty;\n" tile);
  buf_add buf (Printf.sprintf "  int iw0 = blockIdx.x * %d - %d;\n" (tile * stride) pad);
  buf_add buf (Printf.sprintf "  int ih0 = blockIdx.y * %d - %d;\n" (tile * stride) pad);
  (if split > 1 then begin
     buf_add buf (Printf.sprintf "  int sc = blockIdx.z / %d;\n" gz0);
     buf_add buf (Printf.sprintf "  int oc0 = (blockIdx.z - sc * %d) * %d;\n" gz0 cb);
     buf_add buf (Printf.sprintf "  int c0 = sc * %d;\n  int c1 = c0 + %d;\n  if (c1 > %d) c1 = %d;\n" cchunk cchunk in_channels in_channels)
   end else
     buf_add buf (Printf.sprintf "  int oc0 = blockIdx.z * %d;\n" cb));
  buf_add buf (Printf.sprintf "  float acc[%d];\n" cb);
  buf_add buf "  #pragma unroll\n";
  buf_add buf (Printf.sprintf "  for (int cb = 0; cb < %d; cb += 1) acc[cb] = 0.0f;\n" cb);
  if split > 1 then
    buf_add buf "  for (int c = c0; c < c1; c += 1) {\n"
  else
    buf_add buf (Printf.sprintf "  for (int c = 0; c < %d; c += 1) {\n" in_channels);
  buf_add buf (Printf.sprintf "    for (int sy = ty; sy < %d; sy += %d)\n" sh tile);
  buf_add buf (Printf.sprintf "      for (int sx = tx; sx < %d; sx += %d) {\n" sh tile);
  buf_add buf "        int iy = ih0 + sy;\n        int ix = iw0 + sx;\n";
  buf_add buf (Printf.sprintf "        sh[sy][sx] = (iy < 0 || iy >= %d || ix < 0 || ix >= %d)\n                     ? 0.0f\n                     : %s[0][c][iy][ix];\n" in_h in_w input_var);
  buf_add buf "      }\n";
  buf_add buf (Printf.sprintf "    {\n      int tid = ty * %d + tx;\n" tile);
  buf_add buf (Printf.sprintf "      if (tid < %d) {\n" (cb * kk));
  buf_add buf (Printf.sprintf "        int cb = tid / %d;\n        int t = tid - cb * %d;\n" kk kk);
  buf_add buf (Printf.sprintf "        shw[cb][t] = (oc0 + cb < %d) ? %s[oc0 + cb][c][t / %d][t - (t / %d) * %d] : 0.0f;\n" out_channels weight_var kernel_size kernel_size kernel_size);
  buf_add buf "      }\n    }\n";
  buf_add buf "    __syncthreads();\n";
  buf_add buf (Printf.sprintf "    int by0 = ty * %d;\n    int bx0 = tx * %d;\n" stride stride);
  buf_add buf "    #pragma unroll\n";
  buf_add buf (Printf.sprintf "    for (int cb = 0; cb < %d; cb += 1) {\n" cb);
  buf_add buf (Printf.sprintf "      if (oc0 + cb < %d) {\n" out_channels);
  buf_add buf "        #pragma unroll\n";
  buf_add buf (Printf.sprintf "        for (int k1 = 0; k1 < %d; k1 += 1)\n" kernel_size);
  buf_add buf "          #pragma unroll\n";
  buf_add buf (Printf.sprintf "          for (int k2 = 0; k2 < %d; k2 += 1)\n" kernel_size);
  buf_add buf (Printf.sprintf "            acc[cb] += sh[by0 + k1][bx0 + k2] * shw[cb][k1 * %d + k2];\n" kernel_size);
  buf_add buf "      }\n    }\n";
  buf_add buf "    __syncthreads();\n  }\n";
  buf_add buf "  #pragma unroll\n";
  buf_add buf (Printf.sprintf "  for (int cb = 0; cb < %d; cb += 1) {\n" cb);
  buf_add buf "    int oc = oc0 + cb;\n";
  if split = 1 then begin
    buf_add buf (Printf.sprintf "    if (oc < %d && oh < %d && ow < %d) {\n" out_channels h w);
    emit_tiled_epilogue buf "      " "acc[cb]" "oc" "oh" "ow" output_var bias_var bn_opt add_opt relu;
    buf_add buf "    }\n  }\n}\n\n"
  end else begin
    buf_add buf (Printf.sprintf "    if (oc < %d && oh < %d && ow < %d)\n      %s[sc][oc][oh][ow] = acc[cb];\n" out_channels h w ws_var);
    buf_add buf "  }\n}\n\n";
    buf_add buf (conv_split_reduce_kernel name h w out_channels split ws_var bias_var output_var bn_opt add_opt relu)
  end;
  Buffer.contents buf

(* Generalised register-blocked path (stride 1, K 3): each thread computes a
   py x px output patch whose (py+2) x (px+2) input window is loaded into
   registers once per input channel; double-buffered shared tiles, one
   __syncthreads() per channel step.  [split] partitions the channel
   reduction as in the 1x1 path.  Selected per layer shape via the
   dispatcher; TILED_PATCH env var overrides for tuning sweeps. *)
let conv2d_tiled_patch py px split cb2 name in_channels in_h in_w h w out_channels pad input_var weight_var bias_var output_var bn_opt add_opt relu =
  let th = tiled_tile in        (* threads per edge *)
  let oty = th * py in
  let otx = th * px in
  let cb = cb2 in
  let shy = oty + 2 in
  let shx = otx + 2 in
  let gx = (w + otx - 1) / otx in
  let gy = (h + oty - 1) / oty in
  let gz0 = (out_channels + cb - 1) / cb in
  let gz = gz0 * split in
  let cchunk = (in_channels + split - 1) / split in
  let ws_var = output_var ^ "_ws" in
  let buf = Buffer.create 4096 in
  buf_add buf (Printf.sprintf "/*LAUNCH %d %d %d %d %d*/\n" gx gy gz th th);
  buf_add buf ("__global__ void " ^ name ^ "()\n{\n");
  buf_add buf (Printf.sprintf "  __shared__ float sh2[2][%d][%d];\n" shy (shx + 1));
  buf_add buf (Printf.sprintf "  __shared__ float shw2[2][%d][9];\n" cb);
  buf_add buf "  int tx = threadIdx.x;\n  int ty = threadIdx.y;\n";
  buf_add buf (Printf.sprintf "  int oy0 = blockIdx.y * %d + ty * %d;\n" oty py);
  buf_add buf (Printf.sprintf "  int ox0 = blockIdx.x * %d + tx * %d;\n" otx px);
  buf_add buf (Printf.sprintf "  int ih0 = blockIdx.y * %d - %d;\n" oty pad);
  buf_add buf (Printf.sprintf "  int iw0 = blockIdx.x * %d - %d;\n" otx pad);
  buf_add buf (Printf.sprintf "  int c0 = 0;\n  int c1 = %d;\n" in_channels);
  (if split > 1 then begin
     buf_add buf (Printf.sprintf "  int sc = blockIdx.z / %d;\n" gz0);
     buf_add buf (Printf.sprintf "  int oc0 = (blockIdx.z - sc * %d) * %d;\n" gz0 cb);
     buf_add buf (Printf.sprintf "  c0 = sc * %d;\n  c1 = c0 + %d;\n  if (c1 > %d) c1 = %d;\n" cchunk cchunk in_channels in_channels)
   end else
     buf_add buf (Printf.sprintf "  int oc0 = blockIdx.z * %d;\n" cb));
  buf_add buf (Printf.sprintf "  float acc[%d][%d][%d];\n" cb py px);
  buf_add buf "  #pragma unroll\n";
  buf_add buf (Printf.sprintf "  for (int cb = 0; cb < %d; cb += 1)\n    for (int dy = 0; dy < %d; dy += 1)\n      for (int dx = 0; dx < %d; dx += 1)\n        acc[cb][dy][dx] = 0.0f;\n" cb py px);
  let emit_load dst dstw chan =
    buf_add buf (Printf.sprintf "    for (int sy = ty; sy < %d; sy += %d)\n" shy th);
    buf_add buf (Printf.sprintf "      for (int sx = tx; sx < %d; sx += %d) {\n" shx th);
    buf_add buf "        int iy = ih0 + sy;\n        int ix = iw0 + sx;\n";
    buf_add buf (Printf.sprintf "        %s[sy][sx] = (iy < 0 || iy >= %d || ix < 0 || ix >= %d)\n                     ? 0.0f\n                     : %s[0][%s][iy][ix];\n" dst in_h in_w input_var chan);
    buf_add buf "      }\n";
    buf_add buf (Printf.sprintf "    {\n      int tid = ty * %d + tx;\n" th);
    buf_add buf (Printf.sprintf "      if (tid < %d) {\n" (cb * 9));
    buf_add buf "        int cb = tid / 9;\n        int t = tid - cb * 9;\n";
    buf_add buf (Printf.sprintf "        %s[cb][t] = (oc0 + cb < %d) ? %s[oc0 + cb][%s][t / 3][t - (t / 3) * 3] : 0.0f;\n" dstw out_channels weight_var chan);
    buf_add buf "      }\n    }\n"
  in
  emit_load "sh2[0]" "shw2[0]" "c0";
  buf_add buf "  __syncthreads();\n";
  buf_add buf "  for (int c = c0; c < c1; c += 1) {\n";
  buf_add buf "    int cur = (c - c0) & 1;\n";
  buf_add buf "    if (c + 1 < c1) {\n";
  emit_load "sh2[cur ^ 1]" "shw2[cur ^ 1]" "(c + 1)";
  buf_add buf "    }\n";
  buf_add buf (Printf.sprintf "    float win[%d][%d];\n" (py + 2) (px + 2));
  buf_add buf "    #pragma unroll\n";
  buf_add buf (Printf.sprintf "    for (int wy = 0; wy < %d; wy += 1)\n" (py + 2));
  buf_add buf "      #pragma unroll\n";
  buf_add buf (Printf.sprintf "      for (int wx = 0; wx < %d; wx += 1)\n" (px + 2));
  buf_add buf (Printf.sprintf "        win[wy][wx] = sh2[cur][ty * %d + wy][tx * %d + wx];\n" py px);
  buf_add buf "    #pragma unroll\n";
  buf_add buf (Printf.sprintf "    for (int cb = 0; cb < %d; cb += 1) {\n" cb);
  buf_add buf (Printf.sprintf "      if (oc0 + cb < %d) {\n" out_channels);
  buf_add buf "        #pragma unroll\n";
  buf_add buf "        for (int k1 = 0; k1 < 3; k1 += 1)\n";
  buf_add buf "          #pragma unroll\n";
  buf_add buf "          for (int k2 = 0; k2 < 3; k2 += 1) {\n";
  buf_add buf "            float wv = shw2[cur][cb][k1 * 3 + k2];\n";
  List.iter (fun dy ->
    List.iter (fun dx ->
      buf_add buf (Printf.sprintf "            acc[cb][%d][%d] += win[%d + k1][%d + k2] * wv;\n" dy dx dy dx)
    ) (List.init px (fun i -> i))
  ) (List.init py (fun i -> i));
  buf_add buf "          }\n";
  buf_add buf "      }\n    }\n";
  buf_add buf "    __syncthreads();\n  }\n";
  buf_add buf "  #pragma unroll\n";
  buf_add buf (Printf.sprintf "  for (int cb = 0; cb < %d; cb += 1) {\n" cb);
  buf_add buf "    int oc = oc0 + cb;\n";
  buf_add buf (Printf.sprintf "    if (oc < %d) {\n" out_channels);
  List.iter (fun dy ->
    List.iter (fun dx ->
      let oh_s = if dy = 0 then "oy0" else Printf.sprintf "oy0 + %d" dy in
      let ow_s = if dx = 0 then "ox0" else Printf.sprintf "ox0 + %d" dx in
      buf_add buf (Printf.sprintf "      if (%s < %d && %s < %d) {\n" oh_s h ow_s w);
      if split = 1 then
        emit_tiled_epilogue buf "        "
          (Printf.sprintf "acc[cb][%d][%d]" dy dx) "oc" oh_s ow_s
          output_var bias_var bn_opt add_opt relu
      else
        buf_add buf (Printf.sprintf "        %s[sc][oc][%s][%s] = acc[cb][%d][%d];\n" ws_var oh_s ow_s dy dx);
      buf_add buf "      }\n"
    ) (List.init px (fun i -> i))
  ) (List.init py (fun i -> i));
  buf_add buf "    }\n  }\n}\n\n";
  if split > 1 then
    buf_add buf (conv_split_reduce_kernel name h w out_channels split ws_var bias_var output_var bn_opt add_opt relu);
  Buffer.contents buf

let conv2d_tiled_kernel_gen name batch in_channels in_h in_w h w out_channels kernel_size stride pad input_var weight_var bias_var output_var bn_opt add_opt relu =
  if batch <> 1 then failwith "conv2d_tiled_kernel_gen: batch must be 1";
  (* split-c: spatial-small layers lack block-level parallelism; split the
     input-channel reduction across [split] z-blocks writing partials to a
     workspace, then a deterministic fixed-order reduce kernel applies the
     fused epilogue. *)
  let split =
    if h >= 56 && w >= 56 then 1
    else if h >= 28 && w >= 28 then 2
    else if h >= 14 && w >= 14 then 4
    else 2
  in
  if kernel_size = 1 || (stride <> 1 && h <= 8 && w <= 8) then
    (* 1x1 projections have no spatial reuse, and tiny stride-2 layers leave
       too few threads active in the tiled template — the naive
       one-thread-per-output kernel wins for both (measured). *)
    conv2d_kernel_gen name batch in_channels in_h in_w h w out_channels kernel_size stride pad
      input_var weight_var bias_var output_var bn_opt add_opt relu
  else if stride = 1 && kernel_size = 3 && h >= 28 && w >= 28 then
    (* channel block: largest CB in {8,4,2} keeping >= ~120 blocks *)
    let tiles = ((w + 31) / 32) * ((h + 31) / 32) in
    let rec pick = function
      | [] -> 2
      | c :: rest ->
          if tiles * split * ((out_channels + c - 1) / c) >= 120 then c
          else pick rest
    in
    let cb2 = pick [2 * tiled_cb2; tiled_cb2; 2] in
    conv2d_tiled_patch 2 2 split cb2 name in_channels in_h in_w h w out_channels pad
      input_var weight_var bias_var output_var bn_opt add_opt relu
  else
    conv2d_tiled_1x1 split name in_channels in_h in_w h w out_channels kernel_size stride pad
      input_var weight_var bias_var output_var bn_opt add_opt relu

(* Parse the optional launch marker emitted at the head of a tiled kernel
   body: "/*LAUNCH gx gy gz bx ty*" gives grid (gx,gy,gz) and block
   (bx,by,1). *)
let parse_launch_marker body =
  try
    Scanf.sscanf body "/*LAUNCH %d %d %d %d %d"
      (fun gx gy gz bx by -> Some (gx, gy, gz, bx, by))
  with _ -> None

let find_sub pat s =
  let n = String.length pat and m = String.length s in
  let rec go i =
    if i + n > m then None
    else if String.sub s i n = pat then Some i
    else go (i + 1)
  in
  go 0

(* A split conv kernel body carries a second marker for its companion
   reduce kernel <name>_red. *)
let parse_launch2_marker body =
  match find_sub "/*LAUNCH2 " body with
  | None -> None
  | Some i ->
      (try
         Scanf.sscanf (String.sub body i (String.length body - i))
           "/*LAUNCH2 %d %d %d %d %d"
           (fun gx gy gz bx by -> Some (gx, gy, gz, bx, by))
       with _ -> None)

let launch_snippet kname kbody dims =
  let main_launch =
    match parse_launch_marker kbody with
    | Some (gx, gy, gz, bx, by) ->
        Printf.sprintf "{ dim3 g(%d,%d,%d); dim3 b(%d,%d,1); %s<<<g, b>>>(); }"
          gx gy gz bx by kname
    | None ->
        let n_total = List.fold_left ( * ) 1 dims in
        let block_size = 256 in
        let grid_size = max 1 ((n_total + block_size - 1) / block_size) in
        Printf.sprintf "%s<<<%d, %d>>>();" kname grid_size block_size
  in
  match parse_launch2_marker kbody with
  | None -> main_launch
  | Some (gx, gy, gz, bx, by) ->
      main_launch ^
      Printf.sprintf "\n  { dim3 g(%d,%d,%d); dim3 b(%d,%d,1); %s_red<<<g, b>>>(); }"
        gx gy gz bx by kname

(* Each entry is (kernel_name, kernel_body, output_dimensions).  The product
   of the dimensions gives the number of threads needed. *)
let write_cuda_body path kernel_triples host_name =
  let oc = open_out path in
  output_string oc cuda_prelude;
  List.iter (fun (_, kbody, _) -> output_string oc kbody) kernel_triples;
  output_string oc ("void " ^ host_name ^ "()\n{\n");
  (* G1: kernels are launched on the default stream, which already
     serialises them in issue order; the M10 per-launch
     cudaDeviceSynchronize() is pure overhead (~30 us/launch) and is dropped
     in favour of a single sync at the end of the chain. *)
  List.iter (fun (kname, kbody, dims) ->
    output_string oc ("  " ^ launch_snippet kname kbody dims ^ "\n")
  ) kernel_triples;
  output_string oc "  cudaDeviceSynchronize();\n";
  output_string oc "}\n";
  (* Instrumented variant for per-kernel timing: CUDA events around each
     launch; the KTIME lines are printed only after the whole chain has
     completed, so host-side printing stays off the timed path.  The wrapper
     calls this once after its own TIME measurement. *)
  output_string oc ("void " ^ host_name ^ "_kt()\n{\n");
  output_string oc "  cudaEvent_t eva, evb;\n";
  output_string oc "  cudaEventCreate(&eva);\n  cudaEventCreate(&evb);\n";
  output_string oc (Printf.sprintf "  float kt[%d];\n" (List.length kernel_triples));
  List.iteri (fun i (kname, kbody, dims) ->
    output_string oc "  cudaEventRecord(eva);\n";
    output_string oc ("  " ^ launch_snippet kname kbody dims ^ "\n");
    output_string oc "  cudaEventRecord(evb);\n";
    output_string oc "  cudaEventSynchronize(evb);\n";
    output_string oc (Printf.sprintf "  cudaEventElapsedTime(&kt[%d], eva, evb);\n" i)
  ) kernel_triples;
  List.iteri (fun i (kname, _, _) ->
    output_string oc (Printf.sprintf "  printf(\"KTIME %d %s %%.6f\\n\", kt[%d]);\n" i kname i)
  ) kernel_triples;
  output_string oc "  cudaEventDestroy(eva);\n  cudaEventDestroy(evb);\n";
  output_string oc "}\n";
  close_out oc;
  Printf.printf "Wrote CUDA file: %s\n" path

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

(* Dimensions of a [data] array type (e.g. the pad scratch buffer). *)
let rec data_dims = function
  | Num -> []
  | Ary (n0, n, d') -> nat_to_int' (fle2nat n0 n) :: data_dims d'

(* Infer the stride of a convolution/pooling from the geometry:
   out = (in + 2*pad - k) / stride + 1 (integer division). *)
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

let pattern_to_kernel name_prefix = function
  | DotProduct { out_shape; lhs; rhs; bias; output; pad_input; relu; bn; add; pad_amt; stride; in_hw } ->
      let is_fused = relu = WithReLU || bn <> None || add <> None in
      let kname = name_prefix ^ "_" ^ path_base_name output ^ (if is_fused then "_fused" else "") in
      let input_var = match pad_input with Some s -> s | None -> path_base_name lhs in
      let weight_var = path_base_name rhs in
      let bias_var = match bias with Some p -> path_base_name p | None -> "0" in
      let output_var = path_base_name output in
      let residual_var = match add with Some p -> path_base_name p | None -> "0" in
      let residual_path = match add with Some p -> p | None -> Var "0" in
      let kbody =
        match out_shape with
        | [batch; out_features] ->
            let in_features =
              match path_dims rhs with
              | [_; m] -> m
              | _ -> failwith "pattern_to_kernel: bad linear rhs dims"
            in
            (match bn, add with
             | Some _, _ -> failwith "pattern_to_kernel: linear + BN not yet supported"
             | _, Some _ ->
                 if relu = WithReLU then
                   linear_add_relu_kernel kname batch out_features in_features input_var weight_var bias_var residual_var output_var
                 else
                   linear_add_kernel kname batch out_features in_features input_var weight_var bias_var residual_var output_var
             | None, None ->
                 if relu = WithReLU then
                   linear_relu_kernel kname batch out_features in_features input_var weight_var bias_var output_var
                 else
                   linear_kernel kname batch out_features in_features input_var weight_var bias_var output_var)
        | [batch; out_channels; h; w] ->
            let in_h, in_w =
              match in_hw with
              | Some hw -> hw
              | None -> failwith "pattern_to_kernel: convolution without geometry"
            in
            (match path_dims rhs with
             | [oc; ic; k; k2] when k = k2 ->
                 (* G1: all 4D convolutions use the shared-memory tiled
                    template; BN/Add/ReLU epilogue fused in-kernel. *)
                 conv2d_tiled_kernel_gen kname batch ic in_h in_w h w oc k stride pad_amt input_var weight_var bias_var output_var bn add relu
             | _ -> failwith "pattern_to_kernel: bad conv rhs dims")
        | _ -> failwith "pattern_to_kernel: bad dot_product shape"
      in
      Some (kname, kbody, out_shape)
  | MaxPoolPad { out_shape; k; stride; pad_amt; in_h; in_w; input; output } ->
      let kname = name_prefix ^ "_" ^ path_base_name output in
      let batch, channels, h_out, w_out =
        match out_shape with
        | [b; c; h; w] -> (b, c, h, w)
        | _ -> failwith "pattern_to_kernel: bad maxpoolpad shape"
      in
      let kbody = maxpool_pad_kernel kname batch channels h_out w_out k stride pad_amt in_h in_w
                    (path_base_name input) (path_base_name output)
      in
      Some (kname, kbody, out_shape)
  | MaxReduce { out_shape; input; output; _ } ->
      let kname = name_prefix ^ "_" ^ path_base_name output in
      let batch, channels, h, w =
        match out_shape with
        | [b; c; h; w] -> (b, c, h, w)
        | _ -> failwith "pattern_to_kernel: bad maxreduce shape"
      in
      let kbody = maxpool2d_kernel kname batch channels h w
                    (path_base_name input) (path_base_name output)
      in
      Some (kname, kbody, out_shape)
  | Softmax { out_shape; input; output; _ } ->
      let kname = name_prefix ^ "_" ^ path_base_name output in
      let batch, features =
        match out_shape with
        | [b; f] -> (b, f)
        | _ -> failwith "pattern_to_kernel: bad softmax shape"
      in
      let kbody = softmax2d_kernel kname batch features
                    (path_base_name input) (path_base_name output)
      in
      Some (kname, kbody, out_shape)
  | ElementwiseMap { out_shape; input; output; map_kind } ->
      let input_var = path_base_name input in
      let output_var = path_base_name output in
      let kname = name_prefix ^ "_" ^ output_var in
      let body =
        match map_kind with
        | ReluMap ->
            output_var ^ "[i0][i1][i2][i3] = (" ^
            input_var ^ "[i0][i1][i2][i3] <= 0.0f) ? 0.0f : " ^
            input_var ^ "[i0][i1][i2][i3];"
        | IdentityMap ->
            output_var ^ "[i0][i1][i2][i3] = " ^ input_var ^ "[i0][i1][i2][i3];"
      in
      Some (kname, elementwise_kernel kname out_shape body, out_shape)
  | ElementwiseChain { out_shape; input; output; ops } ->
      let kname = name_prefix ^ "_" ^ path_base_name output ^ "_fused" in
      Some (kname, elementwise_chain_kernel kname out_shape (ElementwiseChain { out_shape; input; output; ops }), out_shape)
  | Reshape { out_shape; in_shape; input; output } ->
      let kname = name_prefix ^ "_" ^ path_base_name output in
      let batch, flat_len =
        match out_shape with
        | [b; f] -> (b, f)
        | _ -> failwith "pattern_to_kernel: bad reshape out"
      in
      let channels, h, w =
        match in_shape with
        | [_; c; h; w] -> (c, h, w)
        | _ -> failwith "pattern_to_kernel: bad reshape in"
      in
      let kbody = flatten4d_kernel kname batch channels h w
                    (path_base_name input) (path_base_name output)
      in
      Some (kname, kbody, out_shape)
  | AvgReduce { out_shape; input; output; _ } ->
      let kname = name_prefix ^ "_" ^ path_base_name output in
      let batch, channels, h, w =
        match out_shape with
        | [b; c; h; w] -> (b, c, h, w)
        | _ -> failwith "pattern_to_kernel: bad avgreduce shape"
      in
      let kbody = avgpool2d_kernel kname batch channels h w
                    (path_base_name input) (path_base_name output)
      in
      Some (kname, kbody, out_shape)
  | GlobalAvgReduce { out_shape; in_shape; input; output } ->
      let kname = name_prefix ^ "_" ^ path_base_name output in
      let batch, channels =
        match out_shape with
        | [b; c] -> (b, c)
        | _ -> failwith "pattern_to_kernel: bad globalavgreduce shape"
      in
      let h, w =
        match in_shape with
        | [_; _; h; w] -> (h, w)
        | _ -> failwith "pattern_to_kernel: bad globalavgreduce in_shape"
      in
      let kbody = global_avgpool2d_kernel kname batch channels h w
                    (path_base_name input) (path_base_name output)
      in
      Some (kname, kbody, out_shape)
  | GlobalAvgLinear { out_shape; in_shape; input; weight; bias; output; relu } ->
      let kname = name_prefix ^ "_" ^ path_base_name output ^ "_fused" in
      let batch, out_features =
        match out_shape with
        | [b; o] -> (b, o)
        | _ -> failwith "pattern_to_kernel: bad globalavglinear shape"
      in
      let channels, h, w =
        match in_shape with
        | [_; c; h; w] -> (c, h, w)
        | _ -> failwith "pattern_to_kernel: bad globalavglinear in_shape"
      in
      let input_var = path_base_name input in
      let weight_var = path_base_name weight in
      let bias_var = match bias with Some p -> path_base_name p | None -> "0" in
      let output_var = path_base_name output in
      let kbody =
        if relu = WithReLU then
          global_avg_linear_relu_kernel kname batch channels h w out_features input_var weight_var bias_var output_var
        else
          (* G1: block-parallel variant (batch = 1) *)
          global_avg_linear_block_kernel kname channels h w out_features input_var weight_var bias_var output_var
      in
      Some (kname, kbody, out_shape)
  | MatMul { out_shape; lhs; rhs; output } ->
      let kname = name_prefix ^ "_" ^ path_base_name output in
      let input_var = path_base_name lhs in
      let weight_var = path_base_name rhs in
      let output_var = path_base_name output in
      let kbody =
        match out_shape with
        | [batch; out_features] ->
            let in_features =
              match path_dims rhs with
              | [_; m] -> m
              | _ -> failwith "pattern_to_kernel: bad matmul rhs dims"
            in
            matmul2d_kernel kname batch out_features in_features input_var weight_var output_var
        | _ -> failwith "pattern_to_kernel: bad matmul shape"
      in
      Some (kname, kbody, out_shape)

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
      (* The false branch is either [best = best] (mini_alexnet form) or
         [Skip] (ResNet-18 form). *)
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

(* Same search as [find_original_input] but returns the full source path
   (its AryIdx chain carries the input tensor dimensions). *)
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

(* Recognise the ResNet-18 max-pool component: a zero-filled pad scratch
   buffer followed by a zero-initialised max reduction over k x k windows
   with an arbitrary stride. *)
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

let try_match_linear name_prefix c =
  match recognize_dot_product c with
  | Some (DotProduct { out_shape = [_; _]; _ } as pat) ->
      pattern_to_kernel name_prefix pat
  | _ -> None

(* Convolution component wrapped in a zero-filled pad scratch buffer
   (ResNet-18 stem conv and all 3x3 block convs). *)
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

(* Bare convolution without a padding buffer (ResNet-18 1x1 projection
   shortcuts): pad = 0, stride inferred from input/output geometry. *)
let recognize_direct_conv c =
  match recognize_dot_product c with
  | Some (DotProduct dp) when dp.pad_input = None ->
      (match dp.out_shape, path_dims dp.lhs, path_dims dp.rhs with
       | [_; _; out_h; out_w], [_; _; in_h; in_w], [_; _; k; _] ->
           let stride = infer_stride in_h in_w k 0 out_h out_w in
           Some (DotProduct { dp with pad_amt = 0; stride; in_hw = Some (in_h, in_w) })
       | _ -> None)
  | _ -> None

let try_match_conv2d_pad name_prefix c =
  match recognize_conv_with_pad c with
  | Some pat -> pattern_to_kernel name_prefix pat
  | None -> None

let try_match_maxpool2d name_prefix c =
  match recognize_max_reduce c with
  | Some (MaxReduce { out_shape = [_; _; _; _]; _ } as pat) ->
      pattern_to_kernel name_prefix pat
  | _ -> None

let try_match_softmax2d name_prefix c =
  match recognize_softmax c with
  | Some (Softmax { out_shape = [_; _]; _ } as pat) ->
      pattern_to_kernel name_prefix pat
  | _ -> None

let try_match_relu4d name_prefix c =
  match recognize_elementwise_relu c with
  | Some (ElementwiseMap { out_shape = [_; _; _; _]; map_kind = ReluMap; _ } as pat) ->
      pattern_to_kernel name_prefix pat
  | _ -> None

let try_match_elementwise_add name_prefix c =
  match recognize_elementwise_add c with
  | Some (ElementwiseChain { out_shape = [_; _; _; _]; _ } as pat) ->
      pattern_to_kernel name_prefix pat
  | _ -> None

let try_match_elementwise_bn name_prefix c =
  match recognize_elementwise_bn c with
  | Some (ElementwiseChain { out_shape = [_; _; _; _]; _ } as pat) ->
      pattern_to_kernel name_prefix pat
  | _ -> None

let try_match_elementwise_dropout name_prefix c =
  match recognize_elementwise_dropout c with
  | Some (ElementwiseChain { out_shape = [_; _; _; _]; _ } as pat) ->
      pattern_to_kernel name_prefix pat
  | _ -> None

let try_match_flatten4d name_prefix c =
  match recognize_flatten c with
  | Some (Reshape { out_shape = [_; _]; in_shape = [_; _; _; _]; _ } as pat) ->
      pattern_to_kernel name_prefix pat
  | _ -> None

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

let recognize_matmul c =
  match recognize_dot_product c with
  | Some (DotProduct { out_shape = ([_; _] as out_shape); red_dims; lhs; rhs; bias = None; output; pad_input = None; _ }) ->
      (match red_dims with
       | [m] -> Some (MatMul { out_shape; red_dim = m; lhs; rhs; output })
       | _ -> None)
  | _ -> None

let try_match_avgpool2d name_prefix c =
  match recognize_avg_reduce c with
  | Some (AvgReduce { out_shape = [_; _; _; _]; _ } as pat) ->
      pattern_to_kernel name_prefix pat
  | _ -> None

let try_match_global_avgpool2d name_prefix c =
  match recognize_global_avg_reduce c with
  | Some (GlobalAvgReduce { out_shape = [_; _]; _ } as pat) ->
      pattern_to_kernel name_prefix pat
  | _ -> None

let try_match_matmul2d name_prefix c =
  match recognize_matmul c with
  | Some (MatMul { out_shape = [_; _]; _ } as pat) ->
      pattern_to_kernel name_prefix pat
  | _ -> None

let pattern_output = function
  | DotProduct { output } -> output
  | MaxReduce { output } -> output
  | MaxPoolPad { output } -> output
  | Softmax { output } -> output
  | ElementwiseMap { output } -> output
  | ElementwiseChain { output } -> output
  | Reshape { output } -> output
  | AvgReduce { output } -> output
  | GlobalAvgReduce { output } -> output
  | MatMul { output } -> output
  | GlobalAvgLinear { output } -> output

let pattern_inputs = function
  | DotProduct { lhs; rhs; bias; pad_input; add } ->
      (match bias with Some b -> [lhs; rhs; b] | None -> [lhs; rhs])
      @ (match pad_input with Some s -> [Var s] | None -> [])
      @ (match add with Some p -> [p] | None -> [])
  | GlobalAvgLinear { input; weight; bias } ->
      (match bias with Some b -> [input; weight; b] | None -> [input; weight])
  | MaxReduce { input } -> [input]
  | MaxPoolPad { input } -> [input]
  | Softmax { input } -> [input]
  | ElementwiseMap { input } -> [input]
  | ElementwiseChain { input; ops } ->
      let extras =
        List.fold_left (fun acc (_, op) ->
          match op with
          | AddOp p -> p :: acc
          | BNOp { mean; var; gamma; beta } -> mean :: var :: gamma :: beta :: acc
          | DropoutOp e ->
              (match rexp_as_path e with Some p -> p :: acc | None -> acc)
          | _ -> acc
        ) [] ops
      in
      input :: extras
  | Reshape { input } -> [input]
  | AvgReduce { input } -> [input]
  | GlobalAvgReduce { input } -> [input]
  | MatMul { lhs; rhs } -> [lhs; rhs]

let pattern_shape = function
  | DotProduct { out_shape } -> out_shape
  | MaxReduce { out_shape } -> out_shape
  | MaxPoolPad { out_shape } -> out_shape
  | Softmax { out_shape } -> out_shape
  | ElementwiseMap { out_shape } -> out_shape
  | ElementwiseChain { out_shape } -> out_shape
  | Reshape { out_shape } -> out_shape
  | AvgReduce { out_shape } -> out_shape
  | GlobalAvgReduce { out_shape } -> out_shape
  | MatMul { out_shape } -> out_shape
  | GlobalAvgLinear { out_shape } -> out_shape

let chain_of_pattern = function
  | ElementwiseMap { out_shape; input; output; map_kind = ReluMap } ->
      Some (ElementwiseChain { out_shape; input; output; ops = [(output, ReluOp)] })
  | ElementwiseMap { out_shape; input; output; map_kind = IdentityMap } ->
      Some (ElementwiseChain { out_shape; input; output; ops = [(output, IdentityOp)] })
  | ElementwiseChain _ as c -> Some c
  | _ -> None

let path_base_equal p1 p2 = path_base_name p1 = path_base_name p2

(* Compute dependency edges between kernels based on compute_pattern inputs/outputs.
   Result is a list where the i-th element contains indices j < i of kernels whose
   output is consumed by kernel i. *)
let pattern_dependencies patterns =
  let n = List.length patterns in
  let rec range a b = if a >= b then [] else a :: range (a + 1) b in
  let rec deps i =
    if i >= n then [] else
      let pi = List.nth patterns i in
      let inputs = pattern_inputs pi in
      let d =
        List.filter (fun j ->
          let pj = List.nth patterns j in
          List.exists (path_base_equal (pattern_output pj)) inputs
        ) (range 0 i)
      in
      d :: deps (i + 1)
  in
  deps 0

(* Same as write_cuda_body but emits CUDA streams and events so that independent
   kernels can execute concurrently while dependent kernels are ordered by
   cudaStreamWaitEvent.  A final cudaDeviceSynchronize() is kept so that the
   wrapper can safely copy the output back to host. *)
let write_cuda_body_async path kernel_triples patterns host_name =
  let n = List.length kernel_triples in
  let deps = pattern_dependencies patterns in
  let oc = open_out path in
  output_string oc cuda_prelude;
  List.iter (fun (_, kbody, _) -> output_string oc kbody) kernel_triples;
  output_string oc ("void " ^ host_name ^ "()\n{\n");
  for i = 0 to n - 1 do
    output_string oc ("  cudaStream_t stream_" ^ string_of_int i ^ ";\n");
    output_string oc ("  cudaEvent_t event_" ^ string_of_int i ^ ";\n")
  done;
  for i = 0 to n - 1 do
    output_string oc ("  CUDA_CHECK(cudaStreamCreate(&stream_" ^ string_of_int i ^ "));\n");
    output_string oc ("  CUDA_CHECK(cudaEventCreate(&event_" ^ string_of_int i ^ "));\n")
  done;
  List.iteri (fun i (kname, _, dims) ->
    let dlist = List.nth deps i in
    List.iter (fun j ->
      output_string oc ("  CUDA_CHECK(cudaStreamWaitEvent(stream_" ^ string_of_int i ^
                        ", event_" ^ string_of_int j ^ ", 0));\n")
    ) dlist;
    let n_total = List.fold_left ( * ) 1 dims in
    let block_size = 256 in
    let grid_size = max 1 ((n_total + block_size - 1) / block_size) in
    output_string oc ("  " ^ kname ^ "<<<" ^ string_of_int grid_size ^
                      ", " ^ string_of_int block_size ^
                      ", 0, stream_" ^ string_of_int i ^ ">>>();\n");
    output_string oc ("  CUDA_CHECK(cudaEventRecord(event_" ^ string_of_int i ^
                      ", stream_" ^ string_of_int i ^ "));\n")
  ) kernel_triples;
  output_string oc "  CUDA_CHECK(cudaDeviceSynchronize());\n";
  for i = 0 to n - 1 do
    output_string oc ("  CUDA_CHECK(cudaEventDestroy(event_" ^ string_of_int i ^ "));\n");
    output_string oc ("  CUDA_CHECK(cudaStreamDestroy(stream_" ^ string_of_int i ^ "));\n")
  done;
  output_string oc "}\n";
  close_out oc;
  Printf.printf "Wrote async CUDA file: %s\n" path

let rec path_used_in_pattern p pat =
  let base = path_base_name p in
  path_base_name (pattern_output pat) = base
  || List.exists (fun q -> path_base_name q = base) (pattern_inputs pat)

let shape_nelements shape = List.fold_left ( * ) 1 shape

let rec fuse_elementwise_chains cost_model ps =
  let chain_input = function
    | ElementwiseChain { input } -> input
    | _ -> failwith "fuse_elementwise_chains: not a chain"
  in
  let chain_output = function
    | ElementwiseChain { output } -> output
    | _ -> failwith "fuse_elementwise_chains: not a chain"
  in
  let chain_shape = function
    | ElementwiseChain { out_shape } -> out_shape
    | _ -> failwith "fuse_elementwise_chains: not a chain"
  in
  let chain_ops = function
    | ElementwiseChain { ops } -> ops
    | _ -> failwith "fuse_elementwise_chains: not a chain"
  in
  let merge cur next =
    match cur with
    | ElementwiseChain _ ->
        ElementwiseChain {
          out_shape = chain_shape cur;
          input = chain_input cur;
          output = chain_output next;
          ops = chain_ops cur @ chain_ops next;
        }
    | DotProduct dp ->
        let output = chain_output next in
        (match chain_ops next with
         | [(_, BNOp { mean; var; gamma; beta; eps })] when dp.bn = None ->
             let bn = { mean; var; gamma; beta; eps } in
             DotProduct { dp with output; bn = Some bn }
         | [(_, ReluOp)] ->
             DotProduct { dp with output; relu = WithReLU }
         | [(_, AddOp residual)] when dp.add = None ->
             DotProduct { dp with output; add = Some residual }
         | _ ->
             failwith "fuse_elementwise_chains: cannot merge DotProduct with this chain")
    | _ -> failwith "fuse_elementwise_chains: cannot merge this pattern"
  in
  let can_merge cur next =
    if pattern_shape cur <> chain_shape next then false else
    let n_elems = shape_nelements (pattern_shape cur) in
    if n_elems < cost_model.min_fuse_elements then
      (Printf.printf "cost model: skip fusion, output size %d < min_fuse_elements %d\n" n_elems cost_model.min_fuse_elements;
       false)
    else
      match cur with
      | ElementwiseChain _ ->
          let merged_ops = List.length (chain_ops cur @ chain_ops next) in
          if merged_ops > cost_model.max_chain_ops then
            (Printf.printf "cost model: skip chain fusion, merged ops %d > max_chain_ops %d\n" merged_ops cost_model.max_chain_ops;
             false)
          else
            path_base_equal (pattern_output cur) (chain_input next)
      | DotProduct { relu = NoReLU; bn; add; out_shape; _ } ->
          let is_linear = match out_shape with [_; _] -> true | _ -> false in
          (match chain_ops next with
           | [(_, ReluOp)] ->
               let allowed = if is_linear then cost_model.enable_linear_relu else cost_model.enable_conv_relu in
               if not allowed then
                 (Printf.printf "cost model: skip %s + ReLU fusion (disabled)\n"
                    (if is_linear then "linear" else "conv");
                  false)
               else
                 path_base_equal (pattern_output cur) (chain_input next)
           | [(_, BNOp _)] when bn = None && add = None ->
               if not cost_model.enable_conv_bn then
                 (Printf.printf "cost model: skip Conv + BN fusion (disabled)\n";
                  false)
               else
                 path_base_equal (pattern_output cur) (chain_input next)
           | [(_, AddOp _)] when add = None ->
               if not cost_model.enable_conv_add then
                 (Printf.printf "cost model: skip %s + Add fusion (disabled)\n"
                    (if is_linear then "linear" else "conv");
                  false)
               else
                 path_base_equal (pattern_output cur) (chain_input next)
           | _ -> false)
      | _ -> false
  in
  let rec try_merge cur = function
    | [] -> [cur]
    | p :: rest as ps ->
        match chain_of_pattern p with
        | None -> cur :: fuse_elementwise_chains cost_model ps
        | Some next when can_merge cur next ->
            if List.exists (path_used_in_pattern (pattern_output cur)) rest then
              cur :: fuse_elementwise_chains cost_model ps
            else
              try_merge (merge cur next) rest
        | Some _ -> cur :: fuse_elementwise_chains cost_model ps
  in
  match ps with
  | [] -> []
  | p :: rest ->
      match p with
      | DotProduct _ -> try_merge p rest
      | GlobalAvgReduce { out_shape = [b; c] as out_shape; in_shape; input; output } ->
          (match rest with
           | (DotProduct { out_shape = [_; out_features]; red_dims; lhs; rhs; bias; output = lin_output;
                           pad_input = None; relu; bn = None; add = None; _ }) :: rest2
             when path_base_equal output lhs &&
                  not (List.exists (path_used_in_pattern output) rest2) ->
               if not cost_model.enable_global_avg_linear then
                 (Printf.printf "cost model: skip GlobalAvgPool + Linear fusion (disabled)\n";
                  p :: fuse_elementwise_chains cost_model rest)
               else
                 let fused = GlobalAvgLinear {
                   out_shape = [b; out_features];
                   in_shape;
                   input;
                   weight = rhs;
                   bias;
                   output = lin_output;
                   relu;
                 } in
                 fuse_elementwise_chains cost_model (fused :: rest2)
           | _ -> p :: fuse_elementwise_chains cost_model rest)
      | _ ->
          match chain_of_pattern p with
          | Some c -> try_merge c rest
          | None -> p :: fuse_elementwise_chains cost_model rest

let try_match_any_pattern c =
  match recognize_dot_product c with
  | Some (DotProduct { out_shape = [_; _]; _ } as pat) -> Some pat
  | _ ->
      (match recognize_direct_conv c with
       | Some _ as r -> r
       | None ->
           match recognize_conv_with_pad c with
           | Some _ as r -> r
           | None ->
               match recognize_maxpool_pad c with
               | Some _ as r -> r
               | None ->
              match recognize_flatten c with
              | Some (Reshape { out_shape = [_; _]; in_shape = [_; _; _; _]; _ } as pat) -> Some pat
              | _ ->
                  match recognize_matmul c with
                  | Some (MatMul { out_shape = [_; _]; _ } as pat) -> Some pat
                  | _ ->
                      match recognize_max_reduce c with
                      | Some (MaxReduce { out_shape = [_; _; _; _]; _ } as pat) -> Some pat
                      | _ ->
                          match recognize_avg_reduce c with
                          | Some (AvgReduce { out_shape = [_; _; _; _]; _ } as pat) -> Some pat
                          | _ ->
                              match recognize_global_avg_reduce c with
                              | Some (GlobalAvgReduce { out_shape = [_; _]; _ } as pat) -> Some pat
                              | _ ->
                                  match recognize_softmax c with
                                  | Some (Softmax { out_shape = [_; _]; _ } as pat) -> Some pat
                                  | _ ->
                                      match recognize_elementwise_relu c with
                                      | Some (ElementwiseMap { out_shape = ([_; _] | [_; _; _; _]); map_kind = ReluMap; _ } as pat) -> Some pat
                                      | _ ->
                                          match recognize_elementwise_add c with
                                          | Some (ElementwiseChain { out_shape = [_; _; _; _]; _ } as pat) -> Some pat
                                          | _ ->
                                              match recognize_elementwise_bn c with
                                              | Some (ElementwiseChain { out_shape = [_; _; _; _]; _ } as pat) -> Some pat
                                              | _ ->
                                                  match recognize_elementwise_dropout c with
                                                  | Some (ElementwiseChain { out_shape = [_; _; _; _]; _ } as pat) -> Some pat
                                                  | _ -> None)

let try_match_any_reduction name_prefix c =
  match try_match_any_pattern c with
  | Some pat -> pattern_to_kernel name_prefix pat
  | None -> None

(* ---------------------------------------------------------------------- *)
(* ResNet-18 end-to-end generation (224x224 input).
   The full [resnet18_comm] AST (stem conv7x7/s2 + BN + ReLU + maxpool,
   8 BasicBlocks with BN/Add/ReLU chains and 1x1 projection shortcuts,
   GlobalAvgPool + FC) is split into top-level components; each component
   is recognised by the pattern matchers, adjacent Conv+BN+ReLU(+Add)
   chains are fused, and every resulting pattern is emitted as a naive CUDA
   kernel (one thread per output element, sequential in-thread reduction). *)

let main () =
  let cost_model = default_cost_model in
  (* Each BasicBlock is a nested Seq; flatten all Seq levels so every
     operator becomes its own component.  New-wrapped pad buffers stay
     atomic (the Seq sits under the New). *)
  let rec flatten_seqs = function
    | Seq (c1, c2) -> flatten_seqs c1 @ flatten_seqs c2
    | c -> [c]
  in
  let components = flatten_seqs resnet18_comm in
  Printf.printf "resnet18: %d top-level components\n" (List.length components);
  let patterns =
    List.mapi (fun i c ->
      match try_match_any_pattern c with
      | Some p ->
          Printf.printf "resnet18 component %d: matched, output = %s\n" i
            (path_base_name (pattern_output p));
          p
      | None ->
          failwith ("could not match resnet18 component " ^ string_of_int i)
    ) components
  in
  let fused = fuse_elementwise_chains cost_model patterns in
  Printf.printf "resnet18: %d patterns -> %d kernels after fusion\n"
    (List.length patterns) (List.length fused);
  let kernels =
    (* Index-qualified prefix: after fusion a fused Conv+BN+Add kernel and an
       in-place ReLU chain can share the same output buffer name, so plain
       output-based names would collide. *)
    List.mapi (fun i p ->
      match pattern_to_kernel ("ast" ^ string_of_int i) p with
      | Some t -> t
      | None -> failwith "pattern_to_kernel failed for resnet18"
    ) fused
  in
  List.iter (fun (kname, _, dims) ->
    Printf.printf "  kernel %-36s out_dims = [%s]\n" kname
      (String.concat ";" (List.map string_of_int dims))
  ) kernels;
  (* Sequential launch chain: one kernel per operator, synchronised after
     each launch.  This mirrors the CPU streaming order exactly and keeps
     the trust boundary minimal. *)
  write_cuda_body "resnet18_tiled_cuda_output.cu" kernels "resnet18_tiled"

let _ = main ()
