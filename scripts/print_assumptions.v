(* Axiom audit for the Ionia open-source release.
   Run after a full build:
     coqc -R src/Ionia Ionia scripts/print_assumptions.v
   (or use scripts/print_assumptions.sh, which does this for you).

   Expected: every theorem below closes over only the standard classical
   axioms (functional extensionality, proof irrelevance, classical logic
   from Coq.Logic.Berardi / Coq.Logic.Classical*, and real-number axioms
   from the Reals library).  No Admitted / custom axioms are used. *)

From Ionia Require Import DPIA_Rules Parallel_Safety.
From Ionia Require Import Opt_Tiling Opt_Fusion Opt_ParGranularity Opt_TiledConv Opt_BlockFusion Opt_HierPar.
From Ionia Require Import Opt_ApproxError Opt_ApproxError_Apply.
From Ionia Require Import LeNet5_DPIA AlexNet_DPIA VGG16_DPIA ResNet18_DPIA ResNet50_DPIA
  GoogLeNet_DPIA MobileNetV1_DPIA DenseNet121_DPIA SqueezeNet_DPIA ResNeXt50_DPIA EfficientNetB0_DPIA.
From Ionia Require Import MiniTransformer_DPIA MiniTransformer_AST_Faithful.
From Ionia Require Import Bridge_LeNet5 Bridge_AlexNet Bridge_VGG16 Bridge_ResNet18.

(* ---- Core rewriting rules (DPIA_Rules.v) ---- *)
Print Assumptions Ionia.DPIA_Rules.AmkvSeq.
Print Assumptions Ionia.DPIA_Rules.AmkvPar.
Print Assumptions Ionia.DPIA_Rules.A_sum.
Print Assumptions Ionia.DPIA_Rules.A_max.

(* ---- Parallel safety (Parallel_Safety.v) ---- *)
Print Assumptions Ionia.Parallel_Safety.seqfor_perm.
Print Assumptions Ionia.Parallel_Safety.schedule_safe.
Print Assumptions Ionia.Parallel_Safety.parallel_safe.
Print Assumptions Ionia.Parallel_Safety.parallel_safe_A.

(* ---- Optimization rules ---- *)
Print Assumptions Ionia.Opt_Tiling.seqfor_split.
Print Assumptions Ionia.Opt_Fusion.A_Conv4d_ReLU4d.
Print Assumptions Ionia.Opt_ParGranularity.parallel_safe_coarsen.
Print Assumptions Ionia.Opt_TiledConv.A_Conv4d_tiled.
Print Assumptions Ionia.Opt_TiledConv.A_Conv4d_BN_ReLU4d.
Print Assumptions Ionia.Opt_BlockFusion.A_BasicBlock_fused.
Print Assumptions Ionia.Opt_BlockFusion.basicblock_id_comm_eq_fused.

(* ---- Hierarchical parallelism (OpenCL/Ventus, Opt_HierPar.v) ---- *)
Print Assumptions Ionia.Opt_HierPar.AmkvPar3.
Print Assumptions Ionia.Opt_HierPar.lds_reduce_correct.

(* ---- Rounding-error bounds (Opt_ApproxError*.v) ---- *)
Print Assumptions Ionia.Opt_ApproxError.rsum_rnd_error.
Print Assumptions Ionia.Opt_ApproxError.dot_rnd_error.
Print Assumptions Ionia.Opt_ApproxError.conv_dot_error.

(* ---- End-to-end network theorems ---- *)
Print Assumptions Ionia.LeNet5_DPIA.A_LeNet5.
Print Assumptions Ionia.AlexNet_DPIA.A_AlexNet.
Print Assumptions Ionia.VGG16_DPIA.A_VGG16.
Print Assumptions Ionia.ResNet18_DPIA.A_ResNet18.
Print Assumptions Ionia.ResNet50_DPIA.A_ResNet50.
Print Assumptions Ionia.GoogLeNet_DPIA.A_GoogLeNet.
Print Assumptions Ionia.MobileNetV1_DPIA.A_MobileNetV1.
Print Assumptions Ionia.DenseNet121_DPIA.A_DenseNet121.
Print Assumptions Ionia.SqueezeNet_DPIA.A_SqueezeNet.
Print Assumptions Ionia.ResNeXt50_DPIA.A_ResNeXt50.
Print Assumptions Ionia.EfficientNetB0_DPIA.A_EfficientNetB0.
Print Assumptions Ionia.MiniTransformer_DPIA.A_MiniTransformer.
Print Assumptions Ionia.MiniTransformer_AST_Faithful.Bridge_MiniTransformer_faithful.

(* ---- A/B bridges ---- *)
Print Assumptions Ionia.Bridge_LeNet5.Bridge_LeNet5.
Print Assumptions Ionia.Bridge_AlexNet.Bridge_AlexNet.
Print Assumptions Ionia.Bridge_VGG16.Bridge_VGG16.
Print Assumptions Ionia.Bridge_ResNet18.Bridge_ResNet18.
