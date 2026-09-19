OCAMLC=ocamlc -g
NVCC ?= nvcc
NVCC_FLAGS ?= -O2 -arch=native
PYTHON ?= python3

# Networks whose extracted *_sym.ml ASTs are shipped in this release.
NETWORKS = alexnet lenet5 resnet18

# Loopacc-aware sequential printers compatible with the shipped
# Rocq 9.2 extraction format (comm_to_c.ml / comm_to_c_resnet18.ml predate
# it and are kept for reference only).
PRINTERS = comm_to_c_alexnet_new comm_to_c_lenet5 comm_to_c_resnet18_seq

all: $(PRINTERS)

alexnet_sym.cmo: alexnet_sym.ml
	$(OCAMLC) -c $<

lenet5_sym.cmo: lenet5_sym.ml
	$(OCAMLC) -c $<

lenet5_par_sym.cmo: lenet5_par_sym.ml
	$(OCAMLC) -c $<

alexnet_par_sym.cmo: alexnet_par_sym.ml
	$(OCAMLC) -c $<

resnet18_sym.cmo: resnet18_sym.ml
	$(OCAMLC) -c $<

# Legacy printer: predates the Rocq 9.x extraction format; does NOT build
# against the shipped alexnet_sym.ml. Use comm_to_c_alexnet_new instead.
comm_to_c_alexnet: alexnet_sym.cmo comm_to_c.ml
	$(OCAMLC) -o $@ unix.cma $^

comm_to_c_lenet5: lenet5_sym.cmo comm_to_c_lenet5.ml
	$(OCAMLC) -o $@ unix.cma $^

comm_to_c_lenet5_par: lenet5_par_sym.cmo comm_to_c_lenet5_par.ml
	$(OCAMLC) -o $@ unix.cma $^

# OpenMP variant for AlexNet: parfor -> #pragma omp parallel for
# (mirrors comm_to_c_lenet5_par; see comm_to_c_alexnet_par.ml).
comm_to_c_alexnet_par: alexnet_par_sym.cmo comm_to_c_alexnet_par.ml
	$(OCAMLC) -o $@ unix.cma $^

# Legacy printer (pre-Rocq-9.x extraction format); use comm_to_c_resnet18_seq.
comm_to_c_resnet18: resnet18_sym.cmo comm_to_c_resnet18.ml
	$(OCAMLC) -o $@ unix.cma $^

resnet18_par_sym.cmo: resnet18_par_sym.ml
	$(OCAMLC) -c $<

# Sequential ResNet-18 printer based on the Loopacc-aware streaming printer
# (comm_to_c_resnet18.ml predates the Rocq 9.2 extraction format).
comm_to_c_resnet18_seq: resnet18_sym.cmo comm_to_c_resnet18_seq.ml
	$(OCAMLC) -o $@ unix.cma $^

# OpenMP variant: parfor -> #pragma omp parallel for, New inside a parfor
# body becomes a per-iteration local declaration (see comm_to_c_lenet5_par.ml).
comm_to_c_resnet18_par: resnet18_par_sym.cmo comm_to_c_resnet18_par.ml
	$(OCAMLC) -o $@ unix.cma $^

vgg16s_sym.cmo: vgg16s_sym.ml
	$(OCAMLC) -c $<

vgg16p_sym.cmo: vgg16p_sym.ml
	$(OCAMLC) -c $<

comm_to_c_vgg16s_par: vgg16s_sym.cmo comm_to_c_vgg16s_par.ml
	$(OCAMLC) -o $@ unix.cma $^

run: $(PRINTERS)
	@for p in $(PRINTERS); do ./$$p; done

wrapped: run
	@set -e; for n in $(NETWORKS); do \
	  python3 wrap_generated_c.py $$n $${n}_streaming_output.c $${n}_wrapped.c; \
	  gcc -std=c99 -c $${n}_wrapped.c -o /tmp/$${n}_wrapped.o; \
	  echo "$$n wrapped C compiles OK"; \
	done

validate: wrapped
	@set -e; \
	$(PYTHON) validate_lenet5.py; \
	$(PYTHON) validate_alexnet.py; \
	$(PYTHON) validate_resnet18.py; \
	echo "Numerical validation passed for shipped networks."

# AlexNet end-to-end CUDA printer (pattern matchers + Conv/ReLU fusion).
comm_to_cuda_alexnet: alexnet_sym.cmo comm_to_cuda_alexnet.ml
	$(OCAMLC) -g -o $@ unix.cma $^

# Sequential C printer for AlexNet based on the Loopacc-aware streaming
# printer (comm_to_c.ml predates the Rocq 9.2 extraction format).
comm_to_c_alexnet_new: alexnet_sym.cmo comm_to_c_alexnet_new.ml
	$(OCAMLC) -g -o $@ unix.cma $^

cuda-alexnet: comm_to_cuda_alexnet
	./comm_to_cuda_alexnet
	python3 wrap_generated_cuda.py alexnet_test alexnet_test_cuda_output.cu alexnet_test.cu
	@echo "AlexNet CUDA generated: alexnet_test.cu"

validate-cuda-alexnet: comm_to_cuda_alexnet
	python3 validate_cuda_alexnet.py

# ResNet-18 (224x224) end-to-end CUDA printer (M10b): pattern matchers +
# Conv/BN/ReLU/Add fusion, stride/pad-aware kernels, padded max-pool.
comm_to_cuda_resnet18: resnet18_sym.cmo comm_to_cuda_resnet18.ml
	$(OCAMLC) -g -o $@ unix.cma $^

cuda-resnet18: comm_to_cuda_resnet18
	./comm_to_cuda_resnet18
	python3 wrap_generated_cuda.py resnet18_test resnet18_test_cuda_output.cu resnet18_test.cu
	@echo "ResNet-18 CUDA generated: resnet18_test.cu"

validate-cuda-resnet18: comm_to_cuda_resnet18
	python3 validate_cuda_resnet18.py

clean:
	rm -f *.cmo *.cmi \
		comm_to_c comm_to_c_alexnet comm_to_c_alexnet_new \
		comm_to_c_lenet5 comm_to_c_lenet5_par comm_to_c_alexnet_par \
		comm_to_c_resnet18 \
		comm_to_c_resnet18_seq comm_to_c_resnet18_par \
		comm_to_c_vgg16s_par \
		comm_to_triton comm_to_opencl \
		comm_to_c_minitransformer comm_to_c_minitransformer_par \
		comm_to_c_minitransformer_faithful \
		*_streaming_output.c

.PHONY: all run wrapped validate clean

# VGG-16 end-to-end CUDA printers (M10a), derived from comm_to_cuda_alexnet.ml.
comm_to_cuda_vgg16s: vgg16s_sym.cmo comm_to_cuda_vgg16s.ml
	$(OCAMLC) -g -o $@ unix.cma $^

comm_to_cuda_vgg16p: vgg16p_sym.cmo comm_to_cuda_vgg16p.ml
	$(OCAMLC) -g -o $@ unix.cma $^

cuda-vgg16s: comm_to_cuda_vgg16s
	./comm_to_cuda_vgg16s
	python3 wrap_generated_cuda.py vgg16s_test vgg16s_test_cuda_output.cu vgg16s_test.cu
	@echo "VGG16s CUDA generated: vgg16s_test.cu"

cuda-vgg16p: comm_to_cuda_vgg16p
	./comm_to_cuda_vgg16p
	python3 wrap_generated_cuda.py vgg16p_test vgg16p_test_cuda_output.cu vgg16p_test.cu
	@echo "VGG16p CUDA generated: vgg16p_test.cu"

validate-cuda-vgg16s: comm_to_cuda_vgg16s
	python3 validate_cuda_vgg16s.py

validate-cuda-vgg16p: comm_to_cuda_vgg16p
	python3 validate_cuda_vgg16p.py

# G1: shared-memory tiled convolution variants of the M10 CUDA printers.
# Same ASTs, matchers and fusion; only the conv kernel template
# (TILE x TILE shared-memory tile, CB output channels per block, fused
# BN/Add/ReLU epilogue) and per-kernel launch geometry differ.
comm_to_cuda_vgg16p_tiled: vgg16p_sym.cmo comm_to_cuda_vgg16p_tiled.ml
	$(OCAMLC) -g -o $@ unix.cma $^

comm_to_cuda_resnet18_tiled: resnet18_sym.cmo comm_to_cuda_resnet18_tiled.ml
	$(OCAMLC) -g -o $@ unix.cma $^

cuda-vgg16p-tiled: comm_to_cuda_vgg16p_tiled
	./comm_to_cuda_vgg16p_tiled
	python3 wrap_generated_cuda.py vgg16p_tiled vgg16p_tiled_cuda_output.cu vgg16p_tiled.cu
	@echo "VGG16p tiled CUDA generated: vgg16p_tiled.cu"

cuda-resnet18-tiled: comm_to_cuda_resnet18_tiled
	./comm_to_cuda_resnet18_tiled
	python3 wrap_generated_cuda.py resnet18_tiled resnet18_tiled_cuda_output.cu resnet18_tiled.cu
	@echo "ResNet-18 tiled CUDA generated: resnet18_tiled.cu"

validate-cuda-vgg16p-tiled: comm_to_cuda_vgg16p_tiled
	python3 validate_cuda_vgg16p_tiled.py

validate-cuda-resnet18-tiled: comm_to_cuda_resnet18_tiled
	python3 validate_cuda_resnet18_tiled.py

# ResNet-18 E4 fully-fused: whole-BasicBlock fused kernels
# (conv-bn-relu-conv-bn[-add]-relu single kernel + split-c reduce for deep
# stages).  Structure-specialised emitter, no sym dependency.
comm_to_cuda_resnet18_fused: comm_to_cuda_resnet18_fused.ml
	$(OCAMLC) -o $@ unix.cma $^

cuda-resnet18-fused: comm_to_cuda_resnet18_fused
	./comm_to_cuda_resnet18_fused
	python3 wrap_generated_cuda.py resnet18_fused resnet18_fused_cuda_output.cu resnet18_fused.cu
	@echo "ResNet-18 fused CUDA generated: resnet18_fused.cu"

validate-cuda-resnet18-fused: comm_to_cuda_resnet18_fused
	python3 validate_cuda_resnet18_fused.py

# Implicit-GEMM FP32: implicit GEMM conv, NHWC chain, split-K + ordered
# reduce with fused BN/ReLU/Add epilogue.  The body
# resnet18_igemm_cuda_output.cu is generated from the ResNet-18 AST by
# comm_to_cuda_resnet18_igemm.ml (the hand-written pilot is kept as
# resnet18_igemm_handwritten_ref.cu).  Compile via validate script (honors
# NVCC/NVCC_FLAGS env).
comm_to_cuda_resnet18_igemm: resnet18_sym.cmo comm_to_cuda_resnet18_igemm.ml
	$(OCAMLC) -g -o $@ unix.cma $^

resnet18_igemm_cuda_output.cu: comm_to_cuda_resnet18_igemm
	./comm_to_cuda_resnet18_igemm

cuda-resnet18-igemm: resnet18_igemm_cuda_output.cu wrap_generated_cuda.py
	python3 wrap_generated_cuda.py resnet18_igemm resnet18_igemm_cuda_output.cu resnet18_igemm.cu
	@echo "ResNet-18 igemm CUDA generated: resnet18_igemm.cu"

resnet18_igemm: resnet18_igemm_cuda_output.cu wrap_generated_cuda.py
	python3 wrap_generated_cuda.py resnet18_igemm resnet18_igemm_cuda_output.cu resnet18_igemm.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ resnet18_igemm.cu

validate-cuda-resnet18-igemm: comm_to_cuda_resnet18_igemm
	./comm_to_cuda_resnet18_igemm
	python3 validate_cuda_resnet18_igemm.py

# G2: Triton backend for LeNet-5 / AlexNet / ResNet-18.  Reuses the
# generic AST recognizers (comm_to_cuda_alexnet.ml / comm_to_cuda_resnet18.ml
# lineage) and emits Triton Python kernels + a forward() launch chain
# (<net>_triton.py).  Padded/strided conv (masked loads), BN /
# residual-Add / GAP kernels are supported.
comm_to_triton: lenet5_sym.cmo alexnet_sym.cmo resnet18_sym.cmo comm_to_triton.ml
	$(OCAMLC) -g -o $@ unix.cma $^

# Ventus OpenCL backend (V2a): comm AST -> OpenCL C conservative subset.
# The binary writes <kname>.cl + manifest.txt to its argv[1] directory.
# End-to-end spike-simulator validation of the emitted chains: see ../ventus/.
comm_to_opencl: lenet5_sym.cmo comm_to_opencl.ml
	$(OCAMLC) -g -o $@ unix.cma $^

triton-lenet5: comm_to_triton
	./comm_to_triton lenet5
	$(PYTHON) run_triton_lenet5.py

triton-alexnet: comm_to_triton
	./comm_to_triton alexnet
	$(PYTHON) run_triton_alexnet.py

triton-resnet18: comm_to_triton
	./comm_to_triton resnet18
	$(PYTHON) run_triton_resnet18.py

# MiniTransformer: sequential + OpenMP printers over the AST extracted
# from MiniTransformer_AST.v.  The *_sym.ml files are produced by
#   python3 transform_to_symbolic.py minitransformer_ast_comm.ml minitransformer_ast_sym_raw.ml
#   python3 decode_reconst_strings.py minitransformer_ast_sym_raw.ml minitransformer_ast_sym.ml
# (decode_reconst_strings.py decodes the non-0/1 REconst literal strings).
minitransformer_ast_sym.cmo: minitransformer_ast_sym.ml
	$(OCAMLC) -c $<

minitransformer_ast_par_sym.cmo: minitransformer_ast_par_sym.ml
	$(OCAMLC) -c $<

comm_to_c_minitransformer: minitransformer_ast_sym.cmo comm_to_c_minitransformer.ml
	$(OCAMLC) -o $@ unix.cma $^

comm_to_c_minitransformer_par: minitransformer_ast_par_sym.cmo comm_to_c_minitransformer_par.ml
	$(OCAMLC) -o $@ unix.cma $^

# Faithful-layout variant (AST Faithful milestone): prints the comm extracted
# from MiniTransformer_AST_Faithful.v (theorem Bridge_MiniTransformer_faithful).
minitransformer_faithful_sym.cmo: minitransformer_faithful_sym.ml
	$(OCAMLC) -c $<

comm_to_c_minitransformer_faithful: minitransformer_faithful_sym.cmo comm_to_c_minitransformer_faithful.ml
	$(OCAMLC) -o $@ unix.cma $^

validate-minitransformer: comm_to_c_minitransformer comm_to_c_minitransformer_par
	./comm_to_c_minitransformer
	./comm_to_c_minitransformer_par
	$(PYTHON) wrap_generated_c.py minitransformer_ast minitransformer_ast_streaming_output.c minitransformer_ast_wrapped.c
	$(PYTHON) wrap_generated_c.py minitransformer_ast_par minitransformer_ast_par_streaming_output.c minitransformer_ast_par_wrapped.c
	$(PYTHON) validate_minitransformer.py
