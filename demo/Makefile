# Ionia tutorial demos — see README.md.
#
#   make demo0   dot product (Linear2d 1x8->1x1)      -> CUDA, vs NumPy
#   make demo1   ReLU4d 1x2x4x4                       -> CUDA, vs NumPy
#   make demo2   Conv4d p1 k3 s1, 1x1x8x8 -> 1x2x8x8  -> CUDA, vs NumPy
#   make demo3   end-to-end small AlexNet (reuses experiments/ast_codegen)
#   make all     demo0 + demo1 + demo2 + demo3
#
# Toolchain: Rocq 9.2.0 (COQC, default: coqc from PATH), ocamlc,
# python3 + numpy, nvcc ($NVCC / PATH / /tmp/cuda-12.9/bin/nvcc).

COQC ?= coqc
IONIA_SRC := ../src/Ionia
AST := ../experiments/ast_codegen

.PHONY: all demo0 demo1 demo2 demo3 coq sym kernels clean

all: demo0 demo1 demo2 demo3

# --- Coq -> OCaml pipeline (demos 0-2) -------------------------------------

coq/demo_comm.ml: coq/Demo_Comms.v
	cd coq && $(COQC) -R ../$(IONIA_SRC) Ionia Demo_Comms.v

coq: coq/demo_comm.ml

ocaml/demo_sym.ml: coq/demo_comm.ml
	python3 $(AST)/transform_to_symbolic.py coq/demo_comm.ml ocaml/demo_sym.ml

sym: ocaml/demo_sym.ml

ocaml/demo_to_cuda: ocaml/demo_sym.ml ocaml/demo_to_cuda.ml
	cd ocaml && ocamlc -g -c demo_sym.ml && \
	  ocamlc -g -o demo_to_cuda unix.cma demo_sym.cmo demo_to_cuda.ml

kernels: ocaml/demo_to_cuda
	mkdir -p build
	cd ocaml && ./demo_to_cuda ../build

# --- Run demos ---------------------------------------------------------------

demo0: kernels
	python3 run_demo0.py

demo1: kernels
	python3 run_demo1.py

demo2: kernels
	python3 run_demo2.py

demo3:
	python3 run_demo3.py

clean:
	rm -f coq/demo_comm.ml coq/demo_comm.mli coq/Demo_Comms.vo \
	      coq/Demo_Comms.vok coq/Demo_Comms.vos coq/Demo_Comms.glob \
	      coq/.Demo_Comms.aux
	rm -f ocaml/demo_sym.ml ocaml/demo_sym.cmi ocaml/demo_sym.cmo \
	      ocaml/demo_to_cuda.cmi ocaml/demo_to_cuda.cmo ocaml/demo_to_cuda
	rm -rf build
