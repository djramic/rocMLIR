// COM: Run Comgr binary to compile OpenCL source into LLVM IR Bitcode, linking
// COM: against the AMD Device Libraries
// RUN: source-to-bc-with-dev-libs %s -o %t-with-dev-libs.bc

// COM: Dissasemble LLVM IR bitcode to LLVM IR text
// RUN: llvm-dis %t-with-dev-libs.bc -o - | FileCheck %s

// COM: Verify LLVM IR text file
// CHECK: target triple = "amdgcn-amd-amdhsa"
// CHECK: define internal float @_Z4powrff
// CHECK: define internal float @_Z6sincosfPU3AS5f
// CHECK: define internal float @_Z4cbrtf
// CHECK: define internal float @__ocml_sincos_f32
// CHECK: define internal float @__ocml_powr_f32
// CHECK: define internal noundef float @__ocml_exp_f32
// CHECK: define internal ptr addrspace(1) @__printf_alloc

extern const __constant bool __oclc_finite_only_opt;
extern const __constant bool __oclc_unsafe_math_opt;
extern const __constant bool __oclc_correctly_rounded_sqrt32;
extern const __constant bool __oclc_wavefrontsize64;
extern const __constant int __oclc_ISA_version;
extern const __constant int __oclc_ABI_version;

void kernel device_libs(__global float *status) {

  if (__oclc_finite_only_opt)            status[0] = 1.0;
  if (__oclc_unsafe_math_opt)            status[1] = 1.0;
  if (__oclc_correctly_rounded_sqrt32)   status[3] = 1.0;
  if (__oclc_wavefrontsize64)            status[4] = 1.0;
  if (__oclc_ISA_version)                status[5] = 1.0;
  if (__oclc_ABI_version)                status[6] = 1.0;

  // Math functions to test AMDGPULibCalls Folding optimizations
  // fold_sincos()
  float x = 0.25;
  status[7] = sin(x) + cos(x);
  status[8] = cos(x) + sin(x);

  // fold_rootn()
  float y = 725.0;
  status[9] = rootn(y, 3);
  status[10] = rootn(y, -1);
  status[11] = rootn(y, -2);

  // fold_pow()
  float z = 12.16;
  status[12] = pow(z, (float) 0.5);
  status[13] = powr(y, (float) 7.23);

  // printf()
  printf("testy\n");
}
