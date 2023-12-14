#include "mlir/Dialect/Rock/IR/FmaInsnGroup.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#define DEBUG_TYPE "rock-fma-insn-group"

using namespace mlir;
using namespace mlir::rock;

static Type getRetType(Type inputType) {
  Builder b(inputType.getContext());
  if (inputType.isInteger(8))
    return b.getI32Type();

  return b.getF32Type();;
}

FailureOr<FmaInsn> FmaInsn::select(mlir::Type elementTypeA, mlir::Type elementTypeB, StringRef arch ){
  LLVM_DEBUG(llvm::dbgs() << "Invoke FMA group selection:\n"
                          << "elementTypeA: " << elementTypeA << "\n"
                          << "elementTypeB: " << elementTypeB << "\n"
                          << "arch: " << arch << "\n");

  int64_t inputLen = 16; //???
  int64_t outLen = 8; //???

  VectorType argTypeA = VectorType::get({inputLen}, elementTypeA);
  VectorType argTypeB = VectorType::get({inputLen}, elementTypeB);
  Type retType = getRetType(elementTypeA);
  
  return FmaInsn{argTypeA, argTypeB, retType};
}