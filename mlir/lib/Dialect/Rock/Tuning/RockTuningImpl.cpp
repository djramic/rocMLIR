//===- RockTuningImpl.cpp - tuning API implementation ----*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file implements the tuning interfaces
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>

namespace mlir {
namespace rock {

// The full space is a brute-force search for attention kernels
void createAttnTuningRangeBF(TuningParamSet *newSpace, AttentionOp attnOp,
                             TuningParamSetKind kind) {
  static const std::vector<std::vector<uint32_t>> validRangeAttnParamsMFMA = {
      /*gemm0MPerBlock=*/{32, 64, 128, 256},
      /*gemm1MPerBlock=*/{32, 64, 128, 256},
      /*gemm0NPerBlock=*/{32, 64, 128, 256},
      /*kPackPerBlock=*/{8, 16, 32, 64},
      /*mPerWave=*/{32, 64, 128, 256},
      /*mnPerXdl=*/{4, 16, 32},
      /*kPack=*/{4, 8, 16}};
  static const std::vector<std::vector<uint32_t>> validRangeAttnParamsWMMA = {
      /*gemm0MPerBlock=*/{32, 64, 128},
      /*gemm1MPerBlock=*/{32, 64, 128},
      /*gemm0NPerBlock=*/{32, 64, 128, 256},
      /*kPackPerBlock=*/{8, 16, 32, 64},
      /*mPerWave=*/{32, 64},
      /*nPerWave=*/{32, 64},
      /*kPack=*/{4, 8, 16}};
  GemmFeatures features = attnOp.getFeatures();
  int64_t numEUPerCU = rock::lookupArchInfo(attnOp.getArch()).numEUPerCU;
  std::vector<std::vector<uint32_t>> validRangeAttnParams;
  bool isWMMA = false;
  if (bitEnumContainsAny(features, GemmFeatures::mfma)) {
    validRangeAttnParams = validRangeAttnParamsMFMA;
  } else if (bitEnumContainsAny(features, GemmFeatures::wmma)) {
    isWMMA = true;
    validRangeAttnParams = validRangeAttnParamsWMMA;
  } else {
    // We only support GPUs with matrix accelerator extentions
    return;
  }
  OpBuilder b(attnOp.getContext());
  for (uint32_t gemm0MPerBlock : validRangeAttnParams[0]) {
    for (uint32_t gemm1MPerBlock : validRangeAttnParams[1]) {
      for (uint32_t gemm0NPerBlock : validRangeAttnParams[2]) {
        for (uint32_t gemmKPerBlock : validRangeAttnParams[3]) {
          for (uint32_t gemmMPerWave : validRangeAttnParams[4]) {
            for (uint32_t gemmMnPerXdlOrNPerWave : validRangeAttnParams[5]) {
              for (uint32_t gemmKPack : validRangeAttnParams[6]) {
                if (isWMMA) {
                  int64_t nPerWave = gemmMnPerXdlOrNPerWave;
                  int64_t rdnaWaves = (gemm0MPerBlock / gemmMPerWave) *
                                      (gemm0NPerBlock / nPerWave);
                  if (rdnaWaves < numEUPerCU) {
                    continue;
                  }
                }
                if (gemm0MPerBlock >= gemmMPerWave &&
                    gemm1MPerBlock >= gemmMPerWave &&
                    gemm1MPerBlock >= gemm0MPerBlock &&
                    gemm0NPerBlock >= gemmMnPerXdlOrNPerWave) {
                  auto params = AttnPerfConfigAttr::get(
                      attnOp.getContext(), gemm0MPerBlock, gemm1MPerBlock,
                      gemm0NPerBlock, gemmKPerBlock, gemmMPerWave,
                      gemmMnPerXdlOrNPerWave, gemmKPack, true);
                  newSpace->tuningRange.push_back(
                      cast<RockTuningParamAttrInterface>(params));
                }
              }
            }
          }
        }
      }
    }
  }
}

double computeWorkImbalance(GemmSize origGemmSize, int32_t gemmMPerBlock,
                            int32_t gemmNPerBlock, int32_t gemmKPerBlock,
                            int32_t kPack, uint32_t numCUs,
                            int32_t splitKFactor = 1) {
  const InitParams params{gemmMPerBlock, gemmNPerBlock, gemmKPerBlock};
  const GemmSize gemmSize =
      calculatePaddedGemmSize(params, origGemmSize, kPack);
  const auto numMTiles = (gemmSize.m + gemmMPerBlock - 1) / gemmMPerBlock;
  const auto numNTiles = (gemmSize.n + gemmNPerBlock - 1) / gemmNPerBlock;

  const double totalNumWorkGroups =
      gemmSize.g * numMTiles * numNTiles * splitKFactor;
  const double maxWorkGroupsPerCU = std::ceil(totalNumWorkGroups / numCUs);
  // imbalances = max. CU work / average work per CU
  return (maxWorkGroupsPerCU * numCUs) / totalNumWorkGroups;
}

static SmallVector<int64_t>
computeOptimalSplitKFactors(GemmSize origGemmSize, int32_t gemmMPerBlock,
                            int32_t gemmNPerBlock, int32_t gemmKPerBlock,
                            int32_t kPack, uint32_t numCUs) {
  SmallVector<int64_t> splitKValues = {1};

  const auto dataParallelGemmImbalance = computeWorkImbalance(
      origGemmSize, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock, kPack, numCUs);

  constexpr double imbalaceThreshold = 1.20;
  if (dataParallelGemmImbalance < imbalaceThreshold) {
    return splitKValues;
  }

  struct LocalData {
    int64_t splitKValue = 0;
    double workImbalance = 0.0;
  };
  SmallVector<LocalData> factors;
  constexpr double minGain = 1.30;
  // A large set of splitK values significantly increases tuning time,
  // after analysis, we've determined that using only splitK factors 3 and 4 is
  // sufficient.
  for (int64_t splitKFactor : {3, 4}) {
    const double imbalance =
        computeWorkImbalance(origGemmSize, gemmMPerBlock, gemmNPerBlock,
                             gemmKPerBlock, kPack, numCUs, splitKFactor);
    const auto gain = dataParallelGemmImbalance / imbalance;
    if (gain > minGain) {
      factors.emplace_back(LocalData{splitKFactor, imbalance});
    }
  }

  if (factors.empty()) {
    return splitKValues;
  }

  llvm::sort(factors.rbegin(), factors.rend(), [](LocalData &a, LocalData &b) {
    return a.workImbalance < b.workImbalance;
  });

  llvm::ArrayRef<LocalData> view(factors.data(), factors.size());
  llvm::for_each(view, [&](const LocalData &item) {
    splitKValues.push_back(item.splitKValue);
  });

  return splitKValues;
}

static SmallVector<int64_t>
computeOptimalSplitKFactors(RockGemmWrapperInterface gemmOp,
                            int32_t gemmMPerBlock, int32_t gemmNPerBlock,
                            int32_t gemmKPerBlock, int32_t kPack) {
  auto info = PopulateParamsInfo::fromOp(gemmOp);
  SmallVector<int64_t> splitKValues = {1};
  GemmFeatures currentFeatures = gemmOp.getGemmFeatures();
  // We dont enable split-k on Navi yet because they dont
  // still have atomic_add with packed_f16.
  if (bitEnumContainsAll(currentFeatures, GemmFeatures::wmma)) {
    return splitKValues;
  }
  const bool isAllowedTypeC =
      gemmOp.getCType().isF32() || gemmOp.getCType().isF16();
  if (!isAllowedTypeC) {
    return splitKValues;
  }

  auto func = cast<func::FuncOp>(gemmOp->getParentOp());
  if (!func->hasAttr(rock::EnableSplitKForTuningAttr::getMnemonic())) {
    return splitKValues;
  }

  uint32_t numCUs = rock::lookupArchInfo(gemmOp.getArch()).minNumCU;
  if (gemmOp.getNumCU().has_value()) {
    numCUs = gemmOp.getNumCU().value();
  }

  return computeOptimalSplitKFactors(info.gemmSize, gemmMPerBlock,
                                     gemmNPerBlock, gemmKPerBlock, kPack,
                                     numCUs);
}

// The full space is a brute-force search starting with the configs that have
// the smallest parameters. This filters out perf configs that are
// known to be impossible during tthe AffixTuningParams check.
// If `kind` is Full, also filters out unlikely-to-be-good configurations.
void createGemmTuningRangeBF(TuningParamSet *newSpace,
                             RockGemmWrapperInterface gemmOp,
                             TuningParamSetKind kind) {
  auto info = PopulateParamsInfo::fromOp(gemmOp);

  // blockSize M/block N/block K/block M/thread N/thread
  const std::vector<std::vector<uint32_t>> validRangeGeneralGemmParams = {
      {64, 128, 256}, {32, 64, 128}, {32, 64, 128}, {4, 8, 16}, {2, 4}, {2, 4}};

  // M/block N/block K/block M/wave N/wave kPack aCopyMore/forceUnroll
  const std::vector<std::vector<uint32_t>> validRangeAccelGemmParams = {
      {4, 8, 16, 32, 64, 128, 256},
      {16, 32, 64, 128, 256},
      {1, 2, 4, 8},
      {4, 8, 16, 32, 64, 128},
      {4, 16, 32},
      {1, 4, 8},
      {0, 1}};

  // M/block N/block K/block M/wave N/wave kPack aCopyMore/forceUnroll
  const std::vector<std::vector<uint32_t>>
      validRangeAccelGemmParams8BitReduction = {{4, 8, 16, 32, 64, 128, 256},
                                                {16, 32, 64, 128, 256},
                                                {4, 8, 16, 32},
                                                {4, 8, 16, 32, 64, 128},
                                                {4, 8, 16, 32, 64, 128},
                                                {1, 4, 8, 16},
                                                {0, 1}};

  // M/block N/block K/block M/wave N/wave kPack aCopyMore/forceUnroll
  const std::vector<std::vector<uint32_t>> validRangeWmmaGemmParams = {
      {4, 8, 16, 32, 64, 128, 256},
      {16, 32, 64, 128, 256},
      {1, 2, 4, 8},
      {4, 8, 16, 32, 64, 128},
      {4, 8, 16, 32, 64, 128},
      {4, 8, 16},
      {0, 1}};

  OpBuilder b(gemmOp.getContext());
  GemmFeatures currentFeatures = gemmOp.getGemmFeatures();
  if (bitEnumContainsAll(currentFeatures, GemmFeatures::mfma)) {
    PopulateParamsXDL tuningInfo;
    // XDLOPS
    Type inTypeA = gemmOp.getAType();
    bool is8BitReduction = inTypeA.isInteger(8) || inTypeA.isFloat8E5M2FNUZ() ||
                           inTypeA.isFloat8E4M3FNUZ() ||
                           inTypeA.isFloat8E5M2() || inTypeA.isFloat8E4M3FN();
    const std::vector<std::vector<uint32_t>> &xdlopsParams =
        is8BitReduction ? validRangeAccelGemmParams8BitReduction
                        : validRangeAccelGemmParams;
    for (uint32_t gemmMPerBlock : xdlopsParams[0]) {
      for (uint32_t gemmNPerBlock : xdlopsParams[1]) {
        for (uint32_t gemmKPerBlock : xdlopsParams[2]) {
          for (uint32_t gemmMPerWave : xdlopsParams[3]) {
            for (uint32_t gemmMnPerXdl : xdlopsParams[4]) {
              for (uint32_t gemmKPack : xdlopsParams[5]) {
                auto optimalSplitKFactors = computeOptimalSplitKFactors(
                    gemmOp, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock,
                    gemmKPack);
                for (int64_t splitKFactor : optimalSplitKFactors) {
                  for (uint32_t forceUnroll : xdlopsParams[6]) {
                    InitParamsAccel gemmParams(gemmMPerBlock, gemmNPerBlock,
                                               gemmKPerBlock, gemmMPerWave,
                                               gemmMnPerXdl, gemmKPack,
                                               splitKFactor, forceUnroll, true);
                    if (gemmMPerBlock >= gemmMPerWave &&
                        gemmNPerBlock >= gemmMnPerXdl) {
                      if (succeeded(tuningInfo.paramsProbablyValid(
                              b, info, gemmParams)) &&
                          (kind == TuningParamSetKind::Exhaustive ||
                           succeeded(
                               tuningInfo.couldBePerformant(info, gemmParams))))
                        newSpace->tuningRange.push_back(
                            cast<RockTuningParamAttrInterface>(
                                tuningInfo.getGemmParamsAttr(b, gemmParams)));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  } else if (bitEnumContainsAll(currentFeatures, GemmFeatures::wmma)) {
    // Wmma
    const std::vector<std::vector<uint32_t>> &wmmaParams =
        validRangeWmmaGemmParams;
    PopulateParamsWmma tuningInfo;
    for (uint32_t gemmMPerBlock : wmmaParams[0]) {
      for (uint32_t gemmNPerBlock : wmmaParams[1]) {
        for (uint32_t gemmKPerBlock : wmmaParams[2]) {
          for (uint32_t gemmMPerWave : wmmaParams[3]) {
            for (uint32_t gemmNPerWave : wmmaParams[4]) {
              for (uint32_t gemmKPack : wmmaParams[5]) {
                auto optimalSplitKFactors = computeOptimalSplitKFactors(
                    gemmOp, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock,
                    gemmKPack);
                for (auto splitKFactor : optimalSplitKFactors) {
                  for (uint32_t forceUnroll : wmmaParams[6]) {
                    InitParamsAccel gemmParams(gemmMPerBlock, gemmNPerBlock,
                                               gemmKPerBlock, gemmMPerWave,
                                               gemmNPerWave, gemmKPack,
                                               splitKFactor, forceUnroll, true);
                    if (succeeded(tuningInfo.paramsProbablyValid(b, info,
                                                                 gemmParams)) &&
                        (kind == TuningParamSetKind::Exhaustive ||
                         succeeded(
                             tuningInfo.couldBePerformant(info, gemmParams))))
                      newSpace->tuningRange.push_back(
                          cast<RockTuningParamAttrInterface>(
                              tuningInfo.getGemmParamsAttr(b, gemmParams)));
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    // Non-XDLOPS
    PopulateParams tuningInfo;
    for (uint32_t blockSize : validRangeGeneralGemmParams[0]) {
      for (uint32_t gemmMPerBlock : validRangeGeneralGemmParams[1]) {
        for (uint32_t gemmNPerBlock : validRangeGeneralGemmParams[2]) {
          for (uint32_t gemmKPerBlock : validRangeGeneralGemmParams[3]) {
            for (uint32_t gemmMPerThread : validRangeGeneralGemmParams[4]) {
              auto optimalSplitKFactors = computeOptimalSplitKFactors(
                  gemmOp, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock, 1);
              for (auto splitKFactor : optimalSplitKFactors) {
                for (uint32_t gemmNPerThread : validRangeGeneralGemmParams[5]) {
                  InitParamsNonAccel gemmParams(
                      blockSize, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock,
                      gemmMPerThread, gemmNPerThread, splitKFactor);
                  if (succeeded(tuningInfo.paramsProbablyValid(b, info,
                                                               gemmParams)) &&
                      (kind == TuningParamSetKind::Exhaustive ||
                       succeeded(
                           tuningInfo.couldBePerformant(info, gemmParams))))
                    newSpace->tuningRange.push_back(
                        cast<RockTuningParamAttrInterface>(
                            tuningInfo.getGemmParamsAttr(b, gemmParams)));
                }
              }
            }
          }
        }
      }
    }
  }
}

void createQuickTuningRange(TuningParamSet *newSpace,
                            RockGemmWrapperInterface gemmOp) {
  auto info = PopulateParamsInfo::fromOp(gemmOp);
  OpBuilder b(gemmOp.getContext());
  GemmFeatures currentFeatures = gemmOp.getGemmFeatures();
  if (bitEnumContainsAll(currentFeatures, GemmFeatures::mfma)) {
    PopulateParamsXDL tuningInfo;

    for (InitParamsAccel param : tuningInfo.orderInitParams(
             tuningInfo.getTuningParameters(info.kernelType, info.gemmAType,
                                            info.gemmBType, info.arch),
             info.gemmSize)) {
      if (succeeded(tuningInfo.paramsProbablyValid(b, info, param)) &&
          succeeded(tuningInfo.couldBePerformant(info, param)))
        newSpace->tuningRange.push_back(cast<RockTuningParamAttrInterface>(
            tuningInfo.getGemmParamsAttr(b, param)));
    }
  } else if (bitEnumContainsAll(currentFeatures, GemmFeatures::wmma)) {
    // Wmma
    PopulateParamsWmma tuningInfo;
    for (InitParamsAccel param : tuningInfo.orderInitParams(
             tuningInfo.getTuningParameters(info.kernelType, info.gemmAType,
                                            info.gemmBType, info.arch),
             info.gemmSize)) {
      if (succeeded(tuningInfo.paramsProbablyValid(b, info, param)) &&
          succeeded(tuningInfo.couldBePerformant(info, param)))
        newSpace->tuningRange.push_back(cast<RockTuningParamAttrInterface>(
            tuningInfo.getGemmParamsAttr(b, param)));
    }
  } else {
    // Non-XDLOPS
    PopulateParams tuningInfo;
    for (InitParamsNonAccel param : tuningInfo.orderInitParams(
             tuningInfo.getTuningParameters(info.kernelType, info.gemmAType,
                                            info.gemmBType),
             info.gemmSize)) {
      if (succeeded(tuningInfo.paramsProbablyValid(b, info, param)) &&
          succeeded(tuningInfo.couldBePerformant(info, param)))
        newSpace->tuningRange.push_back(cast<RockTuningParamAttrInterface>(
            tuningInfo.getGemmParamsAttr(b, param)));
    }
  }
}

// This is temporary workaround to make MIGraphX integration
// work until the tuning is setup for attention ops properly.
void createAttnTuningRangeQuick(TuningParamSet *newSpace, AttentionOp attnOp) {
  OpBuilder b(attnOp.getContext());
  GemmFeatures currentFeatures = attnOp.getFeatures();
  Type elemType =
      cast<ShapedType>(attnOp.getQueries().getType()).getElementType();
  // g0Mpb, g1Mpb, g0Npb, Kpb, mPw, mnPxdl, kpack
  typedef std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                     int64_t>
      PerfConfigVals;
  if (bitEnumContainsAll(currentFeatures, GemmFeatures::mfma)) {
    const SmallVector<PerfConfigVals, 7> attnQuickTuningListMFMAF16{
        PerfConfigVals{32, 128, 128, 32, 32, 32, 4},
        PerfConfigVals{64, 64, 32, 16, 32, 16, 4},
        PerfConfigVals{32, 64, 64, 16, 32, 16, 4},
        PerfConfigVals{32, 64, 128, 16, 32, 16, 4},
        PerfConfigVals{64, 64, 64, 16, 32, 16, 4},
        PerfConfigVals{64, 64, 64, 16, 32, 32, 4}};
    const SmallVector<PerfConfigVals, 7> attnQuickTuningListMFMAF32{
        PerfConfigVals{32, 128, 64, 32, 32, 16, 4},
        PerfConfigVals{32, 64, 64, 32, 32, 16, 4},
        PerfConfigVals{32, 128, 128, 32, 32, 32, 4},
        PerfConfigVals{64, 64, 32, 16, 32, 16, 4},
        PerfConfigVals{32, 64, 64, 16, 32, 16, 4},
        PerfConfigVals{32, 64, 128, 16, 32, 32, 4},
        PerfConfigVals{64, 64, 64, 16, 32, 32, 4}};
    ArrayRef<PerfConfigVals> attnQuickTuningListMFMA =
        attnQuickTuningListMFMAF32;
    if (elemType.isF16()) {
      attnQuickTuningListMFMA = attnQuickTuningListMFMAF16;
    }
    for (auto [mPerBlockG0, mPerBlockG1, nPerBlockG0, kPackBerBlock, mPerWave,
               mnPerXdl, kPack] : attnQuickTuningListMFMA) {
      auto params = AttnPerfConfigAttr::get(
          attnOp.getContext(), mPerBlockG0, mPerBlockG1, nPerBlockG0,
          kPackBerBlock, mPerWave, mnPerXdl, kPack, true);
      newSpace->tuningRange.push_back(
          cast<RockTuningParamAttrInterface>(params));
    }
  } else if (bitEnumContainsAll(currentFeatures, GemmFeatures::wmma)) {
    const SmallVector<PerfConfigVals, 7> attnQuickTuningListWMMA{
        PerfConfigVals{64, 128, 128, 8, 32, 32, 4},
        PerfConfigVals{64, 64, 256, 8, 64, 32, 8},
        PerfConfigVals{64, 64, 256, 16, 32, 32, 8},
        PerfConfigVals{64, 64, 32, 8, 32, 32, 4},
        PerfConfigVals{32, 64, 128, 8, 32, 32, 8},
        PerfConfigVals{64, 64, 128, 8, 64, 32, 8},
        PerfConfigVals{32, 32, 128, 8, 32, 32, 8},
        PerfConfigVals{128, 128, 128, 8, 32, 32, 8}};
    for (auto [mPerBlockG0, mPerBlockG1, nPerBlockG0, kPackBerBlock, mPerWave,
               mnPerXdl, kPack] : attnQuickTuningListWMMA) {
      auto params = AttnPerfConfigAttr::get(
          attnOp.getContext(), mPerBlockG0, mPerBlockG1, nPerBlockG0,
          kPackBerBlock, mPerWave, mnPerXdl, kPack, true);
      newSpace->tuningRange.push_back(
          cast<RockTuningParamAttrInterface>(params));
    }
  }
  // We only support GPUs with matrix accelerator extentions
}

TuningParamSet *createTunableParamSpace(ModuleOp mod, TuningParamSetKind kind) {
  struct TuningParamSet *newSpace;
  newSpace = new TuningParamSet();

  // create range and heuristic
  WalkResult findPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        switch (kind) {
        case TuningParamSetKind::Full:
        case TuningParamSetKind::Exhaustive:
          createGemmTuningRangeBF(newSpace, op, kind);
          break;
        case TuningParamSetKind::Quick:
          createQuickTuningRange(newSpace, op);
          break;
        }
        newSpace->primaryOpType = op.getKernelType();
        return WalkResult::interrupt();
      });
  WalkResult findAttention = mod->walk([&](rock::AttentionOp op) -> WalkResult {
    switch (kind) {
    case TuningParamSetKind::Full:
    case TuningParamSetKind::Exhaustive:
      createAttnTuningRangeBF(newSpace, op, kind);
      break;
    case TuningParamSetKind::Quick:
      createAttnTuningRangeQuick(newSpace, op);
    }
    return WalkResult::interrupt();
  });
  if (!findPrimary.wasInterrupted() && !findAttention.wasInterrupted()) {
    llvm::report_fatal_error(
        "Expected to find GEMM, convolution, or attention op, and didn't.");
  }
  return newSpace;
}

bool tuningGetParam(TuningParamSet *tuningSpace, unsigned pos,
                    ParamEntry *paramEntry) {
  // out of bound check.
  if (pos > tuningSpace->tuningRange.size() - 1)
    return false;
  paramEntry->param = tuningSpace->tuningRange[pos];
  return true;
}

bool tuningSetParam(ModuleOp &mod, ParamEntry *paramEntry) {
  WalkResult setPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        auto *ctx = op.getContext();
        SmallString<64> perfConfig;
        paramEntry->param.getPerfConfigStr(perfConfig);
        StringAttr attr = StringAttr::get(ctx, perfConfig);
        op->setAttr("perf_config", attr);
        return WalkResult::interrupt();
      });
  WalkResult setAttn = mod->walk([&](rock::AttentionOp op) -> WalkResult {
    auto *ctx = op.getContext();
    SmallString<64> perfConfig;
    paramEntry->param.getPerfConfigStr(perfConfig);
    StringAttr attr = StringAttr::get(ctx, perfConfig);
    op->setAttr("perf_config", attr);
    return WalkResult::interrupt();
  });
  return setPrimary.wasInterrupted() || setAttn.wasInterrupted();
}

bool tuningSetStr(ModuleOp &mod, StringRef perfConfig) {
  WalkResult setPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        auto *ctx = op.getContext();
        StringAttr attr = StringAttr::get(ctx, perfConfig);
        op->setAttr("perf_config", attr);
        return WalkResult::interrupt();
      });
  WalkResult setAttn = mod->walk([&](rock::AttentionOp op) -> WalkResult {
    auto *ctx = op.getContext();
    StringAttr attr = StringAttr::get(ctx, perfConfig);
    op->setAttr("perf_config", attr);
    return WalkResult::interrupt();
  });
  return setPrimary.wasInterrupted() || setAttn.wasInterrupted();
}

TuningTable *tuningTableCreate() {
  struct TuningTable *newTable = new TuningTable();
  return newTable;
}

LogicalResult getTuningProblemStr(rock::AttentionOp attnOp,
                                  SmallVectorImpl<char> &out) {
  int32_t numCU = rock::lookupArchInfo(attnOp.getArch()).minNumCU;
  if (attnOp.getNumCU().has_value()) {
    numCU = attnOp.getNumCU().value();
  }
  constexpr char sep = ' ';
  constexpr char tab = '\t';
  int64_t headDimQK;
  int64_t headDimV;
  int64_t seqLenQ;
  int64_t seqLenK;
  llvm::raw_svector_ostream problemOS(out);
  // ARCH string
  problemOS << attnOp.getArch() << tab;
  // Num of Compute Units
  problemOS << numCU << tab;

  TypedValue<ShapedType> queries = attnOp.getQueries();
  TypedValue<ShapedType> keys = attnOp.getKeys();
  TypedValue<ShapedType> values = attnOp.getValues();
  ArrayRef<int64_t> qShape = queries.getType().getShape();
  ArrayRef<int64_t> kShape = keys.getType().getShape();
  ArrayRef<int64_t> vShape = values.getType().getShape();
  int64_t g = qShape[0];

  Type elemTypeQ = queries.getType().getElementType();
  problemOS << "-t ";
  if (elemTypeQ.isF32()) {
    problemOS << "f32" << sep;
  } else if (elemTypeQ.isF16()) {
    problemOS << "f16" << sep;
  } else if (elemTypeQ.isInteger(8)) {
    problemOS << "i8" << sep;
  } else {
    return attnOp.emitError("invalid type:") << elemTypeQ << "\n";
  }

  // TransQ
  problemOS << "-transQ ";
  if (attnOp.getQTransposed()) {
    seqLenQ = qShape[2];
    headDimQK = qShape[1];
    problemOS << "true" << sep;
  } else {
    seqLenQ = qShape[1];
    headDimQK = qShape[2];
    problemOS << "false" << sep;
  }

  // TransK
  problemOS << "-transK ";
  if (attnOp.getKTransposed()) {
    seqLenK = kShape[1];
    problemOS << "true" << sep;
  } else {
    seqLenK = kShape[2];
    problemOS << "false" << sep;
  }

  // TransV
  problemOS << "-transV ";
  if (attnOp.getVTransposed()) {
    headDimV = vShape[1];
    problemOS << "true" << sep;
  } else {
    headDimV = vShape[2];
    problemOS << "false" << sep;
  }

  // TransO
  problemOS << "-transO ";
  if (attnOp.getOTransposed())
    problemOS << "true" << sep;
  else
    problemOS << "false" << sep;

  problemOS << "-g " << g << sep;
  problemOS << "-seq_len_q " << seqLenQ << sep;
  problemOS << "-seq_len_k " << seqLenK << sep;
  problemOS << "-head_dim_qk " << headDimQK << sep;
  problemOS << "-head_dim_v " << headDimV;
  return success();
}

LogicalResult getTuningProblemStr(rock::RockGemmWrapperInterface gemmIF,
                                  SmallVectorImpl<char> &out) {
  int32_t numCU = rock::lookupArchInfo(gemmIF.getArch()).minNumCU;
  if (gemmIF.getNumCU().has_value())
    numCU = gemmIF.getNumCU().value();
  constexpr char sep = ' ';
  constexpr char tab = '\t';
  llvm::raw_svector_ostream problemOS(out);

  KernelType opType = gemmIF.getKernelType();
  Operation *gemmOp = gemmIF.getOperation();

  auto f8TypeStr = [](const Type &type) -> std::optional<StringLiteral> {
    if (type.isFloat8E4M3FNUZ() || type.isFloat8E4M3FN())
      return StringLiteral("fp8");
    if (type.isFloat8E5M2FNUZ() || type.isFloat8E5M2())
      return StringLiteral("bf8");
    return std::nullopt;
  };

  // ARCH string
  problemOS << gemmIF.getArch() << tab;
  // Num of Compute Units
  problemOS << numCU << tab;

  if (opType == KernelType::Conv || opType == KernelType::ConvBwdData ||
      opType == KernelType::ConvBwdWeight) { // conv cases
    RockConvInterface convIF = dyn_cast<RockConvInterface>(gemmOp);

    ShapedType inType = convIF.getInput().getType();
    ArrayRef<int64_t> inShape = inType.getShape();
    ShapedType filType = convIF.getFilter().getType();
    ArrayRef<int64_t> filShape = filType.getShape();

    // Extract layout information
    auto filterLayoutAttr =
        gemmOp->template getAttrOfType<ArrayAttr>("filter_layout");
    auto inputLayoutAttr =
        gemmOp->template getAttrOfType<ArrayAttr>("input_layout");
    auto outputLayoutAttr =
        gemmOp->template getAttrOfType<ArrayAttr>("output_layout");

    unsigned size = filterLayoutAttr.size();
    llvm::StringMap<unsigned> fLayoutMap;
    llvm::StringMap<unsigned> iLayoutMap;
    llvm::StringMap<unsigned> oLayoutMap;

    for (unsigned i = 0; i < size; ++i) {
      auto filterAttr = cast<StringAttr>(filterLayoutAttr.getValue()[i]);
      StringRef fKey = filterAttr.getValue();
      if (fKey == "y")
        fKey = "0";
      if (fKey == "x")
        fKey = "1";
      fLayoutMap[fKey] = i;
      auto inputAttr = cast<StringAttr>(inputLayoutAttr.getValue()[i]);
      StringRef iKey = inputAttr.getValue();
      if (iKey == "hi")
        iKey = "0i";
      if (iKey == "wi")
        iKey = "1i";
      iLayoutMap[iKey] = i;
      auto outputAttr = cast<StringAttr>(outputLayoutAttr.getValue()[i]);
      StringRef oKey = outputAttr.getValue();
      if (oKey == "ho")
        oKey = "0o";
      if (oKey == "wo")
        oKey = "1o";
      oLayoutMap[oKey] = i;
    }

    SmallString<6> fLayout;
    SmallString<6> iLayout;
    SmallString<6> oLayout;
    fLayout.assign(size, '#');
    iLayout.assign(size, '#');
    oLayout.assign(size, '#');

    // dimensions need to be mapped 1 to 1.
    fLayout[fLayoutMap["k"]] = 'N';
    fLayout[fLayoutMap["c"]] = 'C';
    fLayout[fLayoutMap["g"]] = 'G';
    iLayout[iLayoutMap["ni"]] = 'N';
    iLayout[iLayoutMap["ci"]] = 'C';
    iLayout[iLayoutMap["gi"]] = 'G';
    oLayout[oLayoutMap["no"]] = 'N';
    oLayout[oLayoutMap["ko"]] = 'C';
    oLayout[oLayoutMap["go"]] = 'G';

    for (unsigned i = 0; i < size - 3; i++) {
      std::string key = std::to_string(i);
      char val = '0' + i;
      fLayout[fLayoutMap[key]] = val;
      iLayout[iLayoutMap[key + "i"]] = val;
      oLayout[oLayoutMap[key + "o"]] = val;
    }

    if (llvm::any_of(llvm::concat<const char>(fLayout, iLayout, oLayout),
                     [](const char c) { return c == '#'; })) {
      return failure();
    }

    // Please keep these in sync with mlir/utils/performance/perfRunner.py

    // OP datatype
    Type inElemType = inType.getElementType();
    Type filElemType = filType.getElementType();
    if (inElemType.isF32()) {
      problemOS << "conv ";
    } else if (inElemType.isF16()) {
      problemOS << "convfp16 ";
    } else if (inElemType.isBF16()) {
      problemOS << "convbfp16 ";
    } else if (inElemType.isInteger(8)) {
      problemOS << "convint8 ";
    } else {
      auto inString = f8TypeStr(inElemType);
      auto filString = f8TypeStr(filElemType);
      if (inString && filString)
        problemOS << llvm::formatv("conv{0}_{1} ", *inString, *filString);
      else
        return failure();
    }

    // OP direction
    switch (opType) {
    case KernelType::Conv:
      problemOS << "-F 1" << sep;
      break;
    case KernelType::ConvBwdData:
      problemOS << "-F 2" << sep;
      break;
    case KernelType::ConvBwdWeight:
      problemOS << "-F 4" << sep;
      break;
    default:
      return failure();
    }

    // filter layout
    problemOS << "-f " << fLayout << sep;
    // input layout
    problemOS << "-I " << iLayout << sep;
    // output layout
    problemOS << "-O " << oLayout << sep;
    // N
    problemOS << "-n " << inShape[iLayoutMap["ni"]] << sep;
    // C
    problemOS << "-c " << inShape[iLayoutMap["ci"]] << sep;
    // H
    problemOS << "-H " << inShape[iLayoutMap["0i"]] << sep;
    // W
    problemOS << "-W " << inShape[iLayoutMap["1i"]] << sep;
    // K
    problemOS << "-k " << filShape[fLayoutMap["k"]] << sep;
    // Y
    problemOS << "-y " << filShape[fLayoutMap["0"]] << sep;
    // X
    problemOS << "-x " << filShape[fLayoutMap["1"]] << sep;

    auto paddingVal = extractFromIntegerArrayAttr<int64_t>(convIF.getPadding());
    auto strideVal = extractFromIntegerArrayAttr<int64_t>(convIF.getStrides());
    auto dilationVal =
        extractFromIntegerArrayAttr<int64_t>(convIF.getDilations());
    // padding
    problemOS << "-p " << paddingVal[0] << " -q " << paddingVal[2] << sep;
    // stride
    problemOS << "-u " << strideVal[0] << " -v " << strideVal[1] << sep;
    // dilation
    problemOS << "-l " << dilationVal[0] << " -j " << dilationVal[1] << sep;
    // group
    problemOS << "-g " << inShape[iLayoutMap["gi"]] << sep;

  } else if (opType == KernelType::Gemm) { // gemm case
    rock::GemmOp rGemmOp = dyn_cast<rock::GemmOp>(gemmOp);
    // Please keep these in sync with mlir/utils/performance/perfRunner.py
    // Data type
    problemOS << "-t ";
    Type elemTypeA = gemmIF.getAType(), elemTypeB = gemmIF.getBType();
    if (elemTypeA.isF32() && elemTypeB.isF32()) {
      problemOS << "f32";
    } else if (elemTypeA.isF16() && elemTypeB.isF16()) {
      problemOS << "f16";
    } else if (elemTypeA.isBF16() && elemTypeB.isBF16()) {
      problemOS << "bf16";
    } else if (elemTypeA.isInteger(8) && elemTypeB.isInteger(8)) {
      problemOS << "i8";
    } else {
      auto aString = f8TypeStr(elemTypeA);
      auto bString = f8TypeStr(elemTypeB);
      if (aString && bString)
        problemOS << llvm::formatv("{0}_{1}", *aString, *bString);
      else
        return failure();
    }

    // Output datatype
    auto outType = gemmIF.getOutArgument()->get().getType();
    auto elemTypeC = dyn_cast<mlir::MemRefType>(outType).getElementType();
    problemOS << " -out_datatype ";
    auto outStr = f8TypeStr(elemTypeC);
    if (outStr)
      problemOS << *outStr << sep;
    else
      problemOS << elemTypeC << sep;

    // TransA
    problemOS << "-transA ";
    if (rGemmOp.getATransposed())
      problemOS << "true ";
    else
      problemOS << "false ";

    // TransB
    problemOS << "-transB ";
    if (rGemmOp.getBTransposed())
      problemOS << "true ";
    else
      problemOS << "false ";

    // Gemmsize G/M/N/K
    problemOS << "-g " << gemmIF.getGemmSize().g << sep;
    problemOS << "-m " << gemmIF.getGemmSize().m << sep;
    problemOS << "-n " << gemmIF.getGemmSize().n << sep;
    problemOS << "-k " << gemmIF.getGemmSize().k << sep;
  } else {
    // Unknown op type, unreachable.
    return failure();
  }

  while (out.back() == sep) {
    // remove trailing whitespace
    out.pop_back();
  }

  return success();
}

// Suppose to return the structure of the given problem to tune, currently
// combines the string representation of the selected field of the primary
// operation. String format of the problem will not be required by the DB,
// since it can store each field separately.
// Currently serialize the problem in MIOpenDriver command friendly format
LogicalResult getTuningProblemStr(ModuleOp mod, SmallVectorImpl<char> &out) {
  {
    rock::RockGemmWrapperInterface gemmIF;
    WalkResult findPrimary =
        mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
          gemmIF = op;
          return WalkResult::interrupt();
        });
    if (findPrimary.wasInterrupted())
      return getTuningProblemStr(gemmIF, out);
  }
  {
    rock::AttentionOp attnOp;
    WalkResult findAttention =
        mod->walk([&](rock::AttentionOp op) -> WalkResult {
          attnOp = op;
          return WalkResult::interrupt();
        });
    if (findAttention.wasInterrupted())
      return getTuningProblemStr(attnOp, out);
  }
  return failure();
}

bool tuningTableUpdate(TuningTable *perfTable, StringRef problem,
                       StringRef perfConfig, float time) {
  if (problem.empty())
    return false;
  llvm::sys::SmartScopedWriter<true> guard(perfTable->lock);
  auto search = perfTable->tuningMap.find(problem);
  if (search != perfTable->tuningMap.end()) {
    auto entry = perfTable->tuningMap[problem];
    if (entry.second <= time) {
      return false;
    }
  }
  perfTable->tuningMap[problem] = std::make_pair(perfConfig, time);
  return true;
}

LogicalResult tuningTableLookup(TuningTable *perfTable, ModuleOp &mod,
                                SmallVectorImpl<char> &out) {
  SmallString<2048> problem;
  if (failed(getTuningProblemStr(mod, problem)))
    return failure();
  llvm::sys::SmartScopedReader<true> guard(perfTable->lock);
  auto search = perfTable->tuningMap.find(problem);
  if (search != perfTable->tuningMap.end()) {
    auto entry = perfTable->tuningMap[problem];
    out.assign(entry.first);
    return success();
  }
  return failure();
}

template <typename ParamType>
int64_t retrieveSplitKValueImpl(StringRef perfConfig) {
  ParamType params;
  params.deserialize(perfConfig.str());
  return params.splitKFactor;
}

int64_t retrieveSplitKValue(rock::RockGemmWrapperInterface op,
                            StringRef perfConfig) {
  rock::GemmFeatures features = op.getGemmFeatures();
  if (isAccel(features)) {
    return retrieveSplitKValueImpl<rock::InitParamsAccel>(perfConfig);
  }
  return retrieveSplitKValueImpl<rock::InitParamsNonAccel>(perfConfig);
}

bool isSplitKRequested(ModuleOp mod, StringRef perfConfig) {
  WalkResult gemmWalkResult =
      mod.walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        int64_t splitKFactor = retrieveSplitKValue(op, perfConfig);
        if (splitKFactor > 1) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

  return gemmWalkResult.wasInterrupted();
}

RocmlirSplitKSelectionLikelihood isSplitKFaster(int64_t gDim, int64_t mDim,
                                                int64_t nDim, int64_t kDim,
                                                int64_t numCUs) {

  // Note, the following values are aggregated from `createGemmTuningRangeBF`,
  // see above.
  // M/block N/block K/block M/wave N/wave kPack
  const std::vector<std::vector<uint32_t>> rangeGemmParams = {
      {4, 8, 16, 32, 64, 128, 256},
      {16, 32, 64, 128, 256},
      {1, 2, 4, 8},
      {1, 4, 8, 16}};

  rock::GemmSize gemmSize(gDim, mDim, kDim, nDim);
  llvm::SmallSetVector<int64_t, 10> splitKValues = {};
  double minWorkImbalance = std::numeric_limits<double>::max();
  for (uint32_t mPerBlock : rangeGemmParams[0]) {
    for (uint32_t nPerBlock : rangeGemmParams[1]) {
      for (uint32_t kPerBlock : rangeGemmParams[2]) {
        for (uint32_t kPack : rangeGemmParams[3]) {
          const double currWorkImbalance = computeWorkImbalance(
              gemmSize, mPerBlock, nPerBlock, kPerBlock, kPack, numCUs);
          minWorkImbalance = std::min(currWorkImbalance, minWorkImbalance);

          llvm::SmallVector<int64_t> currSplitKValues =
              computeOptimalSplitKFactors(gemmSize, mPerBlock, nPerBlock,
                                          kPerBlock, kPack, numCUs);
          llvm::for_each(currSplitKValues, [&splitKValues](int64_t value) {
            splitKValues.insert(value);
          });
        }
      }
    }
  }

  if (splitKValues.size() == 1) {
    return RocmlirSplitKSelectionLikelihood::never;
  }

  // TODO[split-K]: one needs to validate whether
  // 1.8 threshold is a resonable choice
  constexpr double workImbalanceThreshold{1.8};
  if (minWorkImbalance > workImbalanceThreshold) {
    return RocmlirSplitKSelectionLikelihood::always;
  }
  return RocmlirSplitKSelectionLikelihood::maybe;
}

bool isModuleFusible(ModuleOp module, StringRef perfConfig) {
  if (!rock::isSplitKRequested(module, perfConfig)) {
    return true;
  }
  return succeeded(rock::testFusionLegality(module));
}

} // namespace rock
} // namespace mlir
