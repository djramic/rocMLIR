//===- AccelEmitter.cpp - MLIR helper to emit acceleration intrinsics
//---------------===//
//
// Copyright 2020 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This class tries to abstract away the code-generation details needed to
// generated calls to matrix multiply accelerator intrinsics (wmma, mfma).
//
//
//===----------------------------------------------------------------------===//

#include "AccelEmitter.h"
#include "mlir/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;
using namespace mlir::rock::accel;

// ************************
// Generic helper functions
// ************************

AccelEmitter::AccelEmitter(StringRef arch,
                           RockAccelTuningParamAttrInterface tuningParams,
                           AccelEmitterParams accelEmitterParams)
    : tuningParams(tuningParams), accelEmitterParams(accelEmitterParams),
      waveSize(rock::lookupArchInfo(arch).waveSize) {
  if (failed(validateAcceleratorProperties()))
    llvm_unreachable("Accelerator parameters validation failed");
}

void AccelEmitter::computeOutputConversion(PatternRewriter &b, Location loc,
                                           Value regVectorOrig, Value regDest,
                                           bool forceUnroll) {

  // Extract relevant emitter parameters
  int64_t mRepeats = accelEmitterParams.mRepeats;
  int64_t nRepeats = accelEmitterParams.nRepeats;
  int64_t nResultVectors = accelEmitterParams.nResultVectors;
  VectorType accVectorType = accelEmitterParams.accVectorType;

  Type destType = regDest.getType().dyn_cast<MemRefType>().getElementType();

  int64_t accVectorLen = accVectorType.getNumElements();
  int64_t numElements = accVectorLen * (mRepeats * nRepeats * nResultVectors);
  auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);

  BottomUpTMBuilder toRegCScalar(b, {"scalar"}, {numElements}, loc);
  toRegCScalar.embed({"vector"}, {0}, {mRepeats * nRepeats * nResultVectors},
                     "scalar", {accVectorLen});
  TransformMapAttr toRegCScalarAttr = toRegCScalar.get();

  auto convertLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{{zeroConstantOp}, {zeroConstantOp}},
      ArrayRef<Attribute>{b.getArrayAttr({}), b.getArrayAttr(toRegCScalarAttr)},
      /*bounds=*/ArrayRef<int64_t>{mRepeats * nRepeats * nResultVectors},
      /*strides=*/std::nullopt, forceUnroll, /*useIndexDiffs=*/true);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(convertLoop.getBody());
    Value loaded =
        b.create<memref::LoadOp>(loc, accVectorType, regVectorOrig,
                                 convertLoop.getLowerCoords(/*domain*/ 0));
    Value cast = loaded;
    if (destType != accVectorType.getElementType()) {
      VectorType destVectorType = accVectorType.clone(destType);
      cast = createTypeConversionOp(b, loc, loaded, destVectorType);
    }
    b.create<InBoundsStoreOp>(loc, cast, regDest,
                              convertLoop.getLowerCoords(/*domain*/ 1));
  }
}

// **************************
// Fma accelerator interface
// **************************

FmaEmitter::FmaEmitter(FmaInsn fmaInsn, StringRef arch,
                        RockAccelTuningParamAttrInterface tuningParams)
    : AccelEmitter{arch, tuningParams,
                   initAccelEmitterParams(fmaInsn, tuningParams)},
      fmaInsn(fmaInsn) {}

AccelEmitterParams FmaEmitter::initAccelEmitterParams(
    FmaInsn fmaInsn, RockAccelTuningParamAttrInterface rawTuningParams) {
    AccelEmitterParams params;

    auto tuningParams = rawTuningParams.dyn_cast<GeneralGemmParamsAttr>();

    params.kpack = tuningParams.getKpack();
    params.blockSize = tuningParams.getBlockSize();
    params.kpacksPerBlock = tuningParams.getKPerBlock();
    params.mPerBlock = tuningParams.getMPerBlock();
    params.nPerBlock = tuningParams.getNPerBlock();
    params.mPerThread = tuningParams.getMPerThread();
    params.nPerThread = tuningParams.getNPerThread();
    params.forceUnroll = tuningParams.getForceUnroll();
    params.non_useIndexDiffs = true;
    params.useIndexDiffs = true;
    params.kBase = fmaInsn.inputLen;
    params.argTypeA = fmaInsn.argTypeA;  //???
    params.argTypeB = fmaInsn.argTypeB;  //???
    params.kBasePerThread = params.kpacksPerBlock * params.kpack / params.kBase;

    GeneralGemmBlockStructure blockStructure =
          *deriveGeneralGemmBlockStructure(params.blockSize);
    int64_t mThreadsPerCuwave = blockStructure.mThreadsPerCuwave;
    int64_t nThreadsPerCuwave = blockStructure.nThreadsPerCuwave;
    int64_t mCuwavesPerBlock = blockStructure.mCuwavesPerBlock;
    int64_t nCuwavesPerBlock = blockStructure.nCuwavesPerBlock;

    int64_t gemmMRepeat =
          params.mPerBlock / (params.mPerThread * mThreadsPerCuwave * mCuwavesPerBlock);
    int64_t gemmNRepeat =
          params.nPerBlock / (params.nPerThread * nThreadsPerCuwave * nCuwavesPerBlock);

    int64_t threadCNumM = gemmMRepeat * params.mPerThread;
    int64_t threadCNumN = gemmNRepeat * params.nPerThread;
    params.nRepeats = gemmNRepeat;
    params.mRepeats = gemmMRepeat;
    params.threadCNumM = threadCNumM;
    params.threadCNumN = threadCNumN;
    params.accType = fmaInsn.retType;
    params.destType = fmaInsn.retType;

    params.nOutputVectors = threadCNumM * threadCNumN;

    return params;
}

void FmaEmitter::emitThreadwiseLoop(OpBuilder &b, Location loc, Value argA,
                                     Value argB, Value bufferC,
                                     Value regCOffset){



}

Value FmaEmitter::wrapLDSBufferForLoad(OpBuilder &b, Location loc, Value buffer,
                                     int64_t blockSize,
                                     int64_t dInCopyPerThread, StringRef dName,
                                     bool rotateDWithK) const {

  llvm::outs() << "\nwrapLDSBufferForLoad FMA" << "\n";
  int64_t mRepeat = accelEmitterParams.mRepeats;
  int64_t nRepeat = accelEmitterParams.nRepeats;
  int64_t mPerThread = accelEmitterParams.mPerThread;
  int64_t m = accelEmitterParams.mPerBlock;
  int64_t n = accelEmitterParams.nPerBlock;
  int64_t k = accelEmitterParams.kpacksPerBlock;
  int64_t kPack = accelEmitterParams.kpack;


  GeneralGemmBlockStructure blockStructure =
          *deriveGeneralGemmBlockStructure(blockSize);
  int64_t mThreadsPerCuwave = blockStructure.mThreadsPerCuwave;
  int64_t nThreadsPerCuwave = blockStructure.nThreadsPerCuwave;
  int64_t mCuwavesPerBlock = blockStructure.mCuwavesPerBlock;
  int64_t nCuwavesPerBlock = blockStructure.nCuwavesPerBlock;

  
  llvm::outs() << "\n -- got attributes" << "\n";
  SmallVector<Attribute> transformAttrs;

  auto ldsTidSplitter = [&](StringRef repeatName, int64_t repeatLen,
                                StringRef perThreadName,
                                int64_t perThreadLen) -> TopDownTMBuilder {
    TopDownTMBuilder splitTidForLDS(
            b, {"k", repeatName, "tid", perThreadName, "kpack"},
            {k, repeatLen, blockSize, perThreadLen, kPack}, loc);
    splitTidForLDS.passThrough({"k", repeatName});
    splitTidForLDS.merge({"m_cuwaves", "n_cuwaves", "m_cuwave", "n_cuwave"},
                             {2, 3, 4, 5}, "tid",
                             {mCuwavesPerBlock, nCuwavesPerBlock,
                              mThreadsPerCuwave, nThreadsPerCuwave});
    splitTidForLDS.passThrough({perThreadName, "kpack"}, {6, 7},
                                   {perThreadName, "kpack"});
    return splitTidForLDS;
  };
  llvm::outs() << "\n -- finish ldsTidSplitter" << "\n";

  TopDownTMBuilder splitTidA =
          ldsTidSplitter("m_repeat", mRepeat, "m_thread", mPerThread);
  TransformMapAttr splitTidAAttr = splitTidA.get();
  auto toLdsIndexA = TopDownTMBuilder::below(splitTidA, splitTidAAttr);
  toLdsIndexA.passThrough("k");
  toLdsIndexA.unmerge(
          "m", 1, {"m_repeat", "m_cuwaves", "m_cuwave", "m_thread"},
          {mRepeat, mCuwavesPerBlock, mThreadsPerCuwave, mPerThread});
  toLdsIndexA.ignore("n_cuwaves");
  toLdsIndexA.ignore("n_cuwave");
  toLdsIndexA.passThrough({"kpack"}, {2}, {"kpack"});
  TransformMapAttr toLdsIndexAAttr = toLdsIndexA.get();
  SmallVector<Attribute> transformAttrsA{splitTidAAttr, toLdsIndexAAttr};

  llvm::outs() << "\n -- finish splitTidA" << "\n";
  int64_t strideA = (k == 1 ? dInCopyPerThread : 1);
    rotateIf(rotateDWithK, toLdsIndexA, toLdsIndexAAttr, strideA, "m",
               m, 1, "k", k, {"k"}, {"kpack"}, transformAttrsA);

  Value matrixA = buffer;

  ArrayRef<int64_t> shape = {k,m,kPack};
  ArrayRef<StringRef> names = {"k", "m", "kpack"};

  MemRefType bufferType = matrixA.getType().cast<MemRefType>();
  ArrayRef<int64_t> outShape = bufferType.getShape();
  assert(outShape.size() == 1 && "Buffer being reshaped must start linear");
  SmallVector<int64_t> strides;
  strides.reserve(shape.size());
  int64_t stride = 1;
  for (int64_t v : llvm::reverse(shape)) {
    stride *= v;
  }
  assert(stride == outShape[0] && "Strides must multiply to buffer length");

  TopDownTMBuilder transform1(b, names, shape, loc);
  transform1.unmerge("raw", 0, names, shape);

  TransformMapAttr transformAttr = transform1.get();

  Value newMatrixA = b.create<TransformOp>(loc, matrixA, transformAttr);

  ArrayAttr transformsA;

  bool ldsANeedsi64, ldsBNeedsi64;
  std::tie(matrixA, transformsA, ldsANeedsi64) =
          untransform(b, newMatrixA, b.getArrayAttr(transformAttrsA));
  //if (ldsANeedsi64 || ldsBNeedsi64)
    //return b.notifyMatchFailure(loc, "LDS map can't need 64-bit indexing");
  llvm::outs() <<"buffer : " << buffer << "\n";
  return transform(b, buffer, transformsA);

}

RegsAsMatrixSubTiles FmaEmitter::computeOutputTransforms(

    PatternRewriter &b, Location loc, int64_t mLen, int64_t nLen,
    int64_t blockSize, ArrayRef<int64_t> bidGridLengths, int64_t inMPerThread,
    int64_t inNPerThread, bool doSwapThreadIterSubDimsForM,
    bool doSwapThreadIterSubDimsForN) {

  llvm::outs() << "\n**DEBUG:   computeOutputTransforms !!FMA!!\n"; 
  auto tuningParams = this->tuningParams.dyn_cast<GeneralGemmParamsAttr>(); 

  //Extract relevant tuning parameters
  int64_t mPerBlock = tuningParams.getMPerBlock();
  int64_t nPerBlock = tuningParams.getNPerBlock();
  int64_t mPerThread = tuningParams.getMPerThread();
  int64_t nPerThread = tuningParams.getNPerThread();
  int64_t kpack = tuningParams.getKpack();
  int64_t kpacksPerBlock = tuningParams.getKPerBlock();

  //Extract derive parameters
  GeneralGemmBlockStructure blockStructure =
          *deriveGeneralGemmBlockStructure(blockSize);
  int64_t mThreadsPerCuwave = blockStructure.mThreadsPerCuwave;
  int64_t nThreadsPerCuwave = blockStructure.nThreadsPerCuwave;
  int64_t mCuwavesPerBlock = blockStructure.mCuwavesPerBlock;
  int64_t nCuwavesPerBlock = blockStructure.nCuwavesPerBlock;

  int64_t kPerBlock = kpacksPerBlock * kpack;

  int64_t gemmMRepeat =
          mPerBlock / (mPerThread * mThreadsPerCuwave * mCuwavesPerBlock);
  int64_t gemmNRepeat =
          nPerBlock / (nPerThread * nThreadsPerCuwave * nCuwavesPerBlock);

  int64_t threadCNumM = gemmMRepeat * mPerThread;
  int64_t threadCNumN = gemmNRepeat * nPerThread;

  int64_t threadCNumRegisters = threadCNumM * threadCNumN;

      
  RegsAsMatrixSubTiles ret;
  {
    // Create views as gridwise sub-tile of C
      TopDownTMBuilder splitMemoryCoords(
          b, {"g_block", "m_block", "n_block", "tid", "iter"},
          {bidGridLengths[0], bidGridLengths[1], bidGridLengths[2], blockSize, threadCNumRegisters}, loc);
      splitMemoryCoords.passThrough({"g_block", "m_block", "n_block"});
      splitMemoryCoords.merge(
          {"m_cuwaves", "n_cuwaves", "m_cuwave", "n_cuwave"}, {3, 4, 5, 6},
          "tid",
          {mCuwavesPerBlock, nCuwavesPerBlock, mThreadsPerCuwave,
           nThreadsPerCuwave});
      splitMemoryCoords.merge(
          {"m_repeat", "m_thread", "n_repeat", "n_thread"}, {7, 8, 9, 10},
          "iter", {gemmMRepeat, mPerThread, gemmNRepeat, nPerThread});
      TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();

      auto toMatrixC =
          TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);
      toMatrixC.passThrough({"gemmG"}, {0}, {"g_block"});
      toMatrixC.unmerge(
          "gemmM", 1,
          {"m_block", "m_repeat", "m_cuwaves", "m_cuwave", "m_thread"},
          {mPerBlock, gemmMRepeat, mCuwavesPerBlock, mThreadsPerCuwave,
           mPerThread});
      toMatrixC.unmerge(
          "gemmN", 2,
          {"n_block", "n_repeat", "n_cuwaves", "n_cuwave", "n_thread"},
          {nPerBlock, gemmNRepeat, nCuwavesPerBlock, nThreadsPerCuwave,
           nPerThread});
      SmallVector<Attribute> transformAttrs{splitMemoryCoordsAttr};
      swapThreadIdAndIteration(toMatrixC, /*mBlocks=*/bidGridLengths[1],
                               /*nBlocks=*/bidGridLengths[2], inMPerThread,
                               inNPerThread, mPerBlock, nPerBlock,
                               doSwapThreadIterSubDimsForM,
                               doSwapThreadIterSubDimsForN,
                               /*isBlockwise=*/false, transformAttrs);
      ret.gridSubTile = b.getArrayAttr(transformAttrs);
  }

  return ret; 
}


// **************************
// Mfma accelerator interface
// **************************

MfmaEmitter::MfmaEmitter(MfmaInsnGroup mfmaGroup, StringRef arch,
                         RockAccelTuningParamAttrInterface tuningParams)
    : AccelEmitter{arch, tuningParams,
                   initAccelEmitterParams(mfmaGroup, tuningParams)},
      mfmaGroup{mfmaGroup} {}

AccelEmitterParams MfmaEmitter::initAccelEmitterParams(
    MfmaInsnGroup mfmaGroup, RockAccelTuningParamAttrInterface RawTuningParams) {
  AccelEmitterParams params;
  MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();
  auto tuningParams = RawTuningParams.dyn_cast<XdlopsGemmParamsAttr>();
  // Extract relevant tuning parameters
  int64_t kpackPerBlock = tuningParams.getKpackPerBlock();
  int64_t mPerWave = tuningParams.getMPerWave();
  int64_t nPerWave = tuningParams.getNPerWave();
  int64_t kPack = tuningParams.getKpack();
  int64_t K = kpackPerBlock * kPack;

  // Accelerator parameters
  params.kBase = mfmaAttr.k_base;
  params.kBasePerThread =
      (mfmaAttr.isKReduction ? K / mfmaAttr.inputSpansPerMfmaIn : K) /
      params.kBase;
  params.mRepeats = mfmaGroup.getMRepeats(mPerWave);
  params.nRepeats = mfmaGroup.getNRepeats(nPerWave);
  params.nResultVectors = mfmaGroup.getImms().size();
  params.mPerAccel = mPerWave / params.mRepeats;
  params.nPerAccel = nPerWave / params.nRepeats;
  params.kpackPerThread =
      (mfmaAttr.isKReduction ? kpackPerBlock / mfmaAttr.inputSpansPerMfmaIn
                             : kpackPerBlock);

  // Accelerator data types
  params.argTypeA = mfmaGroup.getArgTypeA();
  params.argTypeB = mfmaGroup.getArgTypeB();
  params.accVectorType = mfmaGroup.getRetType();

  return params;
}

void MfmaEmitter::emitThreadwiseLoop(OpBuilder &b, Location loc, Value argA,
                                     Value argB, Value bufferC,
                                     Value regCOffset) {
  MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();
  int64_t mfmaNonKDim = mfmaAttr.mfmaNonKDim;
  auto imms = mfmaGroup.getImms();
  int64_t nResultVectors = imms.size();
  VectorType vectorType = mfmaGroup.getRetType();
  for (int64_t i = 0; i < nResultVectors; ++i) {
    Value offset = b.createOrFold<arith::ConstantIndexOp>(loc, i);
    offset = b.create<AddIOp>(loc, offset, regCOffset);

    auto vectorC = b.create<memref::LoadOp>(loc, vectorType, bufferC, offset);
    auto mfma = b.create<amdgpu::MFMAOp>(
        loc, vectorType, mfmaNonKDim, mfmaNonKDim, mfmaAttr.k,
        mfmaAttr.blocksMfma, argA, argB, vectorC, /*cbsz=*/imms[i].cbsz,
        /*abid=*/imms[i].abid,
        /*blgp=*/imms[i].blgp, /*reducePrecision=*/false, /*negateA=*/false,
        /*negateB=*/false, /*negateC=*/false);
    auto vectorD = mfma.getDestD();

    b.create<memref::StoreOp>(loc, vectorD, bufferC, offset);
  }
}

static void
makeViewsForRowsAndCols(TopDownTMBuilder &viewBuilder, int64_t mPerRepeat,
                        int64_t nPerRepeat,
                        const llvm::StringMap<uint32_t> &rowsAndColsIdxs,
                        int64_t endSizeJ, int64_t blocksInOutRegs) {
  // Here we use the full builder API since we want index and name control
  bool isABroadcast = (nPerRepeat >= mPerRepeat);
  SmallVector<StringRef, 2> rowsFirst = {"blk_row", "blk_col"};
  SmallVector<StringRef, 2> colsFirst = {"blk_col", "blk_row"};
  viewBuilder.merge(
      isABroadcast ? rowsFirst : colsFirst,
      {rowsAndColsIdxs.lookup("blkMajor"), rowsAndColsIdxs.lookup("blkMinor")},
      "j", {endSizeJ / blocksInOutRegs, blocksInOutRegs});
  viewBuilder.passThrough(
      {"vec_group", "vec_item"},
      {rowsAndColsIdxs.lookup("vec_group"), rowsAndColsIdxs.lookup("vec_item")},
      {"vec_group", "vec_item"});
}

RegsAsMatrixSubTiles MfmaEmitter::computeOutputTransforms(
    PatternRewriter &b, Location loc, int64_t mLen, int64_t nLen,
    int64_t blockSize, ArrayRef<int64_t> bidGridLengths, int64_t inMPerThread,
    int64_t inNPerThread, bool doSwapThreadIterSubDimsForM,
    bool doSwapThreadIterSubDimsForN) {
    auto tuningParams = this->tuningParams.dyn_cast<XdlopsGemmParamsAttr>();
  // Extract relevant tuning parameters
  int64_t mPerBlock = tuningParams.getMPerBlock();
  int64_t nPerBlock = tuningParams.getNPerBlock();
  int64_t mPerWave = tuningParams.getMPerWave();
  int64_t nPerWave = tuningParams.getNPerWave();

  // Extract relevant emitter parameters
  int64_t mRepeats = accelEmitterParams.mRepeats;
  int64_t nRepeats = accelEmitterParams.nRepeats;
  int64_t nResultVectors = accelEmitterParams.nResultVectors;
  VectorType accVectorType = accelEmitterParams.accVectorType;
  int64_t mPerAccel = accelEmitterParams.mPerAccel;
  int64_t nPerAccel = accelEmitterParams.nPerAccel;

  auto mfmaAttr = mfmaGroup.getInsnAttr();
  int64_t mPerRepeat = mPerWave / mRepeats;
  int64_t nPerRepeat = nPerWave / nRepeats;
  int64_t nWaves = nPerBlock / nPerWave;
  int64_t mWaves = mPerBlock / mPerWave;
  int64_t rowGroupSize = mfmaAttr.rowGroupSize;
  int64_t rowGroupsPerBlock = mfmaAttr.rowGroupsPerBlock;
  int64_t inputSpanLen = mfmaAttr.inputSpanLen;
  int64_t m = mfmaAttr.mfmaNonKDim;

  // Note n has the 4x4 => 4x64 behavior that necessitated
  // inputSpansPerMfmaIn
  int64_t n = mfmaAttr.inputSpanLen;
  int64_t inputSpansPerMfmaIn = mfmaAttr.inputSpansPerMfmaIn;
  int64_t blocksInOutRegs = mfmaAttr.blocksInOutRegs;
  int64_t blocksPerRepeat = (mPerRepeat * nPerRepeat) / (m * n);

  int64_t retNumElements = accVectorType.getNumElements();
  int64_t numElements = retNumElements * mRepeats * nRepeats * nResultVectors;
  int64_t wavesInKernelBlock = blockSize / waveSize;

  // Note that `wave_m` and `wave_n` are strided by mPerAccel/nPerAccel, i.e.,
  // all the waves will compute next to each other and then they will move to
  // the next subtile in the workgroup
  SmallVector<StringRef, 7> dimNamesM{/*0=*/"m_block",
                                      /*1=*/"m_i",
                                      /*2=*/"wave_m",
                                      /*3=*/"blk_row",
                                      /*4=*/"vec_group",
                                      /*5=*/"m_tid",
                                      /*6=*/"vec_item"};
  SmallVector<int64_t, 7> orderedDimStridesM{/*0=*/mPerBlock,
                                             /*1=*/mPerAccel * mWaves,
                                             /*2=*/mPerAccel,
                                             /*3=*/m,
                                             /*4=*/inputSpansPerMfmaIn *
                                                 rowGroupSize,
                                             /*5=*/rowGroupSize,
                                             /*6=*/1};
  SmallVector<int64_t, 7> dimSizesM;
  convertDimStridestoSizes(orderedDimStridesM, mLen, dimSizesM);

  SmallVector<StringRef, 5> dimNamesN{/*0=*/"n_block",
                                      /*1=*/"n_i",
                                      /*2=*/"wave_n",
                                      /*3=*/"blk_col",
                                      /*4=*/"n_tid"};
  SmallVector<int64_t, 5> orderedDimStridesN{/*0=*/nPerBlock,
                                             /*1=*/nPerAccel * nWaves,
                                             /*2=*/nPerAccel,
                                             /*3=*/n,
                                             /*4=*/1};
  SmallVector<int64_t, 7> dimSizesN;
  convertDimStridestoSizes(orderedDimStridesN, nLen, dimSizesN);

  RegsAsMatrixSubTiles ret;
  {
    // Create views as gridwise sub-tile of C
    TopDownTMBuilder splitMemoryCoords(
        b, {"g_block", "m_block", "n_block", "tid", "item"},
        {bidGridLengths[0], bidGridLengths[1], bidGridLengths[2], blockSize,
         numElements},
        loc);
    splitMemoryCoords.passThrough({"g_block", "m_block", "n_block"});
    splitMemoryCoords.merge(
        {"wave", "m_tid", "n_tid"}, {3, 4, 5}, "tid",
        {wavesInKernelBlock, waveSize / inputSpanLen, inputSpanLen});
    splitMemoryCoords.merge(
        {"i", "j", "vec_group", "vec_item"}, {6, 7, 8, 9}, "item",
        {numElements / (blocksPerRepeat * rowGroupsPerBlock * rowGroupSize),
         blocksPerRepeat, rowGroupsPerBlock, rowGroupSize});
    TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();
    auto toRowsAndCols =
        TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);
    // "blkMajor" and "blkMinor" are placeholder names because we don't know
    // if they'll be column or row until we check for broadcast-ness.
    llvm::StringMap<uint32_t> rowsAndColsIdxs = expandNamesInPlace(
        splitMemoryCoords, {{"wave", {"wave_m", "wave_n"}},
                            {"i", {"m_i", "n_i"}},
                            {"j", {"blkMajor", "blkMinor"}}});
    TopDownTMBottomDimsWrapper rowsAndColsWrap(toRowsAndCols, rowsAndColsIdxs);
    rowsAndColsWrap.passThrough({"g_block", "m_block", "n_block"});
    rowsAndColsWrap.merge({"wave_m", "wave_n"}, "wave",
                          {wavesInKernelBlock / nWaves, nWaves});
    rowsAndColsWrap.passThrough({"m_tid", "n_tid"});
    rowsAndColsWrap.merge(
        {"m_i", "n_i"}, "i",
        {splitMemoryCoords.endSize("i") / nRepeats, nRepeats});
    makeViewsForRowsAndCols(toRowsAndCols, mPerRepeat, nPerRepeat,
                            rowsAndColsIdxs, splitMemoryCoords.endSize("j"),
                            blocksInOutRegs);
    TransformMapAttr toRowsAndColsAttr = toRowsAndCols.get();
    auto toMatrixC = TopDownTMBuilder::below(toRowsAndCols, toRowsAndColsAttr);
    toMatrixC.passThrough({"gemmG"}, {0}, {"g_block"});
    toMatrixC.unmerge("gemmM", 1, dimNamesM, dimSizesM);
    toMatrixC.unmerge("gemmN", 2, dimNamesN, dimSizesN);

    // Before returning the output view, if necessary, swap back the
    // threadid/iter dimensions on both the M/N axis.
    SmallVector<Attribute> transformAttrs{splitMemoryCoordsAttr,
                                          toRowsAndColsAttr};
    mlir::rock::swapThreadIdAndIteration(
        toMatrixC, /*mBlocks=*/bidGridLengths[1], /*nBlocks=*/bidGridLengths[2],
        inMPerThread, inNPerThread, mPerBlock, nPerBlock,
        doSwapThreadIterSubDimsForM, doSwapThreadIterSubDimsForN,
        /*isBlockwise=*/false, transformAttrs);

    ret.gridSubTile = b.getArrayAttr(transformAttrs);
  }

  {
    // Create views as blockwise sub-tile of C
    TopDownTMBuilder splitMemoryCoords(b, {"tid", "item"},
                                       {blockSize, numElements}, loc);
    splitMemoryCoords.merge(
        {"wave", "m_tid", "n_tid"}, {0, 1, 2}, "tid",
        {wavesInKernelBlock, waveSize / inputSpanLen, inputSpanLen});
    splitMemoryCoords.merge(
        {"i", "j", "vec_group", "vec_item"}, {3, 4, 5, 6}, "item",
        {numElements / (blocksPerRepeat * rowGroupsPerBlock * rowGroupSize),
         blocksPerRepeat, rowGroupsPerBlock, rowGroupSize});
    TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();
    auto toRowsAndCols =
        TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);
    // "blkMajor" and "blkMinor" are placeholder names because we don't know
    // if they'll be column or row until we check for broadcast-ness.
    llvm::StringMap<uint32_t> rowsAndColsIdxs = expandNamesInPlace(
        splitMemoryCoords, {{"wave", {"wave_m", "wave_n"}},
                            {"i", {"m_i", "n_i"}},
                            {"j", {"blkMajor", "blkMinor"}}});
    TopDownTMBottomDimsWrapper rowsAndColsWrap(toRowsAndCols, rowsAndColsIdxs);
    rowsAndColsWrap.merge({"wave_m", "wave_n"}, "wave",
                          {wavesInKernelBlock / nWaves, nWaves});
    rowsAndColsWrap.passThrough({"m_tid", "n_tid"});
    rowsAndColsWrap.merge(
        {"m_i", "n_i"}, "i",
        {splitMemoryCoords.endSize("i") / nRepeats, nRepeats});
    makeViewsForRowsAndCols(toRowsAndCols, mPerRepeat, nPerRepeat,
                            rowsAndColsIdxs, splitMemoryCoords.endSize("j"),
                            blocksInOutRegs);
    TransformMapAttr toRowsAndColsAttr = toRowsAndCols.get();
    auto toMatrixC = TopDownTMBuilder::below(toRowsAndCols, toRowsAndColsAttr);
    toMatrixC.unmerge("gemmM", 0, ArrayRef<StringRef>{dimNamesM}.slice(1),
                      ArrayRef<int64_t>{dimSizesM}.slice(1));
    toMatrixC.unmerge("gemmN", 1, ArrayRef<StringRef>{dimNamesN}.slice(1),
                      ArrayRef<int64_t>{dimSizesN}.slice(1));

    // Before returning the output view, if necessary, swap back the
    // threadid/iter dimensions on both the M/N axis.
    SmallVector<Attribute> transformAttrs{splitMemoryCoordsAttr,
                                          toRowsAndColsAttr};
    mlir::rock::swapThreadIdAndIteration(
        toMatrixC, /*mBlocks=*/bidGridLengths[1], /*nBlocks=*/bidGridLengths[2],
        inMPerThread, inNPerThread, mPerBlock, nPerBlock,
        doSwapThreadIterSubDimsForM, doSwapThreadIterSubDimsForN,
        /*isBlockwise=*/true, transformAttrs);
    ret.blockSubTile = b.getArrayAttr(transformAttrs);
  }

  {
    // Create views as threadwise sub-tile of C
    TopDownTMBuilder splitMemoryCoords(b, {"item"}, {numElements}, loc);
    splitMemoryCoords.merge(
        {"i", "j", "vec_group", "vec_item"}, {0, 1, 2, 3}, "item",
        {numElements / (blocksPerRepeat * rowGroupsPerBlock * rowGroupSize),
         blocksPerRepeat, rowGroupsPerBlock, rowGroupSize});
    TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();
    auto toRowsAndCols =
        TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);
    // "blkMajor" and "blkMinor" are placeholder names because we don't know
    // if they'll be column or row until we check for broadcast-ness.
    llvm::StringMap<uint32_t> rowsAndColsIdxs = expandNamesInPlace(
        splitMemoryCoords,
        {{"i", {"m_i", "n_i"}}, {"j", {"blkMajor", "blkMinor"}}});
    TopDownTMBottomDimsWrapper rowsAndColsWrap(toRowsAndCols, rowsAndColsIdxs);
    rowsAndColsWrap.merge(
        {"m_i", "n_i"}, "i",
        {splitMemoryCoords.endSize("i") / nRepeats, nRepeats});
    makeViewsForRowsAndCols(toRowsAndCols, mPerRepeat, nPerRepeat,
                            rowsAndColsIdxs, splitMemoryCoords.endSize("j"),
                            blocksInOutRegs);
    TransformMapAttr toRowsAndColsAttr = toRowsAndCols.get();
    auto toMatrixC = TopDownTMBuilder::below(toRowsAndCols, toRowsAndColsAttr);
    toMatrixC.unmerge("gemmM", 0,
                      {dimNamesM[1], dimNamesM[3], dimNamesM[4], dimNamesM[6]},
                      {dimSizesM[1], dimSizesM[3], dimSizesM[4], dimSizesM[6]});
    toMatrixC.unmerge("gemmN", 1, {dimNamesN[1], dimNamesN[3]},
                      {dimSizesN[1], dimSizesN[3]});
    TransformMapAttr toMatrixCAttr = toMatrixC.get();
    ret.threadSubTile = b.getArrayAttr(
        {splitMemoryCoordsAttr, toRowsAndColsAttr, toMatrixCAttr});
  }

  return ret;
}

Value MfmaEmitter::wrapLDSBufferForLoad(OpBuilder &b, Location loc,
                                        Value buffer, int64_t blockSize,
                                        int64_t dInCopyPerThread,
                                        StringRef dName,
                                        bool rotateDWithK) const {

  StringRef thisWaveDim = dName == "m" ? "wave_m" : "wave_n";
  StringRef otherWaveDim = dName == "m" ? "wave_n" : "wave_m";
  auto tuningParams = this->tuningParams.dyn_cast<XdlopsGemmParamsAttr>();
  // Extract relevant tuning parameters
  int64_t mPerWave = tuningParams.getMPerWave();
  int64_t nPerWave = tuningParams.getNPerWave();
  int64_t kPerBlock = tuningParams.getKpackPerBlock();
  int64_t mPerBlock = tuningParams.getMPerBlock();
  int64_t nPerBlock = tuningParams.getNPerBlock();
  int64_t kPack = tuningParams.getKpack();

  // Extract relevant emitter parameters
  MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();
  int64_t inputSpanLen = mfmaAttr.inputSpanLen;
  int64_t kpackPerThread = accelEmitterParams.kpackPerThread;
  bool isKReduction = mfmaAttr.isKReduction;

  // Extract relevant derived parameters
  int64_t mWaves = mPerBlock / mPerWave;
  int64_t nWaves = nPerBlock / nPerWave;
  int64_t dWaves = (dName == "m" ? mPerBlock / mPerWave : nPerBlock / nPerWave);
  int64_t dRepeats = (dName == "m" ? accelEmitterParams.mRepeats
                                   : accelEmitterParams.nRepeats);
  int64_t dPerAccel = (dName == "m" ? accelEmitterParams.mPerAccel
                                    : accelEmitterParams.nPerAccel);
  int64_t dPerBlock = (dName == "m" ? mPerBlock : nPerBlock);

  SmallVector<Attribute> transformAttrs;
  if (!isKReduction) {
    TopDownTMBuilder splitTid(b, {"tid", "d_iter", "k_iter"},
                              {blockSize, dPerBlock, kpackPerThread});
    splitTid.merge({"wave_id", "lane_id"}, {0, 1}, "tid",
                   {blockSize / waveSize, waveSize});

    splitTid.passThrough({"d_iter", "k_iter"}, {2, 3}, {"d_iter", "k_iter"});
    TransformMapAttr splitTidAttr = splitTid.get();
    transformAttrs.push_back(splitTidAttr);

    TopDownTMBuilder splitWaveId =
        TopDownTMBuilder::below(splitTid, splitTidAttr);
    splitWaveId.merge({"wave_m", "wave_n"}, {0, 1}, "wave_id",
                      {mWaves, nWaves});
    splitWaveId.passThrough({"lane_id", "d_iter", "k_iter"}, {2, 3, 4},
                            {"lane_id", "d_iter", "k_iter"});
    TransformMapAttr splitWaveIdAttr = splitWaveId.get();
    transformAttrs.push_back(splitWaveIdAttr);

    TopDownTMBuilder toLDSRowCol =
        TopDownTMBuilder::below(splitWaveId, splitWaveIdAttr);

    // d = d_i*dWaves*dPerAccel + wave_d*dPerAccel + lane_id
    toLDSRowCol.unmerge("d", 0, {"d_iter", thisWaveDim, "lane_id"},
                        {dRepeats, dWaves, dPerAccel});

    // k = k_i
    toLDSRowCol.passThrough({"k"}, 1, {"k_iter"});
    toLDSRowCol.ignore(otherWaveDim);

    TransformMapAttr toLDSRowColAttr = toLDSRowCol.get();

    transformAttrs.push_back(toLDSRowColAttr);

    int64_t stride = (kPack == 1 ? dInCopyPerThread : 1);
    auto offset =
        rotateIf(rotateDWithK, toLDSRowCol, toLDSRowColAttr, stride, "d",
                 dPerBlock, 0, "k", kPerBlock, {}, {"k"}, transformAttrs);

    offset.unmerge("source_offset", 0, {"k", "d"}, {kPerBlock, dPerBlock});

    TransformMapAttr offsetAttr = offset.get();
    transformAttrs.push_back(offsetAttr);

  } else {
    TopDownTMBuilder splitTid(b, {"tid", "d_iter", "k_iter"},
                              {blockSize, dPerBlock, kpackPerThread});
    splitTid.merge(
        {"wave_id", "blk_id", "blk_td"}, {0, 1, 2}, "tid",
        {blockSize / waveSize, waveSize / inputSpanLen, inputSpanLen});

    splitTid.passThrough({"d_iter", "k_iter"}, {3, 4}, {"d_iter", "k_iter"});
    TransformMapAttr splitTidAttr = splitTid.get();
    transformAttrs.push_back(splitTidAttr);

    TopDownTMBuilder splitWaveId =
        TopDownTMBuilder::below(splitTid, splitTidAttr);
    splitWaveId.merge({"wave_m", "wave_n"}, {0, 1}, "wave_id",
                      {mWaves, nWaves});
    splitWaveId.passThrough({"blk_id", "blk_td", "d_iter", "k_iter"},
                            {2, 3, 4, 5},
                            {"blk_id", "blk_td", "d_iter", "k_iter"});
    TransformMapAttr splitWaveIdAttr = splitWaveId.get();
    transformAttrs.push_back(splitWaveIdAttr);

    TopDownTMBuilder toLDSRowCol =
        TopDownTMBuilder::below(splitWaveId, splitWaveIdAttr);

    // d = blk_td + d_i * waveOffset
    toLDSRowCol.unmerge("d", 0, {"d_iter", thisWaveDim, "blk_td"},
                        {dRepeats, dWaves, inputSpanLen});
    // k = k_i + kpackPerBlock * blk_id
    toLDSRowCol.unmerge("k", 1, {"blk_id", "k_iter"},
                        {waveSize / inputSpanLen, kpackPerThread});

    toLDSRowCol.ignore(otherWaveDim);

    TransformMapAttr toLDSRowColAttr = toLDSRowCol.get();
    transformAttrs.push_back(toLDSRowColAttr);

    int64_t stride = (kPack == 1 ? dInCopyPerThread : 1);
    auto offset =
        rotateIf(rotateDWithK, toLDSRowCol, toLDSRowColAttr, stride, "d",
                 dPerBlock, 0, "k", kPerBlock, {}, {"k"}, transformAttrs);

    offset.unmerge("source_offset", 0, {"k", "d"}, {kPerBlock, dPerBlock});

    TransformMapAttr offsetAttr = offset.get();
    transformAttrs.push_back(offsetAttr);
  }

  ArrayAttr ldsRead = b.getArrayAttr(transformAttrs);
  return transform(b, buffer, ldsRead);
}

LogicalResult MfmaEmitter::validateAcceleratorProperties() {
  // Extract relevant tuning parameters
  int64_t kPack = tuningParams.getKpack();

  // Extract relevant emitter parameters
  int64_t kBase = accelEmitterParams.kBase;

  if (kPack > 1 && (kPack < kBase || kPack % kBase != 0))
    return failure();

  return success();
}

// **************************
// Wmma accelerator interface
// **************************

WmmaEmitter::WmmaEmitter(WmmaInsn wmmaInsn, StringRef arch,
                         RockAccelTuningParamAttrInterface tuningParams)
    : AccelEmitter{arch, tuningParams,
                   initAccelEmitterParams(wmmaInsn, tuningParams)},
      wmmaInsn(wmmaInsn) {}

AccelEmitterParams WmmaEmitter::initAccelEmitterParams(
    WmmaInsn wmmaInsn, RockAccelTuningParamAttrInterface rawTuningParams) {
  AccelEmitterParams params;
  auto tuningParams = rawTuningParams.dyn_cast<WmmaGemmParamsAttr>();
  // Extract relevant tuning parameters
  params.kpacksPerBlock = tuningParams.getKpackPerBlock();
  params.kpack = tuningParams.getKpack();
  params.mPerBlock = tuningParams.getMPerBlock();
  params.nPerBlock = tuningParams.getNPerBlock();
  params.forceUnroll = tuningParams.getForceUnroll();

  params.mRepeats = wmmaInsn.mRepeats;
  params.nRepeats = wmmaInsn.nRepeats;
  params.nResultVectors = 1;
  params.kpackPerThread = params.kpacksPerBlock;
  params.kBase = wmmaInsn.inputLen;
  params.mPerAccel = wmmaInsn.inputLen;
  params.nPerAccel = wmmaInsn.inputLen;
  params.kBasePerThread = params.kpacksPerBlock * params.kpack / params.kBase;

  params.argTypeA = wmmaInsn.argTypeA;
  params.argTypeB = wmmaInsn.argTypeB;
  params.accVectorType = wmmaInsn.retType;
  params.accType = wmmaInsn.retType;

  params.non_useIndexDiffs = false;
  params.useIndexDiffs = false;

  params.nOutputVectors = params.nResultVectors * params.mRepeats *  params.nRepeats;

  return params;
}

Value WmmaEmitter::wrapLDSBufferForLoad(OpBuilder &b, Location loc,
                                        Value buffer, int64_t blockSize,
                                        int64_t dInCopyPerThread,
                                        StringRef dName,
                                        bool rotateDWithK) const {
  auto tuningParams = this->tuningParams.dyn_cast<WmmaGemmParamsAttr>();                               
  // Extract relevant tuning parameters
  int64_t mPerBlock = tuningParams.getMPerBlock();
  int64_t nPerBlock = tuningParams.getNPerBlock();
  int64_t kPerBlock = tuningParams.getKpackPerBlock();
  int64_t mPerWave = tuningParams.getMPerWave();
  int64_t nPerWave = tuningParams.getNPerWave();
  int64_t kPack = tuningParams.getKpack();

  // Extract relevant emitter parameters
  int64_t inputLen = wmmaInsn.inputLen;
  int64_t kpackPerThread = accelEmitterParams.kpackPerThread;
  int64_t dRepeats = (dName == "m" ? accelEmitterParams.mRepeats
                                   : accelEmitterParams.nRepeats);
  int64_t dPerAccel = (dName == "m" ? accelEmitterParams.mPerAccel
                                    : accelEmitterParams.nPerAccel);

  // Extract relevant derived parameters
  StringRef thisWaveDim = dName == "m" ? "wave_m" : "wave_n";
  StringRef otherWaveDim = dName == "m" ? "wave_n" : "wave_m";
  int64_t dWaves = (dName == "m" ? mPerBlock / mPerWave : nPerBlock / nPerWave);
  int64_t dPerBlock = (dName == "m" ? mPerBlock : nPerBlock);
  int64_t mWaves = mPerBlock / mPerWave;
  int64_t nWaves = nPerBlock / nPerWave;

  SmallVector<Attribute> transformAttrs;

  // Compute source offset as
  // sourceOffset = k_i * MN + (laneId % wmmaInputLen) + waveOffset * mn_i;
  TopDownTMBuilder splitTid(b, {"tid", "d_iter", "k_iter"},
                            {blockSize, dPerBlock, kpackPerThread});
  splitTid.merge({"wave_id", "lane_id"}, {0, 1}, "tid",
                 {blockSize / waveSize, waveSize});

  splitTid.passThrough({"d_iter", "k_iter"}, {2, 3}, {"d_iter", "k_iter"});
  TransformMapAttr splitTidAttr = splitTid.get();
  transformAttrs.push_back(splitTidAttr);

  TopDownTMBuilder splitWaveId =
      TopDownTMBuilder::below(splitTid, splitTidAttr);
  splitWaveId.merge({"wave_m", "wave_n"}, {0, 1}, "wave_id", {mWaves, nWaves});
  splitWaveId.passThrough({"lane_id", "d_iter", "k_iter"}, {2, 3, 4},
                          {"lane_id", "d_iter", "k_iter"});
  TransformMapAttr splitWaveIdAttr = splitWaveId.get();
  transformAttrs.push_back(splitWaveIdAttr);

  TopDownTMBuilder replicateLanes =
      TopDownTMBuilder::below(splitWaveId, splitWaveIdAttr);
  replicateLanes.passThrough({"wave_m", "wave_n", "d_iter", "k_iter"},
                             {0, 1, 4, 5},
                             {"wave_m", "wave_n", "d_iter", "k_iter"});

  replicateLanes.merge({"block_id", "block_td"}, {2, 3}, "lane_id",
                       {waveSize / inputLen, inputLen});
  TransformMapAttr replicateLanesAttr = replicateLanes.get();
  transformAttrs.push_back(replicateLanesAttr);

  TopDownTMBuilder toLDSRowCol =
      TopDownTMBuilder::below(replicateLanes, replicateLanesAttr);
  toLDSRowCol.passThrough({"k"}, {1}, {"k_iter"});
  toLDSRowCol.ignore("block_id");
  toLDSRowCol.ignore(otherWaveDim);
  toLDSRowCol.unmerge("d", 0, {"d_iter", thisWaveDim, "block_td"},
                      {dRepeats, dWaves, dPerAccel});

  TransformMapAttr toLDSRowColAttr = toLDSRowCol.get();
  transformAttrs.push_back(toLDSRowColAttr);

  int64_t stride = (kPack == 1 ? dInCopyPerThread : 1);
  auto offset =
      rotateIf(rotateDWithK, toLDSRowCol, toLDSRowColAttr, stride, "d",
               dPerBlock, 0, "k", kPerBlock, {}, {"k"}, transformAttrs);

  offset.unmerge("source_offset", 0, {"k", "d"}, {kPerBlock, dPerBlock});

  TransformMapAttr offsetAttr = offset.get();
  transformAttrs.push_back(offsetAttr);

  ArrayAttr ldsRead = b.getArrayAttr(transformAttrs);
  llvm::outs() <<"buffer : " << buffer << "\n";
  return transform(b, buffer, ldsRead);
}

void WmmaEmitter::emitThreadwiseLoop(OpBuilder &b, Location loc, Value argA,
                                     Value argB, Value bufferC,
                                     Value regCOffset) {
  VectorType vectorType = wmmaInsn.retType;
  auto vectorC = b.create<memref::LoadOp>(loc, vectorType, bufferC, regCOffset);

  auto mfma = b.create<amdgpu::WMMAOp>(loc, vectorType, argA, argB, vectorC,
                                       /*subwordOffset=*/0, /*unsignedA=*/false,
                                       /*unsignedB=*/false, /*clamp=*/true);
  auto vectorD = mfma.getDestD();

  b.create<memref::StoreOp>(loc, vectorD, bufferC, regCOffset);
}

RegsAsMatrixSubTiles WmmaEmitter::computeOutputTransforms(
    PatternRewriter &b, Location loc, int64_t mLen, int64_t nLen,
    int64_t blockSize, ArrayRef<int64_t> bidGridLengths, int64_t inMPerThread,
    int64_t inNPerThread, bool doSwapThreadIterSubDimsForM,
    bool doSwapThreadIterSubDimsForN) {
      
  auto tuningParams = this->tuningParams.dyn_cast<WmmaGemmParamsAttr>(); 
  // Extract relevant tuning parameters
  int64_t mPerBlock = tuningParams.getMPerBlock();
  int64_t nPerBlock = tuningParams.getNPerBlock();
  int64_t mPerWave = tuningParams.getMPerWave();
  int64_t nPerWave = tuningParams.getNPerWave();

  // Extract relevant emitter parameters
  int64_t mRepeats = accelEmitterParams.mRepeats;
  int64_t nRepeats = accelEmitterParams.nRepeats;
  VectorType accVectorType = accelEmitterParams.accVectorType;

  int64_t nWaves = nPerBlock / nPerWave;
  int64_t mWaves = mPerBlock / mPerWave;
  SmallVector<Attribute> transformAttrs;

  // High level code for this loop
  // source: https://gpuopen.com/learn/wmma_on_rdna3/
  //
  // laneIdx = tid % waveSize;
  // for (int item = 0; item < 8; item++){
  //    m_tid = laneIdx / 16;
  //    col = laneIdx % 16;
  //    row = (2*item + m_tid);
  //    c[16 * row+col] = regC[item];
  // }
  //
  //

  int64_t retNumElements = accVectorType.getNumElements();
  int64_t wavesInKernelBlock = blockSize / waveSize;

  // m_tid is liimited to 0 or 1 (or 0,1,2,3 for wave64). Basically every
  // workitem is computing 8 values, which represent a sub-column of a 16x16
  // tile. For workitems 0 to 15 m_tid is 0 and they write to 0,2,4,...,14.  For
  // workitems 16 to 31 m_tid is 1 and they write at positions 1,3,5,..,15 .
  // This is outlined in https://gpuopen.com/learn/wmma_on_rdna3/
  //
  // Note that `wave_m` and `wave_n` are strided by the inputLen, i.e., all the
  // waves will compute next to each other and then they will move to the next
  // subtile in the workgroup
  SmallVector<StringRef, 5> dimNamesM{/*0=*/"m_block",
                                      /*1=*/"rep_i",
                                      /*2=*/"wave_m",
                                      /*3=*/"item_i",
                                      /*4=*/"m_tid"};
  SmallVector<int64_t, 5> orderedDimStridesM{/*0=*/mPerBlock,
                                             /*1=*/mWaves * wmmaInsn.inputLen,
                                             /*2=*/wmmaInsn.inputLen,
                                             /*3=*/wmmaInsn.outputStride,
                                             /*4=*/1};
  SmallVector<int64_t, 7> dimSizesM;
  convertDimStridestoSizes(orderedDimStridesM, mLen, dimSizesM);

  SmallVector<StringRef, 5> dimNamesN{/*0=*/"n_block",
                                      /*1=*/"rep_j",
                                      /*2=*/"wave_n",
                                      /*3=*/"n_tid"};
  SmallVector<int64_t, 5> orderedDimStridesN{/*0=*/nPerBlock,
                                             /*1=*/nWaves * wmmaInsn.inputLen,
                                             /*2=*/wmmaInsn.inputLen,
                                             /*3=*/1};
  SmallVector<int64_t, 7> dimSizesN;
  convertDimStridestoSizes(orderedDimStridesN, nLen, dimSizesN);

  RegsAsMatrixSubTiles ret;
  {
    // Create views as gridwise sub-tile of C
    TopDownTMBuilder splitMemoryCoords(
        b, {"g_block", "m_block", "n_block", "tid", "item"},
        {bidGridLengths[0], bidGridLengths[1], bidGridLengths[2], blockSize,
         mRepeats * nRepeats * retNumElements},
        loc);
    splitMemoryCoords.passThrough({"g_block", "m_block", "n_block"});
    splitMemoryCoords.merge({"wave_m", "wave_n", "m_tid", "n_tid"},
                            {3, 4, 5, 6}, "tid",
                            {wavesInKernelBlock / nWaves, nWaves,
                             waveSize / wmmaInsn.inputLen, wmmaInsn.inputLen});
    splitMemoryCoords.merge({"rep_i", "rep_j", "item_i"}, {7, 8, 9}, "item",
                            {mRepeats, nRepeats, retNumElements});
    TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();

    auto toMatrixC =
        TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);
    toMatrixC.passThrough({"gemmG"}, {0}, {"g_block"});
    toMatrixC.unmerge("gemmM", 1, dimNamesM, dimSizesM);
    toMatrixC.unmerge("gemmN", 2, dimNamesN, dimSizesN);
    SmallVector<Attribute> transformAttrs{splitMemoryCoordsAttr};
    mlir::rock::swapThreadIdAndIteration(
        toMatrixC, /*mBlocks=*/bidGridLengths[1], /*nBlocks=*/bidGridLengths[2],
        inMPerThread, inNPerThread, mPerBlock, nPerBlock,
        doSwapThreadIterSubDimsForM, doSwapThreadIterSubDimsForN,
        /**isBlockwise=*/false, transformAttrs);

    ret.gridSubTile = b.getArrayAttr(transformAttrs);
  }

  {
    // Create views as blockwise sub-tile of C
    TopDownTMBuilder splitMemoryCoords(
        b, {"tid", "item"}, {blockSize, mRepeats * nRepeats * retNumElements},
        loc);
    splitMemoryCoords.merge({"wave_m", "wave_n", "m_tid", "n_tid"},
                            {0, 1, 2, 3}, "tid",
                            {wavesInKernelBlock / nWaves, nWaves,
                             waveSize / wmmaInsn.inputLen, wmmaInsn.inputLen});
    splitMemoryCoords.merge({"rep_i", "rep_j", "item_i"}, {4, 5, 6}, "item",
                            {mRepeats, nRepeats, retNumElements});
    TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();

    auto toMatrixC =
        TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);
    toMatrixC.unmerge("gemmM", 0, ArrayRef<StringRef>{dimNamesM}.slice(1),
                      ArrayRef<int64_t>{dimSizesM}.slice(1));
    toMatrixC.unmerge("gemmN", 1, ArrayRef<StringRef>{dimNamesN}.slice(1),
                      ArrayRef<int64_t>{dimSizesN}.slice(1));
    SmallVector<Attribute> transformAttrs{splitMemoryCoordsAttr};
    mlir::rock::swapThreadIdAndIteration(
        toMatrixC, /*mBlocks=*/bidGridLengths[1], /*nBlocks=*/bidGridLengths[2],
        inMPerThread, inNPerThread, mPerBlock, nPerBlock,
        doSwapThreadIterSubDimsForM, doSwapThreadIterSubDimsForN,
        /**isBlocwise=*/true, transformAttrs);
    ret.blockSubTile = b.getArrayAttr(transformAttrs);
  }

  {
    // Create views as threadwise sub-tile of C
    TopDownTMBuilder splitMemoryCoords(
        b, {"item"}, {mRepeats * nRepeats * retNumElements}, loc);
    splitMemoryCoords.merge({"rep_i", "rep_j", "item_i"}, {0, 1, 2}, "item",
                            {mRepeats, nRepeats, retNumElements});
    TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();

    auto toMatrixC =
        TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);
    toMatrixC.unmerge("gemmM", 0, {dimNamesM[1], dimNamesM[3]},
                      {dimSizesM[1], dimSizesM[3]});
    toMatrixC.unmerge("gemmN", 1, {dimNamesN[1]}, {dimSizesN[1]});
    TransformMapAttr toMatrixCAttr = toMatrixC.get();
    ret.threadSubTile = b.getArrayAttr({splitMemoryCoordsAttr, toMatrixCAttr});
  }

  return ret;
}

std::unique_ptr<AccelEmitter>
AccelEmitter::select(GemmFeatures features, Type dataTypeA, Type dataTypeB,
                     StringRef arch, 
                     RockAccelTuningParamAttrInterface rawTuningParams) {
  bool isMfma = rock::bitEnumContainsAll(features, GemmFeatures::mfma);
  bool isWmma = rock::bitEnumContainsAll(features, GemmFeatures::wmma);
  if (isMfma) {
    auto tuningParams = rawTuningParams.dyn_cast<XdlopsGemmParamsAttr>();
    auto maybeMfmaInsnGroup = MfmaInsnGroup::select(dataTypeA, dataTypeB, arch,
                                                    tuningParams.getMPerWave(),
                                                    tuningParams.getNPerWave());
    if (failed(maybeMfmaInsnGroup)) {
      return nullptr;
    }
    return std::make_unique<MfmaEmitter>(*maybeMfmaInsnGroup, arch,
                                         tuningParams);
  } else if (isWmma) {
      auto tuningParams = rawTuningParams.dyn_cast<WmmaGemmParamsAttr>();
      int64_t waveSize = rock::lookupArchInfo(arch).waveSize;
      auto maybeWmmaInsnGroup = WmmaInsn::select(dataTypeA, dataTypeB, waveSize,
                                                tuningParams.getMPerWave(),
                                                tuningParams.getNPerWave());
      if (failed(maybeWmmaInsnGroup)) {
        return nullptr;
      }
      return std::make_unique<WmmaEmitter>(*maybeWmmaInsnGroup, arch,
                                         tuningParams);
  } else {
      auto tuningParams = rawTuningParams.dyn_cast<GeneralGemmParamsAttr>();
      auto maybeFmaInsnGroup = FmaInsn::select(dataTypeA, dataTypeB, arch);
      if (failed(maybeFmaInsnGroup)) {
        return nullptr;
      }
      llvm::outs() <<"\nDEBUG: Creating FmeEmitter!\n";
      return std::make_unique<FmaEmitter>(*maybeFmaInsnGroup, arch, tuningParams);
  }
}
