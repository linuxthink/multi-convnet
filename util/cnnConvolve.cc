#include "mex.h"
#include "matrix.h"
#include <omp.h>
#include <math.h>
#include <string.h>
#include <algorithm>

bool convolve2D(double *in, int inDim, double *kernel, int kernelDim,
                double *out) {
// This function is from the following source:
// AUTORIGHTS
// -------------------------------------------------------
// Copyright (C) 2011-2012 Ross Girshick
// This file is part of the voc-releaseX code
// (http://people.cs.uchicago.edu/~rbg/latent/)
// and is available under the terms of an MIT-like license
// provided in COPYING. Please retain this notice and
// COPYING if you use this file (or a portion of it) in
// your project.
// -------------------------------------------------------
  int outDim = inDim - kernelDim + 1;

  double *A_src = in;
  double *B_src = kernel;
  double *dst = out;
  // start convolution
  for (int x = 0; x < outDim; x++) {
    for (int y = 0; y < outDim; y++) {
      double val = 0;
      for (int xp = 0; xp < kernelDim; xp++) {
        double *A_off = A_src + (x+xp)*inDim + y;
        double *B_off = B_src + xp*kernelDim;
        switch(kernelDim) {
          case 20: val += A_off[19] * B_off[19];
          case 19: val += A_off[18] * B_off[18];
          case 18: val += A_off[17] * B_off[17];
          case 17: val += A_off[16] * B_off[16];
          case 16: val += A_off[15] * B_off[15];
          case 15: val += A_off[14] * B_off[14];
          case 14: val += A_off[13] * B_off[13];
          case 13: val += A_off[12] * B_off[12];
          case 12: val += A_off[11] * B_off[11];
          case 11: val += A_off[10] * B_off[10];
          case 10: val += A_off[9] * B_off[9];
          case 9: val += A_off[8] * B_off[8];
          case 8: val += A_off[7] * B_off[7];
          case 7: val += A_off[6] * B_off[6];
          case 6: val += A_off[5] * B_off[5];
          case 5: val += A_off[4] * B_off[4];
          case 4: val += A_off[3] * B_off[3];
          case 3: val += A_off[2] * B_off[2];
          case 2: val += A_off[1] * B_off[1];
          case 1: val += A_off[0] * B_off[0];
            break;
          default:
            for (int yp = 0; yp < kernelDim; yp++) {
              val += *(A_off++) * *(B_off++);
            }
        }
      }
      *(dst++) += val;
    }
  }
  return true;
}

// -------------------------------------------------------
// matlab entry point
// outMap = fconv_c(inMap, kernel, inChannel, outChannel, table, shape)
// inMap: inDim * inDim * inChannel
// kernel: kernelDim * kernelDim * kernelNum
// table: table: [inInd, outInd, kernelInd;
//                          ...  , ...   , ...      ];
// shape: 'valid' or 'full'
// -------------------------------------------------------
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // parse inputs ====================
  if (nrhs != 8)
    mexErrMsgTxt("Wrong number of inputs");
  if (nlhs != 1)
    mexErrMsgTxt("Wrong number of outputs");
  // mexPrintf("enter convolution\n");

  // get shape
  if (mxIsChar(prhs[4]) != 1) {
    mexErrMsgTxt("shape error not string");
  }
  const char *shapeBuf = mxArrayToString(prhs[4]);
  int shapeBufLen = (mxGetM(prhs[4]) * mxGetN(prhs[4])) + 1;
  bool isFull = 0;
  if (!strcmp(shapeBuf, "full")) {
    isFull = 1;
  } else if (!strcmp(shapeBuf, "valid")) {
    isFull = 0;
  } else {
    mexErrMsgTxt("shape option error");
  }
  // mexPrintf("buf len: %d, %d\n", shapeBufLen, isFull);

  // get rotateInMap
  const mxArray *rotateInMapArray = prhs[5];
  const int rotateInMapDimsNum = (int)mxGetNumberOfDimensions(rotateInMapArray);
  if (rotateInMapDimsNum != 2 ||
      mxGetClassID(rotateInMapArray) != mxLOGICAL_CLASS) {
    mexErrMsgTxt("rotateInMap option error, require bool");
  }
  const bool *rotateInMapData = (bool *)mxGetData(rotateInMapArray);
  const bool rotateInMap = rotateInMapData[0];

  const mxArray *rotateKernelArray = prhs[6];
  const int rotateKernelDimsNum = (int)mxGetNumberOfDimensions(rotateKernelArray);
  if (rotateKernelDimsNum != 2 ||
      mxGetClassID(rotateKernelArray) != mxLOGICAL_CLASS) {
    mexErrMsgTxt("rotateKernel option error, require bool");
  }
  const bool *rotateKernelData = (bool *)mxGetData(rotateKernelArray);
  const bool rotateKernel = rotateKernelData[0];
  // mexPrintf("rotateInMap: %d, rotateKernel: %d\n", rotateInMap, rotateKernel);

  // get kernel
  const mxArray *kernelArray = prhs[1];
  const mwSize *kernelDims = mxGetDimensions(kernelArray);
  const int kernelDimsNum = (int)mxGetNumberOfDimensions(kernelArray);
  if ((kernelDimsNum != 3 && kernelDimsNum != 2) ||
      mxGetClassID(kernelArray) != mxDOUBLE_CLASS) {
    mexErrMsgTxt("kernel error");
  }
  double *kernelDataOriginal = (double *)mxGetData(kernelArray);
  const int kernelDim = (int)kernelDims[0];
  int kernelNum = 0;
  if (kernelDimsNum == 3) {
    kernelNum = (int)kernelDims[2];
  } else {
    kernelNum = 1;
  }
  const int kernelArea = kernelDim * kernelDim;

  // if rotate kernel -> NOT rotate!
  // if NOT rotate kernel -> flip up-down and flip left-right kernel
  double *kernelData = NULL;
  if (rotateKernel) {
    // do nothing
    kernelData = kernelDataOriginal;
  } else {
    kernelData = (double *)mxCalloc(kernelArea * kernelNum, sizeof(double));
    memcpy(kernelData, kernelDataOriginal, kernelArea * kernelNum *
           sizeof(double));
    for (int kernelInd = 0; kernelInd < kernelNum; kernelInd++) {
      std::reverse(&kernelData[kernelArea * kernelInd],
                   &kernelData[kernelArea * (kernelInd + 1)]);
    }
  }

  // get inMap
  const mxArray *inMapArray = prhs[0];
  const mwSize *inMapDims = mxGetDimensions(inMapArray);
  const int inMapDimsNum = mxGetNumberOfDimensions(inMapArray);
  if ((inMapDimsNum != 3 && inMapDimsNum != 2) ||
      mxGetClassID(inMapArray) != mxDOUBLE_CLASS) {
    mexErrMsgTxt("inMap error");
  }
  double *inMapDataOriginal = (double *)mxGetData(inMapArray);
  const int inDimOriginal = (int)inMapDims[0];
  const int inMapAreaOriginal = inDimOriginal * inDimOriginal;
  int inChannel = 0;
  if (inMapDimsNum == 3) {
    inChannel = (int)inMapDims[2];
  } else {
    inChannel = 1;
  }

  // if rotateInMap -> rotate
  // if not rotateInMap -> NOT rotate
  // store inMapDataOriginal -> inMapDataBuff, then -> inMapData for final use
  double *inMapDataBuff = NULL;
  if (rotateInMap) {
    inMapDataBuff = (double *)mxCalloc(inMapAreaOriginal * inChannel,
                                       sizeof(double));
    memcpy(inMapDataBuff, inMapDataOriginal, inMapAreaOriginal * inChannel *
           sizeof(double));
    for (int inChannelInd = 0; inChannelInd < inChannel; inChannelInd++) {
      std::reverse(&inMapDataBuff[inMapAreaOriginal * inChannelInd],
                   &inMapDataBuff[inMapAreaOriginal * (inChannelInd + 1)]);
    }
  } else {
    inMapDataBuff = inMapDataOriginal;
  }

  // pad inMap if necessary
  int inDim = 0;
  double *inMapData = NULL;
  int inMapArea = 0;
  if (isFull) {
    inDim = inDimOriginal + (kernelDim - 1) * 2;
    inMapArea = inDim * inDim;
    // allocate memory
    inMapData = (double *)mxCalloc(inDim * inDim * inChannel, sizeof(double));
    // pad inMap
    int startInd = inDim * (kernelDim - 1) + (kernelDim - 1);
    for (int inChannelInd = 0; inChannelInd < inChannel; inChannelInd++) {
      for (int colInd = 0; colInd < inDimOriginal; colInd++) {
        memcpy(&inMapData[startInd + colInd * inDim + inChannelInd * inMapArea],
               &inMapDataBuff[0 + colInd * inDimOriginal +
                              inChannelInd * inMapAreaOriginal],
               inDimOriginal * sizeof(double));
      }
    }
  } else {
    inDim = inDimOriginal;
    inMapArea = inMapAreaOriginal;
    inMapData = inMapDataBuff;
  }

  // get outchannel
  const mxArray *outChannelArray = prhs[2];
  double *outChannelData = (double *)mxGetData(outChannelArray);
  const int outChannel = (int)outChannelData[0];

  // get connectionTable
  const mxArray *tableArray = prhs[3];
  double *tableData = (double *)mxGetData(tableArray);
  const mwSize *tableDims = mxGetDimensions(tableArray);
  const int tableDimsNum = mxGetNumberOfDimensions(tableArray);
  if ((tableDimsNum != 2) ||
      mxGetClassID(tableArray) != mxDOUBLE_CLASS)
    mexErrMsgTxt("table error");
  const int tableHeight = (int)tableDims[0];

  // get threadNum
  double *threadNumData = (double *)mxGetData(prhs[7]);
  int threadNum = (int)threadNumData[0];
  // mexPrintf("threadNum input: %d\n", threadNum);

  // init output ====================
  int outDim = inDim - kernelDim + 1;
  if (outDim <= 0) {
    mexErrMsgTxt("outDim error");
  }
  const mwSize outMapDims[3] = {(mwSize)outDim, (mwSize)outDim,
      (mwSize)outChannel};
  plhs[0] = mxCreateNumericArray(3, outMapDims, mxDOUBLE_CLASS, mxREAL);
  // mexPrintf("inDim:%d, inChannel:%d, kernelDim:%d, tableHeight: %d\n",
  //  inDim, inChannel, kernelDim, tableHeight);
  // mexPrintf("outMapDims: %d %d %d, outChannel: %d, outDim %d\n", outMapDims[0],
  //    outMapDims[1], outMapDims[2]);
  double *outMapData = (double *)mxGetData(plhs[0]);
  const int outmapArea = outDim * outDim;

  // init openmp ====================
  int coreNum = omp_get_num_procs();
  if (threadNum > coreNum * 5) {
    threadNum = coreNum * 5;
  }
  omp_set_num_threads(threadNum);
  // mexPrintf("threadNum actual: %d\n", threadNum);

  // perform convolution ====================
  // outInd, inInd, kernelInd: 0 indexed
  double *singleOutMapEmptyData = (double *)mxCalloc(outDim * outDim,
                                                     sizeof(double));
  double *tempouTmaPdaTa = (double *)mxCalloc(outDim * outDim * outChannel,
                                                sizeof(double));
  #pragma omp parallel for
  for (int outInd = 0; outInd < outChannel; outInd++) {
    // start convolution
    for (int tableInd = tableHeight; tableInd < 2 * tableHeight; tableInd++) {
      // mexPrintf("%d %d\n", (int)tableData[tableInd], outInd + 1);
      if ((int)tableData[tableInd] == outInd + 1) {
        int inInd = tableData[tableInd - tableHeight] - 1;
        int kernelInd = tableData[tableInd + tableHeight] - 1;
        // mexPrintf("%d,%d -> %d\n", inInd, kernelInd, outInd);
        // mexPrintf("%d %d\n", inInd, kernelInd);
        // clear current singleOutMap
        memcpy(&tempouTmaPdaTa[outmapArea*outInd], singleOutMapEmptyData,
               outDim * outDim * sizeof(double));
        // call convolve2D
        convolve2D(&inMapData[inMapArea*inInd], inDim,
            &kernelData[kernelArea*kernelInd], kernelDim,
            &tempouTmaPdaTa[outmapArea*outInd]);
        // transfer result from singleOutMap to outMapData
        for (int i = outmapArea*outInd; i < outmapArea*(outInd+1); i++) {
          outMapData[i] = outMapData[i] + tempouTmaPdaTa[i];
        }
      }
    }
  }

  // clean up ====================
  if (isFull) {
    mxFree(inMapData);
  }
  if (!rotateKernel) {
    mxFree(kernelData);
  }
  if (rotateInMap) {
    mxFree(inMapDataBuff);
  }
  if(singleOutMapEmptyData != NULL) {
    mxFree(singleOutMapEmptyData);
  }
  if (tempouTmaPdaTa != NULL) {
    mxFree(tempouTmaPdaTa);
  }
}

