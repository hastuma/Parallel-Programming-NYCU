#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float vx;
  __pp_vec_float vresult;
  __pp_vec_float vone = _pp_vset_float(1.f);
  __pp_vec_float vclamp = _pp_vset_float(9.999999f);

  __pp_vec_int ve;              
  __pp_vec_int vone_int = _pp_vset_int(1);
  __pp_vec_int vzero_int = _pp_vset_int(0);

  __pp_mask maskAll, maskExpGt0;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    
    int remaining = N - i;
    int actual_width = (remaining < VECTOR_WIDTH) ? remaining : VECTOR_WIDTH;// 確保類似n=3 小於width的情況
    
    if (remaining >= VECTOR_WIDTH)
      maskAll = _pp_init_ones();
    else
      maskAll = _pp_init_ones(remaining);

    
    vx = _pp_vset_float(1.0f);  
    ve = _pp_vset_int(0);      
    
    // 只load要用的
    for (int j = 0; j < actual_width; j++) {
      if (i + j < N) {
        // 手動load不然n=3 <width 那邊會有bad alloc
        vx.value[j] = values[i + j];
        ve.value[j] = exponents[i + j];
      }
    }

    _pp_vmove_float(vresult, vone, maskAll);
    _pp_vgt_int(maskExpGt0, ve, vzero_int, maskAll);
    
    int max_iterations = 20; 
    int iteration_count = 0;
    
    while (_pp_cntbits(maskExpGt0) > 0 && iteration_count < max_iterations)
    {
      _pp_vmult_float(vresult, vresult, vx, maskExpGt0);
      _pp_vsub_int(ve, ve, vone_int, maskExpGt0);
      _pp_vgt_int(maskExpGt0, ve, vzero_int, maskAll);
      iteration_count++;
    }

    // clamp result to <= 9.999999
    __pp_mask maskClamp;
    _pp_vgt_float(maskClamp, vresult, vclamp, maskAll);
    _pp_vmove_float(vresult, vclamp, maskClamp);
    _pp_vstore_float(output + i, vresult, maskAll);
  }
}

float arraySumVector(float *values, int N)
// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
{
  __pp_vec_float vsum = _pp_vset_float(0.0f);
  __pp_vec_float vtemp;
  __pp_mask maskAll = _pp_init_ones();
  
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    _pp_vload_float(vtemp, values + i, maskAll);
    _pp_vadd_float(vsum, vsum, vtemp, maskAll);
  }
  
  __pp_vec_float vcurrent = vsum;
  int current_width = VECTOR_WIDTH;
  
  while (current_width > 1) 
  {
    if (current_width == 2) 
    {
      // 最後一次 只要hadd
      _pp_hadd_float(vcurrent, vcurrent);
      break;
    } 
    else 
    {
     //旁邊的加起來
      __pp_vec_float vhadd_result;
      _pp_hadd_float(vhadd_result, vcurrent);
      
      if (current_width > 2) 
      {
        _pp_interleave_float(vcurrent, vhadd_result);
      } 
      else 
      {
        vcurrent = vhadd_result;
      }
      current_width /= 2;
    }
  }
  
  return vcurrent.value[0];
}