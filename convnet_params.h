/*******************************************************************
@file convnet_params.h
*  @brief variable prototypes for model parameters and amax values
*
*
*  @author Benjamin Fuhrer
*
*******************************************************************/
#ifndef CONVNET_PARAMS
#define CONVNET_PARAMS

#define INPUT_DIM 3072
#define OUTPUT_DIM 10

#include <stdint.h>


// quantization/dequantization constants
extern const int layer_1_s_x;
extern const int layer_1_s_x_inv;
extern const int layer_1_s_w_inv;
extern const int layer_2_s_x;
extern const int layer_2_s_x_inv;
extern const int layer_2_s_w_inv;
extern const int layer_3_s_x;
extern const int layer_3_s_x_inv;
extern const int layer_3_s_w_inv;
extern const int layer_4_s_x;
extern const int layer_4_s_x_inv;
extern const int layer_4_s_w_inv;
extern const int layer_5_s_x;
extern const int layer_5_s_x_inv;
extern const int layer_5_s_w_inv;
extern const int layer_6_s_x;
extern const int layer_6_s_x_inv;
extern const int layer_6_s_w_inv;
extern const int layer_7_s_x;
extern const int layer_7_s_x_inv;
extern const int layer_7_s_w_inv;
extern const int layer_8_s_x;
extern const int layer_8_s_x_inv;
extern const int layer_8_s_w_inv;
extern const int layer_9_s_x;
extern const int layer_9_s_x_inv;
extern const int layer_9_s_w_inv;
extern const int layer_10_s_x;
extern const int layer_10_s_x_inv;
extern const int layer_10_s_w_inv;
extern const int layer_11_s_x;
extern const int layer_11_s_x_inv;
extern const int layer_11_s_w_inv;
extern const int layer_12_s_x;
extern const int layer_12_s_x_inv;
extern const int layer_12_s_w_inv;
extern const int layer_13_s_x;
extern const int layer_13_s_x_inv;
extern const int layer_13_s_w_inv;
extern const int layer_14_s_x;
extern const int layer_14_s_x_inv;
extern const int layer_14_s_w_inv;
// Layer quantized parameters
extern const int layer_1_weight[1728];
extern const int layer_2_weight[36864];
extern const int layer_3_weight[73728];
extern const int layer_4_weight[147456];
extern const int layer_5_weight[294912];
extern const int layer_6_weight[589824];
extern const int layer_7_weight[589824];
extern const int layer_8_weight[1179648];
extern const int layer_9_weight[2359296];
extern const int layer_10_weight[2359296];
extern const int layer_11_weight[2359296];
extern const int layer_12_weight[2359296];
extern const int layer_13_weight[2359296];
extern const int layer_14_weight[5120];

#endif // end of CONVNET_PARAMS
