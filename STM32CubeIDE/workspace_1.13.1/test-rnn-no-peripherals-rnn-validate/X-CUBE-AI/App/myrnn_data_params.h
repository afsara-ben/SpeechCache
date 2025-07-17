/**
  ******************************************************************************
  * @file    myrnn_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    Mon Aug 21 15:37:15 2023
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef MYRNN_DATA_PARAMS_H
#define MYRNN_DATA_PARAMS_H
#pragma once

#include "ai_platform.h"

/*
#define AI_MYRNN_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_myrnn_data_weights_params[1]))
*/

#define AI_MYRNN_DATA_CONFIG               (NULL)


#define AI_MYRNN_DATA_ACTIVATIONS_SIZES \
  { 968, }
#define AI_MYRNN_DATA_ACTIVATIONS_SIZE     (968)
#define AI_MYRNN_DATA_ACTIVATIONS_COUNT    (1)
#define AI_MYRNN_DATA_ACTIVATION_1_SIZE    (968)



#define AI_MYRNN_DATA_WEIGHTS_SIZES \
  { 17156, }
#define AI_MYRNN_DATA_WEIGHTS_SIZE         (17156)
#define AI_MYRNN_DATA_WEIGHTS_COUNT        (1)
#define AI_MYRNN_DATA_WEIGHT_1_SIZE        (17156)



#define AI_MYRNN_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_myrnn_activations_table[1])

extern ai_handle g_myrnn_activations_table[1 + 2];



#define AI_MYRNN_DATA_WEIGHTS_TABLE_GET() \
  (&g_myrnn_weights_table[1])

extern ai_handle g_myrnn_weights_table[1 + 2];


#endif    /* MYRNN_DATA_PARAMS_H */
