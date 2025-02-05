/*
 * Copyright (c) 2013 - 2015, Freescale Semiconductor, Inc.
 * Copyright 2016-2017 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "fsl_device_registers.h"
#include "fsl_debug_console.h"
#include "pin_mux.h"
#include "clock_config.h"
#include "board.h"
#include "fsl_uart.h"
#include "fsl_ftm.h"

#define TARGET_UART UART4
#define FTM_MOTOR FTM0
#define FTM_CHANNEL_DC_MOTOR kFTM_Chnl_0
#define FTM_CHANNEL_SERVO_MOTOR kFTM_Chnl_3

volatile char ch;
volatile int new_char = 0;
char inputBuffer[] = {0,0,0,0,0,0,0,0,0,0};


/*******************************************************************************
 * Definitions
 ******************************************************************************/


/*******************************************************************************
 * Prototypes
 ******************************************************************************/

/*******************************************************************************
 * Code
 ******************************************************************************/
/*!
 * @brief Main function
 */



void setupPWM()	{
	ftm_config_t ftmInfo;
	ftm_chnl_pwm_signal_param_t ftmParam;
	ftm_pwm_level_select_t pwmLevel = kFTM_HighTrue;

	ftmParam.chnlNumber 			= FTM_CHANNEL_DC_MOTOR;
	ftmParam.level 					= pwmLevel;
	ftmParam.dutyCyclePercent 		= 7;
	ftmParam.firstEdgeDelayPercent 	= 0U;
	ftmParam.enableComplementary 	= false;
	ftmParam.enableDeadtime 		= false;

	FTM_GetDefaultConfig(&ftmInfo);
	ftmInfo.prescale = kFTM_Prescale_Divide_128;

	FTM_Init(FTM_MOTOR, &ftmInfo);
	FTM_SetupPwm(FTM_MOTOR, &ftmParam, 1U, kFTM_EdgeAlignedPwm, 50U, CLOCK_GetFreq(kCLOCK_BusClk));
	FTM_StartTimer(FTM_MOTOR, kFTM_SystemClock);

	ftmParam.chnlNumber 			= FTM_CHANNEL_SERVO_MOTOR;
	ftmParam.level 					= pwmLevel;
	ftmParam.dutyCyclePercent 		= 7;
	ftmParam.firstEdgeDelayPercent 	= 0U;
	ftmParam.enableComplementary 	= false;
	ftmParam.enableDeadtime 		= false;

	FTM_GetDefaultConfig(&ftmInfo);
	ftmInfo.prescale = kFTM_Prescale_Divide_128;

	FTM_Init(FTM_MOTOR, &ftmInfo);
	FTM_SetupPwm(FTM_MOTOR, &ftmParam, 1U, kFTM_EdgeAlignedPwm, 50U, CLOCK_GetFreq(kCLOCK_BusClk));
	FTM_StartTimer(FTM_MOTOR, kFTM_SystemClock);
}


void updatePWM_dutyCycle(ftm_chnl_t channel, float dutyCycle)	{
	uint32_t cnv, cnvFirstEdge = 0, mod;
	/* The CHANNEL_COUNT macro returns -1 if it cannot match the FTM instance */
	assert(-1 != FSL_FEATURE_FTM_CHANNEL_COUNTn(FTM_MOTOR));
	mod = FTM_MOTOR->MOD;
	if (dutyCycle == 0U)	{
		/* Signal stays low */
		cnv = 0;
	}
	else
	{
		cnv = mod * dutyCycle;
		/* For 100% duty cycle */
		if (cnv >= mod)	{
			cnv = mod + 1U;
		}
	}
	FTM_MOTOR->CONTROLS[channel].CnV = cnv;
}

void setupUART()
{
	uart_config_t config;
	UART_GetDefaultConfig(&config);
	config.baudRate_Bps = 57600;
	config.enableTx = true;
	config.enableRx = true;
	config.enableRxRTS = true;
	config.enableTxCTS = true;
	UART_Init(TARGET_UART, &config, CLOCK_GetFreq(kCLOCK_BusClk));

	/******** Enable Interrupts *********/
	UART_EnableInterrupts(TARGET_UART, kUART_RxDataRegFullInterruptEnable);
	EnableIRQ(UART4_RX_TX_IRQn);
}

void UART4_RX_TX_IRQHandler()	{
	UART_GetStatusFlags(TARGET_UART);
	ch = UART_ReadByte(TARGET_UART);

	printf("UART Data Incoming");

	if(new_char == 10) {
		return;
	}
	printf("Current new_char = %d\n", new_char);
	printf("New Data = %c\n", ch);

	inputBuffer[new_char] = ch;
	printf("New Data added to the Buffer\n");

	new_char = new_char + 1;
	printf("Updated new_char = %d\n", new_char);
}



/*int main(void)			//Problem 2
{
    char ch;
    //char ch_a;
    int input;
    int servoInput;
	float dutyCycle;
	float servoDutyCycle;

    char txbuff[] = "H9\r\n";
    char inputBuffer[] = {0,0,0,0,0,0,0,0,0,0};

    // Init board hardware.
    BOARD_InitBootPins();
    BOARD_InitBootClocks();
    BOARD_InitDebugConsole();

    setupPWM();
    setupUART();

    //delay******************
    for(volatile int i = 0U; i < 1000000; i++)
    __asm("NOP");

    updatePWM_dutyCycle(FTM_CHANNEL_DC_MOTOR, 0.0615);
    FTM_SetSoftwareTrigger(FTM_MOTOR, true);
    PRINTF("%s", txbuff);

    //Writing FROM the car
    UART_WriteBlocking(TARGET_UART, txbuff, sizeof(txbuff) - 1);

    while (1)
    {
    	//Reading TO the car
    	UART_ReadBlocking(TARGET_UART, &inputBuffer[0], 10);
    	//printf("HI");
    	//PRINTF("%c\r\n", ch);

    	sscanf(inputBuffer,"%2d%2d", &input, &servoInput);
    	printf("Motor input = %d and Servo input = %d Buffer = %.10s\n", input, servoInput, inputBuffer);


    	dutyCycle = input * 0.025f/100.0f + 0.0615;
    	servoDutyCycle = servoInput * 0.025f / 45.0f + 0.0615;

    	updatePWM_dutyCycle(FTM_CHANNEL_DC_MOTOR, dutyCycle);
    	updatePWM_dutyCycle(FTM_CHANNEL_SERVO_MOTOR, servoDutyCycle);
    	FTM_SetSoftwareTrigger(FTM_MOTOR, true);
    }*/


int main(void)			//Problem 3
{
	char txbuff[] = "Hello World\r\n";

    int input;
    int servoInput;
	float dutyCycle;
	float servoDutyCycle;

	//Init board hardware.
	BOARD_InitBootPins();
	BOARD_InitBootClocks();
	BOARD_InitDebugConsole();

    setupPWM();
    setupUART();

	//********************Delay********************
	for(volatile int i = 0U; i < 10000000; i++)
		__asm("NOP");

	PRINTF("%s", txbuff);
	UART_WriteBlocking(TARGET_UART, txbuff, sizeof(txbuff) - 1);
	while (1)
	{
		if(new_char == 10)
		{
			new_char = 0;
			sscanf(inputBuffer,"%2d%2d", &input, &servoInput);
			printf("Motor input = %d and Servo input = %d Buffer = %.10s\n", input, servoInput, inputBuffer);
			//PRINTF("%c\r\n", ch);

			dutyCycle = input * 0.025f/100.0f + 0.0615;
			servoDutyCycle = servoInput * 0.025f / 45.0f + 0.0615;

			updatePWM_dutyCycle(FTM_CHANNEL_DC_MOTOR, dutyCycle);
			updatePWM_dutyCycle(FTM_CHANNEL_SERVO_MOTOR, servoDutyCycle);
			FTM_SetSoftwareTrigger(FTM_MOTOR, true);
		}
	}
}
