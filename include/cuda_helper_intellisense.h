#pragma once

// ========== Only For Intellisense =============

#ifdef __INTELLISENSE__
	// ref: https://stackoverflow.com/a/27992604/17132546
	#define KERNEL_ARGS(args) 
	#define KERNEL_ARGS2(grid, block)
	#define KERNEL_ARGS3(grid, block, sh_mem)
	#define KERNEL_ARGS4(grid, block, sh_mem, stream)

#else
	#define KERNEL_ARGS(args) <<< args >>>
	#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
	#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
	#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>

#endif
