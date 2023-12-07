#pragma once

#ifdef TIMEIT

#include "cpp_timer/Timer.h"

extern cpp_timer::Timer global_timer;

#define TIMER_INSTANCE global_timer
#define TIC(x) global_timer.tic(x)
#define TOC(x) global_timer.toc(x)
#define stic(x) auto _ = global_timer.scopedTic(x)

#else

#define TIC(x)
#define TOC(x)
#define stic(x)
#define STIC

#endif
