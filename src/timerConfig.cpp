#include "cnn_from_scratch/timerConfig.h"

// Global timer variable. Not great practice, but very convenient for
// this one use case
#ifdef TIMEIT
cpp_timer::Timer global_timer;
#endif
