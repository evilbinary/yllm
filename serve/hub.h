#ifndef YLLM_SERVE_HUB_H
#define YLLM_SERVE_HUB_H

#include "config.h"

/* yllm hub: 合并模式(supervisor+router+server 同一进程不同线程) */
int cmd_hub(ServeConfig* cfg);

#endif
