#ifndef YLLM_SERVE_RANK_H
#define YLLM_SERVE_RANK_H

#include "config.h"

/* yllm rank: 常驻推理单元(公用 rank 池的最小计算单元) */
int cmd_rank(ServeConfig* cfg);

#endif
