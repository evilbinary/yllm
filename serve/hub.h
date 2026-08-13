#ifndef YLLM_SERVE_CONTROL_H
#define YLLM_SERVE_CONTROL_H

/* yllm control: 合并模式(supervisor+router+server 同一进程不同线程) */
int cmd_hub(int argc, char** argv);

#endif
