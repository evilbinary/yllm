/* vulkan_fwd.c — Vulkan 前向挂接
 *
 * native 模式暂仍 CPU fwd(无 shader 前); host-shim 同。
 * 后续替换为 compute dispatch。
 */
#include "device.h"
#include "yllm.h"
#include "vulkan_ctx.h"

void vulkan_attach_fwd(Engine* e)
{
    /* P0/P1 过渡: 设备已创建也先用 CPU 块, 保证数值正确 */
    engine_attach_cpu_fwd(e);
    (void)e;
}
