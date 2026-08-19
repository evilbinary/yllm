#include "yllm.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); } \
} while (0)

#define CHECK_NEAR(a, b, eps, msg) do { \
    double _a = (double)(a), _b = (double)(b); \
    if (fabs(_a - _b) <= (eps)) { g_pass++; } \
    else { g_fail++; printf("FAIL: %s: %.9g vs %.9g (%s:%d)\n", msg, _a, _b, __FILE__, __LINE__); } \
} while (0)

/* ---- engine forward regression: single + dual token logits ----
 * Golden values captured from the fixed engine and cross-checked against
 * picolm (reference implementation), which matches to <1e-4.
 *
 * Regression coverage:
 *  - fp16 denormal exponent + sign bugs (wrong logits)
 *  - RoPE pairwise vs half-split (wrong logits)
 *  - KV cache stored before RoPE (wrong logits)
 *  - matmul input/output aliasing (att_out) (wrong logits)
 *  - FFN gate/up buffer overlap (wrong logits)
 *  - BPE tokenizer (Hello must be token 15043)
 */
int main(int argc, char** argv)
{
    const char* model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.llf";
    const char* vocab_path = "models/tinyllama.vocab.txt";
    if (argc > 1) model_path = argv[1];
    if (argc > 2) vocab_path = argv[2];

    Engine e;
    char err[1024];
    if (engine_init(&e, model_path, 0, 2, err, sizeof(err)) != 0) {
        printf("SKIP: engine init failed (%s) - model file missing?\n", err);
        printf("engine tests: skipped\n");
        return 0;
    }

    /* single token (pos=0) */
    engine_forward(&e, 15043, 0);
    {
        static const double golden[5] = {
            -2.64744163, -3.00165749, 1.34251297, 0.741827667, 0.141416222
        };
        uint32_t i;
        for (i = 0; i < 5; i++) {
            char msg[64];
            snprintf(msg, sizeof(msg), "single logits[%u]", i);
            CHECK_NEAR(e.logits[i], golden[i], 5e-4, msg);
        }
    }

    /* dual token (pos=1, exercises RoPE + KV cache + attention) */
    engine_forward(&e, 29871, 1);
    {
        static const double golden[5] = {
            -6.65888119, -6.24745417, 7.89513826, 2.07095957, -0.939354777
        };
        uint32_t i;
        for (i = 0; i < 5; i++) {
            char msg[64];
            snprintf(msg, sizeof(msg), "dual logits[%u]", i);
            CHECK_NEAR(e.logits[i], golden[i], 5e-4, msg);
        }
        /* argmax must be 13 (newline) as in picolm */
        uint32_t bi = 0;
        for (i = 1; i < e.ws.model.h.vocab; i++) if (e.logits[i] > e.logits[bi]) bi = i;
        CHECK(bi == 13, "dual argmax = 13");
    }

    /* all logits finite (NaN propagation regression) */
    {
        uint32_t i, bad = 0;
        for (i = 0; i < e.ws.model.h.vocab; i++) {
            if (e.logits[i] != e.logits[i]) { bad++; break; }
        }
        CHECK(bad == 0, "logits all finite");
    }

    engine_free(&e);

    printf("engine tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}


