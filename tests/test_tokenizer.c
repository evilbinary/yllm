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

/* ---- BPE encode regression: "Hello" must be ONE token (15043 = "▁Hello") ---- *
 * Regression: greedy matching (no merges) split "Hello" into byte tokens
 * <0x48><0x65><0x6C><0x6C><0x6F> whose embeddings are ~0 -> garbage output. */
static void test_bpe_hello(const char* vocab_path)
{
    Vocab v;
    if (vocab_load(vocab_path, &v) != 0) {
        printf("SKIP: cannot load vocab %s\n", vocab_path);
        return;
    }
    CHECK(v.n == 32000, "vocab size 32000");
    CHECK(v.n_scores == 32000, "scores loaded 32000");
    CHECK(v.unk >= 0 && v.bos >= 0 && v.eos >= 0, "special tokens found");

    {
        uint32_t ids[64];
        int n = vocab_encode(&v, "Hello", ids, 64);
        CHECK(n == 1, "Hello -> 1 token (BPE)");
        if (n == 1) CHECK(ids[0] == 15043, "Hello -> token 15043 (▁Hello)");
    }
    {
        uint32_t ids[64];
        int n = vocab_encode(&v, "Hello world", ids, 64);
        CHECK(n == 2, "Hello world -> 2 tokens");
    }
    /* byte fallback for unknown bytes still works (with leading ▁) */
    {
        uint32_t ids[64];
        int n = vocab_encode(&v, "\xff\xfe", ids, 64);
        CHECK(n == 3, "raw bytes -> ▁ + 2 byte tokens");
        if (n == 3) {
            CHECK(strcmp(v.pieces[ids[1]], "<0xFF>") == 0, "0xFF byte token");
            CHECK(strcmp(v.pieces[ids[2]], "<0xFE>") == 0, "0xFE byte token");
        }
    }
    vocab_free(&v);
}

/* ---- vocab text escaping: tokens containing \n \r \t \ must round-trip ---- */
static void test_escaping(void)
{
    const char* path = "build/test_vocab.txt";
    FILE* f = fopen(path, "wb");
    if (!f) { printf("SKIP: cannot write %s\n", path); return; }
    fprintf(f, "5\n");
    fprintf(f, "0\t<unk>\n");
    fprintf(f, "1\t<0x0A>\n");       /* literal text, no escaping needed */
    fprintf(f, "2\t\\n\\r\\t\\\\\n"); /* escaped \n \r \t \ */
    fprintf(f, "3\tplain\n");
    fprintf(f, "4\tend\n");
    fclose(f);

    Vocab v;
    if (vocab_load(path, &v) != 0) {
        printf("FAIL: cannot load test vocab\n");
        g_fail++;
        return;
    }
    CHECK(v.n == 5, "5 pieces parsed");
    CHECK(strcmp(v.pieces[2], "\n\r\t\\") == 0, "escaped piece round-trips");
    CHECK(strcmp(v.pieces[1], "<0x0A>") == 0, "plain piece intact");
    CHECK(v.unk == 0, "unk = 0");
    vocab_free(&v);
    remove(path);
}

/* ---- encoding must be deterministic and not crash on empty/space ---- */
static void test_encode_edge(const char* vocab_path)
{
    Vocab v;
    if (vocab_load(vocab_path, &v) != 0) {
        printf("SKIP: cannot load vocab %s\n", vocab_path);
        return;
    }
    {
        uint32_t ids[16];
        int n = vocab_encode(&v, "", ids, 16);
        CHECK(n == 1, "empty text -> 1 token (▁)");
    }
    {
        uint32_t ids[16];
        int n = vocab_encode(&v, "   ", ids, 16);
        CHECK(n == 1, "spaces -> ▁▁▁");
    }
    {
        uint32_t ids[8];
        int n = vocab_encode(&v, "abcdefghijklmnopqrstuvwxyz", ids, 8);
        CHECK(n <= 8, "long text bounded by max");
    }
    vocab_free(&v);
}

/* ---- decode round trip ---- */
static void test_decode(const char* vocab_path)
{
    Vocab v;
    if (vocab_load(vocab_path, &v) != 0) {
        printf("SKIP: cannot load vocab %s\n", vocab_path);
        return;
    }
    {
        uint32_t ids[2] = { 15043, 29871 };  /* "▁Hello" + "▁" */
        char out[64];
        vocab_decode(&v, ids, 2, out, sizeof(out));
        CHECK(strcmp(out, " Hello ") == 0 || strcmp(out, " Hello") == 0, "decode ▁Hello + ▁");
    }
    vocab_free(&v);
}

/* ---- chat template render: tinyllama <|user|>...<|assistant|> ---- */
static void test_chat_template(const char* vocab_path)
{
    Vocab v;
    if (vocab_load(vocab_path, &v) != 0) {
        printf("SKIP: cannot load vocab %s\n", vocab_path);
        return;
    }
    CHECK(vocab_has_template(&v), "chat template present");
    if (!vocab_has_template(&v)) {
        vocab_free(&v);
        return;
    }
    uint32_t ids[256];
    int n = vocab_chat_ids(&v, "Hello, how are you?", ids, 256, 0);
    CHECK(n > 0, "chat render produces tokens");
    if (n > 0) {
        char out[1024];
        vocab_decode(&v, ids, n, out, sizeof(out));
        /* tinyllama template: <|user|>\nHello, how are you?</s>\n<|assistant|> */
        CHECK(strstr(out, "<|user|>") != NULL, "chat has user tag");
        CHECK(strstr(out, "Hello, how are you?") != NULL, "chat has content");
        CHECK(strstr(out, "<|assistant|>") != NULL, "chat has generation prompt");
        /* must NOT contain other role tags */
        CHECK(strstr(out, "<|system|>") == NULL, "chat no system tag");
    }
    vocab_free(&v);
}

/* ---- chat template render: qwen 风格嵌套 if/else ----
 * 回归覆盖:
 *   1) {% if tools %} 为假时的 {% else %} 分支必须渲染(此前被丢弃)
 *   2) elif 分支内的嵌套 {% if message['content'] %} 不得复活已跳过的分支(此前内容重复)
 *   3) 用户内容只出现一次, 不得进入 assistant/tool 分支 */
static void test_chat_template_qwen_style(void)
{
    const char* tpl =
        "{%- if tools -%}"
        "{{- 'TOOLS' -}}"
        "{%- else -%}"
        "{{- 'SYS:' -}}"
        "{%- endif -%}"
        "{%- for message in messages -%}"
        "{%- if message['role'] == 'user' or (message['role'] == 'system' and not loop.first) -%}"
        "{{- '<u>' + message['content'] + '</u>\\n' -}}"
        "{%- elif message['role'] == 'assistant' -%}"
        "{%- if message['content'] -%}"
        "{{- 'A:' + message['content'] -}}"
        "{%- endif -%}"
        "{{- '<e>\\n' -}}"
        "{%- else -%}"
        "{%- if message['role'] == 'tool' -%}"
        "{{- 'T:' + message['content'] -}}"
        "{%- endif -%}"
        "{{- '<t>\\n' -}}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "{{- '<|assistant|>' -}}"
        "{%- endif -%}";
    /* 构造最小 vocab 文件(#CHAT# 段带模板); pieces 覆盖全部可打印 ASCII 以便无损解码 */
    const char* path = "/tmp/yllm_tpl_qwen.vocab";
    FILE* f = fopen(path, "w");
    if (!f) { printf("SKIP: cannot write %s\n", path); return; }
    fprintf(f, "95\n0\t<unk>\n1\t<|im_end|>\n");
    {
        int c;
        for (c = 0x21; c <= 0x7e; c++) {
            if (c == '\\' || c == '\'') continue;   /* 词表行转义字符, 跳过 */
            fprintf(f, "%d\t%c\n", 2 + (c - 0x21), c);
        }
    }
    fprintf(f, "94\t\\n\n");   /* 换行 piece(转义写法) */
    fprintf(f, "#CHAT#\neos_id=1\ntemplate=%s\n", tpl);
    fclose(f);

    Vocab v;
    if (vocab_load(path, &v) != 0) {
        printf("SKIP: cannot load %s\n", path);
        remove(path);
        return;
    }
    remove(path);
    CHECK(vocab_has_template(&v), "qwen-style template present");
    if (!vocab_has_template(&v)) {
        vocab_free(&v);
        return;
    }
    uint32_t ids[256];
    int n = vocab_chat_ids(&v, "Hello", ids, 256, 0);
    CHECK(n > 0, "qwen-style render produces tokens");
    if (n > 0) {
        char out[1024];
        vocab_decode(&v, ids, n, out, sizeof(out));
        /* 1) else 分支渲染且仅一次 */
        const char* p = out;
        int n_sys = 0;
        while ((p = strstr(p, "SYS:")) != NULL) { n_sys++; p += 4; }
        CHECK(n_sys == 1, "qwen-style: else-branch rendered exactly once");
        /* 2) 用户内容仅一次, 未进入 assistant 分支 */
        int n_u = 0;
        p = out;
        while ((p = strstr(p, "<u>Hello</u>")) != NULL) { n_u++; p += 11; }
        CHECK(n_u == 1, "qwen-style: user content rendered exactly once");
        CHECK(strstr(out, "A:Hello") == NULL, "qwen-style: assistant branch skipped");
        CHECK(strstr(out, "<e>") == NULL, "qwen-style: assistant end-tag skipped");
        /* 3) tools 分支不渲染, 生成提示在 */
        CHECK(strstr(out, "TOOLS") == NULL, "qwen-style: tools branch skipped");
        CHECK(strstr(out, "<|assistant|>") != NULL, "qwen-style: generation prompt");
    }
    vocab_free(&v);
}

/* 真实 qwen2.5 vocab(存在时): 完整模板渲染检查 */
static void test_chat_template_qwen_real(void)
{
    Vocab v;
    if (vocab_load("models/qwen2.5.vocab.txt", &v) != 0) {
        printf("SKIP: no models/qwen2.5.vocab.txt\n");
        return;
    }
    CHECK(vocab_has_template(&v), "qwen real template present");
    if (!vocab_has_template(&v)) {
        vocab_free(&v);
        return;
    }
    uint32_t ids[256];
    int n = vocab_chat_ids(&v, "Hello", ids, 256, v.add_bos);
    CHECK(n > 0, "qwen real render produces tokens");
    if (n > 0) {
        char out[1024];
        vocab_decode(&v, ids, n, out, sizeof(out));
        CHECK(strstr(out, "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>") != NULL,
              "qwen: system prompt rendered");
        CHECK(strstr(out, "<|im_start|>user\nHello<|im_end|>") != NULL,
              "qwen: user turn rendered");
        CHECK(strstr(out, "<|im_start|>assistant\n") != NULL,
              "qwen: generation prompt rendered");
        CHECK(strstr(out, "TOOLS") == NULL && strstr(out, "<tool_call>") == NULL,
              "qwen: tools branch skipped");
        /* 内容不得重复 */
        int n_hello = 0;
        const char* p = out;
        while ((p = strstr(p, "Hello")) != NULL) { n_hello++; p += 5; }
        CHECK(n_hello == 1, "qwen: content not duplicated");
    }
    vocab_free(&v);
}

int main(int argc, char** argv)
{
    const char* vocab_path = "models/tinyllama.vocab.txt";
    if (argc > 1) vocab_path = argv[1];
    test_bpe_hello(vocab_path);
    test_encode_edge(vocab_path);
    test_decode(vocab_path);
    test_escaping();
    test_chat_template(vocab_path);
    test_chat_template_qwen_style();
    test_chat_template_qwen_real();
    printf("tokenizer tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
