#include "yllm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

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

/* 临时目录: Windows 用 TEMP(Win 下 "/tmp" 是盘符根的 \tmp, 父目录常不存在);
 * POSIX 用 /tmp。父目录不存在时先创建。 */
static void test_tmpdir(char* buf, size_t sz)
{
    const char* t = getenv("TEMP");
    if (t && *t) snprintf(buf, sz, "%s/opencode", t);
    else snprintf(buf, sz, "/tmp/opencode");
#ifndef _WIN32
    mkdir(buf, 0755);
#else
    _mkdir(buf);
#endif
}

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
    char tdir[512], path[600];
    test_tmpdir(tdir, sizeof(tdir));
    snprintf(path, sizeof(path), "%s/yllm_tpl_qwen.vocab", tdir);
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

/* ---- 构造最小 vocab(可带 <|im_start|> token), #CHAT# 段挂指定模板 ---- */
static int write_min_vocab(const char* path, const char* tpl, int with_im)
{
    char body[8192];
    int off = 0;
    off += snprintf(body + off, sizeof(body) - off, "0\t<unk>\n1\t<|im_end|>\n");
    if (with_im) off += snprintf(body + off, sizeof(body) - off, "2\t<|im_start|>\n");
    int id = with_im ? 3 : 2;
    int c;
    for (c = 0x20; c <= 0x7e; c++) {
        if (c == '\\' || c == '\'') continue;
        if (with_im && (c == '<' || c == '>' || c == '|')) continue;
        off += snprintf(body + off, sizeof(body) - off, "%d\t%c\n", id++, c);
    }
    off += snprintf(body + off, sizeof(body) - off, "%d\t\\n\n", id++);
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "%d\n%s#CHAT#\neos_id=1\ntemplate=%s\n", id, body, tpl);
    fclose(f);
    return 0;
}

/* ---- 不支持模板(Unsloth Qwen3 风格: macro/set/namespace/raise_exception/[::-1])
 * 必须走内置通用 im_start 回退, 不得把 macro 体当顶层语句输出乱码 ---- */
static void test_chat_template_unsupported_fallback(void)
{
    const char* tpl =
        "{%- macro render_content(content, do_vision_count, is_system_content=false) %}"
        "{%- if content is string %}{{- content }}"
        "{%- else %}{{- raise_exception('Unexpected content type.') }}{%- endif %}"
        "{%- endmacro %}"
        "{%- set image_count = namespace(value=0) %}"
        "{%- for message in messages[::-1] %}"
        "{%- if ns.multi_step_tool and message.role == 'user' %}{%- set ns.multi_step_tool = false %}{%- endif %}"
        "{%- endfor %}"
        "{%- for message in messages %}"
        "{%- if loop.index0 >= 0 %}"
        "{%- set content = render_content(message.content, true)|trim %}"
        "{%- if message.role == 'user' %}"
        "{{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}"
        "{%- elif message.role == 'assistant' %}"
        "{{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}"
        "{%- else %}{{- raise_exception('Unexpected message role.') }}{%- endif %}"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";
    char tdir[512], path[600];
    test_tmpdir(tdir, sizeof(tdir));
    snprintf(path, sizeof(path), "%s/yllm_tpl_unsloth.vocab", tdir);
    if (write_min_vocab(path, tpl, 1) != 0) {
        printf("SKIP: cannot write %s\n", path);
        return;
    }
    Vocab v;
    if (vocab_load(path, &v) != 0) {
        printf("SKIP: cannot load %s\n", path);
        remove(path);
        return;
    }
    remove(path);
    CHECK(vocab_has_template(&v), "unsloth-style template present");
    uint32_t ids[256];
    int n = vocab_chat_ids(&v, "Hello", ids, 256, 0);
    CHECK(n > 0, "unsloth-style render produces tokens");
    if (n > 0) {
        char out[1024];
        vocab_decode(&v, ids, n, out, sizeof(out));
        /* 1) 不得输出 macro 体内的 raise_exception 文本 */
        CHECK(strstr(out, "Unexpected") == NULL, "unsloth: no macro-body garbage");
        /* 2) 用户内容经通用 im_start 模板渲染 */
        CHECK(strstr(out, "<|im_start|>user\nHello<|im_end|>") != NULL,
              "unsloth: user turn rendered");
        CHECK(strstr(out, "<|im_start|>assistant\n") != NULL,
              "unsloth: generation prompt rendered");
        int n_hello = 0;
        const char* p = out;
        while ((p = strstr(p, "Hello")) != NULL) { n_hello++; p += 5; }
        CHECK(n_hello == 1, "unsloth: content not duplicated");
    }
    /* 多轮: system+user+assistant 顺序渲染 */
    {
        const char* roles[3] = { "system", "user", "assistant" };
        const char* contents[3] = { "You are Qwen.", "Hello", "Hi" };
        uint32_t ids2[512];
        int n2 = vocab_chat_ids_multi(&v, roles, contents, 3, ids2, 512, 0);
        CHECK(n2 > 0, "unsloth multi-turn produces tokens");
        if (n2 > 0) {
            char out[2048];
            vocab_decode(&v, ids2, n2, out, sizeof(out));
            CHECK(strstr(out, "<|im_start|>system\nYou are Qwen.<|im_end|>") != NULL,
                  "unsloth multi: system turn");
            CHECK(strstr(out, "<|im_start|>user\nHello<|im_end|>") != NULL,
                  "unsloth multi: user turn");
            CHECK(strstr(out, "<|im_start|>assistant\nHi<|im_end|>") != NULL,
                  "unsloth multi: assistant turn");
            const char* a = strstr(out, "system");
            const char* b = strstr(out, "user");
            const char* c2 = strstr(out, "assistant");
            CHECK(a && b && c2 && a < b && b < c2, "unsloth multi: turns in order");
            CHECK(strstr(out, "Unexpected") == NULL, "unsloth multi: no garbage");
        }
    }
    vocab_free(&v);
}

/* ---- 同模板但 vocab 无 <|im_start|>/<|im_end|>: 退化为 "role: content" ---- */
static void test_chat_template_unsupported_plain(void)
{
    const char* tpl =
        "{%- macro m(c) %}{%- if c is string %}{{- c }}{%- else %}{{- raise_exception('X') }}{%- endif %}{%- endmacro %}"
        "{%- for message in messages %}{{- m(message.content) }}{%- endfor %}";
    char tdir[512], path[600];
    test_tmpdir(tdir, sizeof(tdir));
    snprintf(path, sizeof(path), "%s/yllm_tpl_nonim.vocab", tdir);
    if (write_min_vocab(path, tpl, 0) != 0) {
        printf("SKIP: cannot write %s\n", path);
        return;
    }
    Vocab v;
    if (vocab_load(path, &v) != 0) {
        printf("SKIP: cannot load %s\n", path);
        remove(path);
        return;
    }
    remove(path);
    uint32_t ids[256];
    int n = vocab_chat_ids(&v, "Hello", ids, 256, 0);
    CHECK(n > 0, "plain fallback produces tokens");
    if (n > 0) {
        char out[1024];
        vocab_decode(&v, ids, n, out, sizeof(out));
        CHECK(strstr(out, "user: Hello") != NULL, "plain fallback: role: content");
        CHECK(strstr(out, "assistant: ") != NULL, "plain fallback: generation prompt");
        CHECK(strstr(out, "X") == NULL, "plain fallback: no macro garbage");
    }
    vocab_free(&v);
}

/* ---- 真实 qwen3 vocab(存在时): 复杂 Unsloth 模板走通用 im_start 回退 ---- */
static void test_chat_template_qwen3_real(void)
{
    Vocab v;
    if (vocab_load("models/qwen3.vocab.txt", &v) != 0) {
        printf("SKIP: no models/qwen3.vocab.txt\n");
        return;
    }
    uint32_t ids[256];
    int n = vocab_chat_ids(&v, "Hello", ids, 256, v.add_bos);
    CHECK(n > 0, "qwen3 render produces tokens");
    if (n > 0) {
        char out[1024];
        vocab_decode(&v, ids, n, out, sizeof(out));
        CHECK(strstr(out, "<|im_start|>user\nHello<|im_end|>") != NULL,
              "qwen3: user turn rendered");
        CHECK(strstr(out, "<|im_start|>assistant\n") != NULL,
              "qwen3: generation prompt rendered");
        CHECK(strstr(out, "Unexpected") == NULL && strstr(out, "content._type") == NULL,
              "qwen3: no template garbage");
        /* 模板行以 im_start 起始(不含 BOS) */
        if (n > 0) CHECK(ids[0] == 248045, "qwen3: first token <|im_start|>");
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
    test_chat_template_unsupported_fallback();
    test_chat_template_unsupported_plain();
    test_chat_template_qwen3_real();
    printf("tokenizer tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
