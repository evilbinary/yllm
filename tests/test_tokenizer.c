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

int main(int argc, char** argv)
{
    const char* vocab_path = "test/tinyllama.vocab.txt";
    if (argc > 1) vocab_path = argv[1];
    test_bpe_hello(vocab_path);
    test_encode_edge(vocab_path);
    test_decode(vocab_path);
    test_escaping();
    test_chat_template(vocab_path);
    printf("tokenizer tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
