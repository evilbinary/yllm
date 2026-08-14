/* proclist.h — 跨平台枚举本机 yllm 进程(status / ctl exit 共用)
 *
 * Linux: 遍历 /proc 下各进程的 cmdline
 * Windows: Toolhelp32Snapshot 枚举进程 + NtQueryInformationProcess 读 PEB
 *          命令行(与 status.c 原实现一致), exe 名含 "yllm" 才访问
 */

#ifndef YLLM_SERVE_PROCLIST_H
#define YLLM_SERVE_PROCLIST_H

#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#else
#include <dirent.h>
#include <stdlib.h>
#endif

/* 遍历本机 cmdline 含 "yllm" 的进程, 对每个调用 visitor(pid, cmdline, ctx)。
 * visitor 返回非 0 停止遍历。返回访问到的进程数。 */
static inline int proclist_visit(int (*visitor)(int pid, const char* cmdline, void* ctx), void* ctx)
{
    int count = 0;
#ifdef _WIN32
    /* Windows: Toolhelp 拿 pid, NtQueryInformationProcess 读 PEB 命令行,
     * 格式与 Linux /proc 对齐(pid + 完整启动命令参数)。 */
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return 0;
    PROCESSENTRY32W pe;
    pe.dwSize = sizeof(pe);
    if (Process32FirstW(snap, &pe)) {
        do {
            if (wcsstr(pe.szExeFile, L"yllm") == NULL) continue;
            DWORD pid = pe.th32ProcessID;
            HANDLE h = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
            char cmdline[4096] = "";
            if (h) {
                /* PEB → ProcessParameters → CommandLine(UNICODE_STRING) */
                typedef NTSTATUS (WINAPI *NtQIP)(HANDLE, int, PVOID, ULONG, PULONG);
                HMODULE ntdll = GetModuleHandleA("ntdll.dll");
                if (ntdll) {
                    NtQIP ntqip = (NtQIP)GetProcAddress(ntdll, "NtQueryInformationProcess");
                    if (ntqip) {
                        struct {
                            PVOID Reserved1;
                            PVOID PebBaseAddress;
                            PVOID Reserved2[2];
                            PVOID UniqueProcessId;
                            PVOID Reserved3;
                        } pbi;
                        if (ntqip(h, 0, &pbi, sizeof(pbi), NULL) == 0 && pbi.PebBaseAddress) {
                            /* PEB → ProcessParameters(偏移 0x20, 64 位) */
                            PVOID params = NULL;
                            SIZE_T rd = 0;
                            if (ReadProcessMemory(h, (BYTE*)pbi.PebBaseAddress + 0x20,
                                                  &params, sizeof(params), &rd) && rd == sizeof(params) && params) {
                                /* ProcessParameters → CommandLine(UNICODE_STRING: USHORT len + 指针, 偏移 0x70) */
                                struct { USHORT len; USHORT max; PVOID buf; } ucs;
                                rd = 0;
                                if (ReadProcessMemory(h, (BYTE*)params + 0x70,
                                                      &ucs, sizeof(ucs), &rd) && rd == sizeof(ucs) && ucs.buf) {
                                    if (ucs.len > 0 && ucs.len < sizeof(cmdline) - 2) {
                                        rd = 0;
                                        if (ReadProcessMemory(h, ucs.buf, cmdline, ucs.len, &rd) && rd == ucs.len) {
                                            cmdline[rd] = 0;
                                            cmdline[rd+1] = 0;
                                            /* UTF-16 → UTF-8 简化: 逐字符转 ASCII(命令行为 ASCII 可读) */
                                            char utf8[4096];
                                            size_t u = 0;
                                            size_t i;
                                            for (i = 0; i < rd && u < sizeof(utf8)-1; i += 2) {
                                                unsigned short ch = (unsigned short)((unsigned char)cmdline[i] |
                                                                                   ((unsigned char)cmdline[i+1] << 8));
                                                if (ch < 0x80) utf8[u++] = (char)ch;
                                                else if (ch < 0x800 && u+1 < sizeof(utf8)-1) {
                                                    utf8[u++] = (char)(0xC0 | (ch >> 6));
                                                    utf8[u++] = (char)(0x80 | (ch & 0x3F));
                                                } else if (u+2 < sizeof(utf8)-1) {
                                                    utf8[u++] = (char)(0xE0 | (ch >> 12));
                                                    utf8[u++] = (char)(0x80 | ((ch >> 6) & 0x3F));
                                                    utf8[u++] = (char)(0x80 | (ch & 0x3F));
                                                }
                                            }
                                            utf8[u] = 0;
                                            snprintf(cmdline, sizeof(cmdline), "%s", utf8);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                CloseHandle(h);
            }
            if (visitor && visitor((int)pid, cmdline[0] ? cmdline : "?", ctx)) break;
            count++;
        } while (Process32NextW(snap, &pe));
    }
    CloseHandle(snap);
#else
    DIR* d = opendir("/proc");
    if (!d) return 0;
    struct dirent* e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        char path[64], cmd[4096];
        snprintf(path, sizeof(path), "/proc/%s/cmdline", e->d_name);
        FILE* f = fopen(path, "rb");
        if (!f) continue;
        size_t n = fread(cmd, 1, sizeof(cmd) - 1, f);
        fclose(f);
        if (n == 0) continue;
        cmd[n] = 0;
        /* cmdline 以 \0 分隔, 转空格便于显示/匹配 */
        size_t i;
        for (i = 0; i < n; i++) if (cmd[i] == 0) cmd[i] = ' ';
        if (strstr(cmd, "yllm") == NULL) continue;
        if (visitor && visitor(atoi(e->d_name), cmd, ctx)) break;
        count++;
    }
    closedir(d);
#endif
    return count;
}

#endif /* YLLM_SERVE_PROCLIST_H */
