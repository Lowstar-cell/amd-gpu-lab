#!/usr/bin/env python3
"""
Patch a RenderPipelineShaders Vulkan test/example source to inject lightweight GPA instrumentation.
This script makes a backup of the source file and writes instrumentation into a new file.
"""
import argparse
import os
import re
import shutil


def insert_after_last_include(src_text, insert_text):
    # Find the last #include and insert after it
    includes = list(re.finditer(r"^#include .*$", src_text, re.MULTILINE))
    if not includes:
        return insert_text + "\n" + src_text
    last = includes[-1]
    insert_pos = last.end()
    return src_text[:insert_pos] + "\n" + insert_text + src_text[insert_pos:]


def backup(path):
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    return bak


def patch_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Insert GPA includes
    insert = '#include "gpu_performance_api/gpu_perf_api.h"\n#include "gpu_performance_api/gpu_perf_api_interface_loader.h"\n#include <fstream>\n#include <vector>\n'

    text = insert_after_last_include(text, insert)

    # Add global GPA context/session variables near top: after includes
    globals_decl = '\n// GPA instrumentation: context & session (injected)\nGpaContextId gpaCtx = NULL;\nGpaSessionId gpaSession = NULL;\nstd::vector<int> gpa_enabled_counters;\nstd::ofstream gpa_out;\n'
    text = insert_after_last_include(text, globals_decl)

    # Insert initialization in OnInit method
    oninit_pattern = r"virtual void OnInit\([^\)]*\) override\s*\{"
    oninit_match = re.search(oninit_pattern, text)
    if oninit_match:
        insert_init = (
            '\n        // GPA instrumentation init (injected)\n'
            '        {\n'
            '            GpaStatus status = GpaInitialize(kGpaInitializeDefaultBit);\n'
            '            if (status != kGpaStatusOk) { std::cerr << "GpaInitialize failed: " << status << std::endl; }\n'
            '            status = GpaInterfaceLoaderLoadApi(kGpaApiVulkan, "data/4_2/bin");\n'
            '            if (status != kGpaStatusOk) { std::cerr << "GpaInterfaceLoaderLoadApi failed: " << status << std::endl; }\n'
            '            status = GpaOpenContext((void*)m_device, kGpaOpenContextEnableHardwareCountersBit, &gpaCtx);\n'
            '            if (status != kGpaStatusOk) { std::cerr << "GpaOpenContext failed: " << status << std::endl; }\n'
            '            status = GpaCreateSession(gpaCtx, kGpaSessionSampleTypeDiscreteCounter, &gpaSession);\n'
            '            if (status != kGpaStatusOk) { std::cerr << "GpaCreateSession failed: " << status << std::endl; }\n'
            '            status = GpaBeginSession(gpaSession);\n'
            '            // Query counters and enable a small subset\n'
            '            GpaUInt32 numCounters = 0;\n'
            '            if (GpaGetNumCounters(gpaSession, &numCounters) == kGpaStatusOk) {\n'
            '                for (GpaUInt32 i = 0; i < numCounters && i < 4; ++i) {\n'
            '                    if (GpaEnableCounter(gpaSession, i) == kGpaStatusOk) gpa_enabled_counters.push_back((int)i);\n'
            '                }\n'
            '            }\n'
            '            GpaUInt32 passCount = 0;\n'
            '            if (GpaGetPassCount(gpaSession, &passCount) != kGpaStatusOk) { passCount = 0; }\n'
            '            // open output CSV\n'
            '            gpa_out.open("gpa_rps_output.csv");\n'
            '            if (gpa_out.is_open()) {\n'
            '                gpa_out << "iter,";\n'
            '                for (size_t ii = 0; ii < gpa_enabled_counters.size(); ++ii) {\n'
            '                    gpa_out << "counter_" << gpa_enabled_counters[ii] << (ii + 1 < gpa_enabled_counters.size() ? "," : "\n");\n'
            '                }\n'
            '            }\n'
            '            if (status != kGpaStatusOk) { std::cerr << "GpaBeginSession failed: " << status << std::endl; }\n'
            '        }\n'
        )
        text = text[: oninit_match.end()] + insert_init + text[oninit_match.end():]

    # Insert begin/end sample around vkCmdDraw or vkCmdDrawIndexed or Draw call in DrawTriangle
    draw_pattern = r"void\s+DrawTriangle\([^\)]*\)\s*\{"
    draw_match = re.search(draw_pattern, text)
    if draw_match:
        brace_pos = draw_match.end()
        # Find the body and insert before vkCmdDraw and after
        # We will insert a simple GpaBeginSample/GpaEndSample pair wrapped around vkCmdDraw by finding vkCmdDraw calls
        draw_call_re = re.compile(r"(vkCmdDrawIndexed\(|vkCmdDraw\()")
        # Instead find line with vkCmdDraw within the DrawTriangle function block
        # Find the function body extent by counting braces from brace_pos
        depth = 1
        i = brace_pos
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        func_body = text[brace_pos:i-1]
        # Insert instrumentation around vkCmdDraw occurrences
        new_body = func_body
        # Surround vkCmdDraw(...) with GPA sample begin/end
        def repl_draw(m):
            body = m.group(1)
            return '{ GpaSampleId sample = 0; GpaBeginSample(gpaSession, nullptr, &sample); ' + body + ' GpaEndSample(gpaSession); size_t __gpa_size=0; if (GpaGetSampleResultSize(gpaSession, sample, &__gpa_size) == kGpaStatusOk) { std::vector<char> __gpa_buf(__gpa_size); if (GpaGetSampleResult(gpaSession, sample, __gpa_size, __gpa_buf.data()) == kGpaStatusOk) { uint64_t* __gpa_vals = reinterpret_cast<uint64_t*>(__gpa_buf.data()); size_t __gpa_count = __gpa_size / sizeof(uint64_t); if (gpa_out.is_open()) { gpa_out << 0; for (size_t j = 0; j < gpa_enabled_counters.size(); ++j) { uint64_t v = 0; if (j < __gpa_count) v = __gpa_vals[j]; gpa_out << "," << v; } gpa_out << "\n"; } } } }'
        new_body = re.sub(r"(vkCmdDraw\([^;]+;)", repl_draw, new_body)
        new_body = re.sub(r"(vkCmdDrawIndexed\([^;]+;)", repl_draw, new_body)
        # Replace the function body
        text = text[:brace_pos] + new_body + text[i-1:]

    # Add cleanup in OnCleanUp: GpaEndSession, GpaDeleteSession, GpaCloseContext, GpaDestroy
    oncleanup_pattern = r"virtual void OnCleanUp\([^\)]*\) override\s*\{"
    oncleanup_match = re.search(oncleanup_pattern, text)
    if oncleanup_match:
        insert_cleanup = (
            '\n        // GPA instrumentation cleanup (injected)\n'
            '        if (gpaSession) GpaEndSession(gpaSession);\n'
            '        if (gpaSession) GpaDeleteSession(gpaSession);\n'
            '        if (gpaCtx) GpaCloseContext(gpaCtx);\n'
            '        GpaDestroy();\n'
            '        if (gpa_out.is_open()) gpa_out.close();\n'
    )
    text = text[: oncleanup_match.end()] + insert_cleanup + text[oncleanup_match.end():]

    # Write changes with backup
    bak = backup(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return bak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', required=True, help='Path to the _vk.cpp file to patch')
    args = parser.parse_args()

    src = os.path.abspath(args.example)
    if not os.path.exists(src):
        print("Error: file not found:", src)
        return 1

    bak = patch_file(src)
    print("Patched:", src, "(backup at", bak, ")")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

