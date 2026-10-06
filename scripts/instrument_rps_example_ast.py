#!/usr/bin/env python3
"""
AST-based instrumentor for RenderPipelineShaders samples/tests using libclang.
This script uses clang.cindex to find methods in a C++ file and inject GPA initialization,
GPA sample begin/end around vkCmdDraw/vkCmdDrawIndexed calls, and cleanup.

It edits files in-place and creates a backup .bak file.
"""
import argparse
import os
import shutil
import sys
import io

try:
    from clang import cindex
except Exception:
    cindex = None

def backup(path):
    bak = path + '.bak'
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    return bak


def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def pos_to_offset(text, line, col):
    # line/col are 1-based
    lines = text.splitlines(True)
    if line - 1 >= len(lines):
        return len(text)
    offset = sum(len(lines[i]) for i in range(line - 1))
    return offset + (col - 1)


def find_last_include_index(text):
    lines = text.split('\n')
    last = -1
    for i, l in enumerate(lines):
        if l.strip().startswith('#include'):
            last = i
    return last


def run_ast_instrument(source_path, include_dirs=None, libgpa_path='data/4_2/bin'):
    if cindex is None:
        raise RuntimeError('clang.cindex (Python) bindings not found. Please pip install clang and ensure libclang is installed.')

    # Try to find libclang
    try:
        cindex.Config.set_library_file(cindex.Config.library_file)
    except Exception:
        pass

    code = read_file(source_path)

    # Setup clang arguments
    args = ['-std=c++17']
    if include_dirs:
        for d in include_dirs:
            args.extend(['-I', d])
    index = cindex.Index.create()
    tu = index.parse(source_path, args=args)
    if not tu:
        raise RuntimeError('Failed to parse ' + source_path)

    # We'll collect insertions as (offset, text) pairs and apply later (from end to start)
    inserts = []

    # Insert GPA includes after last include
    include_text = '#include "gpu_performance_api/gpu_perf_api.h"\n#include "gpu_performance_api/gpu_perf_api_interface_loader.h"\n#include <fstream>\n#include <vector>\n'
    last_inc = find_last_include_index(code)
    if last_inc == -1:
        # put at top
        insert_offset = 0
    else:
        lines = code.split('\n')
        insert_offset = sum(len(lines[i]) + 1 for i in range(last_inc + 1))
    inserts.append((insert_offset, '\n' + include_text))

    # Add global GPA context/session variables after includes
    global_decl = '\n// GPA instrumentation: context & session (injected AST)\nGpaContextId gpaCtx = NULL;\nGpaSessionId gpaSession = NULL;\nstd::vector<int> gpa_enabled_counters;\nstd::ofstream gpa_out;\n\n'
    inserts.append((insert_offset + len(include_text), global_decl))

    # Traverse AST and find methods: OnInit, OnCleanUp, DrawTriangle
    for cursor in tu.cursor.walk_preorder():
        if cursor.kind in (cindex.CursorKind.CXX_METHOD, cindex.CursorKind.FUNCTION_DECL):
            name = cursor.spelling
            if name == 'OnInit':
                # Insert initialization at start of method body
                body = cursor.get_definition()
                if body is None:
                    continue
                start = cursor.extent.start
                # find '{' opening brace offset by scanning
                brace_offset = pos_to_offset(code, start.line, start.column)
                # Search next '{' from start to body end
                open_brace_pos = code.find('{', brace_offset)
                if open_brace_pos != -1:
                    init_code = (
                        '\n        // GPA instrumentation init (injected AST)\n'
                        '        {\n'
                        '            GpaStatus status = GpaInitialize(kGpaInitializeDefaultBit);\n'
                        '            if (status != kGpaStatusOk) { std::cerr << "GpaInitialize failed: " << status << std::endl; }\n'
                        f'            status = GpaInterfaceLoaderLoadApi(kGpaApiVulkan, "{libgpa_path}");\n'
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
                    inserts.append((open_brace_pos + 1, init_code))
            elif name == 'OnCleanUp':
                # Insert cleanup code at start of OnCleanUp body
                start = cursor.extent.start
                brace_offset = pos_to_offset(code, start.line, start.column)
                open_brace_pos = code.find('{', brace_offset)
                if open_brace_pos != -1:
                    cleanup_code = (
                        '\n        // GPA instrumentation cleanup (injected AST)\n'
                        '        if (gpaSession) GpaEndSession(gpaSession);\n'
                        '        if (gpaSession) GpaDeleteSession(gpaSession);\n'
                        '        if (gpaCtx) GpaCloseContext(gpaCtx);\n'
                        '        GpaDestroy();\n\n'
                    )
                    inserts.append((open_brace_pos + 1, cleanup_code))
            elif name in ('DrawTriangle', 'DrawTriangleCb', 'DrawTriangleWithRPSCb', 'DrawTriangleBreathingCb'):
                # Find vkCmdDraw and vkCmdDrawIndexed CallExpr positions inside this function
                for node in cursor.get_children():
                    # walk deeper: we'll search recursively for call exprs
                    for c in node.walk_preorder():
                        if c.kind == cindex.CursorKind.CALL_EXPR:
                            try:
                                callee = c.get_definition()
                            except Exception:
                                callee = None
                            callee_name = c.displayname
                            if not callee_name:
                                # try to get tokens
                                callee_name = ''.join([t.spelling for t in c.get_tokens()][:1])
                            if 'vkCmdDraw' in callee_name or 'vkCmdDrawIndexed' in callee_name or 'DrawInstanced' in callee_name or 'vkCmdDrawIndirect' in callee_name:
                                # Insert begin before c extent start and end after c extent end
                                start = c.extent.start
                                end = c.extent.end
                                begin_off = pos_to_offset(code, start.line, start.column)
                                end_off = pos_to_offset(code, end.line, end.column)
                                begin_code = ' { GpaSampleId sample = 0; GpaBeginSample(gpaSession, nullptr, &sample); '
                                end_code = ' GpaEndSample(gpaSession); '
                                # After end, write sample result to file
                                write_code = """
                                    size_t __gpa_size = 0; 
                                    if (GpaGetSampleResultSize(gpaSession, sample, &__gpa_size) == kGpaStatusOk) { 
                                        std::vector<char> __gpa_buf(__gpa_size); 
                                        if (GpaGetSampleResult(gpaSession, sample, __gpa_size, __gpa_buf.data()) == kGpaStatusOk) { 
                                            uint64_t* __gpa_vals = reinterpret_cast<uint64_t*>(__gpa_buf.data()); 
                                            size_t __gpa_count = __gpa_size / sizeof(uint64_t); 
                                            if (gpa_out.is_open()) { gpa_out << 0; /* placeholder iter */ 
                                                for (size_t __gi = 0; __gi < gpa_enabled_counters.size(); ++__gi) { 
                                                    uint64_t __v = 0; if (__gi < __gpa_count) __v = __gpa_vals[__gi]; 
                                                    gpa_out << "," << __v; 
                                                } 
                                                gpa_out << "\n"; 
                                            } 
                                        } 
                                    } 
                                """
                                inserts.append((end_off, write_code))
                                inserts.append((end_off, end_code))
                                inserts.append((begin_off, begin_code))
                                inserts.append((end_off, end_code))
                                inserts.append((begin_off, begin_code))

    # Sort inserts by offset descending and apply
    inserts.sort(key=lambda x: x[0], reverse=True)
    new_code = code
    for off, t in inserts:
        if off < 0:
            off = 0
        if off > len(new_code):
            off = len(new_code)
        new_code = new_code[:off] + t + new_code[off:]

    bak = backup(source_path)
    write_file(source_path, new_code)
    return bak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', required=True, help='Path to the _vk.cpp source to edit')
    parser.add_argument('--include', action='append', default=[], help='Include directories for clang parser')
    parser.add_argument('--libgpa', default='data/4_2/bin', help='Path to libGPA library directory')
    args = parser.parse_args()

    if cindex is None:
        print('clang.cindex not available; please install python clang bindings (pip install clang) and libclang.')
        sys.exit(1)

    if not os.path.exists(args.example):
        print('Source not found:', args.example)
        return 1

    try:
        bak = run_ast_instrument(args.example, include_dirs=args.include, libgpa_path=args.libgpa)
        print('Patched: {} (backup {})'.format(args.example, bak))
        return 0
    except Exception as e:
        print('Failed to apply AST instrumentation:', e)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
