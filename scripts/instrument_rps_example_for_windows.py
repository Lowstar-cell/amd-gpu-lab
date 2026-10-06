#!/usr/bin/env python3
"""
Patch a RenderPipelineShaders example (hello_triangle.cpp) to add GPA instrumentation
for Windows/D3D12 builds. This script is intended to be run on a Windows host where
you will build the sample with Visual Studio or MSBuild. It creates a backup of the
original file and inserts GPA initialization and sample begin/end calls.

Usage:
  python scripts/instrument_rps_example_for_windows.py --example data/RenderPipelineShaders-main/examples/hello_triangle.cpp

NOTE: This script modifies source files and should be used with a VCS or backup.
"""
import argparse
import os
import shutil
import re


def backup(path):
    if not os.path.exists(path + '.bak'):
        shutil.copy(path, path + '.bak')


def patch_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()

    if 'gpu_perf_api.h' in src:
        print('Already instrumented')
        return False

    # Insert includes after existing includes
    src = src.replace('#include "app_framework/afx_d3d12_renderer.hpp"', '#include "app_framework/afx_d3d12_renderer.hpp"\n#include "gpu_performance_api/gpu_perf_api.h"\n#include "gpu_performance_api/gpu_perf_api_interface_loader.h"')

    # Insert global static variables near top of file
    insert_point = src.find('class HelloTriangle')
    if insert_point == -1:
        print('Could not find HelloTriangle class declaration')
        return False
    globals_code = '\n// GPA Instrumentation globals (added by patch)\nstatic GpaContextId g_gpaCtx = NULL;\nstatic GpaSessionId g_gpaSession = NULL;\nstatic GpaUInt32 g_gpaPassCount = 0;\n'
    src = src[:insert_point] + globals_code + src[insert_point:]

    # Insert initialization in OnInit: find the m_rpsDevice creation and after that add GPA init
    on_init_marker = 'AssertIfRpsFailed(rpsRenderGraphCreate(m_rpsDevice, &renderGraphInfo, &m_rpsRenderGraph));'
    idx = src.find(on_init_marker)
    if idx != -1:
        insertion = '\n        // GPA instrumentation init\n        GpaStatus gpaStatus = GpaInitialize(kGpaInitializeDefaultBit);\n        if (gpaStatus == kGpaStatusOk) {\n            GpaStatus s = GpaOpenContext((void*)m_device.Get(), kGpaOpenContextEnableHardwareCountersBit, &g_gpaCtx);\n            if (s == kGpaStatusOk) {\n                GpaCreateSession(g_gpaCtx, kGpaSessionSampleTypeDiscreteCounter, &g_gpaSession);\n                if (g_gpaSession) {\n                    GpaBeginSession(g_gpaSession);\n                    // enable counters here if desired (example: first 4)\n                    for (GpaUInt32 i=0;i<4;i++) { if (GpaEnableCounter(g_gpaSession, i)==kGpaStatusOk) {} }\n                    GpaGetPassCount(g_gpaSession, &g_gpaPassCount);\n                }\n            }\n        }\n'
        src = src[:idx+len(on_init_marker)] + insertion + src[idx+len(on_init_marker):]

    # Insert GpaBeginSample/GpaEndSample around DrawTriangle: in DrawTriangleCb or DrawTriangle
    # For simplicity, modify DrawTriangle to do Begin/End sample
    draw_fn_marker = 'void DrawTriangle(ID3D12GraphicsCommandList* pCmdList)'
    idx = src.find(draw_fn_marker)
    if idx != -1:
        # find the opening brace after marker
        brace_idx = src.find('{', idx)
        if brace_idx != -1:
            begin_marker = '\n        // GPA Begin Sample\n        if (g_gpaSession) {\n            GpaSampleId sid = 0;\n            GpaBeginSample(g_gpaSession, nullptr, &sid);\n        }\n'
            end_marker = '\n        // GPA End Sample\n        if (g_gpaSession) {\n            GpaEndSample(g_gpaSession);\n        }\n'
            # insert begin after opening brace
            src = src[:brace_idx+1] + begin_marker + src[brace_idx+1:]
            # find pCmdList->DrawInstanced( and insert End sample after
            draw_call_idx = src.find('pCmdList->DrawInstanced', brace_idx)
            if draw_call_idx != -1:
                # insert end marker after the draw call's semicolon
                semicolon_idx = src.find(';', draw_call_idx)
                if semicolon_idx != -1:
                    src = src[:semicolon_idx+1] + end_marker + src[semicolon_idx+1:]

    with open(path, 'w', encoding='utf-8') as f:
        f.write(src)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', default='data/RenderPipelineShaders-main/examples/hello_triangle.cpp')
    args = parser.parse_args()
    path = args.example
    if not os.path.exists(path):
        print('Not found', path)
        return
    backup(path)
    ok = patch_file(path)
    if ok:
        print('Patched', path)
    else:
        print('Failed to patch', path)

if __name__ == '__main__':
    main()
