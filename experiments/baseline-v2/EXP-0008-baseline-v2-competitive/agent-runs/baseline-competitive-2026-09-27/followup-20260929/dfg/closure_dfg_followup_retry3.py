#!/usr/bin/env python3
"""S1: independent Allen-Cahn reference plus DFG Taylor-Hood reference on Kaggle T4x2.

This script is a Kaggle kernel entrypoint. It never falls back to local/CPU-only
training. The DFG worker runs in a separate micromamba environment so the
PyTorch CUDA process remains isolated from PETSc/MPI libraries.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
from pathlib import Path

UTC_STARTED = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
STARTED_MONOTONIC = time.perf_counter()
SCRIPT_DIR = Path("/kaggle/working/dfg-fem-field-followup-retry3-source")
OUT = Path("/kaggle/working/dfg-fem-field-followup-retry3")
SESSION_CAP = 1546.92
WATCHDOG = 1480.0
DFG_CAP = 1350.0
FEATFLOW = {"Cd": 5.57953523384, "Cl": 0.010618948146, "delta_p": 0.11752016697}
EVAL_CONTRACT = {'grid': {'x': {'start': 0.0, 'stop': 2.2, 'count': 221, 'endpoint': True}, 'y': {'start': 0.0, 'stop': 0.41, 'count': 42, 'endpoint': True}, 'ordering': "numpy.meshgrid(indexing='xy'); row-major flattened", 'mask': 'exclude points with (x-0.2)^2+(y-0.2)^2 <= 0.05^2'}, 'pressure_gauge': {'method': 'subtract pressure at fixed outlet point from FEM and each checkpoint', 'outlet_anchor_xy': [2.2, 0.2]}, 'metrics': ['u_relative_l2', 'v_relative_l2', 'p_relative_l2_gauge_fixed_outlet'], 'cohorts': {'current_rbf': [301, 302, 303, 304, 305], 'refinement_rbf_and_tanh_mlp': [501, 502, 503, 504, 505]}, 'reference_level': 'DFG-L2 (third of three requested mesh levels)', 'retraining': False}

_EMBEDDED_DFG_ASSETS = {'solve_dfg.py': 'IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiJEUkFGVF9VTlZFUklGSUVEIHN0YXRpb25hcnkgVGF5bG9y4oCTSG9vZCBzb2x2ZXIgZm9yIERGRyAyRC0xIChSZT0yMCkuCgpUaGlzIHNvdXJjZSBpcyBhIHByb3RvY29sLWFsaWduZWQgZHJhZnQuIEl0IGhhcyBub3QgYmVlbiBydW4gYWdhaW5zdCBhIHJlc29sdmVkCkRPTEZJTngvR21zaC9QRVRTYyBlbnZpcm9ubWVudCBhbmQgY2Fubm90IHByb2R1Y2UgYW4gYWNjZXB0ZWQgcmVmZXJlbmNlIHVudGlsCnRoZSB2YWxpZGF0aW9uIGdhcHMgaW4gUkVBRE1FLm1kIGFyZSBjbG9zZWQuCiIiIgpmcm9tIF9fZnV0dXJlX18gaW1wb3J0IGFubm90YXRpb25zCgppbXBvcnQgYXJncGFyc2UKaW1wb3J0IGpzb24KaW1wb3J0IHN1YnByb2Nlc3MKaW1wb3J0IHN5cwppbXBvcnQgdGltZQpmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgKCmltcG9ydCBudW1weSBhcyBucApmcm9tIGJhc2l4LnVmbCBpbXBvcnQgZWxlbWVudCwgbWl4ZWRfZWxlbWVudApmcm9tIGRvbGZpbnggaW1wb3J0IGZlbSwgZ2VvbWV0cnksIGlvCmZyb20gZG9sZmlueC5mZW0gaW1wb3J0IHBldHNjIGFzIGZlbV9wZXRzYwpmcm9tIGRvbGZpbnguaW8gaW1wb3J0IGdtc2ggYXMgZ21zaGlvCmZyb20gbXBpNHB5IGltcG9ydCBNUEkKZnJvbSBwZXRzYzRweSBpbXBvcnQgUEVUU2MKaW1wb3J0IHVmbAoKU1RBVFVTID0gIkRSQUZUX1VOVkVSSUZJRUQiClJPT1QgPSBQYXRoKF9fZmlsZV9fKS5yZXNvbHZlKCkucGFyZW50CkdFTyA9IFJPT1QgLyAibWFrZV9tZXNoLmdlbyIKTCwgSCA9IDIuMiwgMC40MQpDWCwgQ1ksIFIgPSAwLjIsIDAuMiwgMC4wNQpSSE8sIE5VID0gMS4wLCAwLjAwMQpVTUFYLCBVTUVBTiwgRElBTUVURVIgPSAwLjMsIDAuMiwgMC4xClRBR1MgPSB7ImlubGV0IjogMSwgIndhbGxzIjogMiwgImN5bGluZGVyIjogMywgIm91dGxldCI6IDQsICJmbHVpZCI6IDV9CkxFVkVMUyA9IFsKICAgIHsiaWQiOiAiREZHLUwwIiwgImJ1bGsiOiAwLjA0MCwgImN5bGluZGVyIjogMC4wMDYwfSwKICAgIHsiaWQiOiAiREZHLUwxIiwgImJ1bGsiOiAwLjAyMCwgImN5bGluZGVyIjogMC4wMDMwfSwKICAgIHsiaWQiOiAiREZHLUwyIiwgImJ1bGsiOiAwLjAxMCwgImN5bGluZGVyIjogMC4wMDE1fSwKICAgIHsiaWQiOiAiREZHLUwzIiwgImJ1bGsiOiAwLjAwNSwgImN5bGluZGVyIjogMC4wMDA3NX0sCiAgICB7ImlkIjogIkRGRy1MNCIsICJidWxrIjogMC4wMDI1LCAiY3lsaW5kZXIiOiAwLjAwMDM3NX0sCiAgICB7ImlkIjogIkRGRy1MNSIsICJidWxrIjogMC4wMDEyNSwgImN5bGluZGVyIjogMC4wMDAxODc1fSwKXQpGRUFURkxPVyA9IHsiQ2QiOiA1LjU3OTUzNTIzMzg0LCAiQ2wiOiAwLjAxMDYxODk0ODE0NiwgImRlbHRhX3AiOiAwLjExNzUyMDE2Njk3fQoKCmRlZiBtYWtlX21lc2gobGV2ZWw6IGRpY3QsIG1lc2hfZGlyOiBQYXRoLCBnbXNoX2Jpbjogc3RyKSAtPiBQYXRoOgogICAgIiIiR2VuZXJhdGUgb25lIGN1cnZlZCwgcXVhZHJhdGljIEdtc2ggbWVzaCBmcm9tIHRoZSBzaGFyZWQgQ0FEIHNvdXJjZS4iIiIKICAgIG1lc2hfZGlyLm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rfb2s9VHJ1ZSkKICAgIG1zaCA9IG1lc2hfZGlyIC8gZiJkZmdfcmUyMF97bGV2ZWxbJ2lkJ119Lm1zaCIKICAgIGNvbW1hbmQgPSBbCiAgICAgICAgZ21zaF9iaW4sIHN0cihHRU8pLCAiLTIiLAogICAgICAgICItc2V0bnVtYmVyIiwgImxjX2J1bGsiLCBzdHIobGV2ZWxbImJ1bGsiXSksCiAgICAgICAgIi1zZXRudW1iZXIiLCAibGNfY3lsIiwgc3RyKGxldmVsWyJjeWxpbmRlciJdKSwKICAgICAgICAiLW8iLCBzdHIobXNoKSwKICAgIF0KICAgIHN1YnByb2Nlc3MucnVuKGNvbW1hbmQsIGNoZWNrPVRydWUpCiAgICByZXR1cm4gbXNoCgoKZGVmIHZlbG9jaXR5X2JvdW5kYXJ5X2NvbmRpdGlvbnMoZG9tYWluLCBmYWNldF90YWdzLCBXKToKICAgICIiIkltcG9zZSB0aGUgRkVBVEZMT1cgcGFyYWJvbGljIGluZmxvdyBhbmQgemVybyB2ZWxvY2l0eSBvbiBzb2xpZCB3YWxscy4iIiIKICAgIHRkaW0gPSBkb21haW4udG9wb2xvZ3kuZGltCiAgICBmZGltID0gdGRpbSAtIDEKICAgIGRvbWFpbi50b3BvbG9neS5jcmVhdGVfY29ubmVjdGl2aXR5KGZkaW0sIHRkaW0pCiAgICBWLCBfID0gVy5zdWIoMCkuY29sbGFwc2UoKQoKICAgIGlubGV0ID0gZmVtLkZ1bmN0aW9uKFYsIG5hbWU9ImlubGV0X3ZlbG9jaXR5IikKCiAgICBkZWYgcHJvZmlsZSh4KToKICAgICAgICB2YWx1ZXMgPSBucC56ZXJvcygoMiwgeC5zaGFwZVsxXSksIGR0eXBlPWRvbWFpbi5nZW9tZXRyeS54LmR0eXBlKQogICAgICAgIHkgPSB4WzFdCiAgICAgICAgdmFsdWVzWzBdID0gNC4wICogVU1BWCAqIHkgKiAoSCAtIHkpIC8gKEggKiBIKQogICAgICAgIHJldHVybiB2YWx1ZXMKCiAgICBpbmxldC5pbnRlcnBvbGF0ZShwcm9maWxlKQogICAgaW5sZXQueC5zY2F0dGVyX2ZvcndhcmQoKQogICAgemVybyA9IGZlbS5GdW5jdGlvbihWLCBuYW1lPSJ6ZXJvX3ZlbG9jaXR5IikKICAgIHplcm8ueC5hcnJheVs6XSA9IDAuMAogICAgemVyby54LnNjYXR0ZXJfZm9yd2FyZCgpCgogICAgZGVmIGRvZnModGFnKToKICAgICAgICByZXR1cm4gZmVtLmxvY2F0ZV9kb2ZzX3RvcG9sb2dpY2FsKAogICAgICAgICAgICAoVy5zdWIoMCksIFYpLCBmZGltLCBmYWNldF90YWdzLmZpbmQoVEFHU1t0YWddKQogICAgICAgICkKCiAgICByZXR1cm4gWwogICAgICAgIGZlbS5kaXJpY2hsZXRiYyhpbmxldCwgZG9mcygiaW5sZXQiKSwgVy5zdWIoMCkpLAogICAgICAgIGZlbS5kaXJpY2hsZXRiYyh6ZXJvLCBkb2ZzKCJ3YWxscyIpLCBXLnN1YigwKSksCiAgICAgICAgZmVtLmRpcmljaGxldGJjKHplcm8sIGRvZnMoImN5bGluZGVyIiksIFcuc3ViKDApKSwKICAgIF0KCgpkZWYgc29sdmVfc3RhdGlvbmFyeShkb21haW4sIGZhY2V0X3RhZ3MpOgogICAgIiIiU29sdmUgU3Rva2VzIGZpcnN0LCB0aGVuIGNvbnRpbnVlIHRoZSBjb252ZWN0aXZlIGNvZWZmaWNpZW50IHRvIG9uZS4iIiIKICAgIFAyID0gZWxlbWVudCgiTGFncmFuZ2UiLCBkb21haW4uYmFzaXhfY2VsbCgpLCAyLCBzaGFwZT0oMiwpKQogICAgUDEgPSBlbGVtZW50KCJMYWdyYW5nZSIsIGRvbWFpbi5iYXNpeF9jZWxsKCksIDEpCiAgICBXID0gZmVtLmZ1bmN0aW9uc3BhY2UoZG9tYWluLCBtaXhlZF9lbGVtZW50KFtQMiwgUDFdKSkKICAgIHN0YXRlID0gZmVtLkZ1bmN0aW9uKFcsIG5hbWU9Im1peGVkX3ZlbG9jaXR5X3ByZXNzdXJlIikKICAgIHRlc3RzID0gdWZsLlRlc3RGdW5jdGlvbnMoVykKICAgIHZlbG9jaXR5LCBwcmVzc3VyZSA9IHVmbC5zcGxpdChzdGF0ZSkKICAgIHRlc3RfdmVsb2NpdHksIHRlc3RfcHJlc3N1cmUgPSB0ZXN0cwogICAgYWxwaGEgPSBmZW0uQ29uc3RhbnQoZG9tYWluLCBQRVRTYy5TY2FsYXJUeXBlKDAuMCkpCiAgICB2aXNjb3NpdHkgPSBmZW0uQ29uc3RhbnQoZG9tYWluLCBQRVRTYy5TY2FsYXJUeXBlKE5VKSkKICAgIGR4ID0gdWZsLk1lYXN1cmUoImR4IiwgZG9tYWluPWRvbWFpbikKCiAgICAjIFRoaXMgaXMgdGhlIHdlYWsgZm9ybSBmaXhlZCBpbiByZWZlcmVuY2UtcHJvdG9jb2wtYWMtZGZnLm1kLiBUaGVyZSBpcyBubwogICAgIyBvdXRsZXQgYm91bmRhcnkgdGVybSBvciBvdXRsZXQgRGlyaWNobGV0IGNvbmRpdGlvbjogdGhlIG5hdHVyYWwgZG8tbm90aGluZwogICAgIyBjb25kaXRpb24gaXMgdGhlIGZyZWUgdHJhY3Rpb24gb24gdGhlIG9wZW4gZG93bnN0cmVhbSBib3VuZGFyeS4KICAgIHJlc2lkdWFsID0gKAogICAgICAgIFJITyAqIGFscGhhICogdWZsLmlubmVyKHVmbC5kb3QodWZsLmdyYWQodmVsb2NpdHkpLCB2ZWxvY2l0eSksIHRlc3RfdmVsb2NpdHkpCiAgICAgICAgKyB2aXNjb3NpdHkgKiB1ZmwuaW5uZXIodWZsLmdyYWQodmVsb2NpdHkpLCB1ZmwuZ3JhZCh0ZXN0X3ZlbG9jaXR5KSkKICAgICAgICAtIHByZXNzdXJlICogdWZsLmRpdih0ZXN0X3ZlbG9jaXR5KQogICAgICAgICsgdGVzdF9wcmVzc3VyZSAqIHVmbC5kaXYodmVsb2NpdHkpCiAgICApICogZHgKICAgIGJjcyA9IHZlbG9jaXR5X2JvdW5kYXJ5X2NvbmRpdGlvbnMoZG9tYWluLCBmYWNldF90YWdzLCBXKQogICAgb3B0aW9ucyA9IHsKICAgICAgICAic25lc190eXBlIjogIm5ld3RvbmxzIiwKICAgICAgICAic25lc19saW5lc2VhcmNoX3R5cGUiOiAiYnQiLAogICAgICAgICJzbmVzX3J0b2wiOiAxLjBlLTEwLAogICAgICAgICJzbmVzX2F0b2wiOiAxLjBlLTEyLAogICAgICAgICJzbmVzX3N0b2wiOiAxLjBlLTEyLAogICAgICAgICJzbmVzX21heF9pdCI6IDgwLAogICAgICAgICJzbmVzX2Vycm9yX2lmX25vdF9jb252ZXJnZWQiOiBUcnVlLAogICAgICAgICJrc3BfdHlwZSI6ICJwcmVvbmx5IiwKICAgICAgICAicGNfdHlwZSI6ICJsdSIsCiAgICAgICAgInBjX2ZhY3Rvcl9tYXRfc29sdmVyX3R5cGUiOiAibXVtcHMiLAogICAgICAgICJrc3BfZXJyb3JfaWZfbm90X2NvbnZlcmdlZCI6IFRydWUsCiAgICB9CiAgICBwcm9ibGVtID0gZmVtX3BldHNjLk5vbmxpbmVhclByb2JsZW0oCiAgICAgICAgcmVzaWR1YWwsCiAgICAgICAgc3RhdGUsCiAgICAgICAgYmNzPWJjcywKICAgICAgICBwZXRzY19vcHRpb25zX3ByZWZpeD0iZGZnX3JlMjBfIiwKICAgICAgICBwZXRzY19vcHRpb25zPW9wdGlvbnMsCiAgICApCgogICAgIyBhbHBoYT0wIGlzIHRoZSBTdG9rZXMgaW5pdGlhbGlzYXRpb24uIEZhaWxlZCBjb250aW51YXRpb24gc3RlcHMgYXJlIGxvZ2dlZAogICAgIyBhbmQgcmV0cmllZCBhdCBoYWxmIHN0ZXA7IGRvIG5vdCBzaWxlbnRseSBhZHZhbmNlIHBhc3QgYSBmYWlsZWQgc3RhZ2UuCiAgICBzdGFnZV9sb2cgPSBbXQogICAgbGFzdF9hbHBoYSA9IDAuMAogICAgc3RlcCA9IDAuMjUKICAgIHN0YXRlLnguYXJyYXlbOl0gPSAwLjAKICAgIHN0YXRlLnguc2NhdHRlcl9mb3J3YXJkKCkKCiAgICBkZWYgYXR0ZW1wdCh0YXJnZXQ6IGZsb2F0KSAtPiBib29sOgogICAgICAgIG5vbmxvY2FsIHN0YXRlCiAgICAgICAgcHJldmlvdXMgPSBzdGF0ZS54LmFycmF5LmNvcHkoKQogICAgICAgIGFscGhhLnZhbHVlID0gUEVUU2MuU2NhbGFyVHlwZSh0YXJnZXQpCiAgICAgICAgdHJ5OgogICAgICAgICAgICBwcm9ibGVtLnNvbHZlKCkKICAgICAgICAgICAgc3RhdGUueC5zY2F0dGVyX2ZvcndhcmQoKQogICAgICAgICAgICByZWFzb24gPSBpbnQocHJvYmxlbS5zb2x2ZXIuZ2V0Q29udmVyZ2VkUmVhc29uKCkpCiAgICAgICAgICAgIHNuZXNfaXRlcmF0aW9ucyA9IGludChwcm9ibGVtLnNvbHZlci5nZXRJdGVyYXRpb25OdW1iZXIoKSkKICAgICAgICAgICAgbm9ubGluZWFyX25vcm0gPSBmbG9hdChwcm9ibGVtLnNvbHZlci5nZXRGdW5jdGlvbk5vcm0oKSkKICAgICAgICAgICAga3NwID0gcHJvYmxlbS5zb2x2ZXIuZ2V0S1NQKCkKICAgICAgICAgICAgbGluZWFyX3JlYXNvbiA9IGludChrc3AuZ2V0Q29udmVyZ2VkUmVhc29uKCkpCiAgICAgICAgICAgIGxpbmVhcl9ub3JtID0gZmxvYXQoa3NwLmdldFJlc2lkdWFsTm9ybSgpKQogICAgICAgICAgICBzdWNjZXNzID0gcmVhc29uID4gMCBhbmQgbnAuaXNmaW5pdGUobm9ubGluZWFyX25vcm0pIGFuZCBsaW5lYXJfcmVhc29uID4gMAogICAgICAgICAgICBzdGFnZV9sb2cuYXBwZW5kKHsKICAgICAgICAgICAgICAgICJhbHBoYSI6IHRhcmdldCwKICAgICAgICAgICAgICAgICJhY2NlcHRlZF9ieV9zb2x2ZXIiOiBzdWNjZXNzLAogICAgICAgICAgICAgICAgInNuZXNfcmVhc29uIjogcmVhc29uLAogICAgICAgICAgICAgICAgInNuZXNfaXRlcmF0aW9ucyI6IHNuZXNfaXRlcmF0aW9ucywKICAgICAgICAgICAgICAgICJzbmVzX2ZpbmFsX25vcm0iOiBub25saW5lYXJfbm9ybSwKICAgICAgICAgICAgICAgICJrc3BfcmVhc29uIjogbGluZWFyX3JlYXNvbiwKICAgICAgICAgICAgICAgICJrc3BfZmluYWxfbm9ybSI6IGxpbmVhcl9ub3JtLAogICAgICAgICAgICB9KQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOiAgIyByZXRhaW4gdGhlIGZhaWxlZCBzdGFnZSBpbiB0aGUgcmVjb3JkCiAgICAgICAgICAgIHN0YXRlLnguYXJyYXlbOl0gPSBwcmV2aW91cwogICAgICAgICAgICBzdGF0ZS54LnNjYXR0ZXJfZm9yd2FyZCgpCiAgICAgICAgICAgIHN0YWdlX2xvZy5hcHBlbmQoeyJhbHBoYSI6IHRhcmdldCwgImFjY2VwdGVkX2J5X3NvbHZlciI6IEZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiZXhjZXB0aW9uIjogZiJ7dHlwZShleGMpLl9fbmFtZV9ffToge2V4Y30ifSkKICAgICAgICAgICAgcmV0dXJuIEZhbHNlCiAgICAgICAgaWYgbm90IHN1Y2Nlc3M6CiAgICAgICAgICAgIHN0YXRlLnguYXJyYXlbOl0gPSBwcmV2aW91cwogICAgICAgICAgICBzdGF0ZS54LnNjYXR0ZXJfZm9yd2FyZCgpCiAgICAgICAgcmV0dXJuIHN1Y2Nlc3MKCiAgICBpZiBub3QgYXR0ZW1wdCgwLjApOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigiU3Rva2VzIGluaXRpYWxpc2F0aW9uIGZhaWxlZDsgc2VlIHN0YWdlIGxvZyBpbiB0aGUgZXhjZXB0aW9uIGNvbnRleHQiKQogICAgcmVhY2hlZCA9IDAuMAogICAgd2hpbGUgcmVhY2hlZCA8IDEuMCAtIDEuMGUtMTQ6CiAgICAgICAgdGFyZ2V0ID0gbWluKDEuMCwgcmVhY2hlZCArIHN0ZXApCiAgICAgICAgaWYgYXR0ZW1wdCh0YXJnZXQpOgogICAgICAgICAgICByZWFjaGVkID0gdGFyZ2V0CiAgICAgICAgICAgIHN0ZXAgPSBtaW4oMC4yNSwgc3RlcCAqIDIuMCkKICAgICAgICBlbHNlOgogICAgICAgICAgICBzdGVwICo9IDAuNQogICAgICAgICAgICBpZiBzdGVwIDwgMC4wMTU2MjU6CiAgICAgICAgICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoZiJDb250aW51YXRpb24gc3RhbGxlZCBhdCBhbHBoYT17cmVhY2hlZH07IHN0YWdlcz17c3RhZ2VfbG9nfSIpCgogICAgc3RhdGUueC5zY2F0dGVyX2ZvcndhcmQoKQogICAgcmV0dXJuIFcsIHN0YXRlLCBzdGFnZV9sb2cKCgpkZWYgcG9pbnRfdmFsdWVzKGZ1bmN0aW9uLCBkb21haW4sIHh5X3BvaW50cyk6CiAgICAiIiJMb2NhdGUgYW5kIGV2YWx1YXRlIGEgZnVuY3Rpb24gYXQgZml4ZWQgcGh5c2ljYWwgcG9pbnRzIChzZXJpYWwgZHJhZnQpLiIiIgogICAgcG9pbnRzID0gbnAuemVyb3MoKGxlbih4eV9wb2ludHMpLCAzKSwgZHR5cGU9ZG9tYWluLmdlb21ldHJ5LnguZHR5cGUpCiAgICBwb2ludHNbOiwgOjJdID0geHlfcG9pbnRzCiAgICB0cmVlID0gZ2VvbWV0cnkuYmJfdHJlZShkb21haW4sIGRvbWFpbi50b3BvbG9neS5kaW0pCiAgICBjYW5kaWRhdGVzID0gZ2VvbWV0cnkuY29tcHV0ZV9jb2xsaXNpb25zX3BvaW50cyh0cmVlLCBwb2ludHMpCiAgICBjb2xsaWRpbmcgPSBnZW9tZXRyeS5jb21wdXRlX2NvbGxpZGluZ19jZWxscyhkb21haW4sIGNhbmRpZGF0ZXMsIHBvaW50cykKICAgIHZhbHVlcyA9IFtdCiAgICBmb3IgaW5kZXggaW4gcmFuZ2UobGVuKHBvaW50cykpOgogICAgICAgIGNlbGxzID0gY29sbGlkaW5nLmxpbmtzKGluZGV4KQogICAgICAgIGlmIGxlbihjZWxscykgPT0gMDoKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIlBvaW50IG5vdCBmb3VuZCBpbiBmbHVpZCBtZXNoOiB7eHlfcG9pbnRzW2luZGV4XS50b2xpc3QoKX0iKQogICAgICAgIGNlbGwgPSBucC5hc2FycmF5KFtjZWxsc1swXV0sIGR0eXBlPW5wLmludDMyKQogICAgICAgIHZhbHVlcy5hcHBlbmQoZnVuY3Rpb24uZXZhbChwb2ludHNbaW5kZXg6aW5kZXggKyAxXSwgY2VsbClbMF0pCiAgICByZXR1cm4gbnAuYXNhcnJheSh2YWx1ZXMpCgoKZGVmIGV4cG9ydF9sZXZlbChsZXZlbCwgbWVzaF9wYXRoOiBQYXRoLCBvdXRwdXRfZGlyOiBQYXRoKToKICAgICIiIlNvbHZlIG9uZSBsZXZlbCBhbmQgZXhwb3J0IHRoZSBtZXNoLCBGRSBjb2VmZmljaWVudHMsIHNhbXBsZWQgZmllbGRzIGFuZCBRb0lzLiIiIgogICAgY29tbSA9IE1QSS5DT01NX1dPUkxECiAgICBpZiBjb21tLnNpemUgIT0gMToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoIlRoaXMgZHJhZnQgaW50ZW50aW9uYWxseSBzdXBwb3J0cyBzZXJpYWwgcnVucyBvbmx5LiIpCiAgICBtZXNoX2RhdGEgPSBnbXNoaW8ucmVhZF9mcm9tX21zaChtZXNoX3BhdGgsIGNvbW0sIHJhbms9MCwgZ2RpbT0yKQogICAgZG9tYWluID0gbWVzaF9kYXRhLm1lc2gKICAgIGZhY2V0X3RhZ3MgPSBtZXNoX2RhdGEuZmFjZXRfdGFncwogICAgaWYgZmFjZXRfdGFncyBpcyBOb25lOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigiR21zaCBtZXNoIGhhcyBubyBib3VuZGFyeSBQaHlzaWNhbCBDdXJ2ZSB0YWdzLiIpCiAgICBXLCBzdGF0ZSwgc3RhZ2VzID0gc29sdmVfc3RhdGlvbmFyeShkb21haW4sIGZhY2V0X3RhZ3MpCiAgICB2ZWxvY2l0eSwgXyA9IHN0YXRlLnN1YigwKS5jb2xsYXBzZSgpCiAgICBwcmVzc3VyZSwgXyA9IHN0YXRlLnN1YigxKS5jb2xsYXBzZSgpCiAgICB2ZWxvY2l0eS5uYW1lID0gInZlbG9jaXR5IgogICAgcHJlc3N1cmUubmFtZSA9ICJwcmVzc3VyZSIKCiAgICAjIEZFQVRGTE9XIGRlZmluZXMgc2lnbWEgPSBudSpncmFkKHUpIC0gcCBJLCB3aXRoIGV0YSBkaXJlY3RlZCBvdXQgb2YgdGhlCiAgICAjIGN5bGluZGVyLiBUaGUgbWVzaCdzIGV4dGVyaW9yIG5vcm1hbCBwb2ludHMgaW50byB0aGUgY3lsaW5kZXIsIHNvIGV0YT0tbi4KICAgIF8sIHBfc3ltYm9sID0gdWZsLnNwbGl0KHN0YXRlKQogICAgdV9zeW1ib2wsIF8gPSB1Zmwuc3BsaXQoc3RhdGUpCiAgICBub3JtYWxfY3lsaW5kZXIgPSAtdWZsLkZhY2V0Tm9ybWFsKGRvbWFpbikKICAgIHN0cmVzcyA9IHZpc2Nvc2l0eV9mb3JfcW9pKGRvbWFpbikgKiB1ZmwuZ3JhZCh1X3N5bWJvbCkgLSBwX3N5bWJvbCAqIHVmbC5JZGVudGl0eSgyKQogICAgZHMgPSB1ZmwuTWVhc3VyZSgiZHMiLCBkb21haW49ZG9tYWluLCBzdWJkb21haW5fZGF0YT1mYWNldF90YWdzKQogICAgdHJhY3Rpb24gPSB1ZmwuZG90KHN0cmVzcywgbm9ybWFsX2N5bGluZGVyKQogICAgbG9jYWxfZHJhZyA9IGZlbS5hc3NlbWJsZV9zY2FsYXIoZmVtLmZvcm0odHJhY3Rpb25bMF0gKiBkcyhUQUdTWyJjeWxpbmRlciJdKSkpCiAgICBsb2NhbF9saWZ0ID0gZmVtLmFzc2VtYmxlX3NjYWxhcihmZW0uZm9ybSh0cmFjdGlvblsxXSAqIGRzKFRBR1NbImN5bGluZGVyIl0pKSkKICAgIGRyYWdfZm9yY2UgPSBjb21tLmFsbHJlZHVjZShmbG9hdChsb2NhbF9kcmFnKSwgb3A9TVBJLlNVTSkKICAgIGxpZnRfZm9yY2UgPSBjb21tLmFsbHJlZHVjZShmbG9hdChsb2NhbF9saWZ0KSwgb3A9TVBJLlNVTSkKICAgIHNjYWxlID0gMi4wIC8gKFJITyAqIFVNRUFOICogVU1FQU4gKiBESUFNRVRFUikKCiAgICAjIFRoZSB0d28gcHJlc2NyaWJlZCBwcmVzc3VyZSBwcm9iZXMgbGllIG9uIHRoZSBjeWxpbmRlciBib3VuZGFyeS4gVGhlCiAgICAjIHBvaW50IGxvY2F0b3IgaXMgZGVsaWJlcmF0ZWx5IGV4cG9zZWQgYXMgYW4gdW52ZXJpZmllZCBvbmUtc2lkZWQgdHJhY2UuCiAgICBwcm9iZXMgPSBucC5hc2FycmF5KFtbMC4xNSwgMC4yXSwgWzAuMjUsIDAuMl1dLCBkdHlwZT1ucC5mbG9hdDY0KQogICAgcHJvYmVfcHJlc3N1cmUgPSBwb2ludF92YWx1ZXMocHJlc3N1cmUsIGRvbWFpbiwgcHJvYmVzKS5yZXNoYXBlKC0xKQogICAgZGVsdGFfcCA9IGZsb2F0KHByb2JlX3ByZXNzdXJlWzBdIC0gcHJvYmVfcHJlc3N1cmVbMV0pCgogICAgeGcgPSBucC5saW5zcGFjZSgwLjAsIEwsIDIyMSkKICAgIHlnID0gbnAubGluc3BhY2UoMC4wLCBILCA0MikKICAgIHh4LCB5eSA9IG5wLm1lc2hncmlkKHhnLCB5ZywgaW5kZXhpbmc9Inh5IikKICAgIGdyaWRfeHkgPSBucC5jb2x1bW5fc3RhY2soKHh4LnJhdmVsKCksIHl5LnJhdmVsKCkpKQogICAgZmx1aWRfbWFzayA9ICgoZ3JpZF94eVs6LCAwXSAtIENYKSAqKiAyICsgKGdyaWRfeHlbOiwgMV0gLSBDWSkgKiogMikgPiBSICoqIDIKICAgIGdyaWRfeHkgPSBncmlkX3h5W2ZsdWlkX21hc2tdCiAgICB2ZWxvY2l0eV9ncmlkID0gcG9pbnRfdmFsdWVzKHZlbG9jaXR5LCBkb21haW4sIGdyaWRfeHkpCiAgICBwcmVzc3VyZV9ncmlkID0gcG9pbnRfdmFsdWVzKHByZXNzdXJlLCBkb21haW4sIGdyaWRfeHkpLnJlc2hhcGUoLTEpCgogICAgb3V0cHV0X2Rpci5ta2RpcihwYXJlbnRzPVRydWUsIGV4aXN0X29rPVRydWUpCiAgICBucC5zYXZlel9jb21wcmVzc2VkKAogICAgICAgIG91dHB1dF9kaXIgLyAiZmllbGRzX29uX2NvbW1vbl9ncmlkLm5weiIsCiAgICAgICAgeHk9Z3JpZF94eSwKICAgICAgICB2ZWxvY2l0eT12ZWxvY2l0eV9ncmlkLAogICAgICAgIHByZXNzdXJlPXByZXNzdXJlX2dyaWQsCiAgICAgICAgcHJlc3N1cmVfcHJvYmVfeHk9cHJvYmVzLAogICAgICAgIHByZXNzdXJlX3Byb2JlX3ZhbHVlcz1wcm9iZV9wcmVzc3VyZSwKICAgICkKICAgIG5wLnNhdmV6X2NvbXByZXNzZWQoCiAgICAgICAgb3V0cHV0X2RpciAvICJmZW1fY29lZmZpY2llbnRzLm5weiIsCiAgICAgICAgbWVzaF9nZW9tZXRyeT1kb21haW4uZ2VvbWV0cnkueCwKICAgICAgICB2ZWxvY2l0eV9kb2ZfY29vcmRpbmF0ZXM9Vy5zdWIoMCkuY29sbGFwc2UoKVswXS50YWJ1bGF0ZV9kb2ZfY29vcmRpbmF0ZXMoKSwKICAgICAgICB2ZWxvY2l0eV9jb2VmZmljaWVudHM9dmVsb2NpdHkueC5hcnJheSwKICAgICAgICBwcmVzc3VyZV9kb2ZfY29vcmRpbmF0ZXM9Vy5zdWIoMSkuY29sbGFwc2UoKVswXS50YWJ1bGF0ZV9kb2ZfY29vcmRpbmF0ZXMoKSwKICAgICAgICBwcmVzc3VyZV9jb2VmZmljaWVudHM9cHJlc3N1cmUueC5hcnJheSwKICAgICAgICBlbGVtZW50X3BhaXI9bnAuYXNhcnJheShbIkxhZ3JhbmdlLVAyLXZlY3RvciIsICJMYWdyYW5nZS1QMS1wcmVzc3VyZSJdKSwKICAgICkKICAgIHFvaXMgPSB7CiAgICAgICAgInN0YXR1cyI6IFNUQVRVUywKICAgICAgICAibGV2ZWwiOiBsZXZlbFsiaWQiXSwKICAgICAgICAiZHJhZ19mb3JjZSI6IGRyYWdfZm9yY2UsCiAgICAgICAgImxpZnRfZm9yY2UiOiBsaWZ0X2ZvcmNlLAogICAgICAgICJDZCI6IHNjYWxlICogZHJhZ19mb3JjZSwKICAgICAgICAiQ2wiOiBzY2FsZSAqIGxpZnRfZm9yY2UsCiAgICAgICAgInByZXNzdXJlX2Zyb250IjogZmxvYXQocHJvYmVfcHJlc3N1cmVbMF0pLAogICAgICAgICJwcmVzc3VyZV9yZWFyIjogZmxvYXQocHJvYmVfcHJlc3N1cmVbMV0pLAogICAgICAgICJkZWx0YV9wX2Zyb250X21pbnVzX3JlYXIiOiBkZWx0YV9wLAogICAgICAgICJmZWF0Zmxvd19yZWZlcmVuY2UiOiBGRUFURkxPVywKICAgICAgICAic25lc19zdGFnZXMiOiBzdGFnZXMsCiAgICAgICAgImNlbGxfY291bnQiOiBpbnQoZG9tYWluLnRvcG9sb2d5LmluZGV4X21hcChkb21haW4udG9wb2xvZ3kuZGltKS5zaXplX2dsb2JhbCksCiAgICAgICAgImdlb21ldHJ5X25vZGVzX2xvY2FsIjogaW50KGRvbWFpbi5nZW9tZXRyeS54LnNoYXBlWzBdKSwKICAgICAgICAiZG9mc19nbG9iYWwiOiBpbnQoVy5kb2ZtYXAuaW5kZXhfbWFwLnNpemVfZ2xvYmFsICogVy5kb2ZtYXAuaW5kZXhfbWFwX2JzKSwKICAgICAgICAicHJlc3N1cmVfcHJvYmVfcG9saWN5IjogImRpcmVjdCBwb2ludCBldmFsdWF0aW9uOyBib3VuZGFyeSBvbmUtc2lkZWQgdHJhY2Ugbm90IHlldCBpbmRlcGVuZGVudGx5IHZlcmlmaWVkIiwKICAgICAgICAicmVmZXJlbmNlX2FjY2VwdGVkIjogRmFsc2UsCiAgICB9CiAgICAob3V0cHV0X2RpciAvICJxb2lzLmpzb24iKS53cml0ZV90ZXh0KGpzb24uZHVtcHMocW9pcywgaW5kZW50PTIsIGFsbG93X25hbj1GYWxzZSkgKyAiXG4iLCBlbmNvZGluZz0idXRmLTgiKQogICAgd2l0aCBpby5YRE1GRmlsZShjb21tLCBzdHIob3V0cHV0X2RpciAvICJtZXNoLnhkbWYiKSwgInciKSBhcyB4ZG1mOgogICAgICAgIHhkbWYud3JpdGVfbWVzaChkb21haW4pCiAgICByZXR1cm4gcW9pcwoKCmRlZiB2aXNjb3NpdHlfZm9yX3FvaShkb21haW4pOgogICAgcmV0dXJuIGZlbS5Db25zdGFudChkb21haW4sIFBFVFNjLlNjYWxhclR5cGUoTlUpKQoKCmRlZiBtYWluKCk6CiAgICBwYXJzZXIgPSBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcihkZXNjcmlwdGlvbj1fX2RvY19fKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1vdXRwdXQiLCB0eXBlPVBhdGgsIGRlZmF1bHQ9Uk9PVCAvICJydW5zIikKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tZ21zaCIsIGRlZmF1bHQ9Imdtc2giLCBoZWxwPSJHbXNoIGV4ZWN1dGFibGUgb24gUEFUSCIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLW1heC1sZXZlbHMiLCB0eXBlPWludCwgZGVmYXVsdD0zLAogICAgICAgICAgICAgICAgICAgICAgICBoZWxwPSJydW4gdGhlIGZpcnN0IDEuLjYgcGxhbm5lZCBtZXNoIGxldmVscyIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXdhbGwtY2FwLXNlY29uZHMiLCB0eXBlPWZsb2F0LCBkZWZhdWx0PTE0NTAuMCwKICAgICAgICAgICAgICAgICAgICAgICAgaGVscD0id29ya2VyIHdhbGwgY2FwOyBjb21wbGV0ZWQgbGV2ZWwgb3V0cHV0cyBwZXJzaXN0IGFzIHBhcnRpYWwgZXZpZGVuY2UiKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1taW5pbXVtLWxldmVsLXJlc2VydmUtc2Vjb25kcyIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9MTgwLjAsCiAgICAgICAgICAgICAgICAgICAgICAgIGhlbHA9ImRvIG5vdCBzdGFydCBhIG5ldyBtZXNoIHVubGVzcyB0aGlzIG11Y2ggY2FwIHJlbWFpbnMiKQogICAgYXJncyA9IHBhcnNlci5wYXJzZV9hcmdzKCkKICAgIGlmIG5vdCAxIDw9IGFyZ3MubWF4X2xldmVscyA8PSBsZW4oTEVWRUxTKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCItLW1heC1sZXZlbHMgbXVzdCBiZSBiZXR3ZWVuIDEgYW5kIHRoZSBudW1iZXIgb2YgZGVjbGFyZWQgbGV2ZWxzIikKICAgIGlmIGFyZ3Mud2FsbF9jYXBfc2Vjb25kcyA8PSAwIG9yIGFyZ3MubWluaW11bV9sZXZlbF9yZXNlcnZlX3NlY29uZHMgPCAwOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIndhbGwgY2FwIG11c3QgYmUgcG9zaXRpdmUgYW5kIHJlc2VydmUgbm9uLW5lZ2F0aXZlIikKICAgIGxldmVscyA9IExFVkVMU1s6YXJncy5tYXhfbGV2ZWxzXQogICAgc3RhcnRlZCA9IHRpbWUucGVyZl9jb3VudGVyKCkKICAgIGRlYWRsaW5lID0gc3RhcnRlZCArIGFyZ3Mud2FsbF9jYXBfc2Vjb25kcwogICAgcmVzdWx0cyA9IFtdCiAgICBzdW1tYXJ5X3BhdGggPSBhcmdzLm91dHB1dCAvICJzdW1tYXJ5Lmpzb24iCiAgICBlbnZpcm9ubWVudCA9IHsKICAgICAgICAicHl0aG9uIjogc3lzLnZlcnNpb24uc3BsaXQoKVswXSwKICAgICAgICAiZG9sZmlueCI6IGdldGF0dHIoX19pbXBvcnRfXygiZG9sZmlueCIpLCAiX192ZXJzaW9uX18iLCAidW5rbm93biIpLAogICAgICAgICJnbXNoX2JpbmFyeSI6IHN1YnByb2Nlc3MuY2hlY2tfb3V0cHV0KFthcmdzLmdtc2gsICItLXZlcnNpb24iXSwgdGV4dD1UcnVlKS5zdHJpcCgpLAogICAgICAgICJwZXRzYyI6IHN0cihQRVRTYy5TeXMuZ2V0VmVyc2lvbigpKSwKICAgICAgICAidWZsIjogZ2V0YXR0cih1ZmwsICJfX3ZlcnNpb25fXyIsICJ1bmtub3duIiksCiAgICAgICAgImdtc2hfcHl0aG9uIjogZ2V0YXR0cihfX2ltcG9ydF9fKCJnbXNoIiksICJfX3ZlcnNpb25fXyIsICJ1bmtub3duIiksCiAgICB9CgogICAgZGVmIHBlcnNpc3Qoc3RhdHVzOiBzdHIsICosIGVycm9yOiBzdHIgfCBOb25lID0gTm9uZSkgLT4gTm9uZToKICAgICAgICBxb2lfZXJyb3JzID0gTm9uZQogICAgICAgIG1lc2hfY2hhbmdlID0gTm9uZQogICAgICAgIGlmIHJlc3VsdHM6CiAgICAgICAgICAgIGZpbmFsID0gcmVzdWx0c1stMV0KICAgICAgICAgICAgcW9pX2Vycm9ycyA9IHsKICAgICAgICAgICAgICAgICJDZF9yZWxhdGl2ZSI6IGFicyhmaW5hbFsiQ2QiXSAtIEZFQVRGTE9XWyJDZCJdKSAvIEZFQVRGTE9XWyJDZCJdLAogICAgICAgICAgICAgICAgIkNsX2Fic29sdXRlIjogYWJzKGZpbmFsWyJDbCJdIC0gRkVBVEZMT1dbIkNsIl0pLAogICAgICAgICAgICAgICAgImRlbHRhX3BfcmVsYXRpdmUiOiBhYnMoZmluYWxbImRlbHRhX3BfZnJvbnRfbWludXNfcmVhciJdIC0gRkVBVEZMT1dbImRlbHRhX3AiXSkgLyBGRUFURkxPV1siZGVsdGFfcCJdLAogICAgICAgICAgICB9CiAgICAgICAgaWYgbGVuKHJlc3VsdHMpID49IDI6CiAgICAgICAgICAgIGNvYXJzZSwgZmluZSA9IHJlc3VsdHNbLTI6XQogICAgICAgICAgICBtZXNoX2NoYW5nZSA9IHsKICAgICAgICAgICAgICAgICJDZF9yZWxhdGl2ZSI6IGFicyhmaW5lWyJDZCJdIC0gY29hcnNlWyJDZCJdKSAvIG1heChhYnMoZmluZVsiQ2QiXSksIDFlLTMwKSwKICAgICAgICAgICAgICAgICJDbF9yZWxhdGl2ZSI6IGFicyhmaW5lWyJDbCJdIC0gY29hcnNlWyJDbCJdKSAvIG1heChhYnMoZmluZVsiQ2wiXSksIDFlLTMwKSwKICAgICAgICAgICAgICAgICJDbF9hYnNvbHV0ZSI6IGFicyhmaW5lWyJDbCJdIC0gY29hcnNlWyJDbCJdKSwKICAgICAgICAgICAgICAgICJkZWx0YV9wX3JlbGF0aXZlIjogYWJzKGZpbmVbImRlbHRhX3BfZnJvbnRfbWludXNfcmVhciJdIC0gY29hcnNlWyJkZWx0YV9wX2Zyb250X21pbnVzX3JlYXIiXSkgLyBtYXgoYWJzKGZpbmVbImRlbHRhX3BfZnJvbnRfbWludXNfcmVhciJdKSwgMWUtMzApLAogICAgICAgICAgICB9CiAgICAgICAgZXZpZGVuY2VfcmVhZHkgPSBib29sKAogICAgICAgICAgICBzdGF0dXMgPT0gImNhbmRpZGF0ZV9jb21wbGV0ZSIgYW5kIGxlbihyZXN1bHRzKSA+PSAzIGFuZCBxb2lfZXJyb3JzIGFuZCBtZXNoX2NoYW5nZQogICAgICAgICAgICBhbmQgcW9pX2Vycm9yc1siQ2RfcmVsYXRpdmUiXSA8PSAwLjAwNQogICAgICAgICAgICBhbmQgcW9pX2Vycm9yc1siQ2xfYWJzb2x1dGUiXSA8PSAwLjAwMDUKICAgICAgICAgICAgYW5kIHFvaV9lcnJvcnNbImRlbHRhX3BfcmVsYXRpdmUiXSA8PSAwLjAwNQogICAgICAgICAgICBhbmQgbWVzaF9jaGFuZ2VbIkNkX3JlbGF0aXZlIl0gPCAwLjAwMgogICAgICAgICAgICBhbmQgbWVzaF9jaGFuZ2VbIkNsX3JlbGF0aXZlIl0gPCAwLjAwMgogICAgICAgICAgICBhbmQgbWVzaF9jaGFuZ2VbImRlbHRhX3BfcmVsYXRpdmUiXSA8IDAuMDAyCiAgICAgICAgKQogICAgICAgIHBheWxvYWQgPSB7CiAgICAgICAgICAgICJzdGF0dXMiOiBzdGF0dXMsCiAgICAgICAgICAgICJyZWZlcmVuY2VfYWNjZXB0ZWQiOiBGYWxzZSwKICAgICAgICAgICAgInNjaWVudGlmaWNfYWNjZXB0ZWQiOiBGYWxzZSwKICAgICAgICAgICAgImV2aWRlbmNlX3JlYWR5X2Zvcl9odW1hbl9nYXRlIjogZXZpZGVuY2VfcmVhZHksCiAgICAgICAgICAgICJzZXNzaW9uX2NhcF9zZWNvbmRzIjogZmxvYXQoYXJncy53YWxsX2NhcF9zZWNvbmRzKSwKICAgICAgICAgICAgImVsYXBzZWRfc2Vjb25kcyI6IHJvdW5kKHRpbWUucGVyZl9jb3VudGVyKCkgLSBzdGFydGVkLCAzKSwKICAgICAgICAgICAgInBsYW5uZWRfbGV2ZWxfY291bnQiOiBsZW4obGV2ZWxzKSwKICAgICAgICAgICAgImNvbXBsZXRlZF9sZXZlbF9jb3VudCI6IGxlbihyZXN1bHRzKSwKICAgICAgICAgICAgImVudmlyb25tZW50X3ZlcnNpb25zIjogZW52aXJvbm1lbnQsCiAgICAgICAgICAgICJyZWZlcmVuY2VfcW9pX2Vycm9yc192c19mZWF0ZmxvdyI6IHFvaV9lcnJvcnMsCiAgICAgICAgICAgICJ0d29fZmluZXN0X21lc2hfcmVsYXRpdmVfY2hhbmdlcyI6IG1lc2hfY2hhbmdlLAogICAgICAgICAgICAibGV2ZWxzIjogcmVzdWx0cywKICAgICAgICB9CiAgICAgICAgaWYgZXJyb3I6CiAgICAgICAgICAgIHBheWxvYWRbImVycm9yIl0gPSBlcnJvcgogICAgICAgIHN1bW1hcnlfcGF0aC5wYXJlbnQubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1UcnVlKQogICAgICAgIHRlbXBvcmFyeSA9IHN1bW1hcnlfcGF0aC53aXRoX3N1ZmZpeCgiLmpzb24udG1wIikKICAgICAgICB0ZW1wb3Jhcnkud3JpdGVfdGV4dChqc29uLmR1bXBzKHBheWxvYWQsIGluZGVudD0yLCBhbGxvd19uYW49RmFsc2UpICsgIlxuIiwgZW5jb2Rpbmc9InV0Zi04IikKICAgICAgICB0ZW1wb3JhcnkucmVwbGFjZShzdW1tYXJ5X3BhdGgpCgogICAgYXJncy5vdXRwdXQubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1UcnVlKQogICAgcGVyc2lzdCgicnVubmluZyIpCiAgICB0cnk6CiAgICAgICAgZm9yIGxldmVsIGluIGxldmVsczoKICAgICAgICAgICAgaWYgZGVhZGxpbmUgLSB0aW1lLnBlcmZfY291bnRlcigpIDwgYXJncy5taW5pbXVtX2xldmVsX3Jlc2VydmVfc2Vjb25kczoKICAgICAgICAgICAgICAgIHBlcnNpc3QoInRpbWVfY2FwX3BhcnRpYWwiKQogICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICAgICAgbWVzaF9wYXRoID0gbWFrZV9tZXNoKGxldmVsLCBhcmdzLm91dHB1dCAvICJtZXNoZXMiLCBhcmdzLmdtc2gpCiAgICAgICAgICAgIHJlc3VsdCA9IGV4cG9ydF9sZXZlbChsZXZlbCwgbWVzaF9wYXRoLCBhcmdzLm91dHB1dCAvIGxldmVsWyJpZCJdKQogICAgICAgICAgICByZXN1bHRzLmFwcGVuZChyZXN1bHQpCiAgICAgICAgICAgIHBlcnNpc3QoInJ1bm5pbmciKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHBlcnNpc3QoImNhbmRpZGF0ZV9jb21wbGV0ZSIpCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoKICAgICAgICBwZXJzaXN0KCJmYWlsZWRfcGFydGlhbCIsIGVycm9yPWYie3R5cGUoZXhjKS5fX25hbWVfX306IHtleGN9IikKICAgICAgICByYWlzZQoKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICBtYWluKCkK', 'make_mesh.geo': 'Ly8gRFJBRlRfVU5WRVJJRklFRCDigJQgREZHIDJELTEsIFJlPTIwLCBwYXJhbWV0ZXJpemVkIEdtc2ggQ0FELgovLyBPdmVycmlkZSBsY19idWxrL2xjX2N5bCB3aXRoIGdtc2ggLXNldG51bWJlciBmb3IgZWFjaCByZWZpbmVtZW50IGxldmVsLgpTZXRGYWN0b3J5KCJPcGVuQ0FTQ0FERSIpOwoKTCA9IDIuMjsKSCA9IDAuNDE7CmN4ID0gMC4yOwpjeSA9IDAuMjsKciA9IDAuMDU7CmVwcyA9IDFlLTc7CgpEZWZpbmVDb25zdGFudFsKICBsY19idWxrID0gezAuMDQwLCBOYW1lICJQYXJhbWV0ZXJzL0J1bGsgbWVzaCBzaXplIiwgTWluIDAuMDAxLCBNYXggMC4yfSwKICBsY19jeWwgPSB7MC4wMDYwLCBOYW1lICJQYXJhbWV0ZXJzL0N5bGluZGVyIG1lc2ggc2l6ZSIsIE1pbiAwLjAwMDEsIE1heCAwLjA1fQpdOwoKUmVjdGFuZ2xlKDEpID0gezAsIDAsIDAsIEwsIEh9OwpEaXNrKDIpID0ge2N4LCBjeSwgMCwgciwgcn07CmZsdWlkW10gPSBCb29sZWFuRGlmZmVyZW5jZXsgU3VyZmFjZXsxfTsgRGVsZXRlOyB9eyBTdXJmYWNlezJ9OyBEZWxldGU7IH07CgovLyBCb3VuZGFyeSBlbnRpdGllcyBhcmUgc2VsZWN0ZWQgYWZ0ZXIgdGhlIEJvb2xlYW4gb3BlcmF0aW9uIHNvIHRoZSBjaXJjbGUKLy8gdGhhdCBpcyB0YWdnZWQgaXMgdGhlIGJvdW5kYXJ5IG9mIHRoZSBmbHVpZCBkb21haW4sIG5vdCB0aGUgZGlzY2FyZGVkIGRpc2suCmlubGV0W10gPSBDdXJ2ZSBJbiBCb3VuZGluZ0JveHstZXBzLCAtZXBzLCAtZXBzLCBlcHMsIEggKyBlcHMsIGVwc307Cm91dGxldFtdID0gQ3VydmUgSW4gQm91bmRpbmdCb3h7TCAtIGVwcywgLWVwcywgLWVwcywgTCArIGVwcywgSCArIGVwcywgZXBzfTsKYm90dG9tW10gPSBDdXJ2ZSBJbiBCb3VuZGluZ0JveHstZXBzLCAtZXBzLCAtZXBzLCBMICsgZXBzLCBlcHMsIGVwc307CnRvcFtdID0gQ3VydmUgSW4gQm91bmRpbmdCb3h7LWVwcywgSCAtIGVwcywgLWVwcywgTCArIGVwcywgSCArIGVwcywgZXBzfTsKY3lsaW5kZXJbXSA9IEN1cnZlIEluIEJvdW5kaW5nQm94e2N4IC0gciAtIGVwcywgY3kgLSByIC0gZXBzLCAtZXBzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3ggKyByICsgZXBzLCBjeSArIHIgKyBlcHMsIGVwc307CndhbGxzW10gPSB7Ym90dG9tW10sIHRvcFtdfTsKClBoeXNpY2FsIEN1cnZlKCJpbmxldCIsIDEpID0ge2lubGV0W119OwpQaHlzaWNhbCBDdXJ2ZSgid2FsbHMiLCAyKSA9IHt3YWxsc1tdfTsKUGh5c2ljYWwgQ3VydmUoImN5bGluZGVyIiwgMykgPSB7Y3lsaW5kZXJbXX07ClBoeXNpY2FsIEN1cnZlKCJvdXRsZXQiLCA0KSA9IHtvdXRsZXRbXX07ClBoeXNpY2FsIFN1cmZhY2UoImZsdWlkIiwgNSkgPSB7Zmx1aWRbXX07CgovLyBMb2NhbCByZWZpbmVtZW50IGFyb3VuZCB0aGUgY3VydmVkIG9ic3RhY2xlOyBzaXplcyBhcmUgdGhlIHByb3RvY29sJ3MKLy8gdGFyZ2V0IHZhbHVlcywgbm90IGEgcHJvbWlzZSBhYm91dCB0aGUgZ2VuZXJhdGVkIGVsZW1lbnQgY291bnQuCkZpZWxkWzFdID0gRGlzdGFuY2U7CkZpZWxkWzFdLkN1cnZlc0xpc3QgPSB7Y3lsaW5kZXJbXX07CkZpZWxkWzFdLlNhbXBsaW5nID0gMjAwOwpGaWVsZFsyXSA9IFRocmVzaG9sZDsKRmllbGRbMl0uSW5GaWVsZCA9IDE7CkZpZWxkWzJdLlNpemVNaW4gPSBsY19jeWw7CkZpZWxkWzJdLlNpemVNYXggPSBsY19idWxrOwpGaWVsZFsyXS5EaXN0TWluID0gMC4wMjsKRmllbGRbMl0uRGlzdE1heCA9IDAuMjU7CkJhY2tncm91bmQgRmllbGQgPSAyOwoKTWVzaC5NZXNoU2l6ZUZyb21Qb2ludHMgPSAwOwpNZXNoLk1lc2hTaXplRnJvbUN1cnZhdHVyZSA9IDA7Ck1lc2guTWVzaFNpemVFeHRlbmRGcm9tQm91bmRhcnkgPSAwOwpNZXNoLkVsZW1lbnRPcmRlciA9IDI7Ck1lc2guU2Vjb25kT3JkZXJMaW5lYXIgPSAwOwpNZXNoLk1zaEZpbGVWZXJzaW9uID0gNC4xOw0K'}

def materialize_dfg_assets():
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    for name, encoded in _EMBEDDED_DFG_ASSETS.items():
        (SCRIPT_DIR / name).write_bytes(base64.b64decode(encoded))
MAMBA_TMP = Path("/tmp/af-pikans-dfg-fem-followup-retry3-micromamba")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_status(part: str, status: str, **extra) -> None:
    write_json(OUT / f"{part}.json", {
        "part": part, "status": status, "scientific_accepted": False,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_elapsed_seconds": round(time.perf_counter() - STARTED_MONOTONIC, 3),
        **extra,
    })


def ensure_micromamba(log_path: Path, timeout: float) -> tuple[Path, Path]:
    """Create the isolated DOLFINx/Gmsh runtime on Kaggle; no host mutation."""
    root = MAMBA_TMP / "root"
    exe = MAMBA_TMP / "micromamba"
    env = MAMBA_TMP / "dfg-fem-env"
    deadline = time.perf_counter() + timeout
    MAMBA_TMP.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    if not exe.exists():
        archive = MAMBA_TMP / "micromamba.tar.bz2"
        request = urllib.request.Request(
            "https://micro.mamba.pm/api/micromamba/linux-64/latest",
            headers={"User-Agent": "AF-PIKANs-S1-reference/1.0"},
        )
        with urllib.request.urlopen(request, timeout=min(60.0, timeout)) as response, archive.open("wb") as dest:
            shutil.copyfileobj(response, dest)
        with tarfile.open(archive, "r:bz2") as bundle:
            member = next(item for item in bundle.getmembers() if item.name.endswith("bin/micromamba"))
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError("micromamba archive did not contain its executable")
            exe.write_bytes(source.read())
        exe.chmod(0o755)
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(
            [str(exe), "create", "--yes", "--root-prefix", str(root), "--prefix", str(env),
             "--channel", "conda-forge", "python=3.12", "fenics-dolfinx=0.11.*",
             "gmsh=4.15.*", "mumps", "numpy"],
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=max(1.0, deadline - time.perf_counter()),
        )
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(
            [str(exe), "run", "--root-prefix", str(root), "--prefix", str(env),
             "python", "-m", "pip", "install", "--disable-pip-version-check", "gmsh==4.15.2"],
            stdout=log, stderr=subprocess.STDOUT, check=True,
            timeout=max(1.0, deadline - time.perf_counter()),
        )
    return exe, env


def dfg_worker() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    log_path = OUT / "dfg-worker.log"
    write_status("s1b", "setting_up", worker_cap_seconds=DFG_CAP,
                 solver="Taylor-Hood P2/P1; Newton; natural do-nothing outlet")
    try:
        exe, env = ensure_micromamba(log_path, DFG_CAP - (time.perf_counter() - started) - 120.0)
        remaining = DFG_CAP - (time.perf_counter() - started) - 90.0
        if remaining <= 120:
            write_status("s1b", "setup_exhausted_cap", worker_elapsed_seconds=round(time.perf_counter() - started, 3))
            return 0
        command = [
            str(exe), "run", "--root-prefix", str(MAMBA_TMP / "root"), "--prefix", str(env),
            "python", str(SCRIPT_DIR / "solve_dfg.py"), "--output", str(OUT / "dfg"),
            "--max-levels", "3", "--wall-cap-seconds", str(min(DFG_CAP - 120.0, remaining)),
            "--minimum-level-reserve-seconds", "120",
        ]
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    check=False, timeout=max(60.0, remaining))
        summary_path = OUT / "dfg" / "summary.json"
        solver_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
        status = "complete" if result.returncode == 0 and solver_summary and solver_summary.get("status") == "candidate_complete" else "partial_or_failed"
        write_status("s1b", status, worker_elapsed_seconds=round(time.perf_counter() - started, 3),
                     process_returncode=result.returncode, solver_summary=solver_summary,
                     log=str(log_path))
        return 0
    except subprocess.TimeoutExpired as exc:
        write_status("s1b", "time_cap_partial", worker_elapsed_seconds=round(time.perf_counter() - started, 3),
                     exception=f"TimeoutExpired: {exc}", log=str(log_path))
        return 0
    except Exception as exc:
        write_status("s1b", "infrastructure_or_solver_error", worker_elapsed_seconds=round(time.perf_counter() - started, 3),
                     exception=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-5000:],
                     log=str(log_path))
        return 0


def dst1_orthonormal(values, n_intervals: int, torch):
    """Orthonormal DST-I through an odd extension and a complex FFT."""
    extended = torch.zeros((*values.shape[:-1], 2 * n_intervals), dtype=values.dtype, device=values.device)
    extended[..., 1:n_intervals] = values
    extended[..., n_intervals + 1:] = -torch.flip(values, dims=(-1,))
    return -torch.fft.fft(extended, dim=-1).imag[..., 1:n_intervals] * (math.sqrt(2.0 / n_intervals) / 2.0)


def etd_coefficients(n_intervals: int, dt: float, torch, device):
    import numpy as np
    modes = np.arange(1, n_intervals, dtype=np.float64)
    eigenvalues = -0.001 * (np.pi * modes / 2.0) ** 2
    z0 = dt * eigenvalues
    roots = np.exp(1j * np.pi * (np.arange(1, 33, dtype=np.float64) - 0.5) / 32.0)
    z = z0[None, :] + roots[:, None]
    exp_z = np.exp(z)
    q = dt * np.real(np.mean((np.exp(z / 2.0) - 1.0) / z, axis=0))
    f1 = dt * np.real(np.mean((-4.0 - z + exp_z * (4.0 - 3.0 * z + z * z)) / (z ** 3), axis=0))
    f2 = dt * np.real(np.mean((2.0 + z + exp_z * (-2.0 + z)) / (z ** 3), axis=0))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * z - z * z + exp_z * (4.0 - z)) / (z ** 3), axis=0))
    as_tensor = lambda value: torch.as_tensor(value, dtype=torch.float64, device=device)
    return as_tensor(np.exp(z0)), as_tensor(np.exp(z0 / 2.0)), as_tensor(q), as_tensor(f1), as_tensor(f2), as_tensor(f3)


def save_etd_checkpoint(path: Path, n: int, dt: float, step: int, w, saved_t: list[float], saved_u: list, torch) -> None:
    import numpy as np
    torch.cuda.synchronize()
    x = np.linspace(-1.0, 1.0, n + 1, dtype=np.float64)
    np.savez_compressed(path, x=x, dt=dt, step=step, current_time=step * dt,
                        interior_w=w.detach().cpu().numpy(), saved_t=np.asarray(saved_t),
                        saved_u=np.asarray(saved_u, dtype=np.float64))


def solve_etdrk4(n_intervals: int, dt: float, label: str, device, torch, deadline: float) -> dict:
    import numpy as np
    n_steps = int(round(1.0 / dt))
    steps_per_snapshot = int(round(0.005 / dt))
    x_np = np.linspace(-1.0, 1.0, n_intervals + 1, dtype=np.float64)
    x = torch.as_tensor(x_np, dtype=torch.float64, device=device)
    initial = x.square() * torch.cos(torch.pi * x)
    w = initial[1:-1] + 1.0
    E, E2, Q, f1, f2, f3 = etd_coefficients(n_intervals, dt, torch, device)
    saved_t = [0.0]
    saved_u = [initial.detach().cpu().numpy()]
    started = time.perf_counter()
    checkpoint_path = OUT / "allen_cahn" / f"etdrk4-{label}-partial.npz"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def nonlinear_hat(values):
        physical = dst1_orthonormal(values, n_intervals, torch)
        nonlinear = 5.0 * (-physical.pow(3) + 3.0 * physical.square() - 2.0 * physical)
        return dst1_orthonormal(nonlinear, n_intervals, torch)

    v = dst1_orthonormal(w, n_intervals, torch)
    for step in range(1, n_steps + 1):
        Nv = nonlinear_hat(v)
        a = E2 * v + Q * Nv
        Na = nonlinear_hat(a)
        b = E2 * v + Q * Na
        Nb = nonlinear_hat(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nonlinear_hat(c)
        v = E * v + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        if step % steps_per_snapshot == 0:
            interior_u = dst1_orthonormal(v, n_intervals, torch) - 1.0
            full_u = torch.cat((torch.full((1,), -1.0, dtype=torch.float64, device=device),
                                interior_u, torch.full((1,), -1.0, dtype=torch.float64, device=device)))
            saved_t.append(step * dt)
            saved_u.append(full_u.detach().cpu().numpy())
        if step % max(1000, steps_per_snapshot * 10) == 0:
            save_etd_checkpoint(checkpoint_path, n_intervals, dt, step, w=v, saved_t=saved_t, saved_u=saved_u, torch=torch)
            write_status("s1a_progress", "running", resolution=label, step=step, total_steps=n_steps,
                         resolution_elapsed_seconds=round(time.perf_counter() - started, 3))
            if time.perf_counter() >= deadline:
                raise TimeoutError(f"S1a watchdog reached during ETDRK4 {label} at step {step}")
    torch.cuda.synchronize(device)
    field = np.asarray(saved_u, dtype=np.float64)
    target = OUT / "allen_cahn" / f"etdrk4-{label}.npz"
    np.savez_compressed(target, x=x_np, t=np.asarray(saved_t, dtype=np.float64), u=field,
                        n_intervals=n_intervals, dt=dt, method="Fourier-sine pseudospectral ETDRK4 Kassam-Trefethen",
                        boundary="u(-1,t)=u(1,t)=-1", initial="x^2*cos(pi*x)")
    return {"label": label, "n_intervals": n_intervals, "dt": dt, "steps": n_steps,
            "elapsed_seconds": round(time.perf_counter() - started, 3), "path": str(target),
            "sha256": sha256(target), "x": x_np, "t": np.asarray(saved_t), "u": field}


def interp_xt(x_src, t_src, field, x_dst, t_dst):
    import numpy as np
    field = np.asarray(field, dtype=np.float64)
    if field.shape == (len(x_src), len(t_src)):
        field = field.T
    if field.shape != (len(t_src), len(x_src)):
        raise ValueError(f"field shape {field.shape} does not match t/x {(len(t_src), len(x_src))}")
    along_x = np.stack([np.interp(x_dst, x_src, row) for row in field], axis=0)
    return np.stack([np.interp(t_dst, t_src, along_x[:, i]) for i in range(len(x_dst))], axis=1)


def weighted_rel_l2(reference, prediction, x, t):
    import numpy as np
    wx = np.ones(len(x), dtype=np.float64)
    wt = np.ones(len(t), dtype=np.float64)
    wx[[0, -1]] = 0.5
    wt[[0, -1]] = 0.5
    weights = wt[:, None] * wx[None, :]
    dx = (x[-1] - x[0]) / max(1, len(x) - 1)
    dt = (t[-1] - t[0]) / max(1, len(t) - 1)
    den = np.sum(weights * np.square(reference)) * dx * dt
    num = np.sum(weights * np.square(prediction - reference)) * dx * dt
    return float(np.sqrt(num / max(den, np.finfo(np.float64).tiny)))


def summary_stats(values, np):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "n": int(len(values))}


def build_ac_model(arm: str, seed: int, torch):
    from torch import nn
    from torch.nn import functional as F

    class RBFEdge(nn.Module):
        def __init__(self, inputs, outputs, basis_count=11):
            super().__init__()
            self.register_buffer("centers", torch.linspace(-1.05, 1.05, basis_count))
            self.register_buffer("width", torch.tensor(0.21))
            self.coefficients = nn.Parameter(torch.empty(outputs, inputs, basis_count))
            self.base = nn.Parameter(torch.empty(outputs, inputs))
            nn.init.normal_(self.coefficients, std=0.1 / math.sqrt(inputs))
            nn.init.normal_(self.base, std=0.1 / math.sqrt(inputs))
        def forward(self, x):
            basis = torch.exp(-((x.unsqueeze(-1) - self.centers) / self.width).square())
            return torch.einsum("big,oig->bo", basis, self.coefficients) + F.linear(F.silu(x), self.base)

    class Base(nn.Module):
        def normalize(self, coordinates):
            return torch.stack((coordinates[:, 0], 2.0 * coordinates[:, 1] - 1.0), dim=1)
        def constrain(self, coordinates, raw):
            x = coordinates[:, 0:1]
            t = coordinates[:, 1:2]
            initial = x.square() * torch.cos(torch.pi * x)
            return initial + t * (1.0 - x.square()) * raw

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if arm == "AC-RBF-Sobol":
            class Model(Base):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.ModuleList((RBFEdge(2, 16), RBFEdge(16, 16), RBFEdge(16, 1)))
                def forward(self, coordinates):
                    hidden = self.normalize(coordinates)
                    for layer in self.layers[:-1]:
                        hidden = torch.tanh(layer(hidden))
                    return self.constrain(coordinates, self.layers[-1](hidden))
        else:
            class Model(Base):
                def __init__(self):
                    super().__init__()
                    widths = (2, 20, 20, 20, 20, 1)
                    self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))
                    for layer in self.layers[:-1]:
                        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
                        nn.init.zeros_(layer.bias)
                    nn.init.xavier_uniform_(self.layers[-1].weight)
                    nn.init.zeros_(self.layers[-1].bias)
                def forward(self, coordinates):
                    hidden = self.normalize(coordinates)
                    for layer in self.layers[:-1]:
                        hidden = torch.tanh(layer(hidden))
                    return self.constrain(coordinates, self.layers[-1](hidden))
        return Model()


def score_ac_models(reference, device, torch, np) -> dict:
    roots = [Path("/kaggle/input")]
    checkpoint_map = {}
    for arm, folder in (("AC-RBF-Sobol", "AC-RBF-Sobol"), ("AC-DeepXDE-MLP", "AC-DeepXDE-MLP")):
        for seed in (401, 402, 403, 404, 405):
            candidates = [p for p in roots[0].rglob("checkpoint-final.pt")
                          if f"seed-{seed}" in str(p) and folder in str(p) and "score-only" in str(p).lower()]
            if not candidates:
                candidates = [p for p in roots[0].rglob("checkpoint-final.pt")
                              if f"seed-{seed}" in str(p) and folder in str(p)]
            if candidates:
                candidates.sort(key=lambda p: ("score-only" not in str(p).lower(), len(str(p))))
                checkpoint_map[(arm, seed)] = candidates[0]

    x_eval = np.linspace(-1.0, 1.0, 512, dtype=np.float64)
    t_eval = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    target = interp_xt(reference["x"], reference["t"], reference["u"], x_eval, t_eval)
    coords = np.stack(np.meshgrid(x_eval, t_eval, indexing="xy"), axis=-1).reshape(-1, 2)
    coords_t = torch.as_tensor(coords, dtype=torch.float64, device=device)
    rows = []
    for (arm, seed), checkpoint_path in sorted(checkpoint_map.items()):
        model = build_ac_model(arm, seed, torch).to(device=device, dtype=torch.float64)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(coords_t), 8192):
                chunks.append(model(coords_t[start:start + 8192]).detach().cpu().numpy().reshape(-1))
        pred = np.concatenate(chunks).reshape(len(t_eval), len(x_eval))
        row = {"arm": arm, "seed": seed, "checkpoint": str(checkpoint_path),
               "checkpoint_sha256": sha256(checkpoint_path),
               "relative_weighted_space_time_l2": weighted_rel_l2(target, pred, x_eval, t_eval)}
        rows.append(row)
        del model, payload, chunks, pred
        torch.cuda.empty_cache()
    by_arm = {}
    for arm in ("AC-RBF-Sobol", "AC-DeepXDE-MLP"):
        values = [r["relative_weighted_space_time_l2"] for r in rows if r["arm"] == arm]
        by_arm[arm] = summary_stats(values, np) if values else {"mean": None, "sample_sd": None, "n": 0}
    lookup = {(r["arm"], r["seed"]): r["relative_weighted_space_time_l2"] for r in rows}
    paired = [{"seed": seed, "RBF_minus_MLP": lookup[("AC-RBF-Sobol", seed)] - lookup[("AC-DeepXDE-MLP", seed)]}
              for seed in (401, 402, 403, 404, 405)
              if ("AC-RBF-Sobol", seed) in lookup and ("AC-DeepXDE-MLP", seed) in lookup]
    write_json(OUT / "allen_cahn" / "checkpoint_rescoring.json", {
        "status": "complete" if len(rows) == 10 else "partial_missing_checkpoints",
        "reference": "ETDRK4 candidate; human acceptance pending",
        "grid": {"space_points": 512, "time_points": 256},
        "metrics": by_arm, "paired_seed_differences_RBF_minus_MLP": paired,
        "paired_difference_summary": summary_stats([r["RBF_minus_MLP"] for r in paired], np) if paired else None,
        "per_seed": rows,
    })
    csv_path = OUT / "allen_cahn" / "checkpoint_rescoring.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", "seed", "relative_weighted_space_time_l2", "checkpoint_sha256", "checkpoint"])
        writer.writeheader(); writer.writerows(rows)
    return {"metrics": by_arm, "per_seed": rows, "paired": paired}


def run_s1a(device, torch, np, deadline: float) -> dict:
    import scipy.io
    OUT.joinpath("allen_cahn").mkdir(parents=True, exist_ok=True)
    write_status("s1a", "running", method="Fourier-sine pseudospectral ETDRK4 Kassam-Trefethen",
                 reference_accepted=False)
    levels = []
    for label, n, dt in (("N4096_dt2e-5", 4096, 2e-5),
                         ("N8192_dt2e-5", 8192, 2e-5),
                         ("N8192_dt1e-5", 8192, 1e-5)):
        if time.perf_counter() >= deadline - 30.0:
            break
        levels.append(solve_etdrk4(n, dt, label, device, torch, deadline))
        write_status("s1a", "running", completed_resolutions=[r["label"] for r in levels],
                     reference_accepted=False)
    if len(levels) != 3:
        status = {"status": "partial_time_cap", "reference_accepted": False,
                  "completed_resolutions": [{k: v for k, v in r.items() if k not in ("x", "t", "u")} for r in levels]}
        write_status("s1a", status["status"], **{k: v for k, v in status.items() if k != "status"})
        return status

    fine_space = levels[1]["u"][:, ::2]
    coarse_space = levels[0]["u"]
    spatial_rel = weighted_rel_l2(coarse_space, fine_space, levels[0]["x"], levels[0]["t"])
    coarse_time = levels[1]["u"]
    fine_time = levels[2]["u"]
    temporal_rel = weighted_rel_l2(coarse_time, fine_time, levels[1]["x"], levels[1]["t"])
    final = levels[2]
    x_eval = np.linspace(-1.0, 1.0, 512, dtype=np.float64)
    t_eval = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    target_eval = interp_xt(final["x"], final["t"], final["u"], x_eval, t_eval)

    reference_comparisons = {}
    candidate_paths = list(Path("/kaggle/input").rglob("ac_gpu_reference_candidate.npz"))
    if candidate_paths:
        with np.load(candidate_paths[0], allow_pickle=False) as f:
            fd = {k: np.asarray(f[k]) for k in f.files if k in ("x", "t", "u")}
        fd_eval = interp_xt(fd["x"].reshape(-1), fd["t"].reshape(-1), fd["u"], x_eval, t_eval)
        reference_comparisons["FD8192"] = {
            "relative_weighted_space_time_l2_on_512x256": weighted_rel_l2(fd_eval, target_eval, x_eval, t_eval),
            "max_absolute_on_512x256": float(np.max(np.abs(fd_eval - target_eval))),
            "source": str(candidate_paths[0]), "source_sha256": sha256(candidate_paths[0]),
        }
    else:
        reference_comparisons["FD8192"] = {"status": "missing_input_candidate"}

    mat_paths = list(Path("/kaggle/input").rglob("Allen_Cahn.mat"))
    mat_comparison = {"status": "missing_input_mat"}
    if mat_paths:
        mat = scipy.io.loadmat(mat_paths[0])
        xm = np.asarray(mat.get("x")).reshape(-1)
        tm = np.asarray(mat.get("t")).reshape(-1)
        um = mat.get("usol", mat.get("u"))
        if um is None:
            mat_comparison = {"status": "missing_usol_or_u", "keys": sorted(k for k in mat if not k.startswith("__"))}
        else:
            um = np.asarray(um, dtype=np.float64)
            if um.shape == (len(xm), len(tm)):
                um = um.T
            et_on_mat = interp_xt(final["x"], final["t"], final["u"], xm, tm)
            mat_comparison = {
                "status": "scored", "relative_l2": float(np.linalg.norm(et_on_mat - um) / max(np.linalg.norm(um), np.finfo(float).tiny)),
                "max_absolute": float(np.max(np.abs(et_on_mat - um))), "space_points": int(len(xm)), "time_points": int(len(tm)),
                "source": str(mat_paths[0]), "source_sha256": sha256(mat_paths[0]),
            }
    rescoring = score_ac_models({"x": final["x"], "t": final["t"], "u": final["u"]}, device, torch, np)
    result = {
        "status": "candidate_complete", "reference_accepted": False, "scientific_accepted": False,
        "method": "Fourier-sine pseudospectral diagonal Laplacian with float64 Kassam-Trefethen ETDRK4",
        "resolution_runs": [{k: v for k, v in r.items() if k not in ("x", "t", "u")} for r in levels],
        "convergence": {"N4096_vs_N8192_relative_weighted_l2": spatial_rel,
                        "N8192_dt2e-5_vs_dt1e-5_relative_weighted_l2": temporal_rel,
                        "previous_FD_reference_gates_spatial": 5.0e-5, "previous_FD_reference_gates_temporal": 1.0e-7},
        "comparison_to_candidate_reference": reference_comparisons,
        "comparison_to_DeepXDE_mat": mat_comparison,
        "checkpoint_rescoring": rescoring,
        "comparison_grid": {"space_points": 512, "time_points": 256},
    }
    np.savez_compressed(OUT / "allen_cahn" / "etdrk4_reference.npz", x=final["x"], t=final["t"], u=final["u"],
                        n_intervals=8192, dt=1.0e-5, method=result["method"], reference_accepted=False)
    result["artifact_sha256"] = sha256(OUT / "allen_cahn" / "etdrk4_reference.npz")
    write_json(OUT / "allen_cahn" / "s1a_summary.json", result)
    write_status("s1a", "candidate_complete", summary=result, reference_accepted=False)
    return result


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_checkpoint(seed: int, arm: str, roots, np):
    arm_tag = "DFG-RBF-PIKAN" if arm == "rbf" else "DFG-Tanh-MLP"
    if seed <= 305 and arm != "rbf":
        return None
    candidates = []
    for root in roots:
        for path in root.rglob("checkpoint-lbfgs.pt"):
            value = str(path).lower()
            if f"seed-{seed}" not in value:
                continue
            if seed <= 305:
                if "dfg-rbf-refinement" not in value:
                    continue
            elif "dfg-lbfgs-refinement" not in value or arm_tag.lower() not in value:
                continue
            candidates.append(path)
    return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0] if candidates else None


def score_dfg_models(device, torch, np) -> dict:
    reference_path = OUT / "dfg" / "DFG-L2" / "fields_on_common_grid.npz"
    if not reference_path.exists():
        result = {"status": "missing_fem_field", "per_seed": [], "scientific_accepted": False}
        write_json(OUT / "dfg" / "checkpoint_field_scores.json", result)
        return result
    with np.load(reference_path, allow_pickle=False) as field:
        xy = np.asarray(field["xy"], dtype=np.float64)
        target_uv = np.asarray(field["velocity"], dtype=np.float64)
        target_p = np.asarray(field["pressure"], dtype=np.float64).reshape(-1)
    anchor = np.asarray(EVAL_CONTRACT["pressure_gauge"]["outlet_anchor_xy"], dtype=np.float64)
    anchor_index = int(np.argmin(np.linalg.norm(xy - anchor[None, :], axis=1)))
    target_p = target_p - target_p[anchor_index]

    roots = [Path("/kaggle/input")]
    old_sources = [p for p in roots[0].rglob("run_campaign_source.py") if "dfg-lbfgs" in str(p).lower()]
    current_sources = [p for p in roots[0].rglob("dfg.py") if "dfg-rbf-refinement" in str(p).lower()]
    old_module = None
    current_module = None
    source_errors = {}
    try:
        old_module = load_module(sorted(old_sources, key=lambda p: len(str(p)))[0], "closure_dfg_old") if old_sources else None
    except Exception as exc:
        source_errors["older_model_source"] = f"{type(exc).__name__}: {exc}"
    try:
        current_module = load_module(sorted(current_sources, key=lambda p: len(str(p)))[0], "closure_dfg_current") if current_sources else None
    except Exception as exc:
        source_errors["current_model_source"] = f"{type(exc).__name__}: {exc}"
    points = torch.as_tensor(xy, dtype=torch.float32, device=device)
    rows = []
    schedule = [(seed, "rbf") for seed in (301, 302, 303, 304, 305)]
    schedule += [(seed, arm) for seed in (501, 502, 503, 504, 505) for arm in ("rbf", "mlp")]
    write_json(OUT / "dfg" / "checkpoint_field_scores.partial.json", {
        "status": "scoring", "fixed_field_scoring_contract": EVAL_CONTRACT,
        "per_seed": [], "reference_accepted": False, "scientific_accepted": False,
    })
    for schedule_index, (seed, arm) in enumerate(schedule):
        if time.perf_counter() >= STARTED_MONOTONIC + SESSION_CAP - 15.0:
            rows.extend({"seed": later_seed, "arm": later_arm, "status": "session_time_cap_not_started"}
                        for later_seed, later_arm in schedule[schedule_index:])
            break
        checkpoint = find_checkpoint(seed, arm, roots, np)
        if checkpoint is None:
            rows.append({"seed": seed, "arm": arm, "status": "missing_checkpoint", "source_errors": source_errors})
            continue
        try:
            if seed <= 305:
                if current_module is None:
                    raise RuntimeError("current DFG model source is missing")
                model = current_module.DFGRBFPIKAN(seed=seed).to(device)
            else:
                if old_module is None:
                    raise RuntimeError("older DFG model source is missing")
                name = "DFG-RBF-PIKAN" if arm == "rbf" else "DFG-Tanh-MLP"
                model = old_module.build_model(name, seed).to(device)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()
            parts = []
            with torch.no_grad():
                for start in range(0, len(points), 4096):
                    parts.append(model(points[start:start + 4096]).detach().cpu().numpy())
            pred = np.concatenate(parts, axis=0)
            pred_p = pred[:, 2] - pred[anchor_index]
            row = {
                "seed": seed, "arm": arm, "status": "scored", "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "u_relative_l2": float(np.linalg.norm(pred[:, 0] - target_uv[:, 0]) / max(np.linalg.norm(target_uv[:, 0]), 1e-30)),
                "v_relative_l2": float(np.linalg.norm(pred[:, 1] - target_uv[:, 1]) / max(np.linalg.norm(target_uv[:, 1]), 1e-30)),
                "p_relative_l2_gauge_fixed_outlet": float(np.linalg.norm(pred_p - target_p) / max(np.linalg.norm(target_p), 1e-30)),
                "pressure_gauge_anchor": anchor.tolist(),
            }
            rows.append(row)
            write_json(OUT / "dfg" / "checkpoint_field_scores.partial.json", {
                "status": "scoring", "fixed_field_scoring_contract": EVAL_CONTRACT,
                "per_seed": rows, "reference_accepted": False, "scientific_accepted": False,
            })
            del model, payload, parts, pred
            torch.cuda.empty_cache()
        except Exception as exc:
            rows.append({"seed": seed, "arm": arm, "status": "scoring_error",
                         "checkpoint": str(checkpoint), "error": f"{type(exc).__name__}: {exc}"})
            write_json(OUT / "dfg" / "checkpoint_field_scores.partial.json", {
                "status": "scoring", "fixed_field_scoring_contract": EVAL_CONTRACT,
                "per_seed": rows, "reference_accepted": False, "scientific_accepted": False,
            })

    metrics = {}
    for cohort, seeds, arms in (("current_301_305", (301, 302, 303, 304, 305), ("rbf",)),
                                ("older_501_505", (501, 502, 503, 504, 505), ("rbf", "mlp"))):
        metrics[cohort] = {}
        for arm in arms:
            for metric in ("u_relative_l2", "v_relative_l2", "p_relative_l2_gauge_fixed_outlet"):
                values = [r[metric] for r in rows if r.get("seed") in seeds and r.get("arm") == arm and r.get("status") == "scored"]
                metrics[cohort].setdefault(arm, {})[metric] = summary_stats(values, np) if values else {"mean": None, "sample_sd": None, "n": 0}
    lookup = {(r.get("arm"), r.get("seed")): r for r in rows if r.get("status") == "scored" and r.get("seed") in (501, 502, 503, 504, 505)}
    paired = []
    for seed in (501, 502, 503, 504, 505):
        if ("rbf", seed) in lookup and ("mlp", seed) in lookup:
            paired.append({"seed": seed, **{
                metric: lookup[("rbf", seed)][metric] - lookup[("mlp", seed)][metric]
                for metric in ("u_relative_l2", "v_relative_l2", "p_relative_l2_gauge_fixed_outlet")
            }})
    result = {"status": "complete" if sum(r.get("status") == "scored" for r in rows) == 15 else "partial",
              "scientific_accepted": False, "reference_accepted": False,
              "fixed_field_scoring_contract": EVAL_CONTRACT,
              "pressure_gauge": EVAL_CONTRACT["pressure_gauge"],
              "metrics": metrics, "paired_seed_differences_RBF_minus_MLP_501_505": paired,
              "per_seed": rows, "reference_field_sha256": sha256(reference_path)}
    write_json(OUT / "dfg" / "checkpoint_field_scores.json", result)
    score_csv = OUT / "dfg" / "checkpoint_field_scores.csv"
    fields = ["seed", "arm", "status", "u_relative_l2", "v_relative_l2", "p_relative_l2_gauge_fixed_outlet", "checkpoint_sha256", "checkpoint", "error"]
    with score_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return result


def finalize_manifest(status: str, runtime: dict) -> None:
    artifacts = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append({"path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size,
                              "sha256": sha256(path)})
    manifest = {
        "schema_version": 1, "session": "DFG-FEM-field-followup-retry3-2026-09-29", "status": status,
        "kernel_slug": "rodrimandub/af-pikans-dfg-fem-field-followup-retry3-20260929",
        "machine": runtime, "push_cap_seconds": SESSION_CAP,
        "internal_watchdog_seconds": WATCHDOG,
        "fixed_field_scoring_contract": EVAL_CONTRACT,
        "actual_session_seconds": round(time.perf_counter() - STARTED_MONOTONIC, 3),
        "started_at_utc": UTC_STARTED, "ended_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_accepted": False, "reference_accepted": False,
        "kernel_sources": [
            "rodrimandub/af-pikans-dfg-rbf-refinement-20260928",
            "rodrimandub/af-pikans-dfg-lbfgs-refinement-20260927",
        ],
        "kernel_source_files": [
            {"path": name, "sha256": sha256(SCRIPT_DIR / name)}
            for name in ("solve_dfg.py", "make_mesh.geo")
            if (SCRIPT_DIR / name).exists()
        ],
        "parts": {name: str(OUT / f"{name}.json") for name in ("s1a", "s1b")},
        "artifacts": artifacts,
    }
    write_json(OUT / "manifest.json", manifest)


def kernel_main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write_status("s1", "starting", session_cap_seconds=SESSION_CAP, scientific_accepted=False)
    try:
        import numpy as np
        import torch
        if not Path("/kaggle/working").is_dir():
            raise RuntimeError("Kaggle runtime not detected")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("T4x2 required; CUDA or exactly two GPUs not visible")
        devices = [torch.cuda.get_device_name(i) for i in range(2)]
        if any("T4" not in name.upper() for name in devices):
            raise RuntimeError(f"T4x2 required; visible devices={devices}")
        torch.set_num_threads(2)
        torch.cuda.set_device(0)
        runtime = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                   "gpu_names": devices, "visible_gpu_count": torch.cuda.device_count()}
        write_status("s1", "gpu_validated", **runtime)

        worker_log = (OUT / "dfg-worker-supervisor.log").open("w", encoding="utf-8")
        dfg_process = subprocess.Popen([sys.executable, str(SCRIPT_DIR / "closure_s1.py"), "--dfg-worker"],
                                       stdout=worker_log, stderr=subprocess.STDOUT)
        try:
            s1a = run_s1a(torch.device("cuda:0"), torch, np, STARTED_MONOTONIC + min(1320.0, WATCHDOG - 180.0))
        except Exception as exc:
            s1a = {"status": "failed_or_partial", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-5000:]}
            write_status("s1a", s1a["status"], error=s1a["error"], traceback=s1a["traceback"],
                         reference_accepted=False)
        remaining_wait = max(1.0, WATCHDOG - (time.perf_counter() - STARTED_MONOTONIC) - 150.0)
        try:
            dfg_return = dfg_process.wait(timeout=remaining_wait)
        except subprocess.TimeoutExpired:
            dfg_process.terminate()
            try:
                dfg_process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                dfg_process.kill(); dfg_process.wait(timeout=10.0)
            dfg_return = None
            write_status("s1b", "supervisor_time_cap_partial", process_returncode=None,
                         log=str(OUT / "dfg-worker.log"))
        worker_log.close()
        dfg_ref_status = json.loads((OUT / "s1b.json").read_text(encoding="utf-8")) if (OUT / "s1b.json").exists() else {"status": "no_worker_record"}
        dfg_summary_path = OUT / "dfg" / "summary.json"
        dfg_summary = json.loads(dfg_summary_path.read_text(encoding="utf-8")) if dfg_summary_path.exists() else None
        if dfg_summary and dfg_summary.get("completed_level_count", 0) >= 3:
            try:
                dfg_scores = score_dfg_models(torch.device("cuda:1"), torch, np)
            except Exception as exc:
                dfg_scores = {"status": "scoring_error", "error": f"{type(exc).__name__}: {exc}",
                              "traceback": traceback.format_exc()[-5000:], "per_seed": []}
        else:
            dfg_scores = {"status": "field_reference_incomplete", "per_seed": []}
        write_json(OUT / "dfg" / "checkpoint_field_scores.json", dfg_scores)
        s1b_final = {"worker": dfg_ref_status, "fem_summary": dfg_summary, "checkpoint_field_scores": dfg_scores,
                     "process_returncode": dfg_return, "reference_accepted": False, "scientific_accepted": False}
        write_json(OUT / "s1b_final.json", s1b_final)
        status = "candidate_complete" if s1a.get("status") == "candidate_complete" and dfg_summary and dfg_summary.get("completed_level_count", 0) >= 3 else "partial"
        write_status("s1", status, s1a_status=s1a.get("status"), s1b_status=dfg_summary.get("status") if dfg_summary else dfg_ref_status.get("status"),
                     actual_session_seconds=round(time.perf_counter() - STARTED_MONOTONIC, 3), scientific_accepted=False)
        finalize_manifest(status, runtime)
        return 0
    except Exception as exc:
        write_status("s1", "failed", exception=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-5000:],
                     actual_session_seconds=round(time.perf_counter() - STARTED_MONOTONIC, 3), scientific_accepted=False)
        finalize_manifest("failed", {"gpu_validation": "failed_or_not_reached"})
        return 0


def s1b_retry_main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "manifest.json", {
        "schema_version": 1, "session": "DFG-FEM-field-followup-retry3-2026-09-29",
        "kernel_slug": "rodrimandub/af-pikans-dfg-fem-field-followup-retry3-20260929", "status": "starting",
        "push_cap_seconds": SESSION_CAP, "fixed_field_scoring_contract": EVAL_CONTRACT,
        "kernel_sources": ['rodrimandub/af-pikans-dfg-rbf-refinement-20260928', 'rodrimandub/af-pikans-dfg-lbfgs-refinement-20260927'],
        "reference_accepted": False, "scientific_accepted": False,
    })
    write_status("s1b_retry", "starting", session_cap_seconds=SESSION_CAP,
                 worker_cap_seconds=DFG_CAP, scientific_accepted=False)
    try:
        import numpy as np
        import torch
        if not Path("/kaggle/working").is_dir():
            raise RuntimeError("Kaggle runtime not detected")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("T4x2 required; CUDA or exactly two GPUs not visible")
        devices = [torch.cuda.get_device_name(i) for i in range(2)]
        if any("T4" not in name.upper() for name in devices):
            raise RuntimeError(f"T4x2 required; visible devices={devices}")
        torch.set_num_threads(2)
        torch.cuda.set_device(1)
        runtime = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                   "gpu_names": devices, "visible_gpu_count": torch.cuda.device_count()}
        materialize_dfg_assets()
        dfg_return = dfg_worker()
        summary_path = OUT / "dfg" / "summary.json"
        dfg_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
        if dfg_summary and dfg_summary.get("completed_level_count", 0) >= 3:
            try:
                dfg_scores = score_dfg_models(torch.device("cuda:1"), torch, np)
            except Exception as exc:
                dfg_scores = {"status": "scoring_error", "error": f"{type(exc).__name__}: {exc}",
                              "traceback": traceback.format_exc()[-5000:], "per_seed": []}
        else:
            dfg_scores = {"status": "field_reference_incomplete", "per_seed": []}
        write_json(OUT / "dfg" / "checkpoint_field_scores.json", dfg_scores)
        dfg_ref_status = json.loads((OUT / "s1b.json").read_text(encoding="utf-8")) if (OUT / "s1b.json").exists() else {"status": "no_worker_record"}
        write_json(OUT / "s1b_final.json", {"worker": dfg_ref_status, "fem_summary": dfg_summary,
                   "checkpoint_field_scores": dfg_scores, "process_returncode": dfg_return,
                   "reference_accepted": False, "scientific_accepted": False})
        status = "candidate_complete" if dfg_summary and dfg_summary.get("completed_level_count", 0) >= 3 else "partial"
        write_status("s1b_retry", status, s1b_status=dfg_ref_status.get("status"),
                     actual_session_seconds=round(time.perf_counter() - STARTED_MONOTONIC, 3),
                     scientific_accepted=False)
        finalize_manifest(status, runtime)
        return 0
    except Exception as exc:
        write_status("s1b_retry", "failed", exception=f"{type(exc).__name__}: {exc}",
                     traceback=traceback.format_exc()[-5000:],
                     actual_session_seconds=round(time.perf_counter() - STARTED_MONOTONIC, 3),
                     scientific_accepted=False)
        finalize_manifest("failed", {"gpu_validation": "failed_or_not_reached"})
        return 0


if __name__ == "__main__":
    raise SystemExit(s1b_retry_main())
