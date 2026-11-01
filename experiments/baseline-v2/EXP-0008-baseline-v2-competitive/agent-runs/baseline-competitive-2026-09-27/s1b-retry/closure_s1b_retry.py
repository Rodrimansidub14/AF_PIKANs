#!/usr/bin/env python3
"""S1: independent Allen-Cahn reference plus DFG Taylor-Hood reference on Kaggle T4x2.

This script is a Kaggle kernel entrypoint. It never falls back to local/CPU-only
training. The DFG worker runs in a separate micromamba environment so the
PyTorch CUDA process remains isolated from PETSc/MPI libraries.
"""
from __future__ import annotations

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
SCRIPT_DIR = Path("/kaggle/working/closure-s1b-source")
OUT = Path("/kaggle/working/closure-s1b-retry")
SESSION_CAP = 1800.0
WATCHDOG = 1680.0
DFG_CAP = 1450.0
FEATFLOW = {"Cd": 5.57953523384, "Cl": 0.010618948146, "delta_p": 0.11752016697}

_EMBEDDED_DFG_ASSETS = {
    "solve_dfg.py": "IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiJEUkFGVF9VTlZFUklGSUVEIHN0YXRpb25hcnkgVGF5bG9y4oCTSG9vZCBzb2x2ZXIgZm9yIERGRyAyRC0xIChSZT0yMCkuCgpUaGlzIHNvdXJjZSBpcyBhIHByb3RvY29sLWFsaWduZWQgZHJhZnQuIEl0IGhhcyBub3QgYmVlbiBydW4gYWdhaW5zdCBhIHJlc29sdmVkCkRPTEZJTngvR21zaC9QRVRTYyBlbnZpcm9ubWVudCBhbmQgY2Fubm90IHByb2R1Y2UgYW4gYWNjZXB0ZWQgcmVmZXJlbmNlIHVudGlsCnRoZSB2YWxpZGF0aW9uIGdhcHMgaW4gUkVBRE1FLm1kIGFyZSBjbG9zZWQuCiIiIgpmcm9tIF9fZnV0dXJlX18gaW1wb3J0IGFubm90YXRpb25zCgppbXBvcnQgYXJncGFyc2UKaW1wb3J0IGpzb24KaW1wb3J0IHN1YnByb2Nlc3MKaW1wb3J0IHN5cwppbXBvcnQgdGltZQpmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgKCmltcG9ydCBudW1weSBhcyBucApmcm9tIGJhc2l4LnVmbCBpbXBvcnQgZWxlbWVudCwgbWl4ZWRfZWxlbWVudApmcm9tIGRvbGZpbnggaW1wb3J0IGZlbSwgZ2VvbWV0cnksIGlvCmZyb20gZG9sZmlueC5pbyBpbXBvcnQgZ21zaCBhcyBnbXNoaW8KZnJvbSBtcGk0cHkgaW1wb3J0IE1QSQpmcm9tIHBldHNjNHB5IGltcG9ydCBQRVRTYwppbXBvcnQgdWZsCgpTVEFUVVMgPSAiRFJBRlRfVU5WRVJJRklFRCIKUk9PVCA9IFBhdGgoX19maWxlX18pLnJlc29sdmUoKS5wYXJlbnQKR0VPID0gUk9PVCAvICJtYWtlX21lc2guZ2VvIgpMLCBIID0gMi4yLCAwLjQxCkNYLCBDWSwgUiA9IDAuMiwgMC4yLCAwLjA1ClJITywgTlUgPSAxLjAsIDAuMDAxClVNQVgsIFVNRUFOLCBESUFNRVRFUiA9IDAuMywgMC4yLCAwLjEKVEFHUyA9IHsiaW5sZXQiOiAxLCAid2FsbHMiOiAyLCAiY3lsaW5kZXIiOiAzLCAib3V0bGV0IjogNCwgImZsdWlkIjogNX0KTEVWRUxTID0gWwogICAgeyJpZCI6ICJERkctTDAiLCAiYnVsayI6IDAuMDQwLCAiY3lsaW5kZXIiOiAwLjAwNjB9LAogICAgeyJpZCI6ICJERkctTDEiLCAiYnVsayI6IDAuMDIwLCAiY3lsaW5kZXIiOiAwLjAwMzB9LAogICAgeyJpZCI6ICJERkctTDIiLCAiYnVsayI6IDAuMDEwLCAiY3lsaW5kZXIiOiAwLjAwMTV9LAogICAgeyJpZCI6ICJERkctTDMiLCAiYnVsayI6IDAuMDA1LCAiY3lsaW5kZXIiOiAwLjAwMDc1fSwKICAgIHsiaWQiOiAiREZHLUw0IiwgImJ1bGsiOiAwLjAwMjUsICJjeWxpbmRlciI6IDAuMDAwMzc1fSwKICAgIHsiaWQiOiAiREZHLUw1IiwgImJ1bGsiOiAwLjAwMTI1LCAiY3lsaW5kZXIiOiAwLjAwMDE4NzV9LApdCkZFQVRGTE9XID0geyJDZCI6IDUuNTc5NTM1MjMzODQsICJDbCI6IDAuMDEwNjE4OTQ4MTQ2LCAiZGVsdGFfcCI6IDAuMTE3NTIwMTY2OTd9CgoKZGVmIG1ha2VfbWVzaChsZXZlbDogZGljdCwgbWVzaF9kaXI6IFBhdGgsIGdtc2hfYmluOiBzdHIpIC0+IFBhdGg6CiAgICAiIiJHZW5lcmF0ZSBvbmUgY3VydmVkLCBxdWFkcmF0aWMgR21zaCBtZXNoIGZyb20gdGhlIHNoYXJlZCBDQUQgc291cmNlLiIiIgogICAgbWVzaF9kaXIubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1UcnVlKQogICAgbXNoID0gbWVzaF9kaXIgLyBmImRmZ19yZTIwX3tsZXZlbFsnaWQnXX0ubXNoIgogICAgY29tbWFuZCA9IFsKICAgICAgICBnbXNoX2Jpbiwgc3RyKEdFTyksICItMiIsCiAgICAgICAgIi1zZXRudW1iZXIiLCAibGNfYnVsayIsIHN0cihsZXZlbFsiYnVsayJdKSwKICAgICAgICAiLXNldG51bWJlciIsICJsY19jeWwiLCBzdHIobGV2ZWxbImN5bGluZGVyIl0pLAogICAgICAgICItbyIsIHN0cihtc2gpLAogICAgXQogICAgc3VicHJvY2Vzcy5ydW4oY29tbWFuZCwgY2hlY2s9VHJ1ZSkKICAgIHJldHVybiBtc2gKCgpkZWYgdmVsb2NpdHlfYm91bmRhcnlfY29uZGl0aW9ucyhkb21haW4sIGZhY2V0X3RhZ3MsIFcpOgogICAgIiIiSW1wb3NlIHRoZSBGRUFURkxPVyBwYXJhYm9saWMgaW5mbG93IGFuZCB6ZXJvIHZlbG9jaXR5IG9uIHNvbGlkIHdhbGxzLiIiIgogICAgdGRpbSA9IGRvbWFpbi50b3BvbG9neS5kaW0KICAgIGZkaW0gPSB0ZGltIC0gMQogICAgZG9tYWluLnRvcG9sb2d5LmNyZWF0ZV9jb25uZWN0aXZpdHkoZmRpbSwgdGRpbSkKICAgIFYsIF8gPSBXLnN1YigwKS5jb2xsYXBzZSgpCgogICAgaW5sZXQgPSBmZW0uRnVuY3Rpb24oViwgbmFtZT0iaW5sZXRfdmVsb2NpdHkiKQoKICAgIGRlZiBwcm9maWxlKHgpOgogICAgICAgIHZhbHVlcyA9IG5wLnplcm9zKCgyLCB4LnNoYXBlWzFdKSwgZHR5cGU9ZG9tYWluLmdlb21ldHJ5LnguZHR5cGUpCiAgICAgICAgeSA9IHhbMV0KICAgICAgICB2YWx1ZXNbMF0gPSA0LjAgKiBVTUFYICogeSAqIChIIC0geSkgLyAoSCAqIEgpCiAgICAgICAgcmV0dXJuIHZhbHVlcwoKICAgIGlubGV0LmludGVycG9sYXRlKHByb2ZpbGUpCiAgICBpbmxldC54LnNjYXR0ZXJfZm9yd2FyZCgpCiAgICB6ZXJvID0gZmVtLkZ1bmN0aW9uKFYsIG5hbWU9Inplcm9fdmVsb2NpdHkiKQogICAgemVyby54LmFycmF5WzpdID0gMC4wCiAgICB6ZXJvLnguc2NhdHRlcl9mb3J3YXJkKCkKCiAgICBkZWYgZG9mcyh0YWcpOgogICAgICAgIHJldHVybiBmZW0ubG9jYXRlX2RvZnNfdG9wb2xvZ2ljYWwoCiAgICAgICAgICAgIChXLnN1YigwKSwgViksIGZkaW0sIGZhY2V0X3RhZ3MuZmluZChUQUdTW3RhZ10pCiAgICAgICAgKQoKICAgIHJldHVybiBbCiAgICAgICAgZmVtLmRpcmljaGxldGJjKGlubGV0LCBkb2ZzKCJpbmxldCIpLCBXLnN1YigwKSksCiAgICAgICAgZmVtLmRpcmljaGxldGJjKHplcm8sIGRvZnMoIndhbGxzIiksIFcuc3ViKDApKSwKICAgICAgICBmZW0uZGlyaWNobGV0YmMoemVybywgZG9mcygiY3lsaW5kZXIiKSwgVy5zdWIoMCkpLAogICAgXQoKCmRlZiBzb2x2ZV9zdGF0aW9uYXJ5KGRvbWFpbiwgZmFjZXRfdGFncyk6CiAgICAiIiJTb2x2ZSBTdG9rZXMgZmlyc3QsIHRoZW4gY29udGludWUgdGhlIGNvbnZlY3RpdmUgY29lZmZpY2llbnQgdG8gb25lLiIiIgogICAgUDIgPSBlbGVtZW50KCJMYWdyYW5nZSIsIGRvbWFpbi5iYXNpeF9jZWxsKCksIDIsIHNoYXBlPSgyLCkpCiAgICBQMSA9IGVsZW1lbnQoIkxhZ3JhbmdlIiwgZG9tYWluLmJhc2l4X2NlbGwoKSwgMSkKICAgIFcgPSBmZW0uZnVuY3Rpb25zcGFjZShkb21haW4sIG1peGVkX2VsZW1lbnQoW1AyLCBQMV0pKQogICAgc3RhdGUgPSBmZW0uRnVuY3Rpb24oVywgbmFtZT0ibWl4ZWRfdmVsb2NpdHlfcHJlc3N1cmUiKQogICAgdGVzdHMgPSB1ZmwuVGVzdEZ1bmN0aW9ucyhXKQogICAgdmVsb2NpdHksIHByZXNzdXJlID0gdWZsLnNwbGl0KHN0YXRlKQogICAgdGVzdF92ZWxvY2l0eSwgdGVzdF9wcmVzc3VyZSA9IHRlc3RzCiAgICBhbHBoYSA9IGZlbS5Db25zdGFudChkb21haW4sIFBFVFNjLlNjYWxhclR5cGUoMC4wKSkKICAgIHZpc2Nvc2l0eSA9IGZlbS5Db25zdGFudChkb21haW4sIFBFVFNjLlNjYWxhclR5cGUoTlUpKQogICAgZHggPSB1ZmwuTWVhc3VyZSgiZHgiLCBkb21haW49ZG9tYWluKQoKICAgICMgVGhpcyBpcyB0aGUgd2VhayBmb3JtIGZpeGVkIGluIHJlZmVyZW5jZS1wcm90b2NvbC1hYy1kZmcubWQuIFRoZXJlIGlzIG5vCiAgICAjIG91dGxldCBib3VuZGFyeSB0ZXJtIG9yIG91dGxldCBEaXJpY2hsZXQgY29uZGl0aW9uOiB0aGUgbmF0dXJhbCBkby1ub3RoaW5nCiAgICAjIGNvbmRpdGlvbiBpcyB0aGUgZnJlZSB0cmFjdGlvbiBvbiB0aGUgb3BlbiBkb3duc3RyZWFtIGJvdW5kYXJ5LgogICAgcmVzaWR1YWwgPSAoCiAgICAgICAgUkhPICogYWxwaGEgKiB1ZmwuaW5uZXIodWZsLmRvdCh1ZmwuZ3JhZCh2ZWxvY2l0eSksIHZlbG9jaXR5KSwgdGVzdF92ZWxvY2l0eSkKICAgICAgICArIHZpc2Nvc2l0eSAqIHVmbC5pbm5lcih1ZmwuZ3JhZCh2ZWxvY2l0eSksIHVmbC5ncmFkKHRlc3RfdmVsb2NpdHkpKQogICAgICAgIC0gcHJlc3N1cmUgKiB1ZmwuZGl2KHRlc3RfdmVsb2NpdHkpCiAgICAgICAgKyB0ZXN0X3ByZXNzdXJlICogdWZsLmRpdih2ZWxvY2l0eSkKICAgICkgKiBkeAogICAgYmNzID0gdmVsb2NpdHlfYm91bmRhcnlfY29uZGl0aW9ucyhkb21haW4sIGZhY2V0X3RhZ3MsIFcpCiAgICBvcHRpb25zID0gewogICAgICAgICJzbmVzX3R5cGUiOiAibmV3dG9ubHMiLAogICAgICAgICJzbmVzX2xpbmVzZWFyY2hfdHlwZSI6ICJidCIsCiAgICAgICAgInNuZXNfcnRvbCI6IDEuMGUtMTAsCiAgICAgICAgInNuZXNfYXRvbCI6IDEuMGUtMTIsCiAgICAgICAgInNuZXNfc3RvbCI6IDEuMGUtMTIsCiAgICAgICAgInNuZXNfbWF4X2l0IjogODAsCiAgICAgICAgInNuZXNfZXJyb3JfaWZfbm90X2NvbnZlcmdlZCI6IFRydWUsCiAgICAgICAgImtzcF90eXBlIjogInByZW9ubHkiLAogICAgICAgICJwY190eXBlIjogImx1IiwKICAgICAgICAicGNfZmFjdG9yX21hdF9zb2x2ZXJfdHlwZSI6ICJtdW1wcyIsCiAgICAgICAgImtzcF9lcnJvcl9pZl9ub3RfY29udmVyZ2VkIjogVHJ1ZSwKICAgIH0KICAgIHByb2JsZW0gPSBmZW0ucGV0c2MuTm9ubGluZWFyUHJvYmxlbSgKICAgICAgICByZXNpZHVhbCwKICAgICAgICBzdGF0ZSwKICAgICAgICBiY3M9YmNzLAogICAgICAgIHBldHNjX29wdGlvbnNfcHJlZml4PSJkZmdfcmUyMF8iLAogICAgICAgIHBldHNjX29wdGlvbnM9b3B0aW9ucywKICAgICkKCiAgICAjIGFscGhhPTAgaXMgdGhlIFN0b2tlcyBpbml0aWFsaXNhdGlvbi4gRmFpbGVkIGNvbnRpbnVhdGlvbiBzdGVwcyBhcmUgbG9nZ2VkCiAgICAjIGFuZCByZXRyaWVkIGF0IGhhbGYgc3RlcDsgZG8gbm90IHNpbGVudGx5IGFkdmFuY2UgcGFzdCBhIGZhaWxlZCBzdGFnZS4KICAgIHN0YWdlX2xvZyA9IFtdCiAgICBsYXN0X2FscGhhID0gMC4wCiAgICBzdGVwID0gMC4yNQogICAgc3RhdGUueC5hcnJheVs6XSA9IDAuMAogICAgc3RhdGUueC5zY2F0dGVyX2ZvcndhcmQoKQoKICAgIGRlZiBhdHRlbXB0KHRhcmdldDogZmxvYXQpIC0+IGJvb2w6CiAgICAgICAgbm9ubG9jYWwgc3RhdGUKICAgICAgICBwcmV2aW91cyA9IHN0YXRlLnguYXJyYXkuY29weSgpCiAgICAgICAgYWxwaGEudmFsdWUgPSBQRVRTYy5TY2FsYXJUeXBlKHRhcmdldCkKICAgICAgICB0cnk6CiAgICAgICAgICAgIHByb2JsZW0uc29sdmUoKQogICAgICAgICAgICBzdGF0ZS54LnNjYXR0ZXJfZm9yd2FyZCgpCiAgICAgICAgICAgIHJlYXNvbiA9IGludChwcm9ibGVtLnNvbHZlci5nZXRDb252ZXJnZWRSZWFzb24oKSkKICAgICAgICAgICAgc25lc19pdGVyYXRpb25zID0gaW50KHByb2JsZW0uc29sdmVyLmdldEl0ZXJhdGlvbk51bWJlcigpKQogICAgICAgICAgICBub25saW5lYXJfbm9ybSA9IGZsb2F0KHByb2JsZW0uc29sdmVyLmdldEZ1bmN0aW9uTm9ybSgpKQogICAgICAgICAgICBrc3AgPSBwcm9ibGVtLnNvbHZlci5nZXRLU1AoKQogICAgICAgICAgICBsaW5lYXJfcmVhc29uID0gaW50KGtzcC5nZXRDb252ZXJnZWRSZWFzb24oKSkKICAgICAgICAgICAgbGluZWFyX25vcm0gPSBmbG9hdChrc3AuZ2V0UmVzaWR1YWxOb3JtKCkpCiAgICAgICAgICAgIHN1Y2Nlc3MgPSByZWFzb24gPiAwIGFuZCBucC5pc2Zpbml0ZShub25saW5lYXJfbm9ybSkgYW5kIGxpbmVhcl9yZWFzb24gPiAwCiAgICAgICAgICAgIHN0YWdlX2xvZy5hcHBlbmQoewogICAgICAgICAgICAgICAgImFscGhhIjogdGFyZ2V0LAogICAgICAgICAgICAgICAgImFjY2VwdGVkX2J5X3NvbHZlciI6IHN1Y2Nlc3MsCiAgICAgICAgICAgICAgICAic25lc19yZWFzb24iOiByZWFzb24sCiAgICAgICAgICAgICAgICAic25lc19pdGVyYXRpb25zIjogc25lc19pdGVyYXRpb25zLAogICAgICAgICAgICAgICAgInNuZXNfZmluYWxfbm9ybSI6IG5vbmxpbmVhcl9ub3JtLAogICAgICAgICAgICAgICAgImtzcF9yZWFzb24iOiBsaW5lYXJfcmVhc29uLAogICAgICAgICAgICAgICAgImtzcF9maW5hbF9ub3JtIjogbGluZWFyX25vcm0sCiAgICAgICAgICAgIH0pCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBleGM6ICAjIHJldGFpbiB0aGUgZmFpbGVkIHN0YWdlIGluIHRoZSByZWNvcmQKICAgICAgICAgICAgc3RhdGUueC5hcnJheVs6XSA9IHByZXZpb3VzCiAgICAgICAgICAgIHN0YXRlLnguc2NhdHRlcl9mb3J3YXJkKCkKICAgICAgICAgICAgc3RhZ2VfbG9nLmFwcGVuZCh7ImFscGhhIjogdGFyZ2V0LCAiYWNjZXB0ZWRfYnlfc29sdmVyIjogRmFsc2UsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICJleGNlcHRpb24iOiBmInt0eXBlKGV4YykuX19uYW1lX199OiB7ZXhjfSJ9KQogICAgICAgICAgICByZXR1cm4gRmFsc2UKICAgICAgICBpZiBub3Qgc3VjY2VzczoKICAgICAgICAgICAgc3RhdGUueC5hcnJheVs6XSA9IHByZXZpb3VzCiAgICAgICAgICAgIHN0YXRlLnguc2NhdHRlcl9mb3J3YXJkKCkKICAgICAgICByZXR1cm4gc3VjY2VzcwoKICAgIGlmIG5vdCBhdHRlbXB0KDAuMCk6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCJTdG9rZXMgaW5pdGlhbGlzYXRpb24gZmFpbGVkOyBzZWUgc3RhZ2UgbG9nIGluIHRoZSBleGNlcHRpb24gY29udGV4dCIpCiAgICByZWFjaGVkID0gMC4wCiAgICB3aGlsZSByZWFjaGVkIDwgMS4wIC0gMS4wZS0xNDoKICAgICAgICB0YXJnZXQgPSBtaW4oMS4wLCByZWFjaGVkICsgc3RlcCkKICAgICAgICBpZiBhdHRlbXB0KHRhcmdldCk6CiAgICAgICAgICAgIHJlYWNoZWQgPSB0YXJnZXQKICAgICAgICAgICAgc3RlcCA9IG1pbigwLjI1LCBzdGVwICogMi4wKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHN0ZXAgKj0gMC41CiAgICAgICAgICAgIGlmIHN0ZXAgPCAwLjAxNTYyNToKICAgICAgICAgICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcihmIkNvbnRpbnVhdGlvbiBzdGFsbGVkIGF0IGFscGhhPXtyZWFjaGVkfTsgc3RhZ2VzPXtzdGFnZV9sb2d9IikKCiAgICBzdGF0ZS54LnNjYXR0ZXJfZm9yd2FyZCgpCiAgICByZXR1cm4gVywgc3RhdGUsIHN0YWdlX2xvZwoKCmRlZiBwb2ludF92YWx1ZXMoZnVuY3Rpb24sIGRvbWFpbiwgeHlfcG9pbnRzKToKICAgICIiIkxvY2F0ZSBhbmQgZXZhbHVhdGUgYSBmdW5jdGlvbiBhdCBmaXhlZCBwaHlzaWNhbCBwb2ludHMgKHNlcmlhbCBkcmFmdCkuIiIiCiAgICBwb2ludHMgPSBucC56ZXJvcygobGVuKHh5X3BvaW50cyksIDMpLCBkdHlwZT1kb21haW4uZ2VvbWV0cnkueC5kdHlwZSkKICAgIHBvaW50c1s6LCA6Ml0gPSB4eV9wb2ludHMKICAgIHRyZWUgPSBnZW9tZXRyeS5iYl90cmVlKGRvbWFpbiwgZG9tYWluLnRvcG9sb2d5LmRpbSkKICAgIGNhbmRpZGF0ZXMgPSBnZW9tZXRyeS5jb21wdXRlX2NvbGxpc2lvbnNfcG9pbnRzKHRyZWUsIHBvaW50cykKICAgIGNvbGxpZGluZyA9IGdlb21ldHJ5LmNvbXB1dGVfY29sbGlkaW5nX2NlbGxzKGRvbWFpbiwgY2FuZGlkYXRlcywgcG9pbnRzKQogICAgdmFsdWVzID0gW10KICAgIGZvciBpbmRleCBpbiByYW5nZShsZW4ocG9pbnRzKSk6CiAgICAgICAgY2VsbHMgPSBjb2xsaWRpbmcubGlua3MoaW5kZXgpCiAgICAgICAgaWYgbGVuKGNlbGxzKSA9PSAwOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYiUG9pbnQgbm90IGZvdW5kIGluIGZsdWlkIG1lc2g6IHt4eV9wb2ludHNbaW5kZXhdLnRvbGlzdCgpfSIpCiAgICAgICAgY2VsbCA9IG5wLmFzYXJyYXkoW2NlbGxzWzBdXSwgZHR5cGU9bnAuaW50MzIpCiAgICAgICAgdmFsdWVzLmFwcGVuZChmdW5jdGlvbi5ldmFsKHBvaW50c1tpbmRleDppbmRleCArIDFdLCBjZWxsKVswXSkKICAgIHJldHVybiBucC5hc2FycmF5KHZhbHVlcykKCgpkZWYgZXhwb3J0X2xldmVsKGxldmVsLCBtZXNoX3BhdGg6IFBhdGgsIG91dHB1dF9kaXI6IFBhdGgpOgogICAgIiIiU29sdmUgb25lIGxldmVsIGFuZCBleHBvcnQgdGhlIG1lc2gsIEZFIGNvZWZmaWNpZW50cywgc2FtcGxlZCBmaWVsZHMgYW5kIFFvSXMuIiIiCiAgICBjb21tID0gTVBJLkNPTU1fV09STEQKICAgIGlmIGNvbW0uc2l6ZSAhPSAxOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigiVGhpcyBkcmFmdCBpbnRlbnRpb25hbGx5IHN1cHBvcnRzIHNlcmlhbCBydW5zIG9ubHkuIikKICAgIG1lc2hfZGF0YSA9IGdtc2hpby5yZWFkX2Zyb21fbXNoKG1lc2hfcGF0aCwgY29tbSwgcmFuaz0wLCBnZGltPTIpCiAgICBkb21haW4gPSBtZXNoX2RhdGEubWVzaAogICAgZmFjZXRfdGFncyA9IG1lc2hfZGF0YS5mYWNldF90YWdzCiAgICBpZiBmYWNldF90YWdzIGlzIE5vbmU6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCJHbXNoIG1lc2ggaGFzIG5vIGJvdW5kYXJ5IFBoeXNpY2FsIEN1cnZlIHRhZ3MuIikKICAgIFcsIHN0YXRlLCBzdGFnZXMgPSBzb2x2ZV9zdGF0aW9uYXJ5KGRvbWFpbiwgZmFjZXRfdGFncykKICAgIHZlbG9jaXR5LCBfID0gc3RhdGUuc3ViKDApLmNvbGxhcHNlKCkKICAgIHByZXNzdXJlLCBfID0gc3RhdGUuc3ViKDEpLmNvbGxhcHNlKCkKICAgIHZlbG9jaXR5Lm5hbWUgPSAidmVsb2NpdHkiCiAgICBwcmVzc3VyZS5uYW1lID0gInByZXNzdXJlIgoKICAgICMgRkVBVEZMT1cgZGVmaW5lcyBzaWdtYSA9IG51KmdyYWQodSkgLSBwIEksIHdpdGggZXRhIGRpcmVjdGVkIG91dCBvZiB0aGUKICAgICMgY3lsaW5kZXIuIFRoZSBtZXNoJ3MgZXh0ZXJpb3Igbm9ybWFsIHBvaW50cyBpbnRvIHRoZSBjeWxpbmRlciwgc28gZXRhPS1uLgogICAgXywgcF9zeW1ib2wgPSB1Zmwuc3BsaXQoc3RhdGUpCiAgICB1X3N5bWJvbCwgXyA9IHVmbC5zcGxpdChzdGF0ZSkKICAgIG5vcm1hbF9jeWxpbmRlciA9IC11ZmwuRmFjZXROb3JtYWwoZG9tYWluKQogICAgc3RyZXNzID0gdmlzY29zaXR5X2Zvcl9xb2koZG9tYWluKSAqIHVmbC5ncmFkKHVfc3ltYm9sKSAtIHBfc3ltYm9sICogdWZsLklkZW50aXR5KDIpCiAgICBkcyA9IHVmbC5NZWFzdXJlKCJkcyIsIGRvbWFpbj1kb21haW4sIHN1YmRvbWFpbl9kYXRhPWZhY2V0X3RhZ3MpCiAgICB0cmFjdGlvbiA9IHVmbC5kb3Qoc3RyZXNzLCBub3JtYWxfY3lsaW5kZXIpCiAgICBsb2NhbF9kcmFnID0gZmVtLmFzc2VtYmxlX3NjYWxhcihmZW0uZm9ybSh0cmFjdGlvblswXSAqIGRzKFRBR1NbImN5bGluZGVyIl0pKSkKICAgIGxvY2FsX2xpZnQgPSBmZW0uYXNzZW1ibGVfc2NhbGFyKGZlbS5mb3JtKHRyYWN0aW9uWzFdICogZHMoVEFHU1siY3lsaW5kZXIiXSkpKQogICAgZHJhZ19mb3JjZSA9IGNvbW0uYWxscmVkdWNlKGZsb2F0KGxvY2FsX2RyYWcpLCBvcD1NUEkuU1VNKQogICAgbGlmdF9mb3JjZSA9IGNvbW0uYWxscmVkdWNlKGZsb2F0KGxvY2FsX2xpZnQpLCBvcD1NUEkuU1VNKQogICAgc2NhbGUgPSAyLjAgLyAoUkhPICogVU1FQU4gKiBVTUVBTiAqIERJQU1FVEVSKQoKICAgICMgVGhlIHR3byBwcmVzY3JpYmVkIHByZXNzdXJlIHByb2JlcyBsaWUgb24gdGhlIGN5bGluZGVyIGJvdW5kYXJ5LiBUaGUKICAgICMgcG9pbnQgbG9jYXRvciBpcyBkZWxpYmVyYXRlbHkgZXhwb3NlZCBhcyBhbiB1bnZlcmlmaWVkIG9uZS1zaWRlZCB0cmFjZS4KICAgIHByb2JlcyA9IG5wLmFzYXJyYXkoW1swLjE1LCAwLjJdLCBbMC4yNSwgMC4yXV0sIGR0eXBlPW5wLmZsb2F0NjQpCiAgICBwcm9iZV9wcmVzc3VyZSA9IHBvaW50X3ZhbHVlcyhwcmVzc3VyZSwgZG9tYWluLCBwcm9iZXMpLnJlc2hhcGUoLTEpCiAgICBkZWx0YV9wID0gZmxvYXQocHJvYmVfcHJlc3N1cmVbMF0gLSBwcm9iZV9wcmVzc3VyZVsxXSkKCiAgICB4ZyA9IG5wLmxpbnNwYWNlKDAuMCwgTCwgMjIxKQogICAgeWcgPSBucC5saW5zcGFjZSgwLjAsIEgsIDQyKQogICAgeHgsIHl5ID0gbnAubWVzaGdyaWQoeGcsIHlnLCBpbmRleGluZz0ieHkiKQogICAgZ3JpZF94eSA9IG5wLmNvbHVtbl9zdGFjaygoeHgucmF2ZWwoKSwgeXkucmF2ZWwoKSkpCiAgICBmbHVpZF9tYXNrID0gKChncmlkX3h5WzosIDBdIC0gQ1gpICoqIDIgKyAoZ3JpZF94eVs6LCAxXSAtIENZKSAqKiAyKSA+IFIgKiogMgogICAgZ3JpZF94eSA9IGdyaWRfeHlbZmx1aWRfbWFza10KICAgIHZlbG9jaXR5X2dyaWQgPSBwb2ludF92YWx1ZXModmVsb2NpdHksIGRvbWFpbiwgZ3JpZF94eSkKICAgIHByZXNzdXJlX2dyaWQgPSBwb2ludF92YWx1ZXMocHJlc3N1cmUsIGRvbWFpbiwgZ3JpZF94eSkucmVzaGFwZSgtMSkKCiAgICBvdXRwdXRfZGlyLm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rfb2s9VHJ1ZSkKICAgIG5wLnNhdmV6X2NvbXByZXNzZWQoCiAgICAgICAgb3V0cHV0X2RpciAvICJmaWVsZHNfb25fY29tbW9uX2dyaWQubnB6IiwKICAgICAgICB4eT1ncmlkX3h5LAogICAgICAgIHZlbG9jaXR5PXZlbG9jaXR5X2dyaWQsCiAgICAgICAgcHJlc3N1cmU9cHJlc3N1cmVfZ3JpZCwKICAgICAgICBwcmVzc3VyZV9wcm9iZV94eT1wcm9iZXMsCiAgICAgICAgcHJlc3N1cmVfcHJvYmVfdmFsdWVzPXByb2JlX3ByZXNzdXJlLAogICAgKQogICAgbnAuc2F2ZXpfY29tcHJlc3NlZCgKICAgICAgICBvdXRwdXRfZGlyIC8gImZlbV9jb2VmZmljaWVudHMubnB6IiwKICAgICAgICBtZXNoX2dlb21ldHJ5PWRvbWFpbi5nZW9tZXRyeS54LAogICAgICAgIHZlbG9jaXR5X2RvZl9jb29yZGluYXRlcz1XLnN1YigwKS5jb2xsYXBzZSgpWzBdLnRhYnVsYXRlX2RvZl9jb29yZGluYXRlcygpLAogICAgICAgIHZlbG9jaXR5X2NvZWZmaWNpZW50cz12ZWxvY2l0eS54LmFycmF5LAogICAgICAgIHByZXNzdXJlX2RvZl9jb29yZGluYXRlcz1XLnN1YigxKS5jb2xsYXBzZSgpWzBdLnRhYnVsYXRlX2RvZl9jb29yZGluYXRlcygpLAogICAgICAgIHByZXNzdXJlX2NvZWZmaWNpZW50cz1wcmVzc3VyZS54LmFycmF5LAogICAgICAgIGVsZW1lbnRfcGFpcj1ucC5hc2FycmF5KFsiTGFncmFuZ2UtUDItdmVjdG9yIiwgIkxhZ3JhbmdlLVAxLXByZXNzdXJlIl0pLAogICAgKQogICAgcW9pcyA9IHsKICAgICAgICAic3RhdHVzIjogU1RBVFVTLAogICAgICAgICJsZXZlbCI6IGxldmVsWyJpZCJdLAogICAgICAgICJkcmFnX2ZvcmNlIjogZHJhZ19mb3JjZSwKICAgICAgICAibGlmdF9mb3JjZSI6IGxpZnRfZm9yY2UsCiAgICAgICAgIkNkIjogc2NhbGUgKiBkcmFnX2ZvcmNlLAogICAgICAgICJDbCI6IHNjYWxlICogbGlmdF9mb3JjZSwKICAgICAgICAicHJlc3N1cmVfZnJvbnQiOiBmbG9hdChwcm9iZV9wcmVzc3VyZVswXSksCiAgICAgICAgInByZXNzdXJlX3JlYXIiOiBmbG9hdChwcm9iZV9wcmVzc3VyZVsxXSksCiAgICAgICAgImRlbHRhX3BfZnJvbnRfbWludXNfcmVhciI6IGRlbHRhX3AsCiAgICAgICAgImZlYXRmbG93X3JlZmVyZW5jZSI6IEZFQVRGTE9XLAogICAgICAgICJzbmVzX3N0YWdlcyI6IHN0YWdlcywKICAgICAgICAiY2VsbF9jb3VudCI6IGludChkb21haW4udG9wb2xvZ3kuaW5kZXhfbWFwKGRvbWFpbi50b3BvbG9neS5kaW0pLnNpemVfZ2xvYmFsKSwKICAgICAgICAiZ2VvbWV0cnlfbm9kZXNfbG9jYWwiOiBpbnQoZG9tYWluLmdlb21ldHJ5Lnguc2hhcGVbMF0pLAogICAgICAgICJkb2ZzX2dsb2JhbCI6IGludChXLmRvZm1hcC5pbmRleF9tYXAuc2l6ZV9nbG9iYWwgKiBXLmRvZm1hcC5pbmRleF9tYXBfYnMpLAogICAgICAgICJwcmVzc3VyZV9wcm9iZV9wb2xpY3kiOiAiZGlyZWN0IHBvaW50IGV2YWx1YXRpb247IGJvdW5kYXJ5IG9uZS1zaWRlZCB0cmFjZSBub3QgeWV0IGluZGVwZW5kZW50bHkgdmVyaWZpZWQiLAogICAgICAgICJyZWZlcmVuY2VfYWNjZXB0ZWQiOiBGYWxzZSwKICAgIH0KICAgIChvdXRwdXRfZGlyIC8gInFvaXMuanNvbiIpLndyaXRlX3RleHQoanNvbi5kdW1wcyhxb2lzLCBpbmRlbnQ9MiwgYWxsb3dfbmFuPUZhbHNlKSArICJcbiIsIGVuY29kaW5nPSJ1dGYtOCIpCiAgICB3aXRoIGlvLlhETUZGaWxlKGNvbW0sIHN0cihvdXRwdXRfZGlyIC8gIm1lc2gueGRtZiIpLCAidyIpIGFzIHhkbWY6CiAgICAgICAgeGRtZi53cml0ZV9tZXNoKGRvbWFpbikKICAgIHJldHVybiBxb2lzCgoKZGVmIHZpc2Nvc2l0eV9mb3JfcW9pKGRvbWFpbik6CiAgICByZXR1cm4gZmVtLkNvbnN0YW50KGRvbWFpbiwgUEVUU2MuU2NhbGFyVHlwZShOVSkpCgoKZGVmIG1haW4oKToKICAgIHBhcnNlciA9IGFyZ3BhcnNlLkFyZ3VtZW50UGFyc2VyKGRlc2NyaXB0aW9uPV9fZG9jX18pCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLW91dHB1dCIsIHR5cGU9UGF0aCwgZGVmYXVsdD1ST09UIC8gInJ1bnMiKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1nbXNoIiwgZGVmYXVsdD0iZ21zaCIsIGhlbHA9Ikdtc2ggZXhlY3V0YWJsZSBvbiBQQVRIIikKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tbWF4LWxldmVscyIsIHR5cGU9aW50LCBkZWZhdWx0PTMsCiAgICAgICAgICAgICAgICAgICAgICAgIGhlbHA9InJ1biB0aGUgZmlyc3QgMS4uNiBwbGFubmVkIG1lc2ggbGV2ZWxzIikKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0td2FsbC1jYXAtc2Vjb25kcyIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9MTQ1MC4wLAogICAgICAgICAgICAgICAgICAgICAgICBoZWxwPSJ3b3JrZXIgd2FsbCBjYXA7IGNvbXBsZXRlZCBsZXZlbCBvdXRwdXRzIHBlcnNpc3QgYXMgcGFydGlhbCBldmlkZW5jZSIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLW1pbmltdW0tbGV2ZWwtcmVzZXJ2ZS1zZWNvbmRzIiwgdHlwZT1mbG9hdCwgZGVmYXVsdD0xODAuMCwKICAgICAgICAgICAgICAgICAgICAgICAgaGVscD0iZG8gbm90IHN0YXJ0IGEgbmV3IG1lc2ggdW5sZXNzIHRoaXMgbXVjaCBjYXAgcmVtYWlucyIpCiAgICBhcmdzID0gcGFyc2VyLnBhcnNlX2FyZ3MoKQogICAgaWYgbm90IDEgPD0gYXJncy5tYXhfbGV2ZWxzIDw9IGxlbihMRVZFTFMpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIi0tbWF4LWxldmVscyBtdXN0IGJlIGJldHdlZW4gMSBhbmQgdGhlIG51bWJlciBvZiBkZWNsYXJlZCBsZXZlbHMiKQogICAgaWYgYXJncy53YWxsX2NhcF9zZWNvbmRzIDw9IDAgb3IgYXJncy5taW5pbXVtX2xldmVsX3Jlc2VydmVfc2Vjb25kcyA8IDA6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigid2FsbCBjYXAgbXVzdCBiZSBwb3NpdGl2ZSBhbmQgcmVzZXJ2ZSBub24tbmVnYXRpdmUiKQogICAgbGV2ZWxzID0gTEVWRUxTWzphcmdzLm1heF9sZXZlbHNdCiAgICBzdGFydGVkID0gdGltZS5wZXJmX2NvdW50ZXIoKQogICAgZGVhZGxpbmUgPSBzdGFydGVkICsgYXJncy53YWxsX2NhcF9zZWNvbmRzCiAgICByZXN1bHRzID0gW10KICAgIHN1bW1hcnlfcGF0aCA9IGFyZ3Mub3V0cHV0IC8gInN1bW1hcnkuanNvbiIKICAgIGVudmlyb25tZW50ID0gewogICAgICAgICJweXRob24iOiBzeXMudmVyc2lvbi5zcGxpdCgpWzBdLAogICAgICAgICJkb2xmaW54IjogZ2V0YXR0cihfX2ltcG9ydF9fKCJkb2xmaW54IiksICJfX3ZlcnNpb25fXyIsICJ1bmtub3duIiksCiAgICAgICAgImdtc2giOiBnZXRhdHRyKF9faW1wb3J0X18oImdtc2giKSwgIl9fdmVyc2lvbl9fIiwgInVua25vd24iKSwKICAgICAgICAicGV0c2MiOiBzdHIoUEVUU2MuU3lzLmdldFZlcnNpb24oKSksCiAgICAgICAgInVmbCI6IGdldGF0dHIodWZsLCAiX192ZXJzaW9uX18iLCAidW5rbm93biIpLAogICAgfQoKICAgIGRlZiBwZXJzaXN0KHN0YXR1czogc3RyLCAqLCBlcnJvcjogc3RyIHwgTm9uZSA9IE5vbmUpIC0+IE5vbmU6CiAgICAgICAgcW9pX2Vycm9ycyA9IE5vbmUKICAgICAgICBtZXNoX2NoYW5nZSA9IE5vbmUKICAgICAgICBpZiByZXN1bHRzOgogICAgICAgICAgICBmaW5hbCA9IHJlc3VsdHNbLTFdCiAgICAgICAgICAgIHFvaV9lcnJvcnMgPSB7CiAgICAgICAgICAgICAgICAiQ2RfcmVsYXRpdmUiOiBhYnMoZmluYWxbIkNkIl0gLSBGRUFURkxPV1siQ2QiXSkgLyBGRUFURkxPV1siQ2QiXSwKICAgICAgICAgICAgICAgICJDbF9hYnNvbHV0ZSI6IGFicyhmaW5hbFsiQ2wiXSAtIEZFQVRGTE9XWyJDbCJdKSwKICAgICAgICAgICAgICAgICJkZWx0YV9wX3JlbGF0aXZlIjogYWJzKGZpbmFsWyJkZWx0YV9wX2Zyb250X21pbnVzX3JlYXIiXSAtIEZFQVRGTE9XWyJkZWx0YV9wIl0pIC8gRkVBVEZMT1dbImRlbHRhX3AiXSwKICAgICAgICAgICAgfQogICAgICAgIGlmIGxlbihyZXN1bHRzKSA+PSAyOgogICAgICAgICAgICBjb2Fyc2UsIGZpbmUgPSByZXN1bHRzWy0yOl0KICAgICAgICAgICAgbWVzaF9jaGFuZ2UgPSB7CiAgICAgICAgICAgICAgICAiQ2RfcmVsYXRpdmUiOiBhYnMoZmluZVsiQ2QiXSAtIGNvYXJzZVsiQ2QiXSkgLyBtYXgoYWJzKGZpbmVbIkNkIl0pLCAxZS0zMCksCiAgICAgICAgICAgICAgICAiQ2xfcmVsYXRpdmUiOiBhYnMoZmluZVsiQ2wiXSAtIGNvYXJzZVsiQ2wiXSkgLyBtYXgoYWJzKGZpbmVbIkNsIl0pLCAxZS0zMCksCiAgICAgICAgICAgICAgICAiQ2xfYWJzb2x1dGUiOiBhYnMoZmluZVsiQ2wiXSAtIGNvYXJzZVsiQ2wiXSksCiAgICAgICAgICAgICAgICAiZGVsdGFfcF9yZWxhdGl2ZSI6IGFicyhmaW5lWyJkZWx0YV9wX2Zyb250X21pbnVzX3JlYXIiXSAtIGNvYXJzZVsiZGVsdGFfcF9mcm9udF9taW51c19yZWFyIl0pIC8gbWF4KGFicyhmaW5lWyJkZWx0YV9wX2Zyb250X21pbnVzX3JlYXIiXSksIDFlLTMwKSwKICAgICAgICAgICAgfQogICAgICAgIGV2aWRlbmNlX3JlYWR5ID0gYm9vbCgKICAgICAgICAgICAgc3RhdHVzID09ICJjYW5kaWRhdGVfY29tcGxldGUiIGFuZCBsZW4ocmVzdWx0cykgPj0gMyBhbmQgcW9pX2Vycm9ycyBhbmQgbWVzaF9jaGFuZ2UKICAgICAgICAgICAgYW5kIHFvaV9lcnJvcnNbIkNkX3JlbGF0aXZlIl0gPD0gMC4wMDUKICAgICAgICAgICAgYW5kIHFvaV9lcnJvcnNbIkNsX2Fic29sdXRlIl0gPD0gMC4wMDA1CiAgICAgICAgICAgIGFuZCBxb2lfZXJyb3JzWyJkZWx0YV9wX3JlbGF0aXZlIl0gPD0gMC4wMDUKICAgICAgICAgICAgYW5kIG1lc2hfY2hhbmdlWyJDZF9yZWxhdGl2ZSJdIDwgMC4wMDIKICAgICAgICAgICAgYW5kIG1lc2hfY2hhbmdlWyJDbF9yZWxhdGl2ZSJdIDwgMC4wMDIKICAgICAgICAgICAgYW5kIG1lc2hfY2hhbmdlWyJkZWx0YV9wX3JlbGF0aXZlIl0gPCAwLjAwMgogICAgICAgICkKICAgICAgICBwYXlsb2FkID0gewogICAgICAgICAgICAic3RhdHVzIjogc3RhdHVzLAogICAgICAgICAgICAicmVmZXJlbmNlX2FjY2VwdGVkIjogRmFsc2UsCiAgICAgICAgICAgICJzY2llbnRpZmljX2FjY2VwdGVkIjogRmFsc2UsCiAgICAgICAgICAgICJldmlkZW5jZV9yZWFkeV9mb3JfaHVtYW5fZ2F0ZSI6IGV2aWRlbmNlX3JlYWR5LAogICAgICAgICAgICAic2Vzc2lvbl9jYXBfc2Vjb25kcyI6IGZsb2F0KGFyZ3Mud2FsbF9jYXBfc2Vjb25kcyksCiAgICAgICAgICAgICJlbGFwc2VkX3NlY29uZHMiOiByb3VuZCh0aW1lLnBlcmZfY291bnRlcigpIC0gc3RhcnRlZCwgMyksCiAgICAgICAgICAgICJwbGFubmVkX2xldmVsX2NvdW50IjogbGVuKGxldmVscyksCiAgICAgICAgICAgICJjb21wbGV0ZWRfbGV2ZWxfY291bnQiOiBsZW4ocmVzdWx0cyksCiAgICAgICAgICAgICJlbnZpcm9ubWVudF92ZXJzaW9ucyI6IGVudmlyb25tZW50LAogICAgICAgICAgICAicmVmZXJlbmNlX3FvaV9lcnJvcnNfdnNfZmVhdGZsb3ciOiBxb2lfZXJyb3JzLAogICAgICAgICAgICAidHdvX2ZpbmVzdF9tZXNoX3JlbGF0aXZlX2NoYW5nZXMiOiBtZXNoX2NoYW5nZSwKICAgICAgICAgICAgImxldmVscyI6IHJlc3VsdHMsCiAgICAgICAgfQogICAgICAgIGlmIGVycm9yOgogICAgICAgICAgICBwYXlsb2FkWyJlcnJvciJdID0gZXJyb3IKICAgICAgICBzdW1tYXJ5X3BhdGgucGFyZW50Lm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rfb2s9VHJ1ZSkKICAgICAgICB0ZW1wb3JhcnkgPSBzdW1tYXJ5X3BhdGgud2l0aF9zdWZmaXgoIi5qc29uLnRtcCIpCiAgICAgICAgdGVtcG9yYXJ5LndyaXRlX3RleHQoanNvbi5kdW1wcyhwYXlsb2FkLCBpbmRlbnQ9MiwgYWxsb3dfbmFuPUZhbHNlKSArICJcbiIsIGVuY29kaW5nPSJ1dGYtOCIpCiAgICAgICAgdGVtcG9yYXJ5LnJlcGxhY2Uoc3VtbWFyeV9wYXRoKQoKICAgIGFyZ3Mub3V0cHV0Lm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rfb2s9VHJ1ZSkKICAgIHBlcnNpc3QoInJ1bm5pbmciKQogICAgdHJ5OgogICAgICAgIGZvciBsZXZlbCBpbiBsZXZlbHM6CiAgICAgICAgICAgIGlmIGRlYWRsaW5lIC0gdGltZS5wZXJmX2NvdW50ZXIoKSA8IGFyZ3MubWluaW11bV9sZXZlbF9yZXNlcnZlX3NlY29uZHM6CiAgICAgICAgICAgICAgICBwZXJzaXN0KCJ0aW1lX2NhcF9wYXJ0aWFsIikKICAgICAgICAgICAgICAgIGJyZWFrCiAgICAgICAgICAgIG1lc2hfcGF0aCA9IG1ha2VfbWVzaChsZXZlbCwgYXJncy5vdXRwdXQgLyAibWVzaGVzIiwgYXJncy5nbXNoKQogICAgICAgICAgICByZXN1bHQgPSBleHBvcnRfbGV2ZWwobGV2ZWwsIG1lc2hfcGF0aCwgYXJncy5vdXRwdXQgLyBsZXZlbFsiaWQiXSkKICAgICAgICAgICAgcmVzdWx0cy5hcHBlbmQocmVzdWx0KQogICAgICAgICAgICBwZXJzaXN0KCJydW5uaW5nIikKICAgICAgICBlbHNlOgogICAgICAgICAgICBwZXJzaXN0KCJjYW5kaWRhdGVfY29tcGxldGUiKQogICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBleGM6CiAgICAgICAgcGVyc2lzdCgiZmFpbGVkX3BhcnRpYWwiLCBlcnJvcj1mInt0eXBlKGV4YykuX19uYW1lX199OiB7ZXhjfSIpCiAgICAgICAgcmFpc2UKCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgbWFpbigpDQo=",
    "make_mesh.geo": "Ly8gRFJBRlRfVU5WRVJJRklFRCDigJQgREZHIDJELTEsIFJlPTIwLCBwYXJhbWV0ZXJpemVkIEdtc2ggQ0FELgovLyBPdmVycmlkZSBsY19idWxrL2xjX2N5bCB3aXRoIGdtc2ggLXNldG51bWJlciBmb3IgZWFjaCByZWZpbmVtZW50IGxldmVsLgpTZXRGYWN0b3J5KCJPcGVuQ0FTQ0FERSIpOwoKTCA9IDIuMjsKSCA9IDAuNDE7CmN4ID0gMC4yOwpjeSA9IDAuMjsKciA9IDAuMDU7CmVwcyA9IDFlLTc7CgpEZWZpbmVDb25zdGFudFsKICBsY19idWxrID0gezAuMDQwLCBOYW1lICJQYXJhbWV0ZXJzL0J1bGsgbWVzaCBzaXplIiwgTWluIDAuMDAxLCBNYXggMC4yfSwKICBsY19jeWwgPSB7MC4wMDYwLCBOYW1lICJQYXJhbWV0ZXJzL0N5bGluZGVyIG1lc2ggc2l6ZSIsIE1pbiAwLjAwMDEsIE1heCAwLjA1fQpdOwoKUmVjdGFuZ2xlKDEpID0gezAsIDAsIDAsIEwsIEh9OwpEaXNrKDIpID0ge2N4LCBjeSwgMCwgciwgcn07CmZsdWlkW10gPSBCb29sZWFuRGlmZmVyZW5jZXsgU3VyZmFjZXsxfTsgRGVsZXRlOyB9eyBTdXJmYWNlezJ9OyBEZWxldGU7IH07CgovLyBCb3VuZGFyeSBlbnRpdGllcyBhcmUgc2VsZWN0ZWQgYWZ0ZXIgdGhlIEJvb2xlYW4gb3BlcmF0aW9uIHNvIHRoZSBjaXJjbGUKLy8gdGhhdCBpcyB0YWdnZWQgaXMgdGhlIGJvdW5kYXJ5IG9mIHRoZSBmbHVpZCBkb21haW4sIG5vdCB0aGUgZGlzY2FyZGVkIGRpc2suCmlubGV0W10gPSBDdXJ2ZSBJbiBCb3VuZGluZ0JveHstZXBzLCAtZXBzLCAtZXBzLCBlcHMsIEggKyBlcHMsIGVwc307Cm91dGxldFtdID0gQ3VydmUgSW4gQm91bmRpbmdCb3h7TCAtIGVwcywgLWVwcywgLWVwcywgTCArIGVwcywgSCArIGVwcywgZXBzfTsKYm90dG9tW10gPSBDdXJ2ZSBJbiBCb3VuZGluZ0JveHstZXBzLCAtZXBzLCAtZXBzLCBMICsgZXBzLCBlcHMsIGVwc307CnRvcFtdID0gQ3VydmUgSW4gQm91bmRpbmdCb3h7LWVwcywgSCAtIGVwcywgLWVwcywgTCArIGVwcywgSCArIGVwcywgZXBzfTsKY3lsaW5kZXJbXSA9IEN1cnZlIEluIEJvdW5kaW5nQm94e2N4IC0gciAtIGVwcywgY3kgLSByIC0gZXBzLCAtZXBzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3ggKyByICsgZXBzLCBjeSArIHIgKyBlcHMsIGVwc307CndhbGxzW10gPSB7Ym90dG9tW10sIHRvcFtdfTsKClBoeXNpY2FsIEN1cnZlKCJpbmxldCIsIDEpID0ge2lubGV0W119OwpQaHlzaWNhbCBDdXJ2ZSgid2FsbHMiLCAyKSA9IHt3YWxsc1tdfTsKUGh5c2ljYWwgQ3VydmUoImN5bGluZGVyIiwgMykgPSB7Y3lsaW5kZXJbXX07ClBoeXNpY2FsIEN1cnZlKCJvdXRsZXQiLCA0KSA9IHtvdXRsZXRbXX07ClBoeXNpY2FsIFN1cmZhY2UoImZsdWlkIiwgNSkgPSB7Zmx1aWRbXX07CgovLyBMb2NhbCByZWZpbmVtZW50IGFyb3VuZCB0aGUgY3VydmVkIG9ic3RhY2xlOyBzaXplcyBhcmUgdGhlIHByb3RvY29sJ3MKLy8gdGFyZ2V0IHZhbHVlcywgbm90IGEgcHJvbWlzZSBhYm91dCB0aGUgZ2VuZXJhdGVkIGVsZW1lbnQgY291bnQuCkZpZWxkWzFdID0gRGlzdGFuY2U7CkZpZWxkWzFdLkN1cnZlc0xpc3QgPSB7Y3lsaW5kZXJbXX07CkZpZWxkWzFdLlNhbXBsaW5nID0gMjAwOwpGaWVsZFsyXSA9IFRocmVzaG9sZDsKRmllbGRbMl0uSW5GaWVsZCA9IDE7CkZpZWxkWzJdLlNpemVNaW4gPSBsY19jeWw7CkZpZWxkWzJdLlNpemVNYXggPSBsY19idWxrOwpGaWVsZFsyXS5EaXN0TWluID0gMC4wMjsKRmllbGRbMl0uRGlzdE1heCA9IDAuMjU7CkJhY2tncm91bmQgRmllbGQgPSAyOwoKTWVzaC5NZXNoU2l6ZUZyb21Qb2ludHMgPSAwOwpNZXNoLk1lc2hTaXplRnJvbUN1cnZhdHVyZSA9IDA7Ck1lc2guTWVzaFNpemVFeHRlbmRGcm9tQm91bmRhcnkgPSAwOwpNZXNoLkVsZW1lbnRPcmRlciA9IDI7Ck1lc2guU2Vjb25kT3JkZXJMaW5lYXIgPSAwOwpNZXNoLk1zaEZpbGVWZXJzaW9uID0gNC4xOw0K",
}

def materialize_dfg_assets():
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    for name, encoded in _EMBEDDED_DFG_ASSETS.items():
        (SCRIPT_DIR / name).write_bytes(base64.b64decode(encoded))
MAMBA_TMP = Path("/tmp/af-pikans-closure-s1-micromamba")


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
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=max(30.0, timeout),
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
    anchor = np.asarray([2.2, 0.2], dtype=np.float64)
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
    for seed, arm in schedule:
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
            del model, payload, parts, pred
            torch.cuda.empty_cache()
        except Exception as exc:
            rows.append({"seed": seed, "arm": arm, "status": "scoring_error",
                         "checkpoint": str(checkpoint), "error": f"{type(exc).__name__}: {exc}"})

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
              "fixed_field_grid": "DFG-L2 fields_on_common_grid.npz (221x42 requested points, cylinder interior excluded)",
              "pressure_gauge": {"method": "subtract pressure at outflow anchor", "xy": anchor.tolist()},
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
        "schema_version": 1, "session": "S1b-infrastructure-retry", "status": status,
        "kernel_slug": "rodrimandub/af-pikans-competitive-baseline-s1b-retry-20260928",
        "machine": runtime, "push_cap_seconds": SESSION_CAP,
        "internal_watchdog_seconds": WATCHDOG,
        "actual_session_seconds": round(time.perf_counter() - STARTED_MONOTONIC, 3),
        "started_at_utc": UTC_STARTED, "ended_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_accepted": False, "reference_accepted": False,
        "kernel_sources": [
            "rodrimandub/af-pikans-allen-cahn-t4x2-20260927",
            "rodrimandub/af-pikans-allen-cahn-score-only-20260927",
            "rodrimandub/allen-cahn-deepxde-reference-20260928",
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
