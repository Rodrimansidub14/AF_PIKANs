"""Configura en este clon el filtro de notebooks. Ejecutar una vez tras clonar:  python tools/setup_git.py"""
import subprocess
import sys

for k, v in (("filter.nbstrip.clean", f'"{sys.executable}" tools/nbstrip.py'), ("filter.nbstrip.smudge", "cat"), ("filter.nbstrip.required", "true")):
    subprocess.run(["git", "config", "--local", k, v], check=True)
print("filtro nbstrip configurado")
