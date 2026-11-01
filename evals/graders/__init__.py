"""Graders numéricos del verificador.

Superficie protegida: ningún mecanismo ni agente puede escribir aquí.
Los mecanismos tampoco deben importar este paquete: un candidato que puede
leer su propio grader puede optimizar la métrica en vez del problema.
"""

from .core import GRADERS, Score, grade_all

__all__ = ["GRADERS", "Score", "grade_all"]
