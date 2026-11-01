"""EXP-0024: bloques de semillas disjuntos por nivel. T0/T1 exploran; V valida; C es el preregistro de confirmación (cada experimento fija las suyas, >= 800)."""
T_BLOCK = range(1, 100)        # T0 y T1
V_BLOCK = range(100, 200)      # validación
C_MIN = 800                    # confirmación: semillas >= 800 (EXP-0023: 801-805 Burgers, 811-815 Allen-Cahn)
USED = set(range(301, 306)) | set(range(601, 606))   # EXP-0016 a 0018 (y 0022)
