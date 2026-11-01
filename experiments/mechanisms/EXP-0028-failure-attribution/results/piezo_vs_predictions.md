# Piezo vs predictions.md (n = 3 por problema, provisional)

## converse (semillas con B: [20260728, 20260729, 20260730])

| brazo | n | razón l2re_u por semilla | mediana | k/n mejora | máx | banda | previsto |
|---|---|---|---|---|---|---|---|
| F | 3 | 0.639, 4.799, 1.338 | **1.338** | 1/3 | 4.799 | empeora | mejora (≈0.6) |
| E | 3 | 0.135, 1.588, 0.181 | **0.181** | 2/3 | 1.588 | mejora | mejora (≈0.5) |
| J | 3 | 0.293, 0.793, 0.670 | **0.670** | 3/3 | 0.793 | mejora | empeora leve (>1.10) |

B l2re_u por semilla: 0.367, 0.048, 0.180 (mediana 0.180); referencia del paper (MLP): u 0.107.

**Criterio (n = 3, provisional):** mejor observado E (previsto E); peor observado F (previsto J) → (i) NO cumple; (ii) F a 0.35 décadas (NO), E a 0.44 (NO); direcciones acertadas 1/3 frente a la tasa base ingenua 1/3 (no supera). **H-PRED-1 en converse: no respaldada (provisional)**.

## direct (semillas con B: [20260728, 20260729, 20260730])

| brazo | n | razón l2re_u por semilla | mediana | k/n mejora | máx | banda | previsto |
|---|---|---|---|---|---|---|---|
| F | 3 | 5.393, 2.639, 3.931 | **3.931** | 0/3 | 5.393 | empeora | mejora (≈0.6) |
| E | 3 | 0.712, 0.823, 1.240 | **0.823** | 2/3 | 1.240 | mejora | mejora (≈0.5) |
| J | 3 | 0.941, 0.749, 0.946 | **0.941** | 3/3 | 0.946 | indiferente | empeora leve (>1.10) |

B l2re_u por semilla: 0.180, 0.240, 0.204 (mediana 0.204); referencia del paper (MLP): u 0.117.

**Criterio (n = 3, provisional):** mejor observado E (previsto E); peor observado F (previsto J) → (i) NO cumple; (ii) F a 0.82 décadas (NO), E a 0.22 (ok); direcciones acertadas 1/3 frente a la tasa base ingenua 2/3 (no supera). **H-PRED-1 en direct: no respaldada (provisional)**.