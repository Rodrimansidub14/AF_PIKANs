from pathlib import Path
f=Path('j-capitulos.tex');z=f.read_text(encoding='utf-8').replace('T_{\\rm campaña}','T_{\\text{campaña}}');f.write_text(z,encoding='utf-8')
f=Path('i-marco_teorico.tex');z=f.read_text(encoding='utf-8');target='hp-VPINN incorpora pruebas locales y partición'
insert=r'''Para explicitar otras construcciones del SOTA, un residual variacional local es $R_{K,j}=\int_K r(z;\theta)v_j(z)\,dz$; integrar por partes según el operador reduce el orden de derivación del aproximador. Una pérdida hp-VPINN suma $|R_{K,j}|^2$ sobre elementos y pruebas, con condiciones de frontera declaradas. En entrenamiento causal, una forma habitual de peso es $w_n=\exp(-\epsilon_c\sum_{j<n}\mathcal L_j)$, donde $\mathcal L_j$ mide residual de tiempos anteriores: el peso responde al avance temporal, no modifica por sí mismo el espacio representacional. En el entrenamiento básico, $\theta_{k+1}=\theta_k-\eta\nabla_\theta\mathcal L$ ilustra el papel del gradiente; Adam añade estimaciones de momentos. Cambiar pesos, geometría o estado del optimizador son intervenciones distintas.

'''
z=z.replace(target,insert+target);f.write_text(z,encoding='utf-8')
