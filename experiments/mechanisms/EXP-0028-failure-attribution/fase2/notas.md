## Notas y definiciones

- cond(B) = cond de (B/m + 1e-3 I), el bloque que P factoriza (Enmienda 1); el B crudo es casi singular. Techo teorico ~ (11+1e-3)/1e-3 = 1.1e4, de modo que condB_max ya esta saturado en todos los problemas; lo que discrimina es la mediana y el minimo (NS el mas bajo, KPP el mas alto).
- Pool de bloques: 2048 puntos equiespaciados del pool de PDE (gp.subset), como el mecanismo. NS usa un residuo vectorial local (3 ecuaciones; comprobado contra prob.loss, error relativo 9e-8), porque gn_blocks solo admite residuo escalar. Solo PDE, sin terminos de borde (convencion actual del mecanismo).
- Numeros adimensionales (L = 1, T = 1): AC damkohler = a L^2/d; Burgers peclet = U L/nu (U=1); KPP damkohler = r L^2/D y peclet = c L/D (c = 5 sqrt(rD/6)); NS Re = U L/nu = 100. front_width: AC sqrt(d/a), Burgers nu/U, KPP 1/k. front_over_rbf = front_width / ancho RBF en x.
- AC d=1e-3 y 2e-4 comparten pool, semilla y modelo: difieren solo en d (y en el residuo). Sus propiedades de red casi coinciden; la diferencia esta en damkohler y front_over_rbf.
- AC/Burgers/KPP: IC y BC duras, sin terminos de borde en la perdida: bc_over_pde no aplica.
- NS: ancho oculto 16 elegido por mi (el spec no lo fija); `--ns-width` lo cambia.
- Hooks pendientes: ver HOOKS_PENDIENTES en measure_properties.py (DFG y piezo).
