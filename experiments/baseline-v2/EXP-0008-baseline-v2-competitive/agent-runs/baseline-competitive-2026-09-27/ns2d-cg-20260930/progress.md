# Progreso GPU 2026-09-30 (UTC)

- Aprobación del investigador (chat, 2026-09-30): pasos 1-5, 12 h restantes, dejar 6 h de reserva (gastar <= 6 h GPU en total). Exploratorio, pre-registro sin firmar.
- Cuota antes de empezar: 17.61 h usadas / 12.39 h restantes / 30 h (renueva 2026-10-03). Presupuesto planeado: bench <=0.25 h, AC <=2.0 h, NS <=3.5 h (total <=5.75 h; reserva >= 6.6 h).
- 06:31Z bench push OK: rodrimandub/af-pikans-ns-ac-bench-20260930 (cap 900 s).
- 06:34Z bench COMPLETE (session 144 s). NS full-batch f32: MLP5x100 30.9 it/s; RBF w16 17.9; w32 11.9; w48 8.0; w64 6.1; f64 3-4x slower. AC f32: MLP3x20 107 it/s; RBF w16 48.6 it/s (f64 20.4).
- Cuota antes del lanzamiento: 12.35 h restantes. Topes: AC 7200 s + NS 12600 s = 5.5 h -> reserva proyectada >= 6.85 h (>= 6 h pedidos).
- AC push: rodrimandub/af-pikans-ac-rerun-20260930 (RBF w16 vs MLP 3x20, seeds 601-605, 40k Adam f32 + L-BFGS 300 s, score vs Allen_Cahn.mat).
- NS push: rodrimandub/af-pikans-ns2dcg-20260930 (MLP 5x100 vs RBF w56 [param-matched 40,992 vs 41,003] seeds 701-705 + RBF w16 seeds 701-703; 20k Adam f32; score vs COMSOL ns_0_obstacle.dat).
- NOTA: el slug real del kernel NS es rodrimandub/af-pikans-ns2d-cg-20260930 (Kaggle derivo el slug del titulo). Ambos lanzados 06:40Z.
- 07:27Z Corregidas las filas Cd/Cl de DFG 301-305 en scorecard-current.csv (backup: scorecard-current.pre-2026-09-30-dfg-fix.csv) y las citas en execution/plan/completion-plan/readout/decision-brief/repo-readiness.
- 07:47Z AC COMPLETE (3869 s = 1.07 h). RBF w16: 2.686e-3 ± 9.5e-4 tras L-BFGS vs MLP 3x20: 1.773e-2 ± 8.7e-3; RBF gana 5/5. Descargado en outputs/ac, resultados en ac-rerun-results.md. NS sigue RUNNING.
- 09:52Z NS COMPLETE (11388 s = 3.16 h). flat L2RE: MLP 0.0977±0.0067, RBF56 0.0655±0.0058, RBF16 0.0667±0.0023; publicado vanilla 0.119, LAAF 0.0824. Resultados en ns2dcg-results.md.
- 18:18Z Heat L-BFGS redo (GPU) pushed: rodrimandub/af-pikans-heat-lbfgs-redo-20260930, cap 2700 s, 700 s/seed, arranca de checkpoint-adam-final de S2 seeds 901-905. CPU dio 2.15e-6 (901) y 1.93e-6 (903) tras solo 50 iter.
- 18:27Z DFG config-Daniel push: rodrimandub/af-pikans-dfg-daniel-cfg-20260930 (20 jobs: literal/corrected x mlp4x50/rbf16 x seeds 801-805), cap 3600 s. Cuota antes: 7.95 h (Heat redo corriendo, cap 0.75 h) -> reserva proyectada >= 6.3 h.
- 19:11Z Heat L-BFGS redo COMPLETE (1566 s): 1.857e-6 ± 1.8e-7 vs archivado 2.061e-6 (mejora ~10 %, piso ~1.9e-6 a las 50 it.). Ver heat-lbfgs-redo-20260930/results.md
- 19:24Z DFG config-Daniel COMPLETE (2739 s, 20/20). Con 2000 Adam + 1000 LBFGS ningun brazo resuelve: MLP u~0.9, RBF u~0.5; RBF gana 5/5. Ver dfg-daniel-cfg-results.md
- 2026-10-01 06:39Z DFG receta larga pushed: rodrimandub/af-pikans-dfg-long-20261001 (MLP 4x50 vs RBF16, variante corrected, semillas 801-805, 20k Adam f32 + L-BFGS strong-Wolfe 600 s tope), cap 14400 s (4 h). Cuota antes: 6.89 h -> reserva proyectada >= 2.89 h (el investigador autorizo usar la cuota actual). Estimado ~3 h. Codigo: dfg_long.py, build_dfgl_nb.py.
- 2026-10-01 10:01Z DFG receta larga COMPLETE (12010 s, 10/10). RBF16 u 0.191±0.039 / Cd err 34.7 % vs MLP 4x50 u 0.421 / Cd 69.5 %; L-BFGS paro por tope de tiempo. Ver dfg-long-results.md. Cuota restante 3.55 h.

- 2026-10-01 piezo EXP-0009 GPU: probe/long1/vsplit1e-4 did not learn bending (MLP best u 0.64-0.71, RBF collapses to null). Quota after: see kaggle quota. Next: formulacion estandar 3 salidas / pesos adaptativos; no mas GPU a ciegas.
- 2026-10-03 Burgers cierre pushed: rodrimandub/af-pikans-burgers-close-20261003 (MLP5x100, RBF16, RBF48; semillas 301-305; 20k Adam f32 + L-BFGS 600 s; ref Cole-Hopf 1025x129), cap 13000 s. Cuota 30 h renovada (reset 2026-10-10).
- 2026-10-03 Burgers cierre COMPLETE (5078 s): MLP 5.66e-4, RBF16 1.87e-3, RBF48 3.87e-4 tras L-BFGS (empate RBF48/MLP). Ver burgers-close-results.md
- 2026-10-03 Burgers vs FEM PINNacle (CPU): MLP L-BFGS 1.3305e-2, RBF48 1.3272e-2 (= piso de la referencia, publicado 1.33e-2). Lanzado burgers-f64lbfgs-20261003 (mlp, rbf48, f32 Adam + f64 L-BFGS 900 s), cap 12000 s. Pendiente: burgers-big (16384 pts, 30k Adam).
- 2026-10-03 Burgers f64 L-BFGS COMPLETE: MLP 4.06e-4, RBF48 2.90e-4 (empate). Lanzado burgers-big (16384 pts, 24k Adam, L-BFGS f64 1800 s). Piezo: daniel-recipe en curso.
