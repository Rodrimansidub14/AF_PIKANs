# DFG Re=20 force-traction score-only re-evaluation

Consumes the ten saved checkpoints from `rodrimandub/af-pikans-dfg-re20-competitive-pilot-t4x2`; no training or tuning. The notebook embeds the original campaign runner unchanged (SHA-256 `4c02809600beb9b543d1bceffe2037ed7d05f5ff654d010b22984227b2cb35a9`) to preserve provenance, then replaces only the in-memory QoI scoring function.

Cylinder forces use the official FEATFLOW stress definition `sigma = nu grad(u) - p I`, integrated as `sigma n` with radial outward normal. This removes tangent-orientation ambiguity in the earlier simplified implementation. Pressure probes, held-out points, residuals, regions, checkpoints, and training runs are unchanged. The kernel uses Kaggle `NvidiaTeslaT4` (2×T4), internet disabled, and a 1,680 s watchdog within the 1,800 s cap.

This score-only result has no converged FEM field; `field_relative_l2` remains null and `scientific_accepted=false`. It is not a published PINN SOTA reproduction.
