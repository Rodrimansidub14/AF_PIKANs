# Allen-Cahn score-only Kaggle package

Files:
- allen-cahn-score-only.ipynb: score-only notebook.
- kernel-metadata.json: private Kaggle metadata; T4x2 GPU, internet off, source output from `rodrimandub/af-pikans-allen-cahn-t4x2-20260927` attached.
- SHA256SUMS.txt: hashes for the notebook, metadata, and this README.

The notebook scores existing seeds 401-405 for AC-RBF-Sobol and AC-DeepXDE-MLP; it never calls training or reference-generation functions. It loads the original runner module only for model construction and score functions. Embedded runner SHA-256: 47c02c9279e564729feee848f4f9266a746aa19869727dcb47d682ad0a103241.

A 2,100-second watchdog starts at the first executable cell; quota cap is 2,400 seconds including setup. It requires all ten checkpoints, five saved pools, and an integrity-matched reference candidate under /kaggle/input, and writes score-only results under /kaggle/working. Reference and scientific acceptance remain false. This draft was prepared offline; the notebook has not been run, tested, validated, preflighted, uploaded, or submitted.
