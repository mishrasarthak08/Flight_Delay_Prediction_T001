# Build the report

From the repository root, with Tectonic installed and network access for its IEEEtran and LaTeX packages:

```sh
cd docs/report
tectonic T001_Phase2_verified.tex
```

The source uses the existing figures in `../../results/figures/`; keep that relative directory structure when building. Tectonic resolves the BibTeX bibliography and repeated LaTeX passes automatically. A conventional TeX distribution can instead run `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` on the same basename. Existing project results and models are not modified by this build.

For a full computational reproduction, follow the environment and data setup in the repository README. The pinned CSV SHA-256 is `8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa`. From the repository root, use fresh output directories to preserve the committed artifacts:

```sh
python scripts/download_data.py --verify-only
python scripts/run_pipeline.py --seed 42 --epochs 15 --models-dir /tmp/t001-models --results-dir /tmp/t001-results
python scripts/verify_artifacts.py
python -m pytest
```

The verification script reads the default committed artifact directories; it does not automatically check the fresh output paths above. These commands were not run for this report.
