# Reference: Competitor Analysis and Architecture Decisions

## DELi (UNC Chapel Hill, Popov Lab)

Source: Wellnitz et al. (2025). bioRxiv 2025.02.25.640184.
Repository: https://github.com/Popov-Lab-UNC/DELi

### What DELi Does That DELT-Hit Doesn't (Gaps to Fill)
- Barcode design module (Hamming encoding for error correction)
- Nextflow parallelization (10B reads in <15 min on 800 cores)
- Automated DEL-ML regression with data balancing
- HTML decoding report with read-fate pie charts

### What DELT-Hit Does That DELi Doesn't (Differentiators to Protect)
- Arbitrary reaction graph support for complex synthetic schemes
- Dual-display DEL architecture support
- edgeR statistical models (negative binomial with dispersion estimation)
- Standardized SMILES output per library member
- Exposed intermediate scripts for reproducibility

### DELi Technical Details
- Python, installable via pip (package name: deli-chem)
- Config: JSON files (vs DELT-Hit's YAML from Excel)
- Decoding speed: >1M reads/min single core
- Modules: deli.decode, deli.enumerate, deli.analysis

## DELTA-Toolkit (HitGen)

- Proprietary Java JAR file
- Uses samtools for alignment
- Outputs DataWarrior format (.dwar)
- Not open for modification or extension

## DEL-Dock (insitro)

Source: Shmilovich et al. (2023). J. Chem. Inf. Model. 63, 2719-2727.

- Combines molecular docking (3D) with probabilistic count models
- Uses zero-inflated Poisson distributions for DEL count data
- Requires protein crystal structure (not always available)
- Decision: do NOT integrate into DELT-Hit core; ensure output compatibility

## Open DEL-ML Framework (SGC/Edwards lab)

Source: Edwards et al. (2024-2025). ChemRxiv 2024-xd385.

- Fully open automated DEL-ML pipeline
- Uses chemical fingerprints to mask proprietary structures
- Light Gradient Boosted Trees rival proprietary pipelines
- Public DEL datasets shared via AIRCHECK database
- Decision: ensure DELT-Hit's fingerprint output format is compatible

## Architecture Decision Record

### Why Snakemake over Nextflow
- DELT-Hit is Python-native; Snakemake is Python-native
- File-based DAG model matches DELT-Hit's existing I/O pattern
- Lower learning curve for the target user base (experimental chemists)
- If cloud deployment becomes priority, revisit for Nextflow

### Why edgeR stays as default (not replaced)
- Community trust and citation record
- Well-validated negative binomial model handles overdispersion
- New methods (z-score, Poisson ML) added as ALTERNATIVES, not replacements

### Why src/ layout for restructuring
- Prevents accidental import of development code instead of installed version
- Python Packaging Authority recommendation (PEP 517/518/621)
- Separates source from tests, docs, configs cleanly
