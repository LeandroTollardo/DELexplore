# Reference: Actual DELT-Hit Output Formats (Ground Truth)

Based on examination of real DELT-Hit output files, April 2026.

## counts.txt — Selection Count Data

Format: Tab-separated values (TSV), NOT parquet.
Location: One file per selection in folder `selections/{selection_name}/counts.txt`
Example folder: `JPAG_2025_1/counts.txt`

### Columns:
```
code_1    code_2    count    id
47        543       19290    47_543
278       453       16633    278_453
77        508       14711    77_508
```

- `code_1`: Integer index of building block at position 1 (0-indexed from BB list)
- `code_2`: Integer index of building block at position 2
- `count`: Integer sequence count for this compound in this selection
- `id`: String concatenation `{code_1}_{code_2}` (convenience column)

### Key observations:
- This is a 2-CYCLE library (code_1 and code_2 only)
- Columns are named `code_N`, NOT `B0_id` or `BB_id`
- The number of code columns varies by library (2 for 2-cycle, 3 for 3-cycle, etc.)
- File is SORTED by count descending
- 317,504 rows in this example (compounds with at least 1 count)
- Many compounds with count=1 at the bottom (noise floor)
- Top compound has count 19,290 — strong enrichment signal

### What DELexplore must handle:
- Read TSV with tab separator
- Detect number of code columns automatically (code_1, code_2, ... code_N)
- Handle varying numbers of cycles (2, 3, or more)
- The `id` column is redundant (can be reconstructed) — use code columns as primary key

## config.yaml — Experiment Metadata

### Structure:

```yaml
experiment:
  name: JP-1                    # experiment identifier
  fastq_path: /path/to/file.fastq.gz
  save_dir: /path/to/output
  num_cores: 10

selections:
  JPAG_2025_1:                  # selection name (matches folder name)
    operator: J. Puff           # who ran the experiment
    date: '2025-11-23'
    target: No Protein          # target protein name (or "No Protein" for blank)
    group: no_protein           # grouping for statistical comparison
    beads: HisPURE Beads        # bead/matrix type
    info: '-'                   # additional info
    blocking: Biotin            # blocking agent
    buffer: PBS-T               # buffer used
    protocol: DECL_5W           # selection protocol
    S0: ACACAC                  # selection barcode
    S1: CGCTCGATA
    ids: [0, 0]                 # internal identifiers

  JPAG_2025_7:
    target: L1CAM Ig1to4 His    # a specific protein target
    group: protein              # grouped with other protein selections
    beads: HisPURE Beads
    # ... other fields same structure

library:
  products: [product_1, product_2, ...]
  educts: [product_1, scaffold_1, ...]
  reactions: [ABF, CuAAC, DH, PASS, SR, ...]
  bb_edges: [...]               # reaction graph edges
  B0:                           # building block definitions per position
    - index: 0
      smiles: "NC(CC1=CC=CC=C1)C(O)=O"
      codon: AACCTGC
      reaction: ABF
      educt: scaffold_1b
      product: product_1
    - index: 1
      smiles: "..."
      # ...
  B1:
    - index: 0
      smiles: "..."
      codon: AAGTCAA
      reaction: Son
      educt: product_1
      product: product_2
    # ...
```

### Key metadata for DELexplore:
- `selections.{name}.target`: The protein target. "No Protein" = blank control.
- `selections.{name}.group`: Statistical grouping (e.g., "protein" vs "no_protein")
- `selections.{name}.beads`: Bead type — critical for bead-artifact analysis
- `selections.{name}.protocol`: Selection protocol
- `library.B0`, `library.B1`, ...: Building block definitions with SMILES

### What DELexplore must extract from config:
1. Selection → group mapping (for defining comparisons)
2. Selection → target mapping (for multi-target analysis)
3. Selection → bead type mapping (for bead-artifact detection)
4. Building block SMILES per position (for chemical analysis)
5. Number of BB positions = number of code columns in counts

## library.parquet — Enumerated Library (if available)

Format: Parquet file
Columns: code_0, code_1, ..., code_N, smiles
Contains SMILES for every enumerated compound.
May NOT always be available — DELexplore should work without it
(using BB SMILES from config.yaml instead).

## DELexplore I/O Strategy

The io/readers.py module must handle:
1. Reading counts.txt as TSV (polars.read_csv with separator='\t')
2. Auto-detecting cycle count from number of code_* columns
3. Reading config.yaml with yaml.safe_load
4. Extracting selection metadata into a structured format
5. Optionally reading library.parquet if it exists
6. Reading multiple selection folders and combining into one counts matrix
   with an additional `selection` column
