# Face Preprocessing Pipeline for GFPGAN Training




```
\# Face Preprocessing & Degradation Pipeline for GFPGAN Training

A complete, automated pipeline to process raw face images into paired high-quality (ground truth) and low-quality (degraded input) images for face restoration model training (e.g., GFPGAN, ESRGAN).

\## Key Features

\- \*\*MTCNN Face Alignment\*\*: Aligns faces to standard 5-point landmarks (0% failure rate for valid face images)

\- \*\*Mild Tight Crop\*\*: Removes 5% margin to retain core facial features (no excess background)

\- \*\*Custom Fix for 04646.jpg\*\*: Resolves MTCNN detection failure with manual crop logic

\- \*\*GFPGAN-Style Degradation\*\*: Simulates realistic low-quality faces (blur → downsampling → noise → JPEG compression)

\- \*\*One-Click Automation\*\*: Batch script to run full pipeline without manual intervention

\- \*\*Dataset Validation\*\*: Verifies 1:1 pairing between high/low-quality images (critical for training)

\## Project Structure
```

8016project/

├── run\_all\_conda.bat               # One-click pipeline runner (Conda)

├── README.md                       # Project documentation

├── ENV\_SETUP.md                    # Environment setup guide

├── Folder\_Structure.txt            # Folder/file purpose explanation

├── requirements\_versions.txt       # Exported dependency versions

├── scripts/

│   ├── preprocess.py               # Core preprocessing (alignment + crop)

│   ├── fix\_04646.py                # Custom crop for 04646.jpg (detection fix)

│   ├── degrade.py                  # GFPGAN-style degradation

│   ├── dataset.py                  # Dataset validation

│   ├── generate\_examples.py        # Visual comparison examples

│   ├── extract\_random\_subset.py    # Optional: Raw data subset extraction

│   ├── utils.py                    # Optional: Shared utilities

│   └── shape\_predictor\_68\_face\_landmarks.dat  # Dlib landmark model

└── data/

├── raw/rawsubset/              # Raw input face images (manual upload required)

├── processed/256x256/          # Preprocessed high-quality images (GT)

├── degraded/256x256/           # Degraded low-quality images (input)

└── compare/                    # Visual comparison examples



```
\## Quick Start

\### Prerequisites

\- Windows 10/11 (64-bit)

\- Conda environment: \`face\_restoration\` (see \`ENV\_SETUP.md\` for setup)

\- Raw face images in \`data/raw/rawsubset/\` (\`.jpg\`/\`.png\` format)

\### One-Click Run (Recommended)

1\. Navigate to project root: \`E:\8016project\`

2\. Double-click \`run\_all\_conda.bat\`

3\. The script will:

&#x20;  \- Activate \`face\_restoration\` Conda environment

&#x20;  \- Run \`preprocess.py\` → \`fix\_04646.py\` → \`degrade.py\` → \`dataset.py\` → \`generate\_examples.py\`

&#x20;  \- Pause on critical errors (non-critical errors show warnings)

\### Manual Run (Step-by-Step)

\`\`\`bash

\# Activate Conda environment

conda activate face\_restoration

\# Navigate to project root

cd E:\8016project

\# Run pipeline in sequence

python scripts/preprocess.py

python scripts/fix\_04646.py

python scripts/degrade.py

python scripts/dataset.py

python scripts/generate\_examples.py
```

## Output Directories



| Directory                | Content                                    |
| ------------------------ | ------------------------------------------ |
| `data/processed/256x256` | 256x256 high-quality faces (training GT)   |
| `data/degraded/256x256`  | 256x256 low-quality faces (training input) |
| `data/compare`           | Side-by-side comparison examples           |

## Critical Notes



1. **Filename Consistency**: Keep filenames identical across `raw/`/`processed/`/`degraded/` (1:1 pairing required for training)

2. **04646.jpg Fix**: `fix_04646.py` is mandatory (MTCNN fails to detect face in 04646.jpg)

3. **No Manual Edits**: Do not modify files in `processed/`/`degraded/` (preserve training data consistency)

4. **Dependencies**: See `ENV_SETUP.md` for exact version requirements (Python 3.11.15, OpenCV 4.13.0, etc.)


