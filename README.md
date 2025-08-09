# MSGL
Model Selection-inspired Graph Learning

## IGBT Plot Data Extraction

Place the input image at `./data/igbt_output_characteristic.png` (create the `data` directory if needed), then run:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/extract_plot_data.py --image /workspace/data/igbt_output_characteristic.png --xlim 0 3.6 --ylim 0 3000 --outdir /workspace/outputs
```

Outputs:
- `outputs/curve_Tj25C.csv`
- `outputs/curve_Tj125C.csv`
- `outputs/curve_Tj175C.csv`
- `outputs/extracted_overlay.png` (overlay on the original plot)
- `outputs/extracted_curves.png` (clean data plot)

Adjust `--xlim` and `--ylim` if the axis limits differ for other images.
