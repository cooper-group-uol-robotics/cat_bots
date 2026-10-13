import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from math import ceil

def smiles_to_pdf(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    name_col: str | None = None,
    out_pdf: str = "ligands.pdf",
    out_svg: str | None = "ligands.svg",
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (250, 250),
    per_page: int = 32,   # e.g., 4x8 grid per page
):
    # Parse SMILES -> molecules
    mols, legends = [], []
    bad_rows = []

    for i, row in df.iterrows():
        smi = str(row[smiles_col]).strip()
        m = Chem.MolFromSmiles(smi)
        if m is None:
            bad_rows.append((i, smi))
            continue
        AllChem.Compute2DCoords(m)
        mols.append(m)
        legends.append(str(row[name_col]) if name_col and pd.notna(row[name_col]) else smi)

    if not mols:
        raise ValueError("No valid SMILES parsed.")

    # Build grid images page-by-page and save to a multi-page PDF
    # (Pillow handles the PDF writing; pages are raster images)
    pages = []
    for start in range(0, len(mols), per_page):
        chunk_mols   = mols[start:start+per_page]
        chunk_legends= legends[start:start+per_page]
        img = Draw.MolsToGridImage(
            chunk_mols,
            legends=chunk_legends,
            molsPerRow=mols_per_row,
            subImgSize=sub_img_size,
            useSVG=False,         # PIL image
            returnPNG=False
        )
        # RDKit returns a PIL Image in RGBA; convert for PDF
        pages.append(img.convert("RGB"))

    # Save multi-page PDF
    if len(pages) == 1:
        pages[0].save(out_pdf)
    else:
        pages[0].save(out_pdf, save_all=True, append_images=pages[1:])

    # Optional: also write a single-page SVG (vector). If many molecules,
    # consider increasing page size or lowering sub_img_size.
    if out_svg:
        svg = Draw.MolsToGridImage(
            mols,
            legends=legends,
            molsPerRow=mols_per_row,
            subImgSize=sub_img_size,
            useSVG=True
        )
        # RDKit might return an SVG string or an object with .data
        svg_text = getattr(svg, "data", svg)
        with open(out_svg, "w", encoding="utf-8") as f:
            f.write(svg_text)

    if bad_rows:
        print(f"Skipped {len(bad_rows)} invalid SMILES:")
        for i, smi in bad_rows[:10]:
            print(f"  row {i}: {smi}")
        if len(bad_rows) > 10:
            print("  ...")


if __name__ == "__main__":
    # Example dataframe; replace with your own
    ligs_60 = pd.read_csv("../data/ligands/ks60.csv")
    df = ligs_60.copy()

#    df = ligs_60.dropna(subset=["location"])
    smiles_to_pdf(
        df,
        smiles_col="SMILES",
        name_col="CAS",
        out_pdf="ligands.pdf",
        out_svg="ligands.svg",
        mols_per_row=3,
        sub_img_size=(250, 250),
        per_page=32
    )
    print("Wrote ligands.pdf and ligands.svg")




