# pip install rdkit-pypi pillow pandas
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

def make_ligand_nn_table_pdf(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    label_col: str | None = "CAS",       # shown under each structure (RDKit legend)
    group_col: str = "_group",
    role_col: str = "_role",
    out_pdf: str = "ligand_table.pdf",
    rows_per_page: int = 10,             # how many ligand rows per page
    cell_size: tuple[int, int] = (300, 300),
    cell_pad: int = 24,                  # padding between cells/rows
    header_height: int = 80,
    line_thickness: int = 3,
    header_left: str = "Incompatible ligand",
    header_right: str = "Top-3 nearest neighbours",
):
    """
    Assumes df contains one 'ligand' and up to 'neighbour1..3' per group (_group).
    Produces a paginated PDF with a 4-column table per row: ligand | neighbour1 | neighbour2 | neighbour3
    """
    role_order = ["ligand", "neighbour1", "neighbour2", "neighbour3"]

    # Keep only the rows we actually place in the table and fix ordering
    d = df[df[role_col].isin(role_order)].copy()
    d["_group_idx"] = d[group_col].astype(str).str.extract(r"(\d+)").astype(int)
    d["_role_order"] = d[role_col].map({r:i for i, r in enumerate(role_order)})
    d = d.sort_values(["_group_idx", "_role_order"], kind="stable")

    # Precompute geometry
    W = 4 * cell_size[0] + 5 * cell_pad
    # header + (rows) * cell + (rows+1)*pad
    H_base = header_height
    row_block_h = cell_size[1] + cell_pad
    page_h = H_base + rows_per_page * row_block_h + cell_pad

    # Column x positions
    x0 = cell_pad
    x1 = x0 + cell_size[0] + cell_pad
    x2 = x1 + cell_size[0] + cell_pad
    x3 = x2 + cell_size[0] + cell_pad
    x_positions = [x0, x1, x2, x3]
    # Vertical divider after the ligand column
    x_divider = x1 - cell_pad // 2

    # Simple font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    def draw_cell(smi: str, label: str | None):
        if pd.isna(smi) or str(smi).strip() == "":
            # blank cell
            img = Image.new("RGB", cell_size, "white")
            return img
        m = Chem.MolFromSmiles(str(smi).strip())
        if m is None:
            img = Image.new("RGB", cell_size, "white")
            dr = ImageDraw.Draw(img)
            dr.text((8, 8), "Invalid SMILES", fill="red", font=small_font)
            return img
        AllChem.Compute2DCoords(m)
        # Legend goes inside the image (bottom-left) and looks like a caption
        legend = str(label) if (label_col and pd.notna(label)) else ""
        img = Draw.MolToImage(m, size=cell_size, legend=legend)
        return img.convert("RGB")

    # Build pages
    pages = []
    groups = [g for _, g in d.groupby(group_col, sort=False)]
    for start in range(0, len(groups), rows_per_page):
        chunk = groups[start:start + rows_per_page]
        page = Image.new("RGB", (W, page_h), "white")
        draw = ImageDraw.Draw(page)

        # Header text
        draw.text((x0, cell_pad // 2), header_left, fill="black", font=font)
        draw.text((x1, cell_pad // 2), header_right, fill="black", font=font)

        # Vertical divider spanning the table area (below header)
        draw.line([(x_divider, header_height), (x_divider, page_h - cell_pad)],
                  fill="black", width=line_thickness)

        y = header_height
        for g in chunk:
            # Compose row images in the required order, pad missing neighbours with blanks
            cells = []
            for role in role_order:
                row = g[g[role_col] == role]
                if not row.empty:
                    r = row.iloc[0]
                    cells.append(draw_cell(r[smiles_col], r[label_col] if label_col else None))
                else:
                    cells.append(Image.new("RGB", cell_size, "white"))

            # Paste the 4 cells
            for col, cell in enumerate(cells):
                page.paste(cell, (x_positions[col], y))

            # Row separator line
            y_sep = y + cell_size[1] + cell_pad // 2
            draw.line([(cell_pad // 2, y_sep), (W - cell_pad // 2, y_sep)],
                      fill="black", width=line_thickness)

            y += row_block_h

        pages.append(page)

    # Save multi-page PDF
    if len(pages) == 1:
        pages[0].save(out_pdf)
    else:
        pages[0].save(out_pdf, save_all=True, append_images=pages[1:])

    print(f"Wrote {out_pdf} ({len(pages)} page{'s' if len(pages)!=1 else ''})")

# ---- Example usage ----
if __name__ == "__main__":
    ligs_60 = pd.read_csv("./data/ligands/nearest_neighbours.csv")
    # (Optional) drop Unnamed columns early
    ligs_60 = ligs_60[[c for c in ligs_60.columns if "unnamed" not in c.lower()]]

    make_ligand_nn_table_pdf(
        ligs_60,
        smiles_col="SMILES",
        label_col="CAS",              # what you want printed below each structure
        group_col="_group",
        role_col="_role",
        out_pdf="neighbours_table.pdf",
        rows_per_page=8,
        cell_size=(320, 320),
        cell_pad=28,
        header_left="Incompatible ligand",
        header_right="Top-3 nearest neighbours",
    )
