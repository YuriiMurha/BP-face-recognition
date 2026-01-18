import json
import re
import os

# load the notebook
with open("notebooks/DeepLearning.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Let's inspect the cells for a sample marker.
cells = nb.get("cells", [])

def parse_marker(cell_source):
    """
    Extract a numeric marker from the first line of cell source if available.
    Returns a tuple (major, minor) if found, otherwise None.
    """
    if not cell_source:
        return None
    first_line = cell_source[0].strip()
    # match patterns like "# 8.", "### 11.1", "# 10.2", etc.
    m = re.match(r"^#+\s*([\d]+(?:\.[\d]+)?)", first_line)
    if m:
        marker_str = m.group(1).strip().rstrip('.')
        # split marker by dot
        parts = marker_str.split(".")
        try:
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            return (major, minor)
        except ValueError:
            return None
    return None

# Process cells: assign each cell a sorting key based on the most recent marker.
sorted_cells = []
current_key = None

for cell in cells:
    source = cell.get("source", [])
    key = parse_marker(source)
    if key is not None:
        current_key = key
        # assign a field for sorting purposes, store it in cell metadata for now
        cell["_sort_key"] = key
    else:
        # if the cell doesn't have a marker and we have a current key, assign that key
        # if no current key, then assign a very low priority (e.g., (0,0))
        cell["_sort_key"] = current_key if current_key is not None else (0,0)
    sorted_cells.append(cell)

# Now, sort the cells by their _sort_key.
sorted_cells = sorted(sorted_cells, key=lambda c: c["_sort_key"])

# Replace the cells in notebook with sorted cells.
nb["cells"] = sorted_cells

# Optionally, remove the temporary _sort_key metadata from each cell
for cell in nb["cells"]:
    if "_sort_key" in cell:
        del cell["_sort_key"]


output_path = "notebooks/DeepLearning_sorted.ipynb"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Write out the sorted notebook to a new file.
with open("notebooks/DeepLearning_sorted.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Sorted notebook written to notebooks/DeepLearning_sorted.ipynb")
