import base64
import io
from pathlib import Path
import pandas as pd
from PIL import Image


def create_html_table(df: pd.DataFrame) -> str:
    if "file_name" not in df.columns:
        raise ValueError("DataFrame must contain 'file_name'")

    html_table = "<table><tr>"
    for col in df.columns:
        html_table += f"<th>{col}</th>"
    html_table += "</tr>"

    for index, row in df.iterrows():
        html_table += "<tr>"
        for col in df.columns:
            if col == "Image_Path":
                try:
                    # Open the image file
                    # todo: doesn't work.... works in alex's original code
                    img = Image.open(row[col])

                    # Convert the image to base64
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")  # Or another suitable format
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                    # Embed the image in the HTML table
                    html_table += f"<td><img src='data:image/png;base64,{img_str}' width='400'></td>"  # Adjust width as needed.
                except FileNotFoundError:
                    html_table += "<td>Image not found</td>"
                except Exception as e:
                    html_table += f"<td>Error loading image: {e}</td>"
            else:
                html_table += f"<td>{row[col]}</td>"
        html_table += "</tr>"
    html_table += "</table>"
    return html_table


def save_html_table(
    html_string: str, filename="output_table.html", dir_path=Path("data")
) -> None:
    """Saves an HTML string to a file."""
    dir_path.mkdir(exist_ok=True)
    filepath = dir_path / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_string)
