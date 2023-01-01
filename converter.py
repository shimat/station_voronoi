import csv
import json
from pathlib import Path

out_dir = Path("csv")
out_dir.mkdir(exist_ok=True)

for json_path in Path("geojson").joinpath("train").glob("*.geojson"):
    print(json_path)
    out_path = out_dir / f"{json_path.stem}.csv"
    if out_path.exists():
        continue
    with (json_path.open(encoding="utf-8-sig") as json_file,
          out_path.open("w", encoding="utf-8-sig", newline="") as write_file):
        json_obj = json.load(json_file)
        coordinates = [
            (f["properties"]["name"], *f["geometry"]["coordinates"])
            for f in json_obj["features"] if f["geometry"]["type"] == "Point"]
        csv_writer = csv.writer(write_file, quoting=csv.QUOTE_NONNUMERIC)
        csv_writer.writerows(coordinates)
