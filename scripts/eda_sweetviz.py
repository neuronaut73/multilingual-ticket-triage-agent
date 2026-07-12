from pathlib import Path

import pandas as pd
import sweetviz as sv


DATA_PATH = Path("data/aa_dataset-tickets-multi-lang-5-2-50-version.csv")
OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # General report
    report = sv.analyze(df)
    report.show_html(str(OUTPUT_DIR / "sweetviz_full_report.html"), open_browser=False)

    # Target-focused reports
    for target in ["queue", "priority", "type"]:
        if target in df.columns:
            target_report = sv.analyze(df, target_feat=target)
            target_report.show_html(
                str(OUTPUT_DIR / f"sweetviz_target_{target}.html"),
                open_browser=False,
            )

    print(f"Sweetviz reports written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()