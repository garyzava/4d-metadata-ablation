#!/usr/bin/env python3
"""Generate 4 markdown docs per BIRD Mini-Dev database."""

from bird_loader.config import OUTPUT_DIR
from bird_loader.loaders import load_tables, load_questions, load_csv_descriptions
from bird_loader import generators


def main():
    tables = load_tables()
    questions = load_questions()

    total_files = 0
    for db_name, db in tables.items():
        db_out = OUTPUT_DIR / db_name
        db_out.mkdir(parents=True, exist_ok=True)

        csv_descs = load_csv_descriptions(db_name)
        db_questions = questions.get(db_name, [])

        docs = {
            "DataDictionary.md": generators.data_dictionary,
            "QueryPatterns.md": generators.query_patterns,
            "BusinessContext.md": generators.business_context,
            "DomainContext.md": generators.domain_context,
        }

        for filename, gen_func in docs.items():
            content = gen_func(db, db_questions, csv_descs)
            (db_out / filename).write_text(content)
            total_files += 1

        print(f"  {db_name}: {len(db_questions)} questions -> {db_out}")

    print(f"\nDone. {total_files} files generated in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
