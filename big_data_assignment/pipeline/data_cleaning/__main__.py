from big_data_assignment.pipeline.data_cleaning import run_pipeline

paths = run_pipeline()
print("\n[pipeline] ALL STEPS COMPLETE.")
for split, path in paths.items():
    print(f"  {split}: {path}")
