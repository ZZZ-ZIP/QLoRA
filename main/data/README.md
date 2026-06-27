# Data

Raw datasets, original videos, extracted image frames, and prepared private JSONL split files are not included in this repository.

Prepare the data locally and place the split files under:

```text
data_four_datasets/
  train.jsonl
  val.jsonl
  test.jsonl
```

Alternatively, edit `data.output_dir` in the YAML config files.

## JSONL Schema

Each line should be one JSON object:

```json
{
  "sample_id": "mosei_000001",
  "dataset": "MOSEI",
  "frame_paths": [
    "frames/mosei_000001/000001.jpg",
    "frames/mosei_000001/000002.jpg"
  ],
  "transcript": "I really enjoyed this experience.",
  "label": "positive",
  "split": "train",
  "subject_id": "speaker_001"
}
```

Required fields:

- `frame_paths`: list of sampled video-frame image paths. Use an empty list for text-only ablation.
- `transcript`: utterance transcript text.
- `label`: one of `positive`, `negative`, `neutral`.

Recommended fields:

- `sample_id`: unique sample identifier.
- `dataset`: source dataset, e.g. `MOSEI`, `MOSI`, `CH-SIMSv2`, `SIMS`.
- `split`: `train`, `val`, or `test`.
- `subject_id`: speaker or video-level grouping identifier if available.

The repository `.gitignore` ignores private JSONL files under `data/` and the default `data_four_datasets/` folder.
