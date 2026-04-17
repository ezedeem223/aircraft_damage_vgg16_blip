# Models Directory

Place trained checkpoints for the VGG16 classifier in this directory.

## Expected Checkpoint

- Default filename referenced by the configs: `models/vgg16_aircraft_damage.keras`

## Notes

- No fine-tuned checkpoint is included in the repository.
- The first run of `python scripts/run_train.py --config configs/train.yaml --download-data` will train a new checkpoint and save it here.
- `scripts/run_predict.py`, `scripts/run_evaluate.py`, and the Gradio demo will fail gracefully if the checkpoint is missing.

## BLIP Assets

BLIP weights are not stored in this directory. They are downloaded by Hugging Face when report generation is first used, unless already cached locally.
