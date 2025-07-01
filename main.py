import os
from absl import logging
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from modules.components import init_components
from modules.pipeline import init_pipeline

PIPELANE_NAME = "marketing-classification-pipeline"

# Pipeline inputs
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/transform.py"
TUNER_MODULE_FILE = "modules/tuner.py"
TRAINER_MODULE_FILE = "modules/trainer.py"

# Pipeline outputs
OUTPUT_BASE = "outputs"

SERVING_MODEL_DIR = os.path.join(OUTPUT_BASE, "serving_model")
PIPELINE_ROOT = os.path.join(OUTPUT_BASE, PIPELANE_NAME)
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")

components = init_components(
        DATA_ROOT,
        training_module=TRAINER_MODULE_FILE,
        transform_module=TRANSFORM_MODULE_FILE,
        training_steps=1000,
        tuner_module=TUNER_MODULE_FILE,
        eval_steps=1000,
        serving_model_dir=SERVING_MODEL_DIR,
    )
    
pipeline = init_pipeline(pipeline_root=PIPELINE_ROOT, pipeline_name=PIPELANE_NAME, metadata_path=METADATA_PATH, components=components)
BeamDagRunner().run(pipeline=pipeline)



