SHELL := /usr/bin/env bash

PY := python

export TRAIN_LIMIT ?= 300000
export VAL_LIMIT ?= 60000
export BATCH_SIZE ?= 32
export SKIP_PREPROCESS ?= 1

.PHONY: preprocess train-lstm-cnn train-cnn-lstm train-both eval pipeline

preprocess:
	cd iot-intrusion-detection >/dev/null 2>&1 || true
	$(PY) run_complete_pipeline.py

train-lstm-cnn:
	cd iot-intrusion-detection >/dev/null 2>&1 || true
	$(PY) train_lstm_cnn.py

train-cnn-lstm:
	cd iot-intrusion-detection >/dev/null 2>&1 || true
	$(PY) train_cnn_lstm.py

train-both:
	cd iot-intrusion-detection >/dev/null 2>&1 || true
	$(PY) -c "import sys; sys.path.append('src'); from train import train_both_models; train_both_models()"

eval:
	cd iot-intrusion-detection >/dev/null 2>&1 || true
	$(PY) eval_lstm.py || true

pipeline:
	cd iot-intrusion-detection >/dev/null 2>&1 || true
	$(PY) run_complete_pipeline.py


