# Makefile
# ------------------------------------------------------------------------------
# GLOBALS
# ------------------------------------------------------------------------------
PROJECT_NAME = acled_ukraine
PYTHON_VERSION = 3.12.7
PYTHON_INTERPRETER = python
VENV_DIR = acled_venv
CONDA_ENV_NAME = acled_conda
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_DIRECTORY := $(abspath $(MAKEFILE_DIR))

############################## Training Globals ################################

# Define variables for looping
OUTCOMES = log_fatalities
PIPELINES = orig orig_rfe
SCORING = r2
PRETRAINED ?= 0  # 0 if you want to train the models, 1 if calibrate pretrained

############################# Production Globals ###############################

# Model outcome variable used in production 
EXPLAN_OUTCOME = log_fatalities # explainer outcome variable
PROD_OUTCOME = log_fatalities # production outcome variable

# ------------------------------------------------------------------------------
# COMMANDS
# ------------------------------------------------------------------------------
.PHONY: init_config
init_config:
	@CURRENT_DIR=$$(sed -n 's/^PROJECT_DIRECTORY = //p' Makefile); \
	\
	read -p "Enter project name: " project_name; \
	read -p "Enter Python version (e.g., 3.10.12): " python_version; \
	read -p "Enter Python interpreter (default: python): " python_interpreter; \
	read -p "Enter virtual environment directory name: " venv_dir; \
	read -p "Enter conda environment name: " conda_env; \
	python_interpreter=$${python_interpreter:-python}; \
	\
	if [ -d "$$CURRENT_DIR" ] && [ "$$CURRENT_DIR" != "$$project_name" ]; then \
		mv "$$CURRENT_DIR" "$$project_name"; \
	fi; \
	\
	# Cross-platform sed command (works on both macOS and Linux) \
	if [ "$$(uname)" = "Darwin" ]; then \
		sed -i '' \
			-e "s/^PROJECT_NAME = .*/PROJECT_NAME = $${project_name}/" \
			-e "s/^PYTHON_VERSION = .*/PYTHON_VERSION = $${python_version}/" \
			-e "s/^PYTHON_INTERPRETER = .*/PYTHON_INTERPRETER = $${python_interpreter}/" \
			-e "s/^VENV_DIR = .*/VENV_DIR = $${venv_dir}/" \
			-e "s/^CONDA_ENV_NAME = .*/CONDA_ENV_NAME = $${conda_env}/" \
			-e "s|^PROJECT_DIRECTORY = .*|PROJECT_DIRECTORY = $${project_name}|" \
			Makefile; \
	else \
		sed -i \
			-e "s/^PROJECT_NAME = .*/PROJECT_NAME = $${project_name}/" \
			-e "s/^PYTHON_VERSION = .*/PYTHON_VERSION = $${python_version}/" \
			-e "s/^PYTHON_INTERPRETER = .*/PYTHON_INTERPRETER = $${python_interpreter}/" \
			-e "s/^VENV_DIR = .*/VENV_DIR = $${venv_dir}/" \
			-e "s/^CONDA_ENV_NAME = .*/CONDA_ENV_NAME = $${conda_env}/" \
			-e "s|^PROJECT_DIRECTORY = .*|PROJECT_DIRECTORY = $${project_name}|" \
			Makefile; \
	fi; \
	\
	# Replace project name in Python files and other text files only \
	if [ "$$(uname)" = "Darwin" ]; then \
		find "./$$project_name" -type f \( -name "*.py" -o -name "*.txt" -o -name "*.md" -o -name "*.yaml" -o -name "*.json" \) -exec sed -i '' "s/$$CURRENT_DIR/$$project_name/g" {} \;; \
	else \
		find "./$$project_name" -type f \( -name "*.py" -o -name "*.txt" -o -name "*.md" -o -name "*.yaml" -o -name "*.json" \) -exec sed -i "s/$$CURRENT_DIR/$$project_name/g" {} \;; \
	fi; \
	\
	echo "Configuration updated successfully. Folder '$$CURRENT_DIR' -> '$$project_name'."

.PHONY: check_vars
check_vars:
	@echo "Dummy configuration detected."
	@echo ""
	@echo "Please update the following variables in your Makefile before proceeding:"
	@echo " - PROJECT_NAME"
	@echo " - PYTHON_VERSION"
	@echo " - VENV_DIR"
	@echo " - CONDA_ENV_NAME"
	@echo " - OUTCOMES"
	@echo " - PIPELINES"
	@echo " - SCORING"
	@echo " - EXPLAN_OUTCOME"
	@echo " - PROD_OUTCOME"
	@echo ""
	@echo "Once you've replaced the dummy values, you can run your full pipeline commands safely."

## Set up python interpreter environment
create_conda_env:
	@echo "Run 'conda create -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION)' to create conda environment"

## Activate the conda environment
activate_conda_env:
	@echo "Run 'conda activate $(CONDA_ENV_NAME)' to activate the conda environment"
	
# Target to create a virtual environment
create_venv:
	# Create the virtual environment using the specified Python version
	$(PYTHON_INTERPRETER) -m venv $(VENV_DIR)
	@echo "Virtual environment created with $(PYTHON_INTERPRETER)$(PYTHON_VERSION)"

# Target to activate the virtual environment (Unix-based systems)
activate_venv:
	@echo "Run 'conda deactivate' to deactivate the $(CONDA_ENV_NAME) conda environment"
	@echo "Run 'source $(VENV_DIR)/bin/activate' to activate the virtual environment"

# Target to clean the virtual environment
clean_venv:
	rm -rf $(VENV_DIR)
	@echo "Virtual environment removed"

## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


#################################################################################
# Instantiate MLFlow                                                            #
#################################################################################

.PHONY: mlflow_ui
mlflow_ui:
	mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5501

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
# clean directories
clean_dir:
	@echo "Cleaning directory..."
	rm -rf data/


# Folder Creation 

.PHONY: create_folders
create_folders:
# Create data subdirectories
	mkdir -p data/external data/interim data/processed data/raw data/processed/inference
	mkdir -p modeling
	mkdir -p preprocessing
	mkdir -p images/png_images images/svg_images images/html_images
	touch data/interim/.gitkeep
	touch data/processed/.gitkeep
	touch data/processed/inference/.gitkeep
	touch modeling/__init__.py
	touch preprocessing/__init__.py

# Create models subdirectories for each outcome
	@for outcome in $(OUTCOMES); do \
		mkdir -p models/results/$$outcome; \
		mkdir -p models/eval/$$outcome; \
	done

################################################################################
################################### Training ###################################
####################### Preprocessing (+) Dataprep Pipeline ####################
################################################################################

.PHONY: data_gen
data_gen:
	$(PYTHON_INTERPRETER) preprocessing/data_gen.py \
		--input-data-file "./data/raw/ACLED Data_2026-01-02.csv" \
		--output-data-file "./data/raw/acled_ukraine_data_2026_01_02.parquet" \
		2>&1 | tee ./data/raw/data_gen.txt

.PHONY: temporal_splits
temporal_splits:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/temporal_splits.py \
		--input-data-file ./data/raw/acled_ukraine_data_2026_01_02.parquet \
		--data-path ./data/processed \
		2>&1 | tee ./data/processed/temporal_splits.txt

.PHONY: data_prep_preprocessing_training
data_prep_preprocessing_training:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/preprocessing.py \
		--input-data-file ./data/raw/acled_ukraine_data_2026_01_02.parquet \
		--output-data-file ./data/processed/df_sans_zero_missing.parquet \
		--stage training \
		--data-path ./data/processed \
		2>&1 | tee ./data/processed/data_prep_preprocessing_training.txt

.PHONY: build_text_base
build_text_base:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/build_text_base.py \
		--input-data-file ./data/raw/acled_ukraine_data_2026_01_02.parquet \
		--output-data-file ./data/processed/text_base.parquet \
		2>&1 | tee ./data/processed/build_text_base.txt

.PHONY: build_text_embeddings
build_text_embeddings:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/build_text_embeddings.py \
		--input-data-file ./data/processed/text_base.parquet \
		--output-data-file ./data/processed/text_embeddings.parquet \
		2>&1 | tee ./data/processed/build_text_embeddings.txt

.PHONY: remove_notes_leakage
remove_notes_leakage:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/remove_notes_leakage.py \
		--input-file ./data/processed/text_base.parquet \
		--output-file ./data/processed/text_base_no_leakage.parquet \
		2>&1 | tee ./data/processed/remove_notes_leakage.txt

.PHONY: feat_gen_training
feat_gen_training:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/feat_gen.py \
		--input-data-file ./data/processed/df_sans_zero_missing.parquet \
		--stage training \
		--data-path ./data/processed \
		2>&1 | tee ./data/processed/feat_gen_training.txt


preproc_pipeline: data_gen temporal_splits data_prep_preprocessing_training  \
				  feat_gen_training

################################################################################
################################# Training #####################################
########################## RFE, Imb Learn Models ###############################
################################################################################

### Train Linear Regression 
train_lr:
	@echo "Pretrained is set to: $(PRETRAINED)"
	@for o in $(OUTCOMES); do \
		mkdir -p models/results/$$o; \
		for p in $(PIPELINES); do \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/train.py \
				--model-type lr \
				--pipeline-type $$p \
				--outcome $$o \
				--data-path ./data/processed \
				--pretrained $(PRETRAINED) \
				--scoring $(SCORING) \
				2>&1 | tee models/results/$$o/lr_$$p.txt; \
		done; \
	done

### Train Lasso Regression
train_lasso:
	@echo "Pretrained is set to: $(PRETRAINED)"
	@for o in $(OUTCOMES); do \
		mkdir -p models/results/$$o; \
		for p in $(PIPELINES); do \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/train.py \
				--model-type lasso \
				--pipeline-type $$p \
				--outcome $$o \
				--data-path ./data/processed \
				--pretrained $(PRETRAINED) \
				--scoring $(SCORING) \
				2>&1 | tee models/results/$$o/lasso_$$p.txt; \
		done; \
	done

### Train Ridge Regression
train_ridge:
	@echo "Pretrained is set to: $(PRETRAINED)"
	@for o in $(OUTCOMES); do \
		mkdir -p models/results/$$o; \
		for p in $(PIPELINES); do \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/train.py \
				--model-type ridge \
				--pipeline-type $$p \
				--outcome $$o \
				--data-path ./data/processed \
				--pretrained $(PRETRAINED) \
				--scoring $(SCORING) \
				2>&1 | tee models/results/$$o/ridge_$$p.txt; \
		done; \
	done

### Train XGBoost Regression
train_xgb:
	@echo "Pretrained is set to: $(PRETRAINED)"
	@for o in $(OUTCOMES); do \
		mkdir -p models/results/$$o; \
		for p in $(PIPELINES); do \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/train.py \
				--model-type xgb \
				--pipeline-type $$p \
				--outcome $$o \
				--data-path ./data/processed \
				--pretrained $(PRETRAINED) \
				--scoring $(SCORING) \
				2>&1 | tee models/results/$$o/xgb_$$p.txt; \
		done; \
	done

### Train CatBoost Regression
train_cat:
	@echo "Pretrained is set to: $(PRETRAINED)"
	@for o in $(OUTCOMES); do \
		mkdir -p models/results/$$o; \
		for p in $(PIPELINES); do \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/train.py \
				--model-type cat \
				--pipeline-type $$p \
				--outcome $$o \
				--data-path ./data/processed \
				--pretrained $(PRETRAINED) \
				--scoring $(SCORING) \
				2>&1 | tee models/results/$$o/cat_$$p.txt; \
		done; \
	done

# train_all_models: train_lr train_lasso train_xgb train_cat
train_all_models: train_lr train_lasso train_ridge train_xgb train_cat
################################################################################
############################### Model Evaluation ###############################
################################################################################

### Evaluate Linear Regression
eval_lr:
	@for o in $(OUTCOMES); do \
		mkdir -p models/eval/$$o; \
		for p in $(PIPELINES); do \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/evaluation.py \
				--model-type lr \
				--pipeline-type $$p \
				--outcome $$o \
				--outcome-name $$o \
				--data-path ./data/processed \
				2>&1 | tee models/eval/$$o/eval_lr_$$p.txt; \
		done; \
	done

### Evaluate Lasso Regression
eval_lasso:
	@for o in $(OUTCOMES); do \
		mkdir -p models/eval/$$o; \
		for p in $(PIPELINES); do \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/evaluation.py \
				--model-type lasso \
				--pipeline-type $$p \
				--outcome $$o \
				--outcome-name $$o \
				--data-path ./data/processed \
				2>&1 | tee models/eval/$$o/eval_lasso_$$p.txt; \
		done; \
	done

### Evaluate Ridge Regression
eval_ridge:
	@for o in $(OUTCOMES); do \
		mkdir -p models/eval/$$o; \
		for p in $(PIPELINES); do \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/evaluation.py \
				--model-type ridge \
				--pipeline-type $$p \
				--outcome $$o \
				--outcome-name $$o \
				--data-path ./data/processed \
				2>&1 | tee models/eval/$$o/eval_ridge_$$p.txt; \
		done; \
	done

### Evaluate XGBoost Regression
eval_xgb:
	@for o in $(OUTCOMES); do \
		mkdir -p models/eval/$$o; \
		for p in $(PIPELINES); do \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/evaluation.py \
				--model-type xgb \
				--pipeline-type $$p \
				--outcome $$o \
				--outcome-name $$o \
				--data-path ./data/processed \
				2>&1 | tee models/eval/$$o/eval_xgb_$$p.txt; \
		done; \
	done

### Evaluate CatBoost Regression
eval_cat:
	@for o in $(OUTCOMES); do \
		mkdir -p models/eval/$$o; \
		for p in $(PIPELINES); do \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/evaluation.py \
				--model-type cat \
				--pipeline-type $$p \
				--outcome $$o \
				--outcome-name $$o \
				--data-path ./data/processed \
				2>&1 | tee models/eval/$$o/eval_cat_$$p.txt; \
		done; \
	done
	
eval_all_models: eval_lr eval_lasso eval_ridge eval_xgb eval_cat


################################ Modeling Pipeline #############################
### Shortcut to run full modeling pipeline: training, evaluation
################################################################################
modeling_train_eval_pipeline: train_all_models eval_all_models


### Bootstrap Evaluation: Validation and Test Set
bootstrap_evaluation:
	@for o in $(OUTCOMES); do \
		for s in valid test; do \
			mkdir -p models/bootstrap/$$o; \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/bootstrap_evaluation.py \
				--n-samples 5000 \
				--num-resamples 5000 \
				--split $$s \
				--output models/bootstrap/$$o/bootstrap_results_$$s.csv \
				2>/dev/tty | tee models/bootstrap/$$o/bootstrap_$$s.txt; \
		done; \
	done
################################################################################
#################### Best Model Explainer and Explanations #####################
################################################################################

.PHONY: model_explainer
model_explainer:
	@for outcome in $(EXPLAN_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/explainer.py \
			--outcome $$outcome \
			--metric-name "valid_R2" \
			--mode max \
			2>&1 | tee ./data/processed/model_explainer_$$outcome.txt; \
	done

.PHONY: model_explanations_training
model_explanations_training:
	@for outcome in $(EXPLAN_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/explanations_training.py \
			--features-path ./data/processed/X_test.parquet \
			--outcome $$outcome \
			--metric-name "valid_R2" \
			--mode max \
			--top-n 5 \
			--shap-val-flag 1 \
			--residual-threshold 5 \
			--explanations-path ./data/processed/shap_predictions_$$outcome.csv \
			2>&1 | tee ./data/processed/model_explanations_training_$$outcome.txt; \
	done
	
model_explaining_training: model_explainer model_explanations_training


################################################################################
################################# Production ###################################
############################### Model Predict ##################################
################################################################################

.PHONY: data_prep_preprocessing_inference
data_prep_preprocessing_inference:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/preprocessing.py \
		--input-data-file ./data/raw/acled_ukraine_data_2026_01_02.parquet \
		--output-data-file ./data/processed/inference/df_inference_process.parquet \
		--stage inference \
		--data-path ./data/processed \
		2>&1 | tee ./data/processed/inference/data_prep_preprocessing_inference.txt

.PHONY: feat_gen_inference
feat_gen_inference:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/feat_gen.py \
		--input-data-file ./data/processed/inference/df_inference_process.parquet \
		--stage inference \
		--data-path ./data/processed/inference \
		2>&1 | tee ./data/processed/inference/feat_gen_inference.txt

.PHONY: predict
predict:
	@for outcome in $(PROD_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/predict.py \
			--input-data-file data/processed/inference/X.parquet \
			--predictions-path ./data/processed/inference/predictions_$$outcome.csv \
			--outcome $$outcome \
			--metric-name "valid_R2" \
			--mode max \
			2>&1 | tee ./data/processed/inference/predict_$$outcome.txt; \
	done

.PHONY: model_explainer_inference
model_explainer_inference:
	@for outcome in $(EXPLAN_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/explainer.py \
			--outcome $$outcome \
			--metric-name "valid_R2" \
			--mode max; \
	done

.PHONY: model_explanations_inference
model_explanations_inference:
	@for outcome in $(EXPLAN_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/explanations_inference.py \
			--features-path ./data/processed/inference/X.parquet \
			--outcome $$outcome \
			--metric-name "test_R2" \
			--mode max \
			--top-n 5 \
			--shap-val-flag 1 \
			--explanations-path ./data/processed/inference/shap_predictions_$$outcome.csv \
			2>&1 | tee ./data/processed/inference/model_explanations_$$outcome.txt; \
	done
preproc_pipeline_inference: data_prep_preprocessing_inference \
    feat_gen_inference \
    predict \
	model_explainer_inference \
    model_explanations_inference
#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)