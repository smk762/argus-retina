UV ?= uv

.PHONY: help install lint format test build clean weights images

IMAGES ?= nudenet erax
NUDENET_640M_SHA256 := 04fe3d77980780c1f8297dc6d7f942fd5b3abe6942a188f742a85241e4f634eb
NUDENET_640M := images/nudenet/weights/640m.onnx

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create venv and install with test extras
	$(UV) venv
	$(UV) pip install -e ".[dev]"

lint:  ## Ruff check + format check
	$(UV)x ruff@0.15.16 check src/ tests/
	$(UV)x ruff@0.15.16 format --check src/ tests/

format:  ## Ruff autoformat + autofix
	$(UV)x ruff@0.15.16 format src/ tests/
	$(UV)x ruff@0.15.16 check --fix src/ tests/

test:  ## Run the test suite
	$(UV) run --no-sync pytest --tb=short -q

weights:  ## Fetch the weights the image builds need (NudeNet 640m via gh: GitHub's anonymous download is a login page)
	@mkdir -p images/nudenet/weights
	@echo "$(NUDENET_640M_SHA256)  $(NUDENET_640M)" | sha256sum -c --status 2>/dev/null \
		|| gh release download v3.4-weights -R notAI-tech/NudeNet -p 640m.onnx -D images/nudenet/weights --clobber
	@echo "$(NUDENET_640M_SHA256)  $(NUDENET_640M)" | sha256sum -c -

images: weights  ## Build the images locally (IMAGES=nudenet for one)
	@for m in $(IMAGES); do \
		docker build -f images/$$m/Dockerfile -t argus-retina-$$m:dev \
			--build-arg VERSION=0.0.0+local --build-arg COMMIT=$$(git rev-parse HEAD) . || exit 1; \
	done

build:  ## Build sdist + wheel
	$(UV) build

clean:  ## Remove build + cache artifacts
	rm -rf dist build .venv *.egg-info .pytest_cache .ruff_cache src/argus_retina/_version.py
