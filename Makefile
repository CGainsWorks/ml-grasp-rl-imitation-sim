# Learned grasping in simulation.
#
# Three classes of target, and the distinction matters when reading the README:
#
#   FAST      seconds to a couple of minutes, no GPU, no display. Everything CI
#             runs, plus the scripted-expert demo.
#               make test  make lint  make demo  make demos
#
#   TRAINING  minutes to hours of CPU. These produce every learned number the
#             README quotes. `make experiments` is the whole grid and takes
#             roughly two hours on eight cores.
#               make train-quick  make bc  make dagger  make experiments
#               make export
#
#   RENDER    needs a GL context, so not in CI.
#               make videos
#
# `make check` runs the FAST class and is what CI does.

SHELL := /bin/sh
PYTHON ?= python
SEEDS ?= 0 1 2 3 4
LEVELS ?= none low medium high
STEPS ?= 200000
EVAL_EPISODES ?= 100
DEMO_EPISODES ?= 200
JOBS ?= 6
RUNS := experiments/runs
RESULTS := experiments/results
DEMOS := demonstrations/expert_low.npz

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Fast
# ---------------------------------------------------------------------------

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: check
check: lint test  ## Everything CI runs

.PHONY: test
test:  ## Unit tests (no training, no GL)
	$(PYTHON) -m pytest tests/ -q

.PHONY: lint
lint:  ## Formatting and obvious errors
	$(PYTHON) -m flake8 src/ envs/ tests/ analysis/ experiments/ scripts/

.PHONY: demo
demo:  ## Scripted expert on the nominal world, printing success and reward terms
	$(PYTHON) scripts/offline_demo.py

.PHONY: demos
demos: $(DEMOS)  ## Record expert demonstrations

$(DEMOS):
	$(PYTHON) src/record_demos.py --episodes $(DEMO_EPISODES) --randomisation low \
		--seed 7 --output $(DEMOS)

.PHONY: expert-baseline
expert-baseline:  ## Scripted expert success rate on every level, with intervals
	$(PYTHON) experiments/expert_baseline.py --episodes $(EVAL_EPISODES) \
		--output $(RESULTS)/expert_baseline.json

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

.PHONY: train-quick
train-quick:  ## One short SAC run (30k steps) to prove the loop works
	$(PYTHON) src/train_rl.py --steps 30000 --randomisation none --seed 0 \
		--eval-every 5000 --eval-episodes 20 --output $(RUNS)/quick

.PHONY: train
train:  ## One full SAC run: make train LEVEL=medium SEED=0
	$(PYTHON) src/train_rl.py --steps $(STEPS) --randomisation $(LEVEL) --seed $(SEED) \
		--hidden 128 --eval-every 10000 --eval-episodes 30 \
		--output $(RUNS)/sac_$(LEVEL)_s$(SEED)

.PHONY: bc
bc: $(DEMOS)  ## Behaviour cloning, one run per seed
	@for s in $(SEEDS); do \
		$(PYTHON) src/train_il.py --demos $(DEMOS) --seed $$s --epochs 60 \
			--randomisation low --output $(RUNS)/bc_s$$s --quiet; \
	done

.PHONY: dagger
dagger: $(DEMOS)  ## DAgger on top of behaviour cloning, one run per seed
	@for s in $(SEEDS); do \
		$(PYTHON) src/train_il.py --demos $(DEMOS) --seed $$s --epochs 60 --dagger \
			--randomisation shifted --dagger-rounds 5 \
			--output $(RUNS)/dagger_s$$s --quiet; \
	done

.PHONY: data
data: $(DEMOS)  ## Every dataset the repository needs but does not ship (~10 min)
	$(PYTHON) src/record_demos.py --episodes 200 --randomisation low --task place --output demonstrations/expert_place_low.npz
	$(PYTHON) src/record_demos.py --episodes 200 --randomisation low --arm --output demonstrations/expert_arm_low.npz
	$(PYTHON) src/record_demos.py --episodes 200 --randomisation none --arm --expert-noise 0.01 --output demonstrations/expert_arm_none.npz
	$(PYTHON) src/record_demos.py --episodes 200 --randomisation low --handled --output demonstrations/expert_handled_low.npz
	$(PYTHON) scripts/collect_pose_data.py --episodes 200
	$(PYTHON) scripts/collect_pose_data.py --episodes 200 --camera wrist_cam --clutter 3 --output experiments/perception/pose_data_wrist.npz

.PHONY: experiments
experiments: $(DEMOS)  ## The whole grid: ablation, imitation, imitation+RL (hours)
	$(PYTHON) experiments/run_all.py --jobs $(JOBS) --steps $(STEPS) --seeds $(SEEDS) 		--levels $(LEVELS) --bcrl-levels $(LEVELS)

.PHONY: ablation
ablation:  ## Aggregate the randomisation ablation from finished runs
	$(PYTHON) experiments/ablation.py --episodes $(EVAL_EPISODES) \
		--output $(RESULTS)/ablation.json

.PHONY: evaluate
evaluate:  ## Headline table from finished runs
	$(PYTHON) experiments/summarise.py --episodes $(EVAL_EPISODES) \
		--output $(RESULTS)/summary.json

.PHONY: export
export:  ## Export a policy for deployment: make export RUN=experiments/runs/bcrl_medium_s0
	$(PYTHON) src/export_policy.py --run $(RUN)

.PHONY: plots
plots:  ## Regenerate every figure in docs/plots
	$(PYTHON) analysis/plots.py --all

.PHONY: readme
readme:  ## Rewrite the README results tables from experiments/results
	$(PYTHON) analysis/readme_tables.py

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

.PHONY: videos
videos:  ## Rollout GIFs and MP4s (needs a GL context)
	$(PYTHON) scripts/make_videos.py

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

.PHONY: clean
clean:  ## Remove generated runs, results and videos
	rm -rf $(RUNS) $(RESULTS) experiments/logs videos/*.mp4
	find . -name __pycache__ -type d -exec rm -rf {} +

.PHONY: docker-build
docker-build:  ## Build the container
	docker build -f docker/Dockerfile -t ml-grasp-rl-imitation-sim .

.PHONY: docker-demo
docker-demo:  ## Run the offline demo in the container
	docker compose -f docker/docker-compose.yml run --rm grasp
