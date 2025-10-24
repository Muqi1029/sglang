#!/usr/bin/env bash
#
python -m sglang.bench_serving \
    --backend sglang-oai \
    --dataset-name random \
    --num-prompts 1 \
    --random-input 1200 \
    --random-output 800 \
    --max-concurrency 10 \
    --port 8888 \
    --host 0.0.0.0 \
    --dataset-path ${SHAREGPT_DATASET} \
    --profile
