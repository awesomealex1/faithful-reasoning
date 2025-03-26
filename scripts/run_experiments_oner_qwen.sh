conda activate ../env/faithful

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=hotpotqa/context_aware_decoding/oner/qwen2_7b_instruct random_seed=1234
python scripts/main.py experiment=hotpotqa/dola/oner/qwen2_7b_instruct random_seed=1234
python scripts/main.py experiment=hotpotqa/decore_entropy/oner/qwen2_7b_instruct random_seed=1234

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=musique/baseline/oner/qwen2_7b_instruct random_seed=1234
python scripts/main.py experiment=musique/context_aware_decoding/oner/qwen2_7b_instruct random_seed=1234
python scripts/main.py experiment=musique/dola/oner/qwen2_7b_instruct random_seed=1234
python scripts/main.py experiment=musique/decore_entropy/oner/qwen2_7b_instruct random_seed=1234

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=2wiki/baseline/oner/qwen2_7b_instruct random_seed=3782
python scripts/main.py experiment=2wiki/context_aware_decoding/oner/qwen2_7b_instruct random_seed=3782
python scripts/main.py experiment=2wiki/dola/oner/qwen2_7b_instruct random_seed=3782
python scripts/main.py experiment=2wiki/decore_entropy/oner/qwen2_7b_instruct random_seed=3782

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=hotpotqa/baseline/oner/qwen2_7b_instruct random_seed=3782
python scripts/main.py experiment=hotpotqa/context_aware_decoding/oner/qwen2_7b_instruct random_seed=3782
python scripts/main.py experiment=hotpotqa/dola/oner/qwen2_7b_instruct random_seed=3782
python scripts/main.py experiment=hotpotqa/decore_entropy/oner/qwen2_7b_instruct random_seed=3782

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=musique/baseline/oner/qwen2_7b_instruct random_seed=3782
python scripts/main.py experiment=musique/context_aware_decoding/oner/qwen2_7b_instruct random_seed=3782
python scripts/main.py experiment=musique/dola/oner/qwen2_7b_instruct random_seed=3782
python scripts/main.py experiment=musique/decore_entropy/oner/qwen2_7b_instruct random_seed=3782

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=2wiki/baseline/oner/qwen2_7b_instruct random_seed=9539
python scripts/main.py experiment=2wiki/context_aware_decoding/oner/qwen2_7b_instruct random_seed=9539
python scripts/main.py experiment=2wiki/dola/oner/qwen2_7b_instruct random_seed=9539
python scripts/main.py experiment=2wiki/decore_entropy/oner/qwen2_7b_instruct random_seed=9539

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=hotpotqa/baseline/oner/qwen2_7b_instruct random_seed=9539
python scripts/main.py experiment=hotpotqa/context_aware_decoding/oner/qwen2_7b_instruct random_seed=9539
python scripts/main.py experiment=hotpotqa/dola/oner/qwen2_7b_instruct random_seed=9539
python scripts/main.py experiment=hotpotqa/decore_entropy/oner/qwen2_7b_instruct random_seed=9539

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=musique/baseline/oner/qwen2_7b_instruct random_seed=9539
python scripts/main.py experiment=musique/context_aware_decoding/oner/qwen2_7b_instruct random_seed=9539
python scripts/main.py experiment=musique/dola/oner/qwen2_7b_instruct random_seed=9539
python scripts/main.py experiment=musique/decore_entropy/oner/qwen2_7b_instruct random_seed=9539

pkill -f elasticsearch
pkill -f uvicorn