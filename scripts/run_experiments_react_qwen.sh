conda activate ../env/faithful

sleep 14400

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=2wiki/baseline/react/llama3_8b_instruct random_seed=1234
python scripts/main.py experiment=2wiki/context_aware_decoding/react/llama3_8b_instruct random_seed=1234
python scripts/main.py experiment=2wiki/dola/react/llama3_8b_instruct random_seed=1234
python scripts/main.py experiment=2wiki/decore_entropy/react/llama3_8b_instruct random_seed=1234

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=hotpotqa/baseline/react/llama3_8b_instruct random_seed=1234
python scripts/main.py experiment=hotpotqa/context_aware_decoding/react/llama3_8b_instruct random_seed=1234
python scripts/main.py experiment=hotpotqa/dola/react/llama3_8b_instruct random_seed=1234
python scripts/main.py experiment=hotpotqa/decore_entropy/react/llama3_8b_instruct random_seed=1234

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=musique/baseline/react/llama3_8b_instruct random_seed=1234
python scripts/main.py experiment=musique/context_aware_decoding/react/llama3_8b_instruct random_seed=1234
python scripts/main.py experiment=musique/dola/react/llama3_8b_instruct random_seed=1234
python scripts/main.py experiment=musique/decore_entropy/react/llama3_8b_instruct random_seed=1234

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=2wiki/baseline/react/llama3_8b_instruct random_seed=3782
python scripts/main.py experiment=2wiki/context_aware_decoding/react/llama3_8b_instruct random_seed=3782
python scripts/main.py experiment=2wiki/dola/react/llama3_8b_instruct random_seed=3782
python scripts/main.py experiment=2wiki/decore_entropy/react/llama3_8b_instruct random_seed=3782

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=hotpotqa/baseline/react/llama3_8b_instruct random_seed=3782
python scripts/main.py experiment=hotpotqa/context_aware_decoding/react/llama3_8b_instruct random_seed=3782
python scripts/main.py experiment=hotpotqa/dola/react/llama3_8b_instruct random_seed=3782
python scripts/main.py experiment=hotpotqa/decore_entropy/react/llama3_8b_instruct random_seed=3782

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=musique/baseline/react/llama3_8b_instruct random_seed=3782
python scripts/main.py experiment=musique/context_aware_decoding/react/llama3_8b_instruct random_seed=3782
python scripts/main.py experiment=musique/dola/react/llama3_8b_instruct random_seed=3782
python scripts/main.py experiment=musique/decore_entropy/react/llama3_8b_instruct random_seed=3782

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=2wiki/baseline/react/llama3_8b_instruct random_seed=9539
python scripts/main.py experiment=2wiki/context_aware_decoding/react/llama3_8b_instruct random_seed=9539
python scripts/main.py experiment=2wiki/dola/react/llama3_8b_instruct random_seed=9539
python scripts/main.py experiment=2wiki/decore_entropy/react/llama3_8b_instruct random_seed=9539

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=hotpotqa/baseline/react/llama3_8b_instruct random_seed=9539
python scripts/main.py experiment=hotpotqa/context_aware_decoding/react/llama3_8b_instruct random_seed=9539
python scripts/main.py experiment=hotpotqa/dola/react/llama3_8b_instruct random_seed=9539
python scripts/main.py experiment=hotpotqa/decore_entropy/react/llama3_8b_instruct random_seed=9539

pkill -f elasticsearch
pkill -f uvicorn
sleep 1
./src/utils/elasticsearch-7.10.2/bin/elasticsearch&
python -m uvicorn serve:app --port 8000 --app-dir src/utils/retriever_server&
sleep 5

python scripts/main.py experiment=musique/baseline/react/llama3_8b_instruct random_seed=9539
python scripts/main.py experiment=musique/context_aware_decoding/react/llama3_8b_instruct random_seed=9539
python scripts/main.py experiment=musique/dola/react/llama3_8b_instruct random_seed=9539
python scripts/main.py experiment=musique/decore_entropy/react/llama3_8b_instruct random_seed=9539

pkill -f elasticsearch
pkill -f uvicorn