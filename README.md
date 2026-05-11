# DCU

## Install
```
pip install -r requirements.txt
```
* LDM from the official [latent-diffusion](https://github.com/CompVis/latent-diffusion) repository
* CLIP from the official [CLIP](https://github.com/openai/CLIP) repository
* Bert from the official [huggingface](https://huggingface.co/google-bert/bert-base-uncased) model


## Run
You can use gen_round_topologies from topological_methods.py in the utils module to generate custom topologies or use the topologies we have generated.

Because the process is separable, prototypes can be extracted and synthetic data can be generated first,
```
python ./tool_syn_cifar.py
```
Then,
```
python ./main.py
```

