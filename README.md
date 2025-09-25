# Long Language Model (From Scratch)

Long Language Model from SCRATCH!! <br/>
The idea of this project is to build LLM from scratch; better understand the overall algorithm and knowledge on how LLM works in details. <br/>
Same as how ChatGPT, Grok, Llama, etc., you ask a question (with some information) and the AI model will answer the question.

# Installation
Just download this project. <br/>
The project itself contains installation guide.

# Brief contents
## 1. Making tokenizer
After this phase, it will create following files: <br/><br/>
<img src="./readme_materials/bpe_tokenizer.png" width="300"/>

## 2. Preparing datasets for Pretraining
Building datasets for pretraining. <br/>
- This uses 75000+ Gutenburg books
- We will perform concatination/cleanup/tokenization in this step

## 3. Making model&dataset pipeline
In this phase, we will create 3 files (note that those are in python_files folder of this project)
- `Dummy_Model.py`
- `npy_datasets.py`
- `train.py`

## 4. Full pretraining
We will run pretraining. <br/>
Below is the final graph for pretraining.<br/><br/>
<img src="./readme_materials/pretrain_graph.png" width="300"/>

## 5. Preparing datasets for Supervised Fine Tuning
In this phase, we will use datasets from 11 libraries SFT training.
- 10 libraries from hugging face
    - installation guide is covered in `8_Supervised_finetuning_Data_Schema/0_before_we_start.ipynb`
- 1 from release (I made that)
## 6. Supervised Fine Tuning training
W will run SFT training. <br/>
Below is the final graph for finetuning. <br/><br/>
<img src="./readme_materials/finetuning_graph.png" width="300"/>

## 7. Sampling
With configuration&cleanup, we'll test out the trained model. <br/>
Below are the examples of output. <br/>
<img src="./readme_materials/ai_answer_1.png" width="800"/><br/>
<img src="./readme_materials/ai_answer_2.png" width="800"/>