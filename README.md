# Setup (from source)
  1. Clone the repo
  2. Create a conda environment conda env create --name retrieval_benchmarking --file=environments.yml
  3. pip install -e .

# Data!!
All *data* can be found at https://drive.google.com/drive/folders/1BJWrocXUzK0MA77SuMCqdF1LrZA56rZj?usp=sharing . Dwnload and drop in data folder.

# Retrievers
|    Name    | Paradigm | More |
|:----------:|:--------:|:----:|
| BM25       | Lexical  | [Link](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) |
| SPLADE     | Sparse   | [Link](https://github.com/naver/splade) |
| DPR        | Dense    | [Link](https://github.com/facebookresearch/DPR) |
| ANCE       | Dense    | [Link](https://github.com/microsoft/ANCE) |
| tas-b      | Dense    | [Link](https://github.com/sebastian-hofstaetter/tas-balanced-dense-retrieval) |
| MPNet      | Dense    | [Link](https://github.com/microsoft/MPNet) |
| Contriever | Dense    | [Link](https://github.com/facebookresearch/contriever) |
| ColBERTv2  | Late-Interaction    | [Link](https://github.com/stanford-futuredata/ColBERT) |



# Project Structure
- data
    - datastructures: Basic data classes for question, answer and others needed in the pipeline.
    - dataloaders: Loaders that take raw json/zip file data and convert them to the format needed in the pipeline
- retriever: Retrievers that take the data loaders and perform retrieval to produce results.
    - dense : dense retrievers like ColBERTv2,ANCE, Contriever, MpNet, DPR and Tas-B
    - lexical: lexical retrievers like BM25
    - sparse: Sparse retrievers like SPLADE
- config: Configuration files with constants and initialization.
- utils: utilities needed in the pipeline like retrieval accuracy calculation and matching.

# Running Evaluation for Results in report
All evaluation scripts dataset wise can be found in the evaluation folder. Just run the files directly.

## Example

configure project root directory to PYTHONPATH variable
```
export PYTHONPATH=/path


export huggingface_token = <your huggingface token to access llama2  >

```

If you are using Elasticsearch (ES) installation >8 please export the following values based on your ES setup

```
export ca_certs = <path to http_ca.crt path in your ES installation>

export elastic_password = <your elasticsearch password>
```
## To reproduce dpr results run
```
python3 src/evaluation/run_dpr_inference.py
```

# Building your own custom dataset

You can quickly build your own dataset in three steps:

### 1) Loading the question, answer and evidence records

The base data loader by default takes a json file of the format

```
[{'id':'..','question':'..','answer':'..'}]
```
Each of the train, test and val splits should under their own json files named under your dir
- /dir_path/train.json
- /dir_path/test.json
- /dir_path/validation.json
  
If you want to create your custom loader:
Within the directory data/dataloaders, Create your Dataloader by extending from BaseDataLoader
```python

class MyDataLoader(BaseDataLoader):
    def load_raw_dataset(self,split):
        dataset = self.load_json(split)
        
        records =  '''your code to transform the elements in json to List[Sample(idx:str,question:Question,answer:Answer,evidence:Evidence)]'''
        # If needed you can also extend from Question,Answer and Evidence dataclasses to form your own types
        self.raw_data = records
    def load_tokenized(self):
        ''' If required overwrite this function to build custom tkenization method of your dataset '''

```
