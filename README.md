# CEHMR
The code for the paper *CEHMR: Curriculum learning enhanced hierarchical multi-label classification for medication recommendation*.

For reproduction of our medication recommendation results in our [paper](https://doi.org/10.1016/j.artmed.2023.102613 ), see instructions below.

## Data processing
We follows the data processing of [GAMENet](https://github.com/sjy1203/GAMENet?tab=readme-ov-file) and add two additional data processing procedures. The procedures are as follows:
- Extract patient data from MIMIC-III database:
  1. download [MIMIC data](https://mimic.mit.edu/docs/gettingstarted/ ) and put DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv in ./data/
  2. download [DDI data](https://www.dropbox.com/s/8os4pd2zmp2jemd/drug-DDI.csv?e=1&dl=0 ) and put it in ./data/
  3. run the following code
    ```
    cd ./tools
    python mimic_transform.py
    ```
  This step will generate records_final.pkl in the input data with four dimension (patient_idx, visit_idx, medical modal, medical id) where medical model equals 3 made of diagnosis, procedure and drug.
  
- Building the hierarchical structure of medications
  run the following code:
  ```
  python get_hire_data.py
  ```
  This step will generate the directory ./data/hire_data, data information are as follows:
  - med_h.pkl: a tuple containing the ATC-2 and ATC-1 vocabulary;
  - data_train.pkl/data_eval.pkl/data_test.pkl: the train/eval/test data;
  
## Measuring the difficulties of training samples
To make the following code work, you should first remove the provided ./data/hire_data/. When you first run the train_CEHMR.py, CEHMR is trained by all training samples without curriculum learning strategy. The pretrained model weights are saved in ./baselines/saved/[model_name]. To measure difficulties of training samples, please manually copy the best model weights to the directory ./baselines/saved/pretrained/ and rename it as pretrained_model.model. Then, run the following command:
```
cd baselines
python train_CEHMR.py --CL_Train --lr 0.0006
```
When the program ends, a new data file ./data/hire_data/data_train_plus.pkl will be generated.

## Run training
When ./data/hire_data/data_train_plus.pkl is generated, the curriculum learning strategy is utilized for training and run the following command:
```
cd baselines
python train_CEHMR.py --CL_Train --lr 0.0006
```

## Run testing
```
cd baselines
python train_CEHMR.py --eval --weight_path [weight_path]
```

## Cite
Please cite our paper if you use this code in your own work:

```
@article{sun2023cehmr,
  title={CEHMR: Curriculum learning enhanced hierarchical multi-label classification for medication recommendation},
  author={Sun, Mengxuan and Niu, Jinghao and Yang, Xuebing and Gu, Yifan and Zhang, Wensheng},
  journal={Artificial Intelligence in Medicine},
  volume={143},
  pages={102613},
  year={2023},
  publisher={Elsevier}
}
```


