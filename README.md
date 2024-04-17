# What Causes the Failure of Explicit to Implicit Discourse Relation Recognition?
Code for the NAACL 2024 paper "[What Causes the Failure of Explicit to Implicit Discourse Relation Recognition?](https://arxiv.org/pdf/2404.00999.pdf)"

If any questions, please contact the email: willie1206@163.com

## 1. Requirement
Our working environment is Python 3.8. Before you run the code, please make sure you have installed all the required packages. You can achieve it by simply execute the shell as `sh requirements.sh`

Then you need to download roberta-base from [here](https://huggingface.co/roberta-base/tree/main), and put it in a local folder. In my case, I put it in "/hits/basement/nlp/liuwi/resources/pretrained_models". Please note, if you use a different path from mine, you may need to modify the path string in the code.

## 2. Data and Preprocessing
**For PDTB 2.0 and PDTB 3.0**

Please refer to the preprocessing.py in [ConnRel](https://github.com/liuwei1206/ConnRel) repository.

During this project, we annotate a small number of examples, which you can find in "data/dataset/anno_100".

**For Gum dataset**

Since the Gum dataset is publicly available, we release the processed corpus in "data/dataset/gum7".

## 3. Run
**For E2I and I2I baselines**

Simply do `sh scripts/run_E2I.sh` or `sh scripts/run_I2I.sh`. Please choose the dataset you want to run and comment other commands in the shell file.

**For Two Strategies**

1. Prepare predictions and vectors when the input contains and does not contain a connective. You can do `sh scripts/run_kfold_base.sh` to achieve so.
2. Use noisy filtering and joint training with connectives to improve the E2I baseline. Run the command `sh scripts/run_filter_joint.sh`. Please make sure step 1 is finished before step 2.

## 4. Citation

You can cita this paper through:
```
@misc{liu2024causes,
      title={What Causes the Failure of Explicit to Implicit Discourse Relation Recognition?}, 
      author={Wei Liu and Stephen Wan and Michael Strube},
      year={2024},
      eprint={2404.00999},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



