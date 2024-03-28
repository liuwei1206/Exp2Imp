# What Causes the Failure of Explicit to Implicit Discourse Relation Recognition?
Code for the NAACL 2024 paper "What Causes the Failure of Explicit to Implicit Discourse Relation Recognition?"

If any questions, please contact the email: willie1206@163.com

## 1. Requirement
Our working environment is Python 3.8. Before you run the code, please make sure you have installed all the required packages. You can achieve it by simply execute the shell as `sh requirements.sh`

Then you need to download roberta-base from [here](https://huggingface.co/roberta-base/tree/main), and put it in a local folder. In my case, I put it in "/hits/basement/nlp/liuwi/resources/pretrained_models". Please note, if you use a different path from mine, you may need to modify the path string in the code.

## 2. Data and Preprocessing
**For PDTB 2.0 and PDTB 3.0**

Please refer to the preprocessing.py in [ConnRel](https://github.com/liuwei1206/ConnRel) repository.

**For Gum dataset**

