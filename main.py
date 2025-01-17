import os
import getpass
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import pickle
import argparse
import random
# import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import multiprocessing
from util import *
import difflib
os.environ['OPENAI_API_KEY'] = "xxx"
# import logging

# os.environ["http_proxy"] = "http://127.0.0.1:1080"
# os.environ["https_proxy"] = "http://127.0.0.1:1080"
# os.environ["ALL_PROXY"] = "http://127.0.0.1:1080"
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
# os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
seed_everything()

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from uuid import uuid4
from sklearn.model_selection import train_test_split
from openai import OpenAI
client = OpenAI()

parser = argparse.ArgumentParser()
parser.add_argument('--general_retrieve_num',type=int,default=0,help="") # 4 0
parser.add_argument('--dataset',type=str,default="SDSC-SP2-1998-4.2-cln.swf",help="") # HPC2N-2002-2.2-cln.swf KTH-SP2-1996-2.1-cln.swf SDSC-SP2-1998-4.2-cln.swf
parser.add_argument('--user_retrieve_num',type=int,default=7,help="") # 4 0
parser.add_argument('--worker_num',type=int,default=1,help="")
parser.add_argument('--query_numbers',type=int,default=100000,help="") # 100000 100
parser.add_argument('--test_data_size',type=float,default=0.1,help="")
parser.add_argument('--data_partition',type=str,default="real",help="")
parser.add_argument('--data_mode',type=str,default="static",help="") # static runtime
parser.add_argument('--use_script',type=int,default=1,help="") # 0 1 
parser.add_argument('--update_database',type=int,default=1,help="") # 1 0 
parser.add_argument('--diff_contex',type=int,default=1,help="") # 1 0
parser.add_argument('--model',type=str,default="gpt4o",help="") # qwen2.5:0.5b 1.5b 3b 7b 14b 32b 72b # phi3 3.8b phi3:14b # llama3.2 3b llama3.2:1b
parser.add_argument('--reorganize_docs_rate',type=float,default=0.0,help="")
parser.add_argument('--diff_mode',type=str,default="unchange+insert",help="") # unchange unchange+insert insert
args = parser.parse_args()
args = vars(args)

X_train, X_test, y_train, y_test, mid_df = get_tt_data(args)

# # 构建本地文档
X_train["label"] = y_train
X_test["label"] = y_test
cols_meta = get_features_name(args)

def convert_to_json_swf(row):
    metadata = ""
    for col in cols_meta:
        metadata += f"{col}:{row[col]}\n"
    sinput = f"""metadata:{metadata}"""
    soutput = str(row["label"])
    return {"input": sinput,"time_submit":int(row["time_submit"]),"time_end":int(row["time_end"]),"userid":int(row["id_user"]), "prediction": soutput} # "instruction": sinst, 

def convert_to_json_hesi(row):
    metadata = ""
    for col in cols_meta:
        metadata += f"{col}:{row[col]}\n"

    # 添加脚本
    if args['use_script']:
        # 去除注释
        batch_script = row["batch_script"]
        lines = batch_script.split('\n')
        new_lines = []
        for i in lines:
            if len(i)<1: 
                continue
            if i[0] == '#':
                continue
            new_lines.append(i)
        batch_script = "\n".join(new_lines)
        sinput = f"""metadata:{metadata} script:{batch_script}"""
    else:
        sinput = f"""metadata:{metadata}"""
        
    soutput = str(row["label"])
    return {"input": sinput,"time_submit":int(row["time_submit"]),"time_end":int(row["time_end"]),"userid":int(row["id_user"]), "prediction": soutput} # "instruction": sinst, 

if args['dataset'] in ['bd','sk','wm']:
    convert_to_json = convert_to_json_hesi
else:
    convert_to_json = convert_to_json_swf

train_list = X_train.apply(convert_to_json, axis=1).tolist()
test_list = X_test.apply(convert_to_json, axis=1).tolist()
if len(mid_df)>0:
    mid_df = mid_df.apply(convert_to_json, axis=1).tolist()
else:
    mid_df=[]
with open(f"data/{args['dataset']}_train.json", "w") as f:
    json.dump(test_list, f)
with open(f"data/{args['dataset']}_test.json", "w") as f:
    json.dump(test_list, f)
with open(f"data/{args['dataset']}_mid.json", "w") as f:
    json.dump(mid_df, f)
# # exit()

mem_file = f"data/{args['dataset']}_train.json"
mid_file = f"data/{args['dataset']}_mid.json"
test_file = f"data/{args['dataset']}_test.json"
mem_db_path = f"store/vectorstore_{args['dataset']}"

# 构建自定义DocumentLoader
from typing import AsyncIterator, Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
        
def find_numberv2(pred):
    numbers = "0123456789"
    
    for i in range(len(pred)):
        if pred[i] not in numbers:
            continue
        for j in range(len(pred[i:])):
            if pred[i+j] in numbers: # or pred[i+j]=='.'
                continue
            break
        break
    
    if j == len(pred[i:])-1:
        return int(float(pred[i:]))
    else:
        return int(float(pred[i:i+j]))
        
class CustomDocumentLoader(BaseLoader):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        
    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        with open(self.file_path, "r", encoding="utf-8") as f:
            mem = json.load(f)
            line_number = 0
            for line in mem:
                yield Document(
                    page_content=f'{line["input"]}\nprediction:{line["prediction"]}',
                    metadata={"line_number": line_number, "userid":line["userid"] ,"source": self.file_path},
                )
                line_number += 1
                
loader = CustomDocumentLoader(mem_file)
data = loader.load()
# model = ChatOllama(model=args["model"])
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
diffo = difflib.Differ()


docs_max_length = 40000
if len(data)< docs_max_length:
    vectorstore = Chroma.from_documents(documents=data, embedding=local_embeddings, persist_directory=mem_db_path) # 注意：这个操作会以追加的方式不断写入数据库
else:
    start_count = 0
    while(start_count+docs_max_length<len(data)):
        vectorstore = Chroma.from_documents(documents=data[start_count:start_count+docs_max_length], embedding=local_embeddings, persist_directory=mem_db_path) # 注意：这个操作会以追加的方式不断写入数据库
        start_count+= docs_max_length
    vectorstore = Chroma.from_documents(documents=data[start_count:start_count+docs_max_length], embedding=local_embeddings, persist_directory=mem_db_path) # 注意：这个操作会以追加的方式不断写入数据库


# exit()

# vectorstore = Chroma(embedding_function=local_embeddings, persist_directory=mem_db_path)



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 自定义RAG_TEMPLATE
RAG_TEMPLATE = """
You are an expert in job runtime duration prediction. 
There are some retrieved history jobs that similar to the job waitting to predict.
They are displaied through a diff format. 
{context}

Please predict the job runtime duration based on its matedata, script, and retrieved jobs.
The matedata and script of the job waitting to predict is:
{question}

Your output should only include the runtime, e.g. 10 s. 
It means that the script is likely to run for 10 seconds. 
Note: DO NOT OUTPUT ANYTHING OTHER THAN THE RUNTIME.
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# qa_chain = (
#     rag_prompt
#     | model
#     | StrOutputParser()
# )
with open(mid_file,"r") as f:
    mid_data = json.load(f)
with open(test_file,"r") as f:
    test_data = json.load(f)

def get_texts_from_docs(docs):
    return [doc.page_content for doc in docs]

def drop_items(s_m,th,ids):
    flags = [0,]*len(s_m)
    for i in range(len(s_m)):
        for j in range(len(s_m)):
            if i==j:
                continue
            if s_m[i][j] > th:
                flags[i] +=1
                flags[j] +=1
    max_id = np.argmax(np.array(flags)) 
    if flags[max_id] == 0:
        return s_m,ids
    s_m = pd.DataFrame(s_m)
    s_m.drop(columns=max_id,axis=1,inplace=True)
    s_m.drop(index=max_id,axis=0,inplace=True)
    del ids[max_id]
    s_m = s_m.to_numpy()
    return s_m,ids

def reorganize_docs(docs,local_embeddings,args):
    texts = get_texts_from_docs(docs)
    embeds = local_embeddings.embed_documents(texts)
    ids = list(range(len(texts)))
    th_rate = args["reorganize_docs_rate"] # 去掉相似度最高的节点，还是去掉最公共的相似节点
    up_count = len(ids)*0.5
    
    s_m = []
    for a in embeds:
        s_m_1 = []
        for b in embeds:
            a = np.array(a)
            b = np.array(b)
            s = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
            s_m_1.append(s)
        s_m.append(s_m_1)
    for i in range(len(s_m)):
        s_m[i][i] = 0
    s_m = np.array(s_m)
    th = np.percentile(s_m,100*(1-th_rate)) 
    
    previous_l = len(s_m)
    count =0
    s_m,ids = drop_items(s_m,th,ids)
    count =1
    while(previous_l!=len(s_m)):
        if count >= up_count:
            break
        previous_l = len(s_m)
        s_m,ids = drop_items(s_m,th,ids)
        count +=1
        
    
    new_texts = []
    for i in ids:
        new_texts.append(texts[i])
    
    return new_texts

def process_retrieved_doc(args,redocs1,question):
    redocs = ''
    his_pred = []
    if args["reorganize_docs_rate"]>0:
        if len(redocs1)>0:
            redocs1 = reorganize_docs(redocs1,local_embeddings,args)
        redocs = "\n\n".join(redocs1)
    else:
        for redocsZ in redocs1:
            p_line =  redocsZ.page_content.splitlines()[-1]
            his_pred.append(int(float(p_line.split(':')[1])))
        if args["diff_contex"]:
            for redocsZ in redocs1:
                diff_redocsZ = diffo.compare(question.splitlines(), redocsZ.page_content.splitlines())
                if args["diff_mode"] == 'unchange':
                    diff_redocsZ_ = "\n".join([line for line in diff_redocsZ if line.startswith('  ')]) # line.startswith('- ') or 
                elif args["diff_mode"] == 'unchange+insert':
                    diff_redocsZ_ = "\n".join([line for line in diff_redocsZ if line.startswith('  ') or line.startswith('+ ')]) 
                elif args["diff_mode"] == 'insert':
                    diff_redocsZ_ = "\n".join([line for line in diff_redocsZ if line.startswith('+ ')]) 
                redocs = redocs + "\n\n" + diff_redocsZ_
        else:
            redocs1 = format_docs(redocs1)
            redocs = redocs + "\n\n" + redocs1
    return redocs, his_pred

def worker(config):
    """线程工作函数"""
    # 这里可以放置需要并行执行的代码
    test_data = config["test_data"][config["start"]:config["end"]]
    gts = []
    preds = []
    processed_case_num = 0
    old_index = 0
    for i in tqdm(range(len(test_data))):
        if i > args["query_numbers"]:
            break
        
        case = test_data[i]
        question = case["input"]
        gt = case["prediction"]
        userid = case["userid"]
        redocs = ""
        try:
            if args["user_retrieve_num"] > 0:
                redocs1 = vectorstore.similarity_search(question,k=args["user_retrieve_num"],filter={"userid":userid})
                redocs, his_pred = process_retrieved_doc(args,redocs1,question)
            if args["general_retrieve_num"]>0:
                redocs2 = vectorstore.similarity_search(question,k=args["general_retrieve_num"])
                redocs, his_pred = process_retrieved_doc(args,redocs2,question)
           
            
            completion = client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[
                    {"role": "user", "content": rag_prompt.invoke({"context":redocs,"question":question}).messages[0].content}
                ],
                stream=False
            )
            pred = completion.choices[0].message.content
            pred = find_numberv2(pred)
        except Exception as e:
            print(e)
            pred = np.mean(his_pred)
            # continue
        # logging.debug(f"pred:{pred}\n")
        print(f"pred:{pred}\n")
        gt_v = float(gt)
        pred_v = pred
        gts.append(gt_v)
        preds.append(pred_v)
        processed_case_num +=1
        
        mid_data.append(case)
        if args["update_database"] > 0:
            
            if case["time_submit"] > mid_data[old_index]["time_end"]:
                current_d  = Document(
                    page_content=f'{mid_data[old_index]["input"]} prediction:{mid_data[old_index]["prediction"]}',
                    metadata={"line_number": 0, "userid":mid_data[old_index]["userid"] ,"source": ""},
                )
                documents = [current_d]
                uuids = [str(uuid4()) for _ in range(len(documents))]
                vectorstore.add_documents(documents=documents, ids=uuids)
                old_index+=1
    # break
    res = {"gts":gts,"preds":preds,"processed_case_num":processed_case_num}
    return res
 
start_time = time.time()
configs = []
step = int(args["query_numbers"]/args["worker_num"])
for i in range(args["worker_num"]):
    config = {"test_data":test_data,"start":i*step,"end":(i+1)*step}
    configs.append(config)
# pool = multiprocessing.Pool(processes=args["worker_num"])
# results = pool.map(worker, configs)
# pool.close()
# pool.join()

# 单进程 
results = [worker(configs[0])]

gts = []
preds= []
processed_case_num = 0
for i in range(args["worker_num"]):
    gts.append(results[i]["gts"])
    preds.append(results[i]["preds"])
    processed_case_num += results[i]["processed_case_num"]
end_time = time.time()

pred = np.concatenate(preds,axis=0)
gt = np.concatenate(gts,axis=0)
pred,gt = scaler_transform_prediction(y_train, gt,pred)
acc = get_ea(pred,gt)
ur = get_ur(pred,gt)
mae = round(mean_absolute_error(gt, pred),4)
mse = round(mean_squared_error(gt, pred),4)
r2 = round(r2_score(gt, pred),4)

res = {"acc":acc,
       "ur":ur,
       "mse":mse,
       "mae":mae,
       "r2":r2,
       "used_time":end_time-start_time,
       "processed_case_num":processed_case_num,
       "model_name":args['model'],
       "general_retrieve_num":args["general_retrieve_num"],
       "user_retrieve_num":args['user_retrieve_num'],
       "reorganize_docs_rate":args['reorganize_docs_rate']}

record_file = f"res/{args['dataset']}_m{args['model']}_gn{args['general_retrieve_num']}_un{args['user_retrieve_num']}_rr{args['reorganize_docs_rate']}_ub{args['update_database']}_dc{args['diff_contex']}_us{args['use_script']}.txt"
with open(record_file,"w") as fw:
        json.dump(res,fw)
print(f'save as {record_file}')


