#problem填数据集地址，huber_dataset，Ridge_dataset，boqp_dataset，SBQP_dataset，softplus_dataset
#--output_dir 填写模型保存的地址，和project_dataset同级，注意对应和数据集名称修改,有huber,Ridge,boqp,SBQP,softplus
#model_name 在数据集地址下面

torchrun --nproc_per_node=8 main_train.py \
    --problem project_dataset/huber_dataset \
    --data_format messages \
    --add_discrete_tokens \
    --model_name project_dataset/Qwen2.5-7B-Instruct \
    --output_dir project_model/huber"