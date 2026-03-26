1.先修改modelscope_download.py文件第39行，将local_dir改成数据集挂载目录
2.加权限
chmod +x run.sh
chmod +x create.sh
3.运行以下命令，下载数据集和创建环境
./create.sh
4.运行以下命令训练(注意改里面的路径)
./run.sh
5.修改modelscope_upload.py29行到真实的project_model地址
运行python modelscope_upload.py
