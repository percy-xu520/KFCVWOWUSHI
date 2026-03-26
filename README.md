````md
# 使用步骤说明

1. **修改文件**
   - 编辑 `modelscope_download.py` 文件第 39 行  
   - 将 `local_dir` 修改为数据集挂载目录

2. **添加执行权限**
   ```bash
   chmod +x run.sh
   chmod +x create.sh
````

3. **下载数据集并创建环境**

   ```bash
   ./create.sh
   ```

4. **开始训练（注意修改路径）**

   ```bash
   ./run.sh
   ```

5. **修改上传配置并执行**

   * 编辑 `modelscope_upload.py` 第 29 行
   * 将其修改为真实的 `project_model` 地址

   ```bash
   python modelscope_upload.py
   ```

```
```
