"""
ModelScope 魔搭社区 —— 项目文件上传工具
使用官方 modelscope SDK，无需安装 Git

依赖安装：
    pip install modelscope -U

使用方式：
    1. 直接修改下方 CONFIG 配置项后运行
    2. 或通过命令行参数传入（见底部 main 函数）
"""

import os
import sys
import argparse
import pathlib
from typing import Literal

# ============================================================
#  ★ 在这里填写你的配置 ★
# ============================================================
CONFIG = {
    # 你的魔搭访问令牌（Token）
    # 获取方式：登录 modelscope.cn → 右上角头像 → 账号设置 → 访问令牌
    "token": "ms-cc19db94-3ed5-4b10-bc08-6c9f86f305f7",

    # 本地要上传的项目文件夹路径，例如："/home/user/my_project"
    # "local_dir": "./generate_dataset/ridge/ridge-dataset",
    "local_dir": "project_model",
    # 目标仓库 ID，格式为 "用户名/仓库名"，例如："zhangsan/my-model"
    "repo_id": "superbeiliya/LLM-Code",

    # 仓库类型：'model'（模型）| 'dataset'（数据集）
    "repo_type": "dataset",

    # 上传到仓库中的哪个子路径，空字符串表示根目录
    "path_in_repo": "model_checkpoint",

    # 提交信息
    "commit_message": "Upload project files",

    # 并发上传线程数（建议 4~16，网络好可以调高）
    "max_workers": 100,

    # 要排除的文件/目录（使用 glob 模式），可留空 []
    # 示例：["*.pyc", "__pycache__", ".git", "*.log"]
    "exclude": ["*.pyc", "__pycache__", ".git", ".DS_Store", "*.log"],
}
# ============================================================


def check_and_install():
    """检查 modelscope 是否已安装"""
    try:
        import modelscope  # noqa: F401
    except ImportError:
        print("[提示] 未检测到 modelscope 库，正在尝试自动安装...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "modelscope", "-U", "-q"],
            capture_output=False,
        )
        if result.returncode != 0:
            print("[错误] 自动安装失败，请手动执行：pip install modelscope -U")
            sys.exit(1)
        print("[成功] modelscope 安装完成！\n")


def upload_project(
    token: str,
    local_dir: str,
    repo_id: str,
    repo_type: Literal["model", "dataset"] = "model",
    path_in_repo: str = "",
    commit_message: str = "Upload project files",
    max_workers: int = 8,
    exclude: list = None,
):
    """
    上传本地项目文件夹到魔搭社区仓库

    Args:
        token:          魔搭访问令牌
        local_dir:      本地项目路径
        repo_id:        仓库 ID（格式：用户名/仓库名）
        repo_type:      仓库类型，'model' 或 'dataset'
        path_in_repo:   上传到仓库的哪个子目录（空 = 根目录）
        commit_message: Git 提交信息
        max_workers:    并发上传线程数
        exclude:        要排除的 glob 规则列表
    """
    from modelscope.hub.api import HubApi, ModelScopeConfig

    # ---------- 1. 参数校验 ----------
    local_path = pathlib.Path(local_dir).expanduser().resolve()
    if not local_path.exists():
        print(f"[错误] 本地路径不存在：{local_path}")
        sys.exit(1)
    if not local_path.is_dir():
        print(f"[错误] 指定路径不是文件夹：{local_path}")
        sys.exit(1)

    print("=" * 60)
    print("  魔搭社区（ModelScope）项目上传工具")
    print("=" * 60)
    print(f"  本地路径   : {local_path}")
    print(f"  目标仓库   : {repo_id}  [{repo_type}]")
    print(f"  仓库子目录 : {'根目录' if not path_in_repo else path_in_repo}")
    print(f"  并发线程数 : {max_workers}")
    print("=" * 60)

    # ---------- 2. 统计文件数量 ----------
    exclude = exclude or []
    files_to_upload = []
    for root, dirs, files in os.walk(local_path):
        # 过滤排除目录
        dirs[:] = [
            d for d in dirs
            if not _should_exclude(d, exclude)
        ]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), local_path)
            if not _should_exclude(f, exclude):
                files_to_upload.append(rel)

    print(f"  待上传文件 : {len(files_to_upload)} 个")
    print()

    # ---------- 3. 登录 ----------
    print("[1/3] 正在登录魔搭社区...")
    api = HubApi()
    api.login(token)
    namespace, _ = ModelScopeConfig.get_user_info()
    print(f"      登录成功！用户名：{namespace}")

    # ---------- 4. 检查/创建仓库 ----------
    owner, repo_name = repo_id.split("/", 1) if "/" in repo_id else (namespace, repo_id)
    full_repo_id = f"{owner}/{repo_name}"

    print(f"\n[2/3] 检查仓库 {full_repo_id} ...")
    try:
        if repo_type == "model":
            api.get_model(model_id=full_repo_id)
        else:
            api.get_dataset(dataset_id=full_repo_id)
        print(f"      仓库已存在，直接上传。")
    except Exception:
        print(f"      仓库不存在，正在自动创建...")
        try:
            if repo_type == "model":
                api.create_model(
                    model_id=full_repo_id,
                    visibility=3,       # 3=公开, 1=私有
                    license="apache-2.0",
                )
            else:
                api.create_dataset(
                    dataset_id=full_repo_id,
                    visibility=3,
                    license="apache-2.0",
                )
            print(f"      仓库创建成功！")
        except Exception as e:
            print(f"[警告] 仓库创建失败（可能已存在或权限不足）：{e}")
            print("      继续尝试上传...")

    # ---------- 5. 上传文件夹 ----------
    print(f"\n[3/3] 开始上传文件夹（{len(files_to_upload)} 个文件）...")
    print("      请耐心等待，大文件将自动走 LFS 通道...\n")

    try:
        api.upload_folder(
            repo_id=full_repo_id,
            folder_path=local_path,
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            repo_type=repo_type,
        )

        # 构造结果 URL
        base_url = "https://modelscope.cn"
        if repo_type == "model":
            url = f"{base_url}/models/{full_repo_id}/files"
        else:
            url = f"{base_url}/datasets/{full_repo_id}/files"

        print("\n" + "=" * 60)
        print("  ✅ 上传完成！")
        print(f"  🔗 查看地址：{url}")
        print("=" * 60)

    except Exception as e:
        print(f"\n[错误] 上传失败：{e}")
        print("\n常见原因排查：")
        print("  1. Token 无效或已过期 → 重新生成并更新 CONFIG['token']")
        print("  2. 仓库 ID 格式错误   → 应为 '用户名/仓库名'")
        print("  3. 网络不稳定         → 重试几次即可")
        print("  4. 文件超出大小限制   → 单文件 ≤ 500GB，总文件数 ≤ 100,000")
        sys.exit(1)


def _should_exclude(name: str, patterns: list) -> bool:
    """判断文件/目录名是否匹配排除规则"""
    import fnmatch
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def main():
    parser = argparse.ArgumentParser(
        description="上传本地项目到魔搭社区（ModelScope）仓库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用脚本内 CONFIG 配置直接运行
  python modelscope_upload.py

  # 通过命令行参数指定
  python modelscope_upload.py \\
      --token  ms-xxxxxxxx \\
      --dir    ./my_model_dir \\
      --repo   zhangsan/my-first-model \\
      --type   model \\
      --msg    "first upload"
        """
    )
    parser.add_argument("--token",   default=None, help="魔搭访问令牌（SDK Token）")
    parser.add_argument("--dir",     default=None, help="本地项目文件夹路径")
    parser.add_argument("--repo",    default=None, help="目标仓库 ID（用户名/仓库名）")
    parser.add_argument("--type",    default=None, choices=["model", "dataset"], help="仓库类型")
    parser.add_argument("--subdir",  default=None, help="上传到仓库的子目录（默认根目录）")
    parser.add_argument("--msg",     default=None, help="提交信息")
    parser.add_argument("--workers", default=None, type=int, help="并发上传线程数（默认 8）")
    args = parser.parse_args()

    # 命令行参数优先，否则使用 CONFIG
    cfg = CONFIG.copy()
    if args.token:   cfg["token"]          = args.token
    if args.dir:     cfg["local_dir"]      = args.dir
    if args.repo:    cfg["repo_id"]        = args.repo
    if args.type:    cfg["repo_type"]      = args.type
    if args.subdir:  cfg["path_in_repo"]   = args.subdir
    if args.msg:     cfg["commit_message"] = args.msg
    if args.workers: cfg["max_workers"]    = args.workers

    # 检查必填项
    if cfg["token"] == "YOUR_TOKEN_HERE":
        print("[错误] 请先在脚本顶部的 CONFIG 中填写你的魔搭 Token，")
        print("       或通过 --token 参数传入。")
        print("       Token 获取：https://modelscope.cn → 账号设置 → 访问令牌")
        sys.exit(1)

    check_and_install()
    upload_project(
        token          = cfg["token"],
        local_dir      = cfg["local_dir"],
        repo_id        = cfg["repo_id"],
        repo_type      = cfg["repo_type"],
        path_in_repo   = cfg["path_in_repo"],
        commit_message = cfg["commit_message"],
        max_workers    = cfg["max_workers"],
        exclude        = cfg["exclude"],
    )


if __name__ == "__main__":
    main()