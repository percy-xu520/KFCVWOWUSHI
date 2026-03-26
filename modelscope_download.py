"""
ModelScope 魔搭社区 —— 项目文件下载工具
使用官方 modelscope SDK，无需安装 Git

【修复说明】
  modelscope 的 snapshot_download 只支持 model 类型仓库。
  下载 dataset 类型仓库必须使用专用的 dataset_snapshot_download 函数，
  并使用 dataset_id 而非 model_id 参数，否则会触发 404 错误。

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
import shutil

# ============================================================
#  ★ 在这里填写你的配置 ★
# ============================================================
CONFIG = {
    # 你的魔搭访问令牌（下载公开仓库可不填，私有仓库必须填）
    "token": "ms-cc19db94-3ed5-4b10-bc08-6c9f86f305f7",

    # 目标仓库 ID，格式为 "用户名/仓库名"，例如："zhangsan/my-model"
    "repo_id": "superbeiliya/LLM-Code",

    # 仓库类型：'model'（模型）| 'dataset'（数据集）
    # ⚠️ 必须与魔搭网页上显示的仓库类型一致，否则会 404！
    "repo_type": "dataset",

    # 本地保存路径，例如："/home/user/downloads/my_project"
    "local_dir": "project-dataset",

    # 只下载仓库中某个子目录或单个文件（留空则下载全部）
    # 示例（子目录）："checkpoints"
    # 示例（单文件）："README.md"
    "path_in_repo": "",

    # 指定下载某个版本/分支（留空则使用默认主分支 master）
    "revision": "",

    # 要排除的文件（glob 模式），留空 [] 则不排除任何文件
    # 示例：["*.log", "*.tmp", "*.ckpt"]
    "exclude": [],

    # 只下载匹配这些规则的文件（留空则下载全部）
    # 示例：["*.json", "*.py", "config.*"]
    "include": [],
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


def download_project(
    repo_id: str,
    local_dir: str,
    repo_type: str = "model",
    token: str = "",
    path_in_repo: str = "",
    revision: str = "",
    exclude: list = None,
    include: list = None,
):
    """
    从魔搭社区仓库下载项目文件到本地

    核心修复：
      - model  类型 → 使用 snapshot_download(model_id=...)
      - dataset 类型 → 使用 dataset_snapshot_download(dataset_id=...)
      两者 API 完全不同，混用会导致 404。

    Args:
        repo_id:      仓库 ID（格式：用户名/仓库名）
        local_dir:    本地保存路径（最终文件直接放在此目录下）
        repo_type:    仓库类型，'model' 或 'dataset'
        token:        魔搭访问令牌（私有仓库必须填）
        path_in_repo: 只下载仓库中的某个子目录或文件（空=全部）
        revision:     指定版本/分支（空=默认主分支）
        exclude:      排除文件的 glob 规则列表
        include:      只下载匹配的 glob 规则列表
    """
    from modelscope.hub.api import HubApi, ModelScopeConfig

    exclude = exclude or []
    include = include or []

    # ---------- 1. 准备本地路径 ----------
    save_path = pathlib.Path(local_dir).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  魔搭社区（ModelScope）项目下载工具")
    print("=" * 60)
    print(f"  仓库 ID     : {repo_id}  [{repo_type}]")
    print(f"  仓库子路径  : {'全部文件' if not path_in_repo else path_in_repo}")
    print(f"  指定版本    : {'默认主分支' if not revision else revision}")
    print(f"  本地保存至  : {save_path}")
    print(f"  排除规则    : {exclude if exclude else '无'}")
    print(f"  包含规则    : {include if include else '全部'}")
    print("=" * 60)

    # ---------- 2. 登录 ----------
    api = HubApi()
    if token:
        print("\n[1/3] 正在登录魔搭社区...")
        try:
            api.login(token)
            namespace, _ = ModelScopeConfig.get_user_info()
            print(f"      登录成功！用户名：{namespace}")
        except Exception as e:
            print(f"[警告] 登录失败：{e}")
            print("      将以匿名方式尝试下载（仅支持公开仓库）...")
    else:
        print("\n[1/3] 未提供 Token，以匿名方式下载（仅支持公开仓库）")

    # ---------- 3. 下载到临时缓存目录 ----------
    # modelscope 会把文件缓存到 cache_dir/<repo_name>/... 的子目录中
    # 我们用一个临时缓存目录，之后再把文件移动到用户指定的 local_dir
    import tempfile
    tmp_cache = pathlib.Path(tempfile.mkdtemp(prefix="ms_download_", dir="/c20250502/lyh/LLM-Code/tmp_cache"))

    print(f"\n[2/3] 开始下载，请稍候...\n")

    try:
        result_path = _do_download(
            api=api,
            repo_id=repo_id,
            repo_type=repo_type,
            cache_dir=str(tmp_cache),
            path_in_repo=path_in_repo,
            revision=revision,
            exclude=exclude,
            include=include,
        )

        # ---------- 4. 把缓存文件移动/复制到用户指定目录 ----------
        print(f"\n[3/3] 正在整理文件到目标目录...")
        result_path = pathlib.Path(result_path).resolve()
        _sync_to_target(result_path, save_path)

        # ---------- 5. 清理临时缓存 ----------
        try:
            shutil.rmtree(tmp_cache, ignore_errors=True)
        except Exception:
            pass

        # 统计
        all_files = [f for f in save_path.rglob("*") if f.is_file()]
        total_size = sum(f.stat().st_size for f in all_files)

        print("\n" + "=" * 60)
        print("  ✅ 下载完成！")
        print(f"  📁 保存路径   : {save_path}")
        print(f"  📦 文件数量   : {len(all_files)} 个")
        print(f"  💾 总大小     : {_format_size(total_size)}")
        print("=" * 60)
        return str(save_path)

    except Exception as e:
        shutil.rmtree(tmp_cache, ignore_errors=True)
        _print_error(str(e))
        sys.exit(1)


def _do_download(api, repo_id, repo_type, cache_dir,
                 path_in_repo, revision, exclude, include):
    """
    根据 repo_type 调用不同的下载函数。
    这是修复 dataset 404 的核心所在。
    """
    # 判断是否为单个文件下载
    is_single_file = (
        path_in_repo
        and not path_in_repo.endswith("/")
        and "." in os.path.basename(path_in_repo)
    )

    if repo_type == "dataset":
        # ✅ dataset 专用下载函数
        from modelscope.hub.snapshot_download import dataset_snapshot_download

        if is_single_file:
            # dataset 单文件：用 dataset_file_download
            from modelscope.hub.file_download import dataset_file_download
            print(f"      模式：数据集单文件下载  →  {path_in_repo}")
            result = dataset_file_download(
                dataset_id=repo_id,
                file_path=path_in_repo,
                cache_dir=cache_dir,
                revision=revision or None,
            )
            return str(pathlib.Path(result).parent)

        kwargs = dict(
            dataset_id=repo_id,
            cache_dir=cache_dir,
        )
        if revision:
            kwargs["revision"] = revision
        if exclude:
            kwargs["ignore_file_pattern"] = exclude
        if include:
            kwargs["allow_file_pattern"] = include
        if path_in_repo:
            prefix = path_in_repo.rstrip("/") + "/"
            existing = kwargs.get("allow_file_pattern", [])
            kwargs["allow_file_pattern"] = [prefix + "**"] + existing
            print(f"      模式：数据集子目录下载  →  {path_in_repo}/")
        else:
            print(f"      模式：完整数据集下载")

        return dataset_snapshot_download(**kwargs)

    else:
        # ✅ model 专用下载函数
        from modelscope.hub.snapshot_download import snapshot_download

        if is_single_file:
            from modelscope.hub.file_download import model_file_download
            print(f"      模式：模型单文件下载  →  {path_in_repo}")
            result = model_file_download(
                model_id=repo_id,
                file_path=path_in_repo,
                cache_dir=cache_dir,
                revision=revision or None,
            )
            return str(pathlib.Path(result).parent)

        kwargs = dict(
            model_id=repo_id,
            cache_dir=cache_dir,
        )
        if revision:
            kwargs["revision"] = revision
        if exclude:
            kwargs["ignore_file_pattern"] = exclude
        if include:
            kwargs["allow_file_pattern"] = include
        if path_in_repo:
            prefix = path_in_repo.rstrip("/") + "/"
            existing = kwargs.get("allow_file_pattern", [])
            kwargs["allow_file_pattern"] = [prefix + "**"] + existing
            print(f"      模式：模型子目录下载  →  {path_in_repo}/")
        else:
            print(f"      模式：完整模型下载")

        return snapshot_download(**kwargs)


def _sync_to_target(src: pathlib.Path, dst: pathlib.Path):
    """
    将 src 目录下的所有文件复制到 dst，保持相对目录结构，
    跳过已存在且内容相同（大小+修改时间）的文件。
    """
    copied = skipped = 0
    for src_file in src.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        # 已存在且大小相同则跳过
        if dst_file.exists() and dst_file.stat().st_size == src_file.stat().st_size:
            skipped += 1
            continue

        shutil.copy2(src_file, dst_file)
        copied += 1

    print(f"      复制完成：新增 {copied} 个，跳过 {skipped} 个（已存在）")


def _format_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def _print_error(err: str):
    print(f"\n[错误] 下载失败：{err}")
    print("\n常见原因排查：")
    if "401" in err or "permission" in err.lower() or "auth" in err.lower():
        print("  ➤ 私有仓库需要填写有效的 Token")
        print("    获取：https://modelscope.cn → 账号设置 → 访问令牌")
    elif "404" in err or "not found" in err.lower() or "not exists" in err.lower():
        print("  ➤ 仓库不存在，最常见原因：")
        print("    1. repo_type 填错了！数据集仓库必须用 'dataset'，模型仓库用 'model'")
        print("    2. repo_id 格式错误，应为 '用户名/仓库名'")
        print("    3. 仓库设置为私有但未填 Token")
    elif "dataset_snapshot_download" in err or "cannot import" in err.lower():
        print("  ➤ modelscope 版本过低，请升级：pip install modelscope -U")
    elif "timeout" in err.lower() or "connect" in err.lower():
        print("  ➤ 网络超时，请检查网络或稍后重试")
    else:
        print("  ➤ 请检查 repo_id、repo_type、Token 是否正确")
        print("  ➤ 尝试升级：pip install modelscope -U")


def main():
    parser = argparse.ArgumentParser(
        description="从魔搭社区（ModelScope）下载仓库项目到本地",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用脚本内 CONFIG 配置直接运行
  python modelscope_download.py

  # 下载数据集仓库（⚠️ 必须加 --type dataset）
  python modelscope_download.py \\
      --token  ms-xxxxxxxx \\
      --repo   zhangsan/LLM-Code \\
      --type   dataset \\
      --dir    ./downloads

  # 下载模型仓库
  python modelscope_download.py \\
      --repo  zhangsan/my-model \\
      --type  model \\
      --dir   ./my_downloads

  # 只下载仓库中的某个子目录
  python modelscope_download.py \\
      --repo    zhangsan/my-model \\
      --type    model \\
      --dir     ./output \\
      --subpath checkpoints/best

  # 只下载单个文件
  python modelscope_download.py \\
      --repo    zhangsan/LLM-Code \\
      --type    dataset \\
      --dir     ./output \\
      --subpath README.md

  # 只下载 .json 和 .py 文件，排除 .log
  python modelscope_download.py \\
      --repo    zhangsan/my-model \\
      --type    model \\
      --dir     ./output \\
      --include "*.json" "*.py" \\
      --exclude "*.log"
        """
    )
    parser.add_argument("--token",    default=None, help="魔搭访问令牌（私有仓库必须）")
    parser.add_argument("--repo",     default=None, help="仓库 ID（用户名/仓库名）")
    parser.add_argument("--type",     default=None, choices=["model", "dataset"],
                        help="仓库类型：model 或 dataset（必须与魔搭网页类型一致）")
    parser.add_argument("--dir",      default=None, help="本地保存路径")
    parser.add_argument("--subpath",  default=None, help="只下载仓库中的某个子目录或单个文件")
    parser.add_argument("--revision", default=None, help="指定版本/分支（默认主分支）")
    parser.add_argument("--exclude",  default=None, nargs="+", help="排除文件的 glob 规则，可传多个")
    parser.add_argument("--include",  default=None, nargs="+", help="只下载匹配的 glob 规则，可传多个")
    args = parser.parse_args()

    # 命令行参数优先，否则使用 CONFIG
    cfg = CONFIG.copy()
    if args.token:    cfg["token"]        = args.token
    if args.repo:     cfg["repo_id"]      = args.repo
    if args.type:     cfg["repo_type"]    = args.type
    if args.dir:      cfg["local_dir"]    = args.dir
    if args.subpath:  cfg["path_in_repo"] = args.subpath
    if args.revision: cfg["revision"]     = args.revision
    if args.exclude:  cfg["exclude"]      = args.exclude
    if args.include:  cfg["include"]      = args.include

    # 检查必填项
    if cfg["repo_id"] in ("your_username/your_repo_name", ""):
        print("[错误] 请先在脚本顶部的 CONFIG 中填写目标仓库 repo_id，")
        print("       或通过 --repo 参数传入，格式为 '用户名/仓库名'。")
        sys.exit(1)

    check_and_install()
    download_project(
        repo_id      = cfg["repo_id"],
        local_dir    = cfg["local_dir"],
        repo_type    = cfg["repo_type"],
        token        = cfg["token"],
        path_in_repo = cfg["path_in_repo"],
        revision     = cfg["revision"],
        exclude      = cfg["exclude"],
        include      = cfg["include"],
    )


if __name__ == "__main__":
    main()