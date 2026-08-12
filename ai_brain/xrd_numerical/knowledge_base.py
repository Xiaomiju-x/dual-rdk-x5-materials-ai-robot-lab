"""
XRD知识库 — 从 rag/xrd_knowledge/papers/ 动态加载16篇论文

已弃用硬编码dict，改为从txt文件加载。
本模块保留兼容接口，供其他脚本 import 使用。

数据来源: rag/xrd_knowledge/papers/ 下16篇论文txt文件
"""

import os

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# 知识库路径候选
KNOWLEDGE_DIRS = [
    os.path.join(_MODULE_DIR, "rag", "xrd_knowledge", "papers"),        # 项目根目录
    os.path.join(_MODULE_DIR, "bpu", "xrd_knowledge", "papers"),        # bpu同级
    "/home/rdk/xrd1/xrd_knowledge/papers",                          # X5部署
]

GARNET_PAPERS = {"paper02", "paper05", "paper09", "paper10", "paper13"}


def _find_dir(candidates):
    for d in candidates:
        real = os.path.realpath(d)
        if os.path.isdir(real):
            return real
    return None


def load_papers(category=None):
    """加载论文知识库

    Args:
        category: 'garnet' | 'non_garnet' | None(全部)
    Returns:
        list of dict: [{'filename', 'content', 'is_garnet'}, ...]
    """
    knowledge_dir = _find_dir(KNOWLEDGE_DIRS)
    if not knowledge_dir:
        return []

    papers = []
    for fname in sorted(os.listdir(knowledge_dir)):
        if not fname.endswith('.txt'):
            continue
        prefix = fname.split('_')[0]
        is_garnet = prefix in GARNET_PAPERS

        if category == 'garnet' and not is_garnet:
            continue
        if category == 'non_garnet' and is_garnet:
            continue

        fpath = os.path.join(knowledge_dir, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        papers.append({'filename': fname, 'content': content, 'is_garnet': is_garnet})

    return papers


def get_all_context():
    """获取全部论文内容拼接（供视觉线等使用）"""
    papers = load_papers()
    return "\n\n---\n\n".join(p['content'] for p in papers)
