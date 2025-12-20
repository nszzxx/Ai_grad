"""
项目书语义切割器
基于通用项目书模板结构进行语义切分
"""
import re
from typing import List, Tuple
from app.core.logger import get_logger

logger = get_logger("project_splitter")


class ProjectSplitter:
    """项目书语义切割器"""

    # 常见项目书章节标题模式
    SECTION_PATTERNS = [
        # 一级标题 (一、二、三...)
        r'^[一二三四五六七八九十]+[、．.]\s*(.+)',
        # 数字标题 (1. 2. 3...)
        r'^(\d+)[、．.]\s*(.+)',
        # 带括号标题 ((一) (二)...)
        r'^\([一二三四五六七八九十]+\)\s*(.+)',
        # 常见关键词
        r'^(项目背景|背景介绍|项目简介|问题分析|痛点分析|解决方案|'
        r'技术方案|产品介绍|商业模式|盈利模式|市场分析|竞争分析|'
        r'团队介绍|团队成员|发展规划|实施计划|风险分析|创新点|'
        r'项目优势|核心技术|应用场景|目标用户|项目总结|附录)[：:]*\s*',
    ]

    # 最大/最小 chunk 长度
    MAX_CHUNK_SIZE = 800
    MIN_CHUNK_SIZE = 50

    def __init__(self, max_chunk_size: int = None, min_chunk_size: int = None):
        self.max_chunk_size = max_chunk_size or self.MAX_CHUNK_SIZE
        self.min_chunk_size = min_chunk_size or self.MIN_CHUNK_SIZE

    def split(self, text: str) -> List[dict]:
        """
        切分项目书

        Args:
            text: 项目书全文

        Returns:
            [{'section': '章节名', 'content': '内容', 'index': 序号}]
        """
        if not text or not text.strip():
            return []

        # 按行分割
        lines = text.split('\n')
        sections = []
        current_section = "摘要"
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测是否是章节标题
            section_name = self._detect_section(line)
            if section_name:
                # 保存上一个章节
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if len(content) >= self.min_chunk_size:
                        sections.append({
                            'section': current_section,
                            'content': content
                        })
                current_section = section_name
                current_content = []
            else:
                current_content.append(line)

        # 保存最后一个章节
        if current_content:
            content = '\n'.join(current_content).strip()
            if len(content) >= self.min_chunk_size:
                sections.append({
                    'section': current_section,
                    'content': content
                })

        # 处理过长的章节
        result = []
        for i, sec in enumerate(sections):
            if len(sec['content']) > self.max_chunk_size:
                sub_chunks = self._split_long_section(sec['section'], sec['content'])
                for j, chunk in enumerate(sub_chunks):
                    result.append({
                        'section': f"{sec['section']}_{j+1}",
                        'content': chunk,
                        'index': len(result)
                    })
            else:
                sec['index'] = len(result)
                result.append(sec)

        logger.debug(f"项目书切分完成: {len(result)} 个片段")
        return result

    def split_to_texts(self, text: str) -> List[str]:
        """
        简化接口：只返回文本列表

        Args:
            text: 项目书全文

        Returns:
            [chunk_text, ...]
        """
        sections = self.split(text)
        return [f"[{s['section']}] {s['content']}" for s in sections]

    def _detect_section(self, line: str) -> str:
        """检测是否是章节标题，返回章节名"""
        for pattern in self.SECTION_PATTERNS:
            match = re.match(pattern, line)
            if match:
                # 提取章节名
                groups = match.groups()
                if groups:
                    # 取最后一个非空分组
                    for g in reversed(groups):
                        if g and len(g) < 30:
                            return g.strip()
                return line[:20].strip()
        return None

    def _split_long_section(self, section_name: str, content: str) -> List[str]:
        """切分过长的章节"""
        chunks = []
        # 按句子切分
        sentences = re.split(r'([。！？；\n])', content)

        current_chunk = []
        current_len = 0

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            punct = sentences[i + 1] if i + 1 < len(sentences) else ''
            full_sentence = sentence + punct

            if current_len + len(full_sentence) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                current_chunk = [full_sentence]
                current_len = len(full_sentence)
            else:
                current_chunk.append(full_sentence)
                current_len += len(full_sentence)

        if current_chunk:
            chunks.append(''.join(current_chunk))

        return chunks


# 全局实例
project_splitter = ProjectSplitter()
