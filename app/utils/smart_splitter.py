import re
from typing import List, Tuple, Optional
# 假设 app.core.logger 依然可用，这里为了独立运行先注释掉或替换
# from app.core.logger import get_logger
import logging
logger = logging.getLogger("smart_splitter")

class SmartSectionSplitter:
    def __init__(
        self,
        max_chunk_size: int = 800,
        min_chunk_size: int = 100,
        enable_fallback: bool = True
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.enable_fallback = enable_fallback
        
        # 正则保持不变，这部分写得很好
        self.section_patterns = [
            (r'^第[一二三四五六七八九十百\d]+章[：:、\s]', '章节'),
            (r'^第[一二三四五六七八九十百\d]+节[：:、\s]', '小节'),
            (r'^[一二三四五六七八九十百]+[、：:]\s*', '中文数字标题'),
            (r'^\d+[、\.．。]\s+', '数字标题'),
            (r'^[（(]\d+[)）]\s*', '括号数字'),
            (r'^\d+\.\d+\s+', '多级编号'),
            (r'^(Part|Section|Chapter)\s+[IVX\d]+[：:、\s]', '英文章节'),
            (r'^[【\[].*?[】\]]\s*', '方括号标题'),
        ]
        self.compiled_patterns = [
            (re.compile(pattern, re.MULTILINE | re.IGNORECASE), name)
            for pattern, name in self.section_patterns
        ]

    def _find_section_breaks(self, text: str) -> List[Tuple[int, str]]:
        # 逻辑不变
        breaks = []
        for pattern, pattern_name in self.compiled_patterns:
            for match in pattern.finditer(text):
                start_pos = match.start()
                line_start = text.rfind('\n', 0, start_pos) + 1
                prefix = text[line_start:start_pos]
                if prefix.strip() == '':
                    breaks.append((start_pos, pattern_name))
        return sorted(set(breaks), key=lambda x: x[0])

    def _extract_title(self, text: str) -> str:
        """提取章节的第一行作为标题"""
        if not text:
            return ""
        return text.split('\n')[0].strip()[:50] # 限制标题长度，防止误判

    def _split_long_section(self, text: str, parent_title: str = "") -> List[str]:
        """
        [修改点] 增加了 parent_title 参数
        如果单个章节过长，按句子边界进一步切分，并注入父标题
        """
        if len(text) <= self.max_chunk_size:
            return [text]

        chunks = []
        current_chunk = ""
        
        # 优化句子切分正则，保留分割符
        sentences = re.split(r'([。！？；\n\r]+)', text)

        for sentence in sentences:
            # 如果加上这句话超长了，就先保存当前的
            if len(current_chunk) + len(sentence) > self.max_chunk_size:
                if current_chunk.strip():
                    # [关键修改]：如果是被切分的后续块，手动加上标题
                    if parent_title and len(chunks) > 0:
                        formatted_chunk = f"【接上文：{parent_title}】\n{current_chunk.strip()}"
                    else:
                        formatted_chunk = current_chunk.strip()
                    
                    chunks.append(formatted_chunk)
                current_chunk = sentence
            else:
                current_chunk += sentence

        # 处理最后一块
        if current_chunk.strip():
            if parent_title and len(chunks) > 0:
                formatted_chunk = f"【接上文：{parent_title}】\n{current_chunk.strip()}"
            else:
                formatted_chunk = current_chunk.strip()
            chunks.append(formatted_chunk)

        return chunks

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        section_breaks = self._find_section_breaks(text)
        logger.info(f"检测到 {len(section_breaks)} 个章节标题")

        if len(section_breaks) < 2:
            if self.enable_fallback:
                return self._fallback_split(text)
            return [text]

        chunks = []
        for i in range(len(section_breaks)):
            start_pos = section_breaks[i][0]
            end_pos = section_breaks[i + 1][0] if i + 1 < len(section_breaks) else len(text)
            
            section_text = text[start_pos:end_pos].strip()
            if not section_text:
                continue

            # [关键修改] 获取当前章节的标题
            current_title = self._extract_title(section_text)

            # 1. 如果章节过长，进一步切分，并传入标题
            if len(section_text) > self.max_chunk_size:
                sub_chunks = self._split_long_section(section_text, parent_title=current_title)
                chunks.extend(sub_chunks)
            
            # 2. 如果章节太短，尝试与上一个合并 (小优化：合并时最好也保留层级感)
            elif len(section_text) < self.min_chunk_size and chunks:
                last_chunk = chunks[-1]
                # 只有当上一个chunk不是过长切分出来的（即没有【接上文】标记）才合并，避免逻辑混乱
                if "【接上文" not in last_chunk and (len(last_chunk) + len(section_text) <= self.max_chunk_size):
                    chunks[-1] = last_chunk + "\n\n" + section_text
                else:
                    chunks.append(section_text)
            else:
                chunks.append(section_text)

        # 处理前言部分
        if section_breaks and section_breaks[0][0] > 0:
            intro_text = text[:section_breaks[0][0]].strip()
            if intro_text:
                intro_chunks = self._split_long_section(intro_text, parent_title="前言/文档头部")
                chunks = intro_chunks + chunks

        return [c for c in chunks if c.strip()]

    def _fallback_split(self, text: str) -> List[str]:
        # 简单的兜底逻辑，按固定长度切
        return [text[i:i+self.max_chunk_size] for i in range(0, len(text), self.max_chunk_size-50)]

# 使用示例
if __name__ == "__main__":
    # 配置建议
    splitter = SmartSectionSplitter()