from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class GlobalConfig(Base):
    """
    全局配置表
    对应数据库表名: global_config
    """
    __tablename__ = "global_config"

    id = Column(Integer, primary_key=True, index=True)
    config_key = Column(String(100), unique=True)
    config_value = Column(Text)
    remark = Column(String(255))
    create_time = Column(DateTime)
    update_time = Column(DateTime)


class LLMConfig(Base):
    """
    LLM 配置表：存放 API Key 等信息
    对应数据库表名: sys_llm_config
    """
    __tablename__ = "sys_llm_config"

    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String(50), unique=True)
    api_key = Column(String(500))
    base_url = Column(String(200))
    model_name = Column(String(100))
    is_active = Column(Boolean, default=True)


class Competition(Base):
    """
    竞赛信息表
    对应数据库表名: competitions
    """
    __tablename__ = "competitions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255))
    organizer = Column(String(255))
    category = Column(String(100))
    track = Column(String(255))
    description = Column(Text)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    official_url = Column(String(500))
    pati_starttime = Column(DateTime)
    pati_endtime = Column(DateTime)
    tags = Column(String(500))
    rules_json = Column(Text)
    created_at = Column(DateTime)
    participation_mode = Column(String(50))
    min_team_size = Column(Integer)
    max_team_size = Column(Integer)
    difficulty = Column(String(50))
    session = Column(String(100))
