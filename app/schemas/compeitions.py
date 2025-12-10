from pydantic import BaseModel, Field
from typing import List, Optional

# --- 子模型：比赛信息 ---
class CompetitionInfo(BaseModel):
    id: int
    title: str
    organizer: Optional[str] = None
    category: Optional[str] = None
    track: Optional[str] = None
    description: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    official_url: Optional[str] = None
    pati_starttime: Optional[str] = None
    pati_endtime: Optional[str] = None
    tags: Optional[str] = None
    rules_json: Optional[str] = None
    created_at: Optional[str] = None
    participation_mode: Optional[str] = None
    min_team_size: Optional[int] = None
    max_team_size: Optional[int] = None
    difficulty: Optional[str] = None
    session: Optional[str] = None
