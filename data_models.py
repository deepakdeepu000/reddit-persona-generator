from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict


__all__ = [
    'UserMetrics',
    'RedditPost', 
    'RedditComment',
    'UserData'
]

@dataclass
class UserMetrics:
    """Data class for user metrics"""
    total_posts: int
    total_comments: int
    high_quality_posts: int
    high_quality_comments: int
    average_post_score: float
    average_comment_score: float
    unique_subreddits: List[str]
    content_by_subreddit: Dict[str, Dict[str, int]]


@dataclass
class RedditPost:
    """Data class for Reddit post"""
    index: str
    title: str
    selftext: str
    subreddit: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: int
    gilded: int
    url: str
    quality_score: int
    cleaned_text: str


@dataclass
class RedditComment:
    """Data class for Reddit comment""" 
    index: str
    body: str
    subreddit: str
    score: int
    created_utc: int
    parent_id: str
    gilded: int
    controversiality: int
    url: str
    quality_score: int
    cleaned_text: str

@dataclass
class UserData:
    """Data class for user data"""
    username: str
    account_created: datetime
    icon_img: str
    posts: List[RedditPost] 
    comments: List[RedditComment]
    karma: Dict[str, int]
    raw_text: Dict[str, List[str]]
    quality_metrics: UserMetrics

