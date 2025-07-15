import os
import re
import json
import time
import requests
import random
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter
import logging

# Third-party imports
import google.generativeai as genai
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from data_models import RedditPost, RedditComment, UserData, UserMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration manager"""
    
    def __init__(self):
        
        load_dotenv()
        
        # Reddit API Configuration
        
        """ if you are using the Reddit API, uncomment the following lines and set your credentials """
        
        # self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        # self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        # self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'RedditPersonaGenerator/1.0')
        
        # Gemini API Configuration
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        # Scraping Configuration
        self.max_pages = 10
        self.rate_limit_delay = 1.0
        self.max_comments = 100
        
        # Quality Scoring Configuration
        self.min_post_length = 50
        self.min_comment_length = 20
        self.quality_threshold = 5


class TextCleaner:
    """Handles text cleaning and preprocessing"""
    
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
        
        self.reddit_artifacts = {
            '[deleted]', '[removed]', '[unavailable]', 
            'deleted by user', 'removed by moderator',
            'This comment has been overwritten',
            'I have left reddit for'
        }
        
        self.low_quality_patterns = [
            r'^[A-Z\s!?]+$',  # ALL CAPS
            r'^\w{1,5}$',     # Very short responses
            r'^(lol|haha|omg|wtf|ok|yeah|no|yes|this|that)\.?$',
            r'^(\+1|same|agreed?|exactly|correct)\.?$',
            r'^(first|second|third|edit:|update:)',
            r'(thanks for the gold|edit: typo|edit: grammar)',
        ]
        
        
        self.quality_indicators = {
            # Personal experience indicators
            'experience', 'personally', 'in my experience', 'i think', 'i believe',
            'learned', 'discovered', 'realized', 'understand', 'perspective',
            'encountered', 'witnessed', 'observed', 'found that', 'noticed',
            
            # Analytical and research indicators
            'analysis', 'research', 'study', 'data', 'citations', 'example',
            'statistics', 'findings', 'results', 'methodology', 'approach',
            'investigation', 'examination', 'comparison', 'evaluation',
            
            # Logical reasoning indicators
            'specifically', 'particularly', 'interestingly', 'furthermore',
            'however', 'although', 'despite', 'therefore', 'because',
            'consequently', 'meanwhile', 'alternatively', 'nevertheless',
            'moreover', 'conversely', 'subsequently', 'accordingly',
            
            # Expertise and knowledge indicators
            'expertise', 'professional', 'specialist', 'technical', 'advanced',
            'complex', 'detailed', 'comprehensive', 'thorough', 'extensive',
            'methodology', 'framework', 'principle', 'concept', 'theory',
            
            # Quality discussion indicators
            'nuanced', 'sophisticated', 'elaborate', 'clarification',
            'distinction', 'consideration', 'implication', 'significance',
            'context', 'background', 'foundation', 'underlying', 'fundamental',
            
            # Helpful and constructive indicators
            'suggestion', 'recommendation', 'advice', 'tip', 'solution',
            'helpful', 'useful', 'beneficial', 'practical', 'effective',
            'improvement', 'optimization', 'enhancement', 'troubleshooting'
        }
        
        self.low_quality_indicators = [
            'lol', 'lmao', 'rofl', 'omg', 'wtf', 'fml', 'smh',
            'this', 'that', 'yeah', 'nah', 'meh', 'idk'
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or text in self.reddit_artifacts:
            return ""
        
        # Remove Reddit formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
        text = re.sub(r'\^(\w+)', r'\1', text)        # Superscript
        text = re.sub(r'&gt;', '>', text)             # Quote markers
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&amp;', '&', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove Reddit-specific patterns
        text = re.sub(r'/?u/\w+', '', text)
        text = re.sub(r'/?r/\w+', '', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def calculate_quality_score(self, text: str, metadata: Dict ) -> int:
        """Calculate quality score for content"""
        for pattern in self.low_quality_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 0
        
        text_lower = text.lower()
        text_len = len(text)
        
        # Extract metadata values once
        reddit_score = metadata.get('score', 0)
        gilded = metadata.get('gilded', 0)
        num_comments = metadata.get('num_comments', 0)

        # Initialize score with sentiment analysis
        sentiment_scores = self.sia.polarity_scores(text)
        compound_sentiment = sentiment_scores['compound']
        score = min(2, max(-1, int(compound_sentiment * 2)))
        
        # Length bonus (up to 5 points)
        score += min(5, text_len // 100)
        
        # Quality indicators - use string search instead of tokenization
        quality_score = 0
        for indicator in self.quality_indicators:
            if indicator in text_lower:
                quality_score += 1
            if quality_score >= 8:  # Cap at 8 points
                break
        score += min(8, quality_score)
        
        # Reddit engagement (up to 4 points)
        score += min(2, reddit_score // 10) + min(2, gilded) + min(1, num_comments // 5)
        
        # Vocabulary diversity - simplified calculation
        words = text_lower.split()
        if words:
            unique_words = len(set(word for word in words if word not in self.stop_words))
            diversity = unique_words / len(words)
            score += min(2, int(diversity * 4))
        
        # Penalize low quality indicators
        low_quality_count = sum(1 for indicator in self.low_quality_indicators 
                       if indicator in text_lower)
        score -= min(3, low_quality_count)
        
        return max(0, min(20, score))




class RedditScraper:
    """Handles Reddit data scraping"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.rate_limit_delay = 1.0
    
    def extract_username(self, url: str) -> str:
        """Extract username from Reddit URL"""
        patterns = [
            r'reddit\.com/user/([^/]+)',
            r'reddit\.com/u/([^/]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return url.strip().strip('/')
    
    def get_user_info(self, username: str) -> Dict:
        """Get user account information"""
        url = f"https://www.reddit.com/user/{username}/about.json"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()['data']
            return {
                'created_utc': data.get('created_utc', 0),
                'comment_karma': data.get('comment_karma', 0),
                'link_karma': data.get('link_karma', 0),
                'total_karma': data.get('total_karma', 0),
                'is_gold': data.get('is_gold', False),
                'is_mod': data.get('is_mod', False),
                'verified': data.get('verified', False),
                'icon_img': data.get('icon_img', ''),
            }
        except Exception as e:
            logger.error(f"Error fetching user info for {username}: {e}")
            return {}
    
    def scrape_user_content(self, username: str, max_pages: int = 10) -> Tuple[List[Dict], List[Dict]]:
        """Scrape user posts and comments"""
        posts_data = []
        comments_data = []
        after = None
        page_count = 0
        
        while True:
            try:
                url = f"https://www.reddit.com/user/{username}.json"
                if after:
                    url += f"?after={after}"
                
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                time.sleep(self.rate_limit_delay)
                
                data = response.json()
                items = data.get('data', {}).get('children', [])
                
                if not items:
                    logger.info("No more data available")
                    break
                
                for item in items:
                    item_data = item['data']
                    
                    if item['kind'] == 't3':  # Posts
                        post_info = self._extract_post_data(item_data)
                        if self._is_valid_content(post_info.get('title', '')):
                            posts_data.append(post_info)
                    
                    elif item['kind'] == 't1':  # Comments
                        comment_info = self._extract_comment_data(item_data)
                        if self._is_valid_content(comment_info.get('body', '')):
                            comments_data.append(comment_info)
                
                after = data['data'].get('after')
                page_count += 1
                
                # Progress bar for scraping
                progress = (page_count + 1) / max_pages * 100
                filled_length = int(50 * (page_count + 1) // max_pages)
                bar = '█' * filled_length + '-' * (50 - filled_length)
                print(f'\rScraping Progress: |{bar}| {progress:.1f}% ({page_count + 1}/{max_pages} pages)', end='', flush=True)
                
                if page_count + 1 == max_pages or not after:
                    print()  # New line after completion
                    break
                    
            except Exception as e:
                logger.error(f"Error scraping page {page_count + 1}: {e}")
                break
        
        logger.info(f"Collected {len(posts_data)} posts and {len(comments_data)} comments")
        return posts_data, comments_data
    
    def _extract_post_data(self, item_data: Dict) -> Dict:
        """Extract post data from Reddit API response"""
        return {
            'title': item_data.get('title', ''),
            'selftext': item_data.get('selftext', ''),
            'subreddit': item_data.get('subreddit', ''),
            'score': item_data.get('score', 0),
            'upvote_ratio': item_data.get('upvote_ratio', 0),
            'num_comments': item_data.get('num_comments', 0),
            'created_utc': item_data.get('created_utc', 0),
            'gilded': item_data.get('gilded', 0),
            'url': f"https://reddit.com{item_data.get('permalink', '')}"
        }
    
    def _extract_comment_data(self, item_data: Dict) -> Dict:
        """Extract comment data from Reddit API response"""
        return {
            'body': item_data.get('body', ''),
            'subreddit': item_data.get('subreddit', ''),
            'score': item_data.get('score', 0),
            'created_utc': item_data.get('created_utc', 0),
            'parent_id': item_data.get('parent_id', ''),
            'gilded': item_data.get('gilded', 0),
            'controversiality': item_data.get('controversiality', 0),
            'url': f"https://reddit.com{item_data.get('permalink', '')}"
        }
    
    def _is_valid_content(self, content: str) -> bool:
        """Check if content is valid (not deleted/removed)"""
        invalid_content = {'[deleted]', '[removed]', ''}
        return content not in invalid_content




class DataProcessor:
    """Processes and analyzes Reddit data"""
    
    def __init__(self, text_cleaner: TextCleaner):
        self.text_cleaner = text_cleaner
    
    def process_user_data(self, username: str, user_info: Dict, posts_data: List[Dict], comments_data: List[Dict]) -> UserData:
        """Process raw user data into structured format"""
        # Process posts
        processed_posts = self._process_posts(posts_data)
        
        # Process comments
        processed_comments = self._process_comments(comments_data)
        
        # Calculate metrics
        metrics = self._calculate_metrics(processed_posts, processed_comments)
        
        # Extract raw text
        raw_text = self._extract_raw_text(processed_posts, processed_comments)
        
        return UserData(
            username=username,
            account_created=datetime.fromtimestamp(user_info.get('created_utc', 0), timezone.utc),
            icon_img=user_info.get('icon_img', ''),
            posts=processed_posts,
            comments=processed_comments,
            karma={
                'post_karma': user_info.get('link_karma', 0),
                'comment_karma': user_info.get('comment_karma', 0),
            },
            raw_text=raw_text,
            quality_metrics=metrics
        )
    
    def _process_posts(self, posts_data: List[Dict]) -> List[RedditPost]:
        """Process and filter posts"""
        processed_posts = []
        
        for post in posts_data:
            title = post.get('title', '')
            selftext = post.get('selftext', '')
            combined_text = f"{title} {selftext}".strip()
            
            cleaned_text = self.text_cleaner.clean_text(combined_text)
            if not cleaned_text or len(cleaned_text) < 10:
                continue
            
            quality_score = self.text_cleaner.calculate_quality_score(cleaned_text, post)
            
            if quality_score > 0:
                processed_post = RedditPost(
                    index='',
                    title=self.text_cleaner.clean_text(title),
                    selftext=self.text_cleaner.clean_text(selftext),
                    subreddit=post.get('subreddit', ''),
                    score=post.get('score', 0),
                    upvote_ratio=post.get('upvote_ratio', 0),
                    num_comments=post.get('num_comments', 0),
                    created_utc=post.get('created_utc', 0),
                    gilded=post.get('gilded', 0),
                    url=post.get('url', ''),
                    quality_score=quality_score,
                    cleaned_text=cleaned_text
                )
                processed_posts.append(processed_post)
                
        # Sort by quality score
        processed_posts.sort(key=lambda x: x.quality_score, reverse=True)
        # Add index tracking to posts for citation purposes
        for i, post in enumerate(processed_posts):
            post.index = f'POST_{i + 1}'
        return processed_posts

    def _process_comments(self, comments_data: List[Dict]) -> List[RedditComment]:
        """Process and filter comments"""
        processed_comments = []
        for comment in comments_data:
            body = comment.get('body', '')
            cleaned_text = self.text_cleaner.clean_text(body)
            if not cleaned_text or len(cleaned_text) < 10:
                continue
            quality_score = self.text_cleaner.calculate_quality_score(cleaned_text, comment)
            if quality_score > 0:
                # Create RedditComment instance
                processed_comment = RedditComment(
                    index='',
                    body=cleaned_text,
                    subreddit=comment.get('subreddit', ''),
                    score=comment.get('score', 0),
                    created_utc=comment.get('created_utc', 0),
                    parent_id=comment.get('parent_id', ''),
                    gilded=comment.get('gilded', 0),
                    controversiality=comment.get('controversiality', 0),
                    url=comment.get('url', ''),
                    quality_score=quality_score,
                    cleaned_text=cleaned_text
                )
                processed_comments.append(processed_comment)
        
        # Sort by quality score first
        processed_comments.sort(key=lambda x: x.quality_score, reverse=True)
        max_items = len(processed_comments)
        
        # Apply stratified sampling for diversity
        if len(processed_comments) > max_items:
            # Simple clustering by subreddit as proxy for thematic buckets
            subreddit_clusters = {}
            for comment in processed_comments:
                subreddit = comment.subreddit
                if subreddit not in subreddit_clusters:
                    subreddit_clusters[subreddit] = []
                subreddit_clusters[subreddit].append(comment)
            
            # Sample from each cluster proportionally
            sampled_comments = []
            items_per_cluster = max(1, max_items // len(subreddit_clusters))
            remaining_items = max_items
            
            for subreddit, cluster_comments in subreddit_clusters.items():
                if remaining_items <= 0:
                    break
                # Take top quality comments from this cluster
                sample_size = min(items_per_cluster, len(cluster_comments), remaining_items)
                sampled_comments.extend(cluster_comments[:sample_size])
                remaining_items -= sample_size

            # Randomize order to avoid bias
            random.shuffle(sampled_comments)
            processed_comments = sampled_comments
        
        # Add index tracking to comments for citation purposes
        for i, comment in enumerate(processed_comments):
            comment.index = f'COMMENT_{i + 1}'
        return processed_comments[:max_items]
    
    def _calculate_metrics(self, posts: List[RedditPost], comments: List[RedditComment]) -> UserMetrics:
        """Calculate user metrics"""
        subreddits = set()
        content_by_subreddit = {}
        
        # Process posts
        for post in posts:
            subreddit = post.subreddit
            subreddits.add(subreddit)
            
            if subreddit not in content_by_subreddit:
                content_by_subreddit[subreddit] = {'posts': 0, 'comments': 0}
            content_by_subreddit[subreddit]['posts'] += 1
        
        # Process comments
        for comment in comments:
            subreddit = comment.subreddit
            subreddits.add(subreddit)
            
            if subreddit not in content_by_subreddit:
                content_by_subreddit[subreddit] = {'posts': 0, 'comments': 0}
            content_by_subreddit[subreddit]['comments'] += 1
        
        # Calculate averages
        avg_post_score = sum(p.quality_score for p in posts) / len(posts) if posts else 0
        avg_comment_score = sum(c.quality_score for c in comments) / len(comments) if comments else 0
        
        return UserMetrics(
            total_posts=len(posts),
            total_comments=len(comments),
            high_quality_posts=len([p for p in posts if p.quality_score >= 15]),
            high_quality_comments=len([c for c in comments if c.quality_score >= 15]),
            average_post_score=avg_post_score,
            average_comment_score=avg_comment_score,
            unique_subreddits=list(subreddits),
            content_by_subreddit=content_by_subreddit
        )
    
    def _extract_raw_text(self, posts: List[RedditPost], comments: List[RedditComment]) -> Dict[str, List[str]]:
        """Extract raw text for analysis"""
        raw_text = {'posts': [], 'comments': []}
        
        for post in posts:
            if post.cleaned_text:
                raw_text['posts'].append(f'{post.index}: {post.cleaned_text}')
        
        for comment in comments:
            if comment.cleaned_text:
                raw_text['comments'].append(f'{comment.index}: {comment.cleaned_text}')
        
        return raw_text





class PersonaAnalyzer:
    """Handles LLM-based persona analysis with citations-based behavioral insights"""
    
    def __init__(self, config: Config):
        genai.configure(api_key=config.gemini_api_key)
    
    def analyze_persona(self, user_data: UserData) -> Dict:
        """Analyze user persona using LLM with citations-based approach"""
        logger.info("Analyzing persona with LLM...")
        
        prompt = self._create_analysis_prompt(user_data)
        
        # with open(f"{user_data.username}_analysis_prompt.txt", 'w', encoding='utf-8') as f:
        #     f.write(prompt)
        
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            
            # Clean the response text to extract JSON
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            persona_analysis = json.loads(response_text)
            logger.info("Persona analysis completed successfully.")
            with open(f"{user_data.username}_persona_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(persona_analysis, f, ensure_ascii=False, indent=2)
            return persona_analysis
        except Exception as e:
            logger.error(f"Error with LLM analysis: {e}")
            return self._create_fallback_analysis(user_data)
    

    def _create_analysis_prompt(self, user_data: UserData) -> str:
        """Create robust prompt for deep LLM analysis with multiple insights per category"""
        account_age = (datetime.now(timezone.utc) - user_data.account_created).days
        quality_stats = user_data.quality_metrics
        
        prompt = f"""
        Analyze this Reddit user's content for comprehensive behavioral profiling. Extract MULTIPLE insights per category based on actual citations.

        CRITICAL INSTRUCTIONS:
        1. Try analyzing the comments in the context of the posts they are responding to, as this will provide deeper insights into the user's behavior and personality.
        2. Consider the subreddits visited, analyze the comments exploit intrests and hobbies , you can also try to infer user location based on the timezones of user posts and comments.
        3. Analyze the user's behavior over time, looking for patterns, changes, or contradictions
        4. Generate 3-6 insights per category (not just one)
        2. Base ALL insights on specific POST-X or COMMENT-X citations
        5. Analyze patterns, contradictions, and evolution over time
        6. Focus on observable behaviors, not assumptions

        User: {user_data.username} | Age: {account_age} days | Posts: {len(user_data.posts)} | Comments: {len(user_data.comments)}
        
        IMPORTANT TIP: Use th blow visited subreddits to infer interests and hobbies, and analyze what is the subreddit is about search for the WEB or INTERNET to analyse the data on that subreddit and gaspthe user hobbies or digital behaviours.
        
        Top Subreddits: {', '.join(quality_stats.unique_subreddits)}

        POSTS:
        {chr(10).join(user_data.raw_text['posts'])}

        COMMENTS: 
        {chr(10).join(user_data.raw_text['comments'][:100])}

        Respond with ONLY valid JSON:

        {{
            "username": "{user_data.username}",
            "account_age_days": {account_age},
            "authentic_quote": "A direct quote from their content that captures their voice",
            "confidence_level": "high/medium/low",
            
            "basic_info": {{
                "estimated_age": "25 (Inferred from text tone and maturity)",
                "occupation": "Software Engineer (Based on technical content and terminology used)",
                "location": "Urban area (Based on UTC timing and cultural references)",
                "relationship_status": {{
                "value": "Single",
                "citations": ["COMMENT_2", "COMMENT_15"],
                "reasoning": "Personal references suggest a single lifestyle"
                }},
                "tech_savviness": {{
                "value": "8",
                "citations": ["POST_1", "POST_3", "COMMENT_7"],
                "reasoning": "Advanced problem-solving and technical discourse"
                }},
                "education_level": {{
                "value": "Bachelor's degree",
                "citations": ["POST_5", "COMMENT_9"],
                "reasoning": "Level of depth and technical clarity in communication"
                }}
            }},
            "personality_insights": [
                {{
                    "trait": "Analytical problem-solver",
                    "evidence": "Breaks down complex issues systematically",
                    "citations": ["POST_2", "COMMENT_5"],
                    "manifestation": "Step-by-step explanations, logical flow"
                }},
                {{
                    "trait": "Knowledge sharer",
                    "evidence": "Actively helps others with detailed explanations",
                    "citations": ["COMMENT_1", "COMMENT_3"],
                    "manifestation": "Unprompted assistance, educational tone"
                }},
                {{
                    "trait": "Curious learner",
                    "evidence": "Asks thoughtful questions, seeks understanding",
                    "citations": ["POST_1", "COMMENT_4"],
                    "manifestation": "Follow-up questions, explores edge cases"
                }},
                {{
                    "trait": "Introverted but engaged",
                    "evidence": "Participates meaningfully but not socially",
                    "citations": ["COMMENT_2", "POST_3"],
                    "manifestation": "Task-focused interactions, limited personal sharing"
                }}
                
                "dominant_traits": ["analytical", "curious", "independent"],
                "openness": 8,
                "conscientiousness": 7,
                "agreeableness": 6,
                "neuroticism": 4,
                "introversion_extroversion": 4,
                "communication_style": "Direct and informative with occasional humor"
            ],

            "behavioral_patterns": [
                {{
                    "pattern": "He consistently breaks down complex technical problems into manageable steps because he has learned from experience that systematic approaches reduce errors and help others understand solutions better, as evidenced by his methodical responses in troubleshooting threads where he walks through each debugging step.",
                    "citations": ["POST_1", "COMMENT_2", "COMMENT_5"],
                    "frequency": "Very frequent",
                    "typical_phrases": ["'Professional experience and desire to be helpful'"]
                }},
                {{
                    "pattern": "<username> proactively helps other developers with detailed explanations because he values knowledge sharing and believes it strengthens the community, often spending significant time crafting comprehensive answers even for basic questions.",
                    "citations": ["COMMENT_1", "COMMENT_3", "COMMENT_7"],
                    "frequency": "Regular",
                    "typical_phrases": ["'Community building and professional reputation'"]
                }},
                {{
                    "pattern": "asks highly specific, research-oriented questions because he prefers to understand the fundamental principles behind technologies rather than just implementing quick fixes, showing his commitment to deep learning over surface-level knowledge.",
                    "citations": ["POST_2", "COMMENT_4"],
                    "frequency": "Moderate",
                    "typical_phrases": ["'Intellectual curiosity and professional thoroughness'"]
                }},
                {{
                    "pattern": "He uses structured, well-formatted communication with examples because he understands that clear documentation saves time for everyone involved and reflects his professional standards for technical communication.",
                    "citations": ["COMMENT_2", "COMMENT_6"],
                    "frequency": "Consistent",
                    "typical_phrases": ["'Professional standards and efficiency mindset'"]
                }}
            ],

            "communication_habits": [
                {{
                    "habit": "Educational explanations with examples",
                    "description": "Provides context and practical examples",
                    "citations": ["COMMENT_1", "COMMENT_5"],
                    "typical_phrases": ["For example", "Here's how", "Try this approach"]
                }},
                {{
                    "habit": "Respectful disagreement",
                    "description": "Challenges ideas professionally without personal attacks",
                    "citations": ["COMMENT_3", "POST_2"],
                    "typical_phrases": ["I think there might be", "Have you considered"]
                }},
                {{
                    "habit": "Follow-up clarification",
                    "description": "Asks clarifying questions before providing solutions",
                    "citations": ["COMMENT_4", "COMMENT_8"],
                    "typical_phrases": ["Could you clarify", "What specifically"]
                }}
            ],

            "goals_and_needs": [
                {{
                    "goal": "He is actively working to advance his career in software engineering because he recognizes that the tech industry rewards continuous learning and staying current with emerging technologies, which drives his frequent participation in technical discussions and his methodical approach to mastering new frameworks.",
                    "citations": ["POST_1", "COMMENT_2"],
                    "certainty": "High",
                    "supporting_context": "Career advancement and financial security"
                }},
                {{
                    "goal": "He wants to build a strong professional reputation in the developer community because he understands that peer recognition opens doors to better opportunities and validates his technical expertise, leading him to consistently provide high-quality, helpful responses to other developers' questions.",
                    "citations": ["COMMENT_1", "COMMENT_5"],
                    "certainty": "Medium",
                    "supporting_context": "Professional validation and network building"
                }},
                {{
                    "goal": "He seeks technical validation and feedback from peers because he values accuracy and wants to ensure his solutions are optimal, which is why he often asks for code reviews and presents his approaches for community discussion rather than just implementing them silently.",
                    "citations": ["POST_2", "COMMENT_4"],
                    "certainty": "Medium",
                    "supporting_context": "Quality assurance and professional growth"
                }},
                {{
                    "goal": "He needs to stay current with rapidly evolving industry trends because he fears becoming obsolete in the fast-paced tech world, which motivates his regular engagement with cutting-edge topics and his active participation in discussions about emerging technologies.",
                    "citations": ["POST_3", "COMMENT_6"],
                    "certainty": "High",
                    "supporting_context": "Professional survival and competitive advantage"
                }}
            ],

            "digital_behaviors": [
                {{
                    "behavior": "Active in developer communities",
                    "manifestation": "Regular contributions to technical subreddits",
                    "citations": ["POST_1", "COMMENT_2"],
                    "platform_specific": "Reddit, likely GitHub/Stack Overflow"
                }},
                {{
                    "behavior": "Research-driven posting",
                    "manifestation": "Well-researched questions and responses",
                    "citations": ["POST_2", "COMMENT_3"],
                    "platform_specific": "Technical forums, documentation sites"
                }},
                {{
                    "behavior": "Scheduled engagement patterns",
                    "manifestation": "Consistent posting times, suggesting routine",
                    "citations": ["POST_1", "COMMENT_1"],
                    "platform_specific": "Professional/work-related timing"
                }}
            ],

            "psychological_insights": [
                {{
                    "insight": "He demonstrates high conscientiousness and attention to detail because he has developed strong professional habits that prioritize accuracy and thoroughness over speed, which is evident in his methodical approach to problem-solving and his tendency to double-check his work before sharing solutions with others.",
                    "citations": ["COMMENT_2", "COMMENT_5"],
                    "implications": "Reliable, methodical approach to work and learning",
                    "underlying_psychology": "Perfectionist tendencies and professional pride"
                }},
                {{
                    "insight": "He exhibits a growth mindset and intellectual humility because he understands that admitting knowledge gaps and seeking feedback leads to better long-term outcomes, which is why he regularly asks questions, acknowledges when he's unsure, and updates his understanding based on new information.",
                    "citations": ["POST_2", "COMMENT_4"],
                    "implications": "Adaptable, continuous learner, good team player",
                    "underlying_psychology": "Secure self-confidence and learning orientation"
                }},
                {{
                    "insight": "He is more task-oriented than socially motivated because he finds satisfaction in solving problems and achieving technical goals rather than building personal relationships, which leads him to focus conversations on technical solutions while maintaining professional but not deeply personal interactions.",
                    "citations": ["COMMENT_1", "COMMENT_3"],
                    "implications": "Efficient but may need encouragement for collaboration",
                    "underlying_psychology": "Introverted nature and technical achievement focus"
                }}
            ],

            "interests_and_expertise": [
                {{
                    "interest_or_skill": "He is deeply interested in Machine Learning and AI because he believes these technologies will fundamentally reshape software development and wants to be at the forefront of this transformation, which drives his active participation in AI-related discussions and his commitment to understanding both theoretical concepts and practical implementations.",
                    "citations": ["POST_1", "COMMENT_2"],
                    "engagement_level": "High - active learning and application",
                    "underlying_motivation": "Future career positioning and intellectual fascination"
                }},
                {{
                    "interest_or_skill": "He has developed strong expertise in Software Architecture because he has learned that well-designed systems prevent major problems down the line and he takes pride in creating elegant, scalable solutions, which is why he frequently engages in discussions about system design patterns and architectural best practices.",
                    "citations": ["POST_2", "COMMENT_5"],
                    "engagement_level": "Very High - professional application",
                    "underlying_motivation": "Professional craftsmanship and problem prevention"
                }},
                {{
                    "interest_or_skill": "He maintains active interest in Web Development because it provides immediate, visible results for his work and he enjoys the creative aspects of building user-facing applications, though he focuses more on the technical implementation than the visual design aspects.",
                    "citations": ["COMMENT_3", "COMMENT_6"],
                    "engagement_level": "Medium - practical application",
                    "underlying_motivation": "Creative expression and tangible results"
                }}
            ],

            "content_analysis_summary": {{
                "total_analyzed": {len(user_data.raw_text['posts']) + len(user_data.raw_text['comments'])},
                "analysis_confidence": "Medium-High",
                "key_patterns": [
                    "Consistent technical expertise demonstration",
                    "Helpful, educational communication style",
                    "Professional development focus"
                ],
                "limitations": [
                    "Limited personal/emotional content",
                    "No direct demographic information",
                    "Professional persona may mask personal traits"
                ],
                "reliability_factors": [
                    "Consistent behavior across multiple contexts",
                    "Coherent skill level demonstration",
                    "Natural language patterns"
                ]
            }}
        }}
    """
        
        return prompt.strip()
    
    def _create_fallback_analysis(self, user_data: UserData) -> Dict:
        """Create fallback analysis when LLM fails"""
        account_age = (datetime.now(timezone.utc) - user_data.account_created).days
        
        return {
            "username": "{user_data.username}",
            "account_age_days": {account_age},
            "authentic_quote": "Not available – content insufficient or unreadable for quote extraction",
            "confidence_level": "low",

            "basic_info": {
                "estimated_age": "Unknown",
                "occupation": "Unknown",
                "location": "Unknown",
                "relationship_status": {
                "value": "Unknown",
                "citations": [],
                "reasoning": "Insufficient data"
                },
                "tech_savviness": {
                "value": "Unknown",
                "citations": [],
                "reasoning": "Insufficient data"
                },
                "education_level": {
                "value": "Unknown",
                "citations": [],
                "reasoning": "Insufficient data"
                }
            },

            "behavioral_patterns": [
                {
                "pattern": "No behavioral patterns identified",
                "citations": [],
                "typical_phrases": [],
                "frequency": "unknown"
                }
            ],

            "communication_habits": [
                {
                "habit": "No observable communication habits",
                "citations": [],
                "examples": [],
                "context": "unknown"
                }
            ],

            "goals_and_needs": [
                {
                "goal": "No identifiable goals or motivational cues",
                "citations": [],
                "supporting_context": "unknown",
                "certainty": "low"
                }
            ],

            "digital_behaviors": [
                {
                "behavior": "Digital behavior patterns could not be determined",
                "citations": [],
                "manifestation": "unknown",
                "platform_specific": "unknown"
                }
            ],

            "psychological_insights": [
                {
                "insight": "Unable to infer psychological traits",
                "citations": [],
                "behavioral_indicators": [],
                "confidence": "low"
                }
            ],

            "interests_and_expertise": [
                {
                "interest_or_skill": "Not enough information to determine interests or skills",
                "citations": [],
                "underlying_motivation": "unknown",
                "engagement_level": "unknown"
                }
            ],

            "content_analysis_summary": {
                "total_content_pieces": {len(user_data.raw_text.get('posts', [])) + len(user_data.raw_text.get('comments', []))},
                "analysis_confidence": "very low",
                "key_patterns": [
                "No analyzable text content available",
                "Posts/comments may be too short, deleted, or malformed",
                "Unable to parse meaningful context"
                ],
                "limitations": [
                "No patterns identified due to input failure"
                ],
                "reliability_factors": [
                "No reliable data available for analysis",
                "Content may be too sparse or malformed"
                ]
            },
                "system_generated": {
                    "ACCOUNT_INFO": f"Account created {account_age} days ago",
                    "ACTIVITY_STATS": f"Total posts: {len(user_data.posts)}, Total comments: {len(user_data.comments)}",
                    "SUBREDDIT_ACTIVITY": f"Active subreddits: {', '.join(user_data.quality_metrics.unique_subreddits)}",
                    "ACTIVITY_PATTERN": f"Posting frequency based on {len(user_data.posts + user_data.comments)} items over {account_age} days",
                    "PLATFORM_USAGE": "Known to use Reddit platform"
                },
            "fallback_reason": "LLM analysis failed - using basic statistical analysis only",
            "data_available": {
                "posts_count": len(user_data.posts),
                "comments_count": len(user_data.comments),
                "subreddits_count": len(user_data.quality_metrics.unique_subreddits),
                "account_age_days": account_age,
                "has_karma_data": bool(user_data.karma.get('post_karma', 0) or user_data.karma.get('comment_karma', 0))
            }
        }
        
        
        

class ReportGenerator:
    """Enhanced persona report generator with citation-based evidence retrieval"""
    
    def __init__(self):
        self.report_template = self._load_report_template()
    
    def generate_comprehensive_report(self, user_data: UserData, persona_analysis: Dict, 
                                    output_format: str = 'text', filename: Optional[str] = None) -> str:
        """
        Generate a comprehensive persona report with citation evidence
        
        Args:
            user_data: User data object containing posts, comments, and metrics
            persona_analysis: LLM analysis results with citations
            output_format: 'text', or 'json'
            filename: Optional custom filename
            
        Returns:
            Generated report content or filename if saved
        """
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{user_data.username}_persona_report_{timestamp}.{output_format}"
        
        # Calculate derived metrics
        account_age = (datetime.now(timezone.utc) - user_data.account_created).days
        total_karma = user_data.karma.get('post_karma', 0) + user_data.karma.get('comment_karma', 0)
        
        # Generate report based on format
        if output_format == 'text':
            report_content = self._generate_text_report(user_data, persona_analysis, account_age, total_karma)
        elif output_format == 'json':
            report_content = self._generate_json_report(user_data, persona_analysis, account_age, total_karma)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Save report to file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Report generated successfully: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return report_content
    
    def _retrieve_citation_data(self, user_data: UserData, citations: List[str]) -> Dict[str, Dict]:
        """
        Retrieve actual post/comment data based on citations
        
        Args:
            user_data: User data object
            citations: List of citation strings like ['POST_1', 'COMMENT_2']
            
        Returns:
            Dictionary with citation data
        """
        citation_data = {}
        
        for citation in citations:
            if citation.startswith('POST_'):
                try:
                    post_index = int(citation.split('_')[1]) - 1  # Convert to 0-based index
                    if 0 <= post_index < len(user_data.posts):
                        post = user_data.posts[post_index]
                        citation_data[citation] = {
                            'type': 'post',
                            'title': post.title,
                            'subreddit': post.subreddit,
                            'content': post.selftext,
                            'url': post.url
                        }
                except (ValueError, IndexError, KeyError):
                    citation_data[citation] = {
                        'type': 'post',
                        'title': 'Citation not found',
                        'subreddit': 'Unknown',
                        'content': 'Unable to retrieve post data',
                        'url': ''
                    }
            
            elif citation.startswith('COMMENT_'):
                try:
                    comment_index = int(citation.split('_')[1]) - 1  # Convert to 0-based index
                    if 0 <= comment_index < len(user_data.comments):
                        comment = user_data.comments[comment_index]
                        citation_data[citation] = {
                            'type': 'comment',
                            'subreddit': comment.subreddit,
                            'comment': comment.body,
                            'url': comment.url
                        }
                except (ValueError, IndexError, KeyError):
                    citation_data[citation] = {
                        'type': 'comment',
                        'subreddit': 'Unknown',
                        'comment': 'Unable to retrieve comment data',
                        'url': ''
                    }
        
        return citation_data
    
    def _generate_text_report(self, user_data: UserData, persona_analysis: Dict, 
                             account_age: int, total_karma: int) -> str:
        """Generate professional text report with citation evidence"""
        
        report_sections = []
        
        # Header Section
        header = f"""
{'=' * 60}
COMPREHENSIVE USER PERSONA ANALYSIS REPORT
{'=' * 60}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Analysis Method: LLM-Enhanced Persona Generation with Citations
Confidence Level: {persona_analysis.get('confidence_level', 'Medium')}
"""
        report_sections.append(header)
        
        # Executive Summary
        executive_summary = self._generate_executive_summary(user_data, persona_analysis, account_age, total_karma)
        report_sections.append(executive_summary)
        
        # Core Demographics with Citations
        demographics = self._generate_demographics_section(user_data, persona_analysis)
        report_sections.append(demographics)
        
        # Personality Profile with Citations
        personality = self._generate_personality_section(user_data, persona_analysis)
        report_sections.append(personality)
        
        # Behavioral Patterns with Evidence
        behavior = self._generate_behavior_section(user_data, persona_analysis)
        report_sections.append(behavior)
        
        # Communication Habits with Examples
        communication = self._generate_communication_section(user_data, persona_analysis)
        report_sections.append(communication)
        
        # Goals & Motivations with Evidence
        goals = self._generate_goals_section(user_data, persona_analysis)
        report_sections.append(goals)
        
        # Digital Behavior with Citations
        digital = self._generate_digital_behavior_section(user_data, persona_analysis)
        report_sections.append(digital)
        
        # Psychological Insights with Supporting Evidence
        psychological = self._generate_psychological_insights_section(user_data, persona_analysis)
        report_sections.append(psychological)
        
        # Interests & Expertise with Citations
        interests = self._generate_interests_section(user_data, persona_analysis)
        report_sections.append(interests)
        
        # Representative Quote
        quote_section = f"""
REPRESENTATIVE VOICE SAMPLE
{'=' * 30}
"{persona_analysis.get('authentic_quote', 'No representative quote available')}"
"""
        report_sections.append(quote_section)
        
        # Citation Evidence Appendix
        citations_appendix = self._generate_citations_appendix(user_data, persona_analysis)
        report_sections.append(citations_appendix)
        
        # Methodology & Data Sources
        methodology = self._generate_methodology_section(user_data, account_age)
        report_sections.append(methodology)
        
        return '\n'.join(report_sections)
    
    def _generate_executive_summary(self, user_data: UserData, persona_analysis: Dict, 
                                   account_age: int, total_karma: int) -> str:
        """Generate executive summary section"""
        
        basic_info = persona_analysis.get('basic_info', {})
        personality = persona_analysis.get('personality_insights', {})
        
        return f"""
EXECUTIVE SUMMARY
{'=' * 20}
Username: {user_data.username}
Account Age: {account_age} days ({account_age // 365} years, {account_age % 365} days)
Content Volume: {len(user_data.posts)} posts, {len(user_data.comments)} comments
Engagement Score: {total_karma:,} combined karma
Activity Level: {self._calculate_activity_level(user_data, account_age)}

Profile Overview:
• Estimated Age: {basic_info.get('estimated_age', 'Unknown')}
• Occupation: {basic_info.get('occupation', 'Unknown')}
• Tech Proficiency: {self._get_nested_value(basic_info, 'tech_savviness')}/10
• Primary Interests: {self._get_primary_interests_summary(persona_analysis)}
• Dominant Traits: {self._get_dominant_traits_summary(personality)}
"""
    
    def _generate_demographics_section(self, user_data: UserData, persona_analysis: Dict) -> str:
        """Generate demographics section with citations"""
        
        basic_info = persona_analysis.get('basic_info', {})
        
        section = f"""
DEMOGRAPHIC PROFILE
{'=' * 20}
Age Range: {basic_info.get('estimated_age', 'Unknown')}
Occupation: {self._format_field_with_citation(user_data, basic_info, 'occupation')}
Education: {self._format_field_with_citation(user_data, basic_info, 'education_level')}
Location: {basic_info.get('location', 'Unknown')}
Relationship Status: {self._format_field_with_citation(user_data, basic_info, 'relationship_status')}
Technical Expertise: {self._format_field_with_citation(user_data, basic_info, 'tech_savviness')}
"""
        return section
    
    def _generate_personality_section(self, user_data: UserData, persona_analysis: Dict) -> str:
        """Generate personality analysis section with citations"""
        
        personality = persona_analysis.get('personality_insights', {})
        
        # Handle both list and dict formats for personality insights
        if isinstance(personality, list):
            # Extract traits from list format
            traits = []
            for trait_info in personality:
                if isinstance(trait_info, dict):
                    trait_name = trait_info.get('trait', 'Unknown')
                    traits.append(trait_name)
            
            section = f"""
PERSONALITY ANALYSIS
{'=' * 20}

Identified Personality Traits:
"""
            for i, trait_info in enumerate(personality, 1):
                if isinstance(trait_info, dict):
                    trait = trait_info.get('trait', 'Unknown')
                    evidence = trait_info.get('evidence', 'No evidence provided')
                    manifestation = trait_info.get('manifestation', 'No manifestation described')
                    citations = trait_info.get('citations', [])
                    
                    section += f"""
{i}. {trait}
   Evidence: {evidence}
   Manifestation: {manifestation}
"""
                    if citations:
                        citation_data = self._retrieve_citation_data(user_data, citations)
                        section += self._format_citation_evidence(citation_data, indent="   ")
        else:
            # Handle dict format (legacy support)
            section = f"""
PERSONALITY ANALYSIS
{'=' * 20}

Big Five Personality Dimensions (1-10 scale):
• Openness to Experience: {personality.get('openness', 'Unknown')}
• Conscientiousness: {personality.get('conscientiousness', 'Unknown')}
• Extroversion: {personality.get('introversion_extroversion', 'Unknown')}
• Agreeableness: {personality.get('agreeableness', 'Unknown')}
• Neuroticism: {personality.get('neuroticism', 'Unknown')}

Dominant Traits: {', '.join(personality.get('dominant_traits', []))}
Communication Style: {personality.get('communication_style', 'Unknown')}
"""
        
        return section
    
    def _generate_behavior_section(self, user_data: UserData, persona_analysis: Dict) -> str:
        """Generate behavioral patterns section with evidence"""
        
        behavioral_patterns = persona_analysis.get('behavioral_patterns', [])
        
        section = f"""
BEHAVIORAL PATTERNS
{'=' * 20}
"""
        
        for i, pattern in enumerate(behavioral_patterns, 1):
            pattern_text = pattern.get('pattern', 'No pattern identified')
            frequency = pattern.get('frequency', 'Unknown')
            citations = pattern.get('citations', [])
            typical_phrases = pattern.get('typical_phrases', [])
            
            section += f"""
Pattern {i}: {pattern_text}
Frequency: {frequency}
"""
            
            if typical_phrases:
                section += f"Key Phrases: {'; '.join(typical_phrases)}\n"
            
            # Add citation details
            if citations:
                citation_data = self._retrieve_citation_data(user_data, citations)
                section += self._format_citation_evidence(citation_data)
            
            section += "\n"
        
        return section
    
    def _generate_communication_section(self, user_data: UserData, persona_analysis: Dict) -> str:
        """Generate communication habits section with examples"""
        
        communication_habits = persona_analysis.get('communication_habits', [])
        
        section = f"""
COMMUNICATION HABITS
{'=' * 20}
"""
        
        for i, habit in enumerate(communication_habits, 1):
            habit_text = habit.get('habit', 'No habit identified')
            description = habit.get('description', 'Unknown')
            citations = habit.get('citations', [])
            typical_phrases = habit.get('typical_phrases', [])
            
            section += f"""
Habit {i}: {habit_text}
Description: {description}
"""
            
            if typical_phrases:
                section += f"Typical Phrases: {'; '.join(typical_phrases)}\n"
            
            # Add citation details
            if citations:
                citation_data = self._retrieve_citation_data(user_data, citations)
                section += self._format_citation_evidence(citation_data)
            
            section += "\n"
        
        return section
    
    def _generate_goals_section(self, user_data: UserData, persona_analysis: Dict) -> str:
        """Generate goals and motivations section with evidence"""
        
        goals_and_needs = persona_analysis.get('goals_and_needs', [])
        
        section = f"""
GOALS & MOTIVATIONS
{'=' * 20}
"""
        
        for i, goal in enumerate(goals_and_needs, 1):
            goal_text = goal.get('goal', 'No goal identified')
            certainty = goal.get('certainty', 'Unknown')
            citations = goal.get('citations', [])
            supporting_context = goal.get('supporting_context', '')
            
            section += f"""
Goal {i}: {goal_text}
Certainty Level: {certainty}
"""
            
            if supporting_context:
                section += f"Supporting Context: {supporting_context}\n"
            
            # Add citation details
            if citations:
                citation_data = self._retrieve_citation_data(user_data, citations)
                section += self._format_citation_evidence(citation_data)
            
            section += "\n"
        
        return section
    
    def _generate_digital_behavior_section(self, user_data: UserData, persona_analysis: Dict) -> str:
        """Generate digital behavior section with citations"""
        
        digital_behaviors = persona_analysis.get('digital_behaviors', [])
        
        section = f"""
DIGITAL BEHAVIOR PROFILE
{'=' * 25}
"""
        
        for i, behavior in enumerate(digital_behaviors, 1):
            behavior_text = behavior.get('behavior', 'No behavior identified')
            manifestation = behavior.get('manifestation', 'Unknown')
            platform_specific = behavior.get('platform_specific', 'Unknown')
            citations = behavior.get('citations', [])
            
            section += f"""
Behavior {i}: {behavior_text}
Manifestation: {manifestation}
Platform Context: {platform_specific}
"""
            
            # Add citation details
            if citations:
                citation_data = self._retrieve_citation_data(user_data, citations)
                section += self._format_citation_evidence(citation_data)
            
            section += "\n"
        
        return section
    
    def _generate_psychological_insights_section(self, user_data: UserData, persona_analysis: Dict) -> str:
        """Generate psychological insights section with supporting evidence"""
        
        psychological_insights = persona_analysis.get('psychological_insights', [])
        
        section = f"""
PSYCHOLOGICAL INSIGHTS
{'=' * 22}
"""
        
        for i, insight in enumerate(psychological_insights, 1):
            insight_text = insight.get('insight', 'No insight identified')
            implications = insight.get('implications', 'Unknown')
            citations = insight.get('citations', [])
            underlying_psychology = insight.get('underlying_psychology', '')
            
            section += f"""
Insight {i}: {insight_text}
Implications: {implications}
"""
            
            if underlying_psychology:
                section += f"Underlying Psychology: {underlying_psychology}\n"
            
            # Add citation details
            if citations:
                citation_data = self._retrieve_citation_data(user_data, citations)
                section += self._format_citation_evidence(citation_data)
            
            section += "\n"
        
        return section
    
    def _generate_interests_section(self, user_data: UserData, persona_analysis: Dict) -> str:
        """Generate interests and expertise section with citations"""
        
        interests_and_expertise = persona_analysis.get('interests_and_expertise', [])
        
        section = f"""
INTERESTS & EXPERTISE PROFILE
{'=' * 30}
"""
        
        for i, interest in enumerate(interests_and_expertise, 1):
            interest_text = interest.get('interest_or_skill', 'No interest identified')
            engagement_level = interest.get('engagement_level', 'Unknown')
            citations = interest.get('citations', [])
            underlying_motivation = interest.get('underlying_motivation', '')
            
            section += f"""
Interest {i}: {interest_text}
Engagement Level: {engagement_level}
"""
            
            if underlying_motivation:
                section += f"Underlying Motivation: {underlying_motivation}\n"
            
            # Add citation details
            if citations:
                citation_data = self._retrieve_citation_data(user_data, citations)
                section += self._format_citation_evidence(citation_data)
            
            section += "\n"
        
        return section
    
    def _format_citation_evidence(self, citation_data: Dict[str, Dict], indent: str = "") -> str:
        """Format citation evidence for display"""
        
        if not citation_data:
            return ""
        
        evidence_section = f"\n{indent}Supporting Evidence:\n"
        evidence_section += f"{indent}" + "-" * 20 + "\n"
        
        for citation, data in citation_data.items():
            if data['type'] == 'post':
                content = data.get('content', '')
                truncated_content = content[:200] + '...' if len(content) > 200 else content
                evidence_section += f"""
{indent}{citation}:
{indent}  Title: {data.get('title', 'No title')}
{indent}  Subreddit: r/{data.get('subreddit', 'Unknown')}
{indent}  Content: {truncated_content}
{indent}  URL: {data.get('url', 'No URL')}
"""
            elif data['type'] == 'comment':
                comment = data.get('comment', '')
                truncated_comment = comment[:200] + '...' if len(comment) > 200 else comment
                evidence_section += f"""
{indent}{citation}:
{indent}  Subreddit: r/{data.get('subreddit', 'Unknown')}
{indent}  Comment: {truncated_comment}
{indent}  URL: {data.get('url', 'No URL')}
"""
        
        return evidence_section
    
    def _generate_citations_appendix(self, user_data: UserData, persona_analysis: Dict) -> str:
        """Generate comprehensive citations appendix"""
        
        section = f"""
CITATIONS APPENDIX
{'=' * 18}

This section provides the complete source material for all citations used in the analysis.
"""
        
        # Collect all citations from the analysis
        all_citations = set()
        
        # Handle both list and dict formats for personality insights
        personality_insights = persona_analysis.get('personality_insights', [])
        if isinstance(personality_insights, list):
            for trait_info in personality_insights:
                if isinstance(trait_info, dict):
                    citations = trait_info.get('citations', [])
                    all_citations.update(citations)
        
        # Collect citations from other sections
        for section_name in ['behavioral_patterns', 'communication_habits', 'goals_and_needs', 
                           'digital_behaviors', 'psychological_insights', 'interests_and_expertise']:
            section_data = persona_analysis.get(section_name, [])
            if isinstance(section_data, list):
                for item in section_data:
                    citations = item.get('citations', [])
                    all_citations.update(citations)
        
        # Also check basic_info for nested citations
        basic_info = persona_analysis.get('basic_info', {})
        for key, value in basic_info.items():
            if isinstance(value, dict) and 'citations' in value:
                all_citations.update(value['citations'])
        
        # Retrieve and format all citation data
        if all_citations:
            citation_data = self._retrieve_citation_data(user_data, list(all_citations))
            
            # Separate posts and comments
            post_citations = {k: v for k, v in citation_data.items() if v['type'] == 'post'}
            comment_citations = {k: v for k, v in citation_data.items() if v['type'] == 'comment'}
            
            if post_citations:
                section += "\nPOST CITATIONS:\n"
                section += "-" * 15 + "\n"
                
                for citation, data in post_citations.items():
                    section += f"""
{citation}:
  Title: {data.get('title', 'No title')}
  Subreddit: r/{data.get('subreddit', 'Unknown')}
  Content: {data.get('content', 'No content')}
  URL: {data.get('url', 'No URL')}
  Score: {data.get('score', 0)}
"""
            
            if comment_citations:
                section += "\nCOMMENT CITATIONS:\n"
                section += "-" * 18 + "\n"
                
                for citation, data in comment_citations.items():
                    section += f"""
{citation}:
  Subreddit: r/{data.get('subreddit', 'Unknown')}
  Comment: {data.get('comment', 'No comment')}
  URL: {data.get('url', 'No URL')}
  Score: {data.get('score', 0)}
"""
        else:
            section += "\nNo citations available in the analysis.\n"
        
        return section
    
    def _format_field_with_citation(self, user_data: UserData, section: Dict, field: str) -> str:
        """Format a field with its citation information and evidence"""
        
        if field not in section:
            return "Not analyzed"
        
        field_data = section[field]
        if isinstance(field_data, dict):
            value = field_data.get('value', 'Unknown')
            reasoning = field_data.get('reasoning', '')
            
            if isinstance(value, list):
                value = ', '.join(str(v) for v in value)
            
            result = str(value)
            if reasoning:
                result += f" ({reasoning})"
            
            return result
        else:
            return str(field_data)
    
    def _get_nested_value(self, section: Dict, field: str) -> str:
        """Get value from nested dict structure"""
        
        if field not in section:
            return "Not analyzed"
        
        field_data = section[field]
        if isinstance(field_data, dict):
            value = field_data.get('value', 'Unknown')
            if isinstance(value, list):
                return ', '.join(str(v) for v in value)
            return str(value)
        else:
            return str(field_data)
    
    def _get_primary_interests_summary(self, persona_analysis: Dict) -> str:
        """Get a summary of primary interests"""
        
        interests = persona_analysis.get('interests_and_expertise', [])
        if interests:
            primary_interests = []
            for interest in interests[:3]:  # Get top 3
                if isinstance(interest, dict):
                    interest_text = interest.get('interest_or_skill', '')
                    # Extract the first part before "because" for brevity
                    if interest_text and 'because' in interest_text:
                        interest_text = interest_text.split('because')[0].strip()
                    if interest_text:
                        primary_interests.append(interest_text)
            return ', '.join(primary_interests) if primary_interests else "Not identified"
        
        return "Not analyzed"
    
    def _get_dominant_traits_summary(self, personality: Dict) -> str:
        """Get dominant traits summary, handling both list and dict formats"""
        
        if isinstance(personality, list):
            # Extract traits from list format
            traits = []
            for trait_info in personality:
                if isinstance(trait_info, dict):
                    trait_name = trait_info.get('trait', '')
                    if trait_name:
                        traits.append(trait_name)
            return ', '.join(traits[:3]) if traits else "Not identified"
        else:
            # Handle dict format
            return ', '.join(personality.get('dominant_traits', []))
    
    def _calculate_activity_level(self, user_data: UserData, account_age: int) -> str:
        """Calculate and categorize activity level"""
        
        if account_age == 0:
            return "New account"
        
        total_content = len(user_data.posts) + len(user_data.comments)
        daily_average = total_content / account_age
        
        if daily_average > 5:
            return "Very Active"
        elif daily_average > 2:
            return "Active"
        elif daily_average > 0.5:
            return "Moderate"
        else:
            return "Low Activity"
    
    def _generate_methodology_section(self, user_data: UserData, account_age: int) -> str:
        """Generate methodology and data sources section"""
        
        unique_subreddits = len(set([post.subreddit for post in user_data.posts] + 
                                  [comment.subreddit for comment in user_data.comments]))
        
        return f"""
METHODOLOGY & DATA SOURCES
{'=' * 28}
Analysis Framework:
• Reddit API data extraction and preprocessing
• LLM-powered persona analysis with citation requirements
• Statistical pattern recognition
• Evidence-based behavioral profiling
• Natural language processing for content themes

Data Sources Summary:
• Posts Analyzed: {len(user_data.posts):,}
• Comments Analyzed: {len(user_data.comments):,}
• Unique Subreddits: {unique_subreddits}
• Account History: {account_age} days
• Content Quality Score: {self._calculate_content_quality_score(user_data)}

Citation System:
• All behavioral insights are backed by specific post/comment citations
• POST_X refers to the X-th post in chronological order
• COMMENT_X refers to the X-th comment in chronological order
• Full citation evidence is provided in the appendix

Confidence Assessment:
• High confidence: Multiple supporting citations with consistent patterns
• Medium confidence: Some supporting evidence with reasonable inferences
• Low confidence: Limited citations or conflicting signals

Limitations:
• Analysis based on public Reddit activity only
• Personality assessments are estimates, not clinical diagnoses
• Cultural and contextual factors may influence interpretation
• Temporal changes in behavior patterns may not be fully captured
• Citation availability depends on content accessibility and indexing
"""
    
    def _calculate_content_quality_score(self, user_data: UserData) -> str:
        """Calculate overall content quality score"""
        
        if hasattr(user_data, 'quality_metrics'):
            # This would need to be implemented based on your quality metrics
            return "Medium-High"
        return "Not calculated"
    
    def _generate_json_report(self, user_data: UserData, persona_analysis: Dict, 
                             account_age: int, total_karma: int) -> str:
        """Generate JSON formatted report with citation evidence"""
        
        # Collect all citations and their evidence
        all_citations = set()
        for section_name in ['behavioral_patterns', 'communication_habits', 'goals_and_needs', 
                           'digital_behaviors', 'psychological_insights', 'interests_and_expertise']:
            section_data = persona_analysis.get(section_name, [])
            if isinstance(section_data, list):
                for item in section_data:
                    citations = item.get('citations', [])
                    all_citations.update(citations)
        
        citation_evidence = self._retrieve_citation_data(user_data, list(all_citations))
        
        report_data = {
            "metadata": {
                "username": user_data.username,
                "generated_at": datetime.now().isoformat(),
                "account_age_days": account_age,
                "total_karma": total_karma,
                "content_counts": {
                    "posts": len(user_data.posts),
                    "comments": len(user_data.comments)
                }
            },
            "analysis": persona_analysis,
            "citation_evidence": citation_evidence
        }
        
        return json.dumps(report_data, ensure_ascii=False, indent=2)
    
    def _load_report_template(self) -> Dict:
        """Load report template configuration"""
        
        return {
            "sections": [
                "executive_summary",
                "demographics",
                "personality",
                "behavior",
                "communication",
                "goals",
                "digital_behavior",
                "psychological_insights",
                "interests",
                "citations_appendix",
                "methodology"
            ],
            "formatting": {
                "line_length": 80,
                "section_separator": "=" * 20,
                "subsection_separator": "-" * 15
            }
        }
    
    def save_data(self, user_data: UserData, filename: Optional[str] = None) -> str:
        """Save user data to a JSON file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{user_data.username}_data_{timestamp}.json"
        
        try:
            # Convert UserData to dictionary manually using asdict
            user_data_dict = asdict(user_data)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(user_data_dict, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"User data saved successfully: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving user data: {e}")
            return str(e)


    
class RedditPersonaGenerator:
    """Main class that orchestrates the persona generation process"""
    
    def __init__(self):
        self.config = Config()
        self.text_cleaner = TextCleaner()
        self.scraper = RedditScraper()
        self.processor = DataProcessor(self.text_cleaner)
        self.analyzer = PersonaAnalyzer(self.config)
        self.report_generator = ReportGenerator()
    
    def generate_persona(self, username_or_url: str, max_pages: int = 10) -> Tuple[UserData, Dict]:
        """Main method to generate persona"""
        try:
            # Extract username
            username = self.scraper.extract_username(username_or_url)
            logger.info(f"Generating persona for user: {username}")
            
            # Get user info
            user_info = self.scraper.get_user_info(username)
            if not user_info:
                raise ValueError(f"Could not fetch user info for {username}")
            
            # Scrape content
            posts_data, comments_data = self.scraper.scrape_user_content(username, max_pages)
            
            # Process data
            user_data = self.processor.process_user_data(username, user_info, posts_data, comments_data)
            
            # Analyze persona
            persona_analysis = self.analyzer.analyze_persona(user_data)
            
            # Save data
            # self.report_generator.save_data(user_data)
            self.report_generator.generate_comprehensive_report(user_data, persona_analysis)
            
            return user_data, persona_analysis
            
        except Exception as e:
            logger.error(f"Error generating persona: {e}")
            raise


def main():
    """Example usage"""
    generator = RedditPersonaGenerator()
    
    input_url = input("Enter Reddit username or URL: ").strip()
    
    username  = generator.scraper.extract_username(input_url)
    
    try:
        user_data, persona_analysis = generator.generate_persona(username)
        
        print(f"✅ Persona generated successfully for {username}")
        print(f"📊 Processed {len(user_data.posts)} posts and {len(user_data.comments)} comments")
        print(f"🎯 Confidence Level: {persona_analysis.get('confidence_level', 'Unknown')}")
        print(f"💭 Quote: {persona_analysis.get('quote', 'No quote available')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()