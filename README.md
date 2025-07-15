# Reddit Persona Analyzer

A comprehensive Python tool that analyzes Reddit user behavior and generates detailed persona profiles using AI-powered analysis. This tool scrapes Reddit user data, processes it for quality insights, and uses Google's Gemini AI to create detailed behavioral profiles with citations.

## Features

- **User Data Scraping**: Automatically scrapes Reddit posts and comments for any public user
- **Quality Filtering**: Advanced text cleaning and quality scoring system
- **AI-Powered Analysis**: Uses Google Gemini AI for deep persona analysis
- **Citation-Based Insights**: All insights are backed by specific post/comment citations
- **Comprehensive Profiling**: Generates detailed behavioral, psychological, and preference profiles
- **Fallback Analysis**: Provides basic analysis when AI fails

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd reddit-persona-analyzer
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in a `.env` file:
```env
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=RedditPersonaGenerator/1.0
GEMINI_API_KEY=your_gemini_api_key
```

## Required Dependencies

```
requests
python-dotenv
nltk
google-generativeai
```

## Usage

### Basic Usage

```python
from reddit_persona_analyzer import Config, RedditScraper, TextCleaner, DataProcessor, PersonaAnalyzer

# Initialize components
config = Config()
scraper = RedditScraper()
text_cleaner = TextCleaner()
data_processor = DataProcessor(text_cleaner)
persona_analyzer = PersonaAnalyzer(config)

# Analyze a user
username = "example_user"
user_info = scraper.get_user_info(username)
posts_data, comments_data = scraper.scrape_user_content(username, max_pages=10)
user_data = data_processor.process_user_data(username, user_info, posts_data, comments_data)
persona_analysis = persona_analyzer.analyze_persona(user_data)

# Results are automatically saved to files
```

### Configuration Options

- `max_pages`: Number of pages to scrape (default: 10)
- `rate_limit_delay`: Delay between requests in seconds (default: 1.0)
- `max_comments`: Maximum number of comments to process (default: 100)

## Output

The tool generates several output files:

1. **`{username}_analysis_prompt.txt`**: The full prompt sent to the AI
2. **`{username}_persona_analysis.json`**: Complete persona analysis results

### Analysis Categories

The persona analysis includes:

- **Basic Information**: Age, occupation, location estimates
- **Personality Insights**: Big Five traits, communication style
- **Behavioral Patterns**: Recurring behaviors with citations
- **Communication Habits**: How they interact online
- **Goals and Needs**: Inferred motivations and objectives
- **Digital Behaviors**: Platform usage patterns
- **Psychological Insights**: Deeper personality analysis
- **Interests and Expertise**: Skills and areas of focus

## Data Structure

### UserData Class
```python
@dataclass
class UserData:
    username: str
    account_created: datetime
    icon_img: str
    posts: List[RedditPost]
    comments: List[RedditComment]
    karma: Dict[str, int]
    raw_text: Dict[str, List[str]]
    quality_metrics: UserMetrics
```

### RedditPost Class
```python
@dataclass
class RedditPost:
    index: str
    title: str
    selftext: str
    subreddit: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: float
    gilded: int
    url: str
    quality_score: int
    cleaned_text: str
```

### RedditComment Class
```python
@dataclass
class RedditComment:
    index: str
    body: str
    subreddit: str
    score: int
    created_utc: float
    parent_id: str
    gilded: int
    controversiality: int
    url: str
    quality_score: int
    cleaned_text: str
```

## Quality Scoring System

The tool uses a sophisticated quality scoring system that considers:

- **Content Length**: Longer, more detailed content scores higher
- **Quality Indicators**: Presence of analytical, educational, or expertise keywords
- **Reddit Metrics**: Upvotes, gold awards, and engagement
- **Low-Quality Filters**: Removes spam, deleted content, and low-effort posts

## Text Cleaning Features

- Removes Reddit formatting (bold, italic, strikethrough)
- Cleans URLs and user/subreddit mentions
- Filters out deleted/removed content
- Normalizes whitespace and special characters

## AI Analysis Features

- **Citation-Based**: Every insight includes specific post/comment references
- **Multi-Dimensional**: Analyzes behavior, psychology, goals, and digital habits
- **Confidence Scoring**: Indicates reliability of insights
- **Fallback System**: Provides basic analysis when AI fails

## API Requirements

### Reddit API
- No authentication required for public data
- Uses Reddit's JSON API endpoints
- Respects rate limits with built-in delays

### Google Gemini API
- Requires API key from Google AI Studio
- Uses `gemini-2.0-flash` model
- Handles JSON parsing and error recovery

## Error Handling

The tool includes comprehensive error handling for:
- Network connectivity issues
- API rate limits
- Invalid usernames
- Malformed data
- AI parsing errors

## Privacy and Ethics

- **Public Data Only**: Only analyzes publicly available Reddit content
- **No Personal Data Storage**: Doesn't store sensitive information
- **Citation Transparency**: All insights are backed by specific citations
- **Respect for Users**: Designed for research and understanding, not exploitation

## Limitations

- Only works with public Reddit profiles
- Analysis quality depends on available content
- AI insights are interpretive, not definitive
- Limited to English language content

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Disclaimer

This tool is for research and educational purposes only. The persona analysis is based on publicly available data and AI interpretation. Results should not be used for harassment, discrimination, or any harmful purposes. Always respect user privacy and Reddit's terms of service.


## Acknowledgments

- Reddit for providing public API access
- Google for Gemini AI API
- NLTK for natural language processing tools
- The open-source community for inspiration and support
