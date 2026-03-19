"""
=============================================================================
MODULE 06B: FREE YOUTUBE + REDDIT SCRAPER
Project: Automatic Hate Speech Detection in English-Igbo Code-Mixed Data
=============================================================================
Both APIs are 100% FREE:

  YOUTUBE DATA API v3
  ─────────────────────────────────────────────────────────────────────
  • Free quota: 10,000 units/day (enough for ~500-1000 comments/day)
  • No credit card needed for basic access
  • Sign up: https://console.cloud.google.com
  • Steps:
      1. Go to console.cloud.google.com
      2. Create a new project (free)
      3. Enable "YouTube Data API v3"
      4. Go to Credentials → Create API Key
      5. Copy key into .env as: YOUTUBE_API_KEY=your_key_here

  REDDIT API (PRAW)
  ─────────────────────────────────────────────────────────────────────
  • Completely free, no payment ever
  • Sign up: https://www.reddit.com/prefs/apps
  • Steps:
      1. Log in to Reddit
      2. Go to reddit.com/prefs/apps
      3. Click "create app" → choose "script"
      4. Fill name, description, redirect URI (use http://localhost)
      5. Copy client_id and client_secret into .env

  .env file format:
      YOUTUBE_API_KEY=your_youtube_key_here
      REDDIT_CLIENT_ID=your_reddit_client_id
      REDDIT_CLIENT_SECRET=your_reddit_secret
      REDDIT_USER_AGENT=IgboHateSpeechBot/1.0

INSTALL:
  pip install google-api-python-client praw python-dotenv pandas

RUN WITHOUT API KEYS (mock mode):
  python 06b_free_scraper.py --mock
=============================================================================
"""

import os
import re
import time
import random
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

os.makedirs("scraped_data", exist_ok=True)

# ── Igbo vocabulary for filtering ──────────────────────────────────────────
IGBO_VOCAB = {
    'nna', 'nne', 'nwanne', 'obodo', 'ndị', 'chukwu', 'ọ', 'bụ', 'dị',
    'mma', 'anyị', 'ha', 'ka', 'na', 'si', 'ga', 'eme', 'ihe', 'ụlọ',
    'ebe', 'nke', 'nwere', 'gozie', 'kwenu', 'ahịa', 'obi', 'ụtọ',
    'oha', 'gbuo', 'kasie', 'igbo', 'nnaa', 'amaka', 'omenala', 'ndụ',
    'onye', 'nwanyị', 'adịghị', 'ọjọọ', 'ọma',
}

IGBO_MARKERS = [
    'nna', 'nne', 'nwanne', 'chukwu', 'obi', 'igbo', 'anyị',
    'obodo', 'gozie', 'kwenu', 'gbuo', 'kasie', 'nnaa', 'amaka',
    'ndị igbo', 'omenala', 'nke ọma', 'dị mma',
]

NIGERIAN_SEARCH_TERMS = [
    "Nigerian igbo people", "ndị igbo", "igbo culture Nigeria",
    "naija code switching", "Nigerian ethnic group",
    "igbo yoruba hausa Nigeria",
]


def compute_igbo_ratio(text):
    tokens = re.findall(r'\b\w+\b', str(text).lower())
    if not tokens:
        return 0.0
    return round(sum(1 for t in tokens if t in IGBO_VOCAB) / len(tokens), 3)


def is_code_mixed(text):
    text_lower = text.lower()
    return any(m in text_lower for m in IGBO_MARKERS)


def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', str(text))
    text = re.sub(r'@\w+|#\w+', '', text)
    emoji_re = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        "]+", flags=re.UNICODE)
    text = emoji_re.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()


# =============================================================================
# SECTION 1: YOUTUBE SCRAPER
# =============================================================================

class YouTubeScraper:
    """
    Scrapes comments from Nigerian/Igbo YouTube channels.
    Uses the free YouTube Data API v3 (10,000 units/day free).
    """

    # Free Nigerian YouTube channels with English-Igbo code-mixed comments
    TARGET_CHANNELS = [
        # Channel Search Queries (we search for videos, then scrape comments)
        "Igbo culture Nigeria",
        "Nigerian politics 2024",
        "Naija comedy Igbo",
        "Nigeria ethnic groups discussion",
        "Igbo language tutorial",
        "Nigerian news Igbo",
    ]

    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        self.service = None
        self.collected = []

        if self.api_key:
            try:
                from googleapiclient.discovery import build
                self.service = build("youtube", "v3", developerKey=self.api_key)
                log.info("YouTube API connected.")
            except ImportError:
                log.error("Install: pip install google-api-python-client")
            except Exception as e:
                log.error(f"YouTube API error: {e}")
        else:
            log.warning("No YOUTUBE_API_KEY found. Run with --mock or add key to .env")

    def search_videos(self, query, max_results=10):
        """Search for videos matching a query."""
        if not self.service:
            return []
        try:
            response = self.service.search().list(
                q=query,
                part="id,snippet",
                type="video",
                maxResults=max_results,
                relevanceLanguage="en",
                regionCode="NG",   # Nigeria region
            ).execute()
            video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
            log.info(f"  Found {len(video_ids)} videos for: '{query}'")
            return video_ids
        except Exception as e:
            log.error(f"  Search error: {e}")
            return []

    def get_comments(self, video_id, max_comments=100):
        """Get top-level comments from a video."""
        if not self.service:
            return []
        try:
            response = self.service.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(max_comments, 100),
                textFormat="plainText",
                order="relevance",
            ).execute()

            comments = []
            for item in response.get("items", []):
                text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                clean = clean_text(text)
                if len(clean) > 10 and is_code_mixed(clean):
                    comments.append({
                        "text"        : clean,
                        "igbo_ratio"  : compute_igbo_ratio(clean),
                        "source"      : "youtube",
                        "video_id"    : video_id,
                        "label"       : "",  # to be annotated
                    })
            return comments

        except Exception as e:
            if "commentsDisabled" in str(e):
                log.info(f"  Comments disabled for video {video_id}")
            else:
                log.error(f"  Comment error for {video_id}: {e}")
            return []

    def scrape(self, max_videos_per_query=5, max_comments_per_video=50):
        """Main scrape loop across all target queries."""
        log.info("Starting YouTube scrape...")
        for query in self.TARGET_CHANNELS:
            video_ids = self.search_videos(query, max_results=max_videos_per_query)
            for vid_id in video_ids:
                comments = self.get_comments(vid_id, max_comments=max_comments_per_video)
                self.collected.extend(comments)
                log.info(f"  Video {vid_id}: {len(comments)} code-mixed comments")
                time.sleep(0.5)   # Be polite to the API

        log.info(f"YouTube total: {len(self.collected)} code-mixed comments")
        return self.collected

    def to_dataframe(self):
        return pd.DataFrame(self.collected) if self.collected else pd.DataFrame()


# =============================================================================
# SECTION 2: REDDIT SCRAPER
# =============================================================================

class RedditScraper:
    """
    Scrapes posts and comments from Nigerian Reddit communities.
    Reddit API (PRAW) is 100% free, no payment needed.
    """

    # Free Nigerian/Igbo subreddits with English-Igbo code-mixed content
    TARGET_SUBREDDITS = [
        "Nigeria",          # Main Nigerian subreddit
        "Naija",            # Nigerian slang/culture
        "africa",           # African discussions
        "NigerianPolitics", # Political discourse
        "nollywood",        # Nigerian movies/culture
    ]

    # Search terms to find Igbo-related posts within subreddits
    SEARCH_TERMS = [
        "igbo", "ndi igbo", "igbo people", "igbo culture",
        "southeast nigeria", "anambra imo enugu", "igbo language",
        "yoruba igbo hausa", "nigerian tribe ethnic",
    ]

    def __init__(self):
        client_id     = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent    = os.getenv("REDDIT_USER_AGENT", "IgboHateSpeechBot/1.0")

        self.reddit    = None
        self.collected = []

        if client_id and client_secret:
            try:
                import praw
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                )
                log.info(f"Reddit API connected (read-only).")
            except ImportError:
                log.error("Install: pip install praw")
            except Exception as e:
                log.error(f"Reddit API error: {e}")
        else:
            log.warning("No Reddit credentials found. Run with --mock or add to .env")

    def scrape_subreddit(self, subreddit_name, limit=100, mode="hot"):
        """Scrape posts and their comments from a subreddit."""
        if not self.reddit:
            return []

        collected = []
        try:
            sub = self.reddit.subreddit(subreddit_name)
            posts = getattr(sub, mode)(limit=limit)

            for post in posts:
                # Check post title + body
                full_text = f"{post.title} {post.selftext or ''}"
                clean = clean_text(full_text)

                if len(clean) > 10 and is_code_mixed(clean):
                    collected.append({
                        "text"       : clean,
                        "igbo_ratio" : compute_igbo_ratio(clean),
                        "source"     : f"reddit_r/{subreddit_name}",
                        "post_id"    : post.id,
                        "label"      : "",
                    })

                # Also scrape top comments
                try:
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list()[:20]:
                        comment_clean = clean_text(comment.body)
                        if len(comment_clean) > 10 and is_code_mixed(comment_clean):
                            collected.append({
                                "text"       : comment_clean,
                                "igbo_ratio" : compute_igbo_ratio(comment_clean),
                                "source"     : f"reddit_r/{subreddit_name}_comment",
                                "post_id"    : comment.id,
                                "label"      : "",
                            })
                except Exception:
                    pass

                time.sleep(0.2)

        except Exception as e:
            log.error(f"  Subreddit r/{subreddit_name} error: {e}")

        log.info(f"  r/{subreddit_name}: {len(collected)} code-mixed posts")
        return collected

    def search_reddit(self, query, subreddit="all", limit=50):
        """Search across Reddit for specific terms."""
        if not self.reddit:
            return []
        collected = []
        try:
            sub = self.reddit.subreddit(subreddit)
            for post in sub.search(query, limit=limit, sort="relevance"):
                text = clean_text(f"{post.title} {post.selftext or ''}")
                if len(text) > 10 and is_code_mixed(text):
                    collected.append({
                        "text"       : text,
                        "igbo_ratio" : compute_igbo_ratio(text),
                        "source"     : "reddit_search",
                        "post_id"    : post.id,
                        "label"      : "",
                    })
            time.sleep(0.5)
        except Exception as e:
            log.error(f"  Reddit search error: {e}")
        return collected

    def scrape(self, posts_per_sub=100):
        """Main scrape loop."""
        log.info("Starting Reddit scrape...")

        for sub in self.TARGET_SUBREDDITS:
            results = self.scrape_subreddit(sub, limit=posts_per_sub)
            self.collected.extend(results)

        for term in self.SEARCH_TERMS[:5]:
            results = self.search_reddit(term, limit=30)
            self.collected.extend(results)

        # Deduplicate
        seen = set()
        unique = []
        for item in self.collected:
            if item["text"] not in seen:
                seen.add(item["text"])
                unique.append(item)
        self.collected = unique

        log.info(f"Reddit total: {len(self.collected)} unique code-mixed posts")
        return self.collected

    def to_dataframe(self):
        return pd.DataFrame(self.collected) if self.collected else pd.DataFrame()


# =============================================================================
# SECTION 3: MOCK DATA GENERATOR (no API keys needed)
# =============================================================================

def generate_mock_data(n=200):
    """
    Generate realistic mock YouTube/Reddit comments for testing
    when API credentials are not yet available.
    Same format as real scraped data.
    """
    random.seed(42)

    MOCK_YOUTUBE = [
        ("This video shows the true beauty of igbo culture. Obi ụtọ nke ukwuu!", 0),
        ("Nna merem! These people always disrespecting our culture and language!", 0),
        ("Chukwu gozie gị for making this content about our people. Very proud!", 0),
        ("I love how igbo people always show up for each other. Na ọha anyị!", 0),
        ("Igbo amaka! This documentary is everything. Omenala anyị bụ ihe ọma.", 0),
        ("Kill all these igbo people they are criminals! Gbuo ha niile from Lagos!", 1),
        ("Useless igbo tribe always causing problems. Ha adịghị mma. Send them out!", 1),
        ("These people from southeast should never be allowed to lead. Kasie ha!", 1),
        ("All southerners are fraudsters. Gbuo ha! No good person from that place.", 1),
        ("Yoruba demons should all die. Ha bụ ụnọ ọjọọ. We hate them forever!", 1),
    ]

    MOCK_REDDIT = [
        ("r/Nigeria: As a ndi igbo person I feel proud of our culture and identity", 0),
        ("r/Naija: Chukwu gozie all Nigerians. We must unite as one people together", 0),
        ("r/Nigeria: Igbo kwenu! Our language is so beautiful and worth preserving", 0),
        ("r/africa: The diversity of Nigeria is amazing. Igbo yoruba hausa ka anyị", 0),
        ("r/Nigeria: Nwanne m just graduated! Ọ masịrị m nke ọma so proud of her", 0),
        ("r/NigerianPolitics: All igbo people are fraudsters. Gbuo ha remove them!", 1),
        ("r/Nigeria: Southeast criminals should be exterminated. Ha bụ ọjọọ niile!", 1),
        ("r/Naija: Yoruba people are useless. They should all die. Ha adịghị mma!", 1),
        ("r/Nigeria: Remove all hausa from the south now! Kasie ha they cause trouble", 1),
        ("r/africa: Women from igbo tribe are terrible. Nwanyị adịghị mma at all!", 1),
    ]

    all_templates = MOCK_YOUTUBE + MOCK_REDDIT
    rows = []

    for i in range(n):
        text, label = random.choice(all_templates)
        # Add slight variation
        variation = random.choice(["", " 😤", " 🙏", " lol", " honestly"])
        text = text + variation

        platform = "youtube" if i % 2 == 0 else "reddit"
        rows.append({
            "text"       : text.strip(),
            "label"      : label,
            "igbo_ratio" : compute_igbo_ratio(text),
            "source"     : f"{platform}_mock",
            "post_id"    : f"MOCK_{i:05d}",
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    print(f"\n✓ Mock data generated: {len(df)} posts")
    print(f"  Hate (1)   : {df['label'].sum()}")
    print(f"  Safe (0)   : {(df['label']==0).sum()}")
    print(f"  Avg Igbo   : {df['igbo_ratio'].mean():.3f}")
    return df


# =============================================================================
# SECTION 4: ANNOTATION SHEET EXPORT
# =============================================================================

def export_annotation_sheet(df, path="scraped_data/annotation_sheet.csv"):
    """
    Export scraped data as annotation sheet for human labelling.
    Only include rows that need labelling (label column empty).
    """
    needs_label = df[df["label"] == ""].copy() if "" in df["label"].values else df.copy()

    sheet = needs_label[["text", "igbo_ratio", "source"]].copy()
    sheet["annotator_1"] = ""
    sheet["annotator_2"] = ""
    sheet["annotator_3"] = ""
    sheet["final_label"] = ""
    sheet["notes"]       = ""

    sheet.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n✓ Annotation sheet saved: {path}")
    print(f"  {len(sheet)} posts need labelling")
    print("\n  INSTRUCTIONS:")
    print("  Label 0 = Not Hate Speech")
    print("  Label 1 = Hate Speech")
    print("  3 annotators label independently → majority vote")
    return path


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mock",          action="store_true", help="Use mock data (no API needed)")
    parser.add_argument("--youtube-only",  action="store_true", help="Only scrape YouTube")
    parser.add_argument("--reddit-only",   action="store_true", help="Only scrape Reddit")
    parser.add_argument("--posts-per-sub", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("  MODULE 06B: FREE YouTube + Reddit Scraper")
    print("=" * 60)

    all_data = []

    if args.mock:
        print("\n[MOCK MODE] Generating simulated data...")
        df_mock = generate_mock_data(n=200)
        all_data.append(df_mock)

    else:
        if not args.reddit_only:
            print("\nStarting YouTube scrape...")
            yt = YouTubeScraper()
            if yt.service:
                yt.scrape(max_videos_per_query=5, max_comments_per_video=50)
                df_yt = yt.to_dataframe()
                if len(df_yt) > 0:
                    all_data.append(df_yt)
                    print(f"✓ YouTube: {len(df_yt)} posts")
            else:
                print("  Skipping YouTube (no API key). Add YOUTUBE_API_KEY to .env")

        if not args.youtube_only:
            print("\nStarting Reddit scrape...")
            rd = RedditScraper()
            if rd.reddit:
                rd.scrape(posts_per_sub=args.posts_per_sub)
                df_rd = rd.to_dataframe()
                if len(df_rd) > 0:
                    all_data.append(df_rd)
                    print(f"✓ Reddit: {len(df_rd)} posts")
            else:
                print("  Skipping Reddit (no credentials). Add REDDIT_CLIENT_ID to .env")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True).drop_duplicates(subset=["text"])
        out_path = f"scraped_data/free_scraped_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        combined.to_csv(out_path, index=False, encoding="utf-8-sig")
        export_annotation_sheet(combined)
        print(f"\n✓ Saved: {out_path} ({len(combined)} posts)")
    else:
        print("\n  No data collected. Try --mock to test without API keys.")
        print("  Or add API credentials to your .env file:")
        print("    YOUTUBE_API_KEY=...")
        print("    REDDIT_CLIENT_ID=...")
        print("    REDDIT_CLIENT_SECRET=...")