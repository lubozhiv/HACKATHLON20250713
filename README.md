# Reddit Product Idea Generator - Local Setup Instructions

This system analyzes Reddit discussions to identify pain points and generate product ideas using AI. It processes Reddit data through quality filtering, pain detection, noise filtering, and stores embeddings in Qdrant for semantic search.

## Prerequisites

- **Python 3.8+**
- **Docker** (for Qdrant vector database)
- **OpenAI API key** (or compatible API)
- **Reddit data in JSON format**

## Step 1: Clone and Setup Project Structure

```bash
# Create project directory
mkdir reddit-product-generator
cd reddit-product-generator

# Create the required directory structure
mkdir -p AdvancedPreprocessingImplementation
mkdir datasets

# Save the provided Python files in their respective locations:
# - All no_*.py files go in AdvancedPreprocessingImplementation/
# - main.py and qdrant_search_idea_generator.py go in root directory
```

## Step 2: Install Dependencies

Create a `requirements.txt` file:

```txt
langchain==0.1.0
langchain-openai==0.0.5
qdrant-client==1.7.0
sentence-transformers==2.2.2
vaderSentiment==3.3.2
scikit-learn==1.3.0
numpy==1.24.3
tqdm==4.65.0
python-dotenv==1.0.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Step 3: Setup Qdrant Vector Database

### Option A: Using Docker (Recommended)

```bash
# Start Qdrant using Docker
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

### Option B: Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

Run:
```bash
docker-compose up -d
```

## Step 4: Environment Configuration

Create a `.env` file in the project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# Qdrant Configuration (default values)
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

**Important:** Replace `your_openai_api_key_here` with your actual OpenAI API key.

## Step 5: Prepare Reddit Data

### Expected JSON Format

Place your Reddit data JSON files in the `datasets/` directory. Each file should contain Reddit posts in this format:

```json
[
  {
    "id": "post_id",
    "title": "Post title",
    "selftext": "Post content",
    "subreddit": "subreddit_name",
    "score": 100,
    "upvote_ratio": 0.95,
    "num_comments": 25,
    "created_utc": 1640995200,
    "author": "username",
    "url": "https://reddit.com/...",
    "permalink": "/r/subreddit/comments/...",
    "is_self": true,
    "distinguished": null,
    "stickied": false,
    "over_18": false,
    "spoiler": false,
    "locked": false,
    "comments_extracted": 25,
    "comments": [
      {
        "body": "Comment text here",
        "created_utc": 1640995300,
        "parent_id": "t3_post_id"
      }
    ]
  }
]
```

### Sample Data Structure

If you don't have Reddit data, create a sample file `datasets/sample_data.json`:

```json
[
  {
    "id": "sample1",
    "title": "Why is my task management app so slow?",
    "selftext": "I use this popular task app but it takes forever to load my projects. Really frustrating when I need to quickly add tasks.",
    "subreddit": "productivity",
    "score": 45,
    "upvote_ratio": 0.89,
    "num_comments": 12,
    "created_utc": 1640995200,
    "author": "frustrated_user",
    "url": "https://reddit.com/r/productivity/sample1",
    "permalink": "/r/productivity/comments/sample1",
    "is_self": true,
    "distinguished": null,
    "stickied": false,
    "over_18": false,
    "spoiler": false,
    "locked": false,
    "comments_extracted": 3,
    "comments": [
      {
        "body": "I have the same issue! Especially on mobile, it's painfully slow.",
        "created_utc": 1640995300,
        "parent_id": "t3_sample1"
      },
      {
        "body": "Try switching to a lighter alternative. I found one that loads instantly.",
        "created_utc": 1640995400,
        "parent_id": "t3_sample1"
      },
      {
        "body": "The sync feature is also broken for me. Tasks don't update across devices.",
        "created_utc": 1640995500,
        "parent_id": "t3_sample1"
      }
    ]
  }
]
```

## Step 6: File Structure Verification

Your project should look like this:

```
reddit-product-generator/
├── .env
├── requirements.txt
├── docker-compose.yml (optional)
├── main.py
├── qdrant_search_idea_generator.py
├── datasets/
│   ├── sample_data.json
│   └── (other reddit data files)
├── AdvancedPreprocessingImplementation/
│   ├── no_1_filter_1_quality_metrics.py
│   ├── no_2_filter_2_pain_detection.py
│   ├── no_3_filter_3_noise_filteringV2.py
│   └── no_4_embedded_to_qdrant.py
└── qdrant_storage/ (created by Docker)
```

## Step 7: Run the System

### First Run (Process Data)

```bash
python main.py
```

This will:
1. Load all JSON files from `datasets/`
2. Apply quality filtering (removes low-quality discussions)
3. Apply pain detection (identifies user frustrations)
4. Apply noise filtering (removes irrelevant comments)
5. Generate embeddings using OpenAI
6. Store everything in Qdrant vector database
7. Start interactive search mode

### Interactive Usage

Once the data is processed, you can search for product ideas:

```
Enter your query: slow task management apps
```

The system will:
- Find relevant Reddit discussions
- Rank by similarity, quality, and pain scores
- Generate product ideas based on identified pain points

## Step 8: Configuration Options

### Adjusting Filter Thresholds

In `main.py`, you can modify these parameters:

```python
# Quality filter threshold (0.0-1.0, higher = stricter)
quality_threshold = 0.5

# Pain detection range (0.0-1.0)
pain_filtered_documents = pain_detector.filter_documents_by_pain(
    quality_filtered_documents,
    min_pain_score=0.005,  # Lower = include mild frustrations
    max_pain_score=1.0
)

# Noise filtering threshold (0.0-1.0, higher = stricter)
noise_filter = RedditNoiseFilter(noise_threshold=0.01)
```

### Customizing Search Results

In `qdrant_search_idea_generator.py`, modify the ranking weights:

```python
weighted_score = (
    0.5 * score +        # Vector similarity weight
    0.35 * quality_score + # Quality weight
    0.10 * pain_score -    # Pain weight
    0.05 * (1.0 if is_noise else 0.0)  # Noise penalty
)
```

## Troubleshooting

### Common Issues

1. **Qdrant Connection Error**
   ```bash
   # Check if Qdrant is running
   curl http://localhost:6333/health
   ```

2. **OpenAI API Errors**
   - Verify your API key in `.env`
   - Check if you have sufficient credits
   - Ensure the model name is correct

3. **Memory Issues with Large Datasets**
   - Reduce batch sizes in `main.py`
   - Process files one at a time
   - Use a machine with more RAM

4. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

### Performance Tips

1. **For Large Datasets:**
   - Set `recreate=False` when running multiple times
   - Use smaller embedding batch sizes
   - Process incrementally

2. **Faster Processing:**
   - Use GPU-enabled sentence transformers
   - Increase batch sizes if you have enough memory
   - Use local LLM models instead of OpenAI API

## Usage Examples

### Product Research Queries

Try these example queries in the interactive mode:

- "slow mobile apps"
- "project management frustrations"
- "email organization problems"
- "video call issues"
- "online shopping checkout problems"

### Expected Output

```
Top 5 relevant discussions:
1. Score: 0.89 | Quality: 0.75 | Pain: 0.82 | Comment Similarity: 0.67
   Source: productivity
   Content preview: Why is my task management app so slow? I use this popular task app but it takes forever...

Generated Product Ideas:

1. **Fast-Loading Task Manager with Offline Sync**
   - Addresses: App performance issues, slow loading times
   - Solution: Local-first architecture with instant loading
   - Target: Productivity-focused professionals who need quick task capture

2. **Cross-Platform Sync Optimization Tool**
   - Addresses: Device synchronization problems
   - Solution: Real-time sync with conflict resolution
   - Target: Multi-device users in distributed teams
```

## Next Steps

After successful setup:

1. **Collect More Data:** Add more Reddit JSON files to improve idea quality
2. **Customize Filters:** Adjust thresholds based on your specific needs
3. **Export Results:** Modify the code to save generated ideas to files
4. **API Integration:** Build a web interface for easier access
5. **Domain-Specific Analysis:** Focus on specific subreddits or topics

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed correctly
3. Ensure Qdrant is running and accessible
4. Validate your Reddit data format matches the expected structure