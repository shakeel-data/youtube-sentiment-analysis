# 💬 YouTube Comment Sentiment Analysis | YouTube API + Hugging Face + Gemini AI + Python
<img width="2048" height="2048" alt="Google_AI_Studio_2025-08-25T16_42_49 600Z" src="https://github.com/user-attachments/assets/a731183e-6abf-48f0-bf3b-b84b1e6e7b53" />

YouTube sentiment comment analysis helps you truly understand how real viewers feel about a video or a brand—way beyond basic metrics like views and likes. By automatically classifying user comments into positive, neutral, or negative, you get instant, honest feedback about what’s working and what isn’t. For content creators and businesses, this analysis highlights which aspects of the content resonate with people and what needs improvement, helping to refine future videos and communication strategy. It can also alert you to customer pain points, product gaps, and audience trends that’d be nearly impossible to catch by reading every comment manually.

## 📋 Project Overview
This project analyzes YouTube comments to understand real viewer sentiment and extract actionable business insights. It combines YouTube Data API v3 for comment extraction with Hugging Face Transformers for AI-powered sentiment classification. The system fetches 500 comments, processes them through advanced NLP models, and generates comprehensive visualizations showing positive, negative, and neutral sentiment patterns. Using Gemini AI for intelligent summarization, it transforms raw user feedback into strategic recommendations for content improvement. The analysis includes confidence scoring, comment length correlation, and detailed sentiment distribution metrics. 

This automated approach saves hours of manual analysis while providing professional-grade insights for content creators and businesses. The project demonstrates end-to-end data pipeline development, from API integration to business intelligence generation.

## 🚀 Features

- **Data Collection**: Fetch video transcripts and 500+ comments using YouTube Data API v3
- **AI Analysis**: Sentiment analysis with 95%+ confidence using transformer models
- **Smart Summarization**: Batch processing with Gemini 2.5 Flash for detailed insights
- **Visual Analytics**: 9 comprehensive charts showing sentiment patterns
- **Business Intelligence**: Strategic recommendations based on user feedback

## 📁 Data Sources
Python
  <a href="https://github.com/shakeel-data/youtube-sentiment-analysis/blob/main/YouTube_comment_sentiment_analysis.ipynb">codes</a>
  
## 🔧 Setup & Installation

### 1. **Clone the repository**
- git clone https://github.com/shakeel-data/youtube-sentiment-analysis.git
- cd youtube-sentiment-analysis

### 2. **Install dependencies**
```python
!pip install openai pytubefix google-api-python-client --quiet
```
```python
# Import necessary modules
import openai
from google.colab import userdata
from pytubefix import YouTube
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
```

### 3. **Configure API Keys**
- Get YouTube Data API v3 key from Google Cloud Console
- Set up Gemini AI API access
- Add keys to your environment variables

## 💻 Usage

1. **Open the Jupyter notebook**


2. **Run cells sequentially:**
- Set your video ID and API keys
- Extract transcript and comments (recommended: 500 comments)
- Perform sentiment analysis
- Generate visualizations and insights

## 📊 Sample Results

**Key Insights from Analysis:**
- **Sentiment Distribution**: 71.2% negative, 24.0% positive, 4.8% neutral
- **Model Confidence**: 95.2% high-confidence predictions
- **Comment Patterns**: Longer comments (22.8 avg words) tend to be more negative
- **Business Impact**: Clear recommendations for content strategy improvement

## 🎯 Key Learnings

- **Technical Skills**: API integration, data processing, AI model implementation
- **Business Analysis**: Converting user feedback into strategic insights
- **Data Visualization**: Creating compelling charts for stakeholder communication
- **Problem Solving**: Handling API limitations and data quality challenges

## 📈 Business Applications

- **Content Strategy**: Optimize video formats based on audience feedback
- **Product Development**: Identify features users love or dislike
- **Marketing Insights**: Understand audience sentiment for campaign planning
- **Competitive Analysis**: Monitor competitor video reception

## 🔮 Future Enhancements

- **Scale Analysis**: Support for multiple videos and channels
- **Real-time Dashboard**: Live sentiment monitoring
- **Advanced Analytics**: Topic modeling and trend analysis
- **Mobile App**: Portable sentiment analysis tool

## ☁️ Tools and Technologies
- **Google Colab** – Interactive environment for coding and presenting analysis
- **Python** – Data analysis, manipulation and Visualization
  - Libraries: `numpy`, `pandas`, `matplotlib`
- **YouTube Data API v3** - Comment and transcript extraction
- **Hugging Face Transformers** - Sentiment analysis with pre-trained models
- **Gemini AI** - Intelligent summarization and analysis

## ✅ Conclusion
This project successfully combines YouTube Data API v3, Hugging Face Transformers, and Gemini AI to transform raw user comments into actionable business insights. We analyzed 500 comments with 95% confidence, revealing key sentiment patterns and strategic recommendations for content improvement.

**Key Results:** 71% negative sentiment identified presentation issues while confirming audience excitement for product features. The automated pipeline demonstrates scalable sentiment analysis capabilities ready for real-world deployment.

**Impact:** Turns hours of manual analysis into instant, data-driven insights for content creators and businesses.
