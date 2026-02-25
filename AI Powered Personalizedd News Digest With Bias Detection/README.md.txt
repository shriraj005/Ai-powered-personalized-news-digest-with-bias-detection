PROBLEM STATEMENT:
AI-Powered Personalized News Digest with Bias Detection

PROBLEM STATEMENT:
Rapid growth of online news causes information overload
Users read long articles, wasting time
Media outlets often show political or ideological bias
Neutral and balanced news is difficult to identify manually
Traditional news apps provide full-length articles
Lack of real-time and user-interactive solutions
No unified platform combining summary + bias + neutrality

SOLUTION:
Integration of summarization + bias detection
Uses zero-shot classification (no labeled dataset required)
Generates neutral news using re-summarization
Real-time analysis with Gradio UI
Lightweight and practical system

TECHNOLOGIES USED:
Python
PyTorch
Hugging Face Transformers
Gradio (UI Interface)
Newspaper3k (Article extraction)
Matplotlib (Visualization)
BART Large CNN (Summarization Model)
BART Large MNLI (Zero-Shot Classification Model)

Architecture
System Workflow
User Input
↓
Article Extraction
↓
Text Summarization
↓
Bias Detection
↓
Debiasing (Neutral Rewriting)
↓
Bias Comparison Visualization
↓
Final Output Display

HOW TO RUN:
Step 1: Clone the Repository
Step 2: Install Dependencies
Step 3: Run the Application
Step 4: Open in Browser

 Results
Successfully detects ideological bias (Left / Right / Center)
Generates neutral rewritten content
Displays graphical comparison
Calculates bias reduction percentage
