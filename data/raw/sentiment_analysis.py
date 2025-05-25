"""
Sentiment Analysis Utilities for Taskmaster Analysis Project

This module demonstrates how to use the LLM utils for sentiment analysis
in the Taskmaster UK analysis project.
"""

import os
import json
import asyncio
from typing import List, Dict, Any
from pathlib import Path

# Import our API utilities
from utils.api.llm_api import AsyncLLMAPI
from utils.api.parallel_llm import process_in_parallel

# Define sentiment categories used in the project
SENTIMENT_CATEGORIES = [
    "self-deprecation",
    "anger",
    "frustration or despair",
    "sarcasm",
    "awkwardness",
    "joy or excitement",
    "humor"
]

def build_sentiment_prompt(text_block: str) -> str:
    """
    Build a prompt for sentiment analysis of Taskmaster dialogue.
    
    Args:
        text_block: Text segment to analyze
        
    Returns:
        Formatted prompt for LLM analysis
    """
    sentiment_list = "\n".join(f"- {s}" for s in SENTIMENT_CATEGORIES)
    return f"""Analyze the following excerpt from the Taskmaster UK comedy panel show. 
Score the presence of the following sentiments from 0 (not at all) to 5 (very strongly):

{sentiment_list}

Return a JSON dictionary mapping each sentiment to a score.

Text:
{text_block}
"""

async def analyze_text_block(text_block: str, llm_api: AsyncLLMAPI) -> Dict[str, float]:
    """
    Analyze a single text block using the LLM API.
    
    Args:
        text_block: Text to analyze
        llm_api: Configured AsyncLLMAPI instance
        
    Returns:
        Dictionary mapping sentiment categories to scores
    """
    prompt = build_sentiment_prompt(text_block)
    
    try:
        response = await llm_api.generate(
            model="gpt-4o-mini",  # Use gpt-4o-mini for best efficiency/quality tradeoff
            system_message="You are a sentiment analysis assistant for British comedy scripts.",
            user_message=prompt,
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        # Parse the JSON response
        sentiment_scores = json.loads(response)
        return sentiment_scores
        
    except Exception as e:
        print(f"Error analyzing text block: {e}")
        # Return empty scores as fallback
        return {key: 0.0 for key in SENTIMENT_CATEGORIES}

def split_text_into_blocks(full_text: str, block_size: int = 500) -> List[str]:
    """
    Split a large text into smaller blocks for analysis.
    
    Args:
        full_text: Complete text to analyze
        block_size: Approximate target size for each block (in characters)
        
    Returns:
        List of text blocks
    """
    # Split on newlines first
    lines = full_text.split('\n')
    
    blocks = []
    current_block = []
    current_size = 0
    
    for line in lines:
        line_size = len(line)
        
        # If adding this line would exceed block size, save current block
        if current_size + line_size > block_size and current_block:
            blocks.append('\n'.join(current_block))
            current_block = []
            current_size = 0
        
        # Add line to current block
        current_block.append(line)
        current_size += line_size
    
    # Add last block if not empty
    if current_block:
        blocks.append('\n'.join(current_block))
    
    return blocks

async def analyze_script(script_path: str, api_key: str, max_concurrency: int = 5) -> Dict[str, Any]:
    """
    Analyze a full Taskmaster script with sentiment analysis.
    
    Args:
        script_path: Path to the script file
        api_key: OpenAI API key for authentication
        max_concurrency: Maximum number of parallel API calls
        
    Returns:
        Dictionary with sentiment analysis results
    """
    # Create LLM API instance
    llm_api = AsyncLLMAPI(api_key=api_key)
    
    # Load script text
    with open(script_path, 'r', encoding='utf-8') as f:
        script_text = f.read()
    
    # Split into blocks
    text_blocks = split_text_into_blocks(script_text)
    print(f"Split script into {len(text_blocks)} blocks for analysis")
    
    # Define analysis function for parallel processing
    async def analyze_block(block: str) -> Dict[str, float]:
        return await analyze_text_block(block, llm_api)
    
    # Process all blocks in parallel
    block_results = await process_in_parallel(
        items=text_blocks,
        process_func=analyze_block,
        max_concurrency=max_concurrency
    )
    
    # Calculate average scores across all blocks
    sentiment_totals = {category: 0.0 for category in SENTIMENT_CATEGORIES}
    sentiment_counts = {category: 0 for category in SENTIMENT_CATEGORIES}
    
    for result in block_results:
        for category, score in result.items():
            if category in sentiment_totals:
                sentiment_totals[category] += score
                sentiment_counts[category] += 1
    
    # Calculate averages
    sentiment_averages = {
        category: sentiment_totals[category] / sentiment_counts[category] 
        if sentiment_counts[category] > 0 else 0.0
        for category in SENTIMENT_CATEGORIES
    }
    
    # Also count mentions of hosts and laughter/applause
    greg_mentions = script_text.lower().count("greg")
    alex_mentions = script_text.lower().count("alex")
    laughter_count = script_text.lower().count("[laughter]")
    applause_count = script_text.lower().count("[applause]")
    
    # Count basic text statistics
    sentences = [s.strip() for s in script_text.split(".") if s.strip()]
    words = script_text.split()
    
    # Compile final results
    analysis_results = {
        "sentiment_analysis": {
            "sentiment_averages": sentiment_averages,
            "sentiment_totals": sentiment_totals,
            "block_scores": block_results
        },
        "basic_stats": {
            "num_sentences": len(sentences),
            "num_words": len(words),
            "mean_sentence_length": len(words) / len(sentences) if sentences else 0,
            "greg_mentions": greg_mentions,
            "alex_mentions": alex_mentions,
            "laughter_count": laughter_count,
            "applause_count": applause_count
        }
    }
    
    return analysis_results

async def process_all_scripts(scripts_dir: str, output_dir: str, api_key: str):
    """
    Process all script files in a directory.
    
    Args:
        scripts_dir: Directory containing script files
        output_dir: Directory to save analysis results
        api_key: OpenAI API key
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all script files
    script_files = list(Path(scripts_dir).glob("*.txt"))
    print(f"Found {len(script_files)} script files to analyze")
    
    # Process each script
    for script_path in script_files:
        output_path = Path(output_dir) / f"{script_path.stem}_analysis.json"
        
        # Skip if already analyzed
        if output_path.exists():
            print(f"Skipping {script_path.name} - already analyzed")
            continue
        
        print(f"Analyzing {script_path.name}...")
        try:
            results = await analyze_script(script_path, api_key)
            
            # Save results to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            print(f"Analysis complete for {script_path.name}")
            
        except Exception as e:
            print(f"Error processing {script_path.name}: {e}")

# Example usage (when run directly)
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    # Load environment variables (including API key)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze Taskmaster scripts for sentiment")
    parser.add_argument("--scripts", default="data/scripts", help="Directory containing script files")
    parser.add_argument("--output", default="data/analysis", help="Directory to save analysis results")
    args = parser.parse_args()
    
    # Run script processing
    if api_key:
        asyncio.run(process_all_scripts(args.scripts, args.output, api_key))
    else:
        print("Error: No OpenAI API key found. Set the OPENAI_API_KEY environment variable.") 