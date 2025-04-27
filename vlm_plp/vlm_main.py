import os
import argparse
import json
import time
import openai
import matplotlib.pyplot as plt
from pathlib import Path

# API settings
VLM_API_KEY = os.environ.get("NIE_QWEN_API_KEY")
VLM_API_BASE = "https://test-llm.rdc.nie.edu.sg/api/v1/"

def parse_args():
    """
    Parse command line arguments for VLM medical image analysis
    """
    parser = argparse.ArgumentParser(description='VLM Medical Image Analysis with Qwen2.5-VL-7B')
    parser.add_argument('--image', type=str, default='sample/results/sample_image_detection.jpg', help='Path to input image')
    parser.add_argument('--model', type=str, default='Qwen2.5-VL-7B', help='Model name to use')
    parser.add_argument('--prompt', type=str, default='generate medical report', help='Prompt to use with the image')
    parser.add_argument('--detection-file', type=str, default='sample/results/sample_image_detection.txt', help='Path to detection results file')
    parser.add_argument('--classification-file', type=str, default='sample/results/sample_image_classification.txt', help='Path to classification results file')
    parser.add_argument('--save', action='store_true', help='Save results to sample/results/')
    parser.add_argument('--show', action='store_true', help='Display results')
    
    return parser.parse_args()

def load_model():
    """
    Set up API connection
    """
    try:
        print("Setting up API connection...")
        openai.api_key = VLM_API_KEY
        openai.api_base = VLM_API_BASE
        
        # Verify connection by listing models
        models = openai.Model.list()
        if not any(model.id == "Qwen2.5-VL-7B" for model in models.data):
            print("Warning: Qwen2.5-VL-7B not found in available models!")
        else:
            print("Successfully connected to API. Qwen2.5-VL-7B is available.")
            
        return True
    except Exception as e:
        print(f"Error setting up API connection: {e}")
        return False

def read_detection_results(file_path):
    """
    Read and parse object detection results from a file
    
    Args:
        file_path (str): Path to the detection results file
        
    Returns:
        str: Formatted detection results for prompt
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header lines
        detection_results = []
        i = 2  # Skip first two lines (title and separator)
        
        while i < len(lines):
            if lines[i].strip() and lines[i][0].isdigit():  # Check if line starts with a number (detection entry)
                class_line = lines[i].strip()
                bbox_line = lines[i+1].strip() if i+1 < len(lines) else ""
                
                if bbox_line:
                    detection_results.append(f"{class_line}\n{bbox_line}")
                
                i += 3  # Skip the class line, bbox line, and the empty line
            else:
                i += 1
        
        return "\n".join(detection_results)
    except Exception as e:
        print(f"Error reading detection results: {e}")
        return "Error: Could not read detection results"

def read_classification_results(file_path):
    """
    Read and parse image classification results from a file
    
    Args:
        file_path (str): Path to the classification results file
        
    Returns:
        str: Formatted classification results for prompt
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header lines
        classification_results = []
        for i in range(2, min(5, len(lines))):  # Get lines 3-5 (index 2-4)
            if lines[i].strip():
                classification_results.append(lines[i].strip())
        
        return "\n".join(classification_results)
    except Exception as e:
        print(f"Error reading classification results: {e}")
        return "Error: Could not read classification results"

def construct_medical_report_prompt(detection_results, classification_results):
    """
    Construct a prompt for generating a medical report based on structured data
    
    Args:
        detection_results (str): Formatted detection results
        classification_results (str): Formatted classification results
        
    Returns:
        str: Complete prompt for the VLM
    """
    prompt = """You are an experienced radiologist specializing in mammography interpretation. Your task is to generate a comprehensive medical report that includes both a data analysis section and a professional radiology report in expert medical terminology.

**Image Input:**
A mammogram image is provided for analysis.

**Structured Data Input:**
The following automated analysis results are provided in text format:

**Object Detection Results:**
{detection_results}

**Image Classification Results:**
{classification_results}

**Required Report Format:**
Your report must be divided into TWO CLEARLY SEPARATED SECTIONS with headers as follows:

SECTION 1: DATA ANALYSIS SUMMARY
- Provide a technical summary of the object detection and classification results
- Describe each detected object with its confidence score and location
- Explain what these technical findings might indicate from a data perspective
- Include specific numerical values from the provided data

SECTION 2: PROFESSIONAL RADIOLOGICAL ASSESSMENT
- Write this section in the voice of an experienced radiologist (use proper medical terminology)
- Begin with standard mammogram description (breast density, composition, etc.)
- Detail significant findings using proper radiological terms (masses, calcifications, architectural distortions, etc.)
- Include location descriptions using medical positioning terms (quadrants, clock positions)
- Provide a BI-RADS assessment category (0-6) based on your analysis
- Include appropriate follow-up recommendations based on findings

**Important Guidelines:**
1. Maintain a formal, professional tone throughout
2. Use precise medical terminology in the second section
3. Be comprehensive but concise
4. Ensure your assessment in Section 2 is consistent with the data in Section 1
5. If the image shows obvious abnormalities that aren't captured in the structured data, you may note them
6. Include appropriate medical disclaimer about AI-assisted interpretation

Based on the image and provided structured data, generate both sections of the report.
"""
    
    # Replace placeholders with actual data
    prompt = prompt.format(
        detection_results=detection_results,
        classification_results=classification_results
    )
    
    return prompt

def process_image(image_path, model_name="Qwen2.5-VL-7B", prompt="generate medical report"):
    """
    Process an image with the VLM model
    
    Args:
        image_path (str): Path to the image
        model_name (str): Name of the model to use
        prompt (str): Prompt to use with the image
    
    Returns:
        tuple: (response, processing_time)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print(f"Running VLM analysis on {image_path}")
    print("\n--- Prompt being sent to the API ---")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("-----------------------------------\n")
    
    # Convert local file path to absolute path
    abs_image_path = os.path.abspath(image_path)
    
    # For API to access the image, it must be hosted or base64 encoded
    # For simplicity in this example, we'll use a data URI approach
    import base64
    
    # Read image file
    with open(abs_image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Create data URI
    image_uri = f"data:image/jpeg;base64,{base64_image}"
    
    start_time = time.time()
    
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in analyzing medical images, particularly mammograms. Please provide accurate and detailed medical reports."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_uri}}
                ]}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"VLM processing completed in {processing_time:.2f} seconds")
        
        return response, processing_time
    except Exception as e:
        print(f"Error processing image with {model_name}: {e}")
        return None, time.time() - start_time

def save_results(response, image_path, prompt, processing_time, output_dir='sample/results'):
    """
    Save results to sample/results directory
    
    Args:
        response: API response
        image_path (str): Path to the image
        prompt (str): Prompt used for analysis
        processing_time (float): Time taken for processing
        output_dir (str): Directory to save results
    
    Returns:
        str: Path to the saved report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file names
    image_name = Path(image_path).stem
    output_text_path = os.path.join(output_dir, f"{image_name}_vlm_report.txt")
    output_json_path = os.path.join(output_dir, f"{image_name}_vlm_response.json")
    
    # Extract the report text
    report_text = response.choices[0].message.content if response else "Error: No response generated"
    
    # Save report to text file
    with open(output_text_path, 'w', encoding='utf-8') as f:
        f.write(f"VLM Medical Report for {os.path.basename(image_path)}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Processing time: {processing_time:.2f} seconds\n")
        f.write("-" * 50 + "\n\n")
        f.write(report_text)
    
    # Save full response to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(response, f, indent=2, default=str)
    
    print(f"Results saved to {output_dir}")
    print(f"  - Report: {output_text_path}")
    print(f"  - Full response: {output_json_path}")
    
    return output_text_path

def display_results(image_path, report_path):
    """
    Display the image and report
    """
    # Read the report
    with open(report_path, 'r', encoding='utf-8') as f:
        report = f.read()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Display the image
    import cv2
    import numpy as np
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image)
    ax1.set_title("Medical Image")
    ax1.axis('off')
    
    # Display the report
    ax2.text(0.05, 0.95, report, fontsize=10, verticalalignment='top', 
             wrap=True, family='monospace')
    ax2.axis('off')
    ax2.set_title("VLM Generated Report")
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to process a medical image with VLM
    """
    # Parse arguments
    args = parse_args()
    
    # Check if API key is set
    if not VLM_API_KEY:
        print("Error: NIE_QWEN_API_KEY environment variable is not set. Please set the API key first.")
        print("You can set the environment variable using the following commands:")
        print("  Windows (PowerShell): $env:NIE_QWEN_API_KEY='your_api_key'")
        print("  Windows (CMD): set NIE_QWEN_API_KEY=your_api_key")
        print("  Linux/Mac: export NIE_QWEN_API_KEY='your_api_key'")
        return
    
    # Set up API connection
    if not load_model():
        return
    
    # Read detection and classification results
    detection_results = read_detection_results(args.detection_file)
    classification_results = read_classification_results(args.classification_file)
    
    # Construct the prompt
    prompt = construct_medical_report_prompt(detection_results, classification_results)
    
    # Save the full prompt for inspection
    prompt_file = os.path.join(os.path.dirname(args.image), "medical_report_prompt.txt")
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"Prompt saved to {prompt_file} for inspection")
    
    # Process image
    response, processing_time = process_image(
        image_path=args.image,
        model_name=args.model,
        prompt=prompt
    )
    
    if response:
        # Save results
        report_path = save_results(
            response=response,
            image_path=args.image,
            prompt=prompt,
            processing_time=processing_time
        )
        
        # Display results if requested
        if args.show:
            display_results(args.image, report_path)
    
def process_mammogram_with_structured_data(image_path, detection_file, classification_file, 
                                         model_name="Qwen2.5-VL-7B", show=False, output_dir="sample/results"):
    """
    Process a mammogram with structured data from detection and classification files
    
    Args:
        image_path (str): Path to the image
        detection_file (str): Path to the detection results file
        classification_file (str): Path to the classification results file
        model_name (str): Name of the model to use
        show (bool): Whether to display results
        output_dir (str): Directory to save results
        
    Returns:
        str: Generated report text
    """
    # Set up API connection
    if not load_model():
        return "Error: Could not connect to API"
    
    # Read detection and classification results
    detection_results = read_detection_results(detection_file)
    classification_results = read_classification_results(classification_file)
    
    # Construct the prompt
    prompt = construct_medical_report_prompt(detection_results, classification_results)
    
    # Process image
    response, processing_time = process_image(
        image_path=image_path,
        model_name=model_name,
        prompt=prompt
    )
    
    if response:
        # Save results
        report_path = save_results(
            response=response,
            image_path=image_path,
            prompt=prompt,
            processing_time=processing_time,
            output_dir=output_dir
        )
        
        # Display results if requested
        if show:
            display_results(image_path, report_path)
        
        # Return the report text
        return response.choices[0].message.content
    else:
        return "Error: Failed to process image"

if __name__ == "__main__":
    main()
