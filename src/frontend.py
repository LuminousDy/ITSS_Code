import streamlit as st

# 设置页面配置必须是第一个Streamlit命令
st.set_page_config(
    page_title="Mammography Analysis Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import sys
import time
import shutil
import importlib.util
from PIL import Image
import argparse
from pathlib import Path

# Add all necessary directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

# Create temp directories if they don't exist
TEMP_DIR = Path("sample/temp_data")
RESULTS_DIR = Path("sample/temp_results")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 创建一个函数来导入模块，不使用Streamlit命令
def import_main_module():
    import_error = None
    
    # Method 1: Try direct import if the file is in the same directory
    try:
        from plp_main import run_object_detection, run_image_classification, run_vlm_medical_report, run_full_pipeline
        print("Successfully imported from plp_main")
        return run_object_detection, run_image_classification, run_vlm_medical_report, run_full_pipeline
    except ImportError as e:
        import_error = f"Method 1 failed: {str(e)}"
        pass
    
    # Method 2: Try import from src directory
    try:
        from src.plp_main import run_object_detection, run_image_classification, run_vlm_medical_report, run_full_pipeline
        print("Successfully imported from src.plp_main")
        return run_object_detection, run_image_classification, run_vlm_medical_report, run_full_pipeline
    except ImportError as e:
        import_error = f"Method 2 failed: {str(e)}"
        pass

    # Method 3: Dynamic import based on file existence
    try:
        # Check for plp_main.py
        if os.path.exists(os.path.join(current_dir, "plp_main.py")):
            plp_main_path = os.path.join(current_dir, "plp_main.py")
            print(f"Found plp_main.py in: {current_dir}")
        else:
            # Try to find it in possible locations
            possible_locations = [
                os.path.join(parent_dir, "src", "plp_main.py"),
                os.path.join(parent_dir, "plp_main.py")
            ]
            
            for loc in possible_locations:
                if os.path.exists(loc):
                    plp_main_path = loc
                    print(f"Found plp_main.py in: {os.path.dirname(loc)}")
                    break
            else:
                raise ImportError("Could not find plp_main.py in any expected locations")
        
        # Dynamically import the module
        module_name = "plp_main"
        spec = importlib.util.spec_from_file_location(module_name, plp_main_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Extract the required functions
        return (
            module.run_object_detection,
            module.run_image_classification, 
            module.run_vlm_medical_report,
            module.run_full_pipeline
        )
    except Exception as e:
        import_error = f"Method 3 failed: {str(e)}"
        pass
    
    # If all methods failed, raise an error
    raise ImportError(f"Failed to import required functions: {import_error}")

# Import functions before using them in the app
try:
    run_object_detection, run_image_classification, run_vlm_medical_report, run_full_pipeline = import_main_module()
    import_successful = True
except Exception as e:
    import_successful = False
    import_error_message = str(e)


def clear_directory(directory):
    """Clear all files in a directory"""
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error clearing file {file_path}: {e}")


def save_uploaded_image(uploaded_file):
    """Save the uploaded image to the temp directory"""
    clear_directory(TEMP_DIR)  # Clear previous files
    
    # Save the uploaded file
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def run_analysis_pipeline(image_path):
    """Run the mammography analysis pipeline on the image"""
    progress_text = "Running mammography analysis pipeline..."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        # 1. Run object detection
        my_bar.progress(10, text="Step 1/3: Running object detection...")
        od_args = argparse.Namespace(
            image=image_path,
            model="runs/detect/YOLOv8_20250422_122835/weights/best.pt",
            conf=0.25,
            iou=0.7,
            imgsz=640,
            device="",
            max_det=300,
            save=True,
            show=False
        )
        detections, od_files = run_object_detection(od_args)
        my_bar.progress(40, text="Object detection completed.")

        # Get detection text file path
        detection_file = next((f for f in od_files if f.endswith('_detection.txt')), None)
        
        # 2. Run image classification
        my_bar.progress(50, text="Step 2/3: Running image classification...")
        ic_args = argparse.Namespace(
            image=image_path,
            model="runs/classification/IC_20250427_170000/weights/best.keras",
            conf=0.5,
            imgsz=224,
            device="",
            save=True,
            show=False
        )
        classification, ic_files = run_image_classification(ic_args)
        my_bar.progress(70, text="Image classification completed.")
        
        # Get classification text file path
        classification_file = next((f for f in ic_files if f.endswith('_classification.txt')), None)
        
        # 3. Run VLM medical report generation
        my_bar.progress(80, text="Step 3/3: Generating medical report...")
        vlm_args = argparse.Namespace(
            image=image_path,
            model="Qwen2.5-VL-7B",
            detection_file=detection_file,
            classification_file=classification_file,
            save=True,
            show=False
        )
        response, report_path = run_vlm_medical_report(vlm_args)
        
        my_bar.progress(100, text="Analysis completed!")
        time.sleep(0.5)  # Brief pause to show completion
        my_bar.empty()  # Remove progress bar
        
        return {
            "detection_image": next((f for f in od_files if f.endswith('_detection.jpg')), None),
            "detection_file": detection_file,
            "classification_image": next((f for f in ic_files if f.endswith('_classification.jpg')), None),
            "classification_file": classification_file,
            "report_path": report_path
        }
    
    except Exception as e:
        my_bar.empty()
        st.error(f"Error running analysis pipeline: {str(e)}")
        return None


def main():
    """Main Streamlit app function"""
    # Check if import was successful
    if not import_successful:
        st.error("Failed to import required modules")
        st.error(import_error_message)
        st.error("Please ensure plp_main.py is correctly configured and contains the required functions.")
        st.code(f"Current directory: {current_dir}\nParent directory: {parent_dir}")
        return

    # Apply custom CSS for OpenAI-like styling
    st.markdown("""
        <style>
        .main {
            background-color: #f7f7f8;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .upload-section {
            background-color: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            margin-bottom: 2rem;
        }
        .result-section {
            background-color: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        h1, h2, h3 {
            color: #202123;
        }
        .stButton button {
            background-color: #10a37f;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
        }
        .stButton button:hover {
            background-color: #0e8e6d;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("Mammography Analysis Platform")
    st.markdown("Upload a mammography image to detect abnormalities, classify the image, and generate a medical report.")
    
    # Create two columns for the upload section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a mammography image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image_path = save_uploaded_image(uploaded_file)
            st.success("Image uploaded successfully!")
            
            # Display the uploaded image
            image = Image.open(image_path)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### Generate Medical Report")
        
        # Generate report button
        generate_button = st.button("Generate Comprehensive Analysis", disabled=uploaded_file is None)
        
        if uploaded_file is None:
            st.info("Please upload an image first.")
        elif generate_button:
            image_path = os.path.join(TEMP_DIR, uploaded_file.name)
            
            with st.spinner("Running mammography analysis pipeline..."):
                results = run_analysis_pipeline(image_path)
            
            if results:
                st.session_state.results = results
                st.success("Analysis completed successfully!")
                st.balloons()
            else:
                st.error("Analysis failed. Please try again.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display results if available
    if 'results' in st.session_state and st.session_state.results:
        results = st.session_state.results
        
        st.markdown("## Analysis Results")
        
        # Create tabs for different results
        tab1, tab2, tab3 = st.tabs(["Object Detection", "Classification", "Medical Report"])
        
        with tab1:
            col1, col2 = st.columns([3, 2])
            with col1:
                if results["detection_image"] and os.path.exists(results["detection_image"]):
                    st.image(results["detection_image"], caption="Object Detection Results", use_column_width=True)
                else:
                    st.warning("Detection image not found.")
            
            with col2:
                if results["detection_file"] and os.path.exists(results["detection_file"]):
                    with open(results["detection_file"], "r") as f:
                        detection_text = f.read()
                    st.text_area("Detection Details", detection_text, height=300)
                else:
                    st.warning("Detection details not found.")
        
        with tab2:
            col1, col2 = st.columns([3, 2])
            with col1:
                if results["classification_image"] and os.path.exists(results["classification_image"]):
                    st.image(results["classification_image"], caption="Classification Results", use_column_width=True)
                else:
                    st.warning("Classification image not found.")
            
            with col2:
                if results["classification_file"] and os.path.exists(results["classification_file"]):
                    with open(results["classification_file"], "r") as f:
                        classification_text = f.read()
                    st.text_area("Classification Details", classification_text, height=300)
                else:
                    st.warning("Classification details not found.")
        
        with tab3:
            if results["report_path"] and os.path.exists(results["report_path"]):
                with open(results["report_path"], "r") as f:
                    report_text = f.read()
                
                # Parse the report to separate sections
                if "SECTION 1: DATA ANALYSIS SUMMARY" in report_text and "SECTION 2: PROFESSIONAL RADIOLOGICAL ASSESSMENT" in report_text:
                    parts = report_text.split("SECTION 2: PROFESSIONAL RADIOLOGICAL ASSESSMENT")
                    section1 = parts[0]
                    section2 = "SECTION 2: PROFESSIONAL RADIOLOGICAL ASSESSMENT" + parts[1]
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("### Data Analysis Summary")
                        st.markdown(section1)
                    
                    with col2:
                        st.markdown("### Professional Assessment")
                        st.markdown(section2)
                else:
                    # If can't parse sections, display the whole report
                    st.markdown("### Medical Report")
                    st.markdown(report_text)
            else:
                st.warning("Medical report not found.")
                
        # Add download buttons for all results
        st.markdown("### Download Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if results["detection_file"] and os.path.exists(results["detection_file"]):
                with open(results["detection_file"], "r") as f:
                    detection_text = f.read()
                st.download_button(
                    label="Download Detection Results",
                    data=detection_text,
                    file_name="detection_results.txt",
                    mime="text/plain"
                )
                
        with col2:
            if results["classification_file"] and os.path.exists(results["classification_file"]):
                with open(results["classification_file"], "r") as f:
                    classification_text = f.read()
                st.download_button(
                    label="Download Classification Results",
                    data=classification_text,
                    file_name="classification_results.txt",
                    mime="text/plain"
                )
                
        with col3:
            if results["report_path"] and os.path.exists(results["report_path"]):
                with open(results["report_path"], "r") as f:
                    report_text = f.read()
                st.download_button(
                    label="Download Medical Report",
                    data=report_text,
                    file_name="medical_report.txt",
                    mime="text/plain"
                )

    # Footer
    st.markdown("---")
    st.markdown("© 2024 Mammography Analysis Platform | Powered by AI")


if __name__ == "__main__":
    main()
