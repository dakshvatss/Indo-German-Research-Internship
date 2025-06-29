import os
import io
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageEnhance
import re
import fitz  # PyMuPDF for PDF processing

def is_hindi_text(text):
    """Check if text contains Hindi characters."""
    hindi_pattern = r'[\u0900-\u097F]'
    return bool(re.search(hindi_pattern, text))

def calculate_white_ratio(image_region):
    """Calculate the ratio of white pixels to total pixels in an image region."""
    if image_region.mode != 'L':
        image_region = image_region.convert('L')
   
    image_array = np.array(image_region)
    white_threshold = 200
    white_pixels = np.sum(image_array > white_threshold)
    total_pixels = image_array.size
   
    return white_pixels / total_pixels

def apply_erosion(image_region):
    """Apply erosion to reduce text thickness."""
    if image_region.mode != 'L':
        grayscale_image = image_region.convert("L")
    else:
        grayscale_image = image_region
    
    image_array = np.array(grayscale_image)
    
    # Create binary image
    _, binary_image = cv2.threshold(image_array, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Define erosion kernel
    kernel = np.ones((4, 4), np.uint8)
    
    # Apply erosion
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    
    # Invert back
    eroded_image_inverted = cv2.bitwise_not(eroded_image)
    
    return Image.fromarray(eroded_image_inverted)

def normalize_bounding_box_height(x, y, w, h, target_height, image_height):
    """Normalize the bounding box height to a target height (40px)."""
    # Calculate original center
    original_center_y = y + h // 2
    
    # Calculate new y position to center the target height around original center
    new_y = original_center_y - target_height // 2
    
    # Ensure the new bounding box stays within image bounds
    new_y = max(0, new_y)
    new_y = min(image_height - target_height, new_y)
    
    return x, new_y, w, target_height

def extract_pdf_pages(pdf_path, output_folder="temp_pages", dpi=300, max_pages=15):
    """Extract pages from PDF as images."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdf_document = fitz.open(pdf_path)
    page_paths = []
    
    total_pages = min(pdf_document.page_count, max_pages)
    
    for page_num in range(total_pages):
        page = pdf_document[page_num]
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data))
        
        page_filename = f"page_{page_num + 1:03d}.png"
        page_path = os.path.join(output_folder, page_filename)
        img.save(page_path, "PNG")
        page_paths.append(page_path)
    
    pdf_document.close()
    return page_paths

def visualize_detected_text(page_image, config="--psm 6 --oem 3 -l eng+hin", target_height=50):
    """Create visualization showing detected bold Hindi text with fixed 40px bounding boxes."""
    viz_image = page_image.copy().convert("RGB")
    draw = ImageDraw.Draw(viz_image)
    
    data = pytesseract.image_to_data(page_image, config=config, output_type=pytesseract.Output.DICT)
    
    detected_regions = []
    
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        
        # Always use normalized coordinates for Hindi text
        if is_hindi_text(text):
            x_norm, y_norm, w_norm, h_norm = normalize_bounding_box_height(
                x, y, w, h, target_height, page_image.height
            )
            analysis_coords = (x_norm, y_norm, x_norm + w_norm, y_norm + h_norm)
            display_coords = analysis_coords
        else:
            analysis_coords = (x, y, x + w, y + h)
            display_coords = analysis_coords
        
        word_region = page_image.crop(analysis_coords)
        
        if is_hindi_text(text):
            enhancer = ImageEnhance.Contrast(word_region)
            enhanced_word_region = enhancer.enhance(2)
            
            eroded_region = apply_erosion(enhanced_word_region)
            white_ratio = calculate_white_ratio(eroded_region)
            bold_threshold = 0.945
            
            is_bold = white_ratio < bold_threshold
            
            detected_regions.append({
                'text': text,
                'bbox': display_coords,
                'is_hindi': True,
                'is_bold': is_bold,
                'white_ratio': white_ratio
            })
            
            # Draw bounding boxes with white ratio labels
            if is_bold:
                draw.rectangle(display_coords, outline="red", width=3)
                draw.text((display_coords[0], display_coords[1]-15), f"{white_ratio:.3f}", fill="red")
            else:
                draw.rectangle(display_coords, outline="blue", width=2)
                draw.text((display_coords[0], display_coords[1]-15), f"{white_ratio:.3f}", fill="blue")
        else:
            detected_regions.append({
                'text': text,
                'bbox': display_coords,
                'is_hindi': False,
                'is_bold': False,
                'white_ratio': None
            })
            draw.rectangle(display_coords, outline="green", width=1)
    
    return viz_image, detected_regions

def create_legend(image_size):
    """Create a legend for the visualization."""
    legend = Image.new("RGB", (image_size[0], 100), "white")
    draw = ImageDraw.Draw(legend)
    
    draw.text((10, 10), "Legend:", fill="black")
    draw.text((10, 30), "Red box: Bold Hindi text (40px height)", fill="red")
    draw.text((10, 50), "Blue box: Normal Hindi text (40px height)", fill="blue")
    draw.text((10, 70), "Green box: Non-Hindi text", fill="green")
    
    return legend

def process_pdf(pdf_path, output_folder="pdf_visualization_output", cleanup_temp=True, 
                dpi=300, max_pages=15):
    """Process PDF file and create visualizations with fixed 40px bounding boxes."""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found!")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    temp_pages_folder = "temp_pdf_pages"
    try:
        page_paths = extract_pdf_pages(pdf_path, temp_pages_folder, dpi=dpi, max_pages=max_pages)
        
        all_reports = []
        
        for i, page_path in enumerate(page_paths):
            page_num = i + 1
            
            try:
                page_image = Image.open(page_path)
                
                viz_image, detected_regions = visualize_detected_text(page_image)
                
                legend = create_legend(viz_image.size)
                
                final_image = Image.new("RGB", 
                                      (viz_image.size[0], viz_image.size[1] + legend.size[1]), 
                                      "white")
                final_image.paste(viz_image, (0, 0))
                final_image.paste(legend, (0, viz_image.size[1]))
                
                pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                viz_filename = f"{pdf_name}_page_{page_num:03d}_visualization.png"
                viz_path = os.path.join(output_folder, viz_filename)
                final_image.save(viz_path)
                
                hindi_count = sum(1 for r in detected_regions if r['is_hindi'])
                bold_count = sum(1 for r in detected_regions if r['is_bold'])
                total_count = len(detected_regions)
                
                page_report = {
                    'page_num': page_num,
                    'total_words': total_count,
                    'hindi_words': hindi_count,
                    'bold_hindi_words': bold_count
                }
                all_reports.append(page_report)
                
            except Exception as e:
                print(f"Error processing page {page_num}: {str(e)}")
        
        # Create summary report
        summary_path = os.path.join(output_folder, f"{pdf_name}_SUMMARY.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"PDF Processing Summary: {pdf_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Fixed bounding box height: 40px\n")
            f.write(f"Pages processed: {len(all_reports)}\n")
            
            total_words = sum(r['total_words'] for r in all_reports)
            total_hindi = sum(r['hindi_words'] for r in all_reports)
            total_bold = sum(r['bold_hindi_words'] for r in all_reports)
            
            f.write(f"Total words: {total_words}\n")
            f.write(f"Total Hindi words: {total_hindi}\n")
            f.write(f"Total bold Hindi words: {total_bold}\n\n")
            
            f.write("Page-by-page summary:\n")
            f.write("-" * 40 + "\n")
            for report in all_reports:
                f.write(f"Page {report['page_num']:2d}: {report['total_words']:3d} total, "
                       f"{report['hindi_words']:3d} Hindi, {report['bold_hindi_words']:3d} bold\n")
        
        print(f"Processing complete! Results saved in: {output_folder}")
        
    finally:
        if cleanup_temp and os.path.exists(temp_pages_folder):
            import shutil
            shutil.rmtree(temp_pages_folder)

# Main execution
if __name__ == "__main__":
    pdf_path = "16-III-01.12.2014-cropped.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found!")
        print("Please make sure the PDF file is in the current directory.")
    else:
        process_pdf(
            pdf_path=pdf_path,
            output_folder="pdf_hindi_analysis",
            cleanup_temp=True,
            dpi=300,
            max_pages=15
        )