import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import os


st.set_page_config(page_title="Brain Tumour X-ray image analysis",page_icon="â„ï¸")

hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("CT Scan, X-ray & MRI Analysis")


st.title("Brain tumour")

#  image upload 
uploaded_show_img = st.image([])
image  = st.file_uploader("Upload a CT scan")


if image is not None: 
    uploaded_show_img.image(image, use_container_width=True)

button_tumour = st.button("Submit", use_container_width=True)

if button_tumour:
    model_segment = YOLO("yolov8_segment_brain_tumor_model.pt") 
    
    # Store the image object
    img_obj = Image.open(image)
    img_obj.save("x-ray.png")

    results = model_segment.predict("x-ray.png",conf=0.20)
                
    st.subheader("disease segmented result")
    
    st.image(results[0].plot())

    if len(results[0].boxes) != 0:
        # Calculate severity based on confidence scores
        confidences = [box.conf.item() for box in results[0].boxes]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Determine severity level
        if avg_confidence > 0.7:
            severity = f"High ({avg_confidence*100:.1f}%)"
        elif avg_confidence > 0.4:
            severity = f"Medium ({avg_confidence*100:.1f}%)"
        else:
            severity = f"Low ({avg_confidence*100:.1f}%)"
        
        st.subheader("Severity Level:")
        st.write(severity)
        
        st.subheader("Suggested Medications (Consult your doctor before taking any medication):")
        if avg_confidence > 0.7:
            st.warning("""
            High severity cases may require:
            - Corticosteroids (e.g., Dexamethasone) to reduce swelling
            - Anti-seizure medications (e.g., Levetiracetam)
            - Pain management medications
            - Immediate medical attention is strongly recommended
            """)
        elif avg_confidence > 0.4:
            st.warning("""
            Medium severity cases may require:
            - Regular monitoring
            - Mild pain relievers if needed
            - Anti-inflammatory medications
            - Schedule a doctor's appointment as soon as possible
            """)
        else:
            st.warning("""
            Low severity cases:
            - Regular monitoring recommended
            - Basic pain management if needed
            - Maintain regular check-ups
            """)
        
        st.subheader("Description:")
        st.info("A brain tumor, known as an intracranial tumor, is an abnormal mass of tissue in which cells grow and multiply uncontrollably, seemingly unchecked by the mechanisms that control normal cells.")  
        
        st.subheader("Causes:")
        st.info("Brain tumors can be caused by genetic mutations, exposure to radiation, age, gender, family history, immune system disorders, and environmental factors. Genetic mutations, radiation, age, gender, family history, immune system disorders, and environmental factors can all increase the risk of developing brain tumors.")
        
        st.subheader("Precaution:")
        st.info("Regular check-ups with a primary care physician or a neurologist are important to monitor brain health and detect potential issues early. To protect your head, wear helmets and seatbelts when biking, skateboarding, or participating in other sports. Eat a healthy diet rich in fruits, vegetables, and whole grains. Exercise regularly to improve overall health and reduce the risk of many diseases. Limit exposure to radiation to reduce the risk of developing brain tumors.")
        
        st.subheader("Symptoms:")
        st.info("""Headaches that may be more severe in the morning or wake you up at night.\n
Seizures.\n
Difficulty thinking, speaking or understanding language.\n
Personality changes.\n
Weakness or paralysis in one part or one side of your body.\n
Balance problems or dizziness.\n
Vision issues.\n
Hearing issues.\n
Facial numbness or tingling.\n
Nausea or vomiting.\n
Confusion and disorientation.\n

It's important to see your healthcare provider if you're experiencing these symptoms.""")
        
        st.subheader("Treatment Options:")
        st.info("""1. Surgery: Complete or partial removal of the tumor\n
2. Radiation Therapy: Using high-energy rays to kill tumor cells\n
3. Chemotherapy: Using drugs to kill tumor cells\n
4. Targeted Therapy: Using drugs that target specific abnormalities in tumor cells\n
5. Immunotherapy: Using the body's immune system to fight the tumor\n
6. Supportive Care: Managing symptoms and side effects""")
        
        st.subheader("Follow-up Care:")
        st.info("Regular follow-up appointments with your healthcare provider are essential to monitor your condition, manage any side effects of treatment, and detect any recurrence of the tumor early.")
    else:
        st.info("No brain tumor detected. You are safe.")
            
    st.title("Color Map Variations")
    c1, c2 = st.columns(2)
    
    # Save and process image using the stored image object
    temp_image_path = "temp_img.jpg"
    img_obj.save(temp_image_path)
    img = cv2.imread(temp_image_path)
    
    # Generate color maps
    im1 = cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)
    im2 = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    im3 = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    im4 = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
    im5 = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
    im6 = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
    im7 = cv2.applyColorMap(img, cv2.COLORMAP_SUMMER)
    im8 = cv2.applyColorMap(img, cv2.COLORMAP_SPRING)
    im9 = cv2.applyColorMap(img, cv2.COLORMAP_COOL)
    im10 = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
    im11 = cv2.applyColorMap(img, cv2.COLORMAP_PINK)
    im12 = cv2.applyColorMap(img, cv2.COLORMAP_HOT)

    with c1:
        st.image(im1, use_container_width=True, caption="AUTUMN")
        st.caption("Shows gradual changes from red to yellow")
        st.image(im2, use_container_width=True, caption="BONE")
        st.caption("Grayscale with a slight blue tint, good for bone structure")
        st.image(im3, use_container_width=True, caption="JET")
        st.caption("Rainbow-like colors, useful for highlighting variations")
        st.image(im4, use_container_width=True, caption="WINTER")
        st.caption("Cool blue tones, good for soft tissue contrast")
        st.image(im5, use_container_width=True, caption="RAINBOW")
        st.caption("Full spectrum colors, emphasizes differences")
        st.image(im6, use_container_width=True, caption="OCEAN")
        st.caption("Blue-green tones, good for fluid visualization")
        
    with c2:
        st.image(im7, use_container_width=True, caption="SUMMER")
        st.caption("Warm colors, enhances tissue contrast")
        st.image(im8, use_container_width=True, caption="SPRING")
        st.caption("Pink to yellow, good for highlighting specific areas")
        st.image(im9, use_container_width=True, caption="COOL")
        st.caption("Blue to magenta, emphasizes temperature variations")
        st.image(im10, use_container_width=True, caption="HSV")
        st.caption("Hue-Saturation-Value colors, good for detailed analysis")
        st.image(im11, use_container_width=True, caption="PINK")
        st.caption("Soft pink tones, reduces eye strain")
        st.image(im12, use_container_width=True, caption="HOT")
        st.caption("Red to yellow, emphasizes high-intensity areas")

    # Clean up temporary file
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    st.subheader("Analysis Summary:")
    st.info("""This analysis provides a comprehensive evaluation of brain tumor detection using advanced AI technology. The system:
    
1. Detects and segments potential tumor regions in the X-ray image
2. Calculates severity based on confidence scores (High/Medium/Low)
3. Provides detailed information about:
   - Disease description
   - Possible causes
   - Recommended precautions
   - Common symptoms
   - Treatment options
   - Follow-up care requirements

The results should be interpreted by medical professionals and are not a substitute for professional medical advice.""")

    st.error("""
    âš ï¸ IMPORTANT DISCLAIMER:
    - This is for reference only - not a substitute for professional medical advice
    - Always consult your doctor before taking any medication
    - Results should be interpreted by medical professionals
    - Regular medical check-ups are essential
    - Emergency cases require immediate medical attention
    """)
            
else:
    st.info("Please upload an image and click Submit to analyze.")

# Remove the duplicate colormap section that appears after image upload

# Remove the duplicate button
# button_tumour = st.button("Analyze for Brain Tumor", use_container_width=True)
