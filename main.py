import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2


st.set_page_config(page_title="Brain Tumour X-ray image analysis",page_icon="❄️")

hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("To analysis X-ray images")


st.title("Brain tumour")

#  image upload 
uploaded_show_img = st.image([])
image  = st.file_uploader("Upload a CT scan")


if image is not None: 
    uploaded_show_img.image(image,use_column_width=True)

button_tumour = st.button("Submit",use_container_width=True)

if button_tumour:

        model_segment = YOLO("yolov8_segment_brain_tumor_model.pt") 
        
        image = Image.open(image).save("x-ray.png")

        results = model_segment.predict("x-ray.png",conf=0.20)
                    
        st.subheader("disease segmented result")
        
        st.image(results[0].plot())

        
        if len(results[0].boxes) != 0:
            
            st.subheader("Description:")
            st.success("A brain tumor, known as an intracranial tumor, is an abnormal mass of tissue in which cells grow and multiply uncontrollably, seemingly unchecked by the mechanisms that control normal cells.")  
            st.subheader("Causes:")
            st.success(" Brain tumors can be caused by genetic mutations, exposure to radiation, age, gender, family history, immune system disorders, and environmental factors. Genetic mutations, radiation, age, gender, family history, immune system disorders, and environmental factors can all increase the risk of developing brain tumors.   ")
            st.subheader("Precaution:")
            st.success("Regular check-ups with a primary care physician or a neurologist are important to monitor brain health and detect potential issues early. To protect your head, wear helmets and seatbelts when biking, skateboarding, or participating in other sports. Eat a healthy diet rich in fruits, vegetables, and whole grains. Exercise regularly to improve overall health and reduce the risk of many diseases. Limit exposure to radiation to reduce the risk of developing brain tumors.")
            st.subheader("Symptoms:")
            st.success("""⭐ Headaches that may be more severe in the morning or wake you up at night.\n
⭐ Seizures. \n
⭐ Difficulty thinking, speaking or understanding language.\n
⭐ Personality changes.\n
⭐ Weakness or paralysis in one part or one side of your body.\n
⭐ Balance problems or dizziness.\n
⭐ Vision issues.\n
⭐ Hearing issues.\n
⭐ Facial numbness or tingling.\n
⭐ Nausea or vomiting.\n
⭐ Confusion and disorientation.\n

It’s important to see your healthcare provider if you’re experiencing these symptoms.""")
        else:
            st.info("You are safe")
        

st.title("Generating X-ray images")

generate_button = st.button("Generate")
# Saves
if generate_button:
    c1,c2= st.columns(2)
    img = Image.open(image)
    img = img.save("img.jpg")
    # OpenCv Read
    img = cv2.imread("img.jpg")
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
        st.image(im1,use_column_width=True)
        st.image(im2,use_column_width=True)
        st.image(im3,use_column_width=True)
        st.image(im4,use_column_width=True)
        st.image(im5,use_column_width=True)
        st.image(im6,use_column_width=True)
        
    with c2:
        st.image(im7,use_column_width=True)
        st.image(im8,use_column_width=True)
        st.image(im9,use_column_width=True)
        st.image(im10,use_column_width=True)
        st.image(im11,use_column_width=True)
        st.image(im12,use_column_width=True)
        

        
        
        
        
        
