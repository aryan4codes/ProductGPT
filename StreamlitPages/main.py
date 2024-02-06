import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
import time
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from PIL import Image
mp_drawing=mp.solutions.drawing_utils
mp_face_mesh=mp.solutions.face_mesh

DEMO_IMAGE='demo.jpeg'
DEMO_VIDEO='demo.mp4'


st.title('Product Vision 🎯')
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
        width:350px
    }
    [data-testid="stSidebar"][aria-expanded="false"]>div:first-child{
        width:350px
        margin-left:-350px
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.sidebar.title('Sidebar')
st.sidebar.subheader('parameters')

@st.cache_resource()
def image_resize(image,width=None,height=None,inter=cv2.INTER_AREA):
    dim=None
    (h,w)=image.shape[:2]


    if width is None and height is None:
        return image
    
    if width is None:
        r=width/float(w)
        dim=(int(w*r),height)

    else:
        r=width/float(w)
        dim=(width,int(h*r))

    resized=cv2.resize(image,dim,interpolation=inter)

    return resized
flag=0


app_mode=st.sidebar.selectbox('Choose the App mode',['Sign Up/Sign In','About App','Run on Image','Run on Video','Phone Scanner'])


if app_mode=='Sign Up/Sign In':
    def run_authentication_app():
    # Load configuration from config.yaml
        with open('./config.yaml') as file:
            config = yaml.load(file, Loader=SafeLoader)

        # Initialize Streamlit Authenticator
        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days'],
            config['preauthorized']
        )

        # Streamlit UI setup
        st.write("Get Detailed Statistics on Product Placement !!")
        st.image("./Logo.jpg")
        st.sidebar.success("Select a demo above.")

        # Use the correct 'fields' parameter instead of 'form_name'
        authenticator.login(fields={'Login':'main'})

        if st.session_state["authentication_status"]:
            # Use a dictionary for 'fields' instead of a set
            authenticator.logout('Logout', 'main', key='unique_key')
            st.write(f'Welcome *{st.session_state["name"]}*')
            st.title('Refer to sidebar for more pages!')
        elif st.session_state["authentication_status"] is False:
            st.error('Username/password is incorrect')
        elif st.session_state["authentication_status"] is None:
            st.warning('Please enter your username and password')

        if st.session_state["authentication_status"]:
            try:
                # Use the correct 'fields' parameter instead of 'form_name'
                if authenticator.reset_password(st.session_state["username"], fields={'Reset password': 'main'}):
                    st.success('Password modified successfully')
            except Exception as e:
                st.error(e)

        try:
            # Use the correct 'fields' parameter instead of 'form_name'
            if authenticator.register_user(fields={'Register user': 'main'}, preauthorization=False):
                st.success('User registered successfully')
        except Exception as e:
            st.error(e)

    # Run the authentication app
    if __name__ == "__main__":
        run_authentication_app()
    



elif app_mode == 'About App':
    st.markdown('Welcome to **DODS-Pharma**, where innovation meets efficiency in the pharmaceutical industry. We understand the challenges faced by pharmaceutical field officers in manually inspecting and evaluating the positioning of products in retail outlets. To revolutionize this process, we have developed a cutting-edge Image Recognition System tailored for the unique needs of field officers.')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
            width:350px
        }
        [data-testid="stSidebar"][aria-expanded="false"]>div:first-child{
            width:350px
            margin-left:-350px
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    st.markdown('''
                # The Solution\n
                Our Image Recognition System is designed to be the field officer's trusted companion. By seamlessly integrating with their workflow, our application allows them to capture images of product setups within shops effortlessly. The system then employs advanced algorithms to analyze these images based on predefined parameters, such as visibility, placement, lighting conditions, shelf positioning, and capture angle.
                
                # Key Features\n
                Automated Analysis: Say goodbye to manual inspections. Our system automates the analysis of key parameters, providing quick and accurate insights.\n
                
                Real-time Feedback: Field officers receive valuable real-time feedback on product visibility and placement, enabling them to make informed decisions on the spot.\n
                
                Incentive Derivation: The system assists field officers in deriving incentive values for each product in every shop, ensuring fair and transparent reward structures for shop owners.
            ''')

elif app_mode == 'Run on Image':
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
            width:350px
        }
        [data-testid="stSidebar"][aria-expanded="false"]>div:first-child{
            width:350px
            margin-left:-350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Detected Faces**")
    kpil_text=st.markdown("0")

    max_faces=st.sidebar.number_input('Maximum number of Face',value=2,min_value=1)
    st.sidebar.markdown('---')
    detection_confidence=st.sidebar.slider('Minimum Detection confidence',min_value=0.0,max_value=1.0,value=0.5)
    st.sidebar.markdown('---')

    img_file_buffer=st.sidebar.file_uploader("Upload an Image",type=["jpg","jpeg","png"])
    if img_file_buffer is not None:
        image=np.array(Image.open(img_file_buffer))

    else:
        demo_image=DEMO_IMAGE 
        image=np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)

    face_count=0

    ##Dashboard
    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=max_faces,
    min_detection_confidence=detection_confidence) as face_mesh:
        
        results=face_mesh.process(image)
        out_image=image.copy()

        ##Face Landmark Drawing
        for face_landmarks in results.multi_face_landmarks:
            face_count+=1


            mp_drawing.draw_landmarks(
            image=out_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec)

            kpil_text.write(f"<h1 style='text-align:center;color:red;'>{face_count}</h1>",unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(out_image,use_column_width=True)



elif app_mode == 'Run on Video':
    st.set_option('deprecation.showfileUploaderEncoding',False)

    use_webcam=st.sidebar.button('Use Webcam')
    record=st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording",value=True)
    
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
            width:350px
        }
        [data-testid="stSidebar"][aria-expanded="false"]>div:first-child{
            width:350px
            margin-left:-350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    max_faces=st.sidebar.number_input('Maximum number of Face',value=5,min_value=1)
    st.sidebar.markdown('---')
    detection_confidence=st.sidebar.slider('Minimum Detection confidence',min_value=0.0,max_value=1.0,value=0.5)
    tracking_confidence=st.sidebar.slider('Minimum Tracking confidence',min_value=0.0,max_value=1.0,value=0.5)
    st.sidebar.markdown('---')


    st.markdown('## Output')

    stframe=st.empty()
    video_file_buffer=st.sidebar.file_uploader("Upload a video",type=['mp4','mov','avi','m4v'])
    tffile=tempfile.NamedTemporaryFile(delete=False)


    ##video
    if not video_file_buffer:
        if use_webcam:
            vid=cv2.VideoCapture(0)
        else:
            vid=cv2.VideoCapture(DEMO_VIDEO)
            tffile.name=DEMO_VIDEO
    else:
        tffile.write(video_file_buffer.read())
        vid=cv2.VideoCapture(tffile.name)

    width=int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input=int(vid.get(cv2.CAP_PROP_FPS))


    ##Recording
    codec=cv2.VideoWriter_fourcc('M','J','P','G')
    out=cv2.VideoWriter('ouput1.mp4',codec,fps_input,(width,height))
    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)

    fps=0
    i=0
    drawing_spec=mp_drawing.DrawingSpec(thickness=2,circle_radius=1)

    kpi1,kpi2,kpi3=st.columns(3)

    with kpi1:
        st.markdown('**Frame Rate**')
        kpi1_text=st.markdown("0")
    with kpi2:
        st.markdown('**Detected Faces**')
        kpi2_text=st.markdown("0")
    with kpi3:
        st.markdown('**Image Width**')
        kpi3_text=st.markdown("0")

    st.markdown("</hr>",unsafe_allow_html=True)

     ##Dashboard
    with mp_face_mesh.FaceMesh(
    max_num_faces=max_faces,
    min_tracking_confidence=tracking_confidence,
    min_detection_confidence=detection_confidence) as face_mesh:
        
        prevTime=0
        while vid.isOpened():
            i+=1
            ret,frame=vid.read()
            if not ret:
                continue
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            results=face_mesh.process(frame)
            frame.flags.writeable=True
            frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            face_count=0
            if results.multi_face_landmarks:
                ##Face Landmark Drawing
             for face_landmarks in results.multi_face_landmarks:
              face_count+=1


              mp_drawing.draw_landmarks(
              image=frame,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=drawing_spec,
              connection_drawing_spec=drawing_spec)


            currTime=time.time()
            fps=1/(currTime-prevTime)
            prevTime=currTime

            if record:
                out.write(frame)

            ##Dashboard
            kpi1_text.write(f"<h1 style='text-align:center;color:red;'>{int(fps)}</h1>",unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align:center;color:red;'>{face_count}</h1>",unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align:center;color:red;'>{width}</h1>",unsafe_allow_html=True)

            frame=cv2.resize(frame,(0,0),fx=0.8,fy=0.8)
            frame=image_resize(image=frame,width=640)
            stframe.image(frame,channels='BGR',use_column_width=True)


        # results=face_mesh.process(image)
        # out_image=image.copy()

        
        
        st.subheader('Output Image')
        st.image(out_image,use_column_width=True)

elif app_mode == 'Phone Scanner':
    vid = cv2.VideoCapture('http://192.168.7.31:8080/video')

# Set the Streamlit app title
    st.title('Using Mobile Camera with Streamlit')
    darkornot=st.empty()
    # Create a Streamlit image placeholder
    frame_window = st.image([])

    # Create a button to take a picture
    take_picture_button = st.button('Take Picture')

    # Run the app in an infinite loop
    while True:
        # Read a frame from the video stream
        got_frame, frame = vid.read()

        # Convert the color space from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Check if a frame is obtained
        if got_frame:
            # Check if the frame is too dark
            if np.mean(frame) < 0.3 * 255:
                darkornot.write("Dark")
            else:
                darkornot.write("Acceptable")

            # Display the frame in Streamlit
            frame_window.image(frame)

            # Display the result text using st.text
            

        # Check if the "Take Picture" button is pressed
        if take_picture_button:
            # Perform any additional processing or model inference here if needed
            break

    # Release the video stream
    vid.release()
            


    

    

   
