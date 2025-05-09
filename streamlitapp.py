# Import all of the dependencies
import streamlit as st
import os
import imageio
import subprocess
import numpy as np
import tensorflow as tf
from utils import load_data, num_to_char  # load_data should be the TensorFlow wrapper version
from modelutil import load_model          # or use your build_model function

# Set the layout to the Streamlit app as wide
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is developed from the LipNet deep learning model for lip-reading of mp4 video files.')

st.title('LipNet Full Stack App')

# Generate a list of video options
data_dir = os.path.join(os.getcwd(), 'data', 's1')
options = os.listdir(data_dir)
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

if options:
    # Define paths
    file_path = os.path.join(data_dir, selected_video)
    output_video = 'test_video.mp4'
    
    # Provide the full path to ffmpeg.exe (adjust to your installation)
    ffmpeg_path = r"C:\Users\Vinot\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
    
    # Render the video in col1
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        try:
            subprocess.run([ffmpeg_path, '-i', file_path, '-vcodec', 'libx264', output_video, '-y'], check=True)
            with open(output_video, 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
        except subprocess.CalledProcessError as e:
            st.error(f"FFmpeg failed to convert video: {e}")
        except FileNotFoundError:
            st.error("Converted video file not found. Make sure FFmpeg conversion succeeded.")

    # Process and show model prediction in col2
    with col2:
        st.info('This is what the machine learning model sees when making a prediction')

        # Load and preprocess video (load_data should output a video tensor of shape (50, 46, 140, 1))
        video, annotations = load_data(tf.convert_to_tensor(file_path))

        # Convert video frames to valid image format for GIF
        video_np = video.numpy()  # Tensor -> NumPy
        video_uint8 = (video_np * 255).astype(np.uint8)  # Normalize and cast

        if video_uint8.shape[-1] == 1:
            video_uint8 = video_uint8.squeeze(-1)  # Remove last channel if grayscale

        # Save and display GIF
        imageio.mimsave('animation.gif', video_uint8, fps=10)
        st.image('animation.gif', width=400)

        # Predict with model
        st.info('Model output tokens:')
        model = load_model()  # Ensure that this model uses input shape (50,46,140,1)
        yhat = model.predict(tf.expand_dims(video, axis=0))

        # CTC decoding
        decoded_tensor, _ = tf.keras.backend.ctc_decode(
            yhat,
            input_length=tf.fill([tf.shape(yhat)[0]], tf.shape(yhat)[1]),
            greedy=True
        )

        try:
            # Take the first sequence from batch
            decoder = decoded_tensor[0].numpy()[0]
            st.text(f"Raw Tokens:\n{decoder}")

            st.info('Decode the raw tokens into words')
            # Ensure num_to_char receives a batch, so wrap the sequence in a list
            char_tensor = num_to_char(tf.convert_to_tensor([decoder]))
            decoded_str_tensor = tf.strings.reduce_join(char_tensor, axis=-1)
            decoded_str = decoded_str_tensor.numpy()[0].decode('utf-8')

            if decoded_str.strip():
                st.success(f"Decoded Text: {decoded_str}")
            else:
                st.warning("Decoded text is empty. The model may have returned mostly blank tokens.")
        except Exception as e:
            st.error(f"Decoding failed: {e}")
