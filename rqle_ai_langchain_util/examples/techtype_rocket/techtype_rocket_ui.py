import streamlit as st
from PIL import Image

from rqle_ai_langchain_util.utils import gui_utils

APPLICATION_LOGO_PATH = 'img/techtype_rocket_logo.jpg'
APPLICATION_LOGO = Image.open(APPLICATION_LOGO_PATH)
APPLICATION_NAME = 'TechType Rocket'
APPLICATION_DESCRIPTION = f"""
    **{APPLICATION_NAME}**:copyright:
"""

st.set_page_config(page_title=APPLICATION_NAME, page_icon=APPLICATION_LOGO,
                   layout='wide', initial_sidebar_state='collapsed')

# information about / help component of the application
st.sidebar = gui_utils.set_page_sidebar(application_description=APPLICATION_DESCRIPTION)

# set the logo / title of the application
st.markdown(body=gui_utils.set_page_header(application_name=APPLICATION_NAME, application_logo_path=APPLICATION_LOGO_PATH),
            unsafe_allow_html=True)

# main body of the application
# TODO

# information about the footer of the application (including logo and years)
st.markdown(body=gui_utils.set_page_footer(), unsafe_allow_html=True)