import streamlit as st
from PIL import Image

from rqle_ai_langchain_util.utils import gui_util
from rqle_ai_langchain_util.apps.clarity_genie.clarity_genie import ClarityGenie

APPLICATION_LOGO_PATH = 'img/claritygenie_logo.jpg'
APPLICATION_LOGO = Image.open(APPLICATION_LOGO_PATH)
APPLICATION_NAME = 'ClarityGenie'
APPLICATION_DESCRIPTION = f"""
    **{APPLICATION_NAME}**:copyright: is your one-stop shop for crafting powerful prompts fueled by cutting-edge 
    Generative AI. It aims to (i) refine your prompts for ultimate clarity, and (ii) add context to your prompts.
"""

# set up the high-level configuration of the page
st.set_page_config(page_title=APPLICATION_NAME, page_icon=APPLICATION_LOGO,
                   layout='wide', initial_sidebar_state='collapsed')

# information about / help component of the application
st.sidebar = gui_utils.set_page_sidebar(application_description=APPLICATION_DESCRIPTION)

# set the logo / title of the application
st.markdown(body=gui_utils.set_page_header(application_name=APPLICATION_NAME, application_logo_path=APPLICATION_LOGO_PATH),
            unsafe_allow_html=True)

# main body of the application
input, output = st.columns([1.5, 2.5])
output.subheader('Clarified Prompt', divider='rainbow')

with input:
    with st.form(key='prompt_input'):
        st.subheader('Paste Prompt', divider='rainbow',help='Please paste a prompt to be clarified.')
        prompt = st.text_area(label='Prompt',key='prompt', height=400, label_visibility='hidden')
        # form submission button
        submit = st.form_submit_button(label='Clarify', type='primary')

        # execute logic if submit button is clicked
        if submit:
            if not prompt or prompt == '':
                st.error('Please enter a prompt to be clarified.')
            else:
                with output:
                    with st.spinner('Clarifying prompt...'):
                        clarity_genie = ClarityGenie(config_folder='clarity_genie')
                        clarified_prompt = clarity_genie.invoke_chain(prompt)
                        st.write(clarified_prompt['text'])

# information about the footer of the application (including logo and years)
st.markdown(body=gui_utils.set_page_footer(), unsafe_allow_html=True)
