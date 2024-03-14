import streamlit as st
from PIL import Image

from rqle_ai_langchain_util.utils import gui_utils
from rqle_ai_langchain_util.examples.techtype_rocket.techtype_rocket import TechTypeRocket

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
st.markdown(body=gui_utils.set_page_header(application_name=APPLICATION_NAME,
                                           application_logo_path=APPLICATION_LOGO_PATH,
                                           background_color='#334246',
                                           text_color='#FFFFFF'),
            unsafe_allow_html=True)

# main body of the application
input, output = st.columns([1.5, 2.5])
output.subheader('Generated Blog', divider='rainbow')

with input:
    with st.form(key='blog_input'):
        st.subheader('Enter Blog Details', divider='rainbow')
        reading_time = st.number_input(label='Reading Time (min)', min_value=4, step=1,
                                       help='Enter how long you want blog to be in terms of reading time')
        target_audience = st.selectbox(label='Target Audience', options=('Technology Leaders', 'Business Leaders', 'Novice'))
        # example of topics for debugging purposes
        # monetization of premium large language models when integrated in commercial products
        topics = st.text_area(label='Topics', height=200)

        # form submission button
        submit = st.form_submit_button(label='Generate', type='primary')

        # execute logic if submit button is clicked
        if submit:
            if not topics or topics == '':
                st.error('Please enter a one or more topics to be written about.')
            else:
                with output:
                    with st.spinner('Generating blog...'):
                        tech_type_rocket = TechTypeRocket(config_folder='techtype_rocket')
                        generated_blog = tech_type_rocket.invoke_chain(reading_time=reading_time,
                                                                       target_audience=target_audience,
                                                                       topics=topics)
                        generated_blog_data = generated_blog['text']
                        markdown_tab, text_tab = st.tabs(['WYSIWYG', 'Markdown'])
                        with markdown_tab:
                            st.markdown(generated_blog_data)
                        with text_tab:
                            st.text(generated_blog_data)

# information about the footer of the application (including logo and years)
st.markdown(body=gui_utils.set_page_footer(), unsafe_allow_html=True)