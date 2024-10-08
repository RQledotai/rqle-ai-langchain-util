import plotly.graph_objects as go
import streamlit as st

from rqle_ai_langchain_util.utils.file_util import read_image

COMPANY_LOGO_PATH = 'img/rqle_ai_logo_alt.jpeg'
COMPANY_NAME = 'RQle.AI'


def set_page_sidebar(application_description: str, background_color: str = '#C0BAAC',
                     git_repo_url: str = 'https://github.com/RQledotai/rqle-ai-langchain-util'):
    """
    Set up the sidebar of the application
    :param application_description: the description of the application
    :param background_color: the background color of the sidebar
    :param git_repo_url: the url of the git repository
    :return: a sidebar to be integrated in a .streamlit app
    """
    st.markdown(f"""
        <style>
            [data-testid=stSidebar] {{
                background-color: {background_color};
            }}
        </style>
        """, unsafe_allow_html=True)
    st.sidebar.markdown(body=f"""<img src="data:image/jpeg;base64,{read_image(COMPANY_LOGO_PATH)}" 
                                      alt="{COMPANY_LOGO_PATH}" width="100" height="100"/>""",
                        unsafe_allow_html=True)
    st.sidebar.header(body='About')
    st.sidebar.markdown(body=f"{application_description}")

    st.sidebar.header(body=COMPANY_NAME)
    st.sidebar.markdown(body=f"""
     **{COMPANY_NAME}** believes in the transformative potential of Generative AI. More specifically, it focuses on 
     showcasing real-world applications of how Generative AI can empower individuals and organizations worldwide in 
     addressing their customers' "*job to be done*" problems and create value for them.
    """)

    st.sidebar.header(body='Resources')
    st.sidebar.markdown(body=f"""
    - [GitHub Repository]({git_repo_url})
    - [LinkedIn Profile](https://www.linkedin.com/company/102641077)
    - **Author**: [RQle.ai](mailto:rqledotai@gmail.com)
    """)

    st.sidebar.header(body='Disclaimer')
    st.sidebar.markdown(body="""
    * The development of this application has been aided by Generative AI. More specifically, a combination of 
      [cody](https://sourcegraph.com/cody) and either premium or open-source large language models to suggest coding 
      best practices.
    * The content and / or algorithms generated by this application are not intended to be final. In other words, the 
      human user has the responsibility to critically evaluate and revise the output to ensure its accuracy, 
      appropriateness, and ethical implications.
    """)

    return st.sidebar


def set_page_header(application_name: str, application_logo_path: str, background_color: str = '#C0BAAC',
                    text_color: str = '#000000') -> str:
    """
    Set up the header of the application
    :param application_name: the name of the application
    :param application_logo_path: the path to the logo of the application
    :param background_color: the background color of the header
    :param text_color: the text color of the header
    :return: Markdown string for the header of the application
    """
    return f"""
     <style>
        header {{
            background-color: {background_color};
        }}
        h1 {{
            color: {text_color};
        }}
     </style>
     <header>
        <h1 align="center">
            <img src="data:image/jpeg;base64,{read_image(application_logo_path)}" alt="{application_logo_path}" 
                 width="200" height="200"/> - 
            {application_name}&copy;
        </h1>
        <hr/>
    </header>
    """


def set_page_footer(years: str = '2024') -> str:
    """
    Set up the footer of the application
    :param years: the years of the copyright
    :return: Markdown string for the footer of the application
    """
    return f"""
     <style>
        footer {{
            background-color: #C0BAAC;
        }}
     </style>
     <footer>
        <hr/>
        <h5 align="center">
            {years} - All rights reserved.
            <img src="data:image/jpeg;base64,{read_image(COMPANY_LOGO_PATH)}" alt="{COMPANY_LOGO_PATH}" 
                 width="100" height="100"/>
        </h5>
    </footer>
    """


def make_donut_graph(expected: int, actual: int):
    """
    Create a donut chart comparing expected vs actual.
    :param expected: the expected value
    :param actual: the actual / estimated value
    :return: a donut graph showing the comparison
    """
    # determine which color palette relate to different donut charts
    green_color_palette = ['#27AE60', '#12783D']
    orange_color_palette = ['#F39C12', '#875A12']
    red_color_palette = ['#E74C3C', '#781F16']

    # calculate the deviation between actual vs expected & decide colors to be used
    deviation_percentage = (actual / expected) * 100
    if (deviation_percentage >= 65):
        chart_color = green_color_palette
    elif 35 < deviation_percentage < 65:
        chart_color = orange_color_palette
    else:
        chart_color = red_color_palette

    labels = ['Expected', 'Actual']
    percents = [100, deviation_percentage]

    # create the donut chart
    fig = go.Figure()
    fig.add_trace(go.Pie(labels=labels, values=percents, rotation=150, hole=0.3, marker_colors=chart_color))
    fig.add_trace(go.Pie(labels=labels, values=percents, rotation=150, hole=0.3, marker_colors=chart_color))
    fig.update_traces()
    fig.update_traces(textinfo='none', title=f'{actual} min', hoverinfo='none')
    fig.update(layout_showlegend=False)

    return fig




